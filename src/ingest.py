"""Chunk scraped markdown and upsert to Pinecone (integrated inference).

Usage:
    python -m src.ingest                    # incremental (skips IDs already in Pinecone)
    python -m src.ingest --force            # re-upsert everything
    python -m src.ingest --throttle 12      # tune sleep between batches

Reads ``data/raw/*.md`` (produced by ``src.scraper``), chunks each file with a
header-aware splitter, and upserts records to a Pinecone integrated index that
auto-embeds the ``text`` field with ``multilingual-e5-large``.

The free tier caps the embedding model at ~250K tokens/min, so we throttle
between batches and retry with backoff on 429 responses. Re-runs are idempotent
because record IDs are deterministic (``<page-slug>-<chunk-index>``).

Each record carries flat metadata fields used for filter + display:
    _id, text, source_url, title, section_path, category, chunk_index
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    PINECONE_CLOUD,
    PINECONE_EMBED_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
    RAW_DIR,
)

load_dotenv()


@dataclass
class Chunk:
    _id: str
    text: str
    source_url: str
    title: str
    section_path: str
    category: str
    chunk_index: int


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, text[m.end() :]


def _split_by_headers(body: str) -> list[tuple[str, str]]:
    """Return [(section_path, section_text), ...] tracking H1>H2>H3 stack."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    last_end = 0
    last_path = ""

    for m in _HEADER_RE.finditer(body):
        chunk_text = body[last_end : m.start()].strip()
        if chunk_text:
            sections.append((last_path or "Introduction", chunk_text))

        level = len(m.group(1))
        title = m.group(2).strip()
        stack = stack[: level - 1]
        while len(stack) < level - 1:
            stack.append("")
        stack.append(title)
        last_path = " > ".join(s for s in stack if s)
        last_end = m.end()

    tail = body[last_end:].strip()
    if tail:
        sections.append((last_path or "Introduction", tail))
    return sections


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    url = meta.get("url", "")
    title = meta.get("title", path.stem)
    category = meta.get("category", "general")
    slug = path.stem

    chunks: list[Chunk] = []
    idx = 0
    for section_path, section_text in _split_by_headers(body):
        for piece in _split_long(section_text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS):
            piece = piece.strip()
            if len(piece) < 80:
                continue
            chunks.append(
                Chunk(
                    _id=f"{slug}-{idx}",
                    text=piece,
                    source_url=url,
                    title=title,
                    section_path=section_path,
                    category=category,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


# --------------------------------------------------------------- Pinecone


def _pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit(
            "PINECONE_API_KEY not set. Add it to .env (https://app.pinecone.io)."
        )
    from pinecone import Pinecone

    return Pinecone(api_key=api_key)


def _ensure_index(pc) -> None:
    """Create the integrated index if it doesn't exist. Idempotent."""
    existing = {idx["name"] for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME in existing:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
        return

    print(
        f"Creating index '{PINECONE_INDEX_NAME}' "
        f"(model={PINECONE_EMBED_MODEL}, cloud={PINECONE_CLOUD}/{PINECONE_REGION})..."
    )
    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        embed={"model": PINECONE_EMBED_MODEL, "field_map": {"text": "text"}},
    )

    # Wait until ready
    for _ in range(60):
        desc = pc.describe_index(PINECONE_INDEX_NAME)
        if desc.status.get("ready"):
            print("  -> index is ready.")
            return
        time.sleep(2)
    raise SystemExit("Index did not become ready in time.")


def _batch(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _existing_ids(index, namespace: str) -> set[str]:
    """Enumerate all record IDs already in the namespace. Paginates."""
    seen: set[str] = set()
    try:
        for page in index.list(namespace=namespace):
            # SDK 5+ yields a list of IDs per page.
            for _id in page:
                seen.add(_id)
    except Exception as e:
        print(f"  ! couldn't list existing IDs ({e}); proceeding without resume.")
    return seen


def _upsert_with_retry(
    index, namespace: str, records: list[dict], max_retries: int = 5
) -> None:
    """Upsert one batch, retrying on 429 with exponential backoff."""
    delay = 60.0
    for attempt in range(1, max_retries + 1):
        try:
            index.upsert_records(namespace=namespace, records=records)
            return
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if not is_rate_limit or attempt == max_retries:
                raise
            print(f"  ! 429 rate limit (attempt {attempt}/{max_retries}); "
                  f"sleeping {delay:.0f}s then retrying...")
            time.sleep(delay)
            delay = min(delay * 2, 300)


def build_index(throttle_seconds: float = 10.0, force: bool = False) -> None:
    if not RAW_DIR.exists():
        raise SystemExit(f"No raw content at {RAW_DIR}. Run `python -m src.scraper` first.")

    md_files = sorted(p for p in RAW_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files in {RAW_DIR}.")

    print(f"Chunking {len(md_files)} files...")
    all_chunks: list[Chunk] = []
    for path in md_files:
        all_chunks.extend(chunk_file(path))
    print(f"  -> {len(all_chunks)} chunks")
    if not all_chunks:
        raise SystemExit("No chunks produced.")

    pc = _pinecone_client()
    _ensure_index(pc)
    index = pc.Index(PINECONE_INDEX_NAME)

    if not force:
        print("Listing already-upserted record IDs (for resume)...")
        seen = _existing_ids(index, PINECONE_NAMESPACE)
        before = len(all_chunks)
        all_chunks = [c for c in all_chunks if c._id not in seen]
        print(f"  -> {len(seen)} already in index, {len(all_chunks)} new "
              f"(skipped {before - len(all_chunks)})")
        if not all_chunks:
            stats = index.describe_index_stats()
            print(f"\nNothing to do. total_vector_count={stats.get('total_vector_count')}")
            return

    # Pinecone integrated upsert_records: max ~96 records / batch.
    BATCH_SIZE = 96
    batches = list(_batch([asdict(c) for c in all_chunks], BATCH_SIZE))
    print(
        f"Upserting {len(all_chunks)} records in {len(batches)} batches "
        f"(throttle={throttle_seconds}s/batch, ~{len(batches) * throttle_seconds / 60:.0f} min)..."
    )
    started = time.time()
    for i, batch in enumerate(batches, 1):
        _upsert_with_retry(index, PINECONE_NAMESPACE, batch)
        if i == 1 or i % 10 == 0 or i == len(batches):
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(batches) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(batches)}] upserted {len(batch)} records  "
                  f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.0f}m)")
        if i < len(batches):
            time.sleep(throttle_seconds)

    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", "?")
    print(f"\nDone. Index '{PINECONE_INDEX_NAME}' total_vector_count={total}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--throttle",
        type=float,
        default=10.0,
        help="Seconds to sleep between batches (default 10; "
             "free tier embedding cap is ~250K tokens/min)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-upsert all chunks; do not skip already-indexed IDs",
    )
    args = p.parse_args()
    build_index(throttle_seconds=args.throttle, force=args.force)


if __name__ == "__main__":
    main()
