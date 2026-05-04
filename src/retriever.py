"""Pinecone retriever with local embedding + rerank.

Originally this used Pinecone integrated inference (server-side embedding +
rerank). The free-tier monthly token quota for ``multilingual-e5-large`` covers
both ingest and query, so once it's exhausted no queries land either. This
module embeds the query locally with the same ``intfloat/multilingual-e5-large``
model that ingest used (so vectors line up against what Pinecone already
stores) and reranks locally with ``BAAI/bge-reranker-v2-m3``.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from src.config import (
    LOCAL_EMBED_MODEL,
    LOCAL_RERANK_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    RETRIEVE_FETCH_K,
    TOP_K_DEFAULT,
)

# bge-small-en-v1.5 retrieval works best with this query-side instruction.
# Documents (passages) need no prefix.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

load_dotenv()


class IndexNotBuilt(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise IndexNotBuilt(
            "PINECONE_API_KEY not set. Add it to .env "
            "(get one at https://app.pinecone.io)."
        )
    from pinecone import Pinecone

    pc = Pinecone(api_key=api_key)
    existing = {i["name"] for i in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing:
        raise IndexNotBuilt(
            f"Pinecone index '{PINECONE_INDEX_NAME}' not found. "
            "Run `python -m src.scraper && python -m src.ingest` first."
        )
    return pc.Index(PINECONE_INDEX_NAME)


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(LOCAL_EMBED_MODEL, device=_pick_device())


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    # Pin reranker to CPU. Reason: on Apple Silicon MPS, the first forward pass
    # through bge-reranker-v2-m3's XLM-R-large architecture triggers ~9 min of
    # kernel compilation per shape — terrible startup UX. CPU rerank for 25
    # candidate pairs takes ~3-5s on M-series Macs with no compile cost.
    return CrossEncoder(LOCAL_RERANK_MODEL, device="cpu")


def _embed_query(text: str) -> list[float]:
    vec = _embedder().encode(
        f"{BGE_QUERY_INSTRUCTION}{text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.tolist()


def search(
    query: str,
    k: int = TOP_K_DEFAULT,
    fetch_k: int = RETRIEVE_FETCH_K,
    categories: list[str] | None = None,
) -> list[dict]:
    """Retrieve + rerank top-k chunks (all local; no Pinecone embedding calls).

    Pipeline:
      1. Embed query locally with ``multilingual-e5-large``.
      2. Dense vector search via ``index.query(vector=...)``, over-fetching ``fetch_k``.
      3. Optional MongoDB-style metadata filter on ``category``.
      4. Local rerank with ``bge-reranker-v2-m3`` to ``k`` final results.

    Each returned dict includes:
      text, source_url, title, section_path, category, chunk_index, score.
    """
    index = _index()

    qvec = _embed_query(query)
    pq_kwargs: dict = {
        "namespace": PINECONE_NAMESPACE,
        "vector": qvec,
        "top_k": max(fetch_k, k),
        "include_metadata": True,
    }
    if categories:
        pq_kwargs["filter"] = {"category": {"$in": list(categories)}}

    response = index.query(**pq_kwargs)
    matches = getattr(response, "matches", None)
    if matches is None:
        matches = response["matches"]

    candidates: list[dict] = []
    for m in matches:
        meta = getattr(m, "metadata", None)
        if meta is None and isinstance(m, dict):
            meta = m.get("metadata", {})
        candidates.append(dict(meta or {}))

    if not candidates:
        return []

    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = _reranker().predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    out: list[dict] = []
    for c, s in ranked[:k]:
        c["score"] = float(s)
        out.append(c)
    return out


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are GitLab's values?"
    for r in search(q, k=5):
        print(f"{r['score']:.3f}  [{r.get('category','?'):>14}]  "
              f"{r.get('title','')}  >  {r.get('section_path','')}")
        print(f"        {r.get('source_url','')}")
        print(f"        {r.get('text','')[:160]}...\n")
