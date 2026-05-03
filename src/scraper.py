"""Fetch GitLab Handbook + Direction pages and convert them to clean markdown.

Usage:
    python -m src.scraper                  # full scrape (skips files already on disk)
    python -m src.scraper --force          # re-fetch all
    python -m src.scraper --concurrency 20 # tune parallelism

Reads URLs from ``data/seed_urls.txt`` and writes one ``.md`` file per URL into
``data/raw/``. The filename is a hash of the URL so re-runs are idempotent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from src.config import (
    RAW_DIR,
    SCRAPER_CONCURRENCY,
    SCRAPER_DELAY_SECONDS,
    SCRAPER_MIN_CONTENT_CHARS,
    SCRAPER_TIMEOUT_SECONDS,
    SCRAPER_USER_AGENT,
    SEED_URLS_FILE,
    categorize_url,
)


@dataclass
class Page:
    url: str
    title: str
    markdown: str
    category: str = ""

    @property
    def slug(self) -> str:
        return hashlib.sha1(self.url.encode("utf-8")).hexdigest()[:16]


def _read_seed_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    return urls


def _extract_main_content(html: str) -> tuple[str, str]:
    """Return (title, cleaned_html_of_main_content)."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )

    return title, str(main)


def fetch_page(client: httpx.Client, url: str) -> Page | None:
    try:
        resp = client.get(url, follow_redirects=True, timeout=SCRAPER_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  ! fetch failed: {url} ({exc})", file=sys.stderr)
        return None

    title, main_html = _extract_main_content(resp.text)
    markdown = md(main_html, heading_style="ATX", strip=["a"]).strip()

    if len(markdown) < SCRAPER_MIN_CONTENT_CHARS:
        print(f"  ! skipping (only {len(markdown)} chars): {url}", file=sys.stderr)
        return None

    return Page(url=url, title=title or url, markdown=markdown, category=categorize_url(url))


def _write_page(page: Page) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{page.slug}.md"
    body = (
        f"---\nurl: {page.url}\ntitle: {page.title}\ncategory: {page.category}\n---\n\n"
        f"# {page.title}\n\n{page.markdown}\n"
    )
    out.write_text(body, encoding="utf-8")
    return out


def _slug_for(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _existing_page_for(url: str) -> Page | None:
    """If we already scraped this URL, reconstruct a minimal Page from disk."""
    out = RAW_DIR / f"{_slug_for(url)}.md"
    if not out.exists():
        return None
    text = out.read_text(encoding="utf-8")
    title = ""
    category = ""
    if text.startswith("---"):
        for line in text.splitlines()[1:]:
            if line.startswith("---"):
                break
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("category:"):
                category = line.split(":", 1)[1].strip()
    return Page(url=url, title=title or url, markdown="", category=category or categorize_url(url))


def scrape_all(concurrency: int = SCRAPER_CONCURRENCY, force: bool = False) -> list[Page]:
    if not SEED_URLS_FILE.exists():
        raise SystemExit(f"Seed URL file not found: {SEED_URLS_FILE}")
    urls = _read_seed_urls(SEED_URLS_FILE)
    if not urls:
        raise SystemExit("No URLs to scrape.")

    if not force:
        new_urls = [u for u in urls if not (RAW_DIR / f"{_slug_for(u)}.md").exists()]
        skipped = len(urls) - len(new_urls)
        print(
            f"{len(urls)} URLs total | "
            f"{skipped} already on disk (skip) | {len(new_urls)} to fetch"
        )
        urls = new_urls
    else:
        print(f"--force: re-fetching all {len(urls)} URLs")

    if not urls:
        print("Nothing to do.")
        return []

    print(f"Scraping with concurrency={concurrency} into {RAW_DIR}")
    headers = {"User-Agent": SCRAPER_USER_AGENT}
    pages: list[Page] = []
    progress_lock = threading.Lock()
    done = 0
    total = len(urls)

    def worker(client: httpx.Client, url: str) -> Page | None:
        # Stagger requests slightly to be polite even at high concurrency.
        time.sleep(SCRAPER_DELAY_SECONDS)
        page = fetch_page(client, url)
        if page is not None:
            _write_page(page)
        return page

    with httpx.Client(headers=headers) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(worker, client, u): u for u in urls}
            for fut in as_completed(futures):
                url = futures[fut]
                try:
                    page = fut.result()
                except Exception as e:
                    print(f"  ! exception on {url}: {e}", file=sys.stderr)
                    page = None
                with progress_lock:
                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f"  [{done}/{total}] kept={len(pages)+(1 if page else 0)}")
                if page:
                    pages.append(page)

    # Merge with anything previously on disk (so index.json is complete).
    all_pages_on_disk: dict[str, dict] = {}
    if (RAW_DIR / "index.json").exists():
        try:
            all_pages_on_disk = json.loads((RAW_DIR / "index.json").read_text(encoding="utf-8"))
        except Exception:
            pass
    for p in pages:
        all_pages_on_disk[p.slug] = {"url": p.url, "title": p.title, "category": p.category}
    (RAW_DIR / "index.json").write_text(
        json.dumps(all_pages_on_disk, indent=2), encoding="utf-8"
    )

    print(f"\nDone. Newly wrote {len(pages)} pages "
          f"(total on disk: {len(list(RAW_DIR.glob('*.md')))})")
    return pages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=SCRAPER_CONCURRENCY)
    p.add_argument("--force", action="store_true", help="Re-fetch URLs already on disk")
    args = p.parse_args()
    scrape_all(concurrency=args.concurrency, force=args.force)


if __name__ == "__main__":
    main()
