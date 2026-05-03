"""Discover GitLab Handbook + Direction URLs from public sitemaps and pick the
top N across categories.

Usage:
    python -m src.discover

Outputs:
    data/seed_urls.txt          # final list (overwritten)
    data/discovered_urls.json   # full discovered set with metadata for inspection
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import httpx
from bs4 import BeautifulSoup

from src.config import (
    CATEGORIES,
    CATEGORY_QUOTAS,
    DISCOVERED_URLS_FILE,
    SCRAPER_TIMEOUT_SECONDS,
    SCRAPER_USER_AGENT,
    SEED_URLS_FILE,
    SITEMAPS,
    TOP_N_URLS,
    categorize_url,
)


@dataclass
class DiscoveredUrl:
    url: str
    category: str
    lastmod: str
    priority: float
    depth: int  # path-segment depth, lower = more general/index page


# ------------------------------------------------------------------ sitemaps


def _fetch(client: httpx.Client, url: str) -> str | None:
    try:
        r = client.get(url, timeout=SCRAPER_TIMEOUT_SECONDS, follow_redirects=True)
        r.raise_for_status()
        return r.text
    except httpx.HTTPError as e:
        print(f"  ! sitemap fetch failed: {url} ({e})")
        return None


def _parse_sitemap(xml: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (sub_sitemap_urls, [(loc, lastmod, priority), ...])."""
    soup = BeautifulSoup(xml, "xml")
    subs = [s.loc.text.strip() for s in soup.find_all("sitemap") if s.loc]
    urls: list[tuple[str, str, str]] = []
    for u in soup.find_all("url"):
        loc = u.loc.text.strip() if u.loc else None
        if not loc:
            continue
        lastmod = u.lastmod.text.strip() if u.lastmod else ""
        priority = u.priority.text.strip() if u.priority else "0.5"
        urls.append((loc, lastmod, priority))
    return subs, urls


def gather_urls() -> list[DiscoveredUrl]:
    headers = {"User-Agent": SCRAPER_USER_AGENT}
    seen: set[str] = set()
    out: list[DiscoveredUrl] = []

    with httpx.Client(headers=headers) as client:
        queue = list(SITEMAPS)
        while queue:
            sm = queue.pop(0)
            print(f"sitemap: {sm}")
            xml = _fetch(client, sm)
            if not xml:
                continue
            subs, urls = _parse_sitemap(xml)
            queue.extend(s for s in subs if s not in SITEMAPS)

            for loc, lastmod, priority in urls:
                if loc in seen:
                    continue
                seen.add(loc)
                if not _is_relevant(loc):
                    continue
                try:
                    pri = float(priority)
                except ValueError:
                    pri = 0.5
                depth = loc.rstrip("/").count("/") - 2
                out.append(
                    DiscoveredUrl(
                        url=loc,
                        category=categorize_url(loc),
                        lastmod=lastmod,
                        priority=pri,
                        depth=max(depth, 0),
                    )
                )
    return out


def _is_relevant(url: str) -> bool:
    """Keep only Handbook + Direction pages, skip noise (job listings, blog, etc.)."""
    if "handbook.gitlab.com/handbook/" in url:
        return True
    if "about.gitlab.com/direction" in url:
        return True
    return False


# ------------------------------------------------------------------ selection


def pick_top_n(
    discovered: list[DiscoveredUrl], n: int | None = TOP_N_URLS
) -> list[DiscoveredUrl]:
    """Pick URLs balanced across categories, capped by ``n`` (or all if None).

    Priority within each category: shallower URLs first (index pages), then
    higher sitemap priority, then most recently modified.
    """
    by_cat: dict[str, list[DiscoveredUrl]] = defaultdict(list)
    for d in discovered:
        by_cat[d.category].append(d)

    for cat, items in by_cat.items():
        items.sort(key=lambda d: (d.depth, -d.priority, d.lastmod), reverse=False)

    picked: list[DiscoveredUrl] = []
    for cat in CATEGORIES:
        quota = CATEGORY_QUOTAS.get(cat, 0)
        picked.extend(by_cat.get(cat, [])[:quota])

    if n is None:
        return picked
    if len(picked) >= n:
        return picked[:n]
    remaining = [d for d in discovered if d not in picked]
    remaining.sort(key=lambda d: (d.depth, -d.priority))
    picked.extend(remaining[: n - len(picked)])
    return picked[:n]


def main() -> None:
    discovered = gather_urls()
    print(f"\nDiscovered {len(discovered)} relevant URLs across sitemaps.")

    by_cat: dict[str, int] = defaultdict(int)
    for d in discovered:
        by_cat[d.category] += 1
    print("By category (discovered):")
    for cat in CATEGORIES:
        print(f"  {cat:>16}: {by_cat[cat]}")

    picked = pick_top_n(discovered)
    target_label = "all" if TOP_N_URLS is None else str(TOP_N_URLS)
    print(f"\nPicked {len(picked)} URLs (target {target_label}).")
    pick_by_cat: dict[str, int] = defaultdict(int)
    for d in picked:
        pick_by_cat[d.category] += 1
    print("By category (picked):")
    for cat in CATEGORIES:
        print(f"  {cat:>16}: {pick_by_cat[cat]}")

    DISCOVERED_URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DISCOVERED_URLS_FILE.write_text(
        json.dumps([asdict(d) for d in discovered], indent=2), encoding="utf-8"
    )

    SEED_URLS_FILE.write_text(
        "# Auto-generated by src.discover from sitemaps\n"
        "# Edit src/config.py CATEGORY_QUOTAS to tune the mix.\n"
        + "\n".join(d.url for d in picked)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {SEED_URLS_FILE}")
    print(f"Wrote {DISCOVERED_URLS_FILE} ({len(discovered)} URLs)")


if __name__ == "__main__":
    main()
