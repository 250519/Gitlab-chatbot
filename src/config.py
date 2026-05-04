"""Central config for paths, models, and tunables."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SEED_URLS_FILE = DATA_DIR / "seed_urls.txt"
DISCOVERED_URLS_FILE = DATA_DIR / "discovered_urls.json"

# --- Pinecone ---
PINECONE_INDEX_NAME = "gitlab-handbook"
PINECONE_NAMESPACE = "default"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# --- Local embedding + rerank ---
# We embed locally to avoid Pinecone's hosted-embedding monthly token cap.
# bge-small-en-v1.5 is 384-dim, English-only, ~25x faster than e5-large.
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_EMBED_DIM = 384
LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# --- Groq ---
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
GROQ_AVAILABLE_MODELS = [GROQ_DEFAULT_MODEL, GROQ_FAST_MODEL]

# --- Chunking / retrieval ---
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
TOP_K_DEFAULT = 5            # final results returned to LLM (post-rerank)
RETRIEVE_FETCH_K = 25        # over-fetch before reranking
TOP_K_MIN = 2
TOP_K_MAX = 10

# --- Categories (metadata tag taxonomy) ---
# Order roughly mirrors a sidebar; "general" is the catch-all.
CATEGORIES = [
    "values",
    "culture",
    "hiring",
    "people",
    "engineering",
    "product",
    "direction",
    "commercial",
    "leadership",
    "legal_security",
    "general",
]

# URL substring → category. First match wins. Keep ordered by specificity.
CATEGORY_RULES: list[tuple[str, str]] = [
    ("/handbook/values", "values"),
    ("/handbook/company/culture/all-remote", "culture"),
    ("/handbook/company/culture", "culture"),
    ("/handbook/communication", "culture"),
    ("/handbook/hiring", "hiring"),
    ("/handbook/people-group", "people"),
    ("/handbook/total-rewards", "people"),
    ("/handbook/engineering", "engineering"),
    ("/handbook/product", "product"),
    ("/handbook/marketing", "commercial"),
    ("/handbook/sales", "commercial"),
    ("/handbook/customer-success", "commercial"),
    ("/handbook/support", "commercial"),
    ("/handbook/finance", "commercial"),
    ("/handbook/legal", "legal_security"),
    ("/handbook/security", "legal_security"),
    ("/handbook/leadership", "leadership"),
    ("/handbook/ceo", "leadership"),
    ("/direction", "direction"),
    ("/handbook/about", "general"),
    ("/handbook/company", "general"),
    ("/handbook", "general"),
]

# Per-category quotas when picking URLs to ingest. Set values >= category sizes
# (or use TOP_N_URLS=None) to ingest everything available.
CATEGORY_QUOTAS: dict[str, int] = {
    "values": 5000,
    "culture": 5000,
    "hiring": 5000,
    "people": 5000,
    "engineering": 5000,
    "product": 5000,
    "direction": 5000,
    "commercial": 5000,
    "leadership": 5000,
    "legal_security": 5000,
    "general": 5000,
}
# Set to None to keep everything, or an int cap.
TOP_N_URLS: int | None = None

# Categories to actually embed + upsert. Set to None to ingest all categories.
INGEST_CATEGORIES: list[str] | None = ["values", "culture", "hiring", "engineering"]

# --- Scraper ---
SCRAPER_USER_AGENT = (
    "GitLabHandbookChatbot/0.1 (educational project; contact: harsh.shivhare@truefoundry.com)"
)
SCRAPER_DELAY_SECONDS = 0.1   # per-request stagger; combined with concurrency
SCRAPER_CONCURRENCY = 10      # max parallel HTTP fetches
SCRAPER_TIMEOUT_SECONDS = 30
SCRAPER_MIN_CONTENT_CHARS = 500  # skip SPA shells / placeholder pages

# --- Sitemaps ---
SITEMAPS = [
    "https://handbook.gitlab.com/sitemap.xml",
    "https://about.gitlab.com/sitemap.xml",
]


def categorize_url(url: str) -> str:
    """Map a URL to one of CATEGORIES via CATEGORY_RULES."""
    for needle, cat in CATEGORY_RULES:
        if needle in url:
            return cat
    return "general"
