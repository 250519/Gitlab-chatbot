# Project Write-up: GitLab Handbook Chatbot

## Goal

Build a chatbot that lets employees (and aspiring ones) query GitLab's Handbook
and Direction pages in plain English, with **grounded, cited answers**. Inspired
by GitLab's "build in public" philosophy — we make their published knowledge searchable.

## Approach: classic RAG with reranking + metadata filtering

```
GitLab sitemaps → discover (~4000 URLs) → category-balanced top-50
                                                │
                              scrape → markdown (+ category tag)
                                                │
                                  chunk (header-aware)
                                                │
                                Pinecone integrated index
                                (auto-embed: multilingual-e5-large)
                                                │
user question → dense search (top-25) → bge-reranker-v2-m3 → top-5
        + optional metadata filter ($in: [categories])
                                                │
                              Groq Llama (streamed) → answer + citations
```

Why this stack? **Quality + auditability per dollar.** Pinecone's integrated
inference removes a local model dependency; reranking adds materially better
top-k quality (BGE rerank scores are far more discriminative than raw cosine);
metadata filters let users scope to a section ("engineering only").

## Key technical choices

### URL discovery: sitemaps, not WebSearch
GitLab publishes `handbook.gitlab.com/sitemap.xml` (~4000 URLs) and
`about.gitlab.com/sitemap.xml` (a sitemap index). We fetch and union them, then
classify each URL by URL substring rules in `src/config.py::CATEGORY_RULES`.
Selection picks the top 50 across categories using a quota-driven algorithm:
shallow URLs (index pages like `/handbook/engineering/`) are preferred, then
sitemap priority, then recency. Sitemaps are dramatically more reliable than
WebSearch crawling for this use case.

### LLM: Llama 3.3 70B on Groq
- **Open-source weights** (Meta's Llama license). Fits the brief's spirit.
- **Groq** offers a generous free tier and ~10x cheaper inference than tier-1
  closed providers, with sub-second time-to-first-token — critical for chat UX.
- Two models exposed in the sidebar: `llama-3.3-70b-versatile` (default, best
  quality) and `llama-3.1-8b-instant` (snappier follow-ups).

### Vector store: Pinecone integrated index (`multilingual-e5-large`)
- **Integrated inference** — Pinecone embeds text at upsert time. No local
  embedding model, no per-record embedding code, simpler ingest pipeline.
- `multilingual-e5-large` is 1024-dim, strong on English knowledge content.
- Hosted, durable, queryable from anywhere — zero ops for the runtime.

### Reranking: `bge-reranker-v2-m3` (over-fetch top-25 → top-5)
- Dense search alone gives "vibe close" results; a cross-encoder rerank pulls
  the actually-relevant chunks to the top.
- Empirically (see comparison below), rerank scores are far more discriminative:
  **0.016** for off-topic vs **0.965+** for on-topic, vs raw cosine which gave
  0.156 vs 0.83 — much harder to threshold on.
- One Pinecone call does both stages (`index.search(query=..., rerank=...)`).

### Metadata filtering: 11-category taxonomy
A `category` field on every record (`values`, `culture`, `hiring`, `people`,
`engineering`, `product`, `direction`, `commercial`, `leadership`,
`legal_security`, `general`). The UI exposes a multiselect; the retriever
applies a `{"category": {"$in": [...]}}` MongoDB-style filter at query time.
This is genuinely useful: "interview process" + filter `[hiring]` ensures all
sources are from the hiring section.

### Chunking: header-aware + character-windowed fallback
Splits on `#`/`##`/`###` first, tracking the heading stack as
`section_path` (e.g., `Engineering > Hiring > Interview Process`). Long sections
fall back to a 2000-char window with 200-char overlap. The `section_path` is
what makes citations *useful* — users see which subsection an answer came from.

### Prompt design
A short, strict system prompt:
- "Answer ONLY using the numbered context snippets."
- "If the answer is not in context, say you couldn't find it." (anti-hallucination)
- "Cite sources inline with `[1]`, `[2]`."

User-turn message embeds the snippets with title + section_path + URL on every
block, then restates the question. Last 6 conversation turns are included so
follow-ups work naturally.

### Streaming
End-to-end token streaming (Groq SDK → `rag.stream_answer` → Streamlit
buffered placeholder). Perceived latency: ~300ms to first token (Pinecone
search ~80-150ms + rerank ~50ms + Groq TTFT).

## UI / UX choices

- **Streamlit** — single-file UI, deploys free to Streamlit Community Cloud.
- **Native chat primitives** (`st.chat_message`, `st.chat_input`).
- **Citations as expanders** under each answer — title, **category badge**,
  section path, URL, rerank score, and a 400-char snippet preview.
- **Category multiselect** in the sidebar — when active, an inline caption
  ("🔎 filtering to: hiring") shows which scope the LLM saw.
- **Starter prompts** on first load to lower the cold-start barrier.
- **Sidebar controls**: model picker, top-k slider, category filter, clear chat.
- **Error handling**: missing API keys, missing index, and Pinecone/Groq errors
  all surface as inline `st.error` messages — never raw stack traces.

## v0 → Pinecone pivot: what changed and why

The first iteration used **FAISS + MiniLM** locally. We pivoted to
**Pinecone + bge-reranker** for three reasons:

1. **Reranking quality.** A cross-encoder pulls the right chunks to the top
   *after* dense recall. Empirically the off-topic / on-topic gap widened from
   0.156 vs 0.83 (cosine) to 0.016 vs 0.99 (rerank) — 50× more discriminative.
2. **Metadata filtering at the index.** Pinecone's `$in` filter is applied at
   the vector layer; with FAISS we'd have post-filtered, hurting recall.
3. **No local embedding model.** Removing sentence-transformers shrinks the
   container, removes a torch dependency, and simplifies the ingest pipeline.

Trade-offs accepted: ~80-150ms of Pinecone network latency per query, and a
free-tier hosted dependency.

## Bonus features delivered

- **Inline citations** linking to live GitLab pages with section_path
  (transparency).
- **Category badges** on each citation (transparency + product thinking).
- **Metadata-filtered search** via UI multiselect (product thinking).
- **Reranker scores** shown per source (transparency).
- **Anti-hallucination** instruction + over-fetch + rerank threshold in spirit
  (rerank below ~0.02 means clearly off-topic; the system prompt and the
  retrieval geometry both push toward "I couldn't find this").
- **Model picker** for quality/speed tradeoff.
- **Starter prompts** to onboard new users.
- **Header-aware section paths** in citations so users land on the *exact*
  subsection.

## Empirical comparison: FAISS+cosine vs Pinecone+rerank

Same 5 prompts, both pipelines, same Groq model:

| Query | FAISS top score | Pinecone rerank score | Quality change |
|-------|----------------:|----------------------:|----------------|
| Six core values | 0.749 | **1.000** | top-2 are both `/values/`; CREDIT mnemonic intact |
| Async communication | 0.829 | **0.990** | sources unified to `/communication/`; tighter context |
| AI direction (hard case) | 0.603 | **0.982** | retrieval honest about gap, no hallucination |
| Interview process (`--cat hiring`) | n/a | **0.965** | filter held — all 5 sources in `[hiring]` |
| Off-topic ("Super Bowl") | 0.156 | **0.016** | reranker confidence collapses cleanly |

The geometric gap between on/off-topic widened by ~50×, making it much easier
to set a hard "couldn't find" threshold in v1.

## Known limitations / what v1 should add

| Area | Limitation | Planned fix |
|------|-----------|-------------|
| Direction pages | `about.gitlab.com/direction/*` are JS-rendered SPAs with no SSR | Playwright fallback, or harvest the same content from the handbook mirror |
| Confidence threshold | Rerank score isn't yet a hard cutoff | Short-circuit to "couldn't find" when top-1 < 0.1 (skip the LLM call) |
| Coverage | 50 URLs of 4000 indexed | Bump `TOP_N_URLS` + raise category quotas; nightly re-ingest |
| Eval | Manual smoke-tests | Golden Q&A set with rerank-score regressions |
| Persistence | Per-session chat | Optional Supabase backing for opt-in users |

## Verification (acceptance)

```bash
python -m src.discover                                       # sitemap → 50 URLs
python -m src.scraper                                        # 49 .md files
python -m src.ingest                                         # 1501 vectors in Pinecone
python -m src.retriever "What are GitLab's values?"          # top-5 reranked hits
python -m src.rag "What are GitLab's values?"                # streamed grounded answer
python -m src.rag --cat hiring "Interview process?"          # filter passes through
streamlit run app.py                                         # browser chat works
```
