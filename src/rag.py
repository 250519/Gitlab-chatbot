"""End-to-end RAG: retrieve -> rerank -> prompt -> stream answer.

CLI:
    python -m src.rag "What are GitLab's values?"
    python -m src.rag --cat engineering "How do you do code review?"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Iterator

from src.config import (
    CONFIDENCE_HIGH_ABOVE,
    CONFIDENCE_LOW_BELOW,
    CONFIDENCE_REFUSE_BELOW,
    GROQ_DEFAULT_MODEL,
    GROQ_FAST_MODEL,
    RETRIEVE_FETCH_K,
    TOP_K_DEFAULT,
)
from src.llm import chat_once, stream_chat
from src.prompts import build_user_prompt, system_prompt_for
from src.retriever import search

REFUSAL_MESSAGE = (
    "I couldn't find anything relevant in the indexed Handbook content for that "
    "question. Try rephrasing, broaden the category filter in the sidebar, or "
    "check whether the topic is in the part of the Handbook that's been indexed "
    "(currently: values, culture, hiring, engineering)."
)


def classify_confidence(snippets: list[dict]) -> str:
    """Classify retrieval confidence into one of: refuse / low / medium / high.

    Uses the top rerank score (bge-reranker-v2-m3 logits). See thresholds in
    ``src.config``. Empty snippet list always refuses.
    """
    if not snippets:
        return "refuse"
    top = max((s.get("score", 0.0) for s in snippets), default=0.0)
    if top < CONFIDENCE_REFUSE_BELOW:
        return "refuse"
    if top < CONFIDENCE_LOW_BELOW:
        return "low"
    if top >= CONFIDENCE_HIGH_ABOVE:
        return "high"
    return "medium"


def retrieve(
    question: str,
    k: int = TOP_K_DEFAULT,
    categories: list[str] | None = None,
    fetch_k: int = RETRIEVE_FETCH_K,
) -> list[dict]:
    return search(question, k=k, fetch_k=fetch_k, categories=categories)


def build_messages(
    question: str,
    snippets: list[dict],
    history: list[dict] | None = None,
    persona: str | None = None,
) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system_prompt_for(persona)}]
    if history:
        for h in history[-6:]:
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": build_user_prompt(question, snippets)})
    return msgs


def stream_answer(
    question: str,
    history: list[dict] | None = None,
    k: int = TOP_K_DEFAULT,
    model: str = GROQ_DEFAULT_MODEL,
    categories: list[str] | None = None,
) -> tuple[Iterator[str], list[dict]]:
    """Run retrieval and start streaming. Returns (token_iterator, snippets_used)."""
    snippets = retrieve(question, k=k, categories=categories)
    messages = build_messages(question, snippets, history)
    return stream_chat(messages, model=model), snippets


def answer(
    question: str,
    history: list[dict] | None = None,
    k: int = TOP_K_DEFAULT,
    model: str = GROQ_DEFAULT_MODEL,
    categories: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """Blocking variant — collects the full answer string."""
    tokens, snippets = stream_answer(
        question, history=history, k=k, model=model, categories=categories
    )
    text = "".join(tokens)
    return text, snippets


_FOLLOWUP_SYSTEM = (
    "You generate short follow-up questions a user might ask next, given the "
    "previous question, the assistant's answer, and the source context. "
    "Return ONLY a JSON array of 3 strings — no prose, no markdown fences. "
    "Each follow-up must be a single concise question under 90 characters, "
    "stay on the GitLab Handbook topic, and explore a different angle than "
    "the previous question."
)


def _extract_json_array(text: str) -> list[str]:
    """Extract a JSON array of strings from a model response. Tolerant to
    leading prose, trailing commas, and ```json fences."""
    # Strip code fences.
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
    # Find the first balanced [ ... ] span.
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try a tolerant fallback: strip trailing commas.
        try:
            data = json.loads(re.sub(r",(\s*[\]}])", r"\1", raw))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [str(s).strip() for s in data if isinstance(s, (str, int, float))]


def suggest_followups(
    question: str,
    answer_text: str,
    sources: list[dict],
    model: str = GROQ_FAST_MODEL,
    n: int = 3,
) -> list[str]:
    """Generate ``n`` short follow-up questions. Returns ``[]`` on failure (fail open)."""
    if not answer_text or not sources:
        return []
    titles = "; ".join(
        f"[{i}] {s.get('title', '')}" for i, s in enumerate(sources[:5], 1)
    )
    user_prompt = (
        f"Previous question: {question}\n\n"
        f"Assistant answer:\n{answer_text[:1500]}\n\n"
        f"Source titles cited: {titles}\n\n"
        f"Return a JSON array of exactly {n} short follow-up questions."
    )
    try:
        raw = chat_once(
            messages=[
                {"role": "system", "content": _FOLLOWUP_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.4,
            max_tokens=200,
        )
    except Exception:
        return []
    items = _extract_json_array(raw)
    return items[:n]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument(
        "--cat",
        action="append",
        help="Restrict to a category (repeatable). e.g. --cat engineering --cat values",
    )
    p.add_argument("-k", type=int, default=TOP_K_DEFAULT)
    args = p.parse_args()

    q = " ".join(args.question) or "What are GitLab's values?"
    cat_filter = args.cat if args.cat else None
    cat_label = f"  [filter: {','.join(cat_filter)}]" if cat_filter else ""
    print(f"Q: {q}{cat_label}\n")

    tokens, snippets = stream_answer(q, k=args.k, categories=cat_filter)
    for tok in tokens:
        print(tok, end="", flush=True)

    print("\n\nSources:")
    for i, s in enumerate(snippets, 1):
        print(f"  [{i}] [{s.get('category','?')}] {s.get('title','')} > "
              f"{s.get('section_path','')}")
        print(f"      {s.get('source_url','')}  (rerank={s.get('score',0):.3f})")
