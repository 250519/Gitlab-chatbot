"""End-to-end RAG: retrieve -> rerank -> prompt -> stream answer.

CLI:
    python -m src.rag "What are GitLab's values?"
    python -m src.rag --cat engineering "How do you do code review?"
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterator

from src.config import GROQ_DEFAULT_MODEL, RETRIEVE_FETCH_K, TOP_K_DEFAULT
from src.llm import stream_chat
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retriever import search


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
) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
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
