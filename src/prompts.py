"""Prompt templates for the GitLab Handbook RAG bot."""
from __future__ import annotations

SYSTEM_PROMPT = """You are GitLab Handbook Assistant, an expert on GitLab's public Handbook and Direction pages.

Rules:
1. Answer ONLY using the numbered context snippets provided below the user's question.
2. If the answer is not contained in the context, say: "I couldn't find this in the GitLab Handbook." Do not guess.
3. Cite sources inline using bracketed numbers like [1], [2] that match the snippet numbers.
4. Be concise and direct. Use bullet points or short sections when it helps clarity.
5. Preserve GitLab's tone: transparent, factual, employee-friendly.
6. Never invent URLs, policies, salaries, names, or dates.
"""


def build_user_prompt(question: str, snippets: list[dict]) -> str:
    """Assemble the user-turn message with retrieved context.

    ``snippets`` is a list of dicts with keys: text, source_url, title, section_path.
    """
    if not snippets:
        return (
            f"Question: {question}\n\n"
            "No context was retrieved. Reply that you couldn't find this in the GitLab Handbook."
        )

    blocks = []
    for i, s in enumerate(snippets, 1):
        header = f"[{i}] {s.get('title', '')} — {s.get('section_path', '')}"
        url = s.get("source_url", "")
        blocks.append(f"{header}\nURL: {url}\n{s['text']}")

    context = "\n\n---\n\n".join(blocks)
    return (
        f"Context snippets from the GitLab Handbook / Direction:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        "Answer using ONLY the context above. Cite sources with [1], [2], etc."
    )
