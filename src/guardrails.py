"""Post-generation guardrails: verify the LLM's `[N]` citations are grounded.

The retrieval pipeline returns the top-k chunks the LLM is *supposed* to cite.
After streaming finishes we parse the bracketed citation numbers from the
answer, sanity-check that each refers to a real source, and score the sentence
containing the citation against the cited chunk via the same cross-encoder
used for reranking. Sentences scoring below ``CITATION_SUPPORT_THRESHOLD`` are
flagged as weakly supported.

This is a transparency feature — we don't block the response; we surface a
single badge plus per-citation detail in verbose mode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import CITATION_SUPPORT_THRESHOLD

_CITATION_RE = re.compile(r"\[(\d+)\]")
# Crude sentence splitter. We don't need NLTK-grade accuracy here — just
# enough to associate each [N] with the surrounding clause.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class CitationCheck:
    citation: int            # the N in [N]
    snippet_index: int       # zero-based index into sources (citation - 1)
    sentence: str
    score: float             # cross-encoder score sentence vs cited chunk
    supported: bool          # score >= CITATION_SUPPORT_THRESHOLD
    out_of_range: bool       # citation > len(sources)


def _sentences_with_citations(answer: str) -> list[tuple[str, list[int]]]:
    """Return (sentence, [citation_numbers]) for each sentence that has one."""
    out: list[tuple[str, list[int]]] = []
    for sent in _SENTENCE_SPLIT_RE.split(answer.strip()):
        nums = [int(m) for m in _CITATION_RE.findall(sent)]
        if nums:
            out.append((sent.strip(), nums))
    return out


def verify_citations(answer: str, sources: list[dict]) -> dict:
    """Check every [N] in ``answer`` against ``sources[N-1]``.

    Returns:
        {
            "badge": "grounded" | "partial" | "unsupported" | "uncited",
            "label": str,                  # human-readable badge label
            "checks": list[CitationCheck], # one per (sentence, citation) pair
            "n_citations": int,
            "n_supported": int,
            "n_out_of_range": int,
        }
    """
    pairs = _sentences_with_citations(answer)
    if not pairs:
        return {
            "badge": "uncited",
            "label": "⚪ No citations",
            "checks": [],
            "n_citations": 0,
            "n_supported": 0,
            "n_out_of_range": 0,
        }

    # Lazy import to avoid loading the reranker at module-import time.
    from src.retriever import _reranker

    rer = _reranker()

    # Build the (sentence, chunk_text) pairs we need scored. Skip out-of-range
    # citations — they're flagged separately and don't need a score.
    flat: list[CitationCheck] = []
    score_inputs: list[tuple[str, str]] = []
    score_input_idx: list[int] = []

    for sentence, cits in pairs:
        for c in cits:
            idx = c - 1
            if idx < 0 or idx >= len(sources):
                flat.append(
                    CitationCheck(
                        citation=c,
                        snippet_index=idx,
                        sentence=sentence,
                        score=float("-inf"),
                        supported=False,
                        out_of_range=True,
                    )
                )
                continue
            score_inputs.append((sentence, sources[idx].get("text", "")))
            score_input_idx.append(len(flat))
            flat.append(
                CitationCheck(
                    citation=c,
                    snippet_index=idx,
                    sentence=sentence,
                    score=0.0,
                    supported=False,
                    out_of_range=False,
                )
            )

    if score_inputs:
        scores = rer.predict(score_inputs)
        for i, s in zip(score_input_idx, scores):
            flat[i].score = float(s)
            flat[i].supported = float(s) >= CITATION_SUPPORT_THRESHOLD

    n_total = len(flat)
    n_oor = sum(1 for c in flat if c.out_of_range)
    n_supported = sum(1 for c in flat if c.supported and not c.out_of_range)

    if n_oor > 0 and n_supported == 0:
        badge = "unsupported"
        label = f"❌ Unsupported ({n_oor} bad citation{'s' if n_oor != 1 else ''})"
    elif n_supported == n_total:
        badge = "grounded"
        label = f"✅ Grounded ({n_total}/{n_total})"
    elif n_supported == 0:
        badge = "unsupported"
        label = f"❌ Unsupported (0/{n_total})"
    else:
        badge = "partial"
        label = f"⚠️ Partially grounded ({n_supported}/{n_total})"

    return {
        "badge": badge,
        "label": label,
        "checks": flat,
        "n_citations": n_total,
        "n_supported": n_supported,
        "n_out_of_range": n_oor,
    }
