"""Streamlit chat UI for the GitLab Handbook chatbot.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import time

import streamlit as st

from src.config import (
    CATEGORIES,
    DEFAULT_PERSONA,
    GROQ_AVAILABLE_MODELS,
    GROQ_DEFAULT_MODEL,
    LOCAL_EMBED_MODEL,
    LOCAL_RERANK_MODEL,
    PERSONAS,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    TOP_K_MIN,
)
from src.guardrails import verify_citations
from src.llm import GroqKeyMissing, stream_chat
from src.rag import (
    REFUSAL_MESSAGE,
    build_messages,
    classify_confidence,
    retrieve,
    suggest_followups,
)
from src.retriever import IndexNotBuilt, _embedder, _reranker

CONFIDENCE_BADGES = {
    "high":   ("🟢 Confidence: High",   "green"),
    "medium": ("🟡 Confidence: Medium", "orange"),
    "low":    ("🟠 Confidence: Low",    "red"),
}

st.set_page_config(
    page_title="GitLab Handbook Chatbot",
    page_icon="📖",
    layout="centered",
)

def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = GROQ_DEFAULT_MODEL
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K_DEFAULT
    if "persona" not in st.session_state:
        st.session_state.persona = DEFAULT_PERSONA
    if "categories" not in st.session_state:
        # Persona-driven default; user can override via the multiselect.
        st.session_state.categories = list(
            PERSONAS[DEFAULT_PERSONA]["default_categories"]
        )
    if "_persona_for_categories" not in st.session_state:
        st.session_state._persona_for_categories = DEFAULT_PERSONA
    if "verbose" not in st.session_state:
        st.session_state.verbose = False
    if "_prewarmed" not in st.session_state:
        st.session_state._prewarmed = False


def _ensure_models_loaded() -> None:
    """Pre-load embedder + reranker before chat is allowed.

    First run downloads ~2.3 GB for the reranker (cached to ~/.cache/huggingface/)
    and ~120 MB for the small embedder, then loads both onto MPS/CUDA/CPU.
    Subsequent app launches are instant.
    """
    if st.session_state._prewarmed:
        return

    with st.status("Loading models (one-time)...", expanded=True) as status:
        t0 = time.time()
        st.write(f"Loading embedder **`{LOCAL_EMBED_MODEL}`** (~120 MB)...")
        emb = _embedder()
        st.write(
            f"Loading reranker **`{LOCAL_RERANK_MODEL}`** "
            "(~2.3 GB on first run; subsequent runs are instant)..."
        )
        rer = _reranker()
        # Warm forward pass so MPS/CUDA kernel compile happens here, not on
        # the user's first question. The reranker compile is the expensive
        # one (~few minutes on MPS for XLM-R-large the first time).
        st.write("Warming up GPU kernels (compiles once per session)...")
        emb.encode("warmup", normalize_embeddings=True, convert_to_numpy=True)
        rer.predict([("warmup question", "warmup passage")])
        elapsed = time.time() - t0
        status.update(
            label=f"Models ready ({elapsed:.1f}s)",
            state="complete",
            expanded=False,
        )
    st.session_state._prewarmed = True


def _render_sources(sources: list[dict], verbose: bool = False) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=verbose):
        for i, s in enumerate(sources, 1):
            cat = s.get("category", "")
            cat_badge = f" `{cat}`" if cat else ""
            st.markdown(
                f"**[{i}] {s.get('title', '')}**{cat_badge} — _{s.get('section_path', '')}_  \n"
                f"[{s.get('source_url', '')}]({s.get('source_url', '')})  \n"
                f"<sub>rerank score: {s.get('score', 0):.3f}  ·  "
                f"chunk #{s.get('chunk_index', '?')}</sub>",
                unsafe_allow_html=True,
            )
            snippet = s.get("text", "")
            if not verbose and len(snippet) > 400:
                snippet = snippet[:400].rstrip() + "..."
            st.caption(snippet)


def _sidebar() -> None:
    with st.sidebar:
        st.header("Settings")
        persona_options = list(PERSONAS.keys())
        new_persona = st.radio(
            "I'm a…",
            persona_options,
            index=persona_options.index(st.session_state.persona),
            help=(
                "Tunes the assistant's tone, the default category filter, and "
                "the starter prompts on the empty state. You can still override "
                "the category filter manually below."
            ),
        )
        # If the user just switched persona, reset the category filter to that
        # persona's defaults — but only when the categories still match the
        # PREVIOUS persona's defaults (i.e. the user hasn't manually customized).
        if new_persona != st.session_state.persona:
            previous_default = list(
                PERSONAS[st.session_state._persona_for_categories]["default_categories"]
            )
            if list(st.session_state.categories) == previous_default:
                st.session_state.categories = list(
                    PERSONAS[new_persona]["default_categories"]
                )
                st.session_state._persona_for_categories = new_persona
            st.session_state.persona = new_persona
        st.session_state.model = st.selectbox(
            "LLM (Groq)",
            GROQ_AVAILABLE_MODELS,
            index=GROQ_AVAILABLE_MODELS.index(st.session_state.model),
            help="Open-source Llama on Groq. 70B = best quality, 8B = fastest.",
        )
        st.session_state.top_k = st.slider(
            "Top-k (after rerank)",
            min_value=TOP_K_MIN,
            max_value=TOP_K_MAX,
            value=st.session_state.top_k,
            help="Final number of chunks sent to the LLM after Pinecone reranking.",
        )
        st.session_state.categories = st.multiselect(
            "Filter by section",
            options=CATEGORIES,
            default=st.session_state.categories,
            help="Restrict retrieval to specific Handbook sections. Empty = search all.",
        )
        st.session_state.verbose = st.checkbox(
            "Verbose mode",
            value=st.session_state.verbose,
            help="Show retrieve/rerank/LLM timings and full untruncated source text.",
        )
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption(
            f"Embedder: **`{LOCAL_EMBED_MODEL}`** (local)  \n"
            f"Reranker: **`{LOCAL_RERANK_MODEL}`** (local)  \n"
            "Vector DB: **Pinecone**  \n"
            "LLM: open-source **Llama** on **Groq**"
        )


def _generate_assistant_reply(question: str) -> None:
    """Generate the assistant reply for the latest user question.

    Renders inline (caller is expected to be inside the scrollable container).
    Shows a live st.status widget that progresses through retrieve → rerank →
    LLM streaming. Verbose mode keeps it expanded with per-source diagnostics.
    Appends the final assistant message to ``st.session_state.messages``.
    """
    history_for_model = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    cat_filter = st.session_state.categories or None
    verbose = st.session_state.verbose

    with st.chat_message("assistant"):
        if cat_filter:
            st.caption(f"🔎 filtering to: {', '.join(cat_filter)}")

        status = st.status("🔍 Retrieving candidates from Pinecone...", expanded=verbose)
        badge_placeholder = st.empty()
        answer_placeholder = st.empty()
        try:
            with status:
                t_retrieve_start = time.time()
                sources = retrieve(
                    question,
                    k=st.session_state.top_k,
                    categories=cat_filter,
                )
                t_retrieve = time.time() - t_retrieve_start
                confidence = classify_confidence(sources)

                if verbose:
                    top_score = max((s.get("score", 0.0) for s in sources), default=0.0)
                    st.write(
                        f"Retrieved + reranked **{len(sources)}** chunks "
                        f"in **{t_retrieve:.2f}s**  ·  "
                        f"top score `{top_score:.3f}` → **{confidence}**"
                    )
                    for i, s in enumerate(sources, 1):
                        st.write(
                            f"  `[{i}]` _{s.get('category','?')}_  "
                            f"**{s.get('title','')}** — score `{s.get('score',0):.3f}`"
                        )

                # Hard refusal: skip LLM call entirely.
                if confidence == "refuse":
                    status.update(
                        label="✗ No relevant context found — refusing.",
                        state="error",
                        expanded=False,
                    )
                    badge_placeholder.markdown(":red[**🚫 Out of corpus — no answer**]")
                    answer_placeholder.markdown(REFUSAL_MESSAGE)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": REFUSAL_MESSAGE,
                        "sources": [],
                        "confidence": "refuse",
                    })
                    return

                badge_label, badge_color = CONFIDENCE_BADGES[confidence]
                badge_placeholder.markdown(f":{badge_color}[**{badge_label}**]")

                status.update(label="💬 Streaming answer from Groq Llama...")
                messages = build_messages(
                    question,
                    sources,
                    history_for_model,
                    persona=st.session_state.persona,
                )
                t_llm_start = time.time()
                tokens = stream_chat(messages, model=st.session_state.model)
                buffer = ""
                for tok in tokens:
                    buffer += tok
                    answer_placeholder.markdown(buffer + "▌")
                answer_placeholder.markdown(buffer)
                t_llm = time.time() - t_llm_start

                status.update(label="🔎 Verifying citations...")
                citation_report = verify_citations(buffer, sources)

                status.update(label="💡 Suggesting follow-ups...")
                followups = suggest_followups(question, buffer, sources)

                total = t_retrieve + t_llm
                status.update(
                    label=(
                        f"✓ Done in {total:.2f}s "
                        f"(retrieve {t_retrieve:.2f}s · LLM {t_llm:.2f}s) · "
                        f"{citation_report['label']}"
                    ),
                    state="complete",
                    expanded=verbose,
                )
        except GroqKeyMissing as e:
            status.update(label="✗ Groq API key missing", state="error")
            answer_placeholder.error(str(e))
            return
        except IndexNotBuilt as e:
            status.update(label="✗ Pinecone index not ready", state="error")
            answer_placeholder.error(str(e))
            return
        except Exception as e:  # pragma: no cover
            status.update(label=f"✗ {type(e).__name__}", state="error")
            answer_placeholder.error(f"Something went wrong: {e}")
            return

        _render_citation_badge(citation_report, verbose=verbose)
        _render_sources(sources, verbose=verbose)
        _render_followups(followups, key_prefix=f"live_{len(st.session_state.messages)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": buffer,
        "sources": sources,
        "confidence": confidence,
        "citation_report": citation_report,
        "followups": followups,
    })


def _render_followups(followups: list[str], key_prefix: str) -> None:
    if not followups:
        return
    st.caption("💡 Follow-up suggestions")
    cols = st.columns(min(3, len(followups)))
    for i, q in enumerate(followups):
        if cols[i % len(cols)].button(
            q,
            key=f"{key_prefix}_followup_{i}",
            use_container_width=True,
        ):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()


def _render_citation_badge(report: dict, verbose: bool) -> None:
    badge = report.get("badge")
    label = report.get("label", "")
    color_for = {
        "grounded":    "green",
        "partial":     "orange",
        "unsupported": "red",
        "uncited":     "gray",
    }
    color = color_for.get(badge, "gray")
    st.markdown(f":{color}[**{label}**]")
    if not verbose or not report.get("checks"):
        return
    with st.expander("Citation details", expanded=False):
        for c in report["checks"]:
            mark = "❌" if c.out_of_range else ("✅" if c.supported else "⚠️")
            score_txt = (
                "out of range"
                if c.out_of_range
                else f"score `{c.score:.2f}`"
            )
            sentence = c.sentence
            if len(sentence) > 240:
                sentence = sentence[:240].rstrip() + "..."
            st.markdown(
                f"{mark} `[{c.citation}]`  ·  {score_txt}  \n"
                f"<sub>{sentence}</sub>",
                unsafe_allow_html=True,
            )


def _render_history(verbose: bool) -> None:
    last_idx = len(st.session_state.messages) - 1
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            confidence = msg.get("confidence")
            if confidence == "refuse":
                st.markdown(":red[**🚫 Out of corpus — no answer**]")
            elif confidence in CONFIDENCE_BADGES:
                label, color = CONFIDENCE_BADGES[confidence]
                st.markdown(f":{color}[**{label}**]")
            st.markdown(msg["content"])
            if msg.get("citation_report"):
                _render_citation_badge(msg["citation_report"], verbose=verbose)
            if msg.get("sources"):
                _render_sources(msg["sources"], verbose=verbose)
            # Show follow-up chips only on the latest assistant message so old
            # turns aren't cluttered with stale suggestions.
            if (
                idx == last_idx
                and msg["role"] == "assistant"
                and msg.get("followups")
            ):
                _render_followups(msg["followups"], key_prefix=f"hist_{idx}")


def main() -> None:
    _init_state()
    _sidebar()

    st.title("📖 GitLab Handbook Chatbot")
    st.caption(
        "Ask anything about GitLab's [Handbook](https://handbook.gitlab.com/) or "
        "[Direction](https://about.gitlab.com/direction/). "
        "Answers are grounded in **Pinecone-indexed** content with reranked retrieval and inline citations."
    )

    _ensure_models_loaded()

    # Scrollable chat history (fixed height; auto-scrolls to bottom on new content).
    chat_box = st.container(height=560, border=True)
    with chat_box:
        _render_history(verbose=st.session_state.verbose)

        # If the last message is from the user with no reply yet, generate one
        # inline so the streaming answer renders inside the scrollable container.
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
        ):
            _generate_assistant_reply(st.session_state.messages[-1]["content"])

    if not st.session_state.messages:
        starter_prompts = PERSONAS[st.session_state.persona]["starter_prompts"]
        st.write(f"**Try one of these — tailored for: _{st.session_state.persona}_**")
        cols = st.columns(2)
        for i, prompt in enumerate(starter_prompts):
            if cols[i % 2].button(prompt, key=f"starter_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

    user_input = st.chat_input("Ask about GitLab...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()


if __name__ == "__main__":
    main()
