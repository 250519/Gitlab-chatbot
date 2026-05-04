"""Streamlit chat UI for the GitLab Handbook chatbot.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import time

import streamlit as st

from src.config import (
    CATEGORIES,
    GROQ_AVAILABLE_MODELS,
    GROQ_DEFAULT_MODEL,
    LOCAL_EMBED_MODEL,
    LOCAL_RERANK_MODEL,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    TOP_K_MIN,
)
from src.llm import GroqKeyMissing, stream_chat
from src.rag import build_messages, retrieve
from src.retriever import IndexNotBuilt, _embedder, _reranker

st.set_page_config(
    page_title="GitLab Handbook Chatbot",
    page_icon="📖",
    layout="centered",
)

STARTER_PROMPTS = [
    "What are GitLab's core values?",
    "How does GitLab approach async communication?",
    "Walk me through the engineering hiring process.",
    "What is GitLab's product direction?",
]


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = GROQ_DEFAULT_MODEL
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K_DEFAULT
    if "categories" not in st.session_state:
        st.session_state.categories = []  # empty = no filter
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

                if verbose:
                    st.write(
                        f"Retrieved + reranked **{len(sources)}** chunks "
                        f"in **{t_retrieve:.2f}s**"
                    )
                    for i, s in enumerate(sources, 1):
                        st.write(
                            f"  `[{i}]` _{s.get('category','?')}_  "
                            f"**{s.get('title','')}** — score `{s.get('score',0):.3f}`"
                        )

                status.update(label="💬 Streaming answer from Groq Llama...")
                messages = build_messages(question, sources, history_for_model)
                t_llm_start = time.time()
                tokens = stream_chat(messages, model=st.session_state.model)
                buffer = ""
                for tok in tokens:
                    buffer += tok
                    answer_placeholder.markdown(buffer + "▌")
                answer_placeholder.markdown(buffer)
                t_llm = time.time() - t_llm_start

                total = t_retrieve + t_llm
                status.update(
                    label=(
                        f"✓ Done in {total:.2f}s "
                        f"(retrieve {t_retrieve:.2f}s · LLM {t_llm:.2f}s)"
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

        _render_sources(sources, verbose=verbose)

    st.session_state.messages.append(
        {"role": "assistant", "content": buffer, "sources": sources}
    )


def _render_history(verbose: bool) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"], verbose=verbose)


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
        st.write("**Try one of these to get started:**")
        cols = st.columns(2)
        for i, prompt in enumerate(STARTER_PROMPTS):
            if cols[i % 2].button(prompt, key=f"starter_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

    user_input = st.chat_input("Ask about GitLab...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()


if __name__ == "__main__":
    main()
