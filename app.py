"""Streamlit chat UI for the GitLab Handbook chatbot.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import (
    CATEGORIES,
    GROQ_AVAILABLE_MODELS,
    GROQ_DEFAULT_MODEL,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    TOP_K_MIN,
)
from src.llm import GroqKeyMissing
from src.rag import stream_answer
from src.retriever import IndexNotBuilt

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


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            cat = s.get("category", "")
            cat_badge = f" `{cat}`" if cat else ""
            st.markdown(
                f"**[{i}] {s.get('title', '')}**{cat_badge} — _{s.get('section_path', '')}_  \n"
                f"[{s.get('source_url', '')}]({s.get('source_url', '')})  \n"
                f"<sub>relevance: {s.get('score', 0):.3f}</sub>",
                unsafe_allow_html=True,
            )
            snippet = s.get("text", "")
            if len(snippet) > 400:
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
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption(
            "Vector DB: **Pinecone** (multilingual-e5-large)  \n"
            "Reranker: **bge-reranker-v2-m3**  \n"
            "LLM: open-source **Llama** on **Groq**"
        )


def _handle_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history_for_model = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    cat_filter = st.session_state.categories or None

    with st.chat_message("assistant"):
        if cat_filter:
            st.caption(f"🔎 filtering to: {', '.join(cat_filter)}")
        placeholder = st.empty()
        try:
            tokens, sources = stream_answer(
                question,
                history=history_for_model,
                k=st.session_state.top_k,
                model=st.session_state.model,
                categories=cat_filter,
            )
            buffer = ""
            for tok in tokens:
                buffer += tok
                placeholder.markdown(buffer + "▌")
            placeholder.markdown(buffer)
        except GroqKeyMissing as e:
            placeholder.error(str(e))
            return
        except IndexNotBuilt as e:
            placeholder.error(str(e))
            return
        except Exception as e:  # pragma: no cover
            placeholder.error(f"Something went wrong: {e}")
            return

        _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": buffer, "sources": sources}
    )


def main() -> None:
    _init_state()
    _sidebar()

    st.title("📖 GitLab Handbook Chatbot")
    st.caption(
        "Ask anything about GitLab's [Handbook](https://handbook.gitlab.com/) or "
        "[Direction](https://about.gitlab.com/direction/). "
        "Answers are grounded in **Pinecone-indexed** content with reranked retrieval and inline citations."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    if not st.session_state.messages:
        st.write("**Try one of these to get started:**")
        cols = st.columns(2)
        for i, prompt in enumerate(STARTER_PROMPTS):
            if cols[i % 2].button(prompt, key=f"starter_{i}", use_container_width=True):
                _handle_question(prompt)
                st.rerun()

    user_input = st.chat_input("Ask about GitLab...")
    if user_input:
        _handle_question(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
