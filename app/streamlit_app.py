"""Streamlit chat UI. A pure client of the FastAPI backend -- it never imports the RAG
pipeline directly, only calls /chat over HTTP, so the UI and the pipeline stay decoupled.

Note: retrieval is single-turn. Each question is retrieved and answered independently of
prior turns in the conversation; there's no query rewriting to fold earlier turns into a
follow-up question. The chat history is displayed for a natural feel, but a follow-up like
"how do I do that with async?" won't have the earlier question's context available to
retrieval. That's a deliberate scope cut -- this project's focus is the chunking/retrieval
eval, not conversational query rewriting.
"""

import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

st.set_page_config(page_title="FastAPI Docs RAG", page_icon="⚡", layout="wide")


@st.cache_data(ttl=300)
def get_strategies() -> dict | None:
    try:
        resp = requests.get(f"{config.API_BASE_URL}/strategies", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def call_chat(question: str, chunking_strategy: str, retrieval_method: str) -> dict:
    resp = requests.post(
        f"{config.API_BASE_URL}/chat",
        json={"question": question, "chunking_strategy": chunking_strategy, "retrieval_method": retrieval_method},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def render_sources(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            breadcrumb = " > ".join(c["heading_path"]) if c["heading_path"] else c["doc_path"]
            st.markdown(f"**[[{c['index']}]]** [{breadcrumb}]({c['url']})")


def render_retrieved_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return
    with st.expander(f"Retrieved chunks ({len(chunks)}) -- what actually fed the answer"):
        for i, c in enumerate(chunks, start=1):
            breadcrumb = " > ".join(c["heading_path"]) if c["heading_path"] else c["doc_path"]
            st.markdown(f"{i}. score={c['score']:.3f} -- [{breadcrumb}]({c['url']})")


st.title("⚡ FastAPI Docs RAG")
st.caption(
    "Chat with FastAPI's documentation. Answers are grounded in retrieved excerpts and cited "
    "back to the exact section -- pick a retrieval strategy in the sidebar to compare."
)

strategies_data = get_strategies()

if strategies_data is None:
    st.error(
        f"Can't reach the backend at `{config.API_BASE_URL}`. Start it first:\n\n"
        "```bash\npython -m uvicorn src.api.main:app --reload\n```"
    )
    st.stop()

with st.sidebar:
    st.subheader("Retrieval strategy")
    options = strategies_data["strategies"]
    labels = [o["label"] for o in options]
    default_label = strategies_data["default"]["label"]
    selected_label = st.selectbox("Config", labels, index=labels.index(default_label))
    selected = next(o for o in options if o["label"] == selected_label)

    st.caption(
        "**fixed_size + dense**: naive token-window chunking, baseline.\n\n"
        "**heading_based + dense**: chunks split on doc section boundaries.\n\n"
        "**heading_based + hybrid**: heading-based chunks, dense + BM25 fused via RRF."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message["citations"])
            render_retrieved_chunks(message["retrieved_chunks"])

if prompt := st.chat_input("Ask about FastAPI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                result = call_chat(prompt, selected["chunking_strategy"], selected["retrieval_method"])
            except requests.exceptions.RequestException as e:
                st.error(f"Request to the backend failed: {e}")
                st.stop()

        st.markdown(result["answer"])
        render_sources(result["citations"])
        render_retrieved_chunks(result["retrieved_chunks"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "retrieved_chunks": result["retrieved_chunks"],
        }
    )
