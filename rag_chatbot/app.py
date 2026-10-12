"""Streamlit UI (driving adapter). Run with: uv run streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from rag_chatbot.container import ConfigError, Container, build_container
from rag_chatbot.domain.models import PipelineResult, ReasoningEffort

SINGLE, COMPARE = "Single pipeline", "Compare all three"

st.set_page_config(page_title="RAG Pipeline Chat", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_container() -> Container:
    # One container per server process: embedded Qdrant allows a single client per index directory.
    return build_container()


try:
    container = get_container()
except ConfigError as e:
    st.error(f"Configuration error: {e}")
    st.info("Set the value in `rag_chatbot/.env` (see `.env.sample`), then restart the app.")
    st.stop()

if "service_ready" not in st.session_state:
    with st.status("Preparing pipelines…", expanded=True) as status:
        try:
            container.start(progress=status.write)
        except Exception as e:
            status.update(label="Startup failed", state="error")
            st.exception(e)
            st.stop()
        status.update(label="Pipelines ready", state="complete", expanded=False)
    st.session_state.service_ready = True

service = container.start()
specs = container.specs
llm = container.llm

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", [SINGLE, COMPARE], horizontal=True)

    pipeline = None
    if mode == SINGLE:
        pipeline = st.selectbox(
            "Pipeline",
            list(specs),
            format_func=lambda n: specs[n].label + ("  (unavailable)" if service.unavailable_reason(n) else ""),
        )
        st.caption(specs[pipeline].description)
        if reason := service.unavailable_reason(pipeline):
            st.warning(reason)

    efforts = llm.supported_reasoning_efforts()
    if efforts:
        default = container.settings.default_reasoning_effort
        effort = st.selectbox(
            "Reasoning level",
            efforts,
            index=efforts.index(default) if default in efforts else min(1, len(efforts) - 1),
            format_func=lambda e: e.value,
        )
    else:
        st.selectbox("Reasoning level", ["not supported by model"], disabled=True)
        effort = None

    st.divider()
    st.caption(f"**LLM:** {llm.provider} · `{llm.model}`")
    st.caption(f"**Embeddings:** `{container.settings.embedding_model}`")
    counts = container.index.counts()
    st.caption("**Index:** " + " · ".join(f"{specs[n].label} {c or 0:,}" for n, c in counts.items()))

    col_a, col_b = st.columns(2)
    if col_a.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col_b.button("Rebuild index", use_container_width=True):
        with st.status("Rebuilding index…", expanded=True) as status:
            container.rebuild_index(progress=status.write)
            status.update(label="Index rebuilt", state="complete", expanded=False)
        st.rerun()


# ---------- rendering ----------
def _escape(text: str) -> str:
    # Streamlit renders $...$ as LaTeX; answers about money shouldn't turn into math.
    return text.replace("$", r"\$")


def render_result(result: PipelineResult, effort_label: str) -> None:
    if not result.ok:
        st.error(result.error)
        return
    st.markdown(_escape(result.answer or ""))

    stats = []
    if result.latency_ms is not None:
        stats.append(f"{result.latency_ms / 1000:.1f}s")
    if result.llm_response and (u := result.llm_response.usage):
        tokens = f"{u.input_tokens:,} in / {u.output_tokens:,} out"
        if u.reasoning_tokens:
            tokens += f" ({u.reasoning_tokens:,} reasoning)"
        stats.append(tokens)
    stats.append(f"effort: {effort_label}")
    st.caption(" · ".join(stats))

    with st.expander(f"Sources ({len(result.chunks)})"):
        for i, chunk in enumerate(result.chunks, 1):
            where = f"{chunk.source}" + (f", p. {chunk.page}" if chunk.page else "")
            st.markdown(f"**{i}. {where}**")
            st.text(chunk.text)


def render_assistant(message: dict) -> None:
    results: dict[str, PipelineResult] = message["results"]
    if message["mode"] == SINGLE:
        (name, result), = results.items()
        st.caption(f"**{specs[name].label}**")
        render_result(result, message["effort"])
    else:
        for col, (name, result) in zip(st.columns(len(results)), results.items()):
            with col:
                st.subheader(specs[name].label)
                render_result(result, message["effort"])


# ---------- chat ----------
st.title("RAG Pipeline Chat")
st.caption("Ask about the documents in `data/`. Pick one pipeline, or compare all three side by side.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(_escape(message["content"]))
        else:
            render_assistant(message)

if question := st.chat_input("Ask a question about the documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(_escape(question))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            if mode == SINGLE:
                results = {pipeline: service.ask(pipeline, question, effort)}
            else:
                results = service.ask_all(question, effort)
        message = {
            "role": "assistant",
            "mode": mode,
            "results": results,
            "effort": effort.value if isinstance(effort, ReasoningEffort) else "n/a",
        }
        render_assistant(message)
    st.session_state.messages.append(message)
