"""Test Sets page: generate synthetic test sets (estimate, confirm, watch progress) and browse them.

All logic lives in SdgService (services/sdg_service.py); this page only calls it and renders.
"""

from __future__ import annotations

import streamlit as st

from rag_chatbot.domain.datasets import TestsetNotFound
from rag_chatbot.services.jobs import JobAlreadyRunning, JobNotFound, JobState
from rag_chatbot.services.sdg_service import (
    MAX_SIZE,
    MIN_SIZE,
    GenerateTestsetRequest,
    GenerationEstimate,
    InvalidRequest,
    TestsetListItem,
)
from rag_chatbot.ui.state import container_or_stop

container = container_or_stop()
sdg = container.sdg  # never container.start(): this page must not build the vector index

ss = st.session_state
ss.setdefault("sdg_job_id", None)  # the generation this session follows: started here, or seen running
ss.setdefault("sdg_job_started_here", False)
ss.setdefault("sdg_job_seen_active", False)  # so the page refreshes once when that job ends
ss.setdefault("sdg_other_job_seen", False)  # a non-SDG job locked the form; unlock it when it ends
ss.setdefault("sdg_estimate", None)  # (inputs, GenerationEstimate) for the inputs it was made for
ss.setdefault("sdg_form_error", None)
ss.setdefault("sdg_selected", None)  # test set id shown in the detail view (the select box's key)
ss.setdefault("sdg_goto", None)  # a test set to show next run; applied before the select box renders


def _short_time(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _testset_label(item: TestsetListItem) -> str:
    s = item.summary
    return f"{s.name} v{s.version}" + (" · stale" if item.is_stale else "") + f"  ({s.id})"


# ---------- active job ----------


@st.fragment(run_every="2s")
def job_panel() -> None:
    active = sdg.active_job()
    if active is None and ss.sdg_other_job_seen:  # the job that locked the form has finished
        ss.sdg_other_job_seen = False
        st.rerun(scope="app")  # the form is outside this fragment
    if active is not None:
        if active.kind != "sdg":
            ss.sdg_other_job_seen = True
            st.info(f"Another job is running: {active.label}. Generation is available when it finishes.")
            return
        if active.job_id != ss.sdg_job_id:  # started in another tab: follow it like our own
            ss.sdg_job_id, ss.sdg_job_started_here, ss.sdg_job_seen_active = active.job_id, False, True
            st.rerun(scope="app")  # lock the form, which is outside this fragment
        ss.sdg_job_seen_active = True
        st.progress(active.progress or 0.0, text=f"{active.label} — {active.message}")
        return

    if not ss.sdg_job_id:
        return
    try:
        status = sdg.job_status(ss.sdg_job_id)
    except JobNotFound:  # pushed out of the runner's history by later jobs: nothing left to show
        ss.sdg_job_id, ss.sdg_job_started_here, ss.sdg_job_seen_active = None, False, False
        return

    elsewhere = "" if ss.sdg_job_started_here else " (started in another tab)"
    if status.state is JobState.COMPLETED:
        st.success(f"Saved test set `{status.result_id}`{elsewhere}.")
        if ss.sdg_job_seen_active:
            ss.sdg_goto = status.result_id
    else:
        st.error(f"Generation failed{elsewhere}: {status.error}")
    if ss.sdg_job_seen_active:  # it finished while this panel was watching: refresh the form and the list
        ss.sdg_job_seen_active = False
        st.rerun(scope="app")


st.title("Test Sets")
st.caption(
    "Generate synthetic question/answer test sets from the PDFs in `data/` with Ragas, then use them "
    "to evaluate the pipelines. Generation runs in the background; one job runs at a time."
)
job_panel()

busy = sdg.active_job() is not None
items = sdg.list_testsets()

# ---------- generation form ----------

st.subheader("Generate a test set")
with_graph = [i for i in items if i.summary.has_knowledge_graph]
graph_labels = {i.summary.id: _testset_label(i) for i in with_graph}

col_name, col_size = st.columns([3, 1])
name = col_name.text_input("Name", value="core-corpus", key="sdg_name", disabled=busy, help="Letters, digits, - and _")
size = col_size.number_input(
    "Questions", min_value=MIN_SIZE, max_value=MAX_SIZE, value=10, step=1, key="sdg_size", disabled=busy
)
reuse = st.selectbox(
    "Reuse knowledge graph from",
    [None, *graph_labels],
    format_func=lambda v: "Don't reuse: build a new graph" if v is None else graph_labels[v],
    key="sdg_reuse",
    disabled=busy,
    help="Reusing a saved graph skips the most expensive step.",
)
request = GenerateTestsetRequest(name=name, size=int(size), reuse_kg_from=reuse)
inputs = (request.name, request.size, request.reuse_kg_from)

if st.button("Estimate", disabled=busy):
    ss.sdg_form_error = None
    try:
        ss.sdg_estimate = (inputs, sdg.estimate(request))
    except (InvalidRequest, TestsetNotFound) as e:
        ss.sdg_estimate = None
        ss.sdg_form_error = str(e) if isinstance(e, InvalidRequest) else f"Test set {e} no longer exists."

if ss.sdg_form_error:
    st.error(ss.sdg_form_error)

# An estimate only counts for the exact inputs and corpus it was made for: change either and it must be redone.
estimate: GenerationEstimate | None = None
stale_reason = None
if ss.sdg_estimate:
    made_for, made = ss.sdg_estimate
    if made_for != inputs:
        stale_reason = "The inputs changed since the last estimate."
    elif made.corpus_digest != sdg.corpus_digest():
        stale_reason = "The PDFs in `data/` changed since the last estimate."
    else:
        estimate = made

if estimate is not None:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Input tokens (est.)", f"~{estimate.est_input_tokens:,}")
        c2.metric("Output tokens (est.)", f"~{estimate.est_output_tokens:,}")
        c3.metric(
            "Cost (est.)", f"~${estimate.est_cost_usd:,.2f}" if estimate.est_cost_usd is not None else "Pricing not configured"
        )
        notes = [f"Model `{estimate.model}`", f"corpus ~{estimate.corpus_tokens:,} tokens"]
        if estimate.reuses_kg:
            notes.append("reusing a saved knowledge graph")
        if estimate.calibrated:
            notes.append("calibrated from past usage")
        st.caption(" · ".join(notes))
elif stale_reason and not busy:
    st.caption(f"{stale_reason} Estimate again before generating.")

if st.button("Confirm and generate", type="primary", disabled=busy or estimate is None):
    try:
        ss.sdg_job_id = sdg.start_generation(request)
        ss.sdg_job_started_here = True
        ss.sdg_job_seen_active = True
        ss.sdg_estimate = None
        ss.sdg_form_error = None
    except InvalidRequest as e:
        ss.sdg_form_error = str(e)
    except TestsetNotFound as e:
        ss.sdg_form_error = f"Test set {e} no longer exists."
    except JobAlreadyRunning:
        ss.sdg_form_error = "A job is already running. Try again when it finishes."
    st.rerun()

# ---------- saved test sets ----------

st.subheader("Saved test sets")
if not items:
    st.info("No test sets yet. Generate one above.")
    st.stop()

st.dataframe(
    [
        {
            "Name": i.summary.name,
            "Version": i.summary.version,
            "Created": _short_time(i.summary.created_at),
            "Questions": (
                f"{i.summary.size} of {i.summary.requested_size}"
                if i.summary.requested_size and i.summary.requested_size != i.summary.size
                else str(i.summary.size)
            ),
            "Graph saved": "yes" if i.summary.has_knowledge_graph else "no",
            "Status": "⚠️ Stale" if i.is_stale else "Current",
            "ID": i.summary.id,
        }
        for i in items
    ],
    hide_index=True,
    width="stretch",
)

labels = {i.summary.id: _testset_label(i) for i in items}
if ss.sdg_goto in labels:
    ss.sdg_selected = ss.sdg_goto
ss.sdg_goto = None
if ss.sdg_selected not in labels:
    ss.sdg_selected = items[0].summary.id
selected_id = st.selectbox("View test set", list(labels), format_func=labels.get, key="sdg_selected")

# ---------- detail ----------

item = next(i for i in items if i.summary.id == selected_id)
testset = sdg.get_testset(selected_id)
m = testset.manifest

st.markdown(f"### {m.name} v{m.version}")
if item.is_stale:
    st.warning(
        "**Stale:** the PDFs in `data/` have changed since this test set was made. Its questions may refer to "
        "content that no longer exists, so scores from it aren't comparable with runs on the current corpus."
    )

left, right = st.columns(2)
with left:
    st.markdown(
        f"- **ID:** `{m.id}`\n"
        f"- **Created:** {_short_time(m.created_at)}\n"
        f"- **Questions:** {m.size}"
        + (f" (requested {m.requested_size})" if m.requested_size and m.requested_size != m.size else "")
        + f"\n- **Generator:** `{m.generator.llm_model}`, embeddings `{m.generator.embedding_model}`"
        + (f", effort {m.generator.reasoning_effort}" if m.generator.reasoning_effort else "")
        + f"\n- **Ragas:** {m.generator.ragas_version}"
    )
with right:
    usage = f"{m.usage.input_tokens:,} in / {m.usage.output_tokens:,} out" if m.usage else "not recorded"
    st.markdown(f"- **Token usage:** {usage}\n- **Corpus files:** {len(m.corpus.files)}")
    for label, target in (("Previous version", m.parent_id), ("Knowledge graph from", m.kg_source_id)):
        if target:
            if target in labels:
                if st.button(f"{label}: {labels[target]}", key=f"goto-{label}"):
                    ss.sdg_goto = target
                    st.rerun()
            else:
                st.markdown(f"- **{label}:** `{target}` (not found)")

with st.expander(f"Corpus files ({len(m.corpus.files)})"):
    for filename, digest in sorted(m.corpus.files.items()):
        st.markdown(f"- `{filename}` — sha256 `{digest[:12]}…`")

st.markdown(f"#### Questions ({len(testset.samples)})")
st.dataframe(
    [
        {"#": n, "Question": s.user_input, "Reference answer": s.reference, "Type": s.synthesizer_name}
        for n, s in enumerate(testset.samples, 1)
    ],
    hide_index=True,
    width="stretch",
)
for n, s in enumerate(testset.samples, 1):
    with st.expander(f"{n}. Reference contexts ({len(s.reference_contexts)})"):
        for ctx in s.reference_contexts:
            st.text(ctx)
