"""Evaluations page: estimate and start evaluations, watch progress, browse and compare runs.

All logic (validation, estimates, comparison math, labels, warnings) lives in EvaluationService
(services/evaluation_service.py); this page only calls it and renders.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_chatbot.domain.datasets import TestsetNotFound
from rag_chatbot.domain.evals import DEFAULT_METRICS, RunNotFound, RunStatus
from rag_chatbot.services.evaluation_service import (
    EvaluationEstimate,
    PipelineUnavailable,
    RunFilter,
    StartEvaluationRequest,
)
from rag_chatbot.services.jobs import JobAlreadyRunning, JobNotFound, JobState
from rag_chatbot.services.sdg_service import InvalidRequest
from rag_chatbot.ui.state import container_or_stop

container = container_or_stop()
evals = container.evaluation
sdg = container.sdg
llm = container.llm

ss = st.session_state
ss.setdefault("eval_job_id", None)  # the evaluation this session follows: started here, or seen running
ss.setdefault("eval_job_started_here", False)
ss.setdefault("eval_job_seen_active", False)  # so the page refreshes once when that job ends
ss.setdefault("eval_other_job_seen", False)  # a non-evaluation job locked the form; unlock it when it ends
ss.setdefault("eval_estimate", None)  # (inputs, EvaluationEstimate) for the inputs it was made for
ss.setdefault("eval_form_error", None)
ss.setdefault("eval_started_warnings", [])
ss.setdefault("eval_compare", [])  # run ids to compare (the multiselect's key)
ss.setdefault("eval_goto_compare", None)  # a selection to apply next run, before the multiselect renders

ANY = "(any)"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _when(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _in_pipeline_order(summaries) -> list[str]:
    """Run ids of a batch in pipeline order (Baseline first), so the default reference is the baseline."""
    order = list(container.specs)
    return [s.run_id for s in sorted(summaries, key=lambda s: order.index(s.pipeline) if s.pipeline in order else len(order))]


def _pipeline_label(name: str) -> str:
    spec = container.specs.get(name)
    return spec.label if spec else name


# ---------- active job ----------


@st.fragment(run_every="2s")
def job_panel() -> None:
    active = evals.active_job()
    if active is None and ss.eval_other_job_seen:  # the job that locked the form has finished
        ss.eval_other_job_seen = False
        st.rerun(scope="app")  # the form is outside this fragment
    if active is not None:
        if active.kind != "eval":
            ss.eval_other_job_seen = True
            st.info(f"Another job is running: {active.label}. Evaluations can start when it finishes.")
            return
        if active.job_id != ss.eval_job_id:  # started in another tab: follow it like our own
            ss.eval_job_id, ss.eval_job_started_here, ss.eval_job_seen_active = active.job_id, False, True
            st.rerun(scope="app")  # lock the form, which is outside this fragment
        ss.eval_job_seen_active = True
        st.progress(active.progress or 0.0, text=f"{active.label} — {active.message}")
        return

    if not ss.eval_job_id:
        return
    try:
        status = evals.job_status(ss.eval_job_id)
    except JobNotFound:  # pushed out of the runner's history by later jobs
        ss.eval_job_id, ss.eval_job_started_here, ss.eval_job_seen_active = None, False, False
        return

    elsewhere = "" if ss.eval_job_started_here else " (started in another tab)"
    if status.state is JobState.COMPLETED:
        batch = [i.summary for i in evals.list_runs(RunFilter(batch_id=status.result_id))]
        failed = [s for s in batch if s.status is not RunStatus.COMPLETED]
        message = f"Evaluation finished{elsewhere}: {len(batch) - len(failed)} of {len(batch)} runs completed."
        (st.warning if failed else st.success)(message)
        if ss.eval_job_seen_active:
            ss.eval_goto_compare = _in_pipeline_order(batch)
    else:
        st.error(f"Evaluation failed{elsewhere}: {status.error}")
    if ss.eval_job_seen_active:  # it finished while this panel was watching: refresh the form and the tables
        ss.eval_job_seen_active = False
        st.rerun(scope="app")


st.title("Evaluations")
st.caption(
    "Answer every question in a test set with one or more pipelines, score the answers with the judge "
    "model (Ragas), and compare runs. Evaluations run in the background; one job runs at a time."
)
for warning in evals.warnings():
    st.warning(warning)

job_panel()
for warning in ss.eval_started_warnings:
    st.warning(warning)

busy = evals.active_job() is not None
testsets = sdg.list_testsets()

# ---------- new evaluation ----------

st.subheader("New evaluation")
if not testsets:
    st.info("There are no test sets yet. Generate one on the **Test Sets** page first.")
else:
    ts_labels = {
        i.summary.id: f"{i.summary.name} v{i.summary.version}" + (" · ⚠️ stale" if i.is_stale else "") for i in testsets
    }
    testset_id = st.selectbox("Test set", list(ts_labels), format_func=ts_labels.get, key="eval_testset", disabled=busy)

    st.markdown("**Pipelines**")
    chosen = []
    columns = st.columns(len(evals.available_pipelines()))
    for col, option in zip(columns, evals.available_pipelines()):
        checked = col.checkbox(
            option.label,
            value=option.available,
            key=f"eval_pipe_{option.name}",
            disabled=busy or not option.available,
            help=option.reason,
        )
        if checked and option.available:
            chosen.append(option.name)

    efforts = llm.supported_reasoning_efforts()
    default_effort = container.settings.default_reasoning_effort
    effort = st.selectbox(
        "Reasoning effort",
        [None, *efforts],
        index=([None, *efforts].index(default_effort) if default_effort in efforts else 0),
        format_func=lambda e: "model default" if e is None else e.value,
        key="eval_effort",
        disabled=busy or not efforts,
        help=f"For the answering model ({llm.model}).",
    )
    request = StartEvaluationRequest(testset_id=testset_id, pipelines=tuple(chosen), reasoning_effort=effort)
    inputs = (testset_id, tuple(chosen), effort.value if effort else None)

    if st.button("Estimate", disabled=busy):
        ss.eval_form_error = None
        ss.eval_estimate = None
        if not chosen:
            ss.eval_form_error = "Select at least one pipeline."
        else:
            try:
                ss.eval_estimate = (inputs, evals.estimate_evaluation(request))
            except (InvalidRequest, PipelineUnavailable) as e:
                ss.eval_form_error = str(e)
            except TestsetNotFound as e:
                ss.eval_form_error = f"Test set {e} no longer exists."

    if ss.eval_form_error:
        st.error(ss.eval_form_error)

    # An estimate only counts for the exact inputs it was made for: change anything and it must be redone.
    estimate: EvaluationEstimate | None = None
    if ss.eval_estimate and ss.eval_estimate[0] == inputs:
        estimate = ss.eval_estimate[1]
    if estimate is not None:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Answers to generate", f"{estimate.questions} × {len(estimate.pipelines)}")
            c2.metric(
                "Answering tokens (est.)", f"~{estimate.est_answer_input_tokens + estimate.est_answer_output_tokens:,}"
            )
            c3.metric("Judging tokens (est.)", f"~{estimate.est_judge_input_tokens + estimate.est_judge_output_tokens:,}")
            c4, c5, _ = st.columns(3)
            for col, label, value in (
                (c4, "Answering cost (est.)", estimate.est_answer_cost_usd),
                (c5, "Judging cost (est.)", estimate.est_judge_cost_usd),
            ):
                col.metric(label, f"~${value:,.2f}" if value is not None else "Pricing not configured")
            notes = [f"Answering with `{estimate.answer_model}`", f"judging with `{estimate.judge_model}`"]
            if estimate.calibrated:
                notes.append("calibrated from past runs")
            st.caption(" · ".join(notes))
            for warning in estimate.warnings:
                st.warning(warning)
    elif ss.eval_estimate is not None and not busy:
        st.caption("The inputs changed since the last estimate. Estimate again before running.")

    if st.button("Confirm and run", type="primary", disabled=busy or estimate is None):
        try:
            started = evals.start_evaluation(request)
            ss.eval_job_id, ss.eval_job_started_here, ss.eval_job_seen_active = started.job_id, True, True
            ss.eval_started_warnings = started.warnings
            ss.eval_estimate = None
            ss.eval_form_error = None
        except (InvalidRequest, PipelineUnavailable) as e:
            ss.eval_form_error = str(e)
        except TestsetNotFound as e:
            ss.eval_form_error = f"Test set {e} no longer exists."
        except JobAlreadyRunning:
            ss.eval_form_error = "A job is already running. Try again when it finishes."
        st.rerun()

# ---------- runs ----------

st.subheader("Runs")
all_runs = evals.list_runs()
if not all_runs:
    st.info("No evaluation runs yet.")
    st.stop()

f1, f2, f3 = st.columns(3)
testset_choices = [ANY, *dict.fromkeys(i.summary.testset_id for i in all_runs)]
labels_by_testset = {i.summary.testset_id: i.testset_label for i in all_runs}
f_testset = f1.selectbox("Test set", testset_choices, format_func=lambda v: labels_by_testset.get(v, v), key="eval_f_testset")
f_pipeline = f2.selectbox("Pipeline", [ANY, *container.specs], format_func=lambda v: v if v == ANY else _pipeline_label(v), key="eval_f_pipeline")
f_status = f3.selectbox("Status", [ANY, *(s.value for s in RunStatus)], key="eval_f_status")
filtered = evals.list_runs(
    RunFilter(
        testset_id=None if f_testset == ANY else f_testset,
        pipeline=None if f_pipeline == ANY else f_pipeline,
        status=None if f_status == ANY else RunStatus(f_status),
    )
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Started": _when(i.summary.started_at),
                "Test set": i.testset_label + (" ⚠️ stale" if i.testset_is_stale else ""),
                "Pipeline": _pipeline_label(i.summary.pipeline),
                "Answer model": i.summary.answer_model,
                "Effort": i.summary.reasoning_effort or "default",
                "Judge": i.summary.judge_model,
                "Status": i.summary.status.value,
                **{m: round(v, 3) if (v := i.summary.aggregate.get(m)) is not None else None for m in DEFAULT_METRICS},
                "Run": i.summary.run_id,
            }
            for i in filtered
        ]
    ),
    hide_index=True,
    width="stretch",
)

# ---------- selection ----------

run_labels = {
    i.summary.run_id: f"{_pipeline_label(i.summary.pipeline)} · {i.testset_label} · {_when(i.summary.started_at)} "
    f"· {i.summary.status.value} ({i.summary.run_id[-4:]})"
    for i in all_runs
}
if ss.eval_goto_compare is not None:
    ss.eval_compare = [r for r in ss.eval_goto_compare if r in run_labels]
    ss.eval_goto_compare = None
ss.eval_compare = [r for r in ss.eval_compare if r in run_labels]

batches = list(dict.fromkeys(i.summary.batch_id for i in all_runs))
b1, b2 = st.columns([3, 1])
batch_labels = {
    b: f"{_when(next(i.summary.started_at for i in all_runs if i.summary.batch_id == b))} · "
    + ", ".join(_pipeline_label(i.summary.pipeline) for i in all_runs if i.summary.batch_id == b)
    for b in batches
}
batch = b1.selectbox("Batch", batches, format_func=batch_labels.get, key="eval_batch")
if b2.button("Select batch", width="stretch"):
    ss.eval_goto_compare = _in_pipeline_order([i.summary for i in all_runs if i.summary.batch_id == batch])
    st.rerun()

selected = st.multiselect("Runs to compare", list(run_labels), format_func=run_labels.get, key="eval_compare")
if not selected:
    st.caption("Select one or more runs (or a whole batch) to compare them and see their details.")
    st.stop()

# ---------- compare ----------

st.subheader("Compare")
if ss.get("eval_reference") not in selected:
    ss.eval_reference = selected[0]
reference = st.selectbox("Reference run", selected, format_func=run_labels.get, key="eval_reference")
try:
    comparison = evals.compare_runs(selected, reference)
except RunNotFound as e:
    st.error(f"Run {e} no longer exists.")
    st.stop()

columns: dict[str, list] = {}
names = {}
for r in comparison.run_ids:
    name = comparison.labels[r]
    if name in names.values():  # two runs with the same label: tell them apart
        name = f"{name} ({r[-4:]})"
    names[r] = name
for r in comparison.run_ids:
    columns[names[r]] = [comparison.values[m][r] for m in comparison.metrics]
    if r != comparison.reference_run_id:
        columns[f"Δ {names[r]}"] = [comparison.deltas[m][r] for m in comparison.metrics]
table = pd.DataFrame(columns, index=comparison.metrics)
delta_columns = [c for c in table.columns if c.startswith("Δ ")]


def _delta_color(value) -> str:
    if value is None or pd.isna(value) or value == 0:
        return ""
    return "color: #1a7f37" if value > 0 else "color: #cf222e"


st.dataframe(
    table.style.format(precision=3, na_rep="—").map(_delta_color, subset=delta_columns),
    width="stretch",
)
st.caption(
    f"Δ is relative to the reference run ({names[comparison.reference_run_id]}). "
    "For noise_sensitivity_relevant, lower is better."
)
if comparison.mixed_testsets:
    st.warning("These runs used different test sets, so their scores aren't directly comparable.")
if comparison.incomplete_run_ids:
    st.info(
        "Not completed, so missing or partial scores: "
        + ", ".join(names[r] for r in comparison.incomplete_run_ids)
    )

# ---------- run detail ----------

st.subheader("Run details")
view = selected[0] if len(selected) == 1 else st.selectbox("Show details for", selected, format_func=run_labels.get, key="eval_view")
run = evals.get_run(view)
m = run.manifest


def _usage(u) -> str:
    return "not recorded" if u is None else f"{u.input_tokens:,} in / {u.output_tokens:,} out"


st.markdown(f"### {_pipeline_label(m.pipeline)} · {m.status.value}")
if m.error:
    st.error(m.error)
left, right = st.columns(2)
left.markdown(
    f"- **Run:** `{m.run_id}` (batch `{m.batch_id}`)\n"
    f"- **Test set:** {run_labels[view].split(' · ')[1]}\n"
    f"- **Started:** {_when(m.started_at)}" + (f", finished {_when(m.finished_at)}" if m.finished_at else "") + "\n"
    f"- **Pipeline settings:** `{m.pipeline_settings}`\n"
    f"- **Index:** `{m.index_manifest_digest[:12]}…`"
)
right.markdown(
    f"- **Answer model:** `{m.answer_model}`, effort {m.reasoning_effort or 'default'}\n"
    f"- **Judge:** `{m.judge_model}`" + (f", effort {m.judge_reasoning_effort}" if m.judge_reasoning_effort else "") + "\n"
    f"- **Embeddings (judge):** `{m.embedding_model}` · Ragas {m.ragas_version}\n"
    f"- **Answering tokens:** {_usage(m.answer_usage)}\n"
    f"- **Judging tokens:** {_usage(m.judge_usage)}"
)

if not run.samples:
    st.caption("No per-question results are available for this run.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "#": n,
                    "Question": s.user_input,
                    "Reference": s.reference,
                    "Answer": s.response if s.response is not None else "(no answer)",
                    **{metric: s.scores.get(metric) for metric in DEFAULT_METRICS},
                }
                for n, s in enumerate(run.samples, 1)
            ]
        ).style.format(precision=3, na_rep="—"),
        hide_index=True,
        width="stretch",
    )
    for n, s in enumerate(run.samples, 1):
        with st.expander(f"{n}. Retrieved contexts ({len(s.retrieved_contexts)})" + (" · error" if s.error else "")):
            if s.error:
                st.error(s.error)
            for ctx in s.retrieved_contexts:
                st.text(ctx)
