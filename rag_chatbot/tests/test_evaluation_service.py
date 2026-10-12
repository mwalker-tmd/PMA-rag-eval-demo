"""EvaluationService with fakes: in-memory repositories, fake LLM/evaluator/embeddings, in-memory Qdrant."""

import threading
from dataclasses import replace
from datetime import timedelta
from statistics import fmean

import pymupdf
import pytest
from factories import T0, make_run_manifest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from rag_chatbot.adapters.datasets.memory_repository import InMemoryDatasetRepository
from rag_chatbot.adapters.eval_runs.memory_repository import InMemoryEvalRunRepository
from rag_chatbot.adapters.evaluation.fake_evaluator import FakeEvaluator
from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
from rag_chatbot.config import ConfigError
from rag_chatbot.container import build_container, build_index_manager
from rag_chatbot.domain.datasets import TestsetNotFound
from rag_chatbot.domain.errors import LLMRateLimitError, ModelIncompatible
from rag_chatbot.domain.evals import DEFAULT_METRICS, EvaluationError, RunNotFound, RunStatus
from rag_chatbot.domain.models import ReasoningEffort
from rag_chatbot.pipelines import rerank
from rag_chatbot.services.evaluation_service import (
    INTERRUPTED,
    PipelineUnavailable,
    RunFilter,
    StartEvaluationRequest,
)
from rag_chatbot.services.jobs import JobAlreadyRunning, JobState
from rag_chatbot.services.sdg_service import GenerateTestsetRequest, InvalidRequest

TIMEOUT = 30


class FakeRerank:
    """Stands in for CohereRerank so the Rerank pipeline runs offline."""

    def __new__(cls, **_kwargs):
        from test_pipelines import FakeRerank as Compressor

        return Compressor(model="fake")


class FailingLLM(FakeLLMAdapter):
    """Fails to answer any question containing `marker`."""

    def __init__(self, marker):
        super().__init__()
        self.marker = marker

    def generate(self, request):
        if self.marker in request.messages[-1].content:
            raise LLMRateLimitError("throttled")
        return super().generate(request)


class EvaluatorFailingOnCall(FakeEvaluator):
    def __init__(self, failing_call):
        super().__init__()
        self.failing_call = failing_call

    def evaluate(self, samples, metrics=DEFAULT_METRICS, **kwargs):
        if len(self.calls) + 1 == self.failing_call:
            self.calls.append({"samples": len(samples)})
            raise EvaluationError("judge timed out")
        return super().evaluate(samples, metrics, **kwargs)


class IncompatibleJudge(FakeEvaluator):
    def preflight(self):
        raise ModelIncompatible("temperature 1e-08 is not supported")


@pytest.fixture
def make_env(settings, monkeypatch):
    """Build (container, testset_id) from fakes; options swap in a failing LLM or evaluator, or enable Rerank."""

    def make(*, llm=None, evaluator=None, with_rerank=False, size=3, **setting_changes):
        s = replace(settings, eval_row_delay_seconds=0, **setting_changes)
        if with_rerank:
            s = replace(s, cohere_api_key="test-key")
            monkeypatch.setattr(rerank, "CohereRerank", FakeRerank)
        container = build_container(
            s,
            llm=llm or FakeLLMAdapter(),
            embeddings=DeterministicFakeEmbedding(size=32),
            client=QdrantClient(":memory:"),
            embedding_dim=32,
            datasets=InMemoryDatasetRepository(),
            eval_runs=InMemoryEvalRunRepository(),
            evaluator=evaluator or FakeEvaluator(),
        )
        testset = container.sdg.generate(GenerateTestsetRequest(name="core-corpus", size=size))
        return container, testset.id

    return make


def run_batch(container, testset_id, **kwargs):
    started = container.evaluation.start_evaluation(StartEvaluationRequest(testset_id=testset_id, **kwargs))
    status = container.jobs.wait(started.job_id, timeout=TIMEOUT)
    return started, status


# ---------- start and complete ----------


def test_start_creates_one_running_run_per_available_pipeline_in_one_batch(make_env):
    container, testset_id = make_env()
    started = container.evaluation.start_evaluation(StartEvaluationRequest(testset_id=testset_id))
    runs = [container.eval_runs.get(r).manifest for r in started.run_ids]
    assert [m.pipeline for m in runs] == ["baseline", "parent_child"]  # rerank needs COHERE_API_KEY
    assert {m.batch_id for m in runs} == {started.batch_id}
    assert all(m.status in (RunStatus.RUNNING, RunStatus.COMPLETED) for m in runs)  # created RUNNING; may finish fast
    assert container.jobs.wait(started.job_id, timeout=TIMEOUT).state is JobState.COMPLETED


def test_a_batch_completes_with_samples_in_order_and_mean_aggregates(make_env):
    container, testset_id = make_env(with_rerank=True)
    started, status = run_batch(container, testset_id)
    assert status.state is JobState.COMPLETED and status.result_id == started.batch_id
    questions = [s.user_input for s in container.datasets.get(testset_id).samples]

    for run_id in started.run_ids:
        run = container.eval_runs.get(run_id)
        assert run.manifest.status is RunStatus.COMPLETED
        assert [s.user_input for s in run.samples] == questions
        assert all(s.response and s.retrieved_contexts for s in run.samples)
        for metric in DEFAULT_METRICS:
            values = [s.scores[metric] for s in run.samples if s.scores[metric] is not None]
            assert run.manifest.aggregate[metric] == pytest.approx(fmean(values))
    assert [container.eval_runs.get(r).manifest.pipeline for r in started.run_ids] == ["baseline", "rerank", "parent_child"]


def test_each_run_records_what_produced_it(make_env):
    container, testset_id = make_env(llm=FakeLLMAdapter(model="answer-model"))
    started, _ = run_batch(container, testset_id, pipelines=("baseline",), reasoning_effort=ReasoningEffort.LOW)
    m = container.eval_runs.get(started.run_ids[0]).manifest
    assert m.pipeline_settings == container.specs["baseline"].settings
    assert (m.answer_model, m.reasoning_effort) == ("answer-model", "low")
    assert (m.judge_model, m.embedding_model, m.ragas_version) == ("fake-judge", "fake-embeddings", "fake")
    assert m.metrics == list(DEFAULT_METRICS)
    assert m.index_manifest_digest == container.index.manifest_digest() and len(m.index_manifest_digest) == 64
    assert m.answer_usage is not None and m.judge_usage is not None
    assert m.testset_id == testset_id


# ---------- refusals (no runs created) ----------


def test_an_unavailable_pipeline_is_refused_without_creating_runs(make_env):
    container, testset_id = make_env()
    with pytest.raises(PipelineUnavailable, match="COHERE_API_KEY"):
        container.evaluation.start_evaluation(StartEvaluationRequest(testset_id=testset_id, pipelines=("baseline", "rerank")))
    assert container.eval_runs.list() == []


def test_a_running_job_blocks_a_new_evaluation_without_creating_runs(make_env):
    container, testset_id = make_env()
    release = threading.Event()
    blocker = container.jobs.submit("sdg", "Generating", lambda report: release.wait(TIMEOUT) and None)
    with pytest.raises(JobAlreadyRunning):
        container.evaluation.start_evaluation(StartEvaluationRequest(testset_id=testset_id))
    assert container.eval_runs.list() == []
    release.set()
    container.jobs.wait(blocker, timeout=TIMEOUT)


@pytest.mark.parametrize(
    "request_kwargs, error",
    [
        ({"testset_id": "ts_20260101_deadbeef"}, TestsetNotFound),
        ({"pipelines": ("baseline", "nonexistent")}, InvalidRequest),
        ({"reasoning_effort": ReasoningEffort.XHIGH}, InvalidRequest),  # the fake LLM supports low/medium/high
    ],
    ids=["unknown-testset", "unknown-pipeline", "unsupported-effort"],
)
def test_invalid_requests_create_nothing(make_env, request_kwargs, error):
    container, testset_id = make_env()
    with pytest.raises(error):
        container.evaluation.start_evaluation(StartEvaluationRequest(**{"testset_id": testset_id, **request_kwargs}))
    assert container.eval_runs.list() == []


def test_missing_judge_configuration_is_reported_as_an_invalid_request(settings):
    container = build_container(
        replace(settings, eval_provider="ragas", openai_api_key=None),  # the Ragas judge needs a key
        llm=FakeLLMAdapter(),
        embeddings=DeterministicFakeEmbedding(size=32),
        client=QdrantClient(":memory:"),
        embedding_dim=32,
        datasets=InMemoryDatasetRepository(),
        eval_runs=InMemoryEvalRunRepository(),
    )
    testset_id = container.sdg.generate(GenerateTestsetRequest(name="core-corpus", size=2)).id
    with pytest.raises(InvalidRequest, match="Evaluation isn't configured.*OPENAI_API_KEY"):
        container.evaluation.start_evaluation(StartEvaluationRequest(testset_id=testset_id))
    assert container.eval_runs.list() == []


# ---------- failures inside a batch ----------


def test_a_failed_answer_is_recorded_and_the_run_still_completes(make_env):
    container, testset_id = make_env(llm=FailingLLM(marker="[1]"))  # the fake generator numbers its questions
    started, _ = run_batch(container, testset_id, pipelines=("baseline",))
    run = container.eval_runs.get(started.run_ids[0])
    assert run.manifest.status is RunStatus.COMPLETED
    failed = [s for s in run.samples if s.error]
    assert len(failed) == 1 and "throttled" in failed[0].error and failed[0].response is None
    assert set(failed[0].scores.values()) == {None}
    assert all(any(v is not None for v in s.scores.values()) for s in run.samples if not s.error)


def test_a_run_where_every_answer_failed_ends_failed(make_env):
    container, testset_id = make_env(llm=FailingLLM(marker="What"))  # every fake question contains "What"
    started, _ = run_batch(container, testset_id, pipelines=("baseline",))
    run = container.eval_runs.get(started.run_ids[0])
    assert run.manifest.status is RunStatus.FAILED
    assert "No answers to score" in run.manifest.error
    assert len(run.samples) == 3 and all(s.error for s in run.samples)


def test_a_judge_failure_fails_only_that_run(make_env):
    container, testset_id = make_env(evaluator=EvaluatorFailingOnCall(failing_call=2), with_rerank=True)
    started, status = run_batch(container, testset_id)
    statuses = [container.eval_runs.get(r).manifest for r in started.run_ids]
    assert [(m.pipeline, m.status) for m in statuses] == [
        ("baseline", RunStatus.COMPLETED),
        ("rerank", RunStatus.FAILED),
        ("parent_child", RunStatus.COMPLETED),
    ]
    rerank_run = container.eval_runs.get(started.run_ids[1])
    assert "judge timed out" in rerank_run.manifest.error
    assert len(rerank_run.samples) == 3 and all(s.response for s in rerank_run.samples)  # partial results kept
    assert status.state is JobState.COMPLETED


def test_an_incompatible_judge_fails_every_run_before_any_answers(make_env):
    llm = FakeLLMAdapter()
    container, testset_id = make_env(llm=llm, evaluator=IncompatibleJudge())
    started, status = run_batch(container, testset_id)
    for run_id in started.run_ids:
        m = container.eval_runs.get(run_id).manifest
        assert m.status is RunStatus.FAILED and "temperature 1e-08" in m.error
    assert llm.requests == []  # never answered anything
    assert status.state is JobState.FAILED


# ---------- comparison ----------


def _completed(container, run_id, pipeline, testset_id, aggregate, started_at=T0, status=RunStatus.COMPLETED):
    container.eval_runs.create(make_run_manifest(run_id, pipeline=pipeline, testset_id=testset_id, started_at=started_at))
    if status is RunStatus.COMPLETED:
        container.eval_runs.complete(run_id, [], aggregate, judge_usage=None, answer_usage=None)
    elif status is RunStatus.FAILED:
        container.eval_runs.fail(run_id, "boom")


def test_compare_runs_values_deltas_and_flags(make_env):
    container, testset_id = make_env()
    _completed(container, "run_a", "baseline", testset_id, {"faithfulness": 0.5, "answer_relevancy": 0.7})
    _completed(container, "run_b", "parent_child", testset_id, {"faithfulness": 0.8, "answer_relevancy": None})
    _completed(container, "run_c", "baseline", "ts_other", {"faithfulness": 0.2}, status=RunStatus.FAILED)

    c = container.evaluation.compare_runs(["run_a", "run_b"], reference_run_id="run_a")
    assert c.run_ids == ["run_a", "run_b"]
    assert c.values["faithfulness"] == {"run_a": 0.5, "run_b": 0.8}
    assert c.deltas["faithfulness"] == {"run_a": None, "run_b": pytest.approx(0.3)}
    assert c.deltas["answer_relevancy"] == {"run_a": None, "run_b": None}  # missing value: no delta
    assert c.metrics == ["faithfulness", "answer_relevancy"]  # DEFAULT_METRICS order, only those present
    assert c.mixed_testsets is False and c.incomplete_run_ids == []
    assert c.labels["run_a"].startswith("Baseline · core-corpus v1 · ")

    _completed(container, "run_x", "baseline", testset_id, {"faithfulness": 0.4, "custom_metric": 0.9})
    assert container.evaluation.compare_runs(["run_a", "run_x"]).metrics == [
        "faithfulness",
        "answer_relevancy",
        "custom_metric",  # extras after the defaults
    ]

    mixed = container.evaluation.compare_runs(["run_a", "run_c"])
    assert mixed.mixed_testsets is True
    assert mixed.incomplete_run_ids == ["run_c"]
    assert mixed.reference_run_id is None and set(mixed.deltas["faithfulness"].values()) == {None}
    assert mixed.labels["run_c"].startswith("Baseline · ts_other · ")


def test_compare_runs_rejects_bad_selections(make_env):
    container, testset_id = make_env()
    _completed(container, "run_a", "baseline", testset_id, {"faithfulness": 0.5})
    with pytest.raises(InvalidRequest):
        container.evaluation.compare_runs([])
    with pytest.raises(InvalidRequest):
        container.evaluation.compare_runs(["run_a"], reference_run_id="run_other")
    with pytest.raises(RunNotFound):
        container.evaluation.compare_runs(["run_a", "run_20260101-000000_baseline_ffff"])


# ---------- listing, warnings, recovery ----------


def test_list_runs_labels_stale_flags_and_filters(make_env, data_dir):
    container, testset_id = make_env()
    _completed(container, "run_a", "baseline", testset_id, {}, started_at=T0)
    _completed(container, "run_b", "parent_child", testset_id, {}, started_at=T0 + timedelta(seconds=1), status=RunStatus.FAILED)
    _completed(container, "run_c", "baseline", "ts_gone", {}, started_at=T0 + timedelta(seconds=2))

    items = container.evaluation.list_runs()
    assert [i.summary.run_id for i in items] == ["run_c", "run_b", "run_a"]
    labels = {i.summary.run_id: (i.testset_label, i.testset_is_stale) for i in items}
    assert labels == {"run_a": ("core-corpus v1", False), "run_b": ("core-corpus v1", False), "run_c": ("ts_gone (missing)", False)}

    ids = lambda f: [i.summary.run_id for i in container.evaluation.list_runs(f)]  # noqa: E731
    assert ids(RunFilter(testset_id=testset_id)) == ["run_b", "run_a"]
    assert ids(RunFilter(pipeline="baseline")) == ["run_c", "run_a"]
    assert ids(RunFilter(status=RunStatus.FAILED)) == ["run_b"]
    assert ids(RunFilter(batch_id="batch_20261012-090000_cd34", pipeline="parent_child")) == ["run_b"]

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "New PDF.")
    doc.save(str(data_dir / "new.pdf"))
    assert all(i.testset_is_stale for i in container.evaluation.list_runs(RunFilter(testset_id=testset_id)))


@pytest.mark.parametrize("judge, warned", [("fake-model", True), ("gpt-4.1-mini", False)])
def test_the_judge_warning_appears_exactly_when_the_models_match(make_env, judge, warned):
    container, testset_id = make_env(eval_judge_model=judge)  # LLM_MODEL is "fake-model" in tests
    assert any("same as the answering model" in w for w in container.evaluation.warnings()) is warned
    started, _ = run_batch(container, testset_id, pipelines=("baseline",))
    assert any("same as the answering model" in w for w in started.warnings) is warned


def test_a_stale_test_set_is_warned_about_when_starting(make_env, data_dir):
    container, testset_id = make_env()
    started, _ = run_batch(container, testset_id, pipelines=("baseline",))
    assert not any("stale" in w for w in started.warnings)

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Changed corpus.")
    doc.save(str(data_dir / "changed.pdf"))
    started, _ = run_batch(container, testset_id, pipelines=("baseline",))
    assert any("stale" in w for w in started.warnings)


def test_recovery_fails_only_interrupted_runs(make_env):
    container, testset_id = make_env()
    container.eval_runs.create(make_run_manifest("run_interrupted", testset_id=testset_id))
    _completed(container, "run_done", "baseline", testset_id, {"faithfulness": 0.5})
    before = container.eval_runs.get("run_done")

    assert container.evaluation.recover_interrupted_runs() == ["run_interrupted"]
    m = container.eval_runs.get("run_interrupted").manifest
    assert m.status is RunStatus.FAILED and m.error == INTERRUPTED
    assert container.eval_runs.get("run_done") == before
    assert container.evaluation.recover_interrupted_runs() == []


def test_building_a_container_recovers_interrupted_runs(settings):
    runs = InMemoryEvalRunRepository()
    runs.create(make_run_manifest("run_left_running"))
    build_container(
        settings,
        llm=FakeLLMAdapter(),
        embeddings=DeterministicFakeEmbedding(size=32),
        client=QdrantClient(":memory:"),
        embedding_dim=32,
        eval_runs=runs,
    )
    assert runs.get("run_left_running").manifest.error == INTERRUPTED


# ---------- jobs and concurrency ----------


def test_an_evaluation_job_reports_monotonic_progress_and_returns_the_batch(make_env):
    container, testset_id = make_env(with_rerank=True)
    started = container.evaluation.start_evaluation(StartEvaluationRequest(testset_id=testset_id))
    seen = []
    while True:
        s = container.evaluation.job_status(started.job_id)
        if s.progress is not None:
            seen.append(s.progress)
        if not s.state.is_active:
            break
    assert s.state is JobState.COMPLETED and s.result_id == started.batch_id
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert container.evaluation.active_job() is None


def test_concurrent_starts_build_the_index_once(settings):
    container = build_container(
        settings,
        llm=FakeLLMAdapter(),
        embeddings=DeterministicFakeEmbedding(size=32),
        client=QdrantClient(":memory:"),
        embedding_dim=32,
    )
    builds = []
    real_build = container.index.build
    container.index.build = lambda progress=lambda m: None: builds.append(1) or real_build(progress)
    threads = [threading.Thread(target=container.start) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    assert builds == [1]


# ---------- EMBEDDING_PROVIDER ----------


def test_fake_embeddings_need_no_api_key_and_are_labelled(settings, tmp_path):
    s = replace(settings, embedding_provider="fake", openai_api_key=None, index_dir=tmp_path / "idx")
    index = build_index_manager(s, client=QdrantClient(":memory:"))
    assert index.embedding_model == "fake:64"  # an index built this way never passes for a real one
    assert len(index.embeddings.embed_query("hello")) == 64
    with pytest.raises(ConfigError, match="EMBEDDING_PROVIDER"):
        build_index_manager(replace(s, embedding_provider="nope"), client=QdrantClient(":memory:"))


# ---------- review regressions (PR #20) ----------


class BlockingLLM(FakeLLMAdapter):
    def __init__(self):
        super().__init__()
        self.started, self.release = threading.Event(), threading.Event()

    def generate(self, request):
        self.started.set()
        self.release.wait(TIMEOUT)
        return super().generate(request)


def test_rebuilding_the_index_is_refused_while_an_evaluation_runs(make_env):
    llm = BlockingLLM()
    container, testset_id = make_env(llm=llm)
    started = container.evaluation.start_evaluation(StartEvaluationRequest(testset_id=testset_id, pipelines=("baseline",)))
    assert llm.started.wait(TIMEOUT)
    counts = container.index.counts()
    with pytest.raises(JobAlreadyRunning):
        container.rebuild_index()
    assert container.index.counts() == counts  # untouched mid-evaluation
    llm.release.set()
    assert container.jobs.wait(started.job_id, timeout=TIMEOUT).state is JobState.COMPLETED
    assert container.rebuild_index()  # allowed once the job is done


def test_an_evaluation_started_during_a_rebuild_waits_for_it(make_env):
    llm = FakeLLMAdapter()
    container, testset_id = make_env(llm=llm)
    container.start()  # index built, service ready
    with container._start_lock:  # what rebuild_index() holds while rebuilding
        started = container.evaluation.start_evaluation(StartEvaluationRequest(testset_id=testset_id, pipelines=("baseline",)))
        assert container.jobs.wait(started.job_id, timeout=0.3).state is JobState.RUNNING
        assert llm.requests == []  # nothing answered against a half-built index
    assert container.jobs.wait(started.job_id, timeout=TIMEOUT).state is JobState.COMPLETED


def test_synchronous_runs_report_batch_warnings(make_env, data_dir):
    container, testset_id = make_env(eval_judge_model="fake-model")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Changed corpus.")
    doc.save(str(data_dir / "changed.pdf"))
    warnings = []
    container.evaluation.run_evaluation(
        StartEvaluationRequest(testset_id=testset_id, pipelines=("baseline",)), on_warning=warnings.append
    )
    assert any("stale" in w for w in warnings)
    assert any("same as the answering model" in w for w in warnings)


@pytest.mark.parametrize("problem", ["missing", "empty"])
def test_a_missing_corpus_is_an_invalid_request_and_creates_nothing(make_env, data_dir, problem):
    import shutil

    container, testset_id = make_env()
    if problem == "missing":
        shutil.rmtree(data_dir)
    else:
        for pdf in data_dir.glob("*.pdf"):
            pdf.unlink()
    with pytest.raises(InvalidRequest, match="No PDFs found"):
        container.evaluation.run_evaluation(StartEvaluationRequest(testset_id=testset_id))
    assert container.eval_runs.list() == []


class SteppedEvaluator(FakeEvaluator):
    """Reports progress halfway through judging, like a finer-grained evaluator would."""

    def evaluate(self, samples, metrics=DEFAULT_METRICS, *, on_progress=None):
        if on_progress:
            on_progress(0.5, "halfway through judging")
        return super().evaluate(samples, metrics, on_progress=on_progress)


def test_judging_progress_flows_into_run_and_batch_progress(make_env):
    container, testset_id = make_env(evaluator=SteppedEvaluator(), size=2)
    events = []
    [summary] = container.evaluation.run_evaluation(
        StartEvaluationRequest(testset_id=testset_id, pipelines=("baseline",)), on_progress=lambda p, m: events.append((p, m))
    )
    halfway = [p for p, m in events if "halfway through judging" in m]
    assert halfway == [pytest.approx(0.75)]  # answered 2 of 4 steps, then half of the 2 scoring steps
    fractions = [p for p, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_a_judge_failure_keeps_the_answering_usage(make_env):
    container, testset_id = make_env(evaluator=EvaluatorFailingOnCall(failing_call=1))
    started, _ = run_batch(container, testset_id, pipelines=("baseline",))
    m = container.eval_runs.get(started.run_ids[0]).manifest
    assert m.status is RunStatus.FAILED
    assert m.answer_usage is not None and m.answer_usage.total_tokens > 0
