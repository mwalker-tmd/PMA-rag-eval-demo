"""Contract every EvalRunRepository adapter must satisfy.

To add an adapter, add it to the `repo` fixture's params; every test below then runs against it.
"""

from dataclasses import replace
from datetime import timedelta, timezone

import pytest
from factories import T0, make_run_manifest, make_sample_results

from rag_chatbot.adapters.eval_runs.file_repository import FileEvalRunRepository
from rag_chatbot.adapters.eval_runs.memory_repository import InMemoryEvalRunRepository
from rag_chatbot.domain.evals import RunAlreadyExists, RunFinalized, RunNotFound, RunStatus
from rag_chatbot.domain.models import TokenUsage
from rag_chatbot.ports.eval_runs import EvalRunRepository

FINISHED = T0 + timedelta(minutes=3)
JUDGE = TokenUsage(input_tokens=5000, output_tokens=800, total_tokens=5800)
ANSWER = TokenUsage(input_tokens=1200, output_tokens=300, total_tokens=1500)
AGGREGATE = {"faithfulness": 0.75, "answer_relevancy": 0.5}


@pytest.fixture(params=["memory", "file"])
def repo(request, tmp_path) -> EvalRunRepository:
    clock = lambda: FINISHED  # noqa: E731
    if request.param == "memory":
        return InMemoryEvalRunRepository(clock=clock)
    return FileEvalRunRepository(tmp_path / ".evals", clock=clock)


@pytest.fixture
def run(repo):
    m = make_run_manifest()
    repo.create(m)
    return m


def test_satisfies_protocol(repo):
    assert isinstance(repo, EvalRunRepository)


def test_create_then_get_returns_the_running_run(repo, run):
    got = repo.get(run.run_id)
    assert got.manifest == run
    assert got.manifest.status is RunStatus.RUNNING
    assert got.samples == []


def test_create_rejects_a_taken_id(repo, run):
    with pytest.raises(RunAlreadyExists):
        repo.create(replace(run, pipeline="rerank"))
    assert repo.get(run.run_id).manifest == run


@pytest.mark.parametrize(
    "bad",
    [
        lambda m: replace(m, status=RunStatus.COMPLETED),
        lambda m: replace(m, finished_at=FINISHED),
        lambda m: replace(m, error="nope"),
        lambda m: replace(m, started_at=T0.replace(tzinfo=None)),
        lambda m: replace(m, started_at=T0.astimezone(timezone(timedelta(hours=2)))),
        lambda m: replace(m, run_id="run_ok\n"),
        lambda m: replace(m, run_id="../escape"),
        lambda m: replace(m, progress=1.5),
        lambda m: replace(m, progress=True),
        lambda m: replace(m, aggregate={"faithfulness": float("nan")}),
    ],
    ids=["status", "finished_at", "error", "naive", "non-utc", "newline-id", "path-id", "progress", "bool-progress", "nan"],
)
def test_create_rejects_an_invalid_new_run(repo, bad):
    with pytest.raises(ValueError):
        repo.create(bad(make_run_manifest()))
    assert repo.list() == []


def test_update_progress_is_reflected(repo, run):
    repo.update_progress(run.run_id, 0.4, "Answering question 4 of 10")
    m = repo.get(run.run_id).manifest
    assert (m.progress, m.message, m.status) == (0.4, "Answering question 4 of 10", RunStatus.RUNNING)
    repo.update_progress(run.run_id, None, "Scoring")
    assert repo.get(run.run_id).manifest.progress is None


@pytest.mark.parametrize("progress", [-0.1, 1.01, float("nan"), True, False])
def test_update_progress_rejects_out_of_range_values(repo, run, progress):
    with pytest.raises(ValueError):
        repo.update_progress(run.run_id, progress, "bad")


def test_complete_records_everything_and_round_trips_samples(repo, run):
    samples = make_sample_results(3)
    repo.complete(run.run_id, samples, AGGREGATE, judge_usage=JUDGE, answer_usage=ANSWER)
    got = repo.get(run.run_id)
    m = got.manifest
    assert m.status is RunStatus.COMPLETED and m.finished_at == FINISHED and m.progress == 1.0
    assert (m.aggregate, m.judge_usage, m.answer_usage) == (AGGREGATE, JUDGE, ANSWER)
    assert got.samples == samples
    assert got.samples[1].scores["faithfulness"] is None  # None scores survive the round trip


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), True])
def test_complete_rejects_non_finite_scores(repo, run, bad_score):
    samples = make_sample_results(1)
    samples[0] = replace(samples[0], scores={"faithfulness": bad_score})
    with pytest.raises(ValueError):
        repo.complete(run.run_id, samples, AGGREGATE, judge_usage=None, answer_usage=None)
    with pytest.raises(ValueError):
        repo.complete(run.run_id, [], {"faithfulness": bad_score}, judge_usage=None, answer_usage=None)
    assert repo.get(run.run_id).manifest.status is RunStatus.RUNNING


def test_fail_records_the_error_and_keeps_partial_samples(repo, run):
    partial = make_sample_results(1)
    repo.fail(run.run_id, "EvaluationError: judge timed out", samples=partial)
    got = repo.get(run.run_id)
    assert got.manifest.status is RunStatus.FAILED
    assert got.manifest.error == "EvaluationError: judge timed out"
    assert got.manifest.finished_at == FINISHED
    assert got.samples == partial


def test_fail_without_samples(repo, run):
    repo.fail(run.run_id, "Interrupted (server restarted)")
    assert repo.get(run.run_id).samples == []


@pytest.mark.parametrize("finish", ["complete", "fail"])
def test_a_finished_run_rejects_every_write(repo, run, finish):
    if finish == "complete":
        repo.complete(run.run_id, make_sample_results(2), AGGREGATE, judge_usage=JUDGE, answer_usage=ANSWER)
    else:
        repo.fail(run.run_id, "boom", samples=make_sample_results(1))
    before = repo.get(run.run_id)

    with pytest.raises(RunFinalized):
        repo.update_progress(run.run_id, 0.5, "late")
    with pytest.raises(RunFinalized):
        repo.complete(run.run_id, [], {}, judge_usage=None, answer_usage=None)
    with pytest.raises(RunFinalized):
        repo.fail(run.run_id, "again")
    assert repo.get(run.run_id) == before


def test_unknown_ids_raise_not_found(repo):
    missing = "run_20260101-000000_baseline_ffff"
    with pytest.raises(RunNotFound):
        repo.get(missing)
    with pytest.raises(RunNotFound):
        repo.update_progress(missing, 0.1, "x")
    with pytest.raises(RunNotFound):
        repo.complete(missing, [], {}, judge_usage=None, answer_usage=None)
    with pytest.raises(RunNotFound):
        repo.fail(missing, "x")


def test_list_is_newest_first_and_filters_combine(repo):
    runs = [
        make_run_manifest("run_a", batch_id="b1", testset_id="ts_1", pipeline="baseline", started_at=T0),
        make_run_manifest("run_b", batch_id="b1", testset_id="ts_1", pipeline="rerank", started_at=T0 + timedelta(seconds=1)),
        make_run_manifest("run_c", batch_id="b2", testset_id="ts_2", pipeline="baseline", started_at=T0 + timedelta(seconds=2)),
        make_run_manifest("run_d", batch_id="b2", testset_id="ts_2", pipeline="rerank", started_at=T0 + timedelta(seconds=3)),
    ]
    for m in runs:
        repo.create(m)
    repo.complete("run_a", [], {}, judge_usage=None, answer_usage=None)
    repo.fail("run_d", "boom")

    def ids(**filters):
        return [s.run_id for s in repo.list(**filters)]

    assert ids() == ["run_d", "run_c", "run_b", "run_a"]
    assert ids(testset_id="ts_1") == ["run_b", "run_a"]
    assert ids(pipeline="baseline") == ["run_c", "run_a"]
    assert ids(batch_id="b2") == ["run_d", "run_c"]
    assert ids(status=RunStatus.RUNNING) == ["run_c", "run_b"]
    assert ids(status=RunStatus.COMPLETED) == ["run_a"]
    assert ids(status=RunStatus.FAILED) == ["run_d"]
    assert ids(testset_id="ts_2", pipeline="rerank") == ["run_d"]
    assert ids(batch_id="b1", status=RunStatus.RUNNING) == ["run_b"]
    assert ids(testset_id="ts_1", batch_id="b2") == []
    assert not hasattr(repo.list()[0], "samples")


def test_stored_runs_are_isolated_from_caller_mutation(repo):
    m = make_run_manifest()
    repo.create(m)
    m.pipeline_settings["k"] = 999
    samples = make_sample_results(1)
    repo.complete(m.run_id, samples, AGGREGATE, judge_usage=None, answer_usage=None)
    samples[0].retrieved_contexts.append("tampered")
    got = repo.get(m.run_id)
    assert got.manifest.pipeline_settings["k"] == 3
    assert "tampered" not in got.samples[0].retrieved_contexts


# Run state is checked before the input: an unknown or finished run reports that, whatever the input.
NAN = float("nan")
BAD_WRITES = {
    "progress": lambda repo, run_id: repo.update_progress(run_id, True, "bad progress"),
    "complete": lambda repo, run_id: repo.complete(run_id, [], {"faithfulness": NAN}, judge_usage=None, answer_usage=None),
    "fail": lambda repo, run_id: repo.fail(
        run_id, "boom", samples=[replace(make_sample_results(1)[0], scores={"faithfulness": NAN})]
    ),
}


@pytest.mark.parametrize("write", BAD_WRITES.values(), ids=BAD_WRITES.keys())
def test_bad_input_to_an_unknown_run_reports_not_found(repo, write):
    with pytest.raises(RunNotFound):
        write(repo, "run_20260101-000000_baseline_ffff")


@pytest.mark.parametrize("write", BAD_WRITES.values(), ids=BAD_WRITES.keys())
def test_bad_input_to_a_finished_run_reports_finalized(repo, run, write):
    repo.complete(run.run_id, [], {}, judge_usage=None, answer_usage=None)
    with pytest.raises(RunFinalized):
        write(repo, run.run_id)
