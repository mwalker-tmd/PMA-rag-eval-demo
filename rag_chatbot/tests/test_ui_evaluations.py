"""UI tests (Streamlit AppTest) for the Evaluations page. Fully offline: the app's container is built from fakes."""

import threading
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
from factories import make_run_manifest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient
from streamlit.testing.v1 import AppTest

import rag_chatbot.ui.state as ui_state
from rag_chatbot.adapters.evaluation.fake_evaluator import FakeEvaluator
from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
from rag_chatbot.adapters.sdg.fake_generator import FakeTestsetGenerator
from rag_chatbot.container import build_container
from rag_chatbot.domain.errors import LLMRateLimitError
from rag_chatbot.domain.evals import RunStatus
from rag_chatbot.pipelines import rerank
from rag_chatbot.services.evaluation_service import RunFilter, StartEvaluationRequest
from rag_chatbot.services.sdg_service import GenerateTestsetRequest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
PAGE = str(ROOT / "src/rag_chatbot/ui/evaluations_page.py")
TIMEOUT = 60


class BlockingLLM(FakeLLMAdapter):
    def __init__(self):
        super().__init__()
        self.started, self.release = threading.Event(), threading.Event()

    def generate(self, request):
        self.started.set()
        self.release.wait(TIMEOUT)
        return super().generate(request)


class FailingLLM(FakeLLMAdapter):
    def generate(self, request):
        if "[1]" in request.messages[-1].content:
            raise LLMRateLimitError("throttled")
        return super().generate(request)


@pytest.fixture
def make_app(monkeypatch, settings):
    """Install an offline container as the app's container. Returns (AppTest on the Evaluations page, container)."""

    def make(*, llm=None, with_rerank=False, testsets=("core-corpus",), entry="page", **changes):
        s = replace(settings, eval_row_delay_seconds=0, **changes)
        if with_rerank:
            from test_evaluation_service import FakeRerank

            s = replace(s, cohere_api_key="test-key")
            monkeypatch.setattr(rerank, "CohereRerank", FakeRerank)
        container = build_container(
            s,
            llm=llm or FakeLLMAdapter(),
            embeddings=DeterministicFakeEmbedding(size=32),
            client=QdrantClient(":memory:"),
            embedding_dim=32,
            testset_generator=FakeTestsetGenerator(),
            evaluator=FakeEvaluator(),
        )
        for name in testsets:
            container.sdg.generate(GenerateTestsetRequest(name=name, size=2))
        monkeypatch.setattr(ui_state, "build_container", lambda: container)
        ui_state.get_container.clear()
        return AppTest.from_file(APP if entry == "app" else PAGE, default_timeout=TIMEOUT), container

    yield make
    ui_state.get_container.clear()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def metrics(at):
    return {m.label: m.value for m in at.metric}


def _testset_id(container, name="core-corpus"):
    return next(i.summary.id for i in container.sdg.list_testsets() if i.summary.name == name)


def evaluate(container, name="core-corpus", pipelines=("baseline",)):
    """Create finished runs directly through the service (fast with fakes)."""
    return container.evaluation.run_evaluation(StartEvaluationRequest(testset_id=_testset_id(container, name), pipelines=pipelines))


def estimate_and_run(at, container):
    button(at, "Estimate").click().run()
    button(at, "Confirm and run").click().run()
    container.jobs.wait(at.session_state["eval_job_id"], timeout=TIMEOUT)
    return at.run()


def runs_table(at):
    return at.dataframe[0].value


# ---------- navigation and banner ----------


def test_navigation_has_three_pages_and_each_loads(make_app):
    at, _ = make_app(entry="app")
    at.run()
    assert not at.exception
    for page, title in (
        ("src/rag_chatbot/ui/test_sets_page.py", "Test Sets"),
        ("src/rag_chatbot/ui/evaluations_page.py", "Evaluations"),
    ):
        at.switch_page(page).run()
        assert not at.exception and [t.value for t in at.title] == [title]
    assert 'title="Evaluations"' in Path(APP).read_text()


@pytest.mark.parametrize("judge, shown", [("fake-model", True), ("gpt-4.1-mini", False)])
def test_judge_warning_banner(make_app, judge, shown):
    at, _ = make_app(eval_judge_model=judge)
    at.run()
    assert any("same as the answering model" in w.value for w in at.warning) is shown


# ---------- the form ----------


def test_pipeline_checkboxes_reflect_availability(make_app):
    at, _ = make_app()
    at.run()
    rerank_box = at.checkbox(key="eval_pipe_rerank")
    assert rerank_box.disabled and rerank_box.value is False
    assert "COHERE_API_KEY" in rerank_box.help
    assert at.checkbox(key="eval_pipe_baseline").value and not at.checkbox(key="eval_pipe_baseline").disabled

    at, _ = make_app(with_rerank=True)
    at.run()
    assert not at.checkbox(key="eval_pipe_rerank").disabled


def test_estimate_shows_tokens_and_unpriced_costs_and_unlocks_confirm(make_app):
    at, _ = make_app()
    at.run()
    assert button(at, "Confirm and run").disabled
    button(at, "Estimate").click().run()
    shown = metrics(at)
    assert shown["Answers to generate"] == "2 × 2"
    assert shown["Answering tokens (est.)"].startswith("~") and shown["Judging tokens (est.)"].startswith("~")
    assert shown["Answering cost (est.)"] == shown["Judging cost (est.)"] == "Pricing not configured"
    assert not button(at, "Confirm and run").disabled


@pytest.mark.parametrize(
    "change",
    [
        lambda at, c: at.selectbox(key="eval_testset").set_value(_testset_id(c, "second")),
        lambda at, c: at.checkbox(key="eval_pipe_parent_child").uncheck(),
        lambda at, c: at.selectbox(key="eval_effort").set_value(c.llm.supported_reasoning_efforts()[-1]),
    ],
    ids=["testset", "pipelines", "effort"],
)
def test_changing_inputs_after_estimating_requires_a_new_estimate(make_app, change):
    at, container = make_app(testsets=("core-corpus", "second"))
    at.run()
    at.selectbox(key="eval_testset").set_value(_testset_id(container, "core-corpus")).run()
    button(at, "Estimate").click().run()
    assert not button(at, "Confirm and run").disabled
    change(at, container).run()
    assert button(at, "Confirm and run").disabled
    assert "Answers to generate" not in metrics(at)


def test_no_pipelines_selected_is_an_inline_error(make_app):
    at, container = make_app()
    at.run()
    at.checkbox(key="eval_pipe_baseline").uncheck()
    at.checkbox(key="eval_pipe_parent_child").uncheck().run()
    button(at, "Estimate").click().run()
    assert any("Select at least one pipeline" in e.value for e in at.error)
    assert button(at, "Confirm and run").disabled
    assert container.eval_runs.list() == []


def test_running_an_evaluation_shows_progress_then_two_completed_runs(make_app):
    llm = BlockingLLM()
    at, container = make_app(llm=llm)
    at.run()
    button(at, "Estimate").click().run()
    button(at, "Confirm and run").click().run()
    assert llm.started.wait(TIMEOUT)
    at.run()
    assert at.get("progress"), "progress panel should show while the job runs"
    assert at.selectbox(key="eval_testset").disabled and at.checkbox(key="eval_pipe_baseline").disabled
    assert button(at, "Estimate").disabled and button(at, "Confirm and run").disabled

    llm.release.set()
    container.jobs.wait(at.session_state["eval_job_id"], timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert any("Evaluation finished: 2 of 2 runs completed" in s.value for s in at.success)
    table = runs_table(at)
    assert list(table["Status"]) == ["completed", "completed"]
    assert len({container.eval_runs.get(r).manifest.batch_id for r in table["Run"]}) == 1
    assert sorted(at.multiselect(key="eval_compare").value) == sorted(table["Run"])  # the batch is preselected


def test_the_form_is_locked_while_any_job_runs(make_app):
    at, container = make_app()
    release = threading.Event()
    job_id = container.jobs.submit("sdg", "Generate 'other' (10)", lambda report: release.wait(TIMEOUT) and None)
    at.run()
    assert any("Another job is running" in i.value for i in at.info)
    assert at.selectbox(key="eval_testset").disabled and button(at, "Estimate").disabled
    release.set()
    container.jobs.wait(job_id, timeout=TIMEOUT)
    at.run()
    assert not button(at, "Estimate").disabled


def test_a_stale_test_set_is_warned_about(make_app, data_dir):
    at, container = make_app()
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Changed corpus.")
    doc.save(str(data_dir / "changed.pdf"))
    at.run()
    assert "stale" in at.selectbox(key="eval_testset").format_func(at.selectbox(key="eval_testset").value)
    button(at, "Estimate").click().run()
    assert any("is stale" in w.value for w in at.warning)  # in the estimate
    button(at, "Confirm and run").click().run()
    container.jobs.wait(at.session_state["eval_job_id"], timeout=TIMEOUT)
    at.run()
    assert any("is stale" in w.value for w in at.warning)  # after starting


# ---------- compare and detail ----------


def test_selecting_two_runs_compares_them_with_deltas(make_app):
    at, container = make_app()
    runs = evaluate(container, pipelines=("baseline", "parent_child"))
    a, b = runs[0].run_id, runs[1].run_id
    at.run()
    at.multiselect(key="eval_compare").set_value([a, b]).run()
    table = at.dataframe[1].value
    expected = container.evaluation.compare_runs([a, b], a)
    label_a, label_b = expected.labels[a], expected.labels[b]
    assert list(table.columns) == [label_a, label_b, f"Δ {label_b}"]
    for metric in expected.metrics:
        assert table.loc[metric, f"Δ {label_b}"] == pytest.approx(expected.deltas[metric][b])

    at.selectbox(key="eval_reference").set_value(b).run()
    table = at.dataframe[1].value
    flipped = container.evaluation.compare_runs([a, b], b)
    assert list(table.columns) == [label_a, f"Δ {label_a}", label_b]
    for metric in flipped.metrics:
        assert table.loc[metric, f"Δ {label_a}"] == pytest.approx(flipped.deltas[metric][a])


def test_mixed_test_sets_and_incomplete_runs_are_flagged(make_app):
    at, container = make_app(testsets=("core-corpus", "second"))
    [one] = evaluate(container, "core-corpus")
    [two] = evaluate(container, "second")
    container.eval_runs.create(make_run_manifest("run_20261012-090000_baseline_dead", testset_id=_testset_id(container)))
    container.eval_runs.fail("run_20261012-090000_baseline_dead", "EvaluationError: judge timed out")
    at.run()
    at.multiselect(key="eval_compare").set_value([one.run_id, two.run_id, "run_20261012-090000_baseline_dead"]).run()
    assert any("different test sets" in w.value for w in at.warning)
    assert any("Not completed" in i.value for i in at.info)


def test_a_single_run_shows_its_manifest_and_every_sample(make_app):
    at, container = make_app(llm=FailingLLM())
    [run] = evaluate(container)
    at.run()
    at.multiselect(key="eval_compare").set_value([run.run_id]).run()
    details = " ".join(m.value for m in at.markdown)
    assert run.run_id in details and "fake-judge" in details and "fake-model" in details and "Answering tokens" in details
    samples = at.dataframe[2].value
    assert len(samples) == 2
    assert {"faithfulness", "answer_relevancy"} <= set(samples.columns)
    assert "(no answer)" in list(samples["Answer"])  # the failed question
    assert any("throttled" in e.value for e in at.error)  # its error, in the sample's expander


def test_filters_narrow_the_runs_table(make_app):
    at, container = make_app(testsets=("core-corpus", "second"))
    evaluate(container, "core-corpus", pipelines=("baseline", "parent_child"))
    evaluate(container, "second", pipelines=("baseline",))
    container.eval_runs.create(make_run_manifest("run_20261012-090000_baseline_dead", testset_id=_testset_id(container)))
    container.eval_runs.fail("run_20261012-090000_baseline_dead", "boom")
    at.run()
    assert len(runs_table(at)) == 4

    at.selectbox(key="eval_f_testset").set_value(_testset_id(container, "second")).run()
    assert list(runs_table(at)["Test set"]) == ["second v1"]
    at.selectbox(key="eval_f_testset").set_value("(any)")
    at.selectbox(key="eval_f_pipeline").set_value("parent_child").run()
    assert list(runs_table(at)["Pipeline"]) == ["Parent-Child"]
    at.selectbox(key="eval_f_pipeline").set_value("(any)")
    at.selectbox(key="eval_f_status").set_value("failed").run()
    assert list(runs_table(at)["Status"]) == ["failed"]


def test_select_batch_selects_every_run_in_it(make_app):
    at, container = make_app()
    runs = evaluate(container, pipelines=("baseline", "parent_child"))
    at.run()
    button(at, "Select batch").click().run()
    assert sorted(at.multiselect(key="eval_compare").value) == sorted(r.run_id for r in runs)


# ---------- lessons from the Test Sets review ----------


def test_an_evaluation_started_in_another_tab_is_followed(make_app):
    llm = BlockingLLM()
    tab_a, container = make_app(llm=llm)
    tab_b = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    tab_a.run()
    tab_b.run()
    button(tab_a, "Estimate").click().run()
    button(tab_a, "Confirm and run").click().run()
    assert llm.started.wait(TIMEOUT)
    tab_b.run()
    assert tab_b.get("progress") and button(tab_b, "Estimate").disabled
    llm.release.set()
    container.jobs.wait(tab_a.session_state["eval_job_id"], timeout=TIMEOUT)
    tab_b.run()
    assert any("another tab" in s.value for s in tab_b.success)
    assert len(runs_table(tab_b)) == 2


def test_a_job_pushed_out_of_history_does_not_break_the_page(make_app):
    from rag_chatbot.services.jobs import HISTORY_LIMIT

    at, container = make_app()
    at.run()
    estimate_and_run(at, container)
    for i in range(HISTORY_LIMIT + 1):
        container.jobs.wait(container.jobs.submit("other", f"filler {i}", lambda report: None), timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert at.session_state["eval_job_id"] is None
    assert len(container.evaluation.list_runs(RunFilter(status=RunStatus.COMPLETED))) == 2



def test_batch_selections_put_the_baseline_first_as_the_reference(make_app):
    at, container = make_app(with_rerank=True)
    evaluate(container, pipelines=("parent_child", "rerank", "baseline"))
    at.run()
    button(at, "Select batch").click().run()
    selected = at.multiselect(key="eval_compare").value
    pipelines = [container.eval_runs.get(r).manifest.pipeline for r in selected]
    assert pipelines == ["baseline", "rerank", "parent_child"]
    assert at.selectbox(key="eval_reference").value == selected[0]
