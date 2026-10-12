"""UI tests (Streamlit AppTest) for the multipage app and the Test Sets page. Fully offline.

The app's cached container is replaced with one built from fakes (LLM, test set generator,
embeddings) and an in-memory Qdrant client, so no network or real index is used.
"""

import threading
from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient
from streamlit.testing.v1 import AppTest

import rag_chatbot.ui.state as ui_state
from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
from rag_chatbot.adapters.sdg.fake_generator import FakeTestsetGenerator
from rag_chatbot.container import build_container
from rag_chatbot.domain.datasets import GenerationError
from rag_chatbot.services.jobs import JobState

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
TEST_SETS_PAGE = "src/rag_chatbot/ui/test_sets_page.py"  # as registered in app.py
TIMEOUT = 60


class BlockingGenerator(FakeTestsetGenerator):
    """Holds generation until released, so tests can look at the page while a job is running."""

    def __init__(self):
        super().__init__()
        self.started, self.release = threading.Event(), threading.Event()

    def generate(self, documents, size, **kwargs):
        self.started.set()
        self.release.wait(TIMEOUT)
        return super().generate(documents, size, **kwargs)


@pytest.fixture
def make_app(monkeypatch, settings):
    """Build an offline container with the given generator and install it as the app's container.

    Returns (AppTest, container). By default the AppTest opens the Test Sets page directly, like
    visiting /test-sets: AppTest ignores switch_page() before the first run, which would run Chat
    (and build the index) first. Pass entry="app" to start from app.py's navigation instead.
    """

    def make(generator=None, entry="test_sets"):
        container = build_container(
            settings,
            llm=FakeLLMAdapter(),
            embeddings=DeterministicFakeEmbedding(size=32),
            client=QdrantClient(":memory:"),
            embedding_dim=32,
            testset_generator=generator or FakeTestsetGenerator(),
        )
        monkeypatch.setattr(ui_state, "build_container", lambda: container)
        ui_state.get_container.clear()
        script = APP if entry == "app" else str(ROOT / TEST_SETS_PAGE)
        return AppTest.from_file(script, default_timeout=TIMEOUT), container

    yield make
    ui_state.get_container.clear()


def open_test_sets(at: AppTest) -> AppTest:
    return at.run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def finish_job(at, container):
    container.jobs.wait(at.session_state["sdg_job_id"], timeout=TIMEOUT)
    return at.run()


def generate(at, container, name="core-corpus", size=3):
    at.text_input(key="sdg_name").set_value(name)
    at.number_input(key="sdg_size").set_value(size)
    button(at, "Estimate").click().run()
    button(at, "Confirm and generate").click().run()
    return finish_job(at, container)


def saved_rows(at):
    return at.dataframe[0].value  # the "Saved test sets" table


# ---------- navigation and the moved chat page ----------


def test_both_pages_load_and_the_chat_flow_still_works(make_app):
    at, _ = make_app(entry="app")
    at.run()  # default page: Chat
    assert not at.exception
    assert [t.value for t in at.title] == ["RAG Pipeline Chat"]

    at.chat_input[0].set_value("What do engineering leaders do?").run()  # single pipeline
    assert not at.exception
    assert [e.label for e in at.expander if e.label.startswith("Sources")]
    assert any(m.value.startswith("[fake answer") for m in at.markdown)

    at.sidebar.radio[0].set_value("Compare all three").run()
    at.chat_input[0].set_value("Why review code early?").run()
    assert not at.exception
    # No Cohere key in tests, so Rerank is listed last and reports itself unavailable.
    assert [h.value for h in at.subheader] == ["Baseline", "Parent-Child", "Rerank"]

    at.switch_page(TEST_SETS_PAGE).run()  # switching works once the app has run
    assert not at.exception
    assert [t.value for t in at.title] == ["Test Sets"]


def test_navigation_lists_chat_and_test_sets():
    source = Path(APP).read_text()
    assert 'title="Chat"' in source and 'title="Test Sets"' in source and "st.navigation" in source


def test_opening_test_sets_does_not_build_the_vector_index(make_app):
    at, container = make_app()
    open_test_sets(at)
    assert not at.exception
    assert set(container.index.counts().values()) == {None}  # no collections created


# ---------- estimate and confirm ----------


def test_estimate_shows_tokens_and_unconfigured_pricing_and_unlocks_confirm(make_app):
    at, _ = open_test_sets(make_app()[0]), None
    assert button(at, "Confirm and generate").disabled
    button(at, "Estimate").click().run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Input tokens (est.)"].startswith("~") and metrics["Output tokens (est.)"].startswith("~")
    assert metrics["Cost (est.)"] == "Pricing not configured"
    assert not button(at, "Confirm and generate").disabled


@pytest.mark.parametrize(
    "change",
    [lambda at: at.text_input(key="sdg_name").set_value("renamed"), lambda at: at.number_input(key="sdg_size").set_value(4)],
    ids=["name", "size"],
)
def test_changing_inputs_after_estimating_requires_a_new_estimate(make_app, change):
    at = open_test_sets(make_app()[0])
    button(at, "Estimate").click().run()
    assert not button(at, "Confirm and generate").disabled
    change(at).run()
    assert button(at, "Confirm and generate").disabled
    assert at.metric == [] or not any(m.label == "Input tokens (est.)" for m in at.metric)


# ---------- running jobs ----------


def test_generating_shows_progress_disables_the_form_and_adds_versions(make_app):
    generator = BlockingGenerator()
    at, container = make_app(generator)
    open_test_sets(at)
    button(at, "Estimate").click().run()
    button(at, "Confirm and generate").click().run()
    assert generator.started.wait(TIMEOUT)
    at.run()

    assert at.get("progress"), "progress panel should be visible while the job runs"
    assert at.text_input(key="sdg_name").disabled
    assert at.number_input(key="sdg_size").disabled
    assert at.selectbox(key="sdg_reuse").disabled
    assert button(at, "Estimate").disabled and button(at, "Confirm and generate").disabled

    generator.release.set()
    finish_job(at, container)
    assert not at.exception
    assert any("Saved test set" in s.value for s in at.success)
    rows = saved_rows(at)
    assert list(rows["Name"]) == ["core-corpus"] and list(rows["Version"]) == [1]
    assert not button(at, "Estimate").disabled  # usable again

    generator.release.set()  # second run doesn't block
    generate(at, container)
    rows = saved_rows(at)
    assert list(rows["Version"]) == [2, 1]


def test_an_invalid_name_shows_an_inline_error_and_starts_nothing(make_app):
    at, container = make_app()
    open_test_sets(at)
    at.text_input(key="sdg_name").set_value("not valid!")
    button(at, "Estimate").click().run()
    assert any("Name must be" in e.value for e in at.error)
    assert button(at, "Confirm and generate").disabled
    assert container.jobs.recent() == []


def test_a_failed_job_shows_its_error_and_the_form_recovers(make_app):
    at, container = make_app(FakeTestsetGenerator(fail_with=GenerationError("the corpus is too small")))
    open_test_sets(at)
    button(at, "Estimate").click().run()
    button(at, "Confirm and generate").click().run()
    finish_job(at, container)
    assert container.jobs.recent()[0].state is JobState.FAILED
    assert any("the corpus is too small" in e.value for e in at.error)
    assert not button(at, "Estimate").disabled
    assert not at.text_input(key="sdg_name").disabled


# ---------- list and detail ----------


def test_test_sets_turn_stale_when_a_pdf_changes(make_app, data_dir):
    at, container = make_app()
    open_test_sets(at)
    generate(at, container)
    assert list(saved_rows(at)["Status"]) == ["Current"]

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A newly added document.")
    doc.save(str(data_dir / "added.pdf"))
    at.run()
    assert list(saved_rows(at)["Status"]) == ["⚠️ Stale"]
    assert any("Stale" in w.value for w in at.warning)


def test_selecting_a_test_set_shows_its_manifest_and_all_samples(make_app):
    at, container = make_app()
    open_test_sets(at)
    generate(at, container, name="first", size=2)
    generate(at, container, name="second", size=4)

    first_id = next(i.summary.id for i in container.sdg.list_testsets() if i.summary.name == "first")
    at.selectbox(key="sdg_selected").set_value(first_id).run()
    assert not at.exception
    assert any(m.value == "### first v1" for m in at.markdown)
    details = " ".join(m.value for m in at.markdown)
    assert first_id in details and "fake-sdg-model" in details and "Token usage" in details
    samples = at.dataframe[1].value
    assert len(samples) == 2 and list(samples.columns) == ["#", "Question", "Reference answer", "Type"]


def test_a_shortfall_shows_requested_and_generated_counts(make_app):
    class Short(FakeTestsetGenerator):
        def generate(self, documents, size, **kwargs):
            from dataclasses import replace

            result = super().generate(documents, size, **kwargs)
            return replace(result, samples=result.samples[:2])

    at, container = make_app(Short())
    open_test_sets(at)
    generate(at, container, size=5)
    assert list(saved_rows(at)["Questions"]) == ["2 of 5"]


# ---------- review regressions ----------


def test_a_job_pushed_out_of_history_does_not_break_the_page(make_app):
    from rag_chatbot.services.jobs import HISTORY_LIMIT

    at, container = make_app()
    open_test_sets(at)
    generate(at, container)
    assert any("Saved test set" in s.value for s in at.success)

    for i in range(HISTORY_LIMIT + 1):  # unrelated jobs push this session's job out of the runner's history
        container.jobs.wait(container.jobs.submit("other", f"filler {i}", lambda report: None), timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert at.session_state["sdg_job_id"] is None
    assert not any("Saved test set" in s.value for s in at.success)
    assert list(saved_rows(at)["Version"]) == [1]  # the page still renders normally


def test_a_generation_started_in_another_tab_is_followed_to_completion(make_app):
    generator = BlockingGenerator()
    tab_a, container = make_app(generator)
    tab_b = AppTest.from_file(str(ROOT / TEST_SETS_PAGE), default_timeout=TIMEOUT)  # same container, new session
    open_test_sets(tab_a)
    open_test_sets(tab_b)

    button(tab_a, "Estimate").click().run()
    button(tab_a, "Confirm and generate").click().run()
    assert generator.started.wait(TIMEOUT)

    tab_b.run()  # tab B's job panel polls while A's job runs
    assert not tab_b.exception
    assert tab_b.get("progress")
    assert button(tab_b, "Confirm and generate").disabled and tab_b.text_input(key="sdg_name").disabled
    assert tab_b.session_state["sdg_job_id"] == tab_a.session_state["sdg_job_id"]

    generator.release.set()
    container.jobs.wait(tab_a.session_state["sdg_job_id"], timeout=TIMEOUT)
    tab_b.run()  # the next poll
    assert not tab_b.exception
    assert any("Saved test set" in s.value and "another tab" in s.value for s in tab_b.success)
    assert list(saved_rows(tab_b)["Version"]) == [1]
    assert not button(tab_b, "Estimate").disabled


def test_changing_the_pdfs_after_estimating_requires_a_new_estimate(make_app, data_dir):
    at, _ = make_app()
    open_test_sets(at)
    button(at, "Estimate").click().run()
    assert not button(at, "Confirm and generate").disabled

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Added after the estimate.")
    doc.save(str(data_dir / "late.pdf"))
    at.run()
    assert button(at, "Confirm and generate").disabled
    assert any("PDFs in `data/` changed" in c.value for c in at.caption)

    button(at, "Estimate").click().run()  # re-estimating against the new corpus unlocks it again
    assert not button(at, "Confirm and generate").disabled
