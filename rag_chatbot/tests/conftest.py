import inspect
from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from factories import RAGAS_SAMPLES, ragas_testset
from qdrant_client import QdrantClient

from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
from rag_chatbot.config import Settings
from rag_chatbot.container import build_container
from rag_chatbot.domain.models import ReasoningEffort

EMBED_DIM = 32

PAGES = {
    "quality.pdf": [
        "Quality is a habit, not an act. Teams that review code early find defects cheaply. " * 12,
        "Testing pyramids put many fast unit tests under fewer slow end-to-end tests. " * 12,
    ],
    "leadership.pdf": [
        "Engineering leaders set direction, remove blockers, and grow people on their teams. " * 12,
    ],
}


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(36, 36, 576, 806), text, fontsize=9)
    doc.save(str(path))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    for name, pages in PAGES.items():
        _write_pdf(d / name, pages)
    return d


@pytest.fixture
def settings(tmp_path: Path, data_dir: Path) -> Settings:
    return Settings(
        openai_api_key=None,
        cohere_api_key=None,
        llm_provider="fake",
        llm_model="fake-model",
        llm_reasoning_efforts=None,
        default_reasoning_effort=ReasoningEffort.MEDIUM,
        embedding_model="fake-embeddings",
        data_dir=data_dir,
        index_dir=tmp_path / ".index",
        datasets_dir=tmp_path / ".datasets",
        evals_dir=tmp_path / ".evals",
        sdg_provider="fake",
        eval_provider="fake",
    )


@pytest.fixture
def fake_llm() -> FakeLLMAdapter:
    return FakeLLMAdapter()


@pytest.fixture
def container(settings, fake_llm):
    return build_container(
        settings,
        llm=fake_llm,
        embeddings=DeterministicFakeEmbedding(size=EMBED_DIM),
        client=QdrantClient(":memory:"),
        embedding_dim=EMBED_DIM,
    )


@pytest.fixture
def mocked_ragas(monkeypatch):
    """Replace the Ragas pieces RagasTestsetGenerator calls, and record how they were called.

    Every call is bound against the *real* pinned Ragas signatures first, so an adapter that
    drifts from the Ragas API fails here with TypeError instead of being hidden by the mock.
    """
    import ragas.testset
    import ragas.testset.synthesizers
    import ragas.testset.transforms

    real = {
        "init": inspect.signature(ragas.testset.TestsetGenerator),
        "generate": inspect.signature(ragas.testset.TestsetGenerator.generate),
        "default_transforms": inspect.signature(ragas.testset.transforms.default_transforms),
        "apply_transforms": inspect.signature(ragas.testset.transforms.apply_transforms),
        "default_query_distribution": inspect.signature(ragas.testset.synthesizers.default_query_distribution),
    }
    calls = {"transforms_applied": 0, "generator_kg": None, "generate_size": None, "sizes": [], "batches": None}

    class FakeRagasGenerator:
        def __init__(self, *args, **kwargs):
            bound = real["init"].bind(*args, **kwargs)
            calls["generator_kg"] = bound.arguments.get("knowledge_graph")

        def generate(self, *args, **kwargs):
            size = real["generate"].bind(self, *args, **kwargs).arguments["testset_size"]
            calls["generate_size"] = size
            calls["sizes"].append(size)
            if calls.get("raise"):
                raise calls["raise"]
            if calls["batches"] is not None:  # scripted results, one list per call
                return ragas_testset(calls["batches"].pop(0) if calls["batches"] else [])
            return ragas_testset(RAGAS_SAMPLES)

    def fake_default_transforms(*args, **kwargs):
        real["default_transforms"].bind(*args, **kwargs)
        return []

    def fake_apply_transforms(*args, **kwargs):
        real["apply_transforms"].bind(*args, **kwargs)
        calls["transforms_applied"] += 1

    def fake_default_query_distribution(*args, **kwargs):
        real["default_query_distribution"].bind(*args, **kwargs)
        return []

    monkeypatch.setattr("ragas.testset.TestsetGenerator", FakeRagasGenerator)
    monkeypatch.setattr("ragas.testset.transforms.default_transforms", fake_default_transforms)
    monkeypatch.setattr("ragas.testset.transforms.apply_transforms", fake_apply_transforms)
    monkeypatch.setattr("ragas.testset.synthesizers.default_query_distribution", fake_default_query_distribution)
    return calls


@pytest.fixture
def mocked_ragas_evaluate(monkeypatch):
    """Replace ragas.evaluate, binding every call against the real pinned signature first.

    Records the dataset, metrics, judge, embeddings and run config it was given. Returns one score
    row per sample: from `calls["rows"]` if a test scripts them, else 0.5 for every metric.
    """
    import types

    import ragas

    real_signature = inspect.signature(ragas.evaluate)
    calls: dict = {"rows": None, "raise": None, "count": 0}

    def fake_evaluate(*args, **kwargs):
        bound = real_signature.bind(*args, **kwargs).arguments
        calls.update(
            count=calls["count"] + 1,
            dataset=bound["dataset"],
            metrics=bound["metrics"],
            llm=bound["llm"],
            embeddings=bound["embeddings"],
            run_config=bound["run_config"],
            show_progress=bound.get("show_progress"),
        )
        if calls["raise"]:
            raise calls["raise"]
        rows = calls["rows"] or [{m.name: 0.5 for m in bound["metrics"]} for _ in bound["dataset"].samples]
        return types.SimpleNamespace(scores=rows)  # the only attribute the adapter reads

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    return calls
