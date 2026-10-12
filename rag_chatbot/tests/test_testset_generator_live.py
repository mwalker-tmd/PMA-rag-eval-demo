"""Live tests for RagasTestsetGenerator. They call OpenAI and cost a few cents: RUN_LIVE_TESTS=1 uv run pytest -m live"""

import os
import threading

import pytest
from factories import SDG_DOCS

from rag_chatbot.config import load_settings
from rag_chatbot.container import build_testset_generator

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]

DOCS = SDG_DOCS


@pytest.fixture(scope="module")
def generator():
    settings = load_settings()
    return build_testset_generator(settings.__class__(**{**settings.__dict__, "sdg_provider": "ragas"}))


def test_preflight_passes_for_the_default_model(generator):
    generator.preflight()


def test_generates_samples_and_reuses_the_graph_on_a_worker_thread(generator):
    first = generator.generate(DOCS, 2)
    assert len(first.samples) >= 2
    assert all(s.user_input and s.reference for s in first.samples)
    assert first.knowledge_graph
    assert first.usage is None or first.usage.total_tokens > 0

    # Reuse the saved graph from a background thread, as the job runner will.
    out: dict = {}

    def worker():
        try:
            out["result"] = generator.generate(DOCS, 2, knowledge_graph=first.knowledge_graph)
        except BaseException as e:  # surface in the main thread
            out["error"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=600)
    assert "error" not in out, out.get("error")
    second = out["result"]
    assert len(second.samples) >= 2
    assert second.knowledge_graph == first.knowledge_graph
    print(f"\nfirst usage: {first.usage}\nsecond usage: {second.usage}")
    print("sample:", first.samples[0])
