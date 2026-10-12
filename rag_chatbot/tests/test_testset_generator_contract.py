"""Contract every TestsetGeneratorPort adapter must satisfy.

To add an adapter, add it to the `adapter` fixture's params; every test below then runs against it.
The real Ragas adapter runs only with RUN_LIVE_TESTS=1 (it calls OpenAI; results are cached per
module so the live run makes as few calls as possible).
"""

import os
from dataclasses import replace

import pytest
from factories import SDG_DOCS

from rag_chatbot.adapters.sdg.fake_generator import FakeTestsetGenerator
from rag_chatbot.adapters.sdg.ragas_generator import RagasTestsetGenerator
from rag_chatbot.config import load_settings
from rag_chatbot.container import build_testset_generator
from rag_chatbot.ports import testset_generator as port

live = pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API")

_live_cache: dict = {}


@pytest.fixture(
    params=[
        pytest.param("fake", id="fake"),
        pytest.param("ragas-mocked", id="ragas-mocked"),
        pytest.param("ragas-live", id="ragas-live", marks=[live, pytest.mark.live]),
    ]
)
def adapter(request, monkeypatch):
    if request.param == "fake":
        return FakeTestsetGenerator()
    if request.param == "ragas-mocked":
        request.getfixturevalue("mocked_ragas")
        monkeypatch.setattr("ragas.llms.LangchainLLMWrapper.generate_text", lambda self, prompt, **_: None)
        return RagasTestsetGenerator(llm_model="gpt-4.1", embedding_model="text-embedding-3-small", api_key="sk-test")
    return build_testset_generator(replace(load_settings(), sdg_provider="ragas"))


def run(request, adapter, size, knowledge_graph=None):
    """Generate and capture progress. Live results are cached so each scenario costs one call."""
    key = (request.node.callspec.id, size, knowledge_graph is not None)
    if "live" in key[0] and key in _live_cache:
        return _live_cache[key]
    events = []
    result = adapter.generate(SDG_DOCS, size, knowledge_graph=knowledge_graph, on_progress=events.append)
    if "live" in key[0]:
        _live_cache[key] = (result, events)
    return result, events


def test_satisfies_protocol_and_describes_itself(adapter):
    assert isinstance(adapter, port.TestsetGeneratorPort)
    info = adapter.info
    assert info.llm_model and info.embedding_model and info.ragas_version


def test_preflight_succeeds_for_a_working_configuration(adapter):
    assert adapter.preflight() is None


def test_generates_at_most_size_complete_samples(request, adapter):
    result, _ = run(request, adapter, 2)
    assert 1 <= len(result.samples) <= 2
    for s in result.samples:
        assert s.user_input.strip() and s.reference.strip() and s.synthesizer_name
        assert s.reference_contexts and all(c.strip() for c in s.reference_contexts)
    assert isinstance(result.knowledge_graph, bytes) and result.knowledge_graph
    assert result.generator == adapter.info
    if result.usage is not None:
        assert result.usage.total_tokens == result.usage.input_tokens + result.usage.output_tokens > 0


def test_reports_progress_by_stage_in_order(request, adapter):
    _, events = run(request, adapter, 2)
    stages = [e.stage for e in events]
    assert stages == sorted(stages, key=["building_knowledge_graph", "generating_samples"].index)
    assert stages[0] == "building_knowledge_graph" and stages[-1] == "generating_samples"
    assert all(e.fraction is None or 0.0 <= e.fraction <= 1.0 for e in events)
    assert events[-1].fraction == 1.0
    assert all(e.message for e in events)


def test_reusing_a_graph_returns_it_and_skips_the_build_stage(request, adapter):
    first, _ = run(request, adapter, 2)
    result, events = run(request, adapter, 1, knowledge_graph=first.knowledge_graph)
    assert result.knowledge_graph == first.knowledge_graph
    assert len(result.samples) == 1
    assert {e.stage for e in events} == {"generating_samples"}


@pytest.mark.parametrize("size", [0, -1])
def test_rejects_a_non_positive_size(adapter, size):
    with pytest.raises(ValueError):
        adapter.generate(SDG_DOCS, size)
