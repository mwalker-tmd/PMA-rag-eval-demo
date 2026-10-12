"""Offline tests for the TestsetGeneratorPort adapters, the usage callback and SDG config."""

import json
from dataclasses import replace
from importlib.metadata import version

import httpx
import openai
import pytest
from factories import RAGAS_SAMPLES, SDG_DOCS as DOCS
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from rag_chatbot.adapters.langchain_usage import TokenUsageCallback
from rag_chatbot.adapters.sdg.fake_generator import FakeTestsetGenerator
from rag_chatbot.adapters.sdg.ragas_generator import RagasTestsetGenerator
from rag_chatbot.config import load_settings
from rag_chatbot.container import ConfigError, build_testset_generator
from rag_chatbot.domain.datasets import GenerationError
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.domain.models import ReasoningEffort
from rag_chatbot.ports import testset_generator as port



def collect():
    events = []
    return events, events.append


# ---------- fake adapter ----------


def test_fake_satisfies_port():
    assert isinstance(FakeTestsetGenerator(), port.TestsetGeneratorPort)
    assert isinstance(RagasTestsetGenerator(llm_model="m", embedding_model="e", api_key="k"), port.TestsetGeneratorPort)


def test_fake_returns_requested_number_of_complete_samples():
    result = FakeTestsetGenerator().generate(DOCS, 5)
    assert len(result.samples) == 5
    for s in result.samples:
        assert s.user_input and s.reference and s.reference_contexts and all(s.reference_contexts)
    assert result.knowledge_graph
    assert result.usage.total_tokens > 0


def test_fake_reports_both_stages_in_order_and_is_deterministic():
    events, cb = collect()
    first = FakeTestsetGenerator().generate(DOCS, 3, on_progress=cb)
    assert [(e.stage, e.fraction) for e in events] == [
        ("building_knowledge_graph", 0.0),
        ("building_knowledge_graph", 1.0),
        ("generating_samples", 0.0),
        ("generating_samples", 1.0),
    ]
    assert FakeTestsetGenerator().generate(DOCS, 3) == first


def test_fake_skips_graph_build_when_a_graph_is_reused():
    events, cb = collect()
    result = FakeTestsetGenerator().generate(DOCS, 2, knowledge_graph=b"saved-graph", on_progress=cb)
    assert {e.stage for e in events} == {"generating_samples"}
    assert result.knowledge_graph == b"saved-graph"


def test_fake_can_be_configured_to_fail():
    with pytest.raises(GenerationError, match="boom"):
        FakeTestsetGenerator(fail_with=GenerationError("boom")).generate(DOCS, 2)


# ---------- Ragas adapter (Ragas mocked; no network) ----------


@pytest.fixture
def ragas_adapter():
    return RagasTestsetGenerator(
        llm_model="gpt-4.1", embedding_model="text-embedding-3-small", api_key="sk-test", reasoning_effort=ReasoningEffort.LOW
    )


def test_ragas_builds_a_graph_and_converts_samples(mocked_ragas, ragas_adapter):
    events, cb = collect()
    result = ragas_adapter.generate(DOCS, 2, on_progress=cb)

    assert mocked_ragas["transforms_applied"] == 1
    assert mocked_ragas["generate_size"] == 2
    graph = json.loads(result.knowledge_graph)
    assert [n["properties"]["document_metadata"]["source"] for n in graph["nodes"]] == ["northwind-quality.pdf", "northwind-leadership.pdf"]
    assert [(e.stage, e.fraction) for e in events] == [
        ("building_knowledge_graph", 0.0),
        ("building_knowledge_graph", 1.0),
        ("generating_samples", 0.0),
        ("generating_samples", 1.0),
    ]
    assert [(s.user_input, s.reference, s.reference_contexts, s.synthesizer_name) for s in result.samples] == [
        (q, r, ctx, name) for q, r, ctx, name in RAGAS_SAMPLES
    ]


def test_ragas_reuses_a_saved_graph_without_applying_transforms(mocked_ragas, ragas_adapter):
    saved = ragas_adapter.generate(DOCS, 2).knowledge_graph
    mocked_ragas["transforms_applied"] = 0

    events, cb = collect()
    result = ragas_adapter.generate(DOCS, 2, knowledge_graph=saved, on_progress=cb)

    assert mocked_ragas["transforms_applied"] == 0
    reused = mocked_ragas["generator_kg"]
    assert [n.properties["document_metadata"]["source"] for n in reused.nodes] == ["northwind-quality.pdf", "northwind-leadership.pdf"]
    assert {e.stage for e in events} == {"generating_samples"}
    assert result.knowledge_graph == saved


def _q(i, kind):
    return (f"Question {i}?", f"Answer {i}.", [f"ctx {i}"], kind)


def test_ragas_trims_overshoot_to_size_keeping_the_query_mix(mocked_ragas, ragas_adapter):
    # Ragas rounds each query type's share up: asking for 4 across 3 types can return 6.
    single, abstract, specific = "single_hop", "multi_hop_abstract", "multi_hop_specific"
    mocked_ragas["batches"] = [
        [_q(0, single), _q(1, single), _q(2, abstract), _q(3, abstract), _q(4, specific), _q(5, specific)]
    ]
    result = ragas_adapter.generate(DOCS, 4)
    assert len(result.samples) == 4
    assert {s.synthesizer_name for s in result.samples} == {single, abstract, specific}
    assert mocked_ragas["sizes"] == [4]


def test_ragas_tops_up_a_shortfall_and_drops_duplicates(mocked_ragas, ragas_adapter):
    mocked_ragas["batches"] = [
        [_q(0, "a")],  # asked for 3, got 1
        [_q(0, "a"), _q(1, "b")],  # top-up for 2: one duplicate, one new
        [_q(2, "a")],  # top-up for 1
    ]
    result = ragas_adapter.generate(DOCS, 3)
    assert [s.user_input for s in result.samples] == ["Question 0?", "Question 1?", "Question 2?"]
    assert mocked_ragas["sizes"] == [3, 2, 1]


def test_ragas_returns_fewer_when_the_corpus_runs_dry(mocked_ragas, ragas_adapter):
    mocked_ragas["batches"] = [[_q(0, "a")], [_q(0, "a")]]  # the top-up only repeats a question
    result = ragas_adapter.generate(DOCS, 3)
    assert len(result.samples) == 1
    assert mocked_ragas["sizes"] == [3, 2]  # stops as soon as a round adds nothing new


def test_the_ragas_mock_rejects_calls_that_do_not_match_the_pinned_api(mocked_ragas):
    """Guards the guard: the mock must fail on signature drift, not silently accept it."""
    import ragas.testset

    with pytest.raises(TypeError):
        ragas.testset.TestsetGenerator(generator_llm=None, critic_llm=None, embeddings=None)  # Ragas 0.1.x style
    gen = ragas.testset.TestsetGenerator(llm=None, embedding_model=None)
    with pytest.raises(TypeError):
        gen.generate(test_size=2)  # misspelled keyword


def test_ragas_errors_become_generation_errors(mocked_ragas, ragas_adapter):
    mocked_ragas["raise"] = RuntimeError("No relationships match the provided condition")
    with pytest.raises(GenerationError, match="No relationships match the provided condition"):
        ragas_adapter.generate(DOCS, 2)


def test_ragas_rejects_multi_turn_samples(mocked_ragas, ragas_adapter, monkeypatch):
    from ragas.dataset_schema import MultiTurnSample
    from ragas.messages import HumanMessage
    from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample

    multi = Testset(samples=[TestsetSample(eval_sample=MultiTurnSample(user_input=[HumanMessage(content="hi")]), synthesizer_name="x")])

    class MultiGen:
        def __init__(self, **_):
            pass

        def generate(self, **_):
            return multi

    monkeypatch.setattr("ragas.testset.TestsetGenerator", MultiGen)
    with pytest.raises(GenerationError, match="single-turn"):
        ragas_adapter.generate(DOCS, 1)


def test_generator_info_records_models_effort_and_ragas_version(ragas_adapter):
    assert ragas_adapter.info.llm_model == "gpt-4.1"
    assert ragas_adapter.info.embedding_model == "text-embedding-3-small"
    assert ragas_adapter.info.reasoning_effort == "low"
    assert ragas_adapter.info.ragas_version == version("ragas") == "0.2.10"


def _bad_request():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        "Unsupported value: 'temperature' does not support 1e-08 with this model.",
        response=httpx.Response(400, request=request),
        body=None,
    )


def test_preflight_maps_a_rejected_request_to_model_incompatible(ragas_adapter, monkeypatch):
    def reject(self, prompt, **_):
        raise _bad_request()

    monkeypatch.setattr("ragas.llms.LangchainLLMWrapper.generate_text", reject)
    with pytest.raises(ModelIncompatible, match="temperature"):
        ragas_adapter.preflight()


def test_preflight_maps_other_failures_to_generation_error(ragas_adapter, monkeypatch):
    def offline(self, prompt, **_):
        raise ConnectionError("network down")

    monkeypatch.setattr("ragas.llms.LangchainLLMWrapper.generate_text", offline)
    with pytest.raises(GenerationError, match="network down"):
        ragas_adapter.preflight()


# ---------- usage callback ----------


def test_usage_callback_sums_llm_output_token_usage():
    cb = TokenUsageCallback()
    for _ in range(2):
        cb.on_llm_end(
            LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content="x"))]],
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "completion_tokens_details": {"reasoning_tokens": 5},
                    }
                },
            )
        )
    u = cb.usage()
    assert (u.input_tokens, u.output_tokens, u.total_tokens, u.reasoning_tokens) == (200, 40, 240, 10)


def test_usage_callback_falls_back_to_usage_metadata():
    cb = TokenUsageCallback()
    message = AIMessage(content="x", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert cb.usage().total_tokens == 10
    assert cb.usage().reasoning_tokens is None


def test_usage_callback_without_any_usage_reports_none():
    cb = TokenUsageCallback()
    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]]))
    assert cb.usage() is None


# ---------- config ----------


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr("rag_chatbot.config.load_dotenv", lambda *_a, **_k: None)  # ignore the developer's .env
    for var in ("SDG_PROVIDER", "SDG_LLM_MODEL", "SDG_EMBEDDING_MODEL", "SDG_REASONING_EFFORT"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_unset_sdg_vars_fall_back_to_documented_defaults(clean_env):
    s = load_settings()
    assert (s.sdg_provider, s.sdg_llm_model, s.sdg_embedding_model, s.sdg_reasoning_effort) == (
        "ragas",
        "gpt-4.1",
        "text-embedding-3-small",
        None,
    )


def test_sdg_vars_are_read(clean_env):
    clean_env.setenv("SDG_LLM_MODEL", "gpt-4.1-mini")
    clean_env.setenv("SDG_REASONING_EFFORT", "Low")
    s = load_settings()
    assert s.sdg_llm_model == "gpt-4.1-mini"
    assert s.sdg_reasoning_effort is ReasoningEffort.LOW


def test_provider_selection(settings):
    assert isinstance(build_testset_generator(replace(settings, sdg_provider="fake")), FakeTestsetGenerator)
    ragas = build_testset_generator(replace(settings, sdg_provider="ragas", openai_api_key="sk-test"))
    assert isinstance(ragas, RagasTestsetGenerator)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        build_testset_generator(replace(settings, sdg_provider="ragas", openai_api_key=None))
    with pytest.raises(ConfigError, match="Unknown SDG_PROVIDER"):
        build_testset_generator(replace(settings, sdg_provider="nope"))


def test_container_exposes_the_configured_generator(container):
    assert isinstance(container.testset_generator, FakeTestsetGenerator)
