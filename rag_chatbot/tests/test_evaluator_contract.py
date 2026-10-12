"""Contract every EvaluatorPort adapter must satisfy.

To add an adapter, add it to the `adapter` fixture's params; every test below then runs against it.
The real Ragas adapter runs only with RUN_LIVE_TESTS=1 (it calls OpenAI; its scoring result is
cached for the module so the live run scores the samples once).
"""

import math
import os
from dataclasses import replace

import pytest
from factories import make_eval_samples

from rag_chatbot.adapters.evaluation.fake_evaluator import FakeEvaluator
from rag_chatbot.adapters.evaluation.ragas_evaluator import RagasEvaluator
from rag_chatbot.config import load_settings
from rag_chatbot.container import build_evaluator
from rag_chatbot.domain.evals import DEFAULT_METRICS, UnknownMetric
from rag_chatbot.ports import evaluator as port

live = pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API")
SAMPLES = make_eval_samples()
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
        return FakeEvaluator()
    if request.param == "ragas-mocked":
        request.getfixturevalue("mocked_ragas_evaluate")
        monkeypatch.setattr("ragas.llms.LangchainLLMWrapper.generate_text", lambda self, prompt, **_: None)
        return RagasEvaluator(judge_model="gpt-4.1-mini", embedding_model="text-embedding-3-small", api_key="sk-test")
    return build_evaluator(replace(load_settings(), eval_provider="ragas"))


def scored(request, adapter):
    key = request.node.callspec.id
    if "live" in key and key in _live_cache:
        return _live_cache[key]
    events = []
    result = adapter.evaluate(SAMPLES, on_progress=lambda p, m: events.append((p, m)))
    if "live" in key:
        _live_cache[key] = (result, events)
    return result, events


def test_satisfies_protocol_and_describes_itself(adapter):
    assert isinstance(adapter, port.EvaluatorPort)
    assert adapter.judge_model and adapter.embedding_model and adapter.ragas_version
    assert set(DEFAULT_METRICS) <= set(adapter.supported_metrics())


def test_preflight_succeeds_for_a_working_configuration(adapter):
    assert adapter.preflight() is None


def test_scores_every_sample_in_order_with_exactly_the_default_metrics(request, adapter):
    result, _ = scored(request, adapter)
    assert len(result.scores) == len(SAMPLES)
    for row in result.scores:
        assert list(row) == list(DEFAULT_METRICS)
        for value in row.values():
            assert value is None or (isinstance(value, float) and math.isfinite(value))
    if result.usage is not None:
        assert result.usage.total_tokens == result.usage.input_tokens + result.usage.output_tokens > 0


def test_reports_progress_from_start_to_finish(request, adapter):
    _, events = scored(request, adapter)
    fractions = [p for p, _ in events if p is not None]
    assert fractions and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert all(m for _, m in events)


def test_no_samples_means_no_scores(adapter):
    assert adapter.evaluate([]).scores == []


def test_unknown_metrics_are_rejected(adapter):
    with pytest.raises(UnknownMetric):
        adapter.evaluate(SAMPLES, ["faithfulness", "not_a_metric"])
