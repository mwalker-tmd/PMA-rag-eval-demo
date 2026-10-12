"""Offline tests for the EvaluatorPort adapters (Ragas mocked) and evaluation config."""

from dataclasses import replace
from importlib.metadata import version

import httpx
import openai
import pytest
from factories import make_eval_samples

from rag_chatbot.adapters.evaluation.fake_evaluator import FakeEvaluator
from rag_chatbot.adapters.evaluation.ragas_evaluator import METRIC_CLASSES, RagasEvaluator
from rag_chatbot.config import judge_matches_answer_model, load_settings
from rag_chatbot.container import ConfigError, build_evaluator
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.domain.evals import DEFAULT_METRICS, EvaluationError, UnknownMetric
from rag_chatbot.domain.models import ReasoningEffort

SAMPLES = make_eval_samples()


@pytest.fixture
def ragas_adapter():
    return RagasEvaluator(
        judge_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        api_key="sk-test",
        judge_reasoning_effort=ReasoningEffort.LOW,
        timeout_seconds=123,
    )


# ---------- fake ----------


def test_fake_scores_every_sample_with_exactly_the_requested_metrics():
    result = FakeEvaluator().evaluate(SAMPLES, ["faithfulness", "answer_relevancy"])
    assert len(result.scores) == 2
    assert all(set(row) == {"faithfulness", "answer_relevancy"} for row in result.scores)
    assert all(0.0 <= v <= 1.0 for row in result.scores for v in row.values())
    assert FakeEvaluator().evaluate(SAMPLES, ["faithfulness", "answer_relevancy"]) == result  # deterministic
    assert result.scores[0] != result.scores[1]  # different answers, different scores


def test_fake_gives_none_for_an_empty_answer():
    result = FakeEvaluator().evaluate([replace(SAMPLES[0], response="")])
    assert set(result.scores[0].values()) == {None}


# ---------- unknown metrics ----------


@pytest.mark.parametrize("make", [FakeEvaluator, lambda: RagasEvaluator(judge_model="m", embedding_model="e", api_key="k")])
def test_unknown_metrics_are_rejected_before_any_work(make, mocked_ragas_evaluate):
    with pytest.raises(UnknownMetric, match="bleu"):
        make().evaluate(SAMPLES, ["faithfulness", "bleu"])
    assert mocked_ragas_evaluate["count"] == 0


# ---------- Ragas adapter (ragas.evaluate mocked) ----------


def test_ragas_builds_the_dataset_metrics_judge_embeddings_and_run_config(ragas_adapter, mocked_ragas_evaluate):
    ragas_adapter.evaluate(SAMPLES)

    dataset = mocked_ragas_evaluate["dataset"]
    first = dataset.samples[0]
    assert (first.user_input, first.reference, first.response) == (
        SAMPLES[0].user_input,
        SAMPLES[0].reference,
        SAMPLES[0].response,
    )
    assert first.retrieved_contexts == SAMPLES[0].retrieved_contexts
    assert first.reference_contexts == SAMPLES[0].reference_contexts
    assert len(dataset.samples) == 2

    assert [type(m).__name__ for m in mocked_ragas_evaluate["metrics"]] == [METRIC_CLASSES[n] for n in DEFAULT_METRICS]
    judge = mocked_ragas_evaluate["llm"].langchain_llm
    assert judge.model_name == "gpt-4.1-mini" and judge.reasoning_effort == "low"
    assert mocked_ragas_evaluate["embeddings"].embeddings.model == "text-embedding-3-small"
    assert mocked_ragas_evaluate["run_config"].timeout == 123
    assert mocked_ragas_evaluate["show_progress"] is False


def test_ragas_maps_nan_to_none_and_keeps_input_order(ragas_adapter, mocked_ragas_evaluate):
    nan = float("nan")
    mocked_ragas_evaluate["rows"] = [
        {"faithfulness": 0.9, "answer_relevancy": nan},
        {"faithfulness": 0.1, "answer_relevancy": 0.7},
    ]
    result = ragas_adapter.evaluate(SAMPLES, ["faithfulness", "answer_relevancy"])
    assert result.scores == [
        {"faithfulness": 0.9, "answer_relevancy": None},
        {"faithfulness": 0.1, "answer_relevancy": 0.7},
    ]


def test_ragas_fills_missing_or_odd_values_with_none(ragas_adapter, mocked_ragas_evaluate):
    mocked_ragas_evaluate["rows"] = [{"faithfulness": float("inf")}, {"faithfulness": "n/a"}]
    result = ragas_adapter.evaluate(SAMPLES, ["faithfulness", "answer_relevancy"])
    assert result.scores == [{"faithfulness": None, "answer_relevancy": None}] * 2


def test_ragas_rejects_a_row_count_mismatch(ragas_adapter, mocked_ragas_evaluate):
    mocked_ragas_evaluate["rows"] = [{"faithfulness": 0.5}]
    with pytest.raises(EvaluationError, match="1 score rows for 2 samples"):
        ragas_adapter.evaluate(SAMPLES, ["faithfulness"])


def test_ragas_metric_classes_report_our_metric_names():
    """Pins the Ragas 0.2.10 output keys: a Ragas upgrade that renames them fails here, not silently."""
    import ragas.metrics

    reported = [getattr(ragas.metrics, METRIC_CLASSES[name])().name for name in DEFAULT_METRICS]
    assert reported == list(DEFAULT_METRICS)
    assert set(METRIC_CLASSES) == set(DEFAULT_METRICS)


def test_ragas_errors_become_evaluation_errors(ragas_adapter, mocked_ragas_evaluate):
    mocked_ragas_evaluate["raise"] = RuntimeError("judge timed out after 360s")
    with pytest.raises(EvaluationError, match="judge timed out after 360s"):
        ragas_adapter.evaluate(SAMPLES)


def test_ragas_with_no_samples_makes_no_call(ragas_adapter, mocked_ragas_evaluate):
    assert ragas_adapter.evaluate([]).scores == []
    assert mocked_ragas_evaluate["count"] == 0


def test_ragas_reports_progress(ragas_adapter, mocked_ragas_evaluate):
    events = []
    ragas_adapter.evaluate(SAMPLES, on_progress=lambda p, m: events.append(p))
    assert events == [0.0, 1.0]


def test_ragas_describes_itself(ragas_adapter):
    assert (ragas_adapter.judge_model, ragas_adapter.embedding_model) == ("gpt-4.1-mini", "text-embedding-3-small")
    assert ragas_adapter.ragas_version == version("ragas") == "0.2.10"
    assert ragas_adapter.supported_metrics() == list(DEFAULT_METRICS)


def _bad_request():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError("Unsupported value: 'temperature'", response=httpx.Response(400, request=request), body=None)


def test_preflight_maps_a_rejected_request_to_model_incompatible(ragas_adapter, monkeypatch):
    def reject(self, prompt, **_):
        raise _bad_request()

    monkeypatch.setattr("ragas.llms.LangchainLLMWrapper.generate_text", reject)
    with pytest.raises(ModelIncompatible, match="EVAL_JUDGE_MODEL"):
        ragas_adapter.preflight()


def test_preflight_maps_other_failures_to_evaluation_error(ragas_adapter, monkeypatch):
    def offline(self, prompt, **_):
        raise ConnectionError("network down")

    monkeypatch.setattr("ragas.llms.LangchainLLMWrapper.generate_text", offline)
    with pytest.raises(EvaluationError, match="network down"):
        ragas_adapter.preflight()


# ---------- config ----------


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr("rag_chatbot.config.load_dotenv", lambda *_a, **_k: None)
    for var in ("EVAL_PROVIDER", "EVAL_JUDGE_MODEL", "EVAL_EMBEDDING_MODEL", "EVAL_JUDGE_REASONING_EFFORT", "EVAL_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_unset_eval_vars_fall_back_to_documented_defaults(clean_env):
    s = load_settings()
    assert (s.eval_provider, s.eval_judge_model, s.eval_embedding_model, s.eval_judge_reasoning_effort, s.eval_timeout_seconds) == (
        "ragas",
        "gpt-4.1-mini",
        "text-embedding-3-small",
        None,
        360,
    )


def test_eval_vars_are_read(clean_env):
    clean_env.setenv("EVAL_JUDGE_MODEL", "gpt-4.1")
    clean_env.setenv("EVAL_JUDGE_REASONING_EFFORT", "LOW")
    clean_env.setenv("EVAL_TIMEOUT_SECONDS", "90")
    s = load_settings()
    assert (s.eval_judge_model, s.eval_judge_reasoning_effort, s.eval_timeout_seconds) == ("gpt-4.1", ReasoningEffort.LOW, 90)


@pytest.mark.parametrize("raw", ["0", "-5", "soon", "1.5"])
def test_invalid_timeouts_are_rejected(clean_env, raw):
    clean_env.setenv("EVAL_TIMEOUT_SECONDS", raw)
    with pytest.raises(ValueError, match="EVAL_TIMEOUT_SECONDS"):
        load_settings()


def test_provider_selection(settings):
    assert isinstance(build_evaluator(replace(settings, eval_provider="fake")), FakeEvaluator)
    assert isinstance(build_evaluator(replace(settings, eval_provider="ragas", openai_api_key="sk-test")), RagasEvaluator)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        build_evaluator(replace(settings, eval_provider="ragas", openai_api_key=None))
    with pytest.raises(ConfigError, match="Unknown EVAL_PROVIDER"):
        build_evaluator(replace(settings, eval_provider="nope"))


def test_container_exposes_the_configured_evaluator(container):
    assert isinstance(container.evaluator, FakeEvaluator)
    assert container.evaluator is container.evaluator  # built once


def test_judge_config_is_only_needed_when_evaluation_is_used(settings):
    """Regression: building the container mustn't require the evaluator's OPENAI_API_KEY."""
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from qdrant_client import QdrantClient

    from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
    from rag_chatbot.container import build_container

    container = build_container(
        replace(settings, eval_provider="ragas", openai_api_key=None),  # judge can't be built without a key
        llm=FakeLLMAdapter(),
        embeddings=DeterministicFakeEmbedding(size=32),
        client=QdrantClient(":memory:"),
        embedding_dim=32,
    )
    assert container.sdg is not None  # other services work
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        container.evaluator  # the problem surfaces only when evaluation is used


@pytest.mark.parametrize(
    "judge, answer, expected",
    [
        ("gpt-4.1-mini", "gpt-4.1-mini", True),
        ("  GPT-4.1-mini ", "gpt-4.1-MINI", True),
        ("gpt-4.1-mini", "gpt-5-mini", False),
        ("gpt-4.1-mini", "", False),
    ],
)
def test_judge_matches_answer_model(settings, judge, answer, expected):
    assert judge_matches_answer_model(replace(settings, eval_judge_model=judge, llm_model=answer)) is expected
