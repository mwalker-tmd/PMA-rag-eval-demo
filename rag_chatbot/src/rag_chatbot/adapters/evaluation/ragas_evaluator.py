"""Ragas implementation of EvaluatorPort. Ragas is imported lazily (see adapters/ragas_support.py)."""

from __future__ import annotations

import math
from typing import Sequence

from rag_chatbot.adapters import ragas_support
from rag_chatbot.adapters.langchain_usage import TokenUsageCallback
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.domain.evals import (
    DEFAULT_METRICS,
    EvalSample,
    EvaluationError,
    EvaluationResult,
    UnknownMetric,
)
from rag_chatbot.domain.models import ReasoningEffort
from rag_chatbot.ports.evaluator import ProgressCallback

# Our metric name (= the name Ragas 0.2.10 reports) -> Ragas metric class in ragas.metrics.
METRIC_CLASSES: dict[str, str] = {
    "context_recall": "LLMContextRecall",
    "faithfulness": "Faithfulness",
    "factual_correctness": "FactualCorrectness",
    "answer_relevancy": "ResponseRelevancy",
    "context_entity_recall": "ContextEntityRecall",
    "noise_sensitivity_relevant": "NoiseSensitivity",
}


class RagasEvaluator:
    def __init__(
        self,
        *,
        judge_model: str,
        embedding_model: str,
        api_key: str | None = None,
        judge_reasoning_effort: ReasoningEffort | None = None,
        timeout_seconds: int = 360,
    ) -> None:
        self._judge_model = judge_model
        self._embedding_model = embedding_model
        self._api_key = api_key
        self._effort = judge_reasoning_effort
        self._timeout = timeout_seconds

    @property
    def judge_model(self) -> str:
        return self._judge_model

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def ragas_version(self) -> str:
        return ragas_support.ragas_version()

    def supported_metrics(self) -> list[str]:
        return list(METRIC_CLASSES)

    def preflight(self) -> None:
        ragas_support.preflight(
            self._llm(), model=self._judge_model, setting="EVAL_JUDGE_MODEL", other_error=EvaluationError
        )

    def evaluate(
        self,
        samples: list[EvalSample],
        metrics: Sequence[str] = DEFAULT_METRICS,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> EvaluationResult:
        metrics = list(metrics)
        unknown = [m for m in metrics if m not in METRIC_CLASSES]
        if unknown:
            raise UnknownMetric(f"Unknown metric(s) {unknown}; supported: {list(METRIC_CLASSES)}")
        if not samples:
            return EvaluationResult(scores=[], usage=None)
        report = on_progress or (lambda _p, _m: None)
        report(0.0, f"Scoring {len(samples)} answers on {len(metrics)} metrics with {self._judge_model}")

        usage = TokenUsageCallback()
        try:
            import ragas.metrics as ragas_metrics
            from ragas import EvaluationDataset, RunConfig, evaluate

            dataset = EvaluationDataset.from_list(
                [
                    {
                        "user_input": s.user_input,
                        "retrieved_contexts": list(s.retrieved_contexts),
                        "reference_contexts": list(s.reference_contexts),
                        "response": s.response,
                        "reference": s.reference,
                    }
                    for s in samples
                ]
            )
            result = evaluate(
                dataset=dataset,
                metrics=[getattr(ragas_metrics, METRIC_CLASSES[name])() for name in metrics],
                llm=self._llm(usage),
                embeddings=ragas_support.ragas_embeddings(self._embedding_model, api_key=self._api_key),
                run_config=RunConfig(timeout=self._timeout),
                show_progress=False,
            )
            rows = list(result.scores)
        except (UnknownMetric, EvaluationError, ModelIncompatible):
            raise
        except Exception as e:
            raise EvaluationError(f"Ragas evaluation failed: {type(e).__name__}: {e}") from e

        if len(rows) != len(samples):
            raise EvaluationError(f"Ragas returned {len(rows)} score rows for {len(samples)} samples")
        scores = [{name: _score(row.get(name)) for name in metrics} for row in rows]
        report(1.0, f"Scored {len(samples)} answers")
        return EvaluationResult(scores=scores, usage=usage.usage())

    def _llm(self, usage: TokenUsageCallback | None = None):
        return ragas_support.ragas_llm(self._judge_model, api_key=self._api_key, effort=self._effort, usage=usage)


def _score(value) -> float | None:
    """Ragas reports a metric it couldn't compute as NaN; the domain uses None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
