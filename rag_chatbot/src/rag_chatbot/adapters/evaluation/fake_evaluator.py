"""Deterministic EvaluatorPort for tests and for exercising the UI without API spend."""

from __future__ import annotations

import hashlib
from typing import Sequence

from rag_chatbot.domain.evals import DEFAULT_METRICS, EvalSample, EvaluationResult, UnknownMetric
from rag_chatbot.domain.models import TokenUsage
from rag_chatbot.ports.evaluator import ProgressCallback


class FakeEvaluator:
    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self._fail_with = fail_with  # lets tests exercise failure handling
        self.calls: list[dict] = []

    @property
    def judge_model(self) -> str:
        return "fake-judge"

    @property
    def embedding_model(self) -> str:
        return "fake-embeddings"

    @property
    def ragas_version(self) -> str:
        return "fake"

    def supported_metrics(self) -> list[str]:
        return list(DEFAULT_METRICS)

    def preflight(self) -> None:
        return None

    def evaluate(
        self,
        samples: list[EvalSample],
        metrics: Sequence[str] = DEFAULT_METRICS,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> EvaluationResult:
        metrics = list(metrics)
        unknown = [m for m in metrics if m not in DEFAULT_METRICS]
        if unknown:
            raise UnknownMetric(f"Unknown metric(s) {unknown}")
        self.calls.append({"samples": len(samples), "metrics": metrics})
        if self._fail_with is not None:
            raise self._fail_with
        report = on_progress or (lambda _p, _m: None)
        report(0.0, f"Scoring {len(samples)} answers")
        scores = [{m: _fake_score(m, s) for m in metrics} for s in samples]
        report(1.0, f"Scored {len(samples)} answers")
        calls = len(samples) * len(metrics)
        usage = TokenUsage(input_tokens=calls * 100, output_tokens=calls * 10, total_tokens=calls * 110) if calls else None
        return EvaluationResult(scores=scores, usage=usage)


def _fake_score(metric: str, sample: EvalSample) -> float | None:
    if not sample.response.strip():
        return None
    digest = hashlib.sha256(f"{metric}\0{sample.user_input}\0{sample.response}".encode()).digest()
    return round(int.from_bytes(digest[:4], "big") / 0xFFFFFFFF, 4)
