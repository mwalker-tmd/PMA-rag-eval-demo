from typing import Callable, Protocol, Sequence, runtime_checkable

from rag_chatbot.domain.evals import DEFAULT_METRICS, EvalSample, EvaluationResult

ProgressCallback = Callable[[float | None, str], None]


@runtime_checkable
class EvaluatorPort(Protocol):
    """Outbound port that scores pipeline answers against reference answers."""

    @property
    def judge_model(self) -> str: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def ragas_version(self) -> str: ...

    def supported_metrics(self) -> list[str]: ...

    def preflight(self) -> None:
        """Cheap check that the judge model works for scoring. Raises ModelIncompatible."""
        ...

    def evaluate(
        self,
        samples: list[EvalSample],
        metrics: Sequence[str] = DEFAULT_METRICS,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> EvaluationResult:
        """One score dict per sample, in input order, with exactly `metrics` as keys.

        A score is a finite float, or None when the metric couldn't be computed for that sample.
        Raises UnknownMetric (before any work) or EvaluationError.
        """
        ...
