from typing import Protocol, runtime_checkable

from rag_chatbot.domain.evals import EvalRun, EvalRunManifest, EvalRunSummary, RunStatus, SampleResult
from rag_chatbot.domain.models import TokenUsage


@runtime_checkable
class EvalRunRepository(Protocol):
    """Outbound port for saving evaluation runs and reading them back.

    Lifecycle: create (RUNNING) -> update_progress* -> complete | fail. A completed or failed run
    is final: every later write raises RunFinalized. Every adapter must pass
    tests/test_eval_run_repository_contract.py.
    """

    def create(self, manifest: EvalRunManifest) -> None:
        """Raises RunAlreadyExists; ValueError unless the manifest is a valid new RUNNING run."""
        ...

    def update_progress(self, run_id: str, progress: float | None, message: str) -> None:
        """Raises RunNotFound, RunFinalized."""
        ...

    def complete(
        self,
        run_id: str,
        samples: list[SampleResult],
        aggregate: dict[str, float | None],
        judge_usage: TokenUsage | None,
        answer_usage: TokenUsage | None,
    ) -> None:
        """Mark COMPLETED and set finished_at. Raises RunNotFound, RunFinalized, ValueError (non-finite scores)."""
        ...

    def fail(self, run_id: str, error: str, samples: list[SampleResult] | None = None) -> None:
        """Mark FAILED, set finished_at, keep any partial samples. Raises RunNotFound, RunFinalized."""
        ...

    def get(self, run_id: str) -> EvalRun:
        """Raises RunNotFound."""
        ...

    def list(
        self,
        *,
        testset_id: str | None = None,
        pipeline: str | None = None,
        batch_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[EvalRunSummary]:
        """Newest first; every filter given must match. Does not load samples."""
        ...
