"""In-memory EvalRunRepository, for tests and for exercising services without touching disk."""

from __future__ import annotations

import copy
import threading
from dataclasses import replace
from datetime import UTC, datetime
from typing import Callable

from rag_chatbot.domain.evals import (
    EvalRun,
    EvalRunManifest,
    EvalRunSummary,
    RunAlreadyExists,
    RunFinalized,
    RunNotFound,
    RunStatus,
    SampleResult,
    matches,
    sort_newest_first,
    validate_new_run,
    validate_progress,
    validate_results,
)
from rag_chatbot.domain.models import TokenUsage


class InMemoryEvalRunRepository:
    def __init__(self, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._runs: dict[str, EvalRun] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def create(self, manifest: EvalRunManifest) -> None:
        validate_new_run(manifest)
        with self._lock:
            if manifest.run_id in self._runs:
                raise RunAlreadyExists(manifest.run_id)
            self._runs[manifest.run_id] = EvalRun(manifest=copy.deepcopy(manifest), samples=[])

    def update_progress(self, run_id: str, progress: float | None, message: str) -> None:
        validate_progress(progress)
        self._change(run_id, lambda m: replace(m, progress=progress, message=message))

    def complete(
        self,
        run_id: str,
        samples: list[SampleResult],
        aggregate: dict[str, float | None],
        judge_usage: TokenUsage | None,
        answer_usage: TokenUsage | None,
    ) -> None:
        validate_results(samples, aggregate)
        self._change(
            run_id,
            lambda m: replace(
                m,
                status=RunStatus.COMPLETED,
                finished_at=self._clock(),
                progress=1.0,
                aggregate=dict(aggregate),
                judge_usage=judge_usage,
                answer_usage=answer_usage,
            ),
            samples=samples,
        )

    def fail(self, run_id: str, error: str, samples: list[SampleResult] | None = None) -> None:
        validate_results(samples or [], {})
        self._change(
            run_id, lambda m: replace(m, status=RunStatus.FAILED, finished_at=self._clock(), error=error), samples=samples
        )

    def get(self, run_id: str) -> EvalRun:
        with self._lock:
            return copy.deepcopy(self._existing(run_id))

    def list(
        self,
        *,
        testset_id: str | None = None,
        pipeline: str | None = None,
        batch_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[EvalRunSummary]:
        with self._lock:
            summaries = [EvalRunSummary.from_manifest(r.manifest) for r in self._runs.values()]
        return sort_newest_first(
            [s for s in summaries if matches(s, testset_id=testset_id, pipeline=pipeline, batch_id=batch_id, status=status)]
        )

    def _existing(self, run_id: str) -> EvalRun:
        try:
            return self._runs[run_id]
        except KeyError:
            raise RunNotFound(run_id) from None

    def _change(self, run_id: str, change, samples: list[SampleResult] | None = None) -> None:
        with self._lock:
            run = self._existing(run_id)
            if run.manifest.is_final:
                raise RunFinalized(f"{run_id} is {run.manifest.status.value}")
            self._runs[run_id] = EvalRun(
                manifest=change(run.manifest),
                samples=copy.deepcopy(samples) if samples is not None else run.samples,
            )
