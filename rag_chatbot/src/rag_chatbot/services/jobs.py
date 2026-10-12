"""In-process background job runner.

Long operations (test set generation, evaluations) run on a worker thread so the Streamlit
page stays responsive. The runner lives in the cached container, so job state survives page
reloads and is shared across browser tabs. It does not survive a server restart: services
that need durable status record it themselves.

One job runs at a time. A second submit while one is queued or running raises
JobAlreadyRunning, which keeps API rate limits under control and avoids concurrent writes to
the embedded Qdrant index.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Callable, Protocol

JobId = str

HISTORY_LIMIT = 50


class JobAlreadyRunning(Exception):
    def __init__(self, active: JobStatus) -> None:
        super().__init__(f"A job is already {active.state.value}: {active.label}")
        self.active = active


class JobNotFound(Exception):
    pass


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_active(self) -> bool:
        return self in (JobState.QUEUED, JobState.RUNNING)


@dataclass(frozen=True)
class JobStatus:
    """An immutable snapshot. The runner replaces it on every change; readers never see a half-update."""

    job_id: JobId
    kind: str  # "sdg" | "eval"
    label: str  # human-readable, e.g. "Generate 'core-corpus' (10)"
    state: JobState
    progress: float | None  # 0.0–1.0, or None if unknown
    message: str  # latest progress message
    error: str | None  # set when FAILED
    result_id: str | None  # set when COMPLETED (testset_id or batch_id)
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ProgressReporter(Protocol):
    def __call__(self, progress: float | None, message: str) -> None: ...


JobFn = Callable[[ProgressReporter], str | None]


class JobRunner:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        thread_factory: Callable[..., threading.Thread] = threading.Thread,  # tests can delay the start
    ) -> None:
        self._clock = clock
        self._thread_factory = thread_factory
        self._jobs: OrderedDict[JobId, JobStatus] = OrderedDict()  # oldest first
        self._changed = threading.Condition()

    def submit(self, kind: str, label: str, fn: JobFn) -> JobId:
        with self._changed:
            if (active := self._active_locked()) is not None:
                raise JobAlreadyRunning(active)
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            self._jobs[job_id] = JobStatus(
                job_id=job_id,
                kind=kind,
                label=label,
                state=JobState.QUEUED,
                progress=None,
                message="Queued",
                error=None,
                result_id=None,
                created_at=self._clock(),
                started_at=None,
                finished_at=None,
            )
            self._trim_history_locked()
        try:
            thread = self._thread_factory(target=self._run, args=(job_id, fn), name=f"job-{job_id}", daemon=True)
            thread.start()
        except BaseException as e:  # a job stuck in QUEUED would block every future submit
            self._update(job_id, state=JobState.FAILED, error=f"{type(e).__name__}: {e}", finished_at=self._clock())
            raise
        return job_id

    def status(self, job_id: JobId) -> JobStatus:
        with self._changed:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise JobNotFound(job_id) from None

    def active(self) -> JobStatus | None:
        with self._changed:
            return self._active_locked()

    def recent(self, limit: int = 20) -> list[JobStatus]:
        with self._changed:
            return list(reversed(self._jobs.values()))[:limit]

    def wait(self, job_id: JobId, timeout: float | None = None) -> JobStatus:
        """Block until the job finishes or `timeout` seconds pass; return its status either way."""
        with self._changed:
            if job_id not in self._jobs:
                raise JobNotFound(job_id)
            self._changed.wait_for(lambda: not self._jobs[job_id].state.is_active, timeout=timeout)
            return self._jobs[job_id]

    # --- worker ---

    def _run(self, job_id: JobId, fn: JobFn) -> None:
        self._update(job_id, state=JobState.RUNNING, started_at=self._clock(), message="Started")
        try:
            result = fn(self._reporter(job_id))
            if result is not None and not isinstance(result, str):
                raise TypeError(f"job function must return a result id (str) or None, got {type(result).__name__}")
        except BaseException as e:  # never leave a job stuck "running": it would block every future submit
            self._update(
                job_id,
                state=JobState.FAILED,
                error=f"{type(e).__name__}: {e}",
                message="Failed",
                finished_at=self._clock(),
            )
            if not isinstance(e, Exception):
                raise
            return
        self._update(
            job_id, state=JobState.COMPLETED, result_id=result, progress=1.0, message="Completed", finished_at=self._clock()
        )

    def _reporter(self, job_id: JobId) -> ProgressReporter:
        def report(progress: float | None, message: str) -> None:
            if progress is not None:
                progress = min(1.0, max(0.0, float(progress)))
            with self._changed:
                job = self._jobs.get(job_id)
                if job is None or job.state is not JobState.RUNNING:
                    return  # late report after the job finished: ignore
                self._jobs[job_id] = replace(self._jobs[job_id], progress=progress, message=message)
                self._changed.notify_all()

        return report

    # --- internals (call with the lock held) ---

    def _update(self, job_id: JobId, **changes) -> None:
        with self._changed:
            self._jobs[job_id] = replace(self._jobs[job_id], **changes)
            self._changed.notify_all()

    def _active_locked(self) -> JobStatus | None:
        return next((j for j in self._jobs.values() if j.state.is_active), None)

    def _trim_history_locked(self) -> None:
        while len(self._jobs) > HISTORY_LIMIT:
            oldest_id = next(iter(self._jobs))
            if self._jobs[oldest_id].state.is_active:  # can't happen with one job at a time, but never drop a live job
                break
            del self._jobs[oldest_id]
