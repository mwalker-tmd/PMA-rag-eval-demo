"""Domain types for evaluation runs. No third-party imports: adapters translate to and from these."""

from __future__ import annotations

import math
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from rag_chatbot.domain.models import TokenUsage

# Run ids become folder names: always check with .fullmatch() (match + "$" would accept a trailing newline).
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")


class RunNotFound(Exception):
    pass


class RunAlreadyExists(Exception):
    pass


class RunFinalized(Exception):
    """The run is COMPLETED or FAILED; it can't be changed any more."""


class RunCorrupt(Exception):
    """A stored run can't be read safely: unreadable or invalid data, or a file that isn't a regular file."""


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SampleResult:
    user_input: str
    reference: str
    response: str | None  # None if answering failed
    retrieved_contexts: list[str]
    scores: dict[str, float | None]  # metric name -> score; None = not scored (or NaN from the evaluator)
    latency_ms: float | None
    usage: TokenUsage | None  # answering-model usage
    error: str | None


@dataclass(frozen=True)
class EvalRunManifest:
    run_id: str  # see new_run_id()
    batch_id: str  # shared by runs started together
    testset_id: str
    pipeline: str
    pipeline_settings: dict[str, Any]  # PipelineSpec.settings snapshot
    answer_model: str
    reasoning_effort: str | None
    judge_model: str
    judge_reasoning_effort: str | None
    embedding_model: str  # EVAL_EMBEDDING_MODEL
    metrics: list[str]
    ragas_version: str
    index_manifest_digest: str
    status: RunStatus
    progress: float | None
    message: str
    started_at: datetime  # UTC
    finished_at: datetime | None
    error: str | None
    aggregate: dict[str, float | None] = field(default_factory=dict)  # mean per metric, ignoring None
    answer_usage: TokenUsage | None = None
    judge_usage: TokenUsage | None = None

    @property
    def is_final(self) -> bool:
        return self.status is not RunStatus.RUNNING


@dataclass(frozen=True)
class EvalRun:
    manifest: EvalRunManifest
    samples: list[SampleResult]


@dataclass(frozen=True)
class EvalRunSummary:
    """Cheap listing entry: no samples."""

    run_id: str
    batch_id: str
    testset_id: str
    pipeline: str
    answer_model: str
    reasoning_effort: str | None
    judge_model: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None
    aggregate: dict[str, float | None]

    @classmethod
    def from_manifest(cls, m: EvalRunManifest) -> EvalRunSummary:
        return cls(
            run_id=m.run_id,
            batch_id=m.batch_id,
            testset_id=m.testset_id,
            pipeline=m.pipeline,
            answer_model=m.answer_model,
            reasoning_effort=m.reasoning_effort,
            judge_model=m.judge_model,
            status=m.status,
            started_at=m.started_at,
            finished_at=m.finished_at,
            aggregate=dict(m.aggregate),
        )


def new_run_id(started_at: datetime, pipeline: str) -> str:
    """`run_<yyyymmdd-hhmmss>_<pipeline>_<rand4>`."""
    return f"run_{started_at:%Y%m%d-%H%M%S}_{pipeline}_{secrets.token_hex(2)}"


def new_batch_id(started_at: datetime) -> str:
    return f"batch_{started_at:%Y%m%d-%H%M%S}_{secrets.token_hex(2)}"


def sort_newest_first(summaries: list[EvalRunSummary]) -> list[EvalRunSummary]:
    return sorted(summaries, key=lambda s: (s.started_at, s.run_id), reverse=True)


def matches(
    s: EvalRunSummary,
    *,
    testset_id: str | None = None,
    pipeline: str | None = None,
    batch_id: str | None = None,
    status: RunStatus | None = None,
) -> bool:
    return (
        (testset_id is None or s.testset_id == testset_id)
        and (pipeline is None or s.pipeline == pipeline)
        and (batch_id is None or s.batch_id == batch_id)
        and (status is None or s.status is status)
    )


# ---------- validation shared by every EvalRunRepository ----------


def _is_utc(dt: datetime | None) -> bool:
    return dt is not None and dt.utcoffset() == timedelta(0)


def _check_score(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or None, got {value!r} (map NaN to None)")


def _check_progress(progress: float | None) -> None:
    if progress is not None and (
        isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0.0 <= progress <= 1.0
    ):
        raise ValueError(f"progress must be between 0 and 1, or None; got {progress!r}")


def validate_new_run(m: EvalRunManifest) -> None:
    if not isinstance(m.run_id, str) or not RUN_ID_PATTERN.fullmatch(m.run_id):
        raise ValueError(f"Invalid run id {m.run_id!r}: use letters, digits, '-' or '_'")
    if m.status is not RunStatus.RUNNING:
        raise ValueError(f"A new run must have status RUNNING, got {m.status.value}")
    if m.finished_at is not None or m.error is not None:
        raise ValueError("A new run can't have finished_at or error set")
    if not _is_utc(m.started_at):
        raise ValueError("started_at must be a UTC datetime")
    _check_progress(m.progress)
    for metric, value in m.aggregate.items():
        _check_score(f"aggregate[{metric!r}]", value)


def validate_progress(progress: float | None) -> None:
    _check_progress(progress)


def validate_results(samples: list[SampleResult], aggregate: dict[str, float | None]) -> None:
    for metric, value in aggregate.items():
        _check_score(f"aggregate[{metric!r}]", value)
    for i, sample in enumerate(samples):
        for metric, value in sample.scores.items():
            _check_score(f"samples[{i}].scores[{metric!r}]", value)


# ---------- scoring (EvaluatorPort) ----------

# Metric names as Ragas reports them. Order is the display order everywhere.
DEFAULT_METRICS: tuple[str, ...] = (
    "context_recall",  # LLMContextRecall: did retrieval find what the reference answer needs?
    "faithfulness",  # Faithfulness: is the answer supported by the retrieved contexts?
    "factual_correctness",  # FactualCorrectness: does the answer agree with the reference?
    "answer_relevancy",  # ResponseRelevancy: does the answer address the question?
    "context_entity_recall",  # ContextEntityRecall: are the reference's entities in the contexts?
    "noise_sensitivity_relevant",  # NoiseSensitivity: how often do irrelevant contexts cause errors? (lower is better)
)


class EvaluationError(Exception):
    """Scoring failed. The message carries the underlying error."""


class UnknownMetric(ValueError):
    """A requested metric isn't supported by the evaluator."""


@dataclass(frozen=True)
class EvalSample:
    user_input: str
    reference: str
    response: str
    retrieved_contexts: list[str]
    reference_contexts: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    scores: list[dict[str, float | None]]  # one dict per input sample, same order; NaN -> None
    usage: TokenUsage | None  # judge usage
