"""Application service for evaluations: run pipelines against a saved test set, score, save, compare.

This module defines the Eval contract used by the UI and the CLI: the request and response types
below, the `EvaluationService` methods, and the errors they raise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Callable

from rag_chatbot.config import ConfigError, Settings, judge_matches_answer_model
from rag_chatbot.domain.datasets import Testset
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.domain.evals import (
    DEFAULT_METRICS,
    EvalRun,
    EvalRunManifest,
    EvalRunSummary,
    EvalSample,
    EvaluationError,
    RunStatus,
    SampleResult,
    new_batch_id,
    new_run_id,
)
from rag_chatbot.domain.models import PipelineResult, ReasoningEffort, TokenUsage
from rag_chatbot.ingestion.fingerprint import corpus_fingerprint
from rag_chatbot.ingestion.loader import list_pdfs
from rag_chatbot.pipelines.base import PipelineSpec
from rag_chatbot.ports.datasets import DatasetRepository
from rag_chatbot.ports.eval_runs import EvalRunRepository
from rag_chatbot.ports.evaluator import EvaluatorPort
from rag_chatbot.ports.llm import LLMPort
from rag_chatbot.services.chat_service import ChatService
from rag_chatbot.services.jobs import JobAlreadyRunning, JobId, JobRunner, JobStatus, ProgressReporter
from rag_chatbot.services.sdg_service import InvalidRequest

INTERRUPTED = "Interrupted (server restarted)"

# Base estimate, per question per pipeline. Rough on purpose: calibration against past completed runs
# with the same answer and judge models replaces them once any exist. Judge figures were measured in
# PR #20's live run (gpt-4.1-mini scoring all six metrics: ~17k input / ~3.7k output tokens per answer).
ANSWER_INPUT_TOKENS = 600
ANSWER_OUTPUT_TOKENS = 300
JUDGE_INPUT_TOKENS = 17_000
JUDGE_OUTPUT_TOKENS = 3_700


class PipelineUnavailable(Exception):
    """A requested pipeline can't run in this configuration (e.g. Rerank without COHERE_API_KEY)."""


@dataclass(frozen=True)
class StartEvaluationRequest:
    testset_id: str
    pipelines: tuple[str, ...] = ()  # empty = all available pipelines
    reasoning_effort: ReasoningEffort | None = None


@dataclass(frozen=True)
class StartedEvaluation:
    job_id: JobId
    batch_id: str
    run_ids: list[str]  # one per pipeline, already saved as RUNNING
    warnings: list[str]  # e.g. stale test set, judge == answer model


@dataclass(frozen=True)
class EvaluationEstimate:
    questions: int  # samples in the test set
    pipelines: list[str]  # what will run (resolved like start_evaluation)
    answer_model: str  # LLM_MODEL
    judge_model: str  # EVAL_JUDGE_MODEL
    est_answer_input_tokens: int
    est_answer_output_tokens: int
    est_judge_input_tokens: int
    est_judge_output_tokens: int
    est_answer_cost_usd: float | None  # None unless LLM_PRICE_* are both set
    est_judge_cost_usd: float | None  # None unless EVAL_JUDGE_PRICE_* are both set
    calibrated: bool  # True if per-answer usage came from past completed runs
    warnings: list[str]  # same as start_evaluation would report


@dataclass(frozen=True)
class _Validated:
    testset: Testset
    pipelines: list[str]
    effort: ReasoningEffort | None
    evaluator: EvaluatorPort
    warnings: list[str]


@dataclass(frozen=True)
class PipelineOption:
    name: str
    label: str
    available: bool
    reason: str | None  # why unavailable, e.g. missing COHERE_API_KEY


@dataclass(frozen=True)
class RunFilter:
    testset_id: str | None = None
    pipeline: str | None = None
    batch_id: str | None = None
    status: RunStatus | None = None


@dataclass(frozen=True)
class RunListItem:
    summary: EvalRunSummary
    testset_label: str  # "core-corpus v3", or "<id> (missing)"
    testset_is_stale: bool


@dataclass(frozen=True)
class RunComparison:
    run_ids: list[str]  # column order, as requested
    labels: dict[str, str]  # run_id -> "Rerank · core-corpus v3 · 2026-10-12 14:03"
    metrics: list[str]  # DEFAULT_METRICS order, then any extras
    values: dict[str, dict[str, float | None]]  # metric -> run_id -> aggregate mean
    reference_run_id: str | None
    deltas: dict[str, dict[str, float | None]]  # metric -> run_id -> value - reference (None: reference/missing)
    mixed_testsets: bool  # runs used different test sets
    incomplete_run_ids: list[str]  # selected runs that aren't COMPLETED


@dataclass(frozen=True)
class _Batch:
    """A validated request with its runs already created (RUNNING)."""

    batch_id: str
    testset: Testset
    effort: ReasoningEffort | None
    evaluator: EvaluatorPort
    run_ids: dict[str, str]  # pipeline -> run_id, in run order
    warnings: list[str]


class EvaluationService:
    def __init__(
        self,
        *,
        settings: Settings,
        specs: dict[str, PipelineSpec],
        llm: LLMPort,
        datasets: DatasetRepository,
        eval_runs: EvalRunRepository,
        jobs: JobRunner,
        evaluator: Callable[[], EvaluatorPort],  # lazy: may raise ConfigError
        chat: Callable[[], ChatService],  # lazy: builds the vector index if needed
        index_digest: Callable[[], str],
        row_delay_seconds: float = 2.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._specs = specs
        self._llm = llm
        self._datasets = datasets
        self._runs = eval_runs
        self._jobs = jobs
        self._evaluator = evaluator
        self._chat = chat
        self._index_digest = index_digest
        self._row_delay = row_delay_seconds
        self._clock = clock
        self._sleep = sleep

    # ---------- contract ----------

    def available_pipelines(self) -> list[PipelineOption]:
        options = []
        for name, spec in self._specs.items():
            reason = "COHERE_API_KEY is not set in .env" if spec.requires_cohere and not self._settings.cohere_api_key else None
            options.append(PipelineOption(name=name, label=spec.label, available=reason is None, reason=reason))
        return options

    def warnings(self) -> list[str]:
        warnings = []
        if judge_matches_answer_model(self._settings):
            warnings.append(
                f"The judge model ({self._settings.eval_judge_model}) is the same as the answering model "
                "(LLM_MODEL). A model grading its own answers tends to score them too generously; "
                "set EVAL_JUDGE_MODEL to a different model."
            )
        return warnings

    def estimate_evaluation(self, req: StartEvaluationRequest) -> EvaluationEstimate:
        """Token and cost estimate for a request. Validates like start_evaluation; creates nothing, calls no model."""
        v = self._validate(req)
        runs = len(v.testset.samples) * len(v.pipelines)
        rates = self._calibrated_rates(self._llm.model, v.evaluator.judge_model)
        a_in, a_out, j_in, j_out = rates or (ANSWER_INPUT_TOKENS, ANSWER_OUTPUT_TOKENS, JUDGE_INPUT_TOKENS, JUDGE_OUTPUT_TOKENS)
        est = [round(runs * rate) for rate in (a_in, a_out, j_in, j_out)]
        s = self._settings
        return EvaluationEstimate(
            questions=len(v.testset.samples),
            pipelines=v.pipelines,
            answer_model=self._llm.model,
            judge_model=v.evaluator.judge_model,
            est_answer_input_tokens=est[0],
            est_answer_output_tokens=est[1],
            est_judge_input_tokens=est[2],
            est_judge_output_tokens=est[3],
            est_answer_cost_usd=_cost(est[0], est[1], s.llm_price_input_per_mtok, s.llm_price_output_per_mtok),
            est_judge_cost_usd=_cost(est[2], est[3], s.eval_judge_price_input_per_mtok, s.eval_judge_price_output_per_mtok),
            calibrated=rates is not None,
            warnings=v.warnings,
        )

    def start_evaluation(self, req: StartEvaluationRequest) -> StartedEvaluation:
        if (active := self._jobs.active()) is not None:  # before creating any runs
            raise JobAlreadyRunning(active)
        batch = self._prepare(req)
        label = f"Evaluate {', '.join(self._specs[p].label for p in batch.run_ids)} on {_testset_label(batch.testset)}"
        try:
            job_id = self._jobs.submit("eval", label, lambda report: self._execute(batch, report))
        except JobAlreadyRunning:  # lost a race with another submit: don't leave orphaned RUNNING runs
            for run_id in batch.run_ids.values():
                self._runs.fail(run_id, "Not started: another job was already running")
            raise
        return StartedEvaluation(
            job_id=job_id, batch_id=batch.batch_id, run_ids=list(batch.run_ids.values()), warnings=batch.warnings
        )

    def run_evaluation(
        self,
        req: StartEvaluationRequest,
        on_progress: ProgressReporter | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> list[EvalRunSummary]:
        """Synchronous evaluation (the CLI). Batch warnings (e.g. a stale test set) go to `on_warning`
        before any answering starts; start_evaluation returns the same warnings in StartedEvaluation."""
        batch = self._prepare(req)
        for warning in batch.warnings:
            (on_warning or (lambda _w: None))(warning)
        self._execute(batch, on_progress or (lambda _p, _m: None))
        return [self._summary(run_id) for run_id in batch.run_ids.values()]

    def job_status(self, job_id: JobId) -> JobStatus:
        return self._jobs.status(job_id)

    def active_job(self) -> JobStatus | None:
        return self._jobs.active()

    def list_runs(self, filter: RunFilter | None = None) -> list[RunListItem]:
        f = filter or RunFilter()
        testsets = {s.id: s for s in self._datasets.list()}
        current = self._corpus_digest()
        items = []
        for s in self._runs.list(testset_id=f.testset_id, pipeline=f.pipeline, batch_id=f.batch_id, status=f.status):
            ts = testsets.get(s.testset_id)
            items.append(
                RunListItem(
                    summary=s,
                    testset_label=f"{ts.name} v{ts.version}" if ts else f"{s.testset_id} (missing)",
                    testset_is_stale=ts is not None and ts.corpus_digest != current,
                )
            )
        return items

    def get_run(self, run_id: str) -> EvalRun:
        return self._runs.get(run_id)

    def compare_runs(self, run_ids: list[str], reference_run_id: str | None = None) -> RunComparison:
        run_ids = list(dict.fromkeys(run_ids))  # keep order, drop duplicates
        if not run_ids:
            raise InvalidRequest("Select at least one run to compare.")
        if reference_run_id is not None and reference_run_id not in run_ids:
            raise InvalidRequest("The reference run must be one of the selected runs.")
        manifests = {run_id: self._runs.get(run_id).manifest for run_id in run_ids}  # RunNotFound
        testsets = {s.id: s for s in self._datasets.list()}

        extras = sorted({m for man in manifests.values() for m in man.aggregate} - set(DEFAULT_METRICS))
        present = {m for man in manifests.values() for m in (*man.metrics, *man.aggregate)}
        metrics = [m for m in DEFAULT_METRICS if m in present] + extras
        values = {m: {r: manifests[r].aggregate.get(m) for r in run_ids} for m in metrics}
        deltas: dict[str, dict[str, float | None]] = {}
        for m in metrics:
            ref = values[m].get(reference_run_id) if reference_run_id else None
            deltas[m] = {
                r: (values[m][r] - ref if r != reference_run_id and ref is not None and values[m][r] is not None else None)
                for r in run_ids
            }

        def label(man: EvalRunManifest) -> str:
            ts = testsets.get(man.testset_id)
            spec = self._specs.get(man.pipeline)
            return " · ".join(
                [
                    spec.label if spec else man.pipeline,
                    f"{ts.name} v{ts.version}" if ts else man.testset_id,
                    f"{man.started_at:%Y-%m-%d %H:%M}",
                ]
            )

        return RunComparison(
            run_ids=run_ids,
            labels={r: label(manifests[r]) for r in run_ids},
            metrics=metrics,
            values=values,
            reference_run_id=reference_run_id,
            deltas=deltas,
            mixed_testsets=len({man.testset_id for man in manifests.values()}) > 1,
            incomplete_run_ids=[r for r in run_ids if manifests[r].status is not RunStatus.COMPLETED],
        )

    def recover_interrupted_runs(self) -> list[str]:
        recovered = []
        for s in self._runs.list(status=RunStatus.RUNNING):
            self._runs.fail(s.run_id, INTERRUPTED)
            recovered.append(s.run_id)
        return recovered

    # ---------- internals ----------

    def _prepare(self, req: StartEvaluationRequest) -> _Batch:
        """Validate the request and create one RUNNING run per pipeline."""
        v = self._validate(req)
        testset, pipelines, effort, evaluator, warnings = v.testset, v.pipelines, v.effort, v.evaluator, v.warnings
        started_at = self._clock()
        return self._create_runs(started_at, testset, pipelines, effort, evaluator, warnings)

    def _validate(self, req: StartEvaluationRequest) -> _Validated:
        """Everything start_evaluation and estimate_evaluation check, in one place, so they always agree."""
        testset = self._datasets.get(req.testset_id)  # TestsetNotFound
        if not testset.samples:
            raise InvalidRequest(f"Test set {req.testset_id} has no questions.")
        data_dir = Path(self._settings.data_dir)
        if not data_dir.is_dir() or not list_pdfs(data_dir):  # answering needs the index, which needs the PDFs
            raise InvalidRequest(f"No PDFs found in {data_dir}: add the corpus before evaluating.")
        pipelines = self._pipelines(req.pipelines)
        effort = req.reasoning_effort
        if effort is not None and effort not in self._llm.supported_reasoning_efforts():
            raise InvalidRequest(f"Reasoning effort '{effort.value}' isn't supported by {self._llm.model}.")
        try:
            evaluator = self._evaluator()
        except ConfigError as e:
            raise InvalidRequest(f"Evaluation isn't configured: {e}") from e

        warnings = self.warnings()
        if testset.manifest.corpus.digest != self._corpus_digest():
            warnings.append(
                f"Test set {_testset_label(testset)} is stale: the PDFs in data/ have changed since it was made, "
                "so its questions may not match the current corpus."
            )
        return _Validated(testset, pipelines, effort, evaluator, warnings)

    def _create_runs(self, started_at, testset, pipelines, effort, evaluator, warnings) -> _Batch:
        batch_id = new_batch_id(started_at)
        index_digest = self._index_digest()
        run_ids: dict[str, str] = {}
        for pipeline in pipelines:
            run_id = new_run_id(started_at, pipeline)
            self._runs.create(
                EvalRunManifest(
                    run_id=run_id,
                    batch_id=batch_id,
                    testset_id=testset.manifest.id,
                    pipeline=pipeline,
                    pipeline_settings=dict(self._specs[pipeline].settings),
                    answer_model=self._llm.model,
                    reasoning_effort=effort.value if effort else None,
                    judge_model=evaluator.judge_model,
                    judge_reasoning_effort=(
                        self._settings.eval_judge_reasoning_effort.value if self._settings.eval_judge_reasoning_effort else None
                    ),
                    embedding_model=evaluator.embedding_model,
                    metrics=list(DEFAULT_METRICS),
                    ragas_version=evaluator.ragas_version,
                    index_manifest_digest=index_digest,
                    status=RunStatus.RUNNING,
                    progress=0.0,
                    message="Waiting to start",
                    started_at=started_at,
                    finished_at=None,
                    error=None,
                )
            )
            run_ids[pipeline] = run_id
        return _Batch(batch_id, testset, effort, evaluator, run_ids, warnings)

    def _pipelines(self, requested: tuple[str, ...]) -> list[str]:
        options = {o.name: o for o in self.available_pipelines()}
        if not requested:
            chosen = [name for name, o in options.items() if o.available]
            if not chosen:
                raise PipelineUnavailable("No pipelines are available in this configuration.")
            return chosen
        chosen = list(dict.fromkeys(requested))
        unknown = [p for p in chosen if p not in options]
        if unknown:
            raise InvalidRequest(f"Unknown pipeline(s): {', '.join(unknown)}. Choose from {', '.join(options)}.")
        unavailable = [options[p] for p in chosen if not options[p].available]
        if unavailable:
            raise PipelineUnavailable("; ".join(f"{o.label}: {o.reason}" for o in unavailable))
        return chosen

    def _execute(self, batch: _Batch, report: ProgressReporter) -> str:
        """Answer and score each pipeline in turn. Returns the batch id (the job's result_id)."""
        n, pipelines = len(batch.testset.samples), list(batch.run_ids)
        total_steps = 2 * n * len(pipelines)
        done = 0
        remaining = list(batch.run_ids.values())
        try:
            report(0.0, "Checking the judge model")
            try:
                batch.evaluator.preflight()
            except (ModelIncompatible, EvaluationError) as e:  # costs nothing yet: fail the whole batch
                for run_id in remaining:
                    self._runs.fail(run_id, f"{type(e).__name__}: {e}")
                remaining = []
                raise

            report(0.0, "Preparing the pipelines")
            chat = self._chat()
            for pipeline in pipelines:
                run_id = batch.run_ids[pipeline]
                done = self._run_pipeline(batch, chat, pipeline, run_id, report, done, total_steps)
                remaining.remove(run_id)
            report(1.0, f"Finished {len(pipelines)} run(s)")
            return batch.batch_id
        except BaseException as e:
            for run_id in remaining:  # never leave this batch's runs RUNNING
                self._runs.fail(run_id, f"Batch stopped: {type(e).__name__}: {e}")
            raise

    def _run_pipeline(self, batch, chat, pipeline, run_id, report, done, total_steps) -> int:
        n = len(batch.testset.samples)
        label = self._specs[pipeline].label
        answers: list[PipelineResult] = []
        for i, sample in enumerate(batch.testset.samples):
            if i and self._row_delay:
                self._sleep(self._row_delay)
            answers.append(chat.ask(pipeline, sample.user_input, batch.effort))
            done += 1
            message = f"{label}: answered {i + 1} of {n}"
            self._runs.update_progress(run_id, (i + 1) / (2 * n), message)
            report(done / total_steps, message)

        results = [_partial_result(s, a) for s, a in zip(batch.testset.samples, answers)]
        answer_usage = _sum_usage(answers)
        answered = [i for i, a in enumerate(answers) if a.ok and a.answer]
        if not answered:
            self._runs.fail(
                run_id,
                f"No answers to score: all {n} questions failed to answer",
                samples=results,
                answer_usage=answer_usage,
            )
            return done + n

        self._runs.update_progress(run_id, 0.5, f"{label}: scoring {len(answered)} answers")
        report(done / total_steps, f"{label}: scoring {len(answered)} answers")
        eval_samples = [
            EvalSample(
                user_input=results[i].user_input,
                reference=results[i].reference,
                response=results[i].response or "",
                retrieved_contexts=results[i].retrieved_contexts,
                reference_contexts=list(batch.testset.samples[i].reference_contexts),
            )
            for i in answered
        ]
        highest = [0.0]

        def judging_progress(fraction: float | None, message: str) -> None:
            # Map the evaluator's 0–1 onto the second half of this run, and onto the batch's scoring steps.
            # Never let progress go backwards, whatever order the evaluator reports in.
            if fraction is None:
                return
            highest[0] = max(highest[0], min(1.0, max(0.0, fraction)))
            self._runs.update_progress(run_id, 0.5 + 0.5 * highest[0], f"{label}: {message}")
            report((done + highest[0] * n) / total_steps, f"{label}: {message}")

        try:
            scored = batch.evaluator.evaluate(eval_samples, on_progress=judging_progress)
        except (EvaluationError, ModelIncompatible) as e:  # this run fails; the batch carries on
            self._runs.fail(run_id, f"{type(e).__name__}: {e}", samples=results, answer_usage=answer_usage)
            return done + n

        by_index = dict(zip(answered, scored.scores))
        empty = {m: None for m in DEFAULT_METRICS}
        results = [replace(r, scores=dict(by_index.get(i, empty))) for i, r in enumerate(results)]
        aggregate = {
            m: (fmean(vals) if (vals := [r.scores.get(m) for r in results if r.scores.get(m) is not None]) else None)
            for m in DEFAULT_METRICS
        }
        self._runs.complete(run_id, results, aggregate, judge_usage=scored.usage, answer_usage=answer_usage)
        done += n
        report(done / total_steps, f"{label}: done")
        return done

    def _summary(self, run_id: str) -> EvalRunSummary:
        return EvalRunSummary.from_manifest(self._runs.get(run_id).manifest)

    def _corpus_digest(self) -> str:
        return corpus_fingerprint(Path(self._settings.data_dir)).digest


    def _calibrated_rates(self, answer_model: str, judge_model: str) -> tuple[float, float, float, float] | None:
        """Average (answer in, answer out, judge in, judge out) tokens per question over past completed runs
        with the same answer and judge models, or None if there are none with recorded usage."""
        totals, answered, judged = [0, 0, 0, 0], 0, 0
        for s in self._runs.list(status=RunStatus.COMPLETED):
            if s.answer_model != answer_model or s.judge_model != judge_model:
                continue
            run = self._runs.get(s.run_id)
            m, n = run.manifest, len(run.samples)
            scored = sum(1 for r in run.samples if r.response)
            if m.answer_usage and n:
                totals[0] += m.answer_usage.input_tokens
                totals[1] += m.answer_usage.output_tokens
                answered += n
            if m.judge_usage and scored:
                totals[2] += m.judge_usage.input_tokens
                totals[3] += m.judge_usage.output_tokens
                judged += scored
        if not answered or not judged:
            return None
        return totals[0] / answered, totals[1] / answered, totals[2] / judged, totals[3] / judged


def _cost(tokens_in: int, tokens_out: int, price_in: float | None, price_out: float | None) -> float | None:
    if price_in is None or price_out is None:
        return None
    return round(tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out, 4)


def _testset_label(testset: Testset) -> str:
    return f"{testset.manifest.name} v{testset.manifest.version}"


def _partial_result(sample, answer: PipelineResult) -> SampleResult:
    """A sample's answer, before scoring (every score None)."""
    return SampleResult(
        user_input=sample.user_input,
        reference=sample.reference,
        response=answer.answer if answer.ok else None,
        retrieved_contexts=[c.text for c in answer.chunks],
        scores={m: None for m in DEFAULT_METRICS},
        latency_ms=answer.latency_ms,
        usage=answer.llm_response.usage if answer.llm_response else None,
        error=answer.error,
    )


def _sum_usage(answers) -> TokenUsage | None:
    usages = [a.llm_response.usage for a in answers if a.llm_response and a.llm_response.usage]
    if not usages:
        return None
    reasoning = [u.reasoning_tokens for u in usages if u.reasoning_tokens]
    return TokenUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        total_tokens=sum(u.total_tokens for u in usages),
        reasoning_tokens=sum(reasoning) if reasoning else None,
    )
