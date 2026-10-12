"""Application service for synthetic test set generation (SDG).

This module defines the SDG contract used by the UI and the CLI: the request and response
types below, the `SdgService` methods, and the errors they raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from rag_chatbot.domain.datasets import (
    TESTSET_NAME_PATTERN,
    CorpusFingerprint,
    GenerationError,
    GenerationProgress,
    SourceDocument,
    Testset,
    TestsetAlreadyExists,
    TestsetManifest,
    TestsetNotFound,
    TestsetSummary,
    new_testset_id,
    samples_sha256,
)
from rag_chatbot.ingestion.fingerprint import corpus_fingerprint
from rag_chatbot.ingestion.loader import load_pdfs
from rag_chatbot.ports.datasets import DatasetRepository
from rag_chatbot.ports.testset_generator import TestsetGeneratorPort
from rag_chatbot.services.jobs import JobId, JobRunner, JobStatus, ProgressReporter

MIN_SIZE, MAX_SIZE = 1, 100

# Base estimate constants. Rough on purpose: calibration against a past test set's actual
# usage (same corpus and model) replaces them once one exists.
CHARS_PER_TOKEN = 4
KG_INPUT_PER_CORPUS_TOKEN = 3.0  # graph build: summaries, headlines, entities, themes over every chunk
KG_OUTPUT_PER_CORPUS_TOKEN = 0.3
SAMPLE_INPUT_TOKENS = 1_800  # per generated sample (measured ~1,770 with gpt-4.1 on the fixture corpus)
SAMPLE_OUTPUT_TOKENS = 200  # (measured ~180)

SAVE_ATTEMPTS = 5

# Progress split when the knowledge graph is built in this run.
KG_SHARE = 0.7


class InvalidRequest(Exception):
    """The request can't be run as given. The message is meant for the user."""


@dataclass(frozen=True)
class GenerateTestsetRequest:
    name: str  # 1–64 chars: letters, digits, '-', '_'
    size: int = 10  # 1–100
    reuse_kg_from: str | None = None  # testset_id whose knowledge graph to reuse


@dataclass(frozen=True)
class GenerationEstimate:
    corpus_tokens: int  # approx. tokens in the current data/ PDFs
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float | None  # None when pricing env vars are unset
    reuses_kg: bool
    calibrated: bool  # True if adjusted using a past test set's actual usage
    model: str  # SDG_LLM_MODEL


@dataclass(frozen=True)
class TestsetListItem:
    __test__ = False

    summary: TestsetSummary
    is_stale: bool  # corpus fingerprint != current data/


class SdgService:
    def __init__(
        self,
        *,
        datasets: DatasetRepository,
        generator: TestsetGeneratorPort,
        jobs: JobRunner,
        data_dir: Path,
        price_input_per_mtok: float | None = None,
        price_output_per_mtok: float | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._datasets = datasets
        self._generator = generator
        self._jobs = jobs
        self._data_dir = data_dir
        self._price_in = _validated_price("price_input_per_mtok", price_input_per_mtok)
        self._price_out = _validated_price("price_output_per_mtok", price_output_per_mtok)
        self._clock = clock
        self._corpus_cache: tuple[str, list[SourceDocument]] | None = None  # (digest, docs)

    # ---------- contract ----------

    def estimate(self, req: GenerateTestsetRequest) -> GenerationEstimate:
        reuses_kg = self._validate(req)
        fingerprint, docs = self._load_corpus()
        corpus_tokens = _corpus_tokens(docs)
        est_in, est_out = _base_estimate(corpus_tokens, req.size, reuses_kg)

        calibrated = False
        factors = self._calibration_factors(fingerprint, corpus_tokens)
        if factors is not None:
            est_in, est_out = round(est_in * factors[0]), round(est_out * factors[1])
            calibrated = True

        return GenerationEstimate(
            corpus_tokens=corpus_tokens,
            est_input_tokens=int(est_in),
            est_output_tokens=int(est_out),
            est_cost_usd=self._cost(est_in, est_out),
            reuses_kg=reuses_kg,
            calibrated=calibrated,
            model=self._generator.info.llm_model,
        )

    def start_generation(self, req: GenerateTestsetRequest) -> JobId:
        self._validate(req)  # fail fast, before a job is queued
        label = f"Generate '{req.name}' ({req.size})"
        return self._jobs.submit("sdg", label, lambda report: self.generate(req, report).id)

    def generate(self, req: GenerateTestsetRequest, on_progress: ProgressReporter | None = None) -> TestsetSummary:
        progress = _MonotonicProgress(on_progress)
        reuses_kg = self._validate(req)

        progress(0.0, "Loading PDFs")
        fingerprint, docs = self._load_corpus()
        if not docs:
            raise InvalidRequest(f"No PDFs found in {self._data_dir}")
        knowledge_graph = self._datasets.load_knowledge_graph(req.reuse_kg_from) if reuses_kg else None

        progress(0.0, f"Checking model {self._generator.info.llm_model}")
        self._generator.preflight()

        result = self._generator.generate(
            docs, req.size, knowledge_graph=knowledge_graph, on_progress=progress.stage_mapper(reuses_kg)
        )
        if not result.samples:
            raise GenerationError("The generator produced no samples; the corpus may be too small for this request")

        progress(1.0, "Saving test set")
        return self._save(req, fingerprint, result)

    def job_status(self, job_id: JobId) -> JobStatus:
        return self._jobs.status(job_id)

    def active_job(self) -> JobStatus | None:
        return self._jobs.active()

    def list_testsets(self) -> list["TestsetListItem"]:
        current = corpus_fingerprint(self._data_dir).digest  # once per call
        return [TestsetListItem(summary=s, is_stale=s.corpus_digest != current) for s in self._datasets.list()]

    def get_testset(self, testset_id: str) -> Testset:
        return self._datasets.get(testset_id)

    # ---------- internals ----------

    def _validate(self, req: GenerateTestsetRequest) -> bool:
        """Raise InvalidRequest / TestsetNotFound; return whether a saved graph will be reused."""
        if not isinstance(req.name, str) or not TESTSET_NAME_PATTERN.fullmatch(req.name):
            raise InvalidRequest("Name must be 1–64 characters: letters, digits, '-' or '_'.")
        if isinstance(req.size, bool) or not isinstance(req.size, int) or not MIN_SIZE <= req.size <= MAX_SIZE:
            raise InvalidRequest(f"Size must be a whole number from {MIN_SIZE} to {MAX_SIZE}.")
        if req.reuse_kg_from is None:
            return False
        source = self._datasets.get(req.reuse_kg_from)  # raises TestsetNotFound
        if not source.manifest.has_knowledge_graph:
            raise InvalidRequest(f"Test set {req.reuse_kg_from} has no saved knowledge graph to reuse.")
        return True

    def _load_corpus(self) -> tuple[CorpusFingerprint, list[SourceDocument]]:
        """The current corpus, loaded once per corpus version (keyed by its fingerprint)."""
        fingerprint = corpus_fingerprint(self._data_dir)
        if self._corpus_cache is None or self._corpus_cache[0] != fingerprint.digest:
            docs = [SourceDocument(d.page_content, dict(d.metadata)) for d in load_pdfs(self._data_dir)]
            self._corpus_cache = (fingerprint.digest, docs)
        return fingerprint, list(self._corpus_cache[1])

    def _save(self, req: GenerateTestsetRequest, fingerprint: CorpusFingerprint, result) -> TestsetSummary:
        digest = samples_sha256(result.samples)
        # The repository rejects a (name, version) that's already taken, even by another process
        # (CLI vs. app). On a clash, re-read the next free version and try again.
        for _ in range(SAVE_ATTEMPTS):
            version = self._datasets.next_version(req.name)
            # parent: the newest version actually saved right now. A version another writer has claimed
            # but not finished saving isn't linked (its save could still fail and leave a dangling parent).
            previous = self._datasets.latest(req.name)
            created_at = self._clock()
            manifest = TestsetManifest(
                id=new_testset_id(created_at, req.name, version, digest),
                name=req.name,
                version=version,
                parent_id=previous.id if previous else None,
                kg_source_id=req.reuse_kg_from,
                created_at=created_at,
                size=len(result.samples),
                generator=result.generator,
                corpus=fingerprint,
                samples_sha256=digest,
                usage=result.usage,
                has_knowledge_graph=True,
                requested_size=req.size,
            )
            try:
                self._datasets.save(Testset(manifest=manifest, samples=result.samples), result.knowledge_graph)
            except TestsetAlreadyExists:
                continue
            return TestsetSummary.from_manifest(manifest)
        raise GenerationError(f"Could not save test set '{req.name}': its version kept being taken by another writer")

    def _calibration_factors(self, fingerprint: CorpusFingerprint, corpus_tokens: int) -> tuple[float, float] | None:
        """(input, output) multipliers from the newest past test set with the same corpus and model and recorded usage."""
        model = self._generator.info.llm_model
        for summary in self._datasets.list():  # newest first
            if summary.corpus_digest != fingerprint.digest:
                continue
            past = self._datasets.get(summary.id).manifest
            if past.generator.llm_model != model or past.usage is None:
                continue
            base_in, base_out = _base_estimate(
                corpus_tokens, past.requested_size or past.size, reuses_kg=past.kg_source_id is not None
            )
            if base_in <= 0 or base_out <= 0:
                continue
            return past.usage.input_tokens / base_in, past.usage.output_tokens / base_out
        return None

    def _cost(self, tokens_in: float, tokens_out: float) -> float | None:
        if self._price_in is None or self._price_out is None:
            return None
        return round(tokens_in / 1e6 * self._price_in + tokens_out / 1e6 * self._price_out, 4)


def _validated_price(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number of USD per 1M tokens, got {value!r}")
    return float(value)


def _corpus_tokens(docs: list[SourceDocument]) -> int:
    return sum(len(d.text) for d in docs) // CHARS_PER_TOKEN


def _base_estimate(corpus_tokens: int, size: int, reuses_kg: bool) -> tuple[float, float]:
    kg_in = 0.0 if reuses_kg else KG_INPUT_PER_CORPUS_TOKEN * corpus_tokens
    kg_out = 0.0 if reuses_kg else KG_OUTPUT_PER_CORPUS_TOKEN * corpus_tokens
    return kg_in + size * SAMPLE_INPUT_TOKENS, kg_out + size * SAMPLE_OUTPUT_TOKENS


class _MonotonicProgress:
    """Forwards progress to a reporter, never letting the reported fraction go backwards."""

    def __init__(self, reporter: ProgressReporter | None) -> None:
        self._reporter = reporter
        self._last = 0.0

    def __call__(self, progress: float | None, message: str) -> None:
        if progress is not None:
            progress = max(self._last, progress)
            self._last = progress
        if self._reporter is not None:
            self._reporter(progress if progress is not None else self._last, message)

    def stage_mapper(self, reuses_kg: bool) -> Callable[[GenerationProgress], None]:
        """Map the generator's per-stage progress onto 0–1 overall (graph 0–0.7, samples 0.7–1)."""

        def on_progress(p: GenerationProgress) -> None:
            if p.stage == "building_knowledge_graph":
                start, span = 0.0, KG_SHARE
            else:
                start, span = (0.0, 1.0) if reuses_kg else (KG_SHARE, 1.0 - KG_SHARE)
            self(start + span * p.fraction if p.fraction is not None else None, p.message)

        return on_progress

