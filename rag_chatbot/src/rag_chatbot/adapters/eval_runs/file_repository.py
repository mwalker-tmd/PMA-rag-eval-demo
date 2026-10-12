"""File-backed EvalRunRepository.

Layout, one folder per run:

    <root>/<run_id>/manifest.json    rewritten atomically on every update
    <root>/<run_id>/samples.jsonl    written once, by complete() or fail()

Every file is written to a temporary file in the same folder and moved into place with
`os.replace`, so a reader (or a crash) only ever sees the previous or the next complete
version. On complete/fail the samples are written first and the manifest last: the manifest's
status is the commit point. Leftover temporary files are removed when the repository is created.

Writes are serialized within this process by a lock. Evaluations are run by one process at a
time (the app, or the CLI with the app stopped, since both need the vector index).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from rag_chatbot.adapters.eval_runs.serialization import (
    manifest_from_dict,
    manifest_to_dict,
    sample_from_dict,
    sample_to_dict,
)
from rag_chatbot.domain.evals import (
    RUN_ID_PATTERN,
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

MANIFEST = "manifest.json"
SAMPLES = "samples.jsonl"
TMP_MARKER = ".tmp-"


class FileEvalRunRepository:
    def __init__(self, root: Path, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._clock = clock
        for leftover in self.root.glob(f"*/.*{TMP_MARKER}*"):
            leftover.unlink(missing_ok=True)

    def create(self, manifest: EvalRunManifest) -> None:
        validate_new_run(manifest)
        folder = self.root / manifest.run_id
        with self._lock:
            try:
                folder.mkdir()  # exclusive: an existing folder means the id is taken
            except FileExistsError:
                raise RunAlreadyExists(manifest.run_id) from None
            try:
                self._write_manifest(folder, manifest)
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)  # don't leave a half-created run behind
                raise

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
            folder = self._folder(run_id)
            manifest = self._read_manifest(folder)
            samples_path = folder / SAMPLES
            samples = []
            if samples_path.exists():
                with samples_path.open(encoding="utf-8") as f:
                    samples = [sample_from_dict(json.loads(line)) for line in f if line.strip()]
            return EvalRun(manifest=manifest, samples=samples)

    def list(
        self,
        *,
        testset_id: str | None = None,
        pipeline: str | None = None,
        batch_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[EvalRunSummary]:
        with self._lock:
            summaries = [
                EvalRunSummary.from_manifest(self._read_manifest(folder))
                for folder in self.root.iterdir()
                if folder.is_dir() and not folder.name.startswith(".") and (folder / MANIFEST).exists()
            ]
        return sort_newest_first(
            [s for s in summaries if matches(s, testset_id=testset_id, pipeline=pipeline, batch_id=batch_id, status=status)]
        )

    # --- internals ---

    def _change(self, run_id: str, change, samples: list[SampleResult] | None = None) -> None:
        with self._lock:
            folder = self._folder(run_id)
            manifest = self._read_manifest(folder)
            if manifest.is_final:
                raise RunFinalized(f"{run_id} is {manifest.status.value}")
            if samples is not None:
                lines = "".join(json.dumps(sample_to_dict(s), ensure_ascii=False) + "\n" for s in samples)
                _atomic_write(folder / SAMPLES, lines)
            self._write_manifest(folder, change(manifest))  # last: the manifest's status commits the change

    def _folder(self, run_id: str) -> Path:
        folder = self.root / run_id
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id) or not (folder / MANIFEST).exists():
            raise RunNotFound(run_id)
        return folder

    @staticmethod
    def _read_manifest(folder: Path) -> EvalRunManifest:
        return manifest_from_dict(json.loads((folder / MANIFEST).read_text(encoding="utf-8")))

    @staticmethod
    def _write_manifest(folder: Path, manifest: EvalRunManifest) -> None:
        _atomic_write(folder / MANIFEST, json.dumps(manifest_to_dict(manifest), indent=2, ensure_ascii=False))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}{TMP_MARKER}{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
