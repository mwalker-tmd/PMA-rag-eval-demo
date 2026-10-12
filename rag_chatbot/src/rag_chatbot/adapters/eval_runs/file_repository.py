"""File-backed EvalRunRepository.

Layout, one folder per run:

    <root>/<run_id>/manifest.json    rewritten atomically on every update
    <root>/<run_id>/samples.jsonl    written once, by complete() or fail()

Every file is written to a temporary file in the same folder and moved into place with
`os.replace`, so a reader (or a crash) only ever sees the previous or the next complete
version. On complete/fail the samples are written first and the manifest last: the manifest's
status is the commit point. When the repository is created it cleans up after interrupted writes
(under the write lock, so it never touches another writer's in-flight files): leftover temporary
files, and run folders that never got a manifest because their create() was interrupted.

Writes are serialized for every instance and process that uses the same folder: an in-process
lock shared by all instances on that folder, plus an exclusive file lock on `<root>/.lock`. The
"is this run already final?" check happens while both are held, so two writers can never both
finish a run. Reads take no lock; they rely on the atomic replaces and only expose a run's
samples once its manifest is final.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterator

import portalocker

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
LOCK_FILE = ".lock"
LOCK_TIMEOUT_SECONDS = 30

# One in-process lock per store folder, shared by every repository instance pointed at it.
_ROOT_LOCKS: dict[Path, threading.Lock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


def _root_lock(root: Path) -> threading.Lock:
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(root.resolve(), threading.Lock())


class FileEvalRunRepository:
    def __init__(self, root: Path, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = _root_lock(self.root)
        self._clock = clock
        with self._writing():  # wait for in-flight writes; never delete another writer's temp files
            for leftover in self.root.glob(f"*/.*{TMP_MARKER}*"):
                leftover.unlink(missing_ok=True)
            for folder in self.root.iterdir():
                if _is_abandoned(folder):
                    shutil.rmtree(folder, ignore_errors=True)

    def create(self, manifest: EvalRunManifest) -> None:
        validate_new_run(manifest)
        folder = self.root / manifest.run_id
        with self._writing():
            if _is_abandoned(folder):  # an interrupted create left it without a manifest: reclaim it
                shutil.rmtree(folder, ignore_errors=True)
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
        self._change(
            run_id, lambda m: replace(m, progress=progress, message=message), validate=lambda: validate_progress(progress)
        )

    def complete(
        self,
        run_id: str,
        samples: list[SampleResult],
        aggregate: dict[str, float | None],
        judge_usage: TokenUsage | None,
        answer_usage: TokenUsage | None,
    ) -> None:
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
            validate=lambda: validate_results(samples, aggregate),
        )

    def fail(self, run_id: str, error: str, samples: list[SampleResult] | None = None) -> None:
        self._change(
            run_id,
            lambda m: replace(m, status=RunStatus.FAILED, finished_at=self._clock(), error=error),
            samples=samples if samples is not None else [],  # none given: commit an empty set, not leftovers
            validate=lambda: validate_results(samples or [], {}),
        )

    def get(self, run_id: str) -> EvalRun:
        folder = self._folder(run_id)
        manifest = self._read_manifest(folder)
        samples: list[SampleResult] = []
        # Samples are written before the manifest commits the run, so only trust them once it's final;
        # until then they may belong to a complete()/fail() that's still in progress or crashed.
        samples_path = folder / SAMPLES
        if manifest.is_final and samples_path.exists():
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
        summaries = [
            EvalRunSummary.from_manifest(self._read_manifest(folder))
            for folder in self.root.iterdir()
            if folder.is_dir() and not folder.name.startswith(".") and (folder / MANIFEST).exists()
        ]
        return sort_newest_first(
            [s for s in summaries if matches(s, testset_id=testset_id, pipeline=pipeline, batch_id=batch_id, status=status)]
        )

    # --- internals ---

    @contextmanager
    def _writing(self) -> Iterator[None]:
        """Exclusive write access to this store, across instances (shared lock) and processes (file lock)."""
        with self._lock, portalocker.Lock(self.root / LOCK_FILE, mode="a", timeout=LOCK_TIMEOUT_SECONDS):
            yield

    def _change(self, run_id: str, change, samples: list[SampleResult] | None = None, validate=lambda: None) -> None:
        """Apply a write. Existence and finality are checked first (under the write lock), then the input,
        so an unknown or finished run always reports RunNotFound / RunFinalized, whatever the input."""
        with self._writing():
            folder = self._folder(run_id)
            manifest = self._read_manifest(folder)
            if manifest.is_final:
                raise RunFinalized(f"{run_id} is {manifest.status.value}")
            validate()
            if samples:
                lines = "".join(json.dumps(sample_to_dict(s), ensure_ascii=False) + "\n" for s in samples)
                _atomic_write(folder / SAMPLES, lines)
            elif samples is not None:  # finishing with no samples: drop any left by an interrupted complete()/fail()
                (folder / SAMPLES).unlink(missing_ok=True)
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


def _is_abandoned(folder: Path) -> bool:
    """A run folder whose create() was interrupted before its manifest was written."""
    return (
        folder.is_dir()
        and not folder.name.startswith(".")
        and RUN_ID_PATTERN.fullmatch(folder.name) is not None
        and not (folder / MANIFEST).exists()
    )


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}{TMP_MARKER}{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
