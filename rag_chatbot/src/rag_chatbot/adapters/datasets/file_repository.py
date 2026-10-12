"""File-backed DatasetRepository.

Layout, one folder per test set:

    <root>/<id>/manifest.json
    <root>/<id>/samples.jsonl
    <root>/<id>/knowledge_graph.json   (optional; opaque bytes from the generator)

Saves are atomic: everything is written to `<root>/.tmp-<uuid>/` and the folder is
renamed into place only once complete. Leftover `.tmp-*` folders from interrupted
saves are removed when the repository is created.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from rag_chatbot.adapters.datasets.serialization import (
    manifest_from_dict,
    manifest_to_dict,
    sample_from_dict,
    sample_to_dict,
)
from rag_chatbot.domain.datasets import (
    Testset,
    TestsetAlreadyExists,
    TestsetManifest,
    TestsetNotFound,
    TestsetSummary,
    sort_newest_first,
    validate_for_save,
)

MANIFEST = "manifest.json"
SAMPLES = "samples.jsonl"
KNOWLEDGE_GRAPH = "knowledge_graph.json"
TMP_PREFIX = ".tmp-"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")  # ids are used as folder names; reject anything path-like


class FileDatasetRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for leftover in self.root.glob(f"{TMP_PREFIX}*"):
            shutil.rmtree(leftover, ignore_errors=True)

    def save(self, testset: Testset, knowledge_graph: bytes | None = None) -> None:
        validate_for_save(testset, knowledge_graph)
        testset_id = testset.manifest.id
        if not _SAFE_ID.match(testset_id):
            raise ValueError(f"Invalid test set id {testset_id!r}")
        target = self.root / testset_id
        if target.exists():
            raise TestsetAlreadyExists(testset_id)

        tmp = self.root / f"{TMP_PREFIX}{uuid.uuid4().hex}"
        tmp.mkdir()
        try:
            self._write_samples(tmp, testset)
            if knowledge_graph is not None:
                (tmp / KNOWLEDGE_GRAPH).write_bytes(knowledge_graph)
            self._write_manifest(tmp, testset.manifest)  # last: a folder with a manifest is complete
            try:
                os.rename(tmp, target)
            except OSError as e:
                if target.exists():  # another save took this id since the check above
                    raise TestsetAlreadyExists(testset_id) from e
                raise  # a real filesystem failure (permissions, I/O, …): don't disguise it
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def get(self, testset_id: str) -> Testset:
        folder = self._folder(testset_id)
        with (folder / SAMPLES).open(encoding="utf-8") as f:
            samples = [sample_from_dict(json.loads(line)) for line in f if line.strip()]
        return Testset(manifest=self._read_manifest(folder), samples=samples)

    def list(self) -> list[TestsetSummary]:
        summaries = [
            TestsetSummary.from_manifest(self._read_manifest(folder))
            for folder in self.root.iterdir()
            if folder.is_dir() and not folder.name.startswith(".") and (folder / MANIFEST).exists()
        ]
        return sort_newest_first(summaries)

    def next_version(self, name: str) -> int:
        latest = self.latest(name)
        return latest.version + 1 if latest else 1

    def latest(self, name: str) -> TestsetSummary | None:
        versions = [s for s in self.list() if s.name == name]
        return max(versions, key=lambda s: s.version, default=None)

    def load_knowledge_graph(self, testset_id: str) -> bytes | None:
        path = self._folder(testset_id) / KNOWLEDGE_GRAPH
        return path.read_bytes() if path.exists() else None

    # --- internals ---

    def _folder(self, testset_id: str) -> Path:
        folder = self.root / testset_id
        if not _SAFE_ID.match(testset_id) or not (folder / MANIFEST).exists():
            raise TestsetNotFound(testset_id)
        return folder

    @staticmethod
    def _read_manifest(folder: Path) -> TestsetManifest:
        return manifest_from_dict(json.loads((folder / MANIFEST).read_text(encoding="utf-8")))

    @staticmethod
    def _write_manifest(folder: Path, manifest: TestsetManifest) -> None:
        (folder / MANIFEST).write_text(
            json.dumps(manifest_to_dict(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _write_samples(folder: Path, testset: Testset) -> None:
        with (folder / SAMPLES).open("w", encoding="utf-8") as f:
            for sample in testset.samples:
                f.write(json.dumps(sample_to_dict(sample), ensure_ascii=False) + "\n")
