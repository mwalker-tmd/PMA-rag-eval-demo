"""File-backed DatasetRepository.

Layout, one folder per test set:

    <root>/<id>/manifest.json
    <root>/<id>/samples.jsonl
    <root>/<id>/knowledge_graph.json   (optional; opaque bytes from the generator)

Saves are atomic: everything is written to `<root>/.tmp-<uuid>/` and the folder is
renamed into place only once complete. Leftover `.tmp-*` folders from interrupted
saves are removed when the repository is created.

Each (name, version) can be saved once, even by separate processes (e.g. the CLI and the
app): a save first claims `<root>/.versions/<name>@<version>` with an exclusive create, which
only one writer can win. A failed save releases its claim; a crash leaves the claim behind,
and `next_version` counts it, so a crash costs a version number (a gap) rather than blocking
the name.
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
VERSIONS_DIR = ".versions"
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")  # ids are folder names; always .fullmatch() (match + $ allows "\n")


class FileDatasetRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / VERSIONS_DIR).mkdir(exist_ok=True)
        for leftover in self.root.glob(f"{TMP_PREFIX}*"):
            shutil.rmtree(leftover, ignore_errors=True)

    def save(self, testset: Testset, knowledge_graph: bytes | None = None) -> None:
        validate_for_save(testset, knowledge_graph)
        testset_id = testset.manifest.id
        if not _SAFE_ID.fullmatch(testset_id):
            raise ValueError(f"Invalid test set id {testset_id!r}")
        target = self.root / testset_id
        if target.exists():
            raise TestsetAlreadyExists(testset_id)
        claim = self._claim_version(testset.manifest)  # raises TestsetAlreadyExists if (name, version) is taken

        tmp = self.root / f"{TMP_PREFIX}{uuid.uuid4().hex}"
        try:
            tmp.mkdir()
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
            claim.unlink(missing_ok=True)  # this version wasn't saved after all: release it
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
        saved = [s.version for s in self.list() if s.name == name]
        return max(saved + self._claimed_versions(name), default=0) + 1

    def latest(self, name: str) -> TestsetSummary | None:
        versions = [s for s in self.list() if s.name == name]
        return max(versions, key=lambda s: s.version, default=None)

    def load_knowledge_graph(self, testset_id: str) -> bytes | None:
        path = self._folder(testset_id) / KNOWLEDGE_GRAPH
        return path.read_bytes() if path.exists() else None

    # --- internals ---

    def _claim_version(self, manifest: TestsetManifest) -> Path:
        if any(s.name == manifest.name and s.version == manifest.version for s in self.list()):
            raise TestsetAlreadyExists(f"{manifest.name} v{manifest.version}")  # saved before claims existed
        claim = self.root / VERSIONS_DIR / f"{manifest.name}@{manifest.version}"
        try:
            with claim.open("x", encoding="utf-8") as f:  # exclusive create: exactly one writer wins
                f.write(manifest.id)
        except FileExistsError:
            raise TestsetAlreadyExists(f"{manifest.name} v{manifest.version}") from None
        return claim

    def _claimed_versions(self, name: str) -> list[int]:
        versions = []
        for claim in (self.root / VERSIONS_DIR).glob(f"{name}@*"):
            suffix = claim.name.rpartition("@")[2]
            if suffix.isdigit():
                versions.append(int(suffix))
        return versions

    def _folder(self, testset_id: str) -> Path:
        folder = self.root / testset_id
        if not _SAFE_ID.fullmatch(testset_id) or not (folder / MANIFEST).exists():
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
