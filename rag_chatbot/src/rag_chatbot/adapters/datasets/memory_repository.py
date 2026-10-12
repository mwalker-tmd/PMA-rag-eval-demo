"""In-memory DatasetRepository, for tests and for exercising services without touching disk."""

from __future__ import annotations

import copy
import threading

from rag_chatbot.domain.datasets import (
    Testset,
    TestsetAlreadyExists,
    TestsetNotFound,
    TestsetSummary,
    sort_newest_first,
    validate_for_save,
)


class InMemoryDatasetRepository:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Testset, bytes | None]] = {}
        self._lock = threading.Lock()

    def save(self, testset: Testset, knowledge_graph: bytes | None = None) -> None:
        validate_for_save(testset, knowledge_graph)
        with self._lock:
            m = testset.manifest
            if m.id in self._items:
                raise TestsetAlreadyExists(m.id)
            if any(t.manifest.name == m.name and t.manifest.version == m.version for t, _ in self._items.values()):
                raise TestsetAlreadyExists(f"{m.name} v{m.version}")
            # Deep copy so later mutation of the caller's lists can't change what's stored.
            self._items[testset.manifest.id] = (copy.deepcopy(testset), knowledge_graph)

    def get(self, testset_id: str) -> Testset:
        return copy.deepcopy(self._entry(testset_id)[0])

    def list(self) -> list[TestsetSummary]:
        with self._lock:
            manifests = [t.manifest for t, _ in self._items.values()]
        return sort_newest_first([TestsetSummary.from_manifest(m) for m in manifests])

    def next_version(self, name: str) -> int:
        latest = self.latest(name)
        return latest.version + 1 if latest else 1

    def latest(self, name: str) -> TestsetSummary | None:
        versions = [s for s in self.list() if s.name == name]
        return max(versions, key=lambda s: s.version, default=None)

    def load_knowledge_graph(self, testset_id: str) -> bytes | None:
        return self._entry(testset_id)[1]

    def _entry(self, testset_id: str) -> tuple[Testset, bytes | None]:
        with self._lock:
            try:
                return self._items[testset_id]
            except KeyError:
                raise TestsetNotFound(testset_id) from None
