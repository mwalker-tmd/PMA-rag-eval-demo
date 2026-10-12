"""Domain types for synthetic test sets. No third-party imports: adapters translate to and from these."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from rag_chatbot.domain.models import TokenUsage


class TestsetNotFound(Exception):
    __test__ = False


class TestsetAlreadyExists(Exception):
    __test__ = False


@dataclass(frozen=True)
class TestSample:
    __test__ = False  # stop pytest from collecting this as a test class

    user_input: str
    reference: str
    reference_contexts: list[str]
    synthesizer_name: str


@dataclass(frozen=True)
class CorpusFingerprint:
    files: dict[str, str]  # PDF filename -> sha256

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for name in sorted(self.files):
            h.update(f"{name}\0{self.files[name]}\n".encode())
        return h.hexdigest()


@dataclass(frozen=True)
class GeneratorInfo:
    llm_model: str
    embedding_model: str
    reasoning_effort: str | None
    ragas_version: str


@dataclass(frozen=True)
class TestsetManifest:
    __test__ = False

    id: str
    name: str
    version: int
    parent_id: str | None  # the test set this one was derived from
    kg_source_id: str | None  # the test set whose knowledge graph was reused
    created_at: datetime  # UTC
    size: int  # number of samples
    generator: GeneratorInfo
    corpus: CorpusFingerprint
    samples_sha256: str
    usage: TokenUsage | None
    has_knowledge_graph: bool


@dataclass(frozen=True)
class Testset:
    __test__ = False

    manifest: TestsetManifest
    samples: list[TestSample]


@dataclass(frozen=True)
class TestsetSummary:
    """Cheap listing entry: everything except the samples."""

    __test__ = False

    id: str
    name: str
    version: int
    created_at: datetime
    size: int
    parent_id: str | None
    corpus_digest: str
    has_knowledge_graph: bool

    @classmethod
    def from_manifest(cls, m: TestsetManifest) -> TestsetSummary:
        return cls(
            id=m.id,
            name=m.name,
            version=m.version,
            created_at=m.created_at,
            size=m.size,
            parent_id=m.parent_id,
            corpus_digest=m.corpus.digest,
            has_knowledge_graph=m.has_knowledge_graph,
        )


def samples_sha256(samples: list[TestSample]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(json.dumps(asdict(s), sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def new_testset_id(created_at: datetime, name: str, version: int, samples_digest: str) -> str:
    """`ts_<yyyymmdd>_<hash8>`, stable for the same inputs.

    Name and version are part of the hash so that two versions with identical samples
    (e.g. from a deterministic generator) still get distinct ids.
    """
    h = hashlib.sha256(f"{name}\n{version}\n{samples_digest}".encode()).hexdigest()
    return f"ts_{created_at:%Y%m%d}_{h[:8]}"


def sort_newest_first(summaries: list[TestsetSummary]) -> list[TestsetSummary]:
    return sorted(summaries, key=lambda s: (s.created_at, s.version, s.id), reverse=True)


def validate_for_save(testset: Testset, knowledge_graph: bytes | None) -> None:
    """Integrity checks every DatasetRepository applies before saving. Raises ValueError."""
    m = testset.manifest
    if m.size != len(testset.samples):
        raise ValueError(f"manifest.size={m.size} but {len(testset.samples)} samples were given")
    if m.samples_sha256 != samples_sha256(testset.samples):
        raise ValueError("manifest.samples_sha256 does not match the samples")
    if m.has_knowledge_graph != (knowledge_graph is not None):
        raise ValueError("manifest.has_knowledge_graph does not match whether a knowledge graph was given")
    if m.version < 1:
        raise ValueError("manifest.version must be >= 1")
    if m.created_at.tzinfo is None:
        raise ValueError("manifest.created_at must be timezone-aware (UTC)")
