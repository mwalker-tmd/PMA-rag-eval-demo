"""JSON (de)serialization of test set domain types, for adapters that persist them."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from rag_chatbot.domain.datasets import CorpusFingerprint, GeneratorInfo, TestSample, TestsetManifest
from rag_chatbot.domain.models import TokenUsage

SCHEMA_VERSION = 1


def manifest_to_dict(m: TestsetManifest) -> dict[str, Any]:
    d = asdict(m)
    d["created_at"] = m.created_at.isoformat()
    d["corpus"] = {"files": m.corpus.files, "digest": m.corpus.digest}  # digest is for humans reading the file
    return {"schema_version": SCHEMA_VERSION, **d}


def manifest_from_dict(d: dict[str, Any]) -> TestsetManifest:
    usage = d.get("usage")
    return TestsetManifest(
        id=d["id"],
        name=d["name"],
        version=d["version"],
        parent_id=d.get("parent_id"),
        kg_source_id=d.get("kg_source_id"),
        created_at=datetime.fromisoformat(d["created_at"]),
        size=d["size"],
        generator=GeneratorInfo(**d["generator"]),
        corpus=CorpusFingerprint(files=d["corpus"]["files"]),
        samples_sha256=d["samples_sha256"],
        usage=TokenUsage(**usage) if usage else None,
        has_knowledge_graph=d["has_knowledge_graph"],
    )


def sample_to_dict(s: TestSample) -> dict[str, Any]:
    return asdict(s)


def sample_from_dict(d: dict[str, Any]) -> TestSample:
    return TestSample(
        user_input=d["user_input"],
        reference=d["reference"],
        reference_contexts=list(d["reference_contexts"]),
        synthesizer_name=d["synthesizer_name"],
    )
