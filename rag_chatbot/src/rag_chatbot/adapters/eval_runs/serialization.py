"""JSON (de)serialization of evaluation run domain types, for adapters that persist them."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from rag_chatbot.domain.evals import EvalRunManifest, RunStatus, SampleResult
from rag_chatbot.domain.models import TokenUsage

SCHEMA_VERSION = 1


def _usage(d: dict | None) -> TokenUsage | None:
    return TokenUsage(**d) if d else None


def manifest_to_dict(m: EvalRunManifest) -> dict[str, Any]:
    d = asdict(m)
    d["status"] = m.status.value
    d["started_at"] = m.started_at.isoformat()
    d["finished_at"] = m.finished_at.isoformat() if m.finished_at else None
    return {"schema_version": SCHEMA_VERSION, **d}


def manifest_from_dict(d: dict[str, Any]) -> EvalRunManifest:
    return EvalRunManifest(
        run_id=d["run_id"],
        batch_id=d["batch_id"],
        testset_id=d["testset_id"],
        pipeline=d["pipeline"],
        pipeline_settings=d.get("pipeline_settings") or {},
        answer_model=d["answer_model"],
        reasoning_effort=d.get("reasoning_effort"),
        judge_model=d["judge_model"],
        judge_reasoning_effort=d.get("judge_reasoning_effort"),
        embedding_model=d["embedding_model"],
        metrics=list(d.get("metrics") or []),
        ragas_version=d["ragas_version"],
        index_manifest_digest=d["index_manifest_digest"],
        status=RunStatus(d["status"]),
        progress=d.get("progress"),
        message=d.get("message", ""),
        started_at=datetime.fromisoformat(d["started_at"]),
        finished_at=datetime.fromisoformat(d["finished_at"]) if d.get("finished_at") else None,
        error=d.get("error"),
        aggregate=dict(d.get("aggregate") or {}),
        answer_usage=_usage(d.get("answer_usage")),
        judge_usage=_usage(d.get("judge_usage")),
    )


def sample_to_dict(s: SampleResult) -> dict[str, Any]:
    return asdict(s)


def sample_from_dict(d: dict[str, Any]) -> SampleResult:
    return SampleResult(
        user_input=d["user_input"],
        reference=d["reference"],
        response=d.get("response"),
        retrieved_contexts=list(d.get("retrieved_contexts") or []),
        scores=dict(d.get("scores") or {}),
        latency_ms=d.get("latency_ms"),
        usage=_usage(d.get("usage")),
        error=d.get("error"),
    )
