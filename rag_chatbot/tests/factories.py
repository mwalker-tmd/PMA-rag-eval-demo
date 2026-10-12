"""Shared builders for test data."""

from datetime import UTC, datetime

from rag_chatbot.domain.datasets import (
    CorpusFingerprint,
    GeneratorInfo,
    TestSample,
    Testset,
    TestsetManifest,
    new_testset_id,
    samples_sha256,
)
from rag_chatbot.domain.models import TokenUsage

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def make_testset(
    name: str = "core-corpus",
    version: int = 1,
    *,
    created_at: datetime = T0,
    n: int = 3,
    parent_id: str | None = None,
    with_kg: bool = False,
    tag: str = "",
) -> Testset:
    samples = [
        TestSample(
            user_input=f"Question {i}{tag}?",
            reference=f"Answer {i}{tag}.",
            reference_contexts=[f"context {i}a", f"context {i}b"],
            synthesizer_name="single_hop_specifc_query_synthesizer",
        )
        for i in range(n)
    ]
    digest = samples_sha256(samples)
    manifest = TestsetManifest(
        id=new_testset_id(created_at, name, version, digest),
        name=name,
        version=version,
        parent_id=parent_id,
        kg_source_id=None,
        created_at=created_at,
        size=n,
        generator=GeneratorInfo("gpt-4.1", "text-embedding-3-small", None, "0.2.10"),
        corpus=CorpusFingerprint({"a.pdf": "1" * 64, "b.pdf": "2" * 64}),
        samples_sha256=digest,
        usage=TokenUsage(input_tokens=1000, output_tokens=200, total_tokens=1200),
        has_knowledge_graph=with_kg,
    )
    return Testset(manifest=manifest, samples=samples)
