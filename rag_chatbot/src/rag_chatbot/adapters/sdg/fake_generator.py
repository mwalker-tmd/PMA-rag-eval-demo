"""Deterministic TestsetGeneratorPort for tests and for exercising the UI without API spend."""

from __future__ import annotations

import hashlib
import json

from rag_chatbot.domain.datasets import (
    GenerationProgress,
    GenerationResult,
    GeneratorInfo,
    SourceDocument,
    TestSample,
)
from rag_chatbot.domain.models import TokenUsage
from rag_chatbot.ports.testset_generator import ProgressCallback


class FakeTestsetGenerator:
    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self._fail_with = fail_with  # lets tests exercise failure handling
        self.calls: list[dict] = []

    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            llm_model="fake-sdg-model", embedding_model="fake-embeddings", reasoning_effort=None, ragas_version="fake"
        )

    def preflight(self) -> None:
        return None

    def generate(
        self,
        documents: list[SourceDocument],
        size: int,
        *,
        knowledge_graph: bytes | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        if size < 1:
            raise ValueError("size must be >= 1")
        if not documents:
            raise ValueError("at least one document is required")
        self.calls.append({"size": size, "knowledge_graph": knowledge_graph, "documents": len(documents)})
        report = on_progress or (lambda _p: None)

        if knowledge_graph is None:
            report(GenerationProgress("building_knowledge_graph", 0.0, f"Building graph from {len(documents)} documents"))
            knowledge_graph = json.dumps(
                {"fake": True, "documents": [_digest(d.text) for d in documents]}, sort_keys=True
            ).encode()
            report(GenerationProgress("building_knowledge_graph", 1.0, "Knowledge graph built"))

        if self._fail_with is not None:
            raise self._fail_with

        report(GenerationProgress("generating_samples", 0.0, f"Generating {size} samples"))
        samples = [_sample(documents[i % len(documents)], i) for i in range(size)]
        report(GenerationProgress("generating_samples", 1.0, f"Generated {size} samples"))

        words = sum(len(d.text.split()) for d in documents)
        return GenerationResult(
            samples=samples,
            knowledge_graph=knowledge_graph,
            generator=self.info,
            usage=TokenUsage(input_tokens=words, output_tokens=size * 50, total_tokens=words + size * 50),
        )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _sample(doc: SourceDocument, i: int) -> TestSample:
    words = doc.text.split()
    start = (i * 7) % max(len(words), 1)
    snippet = " ".join(words[start : start + 12]) or doc.text[:80] or "(empty document)"
    source = doc.metadata.get("source", "the document")
    return TestSample(
        user_input=f"[{i}] What does {source} say about: {snippet}?",
        reference=f"According to {source}: {snippet}",
        reference_contexts=[snippet],
        synthesizer_name="fake_single_hop",
    )
