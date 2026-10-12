"""Ragas implementation of TestsetGeneratorPort.

Ragas is imported lazily, inside the methods that use it: importing it applies
`nest_asyncio` to the process, which the chat app doesn't need unless SDG actually runs.
"""

from __future__ import annotations

import tempfile
from importlib.metadata import version
from pathlib import Path

import openai
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_chatbot.adapters.langchain_usage import TokenUsageCallback
from rag_chatbot.domain.datasets import (
    GenerationError,
    GenerationProgress,
    GenerationResult,
    GeneratorInfo,
    SourceDocument,
    TestSample,
)
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.domain.models import ReasoningEffort
from rag_chatbot.ports.testset_generator import ProgressCallback

PREFLIGHT_PROMPT = "Reply with the single word OK."
# Ragas splits `testset_size` across query types with ceil(), so it can return more samples than
# asked for, and fewer when a type can't find enough scenarios in a small graph. Extra samples are
# trimmed; a shortfall gets this many extra rounds for the remainder.
MAX_TOP_UP_ROUNDS = 2


class RagasTestsetGenerator:
    def __init__(
        self,
        *,
        llm_model: str,
        embedding_model: str,
        api_key: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._api_key = api_key
        self._effort = reasoning_effort
        self._info = GeneratorInfo(
            llm_model=llm_model,
            embedding_model=embedding_model,
            reasoning_effort=reasoning_effort.value if reasoning_effort else None,
            ragas_version=version("ragas"),
        )

    @property
    def info(self) -> GeneratorInfo:
        return self._info

    def preflight(self) -> None:
        """One tiny call through the same wrapper and temperature path Ragas uses for generation."""
        from langchain_core.prompt_values import StringPromptValue

        try:
            self._ragas_llm().generate_text(StringPromptValue(text=PREFLIGHT_PROMPT))
        except openai.BadRequestError as e:
            raise ModelIncompatible(
                f"SDG_LLM_MODEL '{self._llm_model}' rejected the request Ragas sends "
                f"(Ragas uses a near-zero temperature, which many reasoning models refuse): {e}"
            ) from e
        except Exception as e:
            raise GenerationError(f"Preflight call to '{self._llm_model}' failed: {type(e).__name__}: {e}") from e

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
        report = on_progress or (lambda _p: None)
        usage = TokenUsageCallback()
        try:
            from ragas.testset import TestsetGenerator
            from ragas.testset.synthesizers import default_query_distribution

            llm = self._ragas_llm(usage)
            if knowledge_graph is None:
                report(GenerationProgress("building_knowledge_graph", 0.0, f"Building graph from {len(documents)} documents"))
                kg = self._build_knowledge_graph(documents, llm)
                report(GenerationProgress("building_knowledge_graph", 1.0, "Knowledge graph built"))
            else:
                kg = _load_knowledge_graph(knowledge_graph)

            report(GenerationProgress("generating_samples", 0.0, f"Generating {size} samples"))
            generator = TestsetGenerator(llm=llm, embedding_model=self._ragas_embeddings(), knowledge_graph=kg)
            distribution = default_query_distribution(llm, kg)
            samples: list[TestSample] = []
            for round_ in range(1 + MAX_TOP_UP_ROUNDS):
                missing = size - len(samples)
                if missing <= 0:
                    break
                if round_:
                    report(GenerationProgress("generating_samples", len(samples) / size, f"Topping up {missing} samples"))
                testset = generator.generate(testset_size=missing, query_distribution=distribution)
                new = _dedupe(samples, [_to_test_sample(s) for s in testset.samples])
                if not new:
                    break
                samples += new
            samples = _balanced_trim(samples, size)
            report(GenerationProgress("generating_samples", 1.0, f"Generated {len(samples)} samples"))
            graph_bytes = knowledge_graph if knowledge_graph is not None else _dump_knowledge_graph(kg)
        except (GenerationError, ModelIncompatible):
            raise
        except Exception as e:
            raise GenerationError(f"Ragas test set generation failed: {type(e).__name__}: {e}") from e

        return GenerationResult(samples=samples, knowledge_graph=graph_bytes, generator=self._info, usage=usage.usage())

    # --- internals ---

    def _build_knowledge_graph(self, documents: list[SourceDocument], llm):
        """What TestsetGenerator.generate_with_langchain_docs does, split out so progress can be reported per stage."""
        from ragas.testset.graph import KnowledgeGraph, Node, NodeType
        from ragas.testset.transforms import apply_transforms, default_transforms

        lc_docs = [Document(page_content=d.text, metadata=d.metadata) for d in documents]
        transforms = default_transforms(documents=lc_docs, llm=llm, embedding_model=self._ragas_embeddings())
        kg = KnowledgeGraph(
            nodes=[
                Node(type=NodeType.DOCUMENT, properties={"page_content": d.page_content, "document_metadata": d.metadata})
                for d in lc_docs
            ]
        )
        apply_transforms(kg, transforms)
        return kg

    def _ragas_llm(self, usage: TokenUsageCallback | None = None):
        from ragas.llms import LangchainLLMWrapper

        chat = ChatOpenAI(
            model=self._llm_model,
            api_key=self._api_key,
            reasoning_effort=self._effort.value if self._effort else None,
            callbacks=[usage] if usage else None,
        )
        return LangchainLLMWrapper(chat)

    def _ragas_embeddings(self):
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=self._embedding_model, api_key=self._api_key))


def _to_test_sample(sample) -> TestSample:
    from ragas.dataset_schema import SingleTurnSample

    e = sample.eval_sample
    if not isinstance(e, SingleTurnSample):
        raise GenerationError(f"Unsupported Ragas sample type {type(e).__name__}; only single-turn samples are supported")
    return TestSample(
        user_input=e.user_input or "",
        reference=e.reference or "",
        reference_contexts=list(e.reference_contexts or []),
        synthesizer_name=sample.synthesizer_name,
    )


def _dedupe(existing: list[TestSample], new: list[TestSample]) -> list[TestSample]:
    seen = {s.user_input for s in existing}
    unique = []
    for s in new:
        if s.user_input not in seen:
            seen.add(s.user_input)
            unique.append(s)
    return unique


def _balanced_trim(samples: list[TestSample], size: int) -> list[TestSample]:
    """Keep at most `size` samples, taking them round-robin by synthesizer so the query-type mix stays balanced."""
    if len(samples) <= size:
        return samples
    by_type: dict[str, list[TestSample]] = {}
    for s in samples:
        by_type.setdefault(s.synthesizer_name, []).append(s)
    kept: list[TestSample] = []
    queues = list(by_type.values())
    while len(kept) < size:
        for q in queues:
            if q and len(kept) < size:
                kept.append(q.pop(0))
    keep_ids = {id(s) for s in kept}
    return [s for s in samples if id(s) in keep_ids]  # original order


def _dump_knowledge_graph(kg) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kg.json"
        kg.save(path)
        return path.read_bytes()


def _load_knowledge_graph(data: bytes):
    from ragas.testset.graph import KnowledgeGraph

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kg.json"
        path.write_bytes(data)
        return KnowledgeGraph.load(path)
