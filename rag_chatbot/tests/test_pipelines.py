from typing import Sequence

import pytest
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from rag_chatbot.domain.models import ReasoningEffort
from rag_chatbot.domain.prompts import RAG_PROMPT
from rag_chatbot.pipelines import baseline, parent_child, rerank
from rag_chatbot.pipelines.registry import PIPELINES

EXPECTED_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""


class FakeRerank(BaseDocumentCompressor):
    model: str
    cohere_api_key: str | None = None
    top_n: int = 3  # CohereRerank's default

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Callbacks = None):
        return list(documents)[: self.top_n]


def test_prompt_text():
    assert RAG_PROMPT == EXPECTED_PROMPT


def test_pipeline_settings():
    assert (baseline.CHUNK_SIZE, baseline.CHUNK_OVERLAP, baseline.K) == (50, 0, 3)
    assert (rerank.CHUNK_SIZE, rerank.CHUNK_OVERLAP, rerank.K, rerank.RERANK_MODEL) == (500, 30, 20, "rerank-v3.5")
    assert (
        parent_child.PARENT_CHUNK_SIZE,
        parent_child.PARENT_CHUNK_OVERLAP,
        parent_child.CHILD_CHUNK_SIZE,
        parent_child.CHILD_CHUNK_OVERLAP,
        parent_child.K,
    ) == (1000, 100, 200, 40, 8)
    assert list(PIPELINES) == ["baseline", "rerank", "parent_child"]


@pytest.fixture
def built(container, monkeypatch):
    monkeypatch.setattr(rerank, "CohereRerank", FakeRerank)
    container.index.build()
    return container


@pytest.mark.parametrize(
    "name, node, max_docs",
    [("baseline", "retrieve", 3), ("rerank", "retrieve_adjusted", 3), ("parent_child", "retrieve_custom", 8)],
)
def test_graph_runs_retrieve_then_generate(built, fake_llm, name, node, max_docs):
    graph = PIPELINES[name].build_graph(built.index.context(name), fake_llm, cohere_api_key="test")
    assert list(graph.get_graph().nodes) == ["__start__", node, "generate", "__end__"]

    state = graph.invoke(
        {"question": "What do engineering leaders do?"}, config={"configurable": {"reasoning_effort": "high"}}
    )
    assert 0 < len(state["context"]) <= max_docs
    assert state["response"] == state["llm_response"].text

    request = fake_llm.requests[-1]
    assert request.reasoning_effort is ReasoningEffort.HIGH
    assert [m.role for m in request.messages] == ["user"]
    expected_context = "\n\n".join(d.page_content for d in state["context"])
    assert request.messages[0].content == EXPECTED_PROMPT.format(
        question="What do engineering leaders do?", context=expected_context
    )


def test_baseline_chunks_are_at_most_50_chars(built, fake_llm):
    graph = baseline.build_graph(built.index.context("baseline"), fake_llm)
    state = graph.invoke({"question": "quality"})
    assert all(len(d.page_content) <= 50 for d in state["context"])


def test_parent_child_returns_parents(built, fake_llm):
    graph = parent_child.build_graph(built.index.context("parent_child"), fake_llm)
    state = graph.invoke({"question": "testing pyramids"})
    assert any(len(d.page_content) > parent_child.CHILD_CHUNK_SIZE for d in state["context"])
    assert all(len(d.page_content) <= parent_child.PARENT_CHUNK_SIZE for d in state["context"])
    assert all(d.metadata.get("start_index") is not None for d in state["context"])


def test_effort_none_when_not_configured(built, fake_llm):
    graph = baseline.build_graph(built.index.context("baseline"), fake_llm)
    graph.invoke({"question": "quality"})
    assert fake_llm.requests[-1].reasoning_effort is None
