"""Shared pieces of the three pipelines: State, the generate node, and the spec interface."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_qdrant import QdrantVectorStore
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired, TypedDict

from rag_chatbot.domain.models import ChatMessage, LLMRequest, LLMResponse, ReasoningEffort
from rag_chatbot.domain.prompts import rag_prompt
from rag_chatbot.ports.llm import LLMPort


class State(TypedDict):
    question: str
    context: List[Document]
    response: str
    # Carries usage/latency from the LLM port out to the UI.
    llm_response: NotRequired[LLMResponse]


_ROLE_BY_TYPE = {"human": "user", "ai": "assistant", "system": "system"}


def to_chat_messages(messages: list[BaseMessage]) -> list[ChatMessage]:
    return [ChatMessage(role=_ROLE_BY_TYPE[m.type], content=m.content) for m in messages]


def make_generate(llm: LLMPort) -> Callable[[State, RunnableConfig], dict]:
    """Build the generate node: fill the RAG prompt with the retrieved context and call the LLM port.

    Reasoning effort travels in config["configurable"]["reasoning_effort"] rather than in State.
    """

    def generate(state: State, config: RunnableConfig) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
        effort = (config.get("configurable") or {}).get("reasoning_effort")
        response = llm.generate(
            LLMRequest(
                messages=to_chat_messages(messages),
                reasoning_effort=ReasoningEffort(effort) if effort else None,
            )
        )
        return {"response": response.text, "llm_response": response}

    return generate


@dataclass(frozen=True)
class IndexContext:
    """Storage a pipeline indexes into and retrieves from."""

    vector_store: QdrantVectorStore
    embeddings: Embeddings
    docstore_dir: Path  # used only by pipelines that need a document store
    # Embedded (local-path) Qdrant isn't documented as thread-safe; searches are serialized.
    search_lock: threading.Lock


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    label: str
    description: str
    settings: dict[str, Any]  # recorded in the index manifest; a change forces a rebuild
    index: Callable[[list[Document], IndexContext], int]  # returns number of vectors written
    build_graph: Callable[..., CompiledStateGraph]  # (ctx, llm, *, cohere_api_key) -> graph
    requires_cohere: bool = False
