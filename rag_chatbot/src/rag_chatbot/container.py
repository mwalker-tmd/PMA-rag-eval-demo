"""Composition root: the only place that chooses concrete adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
from rag_chatbot.adapters.llm.openai_adapter import OpenAILLMAdapter
from rag_chatbot.config import Settings, load_settings
from rag_chatbot.ingestion.indexer import IndexManager, Progress
from rag_chatbot.pipelines.base import PipelineSpec
from rag_chatbot.pipelines.registry import PIPELINES
from rag_chatbot.ports.llm import LLMPort
from rag_chatbot.services.chat_service import ChatService


class ConfigError(Exception):
    pass


def build_llm(settings: Settings) -> LLMPort:
    if settings.llm_provider == "openai":
        if not settings.llm_model:
            raise ConfigError("LLM_MODEL is not set in .env")
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set in .env")
        return OpenAILLMAdapter(
            settings.llm_model,
            api_key=settings.openai_api_key,
            reasoning_efforts=settings.llm_reasoning_efforts,
        )
    if settings.llm_provider == "fake":
        return FakeLLMAdapter(settings.llm_model or "fake-model", reasoning_efforts=settings.llm_reasoning_efforts)
    raise ConfigError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (expected: openai, fake)")


@dataclass
class Container:
    settings: Settings
    llm: LLMPort
    index: IndexManager
    specs: dict[str, PipelineSpec] = field(default_factory=lambda: dict(PIPELINES))
    _service: ChatService | None = None

    def start(self, progress: Progress = lambda _msg: None) -> ChatService:
        """Make sure the index is built, then compile the pipelines. Idempotent."""
        if self._service is None:
            self.index.ensure(progress)
            graphs, unavailable = {}, {}
            for name, spec in self.specs.items():
                if spec.requires_cohere and not self.settings.cohere_api_key:
                    unavailable[name] = "COHERE_API_KEY is not set in .env"
                    continue
                graphs[name] = spec.build_graph(
                    self.index.context(name), self.llm, cohere_api_key=self.settings.cohere_api_key
                )
            self._service = ChatService(graphs, unavailable)
        return self._service

    def rebuild_index(self, progress: Progress = lambda _msg: None) -> dict[str, int]:
        # Compiled graphs hold references to the same collections/docstore paths, so they stay valid.
        return self.index.build(progress)


def build_index_manager(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
    client: QdrantClient | None = None,
    embedding_dim: int | None = None,
) -> IndexManager:
    if embeddings is None:
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set in .env (needed for embeddings)")
        embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    if client is None:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(settings.index_dir / "qdrant"))
    return IndexManager(
        client=client,
        embeddings=embeddings,
        embedding_model=settings.embedding_model,
        data_dir=settings.data_dir,
        index_dir=settings.index_dir,
        specs=PIPELINES,
        embedding_dim=embedding_dim,
    )


def build_container(
    settings: Settings | None = None,
    *,
    llm: LLMPort | None = None,
    embeddings: Embeddings | None = None,
    client: QdrantClient | None = None,
    embedding_dim: int | None = None,
) -> Container:
    settings = settings or load_settings()
    # Build the LLM first: a config error then fails fast, before the embedded Qdrant lock is taken.
    llm = llm or build_llm(settings)
    index = build_index_manager(settings, embeddings=embeddings, client=client, embedding_dim=embedding_dim)
    return Container(settings=settings, llm=llm, index=index)
