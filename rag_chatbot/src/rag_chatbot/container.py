"""Composition root: the only place that chooses concrete adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from rag_chatbot.adapters.datasets.file_repository import FileDatasetRepository
from rag_chatbot.adapters.eval_runs.file_repository import FileEvalRunRepository
from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
from rag_chatbot.adapters.llm.openai_adapter import OpenAILLMAdapter
from rag_chatbot.adapters.sdg.fake_generator import FakeTestsetGenerator
from rag_chatbot.adapters.sdg.ragas_generator import RagasTestsetGenerator
from rag_chatbot.config import Settings, load_settings
from rag_chatbot.ingestion.indexer import IndexManager, Progress
from rag_chatbot.pipelines.base import PipelineSpec
from rag_chatbot.pipelines.registry import PIPELINES
from rag_chatbot.ports.datasets import DatasetRepository
from rag_chatbot.ports.eval_runs import EvalRunRepository
from rag_chatbot.ports.llm import LLMPort
from rag_chatbot.ports.testset_generator import TestsetGeneratorPort
from rag_chatbot.services.chat_service import ChatService
from rag_chatbot.services.jobs import JobRunner
from rag_chatbot.services.sdg_service import SdgService


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


def build_testset_generator(settings: Settings) -> TestsetGeneratorPort:
    if settings.sdg_provider == "ragas":
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set in .env (needed for test set generation)")
        return RagasTestsetGenerator(
            llm_model=settings.sdg_llm_model,
            embedding_model=settings.sdg_embedding_model,
            api_key=settings.openai_api_key,
            reasoning_effort=settings.sdg_reasoning_effort,
        )
    if settings.sdg_provider == "fake":
        return FakeTestsetGenerator()
    raise ConfigError(f"Unknown SDG_PROVIDER '{settings.sdg_provider}' (expected: ragas, fake)")


@dataclass
class Container:
    settings: Settings
    llm: LLMPort
    index: IndexManager
    datasets: DatasetRepository
    testset_generator: TestsetGeneratorPort
    eval_runs: EvalRunRepository
    specs: dict[str, PipelineSpec] = field(default_factory=lambda: dict(PIPELINES))
    # One runner per container (the app caches one container per process), so jobs are shared and run one at a time.
    jobs: JobRunner = field(default_factory=JobRunner)
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

    @cached_property
    def sdg(self) -> SdgService:
        return SdgService(
            datasets=self.datasets,
            generator=self.testset_generator,
            jobs=self.jobs,
            data_dir=self.settings.data_dir,
            price_input_per_mtok=self.settings.sdg_price_input_per_mtok,
            price_output_per_mtok=self.settings.sdg_price_output_per_mtok,
        )

    def rebuild_index(self, progress: Progress = lambda _msg: None) -> dict[str, int]:
        # Compiled graphs hold references to the same collections/docstore paths, so they stay valid.
        return self.index.build(progress)


def build_sdg_service(
    settings: Settings | None = None,
    *,
    datasets: DatasetRepository | None = None,
    generator: TestsetGeneratorPort | None = None,
    jobs: JobRunner | None = None,
) -> SdgService:
    """SDG service without the rest of the app. Never opens the Qdrant index, so the CLI can run alongside the app."""
    settings = settings or load_settings()
    return SdgService(
        datasets=datasets or FileDatasetRepository(settings.datasets_dir),
        generator=generator or build_testset_generator(settings),
        jobs=jobs or JobRunner(),
        data_dir=settings.data_dir,
        price_input_per_mtok=settings.sdg_price_input_per_mtok,
        price_output_per_mtok=settings.sdg_price_output_per_mtok,
    )


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
    datasets: DatasetRepository | None = None,
    testset_generator: TestsetGeneratorPort | None = None,
    eval_runs: EvalRunRepository | None = None,
) -> Container:
    settings = settings or load_settings()
    # Build config-checked adapters first: a config error then fails fast, before the embedded Qdrant lock is taken.
    llm = llm or build_llm(settings)
    testset_generator = testset_generator or build_testset_generator(settings)
    index = build_index_manager(settings, embeddings=embeddings, client=client, embedding_dim=embedding_dim)
    datasets = datasets or FileDatasetRepository(settings.datasets_dir)
    return Container(
        settings=settings,
        llm=llm,
        index=index,
        datasets=datasets,
        testset_generator=testset_generator,
        eval_runs=eval_runs or FileEvalRunRepository(settings.evals_dir),
    )
