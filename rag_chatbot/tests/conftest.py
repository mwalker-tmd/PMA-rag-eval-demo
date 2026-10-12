from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
from rag_chatbot.config import Settings
from rag_chatbot.container import build_container
from rag_chatbot.domain.models import ReasoningEffort

EMBED_DIM = 32

PAGES = {
    "quality.pdf": [
        "Quality is a habit, not an act. Teams that review code early find defects cheaply. " * 12,
        "Testing pyramids put many fast unit tests under fewer slow end-to-end tests. " * 12,
    ],
    "leadership.pdf": [
        "Engineering leaders set direction, remove blockers, and grow people on their teams. " * 12,
    ],
}


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(36, 36, 576, 806), text, fontsize=9)
    doc.save(str(path))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    for name, pages in PAGES.items():
        _write_pdf(d / name, pages)
    return d


@pytest.fixture
def settings(tmp_path: Path, data_dir: Path) -> Settings:
    return Settings(
        openai_api_key=None,
        cohere_api_key=None,
        llm_provider="fake",
        llm_model="fake-model",
        llm_reasoning_efforts=None,
        default_reasoning_effort=ReasoningEffort.MEDIUM,
        embedding_model="fake-embeddings",
        data_dir=data_dir,
        index_dir=tmp_path / ".index",
        datasets_dir=tmp_path / ".datasets",
        sdg_provider="fake",
    )


@pytest.fixture
def fake_llm() -> FakeLLMAdapter:
    return FakeLLMAdapter()


@pytest.fixture
def container(settings, fake_llm):
    return build_container(
        settings,
        llm=fake_llm,
        embeddings=DeterministicFakeEmbedding(size=EMBED_DIM),
        client=QdrantClient(":memory:"),
        embedding_dim=EMBED_DIM,
    )
