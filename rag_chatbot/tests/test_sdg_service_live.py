"""Live end-to-end SDG: real Ragas + OpenAI, real PDFs, file repository. RUN_LIVE_TESTS=1 uv run pytest -m live"""

import os
from dataclasses import replace

import pymupdf
import pytest
from factories import NORTHWIND_LEADERSHIP, NORTHWIND_QUALITY

from rag_chatbot.adapters.datasets.file_repository import FileDatasetRepository
from rag_chatbot.config import load_settings
from rag_chatbot.container import build_sdg_service
from rag_chatbot.services.sdg_service import GenerateTestsetRequest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]


def write_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    assert page.insert_textbox(pymupdf.Rect(36, 36, 576, 806), text, fontsize=9) >= 0, "text didn't fit"
    doc.save(str(path))


def test_end_to_end_generation_saves_a_complete_test_set(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write_pdf(data / "northwind-quality.pdf", NORTHWIND_QUALITY)
    write_pdf(data / "northwind-leadership.pdf", NORTHWIND_LEADERSHIP)
    settings = replace(load_settings(), sdg_provider="ragas", data_dir=data, datasets_dir=tmp_path / ".datasets")
    service = build_sdg_service(settings)

    events = []
    summary = service.generate(GenerateTestsetRequest(name="live-e2e", size=2), lambda p, m: events.append((p, m)))

    repo = FileDatasetRepository(tmp_path / ".datasets")
    saved = repo.get(summary.id)
    assert len(saved.samples) == summary.size == 2
    assert repo.load_knowledge_graph(summary.id)
    usage = saved.manifest.usage
    assert usage is None or usage.total_tokens > 0
    assert [p for p, _ in events] == sorted(p for p, _ in events)
    assert service.list_testsets()[0].is_stale is False
    print(f"\nsaved {summary.id}: {summary.size} samples, usage {usage}")
