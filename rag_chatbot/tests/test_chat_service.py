import threading
import time

from rag_chatbot.domain.errors import LLMRateLimitError
from rag_chatbot.domain.models import ReasoningEffort


def test_start_marks_rerank_unavailable_without_cohere_key(container):
    service = container.start()
    assert service.pipelines == ["baseline", "parent_child", "rerank"]
    assert "COHERE_API_KEY" in service.unavailable_reason("rerank")
    result = service.ask("rerank", "anything")
    assert not result.ok and "COHERE_API_KEY" in result.error


def test_ask_returns_answer_sources_and_pages(container):
    service = container.start()
    result = service.ask("parent_child", "What do engineering leaders do?", ReasoningEffort.LOW)
    assert result.ok, result.error
    assert result.answer.startswith("[fake answer | effort=low")
    assert result.chunks and all(c.source.endswith(".pdf") for c in result.chunks)
    assert all(c.page and c.page >= 1 for c in result.chunks)
    assert result.llm_response.usage.total_tokens > 0
    assert result.latency_ms > 0


def test_ask_all_isolates_failures_and_runs_concurrently(container, fake_llm, monkeypatch):
    service = container.start()
    in_flight, peak, lock = 0, 0, threading.Lock()
    original = fake_llm.generate

    def generate(request):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            time.sleep(0.2)
            if "Testing pyramids" in request.messages[0].content and len(request.messages[0].content) > 2000:
                raise LLMRateLimitError("throttled")
            return original(request)
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(fake_llm, "generate", generate)
    results = service.ask_all("testing pyramids", ReasoningEffort.MEDIUM)

    assert set(results) == {"baseline", "parent_child", "rerank"}
    assert results["baseline"].ok
    assert not results["rerank"].ok  # unavailable, but still reported
    assert peak == 2  # baseline and parent_child generated at the same time
    # parent_child returns long parents, so it hits the injected failure without affecting the others
    assert not results["parent_child"].ok and "throttled" in results["parent_child"].error


def test_unknown_pipeline(container):
    result = container.start().ask("nope", "q")
    assert not result.ok and "Unknown pipeline" in result.error
