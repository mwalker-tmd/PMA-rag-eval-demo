"""Contract every LLMPort adapter must satisfy. The OpenAI adapter runs only with RUN_LIVE_TESTS=1."""

import os

import pytest

from rag_chatbot.adapters.llm.fake_adapter import FakeLLMAdapter
from rag_chatbot.domain.models import ChatMessage, LLMRequest, LLMResponse
from rag_chatbot.ports.llm import LLMPort


def _fake():
    return FakeLLMAdapter()


def _openai():
    from rag_chatbot.config import load_settings
    from rag_chatbot.container import build_llm

    settings = load_settings()
    if settings.llm_provider != "openai":
        pytest.skip("LLM_PROVIDER is not openai")
    return build_llm(settings)


live = pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API")


@pytest.fixture(params=[pytest.param(_fake, id="fake"), pytest.param(_openai, id="openai", marks=[live, pytest.mark.live])])
def adapter(request) -> LLMPort:
    return request.param()


def test_satisfies_protocol(adapter):
    assert isinstance(adapter, LLMPort)
    assert adapter.provider
    assert adapter.model


def test_generate_returns_text(adapter):
    response = adapter.generate(LLMRequest(messages=[ChatMessage(role="user", content="Reply with the word: ok")]))
    assert isinstance(response, LLMResponse)
    assert response.text.strip()
    assert response.latency_ms >= 0


def test_generate_accepts_each_supported_effort(adapter):
    for effort in adapter.supported_reasoning_efforts():
        response = adapter.generate(
            LLMRequest(messages=[ChatMessage(role="user", content="Reply with the word: ok")], reasoning_effort=effort)
        )
        assert response.text.strip()
