"""Deterministic LLMPort for tests and for exercising the UI without API spend."""

from __future__ import annotations

import threading

from rag_chatbot.domain.models import LLMRequest, LLMResponse, ReasoningEffort, TokenUsage


class FakeLLMAdapter:
    def __init__(
        self,
        model: str = "fake-model",
        *,
        reasoning_efforts: list[ReasoningEffort] | None = None,
        reply: str | None = None,
    ) -> None:
        self._model = model
        self._efforts = (
            reasoning_efforts
            if reasoning_efforts is not None
            else [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH]
        )
        self._reply = reply
        self._lock = threading.Lock()
        self.requests: list[LLMRequest] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def supported_reasoning_efforts(self) -> list[ReasoningEffort]:
        return list(self._efforts)

    def generate(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self.requests.append(request)
        prompt = request.messages[-1].content if request.messages else ""
        effort = request.reasoning_effort.value if request.reasoning_effort else "n/a"
        text = self._reply or f"[fake answer | effort={effort} | prompt chars={len(prompt)}]"
        words = len(prompt.split())
        return LLMResponse(
            text=text,
            model=self._model,
            latency_ms=0.0,
            usage=TokenUsage(input_tokens=words, output_tokens=len(text.split()), total_tokens=words + len(text.split())),
        )
