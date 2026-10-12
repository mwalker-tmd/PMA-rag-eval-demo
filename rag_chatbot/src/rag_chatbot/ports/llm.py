from typing import Protocol, runtime_checkable

from rag_chatbot.domain.models import LLMRequest, LLMResponse, ReasoningEffort


@runtime_checkable
class LLMPort(Protocol):
    """Outbound port for text generation. Adapters raise rag_chatbot.domain.errors.LLMError subclasses."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def supported_reasoning_efforts(self) -> list[ReasoningEffort]:
        """Effort levels the configured model accepts. Empty means reasoning effort is not supported."""
        ...

    def generate(self, request: LLMRequest) -> LLMResponse: ...
