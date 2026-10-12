"""OpenAI implementation of LLMPort, using the Responses API directly (no LangChain at this boundary)."""

from __future__ import annotations

import re
import time

import openai
from openai import OpenAI

from rag_chatbot.domain.errors import LLMAuthError, LLMBadRequestError, LLMError, LLMRateLimitError
from rag_chatbot.domain.models import LLMRequest, LLMResponse, ReasoningEffort, TokenUsage

E = ReasoningEffort

# First matching pattern wins. Models not listed fall back to DEFAULT_EFFORTS;
# LLM_REASONING_EFFORTS in .env overrides both.
_EFFORTS_BY_MODEL: list[tuple[re.Pattern[str], list[ReasoningEffort]]] = [
    (re.compile(r"^gpt-4"), []),  # gpt-4o / gpt-4.1 family: not reasoning models
    (re.compile(r"^gpt-5(-|$)"), [E.MINIMAL, E.LOW, E.MEDIUM, E.HIGH]),
    (re.compile(r"^o\d"), [E.LOW, E.MEDIUM, E.HIGH]),
]
DEFAULT_EFFORTS = [E.LOW, E.MEDIUM, E.HIGH]


def efforts_for_model(model: str) -> list[ReasoningEffort]:
    for pattern, efforts in _EFFORTS_BY_MODEL:
        if pattern.search(model):
            return list(efforts)
    return list(DEFAULT_EFFORTS)


class OpenAILLMAdapter:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        reasoning_efforts: list[ReasoningEffort] | None = None,
        max_retries: int = 3,
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        # The SDK retries rate limits and 5xx errors with exponential backoff.
        self._client = client or OpenAI(api_key=api_key, max_retries=max_retries)
        self._efforts = reasoning_efforts if reasoning_efforts is not None else efforts_for_model(model)

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def supported_reasoning_efforts(self) -> list[ReasoningEffort]:
        return list(self._efforts)

    def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "input": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        # Only send reasoning when the model supports it; unsupported params are a 400.
        if request.reasoning_effort is not None and self._efforts:
            kwargs["reasoning"] = {"effort": ReasoningEffort(request.reasoning_effort).value}
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens

        start = time.perf_counter()
        try:
            response = self._client.responses.create(**kwargs)
        except openai.AuthenticationError as e:
            raise LLMAuthError(f"OpenAI rejected the API key: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit hit: {e}") from e
        except (openai.BadRequestError, openai.NotFoundError) as e:
            raise LLMBadRequestError(f"OpenAI rejected the request for model '{self._model}': {e}") from e
        except openai.OpenAIError as e:
            raise LLMError(f"OpenAI request failed: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response.output_text,
            model=getattr(response, "model", None) or self._model,
            latency_ms=latency_ms,
            usage=_to_usage(getattr(response, "usage", None)),
            raw_id=getattr(response, "id", None),
        )


def _to_usage(usage) -> TokenUsage | None:
    if usage is None:
        return None
    details = getattr(usage, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
    )
