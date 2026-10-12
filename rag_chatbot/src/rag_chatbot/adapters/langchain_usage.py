"""Token accounting for LangChain chat models used inside third-party libraries (e.g. Ragas).

Attach a `TokenUsageCallback` to the chat model itself (`ChatOpenAI(callbacks=[cb])`): model-level
callbacks fire on every call, including calls made from the library's own threads and event
loops, where context-based helpers like `get_openai_callback()` can miss them.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from rag_chatbot.domain.models import TokenUsage


class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._input = self._output = self._reasoning = 0
        self._calls_with_usage = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        counts = _from_llm_output(response) or _from_usage_metadata(response)
        if counts is None:
            return
        with self._lock:
            self._input += counts[0]
            self._output += counts[1]
            self._reasoning += counts[2]
            self._calls_with_usage += 1

    def usage(self) -> TokenUsage | None:
        """Totals so far, or None if no call reported usage."""
        with self._lock:
            if not self._calls_with_usage:
                return None
            return TokenUsage(
                input_tokens=self._input,
                output_tokens=self._output,
                total_tokens=self._input + self._output,
                reasoning_tokens=self._reasoning or None,
            )


def _from_llm_output(response: LLMResult) -> tuple[int, int, int] | None:
    # One aggregate per call (covers n > 1 completions without double counting).
    usage = (response.llm_output or {}).get("token_usage") or {}
    if "prompt_tokens" not in usage:
        return None
    details = usage.get("completion_tokens_details") or {}
    return usage["prompt_tokens"], usage.get("completion_tokens", 0), details.get("reasoning_tokens") or 0


def _from_usage_metadata(response: LLMResult) -> tuple[int, int, int] | None:
    found, total = False, [0, 0, 0]
    for generations in response.generations:
        message = getattr(generations[0], "message", None) if generations else None
        meta = getattr(message, "usage_metadata", None)
        if not meta:
            continue
        found = True
        total[0] += meta.get("input_tokens", 0)
        total[1] += meta.get("output_tokens", 0)
        total[2] += (meta.get("output_token_details") or {}).get("reasoning", 0) or 0
    return tuple(total) if found else None
