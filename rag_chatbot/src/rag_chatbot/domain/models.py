"""Domain types. No third-party imports: adapters translate to and from these."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ReasoningEffort(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: list[ChatMessage]
    reasoning_effort: ReasoningEffort | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    usage: TokenUsage | None = None
    raw_id: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page: int | None = None


@dataclass
class PipelineResult:
    pipeline: str
    answer: str | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float | None = None
    llm_response: LLMResponse | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
