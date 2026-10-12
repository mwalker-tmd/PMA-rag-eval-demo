from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph.state import CompiledStateGraph

from rag_chatbot.domain.errors import LLMError
from rag_chatbot.domain.models import PipelineResult, ReasoningEffort, RetrievedChunk
from rag_chatbot.ingestion.loader import page_for


class ChatService:
    """Application service: runs a question through one pipeline, or all of them concurrently.

    Queries are single-turn: the prompt carries no conversation history.
    """

    def __init__(self, graphs: dict[str, CompiledStateGraph], unavailable: dict[str, str] | None = None) -> None:
        self._graphs = graphs
        self._unavailable = unavailable or {}

    @property
    def pipelines(self) -> list[str]:
        return list(self._graphs) + [n for n in self._unavailable if n not in self._graphs]

    def unavailable_reason(self, name: str) -> str | None:
        return self._unavailable.get(name)

    def ask(self, pipeline: str, question: str, effort: ReasoningEffort | None = None) -> PipelineResult:
        if pipeline in self._unavailable:
            return PipelineResult(pipeline=pipeline, error=self._unavailable[pipeline])
        if pipeline not in self._graphs:
            return PipelineResult(pipeline=pipeline, error=f"Unknown pipeline '{pipeline}'")

        config = {"configurable": {"reasoning_effort": effort.value if effort else None}}
        start = time.perf_counter()
        try:
            state = self._graphs[pipeline].invoke({"question": question}, config=config)
        except LLMError as e:
            return PipelineResult(pipeline=pipeline, error=str(e), latency_ms=_ms_since(start))
        except Exception as e:  # retrieval / reranker failures shouldn't take down the UI
            return PipelineResult(pipeline=pipeline, error=f"{type(e).__name__}: {e}", latency_ms=_ms_since(start))

        return PipelineResult(
            pipeline=pipeline,
            answer=state["response"],
            chunks=[
                RetrievedChunk(text=d.page_content, source=d.metadata.get("source", "?"), page=page_for(d.metadata))
                for d in state["context"]
            ],
            latency_ms=_ms_since(start),
            llm_response=state.get("llm_response"),
        )

    def ask_all(
        self, question: str, effort: ReasoningEffort | None = None, pipelines: list[str] | None = None
    ) -> dict[str, PipelineResult]:
        names = pipelines or self.pipelines
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {name: pool.submit(self.ask, name, question, effort) for name in names}
            return {name: f.result() for name, f in futures.items()}


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000
