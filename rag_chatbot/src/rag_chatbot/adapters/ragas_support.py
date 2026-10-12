"""Pieces shared by the Ragas adapters (test set generation and evaluation).

Ragas is imported lazily, inside these functions: importing it applies `nest_asyncio` to the
whole process, which the chat app doesn't need unless SDG or evaluation actually runs.
"""

from __future__ import annotations

from importlib.metadata import version

import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_chatbot.adapters.langchain_usage import TokenUsageCallback
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.domain.models import ReasoningEffort

PREFLIGHT_PROMPT = "Reply with the single word OK."


def ragas_version() -> str:
    return version("ragas")


def ragas_llm(model: str, *, api_key: str | None, effort: ReasoningEffort | None, usage: TokenUsageCallback | None = None):
    """A Ragas LLM wrapper around ChatOpenAI. `usage` is attached to the model itself, so it counts every call."""
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=model,
        api_key=api_key,
        reasoning_effort=effort.value if effort else None,
        callbacks=[usage] if usage else None,
    )
    return LangchainLLMWrapper(chat)


def ragas_embeddings(model: str, *, api_key: str | None):
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=model, api_key=api_key))


def preflight(llm, *, model: str, setting: str, other_error: type[Exception]) -> None:
    """One tiny call through the same wrapper and near-zero temperature Ragas uses for real work.

    A rejected request means the model can't be used with Ragas (e.g. a reasoning model refusing
    the temperature): ModelIncompatible. Anything else is raised as `other_error`.
    """
    from langchain_core.prompt_values import StringPromptValue

    try:
        llm.generate_text(StringPromptValue(text=PREFLIGHT_PROMPT))
    except openai.BadRequestError as e:
        raise ModelIncompatible(
            f"{setting} '{model}' rejected the request Ragas sends "
            f"(Ragas uses a near-zero temperature, which many reasoning models refuse): {e}"
        ) from e
    except Exception as e:
        raise other_error(f"Preflight call to '{model}' failed: {type(e).__name__}: {e}") from e
