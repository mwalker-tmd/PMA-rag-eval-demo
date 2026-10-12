from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from rag_chatbot.adapters.llm.openai_adapter import DEFAULT_EFFORTS, OpenAILLMAdapter, efforts_for_model
from rag_chatbot.domain.errors import LLMAuthError, LLMBadRequestError, LLMError, LLMRateLimitError
from rag_chatbot.domain.models import ChatMessage, LLMRequest, ReasoningEffort

E = ReasoningEffort
MSGS = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]


def _response(text="hello"):
    return SimpleNamespace(
        id="resp_1",
        model="sol-test",
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )


def _adapter(efforts=None):
    client = MagicMock()
    client.responses.create.return_value = _response()
    return OpenAILLMAdapter("sol-test", client=client, reasoning_efforts=efforts), client


def test_request_shape_with_effort():
    adapter, client = _adapter()
    adapter.generate(LLMRequest(messages=MSGS, reasoning_effort=E.HIGH, max_output_tokens=200))
    client.responses.create.assert_called_once_with(
        model="sol-test",
        input=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        reasoning={"effort": "high"},
        max_output_tokens=200,
    )


def test_effort_omitted_when_none():
    adapter, client = _adapter()
    adapter.generate(LLMRequest(messages=MSGS))
    assert "reasoning" not in client.responses.create.call_args.kwargs


def test_effort_omitted_when_model_unsupported():
    adapter, client = _adapter(efforts=[])
    adapter.generate(LLMRequest(messages=MSGS, reasoning_effort=E.HIGH))
    assert "reasoning" not in client.responses.create.call_args.kwargs


def test_response_mapping():
    adapter, _ = _adapter()
    r = adapter.generate(LLMRequest(messages=MSGS))
    assert r.text == "hello"
    assert r.model == "sol-test"
    assert r.raw_id == "resp_1"
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.total_tokens, r.usage.reasoning_tokens) == (10, 5, 15, 3)


def _api_error(cls, status):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_api_error(openai.AuthenticationError, 401), LLMAuthError),
        (_api_error(openai.RateLimitError, 429), LLMRateLimitError),
        (_api_error(openai.BadRequestError, 400), LLMBadRequestError),
        (_api_error(openai.NotFoundError, 404), LLMBadRequestError),
        (_api_error(openai.InternalServerError, 500), LLMError),
        (openai.APIConnectionError(request=httpx.Request("POST", "https://x")), LLMError),
    ],
)
def test_error_translation(exc, expected):
    adapter, client = _adapter()
    client.responses.create.side_effect = exc
    with pytest.raises(expected):
        adapter.generate(LLMRequest(messages=MSGS))


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-4.1-nano", []),
        ("gpt-4o-mini", []),
        ("gpt-5", [E.MINIMAL, E.LOW, E.MEDIUM, E.HIGH]),
        ("gpt-5-mini", [E.MINIMAL, E.LOW, E.MEDIUM, E.HIGH]),
        ("o4-mini", [E.LOW, E.MEDIUM, E.HIGH]),
        ("sol", DEFAULT_EFFORTS),
    ],
)
def test_efforts_for_model(model, expected):
    assert efforts_for_model(model) == expected


def test_config_override_wins():
    adapter, _ = _adapter(efforts=[E.LOW])
    assert adapter.supported_reasoning_efforts() == [E.LOW]
