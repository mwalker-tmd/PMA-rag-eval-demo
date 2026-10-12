class LLMError(Exception):
    """Base error raised by any LLMPort adapter."""


class LLMAuthError(LLMError):
    """Credentials are missing or rejected."""


class LLMRateLimitError(LLMError):
    """The provider throttled the request (after the adapter's own retries)."""


class LLMBadRequestError(LLMError):
    """The provider rejected the request, e.g. an unknown model or unsupported parameter."""


class ModelIncompatible(Exception):
    """A configured model can't be used for a job, e.g. it rejects the temperature Ragas sends.

    Shared by the test set generator and the evaluator.
    """
