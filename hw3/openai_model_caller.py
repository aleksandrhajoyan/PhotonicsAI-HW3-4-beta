"""LumenOpenAIModelCaller: a minimal OpenAI Responses API adapter with the ModelCaller shape.

Exposes the same contract as the task6 ModelCaller:

    caller.call(instructions: str, input_text: str) -> str

so it can sit behind BoundedDecisionProvider (hw3/llm_decision_provider.py),
which in turn serves AdaptiveMCPExecutor (hw3/executor_mcp_loop.py):

    client = create_live_openai_client()          # only when a live run is authorized
    caller = LumenOpenAIModelCaller(client)       # gpt-6-luna, effort low, 512 tokens
    provider = BoundedDecisionProvider(caller, max_calls=8, ...)

The client is always injected. Importing this module or constructing the
adapter performs no request, imports no OpenAI SDK and reads no credentials.

Per call:
    1. require non-empty str instructions and input_text;
    2. call client.responses.create exactly once with model, instructions,
       input, reasoning={"effort": ...} and max_output_tokens (no retries; an
       SDK exception propagates unchanged);
    3. record the usage metadata the response actually reports (see ModelUsage),
       also when the response is then rejected;
    4. require response.status to be exactly "completed"; incomplete, failed,
       cancelled, queued, in_progress, missing or unknown status is refused;
    5. require response.output_text to be a non-blank str and return it
       unchanged (no stripping, no parsing of tool decisions).

Live client (create_live_openai_client):
    Builds the official OpenAI client with max_retries=0 and timeout=30.0 s.
    The SDK itself reads OPENAI_API_KEY when, and only when, the factory is
    called; this module never reads, stores, prints or logs credentials.

Limits and caveats:
    - Token counts are copied from the response's usage field when present;
      missing or malformed counts are reported as None, never estimated.
    - Nothing here is a USD spending limit. Cost depends on model, tokens and
      pricing; BoundedDecisionProvider bounds the number of calls.
    - A response cut off by max_output_tokens has status "incomplete" and may
      still carry partial text; it is refused with ModelOutputError, never
      returned. There is no retry and max_output_tokens is never raised.
    - The default model identifier is a configuration choice; nothing here
      (including the SDK's type annotations) verifies that it is available.
    - Raw requests and responses are not persisted; the Executor owns the
      experimental record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 512
MIN_OUTPUT_TOKENS = 16  # the Responses API rejects smaller max_output_tokens
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
LIVE_MAX_RETRIES = 0
LIVE_TIMEOUT_SECONDS = 30.0
COMPLETED_STATUS = "completed"
# ResponseStatus values of the installed openai SDK (types/responses/response_status.py).
KNOWN_RESPONSE_STATUSES = ("completed", "failed", "in_progress", "cancelled", "queued",
                           "incomplete")


class ModelCallerError(Exception):
    """Base class for refusals by LumenOpenAIModelCaller."""


class ModelInputError(ModelCallerError):
    """instructions or input_text is unusable; the API was not called."""


class ModelOutputError(ModelCallerError):
    """The response did not complete or has no usable text output."""


@dataclass(frozen=True)
class ModelUsage:
    """Read-only metadata reported by one response; None where it was not reported."""

    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


def _count(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def extract_usage(response: Any) -> ModelUsage:
    """Usage metadata actually present on a response; nothing is filled in."""
    model = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    return ModelUsage(
        model=model if isinstance(model, str) and model else None,
        input_tokens=_count(usage, "input_tokens"),
        output_tokens=_count(usage, "output_tokens"),
        total_tokens=_count(usage, "total_tokens"),
    )


def create_live_openai_client() -> Any:
    """The official OpenAI client with retries disabled and a 30 s timeout.

    Only for an explicitly authorized live run. The SDK is imported here, not
    at module import, and it resolves credentials from its usual environment
    variable itself; no key is passed, read or logged by this module.
    """
    from openai import OpenAI

    return OpenAI(max_retries=LIVE_MAX_RETRIES, timeout=LIVE_TIMEOUT_SECONDS)


class LumenOpenAIModelCaller:
    """ModelCaller-compatible adapter around an injected OpenAI-compatible client."""

    def __init__(self, client: Any, *, model: str = DEFAULT_MODEL,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS):
        responses = getattr(client, "responses", None)
        if not callable(getattr(responses, "create", None)):
            raise TypeError("client must provide responses.create(...)")
        if not isinstance(model, str) or not model or model != model.strip():
            raise ValueError("model must be a non-empty str without surrounding whitespace")
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
            raise TypeError(
                f"max_output_tokens must be an int, got {type(max_output_tokens).__name__}")
        if max_output_tokens < MIN_OUTPUT_TOKENS:
            raise ValueError(f"max_output_tokens must be >= {MIN_OUTPUT_TOKENS}, "
                             f"got {max_output_tokens}")
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._calls_made = 0
        self._last_usage: ModelUsage | None = None

    def __repr__(self) -> str:
        # Deliberately omits the client, which may hold credentials.
        return (f"LumenOpenAIModelCaller(model={self._model!r}, "
                f"reasoning_effort={self._reasoning_effort!r}, "
                f"max_output_tokens={self._max_output_tokens})")

    # -- read-only settings and usage -----------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_effort(self) -> str:
        return self._reasoning_effort

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    @property
    def calls_made(self) -> int:
        """API requests attempted through this adapter (including failed ones)."""
        return self._calls_made

    @property
    def last_usage(self) -> ModelUsage | None:
        """Metadata of the last response received; None if the last call got no response."""
        return self._last_usage

    # -- ModelCaller ------------------------------------------------------------

    def call(self, instructions: str, input_text: str) -> str:
        for name, value in (("instructions", instructions), ("input_text", input_text)):
            if not isinstance(value, str):
                raise ModelInputError(f"{name} must be a str, got {type(value).__name__}")
            if not value.strip():
                raise ModelInputError(f"{name} must not be empty")

        self._calls_made += 1
        self._last_usage = None
        response = self._client.responses.create(  # exactly once; errors propagate
            model=self._model,
            instructions=instructions,
            input=input_text,
            reasoning={"effort": self._reasoning_effort},
            max_output_tokens=self._max_output_tokens,
        )
        if response is None:
            raise ModelOutputError("response is missing")
        self._last_usage = extract_usage(response)

        status = getattr(response, "status", None)
        if status is None:
            raise ModelOutputError("response has no status")
        if not isinstance(status, str) or status not in KNOWN_RESPONSE_STATUSES:
            raise ModelOutputError("response status is not recognized")
        if status != COMPLETED_STATUS:
            raise ModelOutputError(f"response status is {status!r}, not 'completed'")

        text = getattr(response, "output_text", None)
        if text is None:
            raise ModelOutputError("response has no output_text")
        if not isinstance(text, str):
            raise ModelOutputError(f"output_text is {type(text).__name__}, not str")
        if not text.strip():
            raise ModelOutputError("output_text is empty")
        return text
