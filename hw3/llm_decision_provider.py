"""BoundedDecisionProvider: a call-limited, context-compacting adapter around a ModelCaller.

Sits between AdaptiveMCPExecutor (hw3/executor_mcp_loop.py) and any object
with the task6 ModelCaller shape:

    model_caller.call(instructions: str, input_text: str) -> str

and exposes that same call() itself, so the Executor can use it as its
decision provider:

    provider = BoundedDecisionProvider(model_caller, max_calls=8, max_input_chars=12000,
                                       max_output_chars=2000, max_recent_events=6)
    executor = AdaptiveMCPExecutor(host, provider, goal, max_decisions=8)

The model caller is always injected; this module never builds an LLM client,
never reads credentials and never logs anything. It does not choose tools,
approve sessions or parse decisions: the Executor validates the returned
decision and the MCP server/SessionGate remain the authority on what runs.

Per call:
    1. refuse if max_calls model calls have already been made;
    2. parse the Executor's JSON context strictly (no NaN/Infinity, no duplicate
       keys, exactly the fields research_goal, tools, history,
       latest_measurement, decision_budget);
    3. build a compact context (see compact_context) and refuse, before any
       model call, if instructions + compact context exceed max_input_chars;
    4. reserve one call, then call the model caller exactly once (no retries;
       a failed call still counts and its exception propagates unchanged);
    5. require a str of at most max_output_chars and return it unchanged.

Compact context (deterministic JSON, sorted keys, no whitespace):
    research_goal, tools, latest_measurement and decision_budget are copied
    verbatim and are never removed. Only the most recent max_recent_events
    history events are kept, each reduced to its decision or tool outcome
    (measurement results reduced to frame_id, exposure, source and the main
    brightness/saturation metrics actually present). Older events are counted
    in history_summary. If the context is still too large, the oldest kept
    events are dropped one at a time; if the indispensable part alone is too
    large, the call is refused.

Limits and caveats:
    - Character counts are NOT token counts; a character limit only bounds the
      request size approximately.
    - max_calls bounds calls made through this adapter. It is NOT a guaranteed
      USD spending limit: price per call depends on model, tokens and pricing.
    - A real ModelCaller may retry internally (the OpenAI SDK client retries
      by default) unless that has been separately verified and disabled; such
      retries are invisible here and are not counted.
    - Model-caller exceptions propagate to the Executor, which stops the run.
    - Raw API credentials must never be logged; this module logs nothing and
      never touches credentials.
"""

from __future__ import annotations

import copy
import json
from typing import Any

CONTEXT_FORMAT = "lumen.hw3.compact_executor_context.v1"
REQUIRED_FIELDS = ("research_goal", "tools", "history", "latest_measurement", "decision_budget")
MEASUREMENT_TOOL = "camera_measure"
_MEASUREMENT_FIELDS = ("frame_id", "exposure_us", "source", "synthetic")
_METRIC_FIELDS = ("mean_dn", "mean_fraction_of_full_scale", "saturation_fraction", "max_dn",
                  "min_dn", "spatial_std_dn", "full_scale_dn")


class BoundedProviderError(Exception):
    """Base class for refusals by BoundedDecisionProvider."""


class ProviderInputError(BoundedProviderError):
    """The Executor context is malformed or incomplete; the model was not called."""


class ContextTooLargeError(BoundedProviderError):
    """The indispensable context exceeds max_input_chars; the model was not called."""


class CallLimitError(BoundedProviderError):
    """max_calls model calls have already been made; the model was not called."""


class ProviderOutputError(BoundedProviderError):
    """The model returned something other than a str within max_output_chars."""


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key!r}")
        result[key] = value
    return result


def dumps(data: Any) -> str:
    """The deterministic serialization used for the compact context."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def parse_context(input_text: Any) -> dict:
    """Strictly parse the Executor's JSON context; raise ProviderInputError if unusable."""
    if not isinstance(input_text, str):
        raise ProviderInputError(f"input_text must be a str, got {type(input_text).__name__}")
    try:
        data = json.loads(input_text, parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)
    except ValueError as exc:
        raise ProviderInputError(f"input_text is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProviderInputError("input_text must be a JSON object")
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ProviderInputError(f"context is missing required fields {missing}")
    extra = sorted(set(data) - set(REQUIRED_FIELDS))
    if extra:
        raise ProviderInputError(f"context has unexpected fields {extra}")
    if not isinstance(data["research_goal"], str) or not data["research_goal"].strip():
        raise ProviderInputError("research_goal must be a non-empty string")
    if not isinstance(data["tools"], list) or not data["tools"]:
        raise ProviderInputError("tools must be a non-empty list")
    if not isinstance(data["history"], list) or not all(isinstance(e, dict) for e in data["history"]):
        raise ProviderInputError("history must be a list of event objects")
    if data["latest_measurement"] is not None and not isinstance(data["latest_measurement"], dict):
        raise ProviderInputError("latest_measurement must be an object or null")
    if not isinstance(data["decision_budget"], dict):
        raise ProviderInputError("decision_budget must be an object")
    return data


def _measurement_summary(result: dict) -> dict:
    """Only the fields actually present; nothing is filled in or estimated."""
    summary = {key: result[key] for key in _MEASUREMENT_FIELDS if key in result}
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        summary["metrics"] = {key: metrics[key] for key in _METRIC_FIELDS if key in metrics}
    return summary


def compact_event(event: dict) -> dict:
    """A smaller copy of one Executor event: its decision or its tool outcome."""
    kind = event.get("type")
    keep = ["sequence", "type", "decision_number"]
    if kind == "decision":
        keep += ["valid", "decision", "error"]
    elif kind == "tool_call":
        keep += ["tool", "arguments", "status", "error", "source", "frame_id"]
    else:
        return copy.deepcopy(event)
    compact = {key: copy.deepcopy(event[key]) for key in keep if key in event}
    if kind == "tool_call" and isinstance(event.get("result"), dict):
        result = event["result"]
        compact["result"] = (_measurement_summary(result) if event.get("tool") == MEASUREMENT_TOOL
                             else {k: copy.deepcopy(v) for k, v in result.items()
                                   if k not in ("action_id", "session_id")})
    return compact


def _summary(omitted: list[dict]) -> dict:
    tool_calls: dict[str, int] = {}
    for event in omitted:
        if event.get("type") == "tool_call":
            tool = str(event.get("tool"))
            tool_calls[tool] = tool_calls.get(tool, 0) + 1
    sequences = [e["sequence"] for e in omitted if type(e.get("sequence")) is int]
    frame_ids = [e["frame_id"] for e in omitted
                 if e.get("type") == "tool_call" and type(e.get("frame_id")) is int]
    return {
        "omitted_event_count": len(omitted),
        "omitted_sequence_range": [min(sequences), max(sequences)] if sequences else None,
        "omitted_tool_calls": tool_calls,
        "omitted_successful_measurements": sum(
            1 for e in omitted if e.get("type") == "tool_call"
            and e.get("tool") == MEASUREMENT_TOOL and e.get("status") == "success"),
        "omitted_frame_id_range": [min(frame_ids), max(frame_ids)] if frame_ids else None,
    }


def compact_context(context: dict, max_recent_events: int, char_limit: int) -> str:
    """Deterministic compact JSON of at most char_limit characters, or ContextTooLargeError.

    The input dict is not modified. Older events are dropped first.
    """
    history = context["history"]
    start = max(len(history) - max_recent_events, 0)
    while True:
        compact = {
            "context_format": CONTEXT_FORMAT,
            "research_goal": context["research_goal"],
            "tools": context["tools"],
            "latest_measurement": context["latest_measurement"],
            "decision_budget": context["decision_budget"],
            "history_summary": _summary(history[:start]),
            "recent_history": [compact_event(event) for event in history[start:]],
        }
        try:
            text = dumps(compact)
        except ValueError as exc:
            raise ProviderInputError(f"context cannot be serialized: {exc}") from exc
        if len(text) <= char_limit:
            return text
        if start >= len(history):
            raise ContextTooLargeError(
                f"indispensable context needs {len(text)} characters, limit is {char_limit}")
        start += 1


class BoundedDecisionProvider:
    """Decision provider for AdaptiveMCPExecutor that bounds calls and request/response size."""

    def __init__(self, model_caller: Any, *, max_calls: int, max_input_chars: int,
                 max_output_chars: int, max_recent_events: int):
        if not callable(getattr(model_caller, "call", None)):
            raise TypeError("model_caller must have a call(instructions, input_text) method")
        self._caller = model_caller
        self._max_calls = _positive_int("max_calls", max_calls)
        self._max_input_chars = _positive_int("max_input_chars", max_input_chars)
        self._max_output_chars = _positive_int("max_output_chars", max_output_chars)
        self._max_recent_events = _positive_int("max_recent_events", max_recent_events, minimum=0)
        self._calls_used = 0
        self._last_input_chars: int | None = None
        self._last_output_chars: int | None = None

    # -- read-only usage ----------------------------------------------------

    @property
    def calls_used(self) -> int:
        """Model calls attempted (including failed ones)."""
        return self._calls_used

    @property
    def calls_remaining(self) -> int:
        return self._max_calls - self._calls_used

    @property
    def last_input_chars(self) -> int | None:
        """len(instructions) + len(compact context) of the last attempted model call."""
        return self._last_input_chars

    @property
    def last_output_chars(self) -> int | None:
        """Length of the last str the model returned; None if it returned none."""
        return self._last_output_chars

    # -- decision provider ----------------------------------------------------

    def call(self, instructions: str, input_text: str) -> str:
        if self._calls_used >= self._max_calls:
            raise CallLimitError(f"model call limit of {self._max_calls} reached")
        if not isinstance(instructions, str):
            raise ProviderInputError(f"instructions must be a str, got {type(instructions).__name__}")
        context = parse_context(input_text)
        budget = self._max_input_chars - len(instructions)
        if budget <= 0:
            raise ContextTooLargeError(
                f"instructions alone need {len(instructions)} characters, "
                f"limit is {self._max_input_chars}")
        compact = compact_context(context, self._max_recent_events, budget)

        self._calls_used += 1  # reserved before the call; never refunded
        self._last_input_chars = len(instructions) + len(compact)
        self._last_output_chars = None
        output = self._caller.call(instructions, compact)  # exactly once; errors propagate
        if not isinstance(output, str):
            raise ProviderOutputError(f"model returned {type(output).__name__}, not str")
        self._last_output_chars = len(output)
        if len(output) > self._max_output_chars:
            raise ProviderOutputError(
                f"model output has {len(output)} characters, limit is {self._max_output_chars}")
        return output
