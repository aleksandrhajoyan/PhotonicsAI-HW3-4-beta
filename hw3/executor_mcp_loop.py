"""Offline adaptive Executor: decision provider -> validated decision -> real MCP tool call -> feedback.

AdaptiveMCPExecutor drives ONE already approved OfflineApprovedSessionHost:

    executor = AdaptiveMCPExecutor(host, provider, research_goal, max_decisions)
    result = await executor.run()      # single use; always ends with host.shutdown()

Loop (sequential, one call at a time):
    1. open an official MCP SDK Client on host.get_mcp_server() and list its tools;
    2. ask the decision provider for the next action (one provider call = one
       decision of the budget);
    3. validate that decision strictly (nothing is repaired, clipped or coerced);
    4. for call_tool: make exactly that one MCP call and record request and response;
       a measurement becomes the "latest measurement" context;
    5. repeat from 2 with the updated context, until the provider finishes, the
       decision budget is used up, or anything fails;
    6. close the client and call the trusted host.shutdown().

The Executor chooses nothing itself: every tool and every exposure value comes
from the provider's own decision. It never approves a session, never sees an
approval, never touches a camera backend or the runtime, and never executes
code. The MCP server and its SessionGate remain the authority on what may run;
validation here only refuses to forward decisions that are malformed or carry
fields the model must not control.

Decision provider contract (shaped like task6 ModelCaller, so a ModelCaller can
later be injected unchanged):

    provider.call(instructions: str, input_text: str) -> str

input_text is JSON with research_goal, tools (name, description, input_schema),
history (the event log so far), latest_measurement (or null) and
decision_budget {max, used, remaining} where remaining includes this call.
The reply must be exactly one JSON object:

    {"action": "call_tool", "tool": "<camera tool>", "arguments": {...}}
    {"action": "finish", "reason": "<non-empty text>"}

Failure handling: malformed decisions, provider exceptions, MCP tool errors and
uninterpretable MCP responses all stop the run; no tool is retried. Status
"completed" is reported only when the provider finished after at least one
successful camera_measure and the host shut down cleanly.
"""

from __future__ import annotations

import json
import math
from typing import Any

from mcp import Client

from approved_session_host import HostError, OfflineApprovedSessionHost
from session_policy import CAMERA_TOOLS, EXPOSURE_TOOL, MEASUREMENT_TOOL

RESULT_SCHEMA = "lumen.hw3.adaptive_executor_run.v1"
CLIENT_READ_TIMEOUT_SECONDS = 30

ACTION_CALL_TOOL = "call_tool"
ACTION_FINISH = "finish"

# Exact argument names per tool; must agree with the server's listed schemas.
TOOL_ARGUMENTS = {tool: () for tool in CAMERA_TOOLS}
TOOL_ARGUMENTS[EXPOSURE_TOOL] = ("exposure_us",)

# Fields a decision must never carry: authority lives outside the model.
FORBIDDEN_FIELDS = frozenset({
    "session_id", "action_id", "approval", "approved", "approval_token",
    "envelope_sha256", "sha256", "token",
})

STATUS_COMPLETED = "completed"
STATUS_FINISHED_WITHOUT_MEASUREMENT = "finished_without_measurement"
STATUS_DECISION_BUDGET_EXHAUSTED = "decision_budget_exhausted"
STATUS_INVALID_DECISION = "invalid_decision"
STATUS_PROVIDER_ERROR = "provider_error"
STATUS_TOOL_ERROR = "tool_error"
STATUS_INVALID_TOOL_RESPONSE = "invalid_tool_response"
STATUS_HOST_NOT_ACTIVE = "host_not_active"
STATUS_ERROR = "error"
STATUS_CLEANUP_FAILED = "cleanup_failed"

_KNOWN_SOURCES = ("synthetic", "hardware")

DECISION_INSTRUCTIONS = """You are the EXECUTOR decision maker for one approved camera session.

You choose exactly ONE next action per reply, using the research goal, the tool
schemas, every previous decision and tool result, and the latest measurement.

Return ONLY one JSON object, nothing else, in one of these two forms:
{"action": "call_tool", "tool": "<tool name>", "arguments": {<arguments>}}
{"action": "finish", "reason": "<why the run should end>"}

Rules:
- tool must be one of the listed camera tools; arguments must match its schema exactly.
- camera_set_exposure takes exactly {"exposure_us": <finite number>}; other tools take {}.
- Never include session, approval, action or hash fields; they are not yours to set.
- Exposure values outside the approved range are refused, never clipped.
- Results are labelled with their source; synthetic data is not a laboratory measurement.
- Finish only when the goal is met or cannot be met within the remaining budget.
"""


class DecisionError(ValueError):
    """The provider's output is not one valid, authorized decision."""


class ExecutorError(Exception):
    """The Executor was used incorrectly (for example run twice)."""


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key!r}")
        result[key] = value
    return result


def _check_fields(where: str, data: dict, expected: set[str]) -> None:
    forbidden = sorted(FORBIDDEN_FIELDS & set(data))
    if forbidden:
        raise DecisionError(f"{where} must not contain {forbidden}")
    if set(data) != expected:
        raise DecisionError(
            f"{where} must have exactly the fields {sorted(expected)}, got {sorted(data)}"
        )


def _check_exposure(value: Any) -> None:
    if type(value) not in (int, float):
        raise DecisionError(f"exposure_us must be a JSON number, got {type(value).__name__}")
    try:
        as_float = float(value)
    except OverflowError as exc:
        raise DecisionError(f"exposure_us {value!r} is not a finite number") from exc
    if not math.isfinite(as_float) or as_float != value:
        raise DecisionError(f"exposure_us {value!r} is not a finite number")


def parse_decision(raw: Any) -> dict:
    """Validate one raw provider reply; return the decision unchanged or raise DecisionError."""
    if not isinstance(raw, str):
        raise DecisionError(f"provider output must be a JSON string, got {type(raw).__name__}")
    try:
        data = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except ValueError as exc:
        raise DecisionError(f"provider output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DecisionError(f"decision must be a JSON object, got {type(data).__name__}")

    action = data.get("action")
    if action == ACTION_FINISH and type(action) is str:
        _check_fields("finish decision", data, {"action", "reason"})
        reason = data["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise DecisionError("finish decision needs a non-empty reason")
        return {"action": ACTION_FINISH, "reason": reason}
    if action != ACTION_CALL_TOOL or type(action) is not str:
        raise DecisionError(f"action must be {ACTION_CALL_TOOL!r} or {ACTION_FINISH!r}, got {action!r}")

    _check_fields("call_tool decision", data, {"action", "tool", "arguments"})
    tool = data["tool"]
    if type(tool) is not str or tool not in CAMERA_TOOLS:
        raise DecisionError(f"unknown tool {tool!r}; allowed: {list(CAMERA_TOOLS)}")
    arguments = data["arguments"]
    if not isinstance(arguments, dict):
        raise DecisionError(f"arguments must be a JSON object, got {type(arguments).__name__}")
    _check_fields(f"{tool} arguments", arguments, set(TOOL_ARGUMENTS[tool]))
    if tool == EXPOSURE_TOOL:
        _check_exposure(arguments["exposure_us"])
    return {"action": ACTION_CALL_TOOL, "tool": tool, "arguments": dict(arguments)}


def _response_problem(tool: str, content: Any) -> str:
    """Why a successful MCP response cannot be interpreted ("" if it can)."""
    if not isinstance(content, dict):
        return "tool result has no structured JSON object"
    if content.get("tool") != tool:
        return f"tool result names tool {content.get('tool')!r}, not {tool!r}"
    if "pixels" in content:
        return "tool result unexpectedly contains raw pixels"
    if tool in ("camera_info", MEASUREMENT_TOOL):
        source = content.get("source")
        if source not in _KNOWN_SOURCES:
            return f"tool result has unknown source {source!r}"
        if content.get("synthetic") is not (source == "synthetic"):
            return "tool result source and synthetic flag disagree"
    if tool == MEASUREMENT_TOOL:
        frame_id = content.get("frame_id")
        if type(frame_id) is not int:
            return "measurement has no integer frame_id"
        if not isinstance(content.get("metrics"), dict):
            return "measurement has no metrics object"
        exposure = content.get("exposure_us")
        if type(exposure) not in (int, float) or not math.isfinite(exposure):
            return "measurement has no finite exposure_us"
    return ""


def _error_text(result: Any) -> str:
    return " ".join(getattr(block, "text", "") for block in (result.content or []))


class AdaptiveMCPExecutor:
    """Repeated provider decision -> MCP tool call -> feedback, for one approved host."""

    def __init__(self, host: OfflineApprovedSessionHost, decision_provider: Any,
                 research_goal: str, max_decisions: int):
        if not isinstance(host, OfflineApprovedSessionHost):
            raise TypeError(f"host must be an OfflineApprovedSessionHost, got {type(host).__name__}")
        if not callable(getattr(decision_provider, "call", None)):
            raise TypeError("decision_provider must have a call(instructions, input_text) method")
        if not isinstance(research_goal, str) or not research_goal.strip():
            raise ValueError("research_goal must be a non-empty string")
        if isinstance(max_decisions, bool) or not isinstance(max_decisions, int):
            raise TypeError(f"max_decisions must be an int, got {type(max_decisions).__name__}")
        if max_decisions <= 0:
            raise ValueError("max_decisions must be a positive integer")
        self._host = host
        self._provider = decision_provider
        self._goal = research_goal
        self._max_decisions = max_decisions
        self._ran = False
        self._tools: list[dict] = []
        self._events: list[dict] = []
        self._decision_count = 0
        self._tool_call_count = 0
        self._measurement_count = 0
        self._latest_measurement: dict | None = None
        self._sources: set[str] = set()
        self._status: str | None = None
        self._reason = ""
        self._shutdown: dict = {"attempted": False, "succeeded": False, "result": None, "error": None}

    async def run(self) -> dict:
        """Run the loop once and return the JSON-compatible evidence record."""
        if self._ran:
            raise ExecutorError("an AdaptiveMCPExecutor can run only once")
        self._ran = True
        try:
            try:
                server = self._host.get_mcp_server()
            except HostError as exc:
                self._stop(STATUS_HOST_NOT_ACTIVE, f"host is not approved and active: {exc}")
            else:
                async with Client(server, mode="legacy",
                                  read_timeout_seconds=CLIENT_READ_TIMEOUT_SECONDS) as client:
                    await self._loop(client)
        except Exception as exc:
            previous = f" (after status {self._status!r})" if self._status else ""
            self._stop(STATUS_ERROR, f"executor error{previous}: {type(exc).__name__}: {exc}")
        finally:
            self._shutdown_host()
        return self._result()

    # -- loop -----------------------------------------------------------------

    async def _loop(self, client: Client) -> None:
        if not await self._load_tools(client):
            return
        while True:
            if self._decision_count >= self._max_decisions:
                self._stop(STATUS_DECISION_BUDGET_EXHAUSTED,
                           f"decision budget of {self._max_decisions} used without finish")
                return
            decision = self._decide()
            if decision is None:
                return
            if decision["action"] == ACTION_FINISH:
                if self._measurement_count:
                    self._stop(STATUS_COMPLETED, decision["reason"])
                else:
                    self._stop(STATUS_FINISHED_WITHOUT_MEASUREMENT,
                               "provider finished before any successful camera_measure: "
                               + decision["reason"])
                return
            if not await self._call_tool(client, decision["tool"], decision["arguments"]):
                return

    async def _load_tools(self, client: Client) -> bool:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
        if set(listed) != set(CAMERA_TOOLS):
            self._stop(STATUS_INVALID_TOOL_RESPONSE,
                       f"server lists tools {sorted(listed)}, expected {sorted(CAMERA_TOOLS)}")
            return False
        for name in CAMERA_TOOLS:
            schema = listed[name].input_schema
            properties = schema.get("properties", {}) if isinstance(schema, dict) else None
            if not isinstance(properties, dict) or sorted(properties) != sorted(TOOL_ARGUMENTS[name]):
                self._stop(STATUS_INVALID_TOOL_RESPONSE,
                           f"server schema for {name} does not match the expected arguments")
                return False
            self._tools.append({"name": name, "description": listed[name].description,
                                "input_schema": schema})
        return True

    def _decide(self) -> dict | None:
        """One provider call (one budget unit); the validated decision, or None after stopping."""
        self._decision_count += 1
        number = self._decision_count
        input_text = json.dumps(self._context(number), ensure_ascii=False, allow_nan=False)
        event = {"type": "decision", "decision_number": number}
        try:
            raw = self._provider.call(DECISION_INSTRUCTIONS, input_text)
        except Exception as exc:
            error = f"provider raised {type(exc).__name__}: {exc}"
            self._record(event, raw_output=None, valid=False, error=error)
            self._stop(STATUS_PROVIDER_ERROR, error)
            return None
        raw_output = raw if isinstance(raw, str) else repr(raw)
        try:
            decision = parse_decision(raw)
        except DecisionError as exc:
            self._record(event, raw_output=raw_output, valid=False, error=str(exc))
            self._stop(STATUS_INVALID_DECISION, f"decision {number} rejected: {exc}")
            return None
        self._record(event, raw_output=raw_output, valid=True, decision=decision)
        return decision

    async def _call_tool(self, client: Client, tool: str, arguments: dict) -> bool:
        self._tool_call_count += 1
        event = {"type": "tool_call", "decision_number": self._decision_count,
                 "tool": tool, "arguments": dict(arguments)}
        try:
            result = await client.call_tool(tool, arguments)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._record(event, status="transport_error", result=None, error=error)
            self._stop(STATUS_ERROR, f"MCP call {tool} failed: {error}")
            return False
        if result.is_error is True:
            error = _error_text(result)
            self._record(event, status="mcp_error", result=None, error=error)
            self._stop(STATUS_TOOL_ERROR, f"{tool} returned an MCP error (not retried): {error}")
            return False
        content = result.structured_content
        problem = "MCP is_error flag is not a boolean" if result.is_error is not False else ""
        problem = problem or _response_problem(tool, content)
        if problem:
            self._record(event, status="invalid_response",
                         result=content if isinstance(content, dict) else None, error=problem)
            self._stop(STATUS_INVALID_TOOL_RESPONSE, f"{tool}: {problem}")
            return False
        self._record(event, status="success", result=content, error=None,
                     source=content.get("source"), frame_id=content.get("frame_id"))
        if "source" in content:
            self._sources.add(content["source"])
        if tool == MEASUREMENT_TOOL:
            self._measurement_count += 1
            self._latest_measurement = content
        return True

    # -- bookkeeping ----------------------------------------------------------

    def _context(self, number: int) -> dict:
        return {
            "research_goal": self._goal,
            "tools": self._tools,
            "history": self._events,
            "latest_measurement": self._latest_measurement,
            "decision_budget": {"max": self._max_decisions, "used": number - 1,
                                "remaining": self._max_decisions - number + 1},
        }

    def _record(self, event: dict, **fields: Any) -> None:
        event = {"sequence": len(self._events) + 1, **event, **fields}
        # A JSON round trip keeps the log JSON-compatible and detached from live objects.
        self._events.append(json.loads(json.dumps(event, allow_nan=False)))

    def _stop(self, status: str, reason: str) -> None:
        self._status = status
        self._reason = reason

    def _shutdown_host(self) -> None:
        self._shutdown["attempted"] = True
        try:
            self._shutdown["result"] = json.loads(json.dumps(self._host.shutdown()))
            self._shutdown["succeeded"] = True
        except Exception as exc:
            self._shutdown["error"] = f"{type(exc).__name__}: {exc}"
            self._reason = (f"host shutdown failed ({self._shutdown['error']}); "
                            f"run had ended with status {self._status!r}: {self._reason}")
            self._status = STATUS_CLEANUP_FAILED

    def _result(self) -> dict:
        sources = sorted(self._sources)
        return {
            "schema": RESULT_SCHEMA,
            "research_goal": self._goal,
            "status": self._status,
            "final_reason": self._reason,
            "decision_count": self._decision_count,
            "max_decisions": self._max_decisions,
            "tool_call_count": self._tool_call_count,
            "successful_measurement_count": self._measurement_count,
            "latest_measurement": self._latest_measurement,
            "events": list(self._events),
            "host_shut_down": self._shutdown["succeeded"],
            "host_shutdown": dict(self._shutdown),
            "provenance": {
                "transport": "mcp-client (in-process, official SDK)",
                "decision_provider": f"{type(self._provider).__module__}."
                                     f"{type(self._provider).__qualname__}",
                "sources_observed": sources,
                "synthetic_only": bool(sources) and sources == ["synthetic"],
                "hardware_evidence": "hardware" in sources,
            },
        }
