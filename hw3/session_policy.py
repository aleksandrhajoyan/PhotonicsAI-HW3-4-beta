"""Fail-closed authorization policy for future hardware measurement sessions.

Pure, offline bookkeeping: this module never touches a camera, a vendor SDK,
the MCP server, LangGraph, the network or an LLM, and it never grants any
permission to execute Python code. It only decides whether a connector may
issue one bounded camera tool call.

Pieces:
    SessionEnvelope  immutable, validated limits of one session, built by
                     trusted application code (never directly from LLM output)
    envelope_sha256  SHA-256 of the envelope's canonical JSON (an integrity
                     check, not authentication of the human)
    SessionGate      owns one envelope, its approval and its budgets

State machine (SessionGate.state):
    awaiting_approval --valid approval--------------------> active
    awaiting_approval --refusal / any invalid decision----> refused   (terminal)
    active --tool-call budget fully used------------------> exhausted (terminal)
    active --elapsed >= max_duration_seconds--------------> expired   (terminal)
    active --ambiguous outcome / clock fault / abort()----> aborted   (terminal)
    any non-terminal or exhausted/expired --close()-------> closed    (terminal)
    close() while an action is still in flight------------> aborted   (outcome unknown)

Only "active" authorizes actions. Every terminal state refuses all further
acquisition and exposure changes; close() and abort() never raise and stay
available in every state.

Action protocol (one action in flight at a time):
    grant = gate.begin_action(action_id, tool, arguments)  # reserves budget first
    ... the connector performs exactly that one tool call ...
    gate.complete_action(action_id, "succeeded" | "failed" | "ambiguous")

Reserved budget is never refunded. An action_id is single-use, even if its
request was rejected. "failed" means the call definitely did not take effect;
when the outcome is unknown the connector must report "ambiguous", which
aborts the session instead of allowing a retry. The gate does not and cannot
guarantee physical exactly-once execution across process crashes.

Physical cleanup is NOT this gate's job. The eventual connector/server must
separately close the physical device (camera_close) whenever a session ends,
fails or is aborted, without routing that cleanup through begin_action and
without conditioning it on this gate's state or budgets: exhausted or expired
budgets must never block safe cleanup. close() and abort() here are local
state transitions only and make no hardware call.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

# The camera MCP tool vocabulary of hw3/mcp_server.py. No other tool, device
# command or code-execution permission can ever enter an envelope.
CAMERA_TOOLS = (
    "camera_info",
    "camera_open",
    "camera_set_exposure",
    "camera_measure",
    "camera_close",
)
MEASUREMENT_TOOL = "camera_measure"
EXPOSURE_TOOL = "camera_set_exposure"

ENVELOPE_SCHEMA = "lumen.hw3.session_envelope.v1"
MAX_IDENTIFIER_LENGTH = 128

AWAITING_APPROVAL = "awaiting_approval"
ACTIVE = "active"
REFUSED = "refused"
EXHAUSTED = "exhausted"
EXPIRED = "expired"
ABORTED = "aborted"
CLOSED = "closed"
STATES = (AWAITING_APPROVAL, ACTIVE, REFUSED, EXHAUSTED, EXPIRED, ABORTED, CLOSED)
TERMINAL_STATES = frozenset({REFUSED, EXHAUSTED, EXPIRED, ABORTED, CLOSED})

IN_FLIGHT = "in_flight"
SUCCEEDED = "succeeded"
FAILED = "failed"
AMBIGUOUS = "ambiguous"
OUTCOMES = (SUCCEEDED, FAILED, AMBIGUOUS)


class PolicyError(Exception):
    """An action or completion that the session policy does not authorize."""


def _check_identifier(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if not value.isprintable():
        raise ValueError(f"{name} must contain only printable characters")
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{name} must be at most {MAX_IDENTIFIER_LENGTH} characters")


def _check_real(name: str, value: Any) -> float:
    """A finite int/float (never bool), returned as float without altering its value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if result != value:
        raise ValueError(f"{name} cannot be represented exactly as a float: {value!r}")
    return result


def _check_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class SessionEnvelope:
    """Immutable limits of one hardware session. Invalid values are rejected, never corrected."""

    session_id: str
    device_label: str
    allowed_tools: tuple[str, ...]
    min_exposure_us: float
    max_exposure_us: float
    max_measurements: int
    max_tool_calls: int
    max_duration_seconds: float

    def __post_init__(self) -> None:
        _check_identifier("session_id", self.session_id)
        _check_identifier("device_label", self.device_label)

        if not isinstance(self.allowed_tools, tuple):
            raise TypeError("allowed_tools must be a tuple of tool names")
        if not self.allowed_tools:
            raise ValueError("allowed_tools must not be empty")
        for tool in self.allowed_tools:
            if not isinstance(tool, str):
                raise TypeError(f"tool names must be str, got {type(tool).__name__}")
            if tool not in CAMERA_TOOLS:
                raise ValueError(f"unknown tool {tool!r}; allowed vocabulary: {CAMERA_TOOLS}")
        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("allowed_tools must not contain duplicates")

        # Normalize int -> float type only (the value is unchanged), so equal
        # envelopes always share one canonical form and fingerprint.
        min_us = _check_real("min_exposure_us", self.min_exposure_us)
        max_us = _check_real("max_exposure_us", self.max_exposure_us)
        duration = _check_real("max_duration_seconds", self.max_duration_seconds)
        measurements = _check_int("max_measurements", self.max_measurements)
        tool_calls = _check_int("max_tool_calls", self.max_tool_calls)
        if min_us <= 0:
            raise ValueError("min_exposure_us must be > 0")
        if max_us < min_us:
            raise ValueError("max_exposure_us must be >= min_exposure_us")
        if measurements <= 0:
            raise ValueError("max_measurements must be > 0")
        if tool_calls <= 0:
            raise ValueError("max_tool_calls must be > 0")
        if tool_calls < measurements:
            raise ValueError("max_tool_calls must be >= max_measurements")
        if duration <= 0:
            raise ValueError("max_duration_seconds must be > 0")
        object.__setattr__(self, "min_exposure_us", min_us)
        object.__setattr__(self, "max_exposure_us", max_us)
        object.__setattr__(self, "max_duration_seconds", duration)

    def canonical_dict(self) -> dict[str, Any]:
        """Every field plus a schema tag; contains no secrets or credentials."""
        return {
            "schema": ENVELOPE_SCHEMA,
            "session_id": self.session_id,
            "device_label": self.device_label,
            "allowed_tools": list(self.allowed_tools),
            "min_exposure_us": self.min_exposure_us,
            "max_exposure_us": self.max_exposure_us,
            "max_measurements": self.max_measurements,
            "max_tool_calls": self.max_tool_calls,
            "max_duration_seconds": self.max_duration_seconds,
        }

    def canonical_json(self) -> str:
        """Sorted keys, no whitespace, ASCII only, NaN/infinity forbidden."""
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def sha256(self) -> str:
        return envelope_sha256(self)


def envelope_sha256(envelope: SessionEnvelope) -> str:
    """Lowercase hex SHA-256 of the envelope's canonical JSON (UTF-8)."""
    if not isinstance(envelope, SessionEnvelope):
        raise TypeError(f"expected SessionEnvelope, got {type(envelope).__name__}")
    return hashlib.sha256(envelope.canonical_json().encode("utf-8")).hexdigest()


def session_approval(session_id: str, envelope_sha256_hex: str, approved: bool) -> dict:
    """The structured decision a human returns for a session approval request."""
    return {"approved": approved, "session_id": session_id, "envelope_sha256": envelope_sha256_hex}


@dataclass(frozen=True)
class ActionGrant:
    """Authorization for exactly one tool call. exposure_us is set only for camera_set_exposure."""

    session_id: str
    envelope_sha256: str
    action_id: str
    tool: str
    exposure_us: float | None


class SessionGate:
    """Approval state and budgets for one SessionEnvelope.

    clock is an injectable monotonic clock returning seconds (default
    time.monotonic). It is read only during approval, begin_action and
    complete_action; there are no background threads.
    """

    def __init__(self, envelope: SessionEnvelope, clock: Callable[[], float] = time.monotonic):
        if not isinstance(envelope, SessionEnvelope):
            raise TypeError(f"envelope must be a SessionEnvelope, got {type(envelope).__name__}")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._envelope = envelope
        self._sha256 = envelope_sha256(envelope)
        self._clock = clock
        self._state = AWAITING_APPROVAL
        self._reason = ""
        self._started_at: float | None = None
        self._last_clock: float | None = None
        self._tool_calls_used = 0
        self._measurements_used = 0
        self._seen_action_ids: set[str] = set()
        self._actions: dict[str, dict[str, Any]] = {}
        self._in_flight: str | None = None

    # -- read-only views ---------------------------------------------------

    @property
    def envelope(self) -> SessionEnvelope:
        return self._envelope

    @property
    def envelope_sha256(self) -> str:
        return self._sha256

    @property
    def state(self) -> str:
        return self._state

    @property
    def reason(self) -> str:
        """Why the gate entered its current terminal state ("" otherwise)."""
        return self._reason

    @property
    def tool_calls_used(self) -> int:
        return self._tool_calls_used

    @property
    def measurements_used(self) -> int:
        return self._measurements_used

    @property
    def in_flight_action_id(self) -> str | None:
        return self._in_flight

    def action_status(self, action_id: str) -> str | None:
        record = self._actions.get(action_id)
        return None if record is None else record["status"]

    def approval_request(self) -> dict[str, Any]:
        """What a human must see to approve: the exact envelope and its fingerprint."""
        return {
            "kind": "hardware_session_approval",
            "session_id": self._envelope.session_id,
            "envelope_sha256": self._sha256,
            "envelope": self._envelope.canonical_dict(),
        }

    # -- approval ----------------------------------------------------------

    def submit_decision(self, reply: Any) -> bool:
        """Apply the human decision; True only if the session became active.

        Only {"approved": True, "session_id": <exact>, "envelope_sha256":
        <exact>} approves. Anything else (refusal, free text, missing or
        mismatched fields) refuses the gate permanently. Outside
        awaiting_approval nothing changes and False is returned.
        """
        if self._state != AWAITING_APPROVAL:
            return False
        reason = self._approval_problem(reply)
        if reason:
            self._terminate(REFUSED, reason)
            return False
        try:
            now = self._read_clock()
        except PolicyError:
            return False
        self._started_at = now
        self._state = ACTIVE
        return True

    def _approval_problem(self, reply: Any) -> str:
        if not isinstance(reply, dict):
            return "reply was not a structured approval decision"
        if reply.get("approved") is not True:
            return "human refused"
        session_id = reply.get("session_id")
        if type(session_id) is not str or session_id != self._envelope.session_id:
            return f"approval names session {session_id!r}, not {self._envelope.session_id!r}"
        digest = reply.get("envelope_sha256")
        if type(digest) is not str or digest != self._sha256:
            return "approval does not match the envelope SHA-256"
        return ""

    # -- actions -----------------------------------------------------------

    def begin_action(
        self, action_id: str, tool: str, arguments: Mapping[str, Any] | None = None
    ) -> ActionGrant:
        """Authorize one tool call and reserve its budget before it executes.

        Raises PolicyError (or TypeError/ValueError for a malformed action_id)
        when the call is not authorized; nothing is reserved in that case.
        """
        _check_identifier("action_id", action_id)
        if action_id in self._seen_action_ids:
            raise PolicyError(f"action_id {action_id!r} was already used")
        self._seen_action_ids.add(action_id)

        if self._state != ACTIVE:
            raise PolicyError(f"session is {self._state}; no actions are authorized")
        self._check_not_expired()
        if self._in_flight is not None:
            raise PolicyError(f"action {self._in_flight!r} is still in flight")
        if not isinstance(tool, str) or tool not in self._envelope.allowed_tools:
            raise PolicyError(f"tool {tool!r} is not authorized for this session")
        exposure_us = self._check_arguments(tool, arguments)
        if self._tool_calls_used >= self._envelope.max_tool_calls:
            raise PolicyError("tool-call budget exhausted")
        if tool == MEASUREMENT_TOOL and self._measurements_used >= self._envelope.max_measurements:
            raise PolicyError("measurement budget exhausted")

        self._tool_calls_used += 1
        if tool == MEASUREMENT_TOOL:
            self._measurements_used += 1
        self._actions[action_id] = {"tool": tool, "exposure_us": exposure_us, "status": IN_FLIGHT}
        self._in_flight = action_id
        return ActionGrant(
            session_id=self._envelope.session_id,
            envelope_sha256=self._sha256,
            action_id=action_id,
            tool=tool,
            exposure_us=exposure_us,
        )

    def _check_arguments(self, tool: str, arguments: Mapping[str, Any] | None) -> float | None:
        if tool != EXPOSURE_TOOL:
            if arguments:
                raise PolicyError(f"{tool} takes no arguments")
            return None
        if not isinstance(arguments, Mapping) or set(arguments) != {"exposure_us"}:
            raise PolicyError("camera_set_exposure requires exactly the argument exposure_us")
        try:
            value = _check_real("exposure_us", arguments["exposure_us"])
        except (TypeError, ValueError) as exc:
            raise PolicyError(str(exc)) from exc
        if not self._envelope.min_exposure_us <= value <= self._envelope.max_exposure_us:
            raise PolicyError(
                f"exposure_us {value!r} is outside the approved range "
                f"[{self._envelope.min_exposure_us!r}, {self._envelope.max_exposure_us!r}]"
            )
        return value

    def complete_action(self, action_id: str, outcome: str) -> str:
        """Acknowledge the outcome of the in-flight action; returns the new gate state.

        "ambiguous" (or any unrecognized outcome) aborts the session: an action
        whose effect is unknown is never retried automatically.
        """
        record = self._actions.get(action_id)
        if record is None:
            raise PolicyError(f"action {action_id!r} was never authorized")
        if record["status"] != IN_FLIGHT:
            raise PolicyError(f"action {action_id!r} is already {record['status']}")
        self._in_flight = None
        if outcome not in (SUCCEEDED, FAILED):
            record["status"] = AMBIGUOUS
            self._terminate(ABORTED, f"action {action_id!r} ended with ambiguous outcome {outcome!r}")
            return self._state
        record["status"] = outcome
        if self._state == ACTIVE:
            try:
                self._check_not_expired()
            except PolicyError:
                return self._state
            if self._tool_calls_used >= self._envelope.max_tool_calls:
                self._terminate(EXHAUSTED, "tool-call budget fully used")
        return self._state

    # -- local termination (no hardware calls) -----------------------------

    def abort(self, reason: str = "aborted") -> str:
        """Locally abort the session. Never raises; idempotent once terminal-aborted/closed/refused."""
        if self._state not in (ABORTED, CLOSED, REFUSED):
            self._terminate(ABORTED, str(reason))
        return self._state

    def close(self) -> str:
        """Locally close the session. Never raises; never makes a hardware call.

        With an action still in flight its outcome is unknown, so the session
        is aborted rather than closed. Already aborted/closed/refused gates are
        left unchanged.
        """
        if self._state in (ABORTED, CLOSED, REFUSED):
            return self._state
        if self._in_flight is not None:
            self._actions[self._in_flight]["status"] = AMBIGUOUS
            self._terminate(ABORTED, f"closed while action {self._in_flight!r} was in flight")
            self._in_flight = None
            return self._state
        self._terminate(CLOSED, "closed")
        return self._state

    # -- internals ---------------------------------------------------------

    def _terminate(self, state: str, reason: str) -> None:
        self._state = state
        self._reason = reason

    def _read_clock(self) -> float:
        try:
            now = self._clock()
            now = _check_real("clock", now)
        except Exception as exc:  # any clock fault fails closed
            self._terminate(ABORTED, f"clock fault: {exc}")
            raise PolicyError(self._reason) from exc
        if self._last_clock is not None and now < self._last_clock:
            self._terminate(ABORTED, "monotonic clock went backwards")
            raise PolicyError(self._reason)
        self._last_clock = now
        return now

    def _check_not_expired(self) -> None:
        now = self._read_clock()
        if now - self._started_at >= self._envelope.max_duration_seconds:
            self._terminate(EXPIRED, "max_duration_seconds elapsed")
            raise PolicyError("session duration expired")
