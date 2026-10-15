"""Trusted offline host: LangGraph human approval -> SessionGate -> authorized MCP server.

A small application control-plane component for ONE synthetic camera session.
It connects the existing pieces without adding a policy of its own:

    hitl_approval       real LangGraph interrupt/resume approval workflow
    session_policy      SessionEnvelope and SessionGate (the only validator)
    authorized_camera   AuthorizedCameraRuntime (guards every camera call)
    authorized_mcp      create_authorized_server(runtime)
    fake_camera         FakeCamera, the only backend this host ever uses

It is NOT an MCP tool and must never be reachable by an Executor LLM: start(),
resume() and shutdown() are trusted entry points that the eventual CLI/UI must
leave under the control of a real human. Nothing here authenticates that human.

Lifecycle (synchronous, single process, one approval per host):

    host = OfflineApprovedSessionHost(envelope)   # no backend call
    request = host.start(thread_id)               # graph pauses at interrupt()
    host.pending()                                # the request to display
    result = host.resume(decision)                # human's structured decision
    server = host.get_mcp_server()                # only after valid approval
    host.shutdown()                               # runtime.shutdown(), then refuse all

resume() activates execution only if ALL of these hold:
    1. this host's own graph thread, resumed with `decision`, finished with
       status "approved" for exactly this host's session_id and envelope SHA-256;
    2. a fresh execution-side SessionGate for the same envelope independently
       accepts the same `decision` through submit_decision().
Only then are AuthorizedCameraRuntime(FakeCamera, gate) and the MCP server
built. A refusal, mismatch or error is terminal: no runtime or server is ever
created and the backend is never touched. There is no way to pass an approval
outcome, status, token or fingerprint in as authority.

States: new -> awaiting_approval -> active | refused | failed; any -> shut_down.
start() and resume() each run at most once. This host is not thread-safe and
does not survive a process crash; authorization lives only in memory.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from authorized_camera import AuthorizedCameraRuntime
from authorized_mcp import create_authorized_server
from fake_camera import FakeCamera
from hitl_approval import (
    STATUS_APPROVED,
    build_approval_graph,
    pending_approval,
    resume_approval,
    start_approval,
)
from session_policy import SessionEnvelope, SessionGate

HOST_NEW = "new"
HOST_AWAITING_APPROVAL = "awaiting_approval"
HOST_ACTIVE = "active"
HOST_REFUSED = "refused"
HOST_FAILED = "failed"
HOST_SHUT_DOWN = "shut_down"


class HostError(Exception):
    """The host was used out of order, is not approved, or has been shut down."""


class OfflineApprovedSessionHost:
    """Owns one envelope, one approval thread, one FakeCamera and, once approved, one MCP server."""

    def __init__(self, envelope: SessionEnvelope, *, clock: Callable[[], float] = time.monotonic):
        if not isinstance(envelope, SessionEnvelope):
            raise TypeError(f"envelope must be a SessionEnvelope, got {type(envelope).__name__}")
        if not callable(clock):
            raise TypeError("clock must be callable")
        # SessionEnvelope is frozen; it is only read, never rebuilt or mutated.
        self._envelope = envelope
        self._envelope_sha256 = envelope.sha256()
        self._clock = clock
        self._graph = build_approval_graph()  # private InMemorySaver: no shared approval state
        self._camera = FakeCamera()  # constructed only; no backend call before approval
        self._state = HOST_NEW
        self._thread_id: str | None = None
        self._runtime: AuthorizedCameraRuntime | None = None
        self._server = None

    # -- trusted control plane ---------------------------------------------

    def start(self, thread_id: str) -> dict:
        """Start the approval thread and return the pending request (graph paused at interrupt)."""
        self._require(HOST_NEW, "start")
        self._state = HOST_FAILED  # until the graph has really paused for this envelope
        try:
            request = start_approval(self._graph, thread_id, self._envelope)
        except Exception as exc:
            raise HostError(f"approval could not be started: {exc}") from exc
        self._thread_id = thread_id
        self._state = HOST_AWAITING_APPROVAL
        return request

    def pending(self) -> dict | None:
        """The approval request waiting for a human decision, or None."""
        self._require_not_shut_down("pending")
        if self._thread_id is None:
            return None
        return pending_approval(self._graph, self._thread_id)

    def resume(self, decision: Any) -> dict:
        """Resume this host's own thread with the human decision; activate only on valid approval.

        Returns {"activated", "reason", "approval_outcome"}. The outcome is
        informational; only the host's own state decides whether execution is
        available. Can be called once; every result is final.
        """
        self._require(HOST_AWAITING_APPROVAL, "resume")
        self._state = HOST_FAILED  # until a decision has been fully applied
        try:
            outcome = resume_approval(self._graph, self._thread_id, decision)
        except Exception as exc:
            raise HostError(f"approval could not be resumed: {exc}") from exc

        problem = self._outcome_problem(outcome)
        if problem:
            self._state = HOST_REFUSED
            return self._resume_result(False, problem, outcome)

        # Independent execution-side validation of the very same decision.
        gate = SessionGate(self._envelope, self._clock)
        if not gate.submit_decision(decision):
            self._state = HOST_REFUSED
            return self._resume_result(
                False, f"execution gate refused the decision: {gate.reason}", outcome
            )
        try:
            runtime = AuthorizedCameraRuntime(self._camera, gate)
            server = create_authorized_server(runtime)
        except Exception as exc:
            gate.close()
            raise HostError(f"execution could not be activated: {exc}") from exc
        self._runtime = runtime
        self._server = server
        self._state = HOST_ACTIVE
        return self._resume_result(True, "human approved the exact envelope", outcome)

    def get_mcp_server(self):
        """The authorized MCP server for in-process clients; only while approved and not shut down."""
        self._require(HOST_ACTIVE, "get_mcp_server")
        return self._server

    def shutdown(self) -> dict:
        """Refuse all further host use and release the runtime via runtime.shutdown().

        Idempotent. Without a runtime the backend was never contacted, so
        nothing is closed. A CleanupError from runtime.shutdown() propagates;
        calling shutdown() again retries only runtime.shutdown().
        """
        self._state = HOST_SHUT_DOWN
        self._server = None
        if self._runtime is None:
            return {"host_state": HOST_SHUT_DOWN, "runtime": None}
        return {"host_state": HOST_SHUT_DOWN, "runtime": self._runtime.shutdown()}

    # -- internals ----------------------------------------------------------

    def _outcome_problem(self, outcome: Any) -> str:
        """Check the graph's result against the exact envelope this host owns."""
        if not isinstance(outcome, dict):
            return "approval workflow returned no outcome"
        if outcome.get("status") != STATUS_APPROVED:
            return str(outcome.get("reason") or "approval was not granted")
        if outcome.get("session_id") != self._envelope.session_id:
            return "approval outcome names a different session"
        if (outcome.get("envelope_sha256") != self._envelope_sha256
                or self._envelope.sha256() != self._envelope_sha256):
            return "approval outcome does not match this host's envelope SHA-256"
        return ""

    @staticmethod
    def _resume_result(activated: bool, reason: str, outcome: Any) -> dict:
        return {
            "activated": activated,
            "reason": reason,
            "approval_outcome": dict(outcome) if isinstance(outcome, dict) else None,
        }

    def _require(self, state: str, operation: str) -> None:
        self._require_not_shut_down(operation)
        if self._state != state:
            raise HostError(f"{operation} is not allowed: host is {self._state}")

    def _require_not_shut_down(self, operation: str) -> None:
        if self._state == HOST_SHUT_DOWN:
            raise HostError(f"{operation} is not allowed: host is shut down")
