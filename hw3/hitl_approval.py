"""LangGraph human approval workflow for one hardware SessionEnvelope.

A small, offline, reusable component: it asks a human to approve the exact
limits of one hardware session through the real langgraph.types.interrupt()
and Command(resume=...) mechanism, and validates the reply with the existing
SessionGate.submit_decision(). It never opens a camera, calls the MCP server,
the vendor SDK, the network or an LLM, and it performs no hardware action.

Graph (compiled with an InMemorySaver checkpointer by default):

    START -> hardware_approval -> END

hardware_approval rebuilds the trusted SessionEnvelope from state, creates a
throwaway SessionGate, interrupts with gate.approval_request() (the exact
envelope and its SHA-256) and, on resume, applies the human reply with
gate.submit_decision(). LangGraph re-runs the node from the top on resume, so
everything before interrupt() is pure: building an envelope and a gate has
no side effects. The gate is never stored in checkpoint state; state holds
only JSON-compatible values.

Public API (one approval per thread_id; thread_id is required):

    graph = build_approval_graph()
    request = start_approval(graph, thread_id, envelope)    # pauses at interrupt
    pending_approval(graph, thread_id)                      # the pending request or None
    outcome = resume_approval(graph, thread_id, decision)   # Command(resume={interrupt_id: decision})
    approval_outcome(graph, thread_id)                      # final outcome or None

Only {"approved": True, "session_id": <exact>, "envelope_sha256": <exact>}
(see session_policy.session_approval) approves; anything else refuses, and a
refusal ends the workflow. A thread can be started once and resumed once:
re-starting or re-resuming a finished thread raises ApprovalError, so a
recorded decision cannot be replayed to approve again.

The outcome is an informational record of what the human decided. It is NOT
an authorization token: nothing accepts it as permission to act. A later
integration step must enforce the approved envelope at the MCP hardware
execution boundary with its own SessionGate.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from session_policy import ENVELOPE_SCHEMA, SessionEnvelope, SessionGate

APPROVAL_NODE = "hardware_approval"
STATUS_APPROVED = "approved"
STATUS_REFUSED = "refused"


class ApprovalError(Exception):
    """The approval workflow was used out of order (unknown, reused or finished thread)."""


class ApprovalInput(TypedDict):
    """The only input a new thread accepts; any other key (e.g. "status") is dropped."""

    envelope: dict[str, Any]


class ApprovalState(TypedDict, total=False):
    envelope: dict[str, Any]  # SessionEnvelope.canonical_dict()
    session_id: str
    envelope_sha256: str
    status: str  # STATUS_APPROVED or STATUS_REFUSED, written only by the approval node
    reason: str


def _envelope_from_state(data: Any) -> SessionEnvelope:
    """Rebuild the envelope; SessionEnvelope itself re-validates every field."""
    if not isinstance(data, dict):
        raise ApprovalError("state does not hold an envelope")
    fields = dict(data)
    if fields.pop("schema", None) != ENVELOPE_SCHEMA:
        raise ApprovalError("envelope has an unknown schema")
    fields["allowed_tools"] = tuple(fields.get("allowed_tools", ()))
    return SessionEnvelope(**fields)


def _hardware_approval(state: ApprovalState) -> dict[str, Any]:
    # No side effects before interrupt(): this node re-runs from the top on resume.
    gate = SessionGate(_envelope_from_state(state.get("envelope")))
    reply = interrupt(gate.approval_request())
    approved = gate.submit_decision(reply)
    return {
        "session_id": gate.envelope.session_id,
        "envelope_sha256": gate.envelope_sha256,
        "status": STATUS_APPROVED if approved else STATUS_REFUSED,
        "reason": "human approved the exact envelope" if approved else gate.reason,
    }


def build_approval_graph(checkpointer=None):
    """Compile START -> hardware_approval -> END with a checkpointer (InMemorySaver by default)."""
    builder = StateGraph(ApprovalState, input_schema=ApprovalInput)
    builder.add_node(APPROVAL_NODE, _hardware_approval)
    builder.add_edge(START, APPROVAL_NODE)
    builder.add_edge(APPROVAL_NODE, END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def thread_config(thread_id: str) -> dict:
    """Run config for one approval thread. There is no default thread_id."""
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ApprovalError("an explicit, non-empty thread_id is required")
    return {"configurable": {"thread_id": thread_id}}


def pending_approval(graph, thread_id: str) -> dict | None:
    """The approval request this thread is waiting on, or None."""
    snapshot = graph.get_state(thread_config(thread_id))
    if not snapshot.interrupts:
        return None
    return snapshot.interrupts[0].value


def start_approval(graph, thread_id: str, envelope: SessionEnvelope) -> dict:
    """Start a new thread for a trusted envelope and return its pending approval request.

    Raises ApprovalError if the thread was already used, so a thread can
    never be restarted after it was approved or refused.
    """
    if not isinstance(envelope, SessionEnvelope):
        raise TypeError(f"envelope must be a SessionEnvelope, got {type(envelope).__name__}")
    config = thread_config(thread_id)
    if graph.get_state(config).created_at is not None:
        raise ApprovalError(f"thread {thread_id!r} was already used")
    graph.invoke({"envelope": envelope.canonical_dict()}, config)
    request = pending_approval(graph, thread_id)
    if request is None or request.get("envelope_sha256") != envelope.sha256():
        raise ApprovalError("workflow did not pause for approval of this envelope")
    return request


def resume_approval(graph, thread_id: str, decision: Any) -> dict:
    """Resume a pending thread with the human decision and return the final outcome.

    The decision is validated only by SessionGate.submit_decision(). Raises
    ApprovalError if the thread has no pending approval (never started,
    already finished), so a decision cannot be replayed.
    """
    config = thread_config(thread_id)
    interrupts = graph.get_state(config).interrupts
    if not interrupts:
        raise ApprovalError(f"thread {thread_id!r} has no pending approval")
    # Keyed by interrupt id: the decision answers exactly this pending request,
    # and values such as {} or None are delivered instead of read as "no resume".
    graph.invoke(Command(resume={interrupts[0].id: decision}), config)
    outcome = approval_outcome(graph, thread_id)
    if outcome is None:
        raise ApprovalError("workflow did not finish after the decision")
    return outcome


def approval_outcome(graph, thread_id: str) -> dict | None:
    """The finished thread's outcome, or None while it is pending or unknown.

    Informational only; it is not an authorization token.
    """
    snapshot = graph.get_state(thread_config(thread_id))
    if snapshot.interrupts or snapshot.next or "status" not in snapshot.values:
        return None
    values = snapshot.values
    return {
        "session_id": values["session_id"],
        "envelope_sha256": values["envelope_sha256"],
        "status": values["status"],
        "reason": values["reason"],
    }
