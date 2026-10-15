"""Trusted offline driver for one LUMEN four-agent session (HW3.5-C3).

One synchronous application-level entry point owns the whole lifecycle:

    trusted envelope template -> fresh session identity -> LangGraph human
    approval -> four-agent LumenAgentSystem graph -> mandatory host.shutdown()

    report = run_lumen_session(envelope_template, outer_model_caller,
                               hardware_decision_provider, research_goal,
                               max_hardware_decisions, human_decision_callback)

It is NOT an MCP tool and must never be reachable by a model. It only wires the
existing pieces together and adds no approval or instrument-control logic:

    OfflineApprovedSessionHost   start() / pending() / resume() / shutdown();
                                 its graph and SessionGate validate the decision
    LumenAgentSystem             the unmodified four-role graph; its Executor is
                                 the only role that reaches the MCP camera tools

Session identity: every call draws a new session_id from the secrets module
(128 random bits) and builds a fresh SessionEnvelope with dataclasses.replace;
every other limit comes unchanged from the trusted template, which is not
mutated. The session_id is also the approval thread_id. No identity is taken
from model output, the research goal, tool arguments or an earlier approval,
so a recorded decision for an older session (or its SHA-256) cannot approve
this one. Neither the session_id nor the SHA-256 authenticates a human.

Approval: human_decision_callback(request) is required and called exactly once
with the request the host's paused approval graph reports. Its reply is passed
unchanged to host.resume(); the driver never builds, edits or completes a
decision. If the host does not activate, the outer graph is never constructed.

Cleanup: once the host exists, host.shutdown() is attempted in a finally block
on every exit path (success, early finish, refusal, invalid approval, callback,
routing, graph or provider failure, an Executor that already shut down). A
shutdown failure is never reported as success: LumenDriverError is raised with
the original error, if any, as its __cause__. KeyboardInterrupt and SystemExit
propagate unchanged after the cleanup attempt (a cleanup failure is added as a
note). Nothing survives process termination or power loss, and nothing is
retried.

Report (JSON-compatible; also attached to LumenDriverError.report): schema,
session_status, approval_status, session_id, envelope_sha256, envelope,
outer_graph_status, dialogue, experiment_status, experiment_result (the
Executor's complete evidence), experiment_records, measurements_occurred,
evidence_provenance (synthetic | hardware | mixed | unknown, from observed
source labels only), shutdown_attempted, shutdown_succeeded, reason. It never
contains the human decision, credentials or SDK buffers. With the FakeCamera
host every measurement is labelled synthetic.
"""

from __future__ import annotations

import dataclasses
import json
import secrets
from collections.abc import Callable
from typing import Any

from approved_session_host import OfflineApprovedSessionHost
from executor_dialogue_bridge import EVIDENCE_UNKNOWN, evidence_type, is_completed
from lumen_agent_system import LumenAgentSystem
from session_policy import SessionEnvelope

REPORT_SCHEMA = "lumen.hw3.session_driver_report.v1"
SESSION_ID_PREFIX = "lumen-"
SESSION_ID_RANDOM_BYTES = 16  # 128 bits from secrets.token_hex
RECURSION_LIMIT = 50

APPROVAL_NOT_STARTED = "not_started"
APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_REFUSED = "refused"

GRAPH_NOT_RUN = "not_run"
GRAPH_RUNNING = "running"
GRAPH_COMPLETED = "completed"
GRAPH_FAILED = "failed"

SESSION_REFUSED = "refused"
SESSION_EXPERIMENT_COMPLETED = "experiment_completed"
SESSION_EXPERIMENT_NOT_COMPLETED = "experiment_not_completed"
SESSION_FINISHED_WITHOUT_EXPERIMENT = "finished_without_experiment"
SESSION_FAILED = "failed"
SESSION_CLEANUP_FAILED = "cleanup_failed"


class LumenDriverError(Exception):
    """The session did not end cleanly; report holds the non-sensitive state reached."""

    def __init__(self, message: str, report: dict | None = None,
                 cleanup_error: BaseException | None = None):
        super().__init__(message)
        self.report = report
        self.cleanup_error = cleanup_error


def new_session_id() -> str:
    """A fresh, unpredictable session identifier (128 random bits)."""
    return SESSION_ID_PREFIX + secrets.token_hex(SESSION_ID_RANDOM_BYTES)


def _json_copy(value: Any) -> Any:
    """A detached, strictly JSON-compatible copy (values unchanged)."""
    return json.loads(json.dumps(value, allow_nan=False))


def _check_inputs(envelope_template: Any, outer_model_caller: Any,
                  hardware_decision_provider: Any, research_goal: Any,
                  max_hardware_decisions: Any, human_decision_callback: Any) -> None:
    if not isinstance(envelope_template, SessionEnvelope):
        raise LumenDriverError("envelope_template must be a SessionEnvelope, "
                               f"got {type(envelope_template).__name__}")
    if not callable(getattr(outer_model_caller, "call", None)):
        raise LumenDriverError("outer_model_caller must have a call(instructions, input_text) "
                               "method")
    if not callable(getattr(hardware_decision_provider, "call", None)):
        raise LumenDriverError("hardware_decision_provider must have a "
                               "call(instructions, input_text) method")
    if not isinstance(research_goal, str) or not research_goal.strip():
        raise LumenDriverError("research_goal must be a non-empty string")
    if isinstance(max_hardware_decisions, bool) or not isinstance(max_hardware_decisions, int) \
            or max_hardware_decisions <= 0:
        raise LumenDriverError("max_hardware_decisions must be a positive int")
    if not callable(human_decision_callback):
        raise LumenDriverError("human_decision_callback is required and must be callable; "
                               "there is no default approval")


def _new_report(envelope: SessionEnvelope) -> dict:
    return {
        "schema": REPORT_SCHEMA,
        "session_status": SESSION_FAILED,
        "approval_status": APPROVAL_NOT_STARTED,
        "session_id": envelope.session_id,
        "envelope_sha256": envelope.sha256(),
        "envelope": envelope.canonical_dict(),
        "outer_graph_status": GRAPH_NOT_RUN,
        "dialogue": None,
        "experiment_status": GRAPH_NOT_RUN,
        "experiment_result": None,
        "experiment_records": [],
        "measurements_occurred": False,
        "evidence_provenance": EVIDENCE_UNKNOWN,
        "shutdown_attempted": False,
        "shutdown_succeeded": False,
        "reason": None,
    }


def _record_experiment(report: dict, system: LumenAgentSystem) -> None:
    """Copy the Executor's complete, unmodified evidence into the report."""
    records = system.experiment_records
    report["experiment_records"] = [
        {"visit": record["visit"], "outcome": record["outcome"],
         "result": None if record["result"] is None else _json_copy(record["result"]),
         "error_type": None if record["error"] is None else type(record["error"]).__name__}
        for record in records]
    latest = system.latest_experiment_result
    if latest is None:
        report["experiment_status"] = (records[-1]["outcome"] if records else GRAPH_NOT_RUN)
        return
    report["experiment_result"] = _json_copy(latest)
    report["experiment_status"] = latest.get("status")
    count = latest.get("successful_measurement_count")
    report["measurements_occurred"] = type(count) is int and count >= 1
    report["evidence_provenance"] = evidence_type(latest)


def _approve_and_run(host: OfflineApprovedSessionHost, report: dict, outer_model_caller: Any,
                     hardware_decision_provider: Any, research_goal: str,
                     max_hardware_decisions: int,
                     human_decision_callback: Callable[[dict], Any]) -> None:
    session_id = report["session_id"]
    host.start(session_id)  # the fresh session_id is the approval thread_id
    report["approval_status"] = APPROVAL_PENDING
    request = host.pending()
    if not isinstance(request, dict) or request.get("session_id") != session_id \
            or request.get("envelope_sha256") != report["envelope_sha256"]:
        raise LumenDriverError("the approval graph is not waiting on this session's envelope")

    # Trusted application-side human interface; its reply goes to the host unchanged.
    decision = human_decision_callback(request)
    activation = host.resume(decision)
    if activation.get("activated") is not True:
        report["approval_status"] = APPROVAL_REFUSED
        report["session_status"] = SESSION_REFUSED
        report["reason"] = ("human approval was refused or invalid for this session; "
                            "the four-agent graph was not run and no measurement occurred")
        return
    report["approval_status"] = APPROVAL_APPROVED

    system = LumenAgentSystem(outer_model_caller, experiment_host=host,
                              decision_provider=hardware_decision_provider,
                              research_goal=research_goal,
                              max_hardware_decisions=max_hardware_decisions)
    graph = system.build()
    report["outer_graph_status"] = GRAPH_RUNNING
    try:
        final = graph.invoke({"dialogue": [{"role": "user", "text": research_goal}]},
                             {"recursion_limit": RECURSION_LIMIT})
    except Exception:
        report["outer_graph_status"] = GRAPH_FAILED
        _record_experiment(report, system)
        raise
    report["outer_graph_status"] = GRAPH_COMPLETED
    report["dialogue"] = _json_copy(final["dialogue"])
    _record_experiment(report, system)

    latest = system.latest_experiment_result
    if latest is None:
        report["session_status"] = SESSION_FINISHED_WITHOUT_EXPERIMENT
        report["reason"] = ("the orchestrator finished without a dispatched experiment; "
                            "no measurement occurred")
    elif is_completed(latest):
        report["session_status"] = SESSION_EXPERIMENT_COMPLETED
    else:
        report["session_status"] = SESSION_EXPERIMENT_NOT_COMPLETED
        report["reason"] = f"executor experiment ended with status {latest.get('status')!r}"


def _shutdown_host(host: OfflineApprovedSessionHost, report: dict) -> Exception | None:
    report["shutdown_attempted"] = True
    try:
        host.shutdown()  # idempotent; harmless after the Executor already shut down
    except Exception as exc:
        report["shutdown_succeeded"] = False
        return exc
    report["shutdown_succeeded"] = True
    return None


def run_lumen_session(envelope_template: SessionEnvelope, outer_model_caller: Any,
                      hardware_decision_provider: Any, research_goal: str,
                      max_hardware_decisions: int,
                      human_decision_callback: Callable[[dict], Any]) -> dict:
    """Approve and run one fresh LUMEN session; always attempt host shutdown.

    Returns the report for a run that ended normally (completed experiment,
    finish without experiment, experiment not completed) or for a refused
    approval. Raises LumenDriverError (report attached, original error as
    __cause__) for any failure, including a failed shutdown.
    """
    _check_inputs(envelope_template, outer_model_caller, hardware_decision_provider,
                  research_goal, max_hardware_decisions, human_decision_callback)
    envelope = dataclasses.replace(envelope_template, session_id=new_session_id())
    report = _new_report(envelope)
    try:
        host = OfflineApprovedSessionHost(envelope)  # no backend call
    except Exception as exc:
        report["reason"] = f"session host could not be created ({type(exc).__name__})"
        raise LumenDriverError(report["reason"], _json_copy(report)) from exc

    failure: Exception | None = None
    interrupt: BaseException | None = None
    cleanup_error: Exception | None = None
    try:
        _approve_and_run(host, report, outer_model_caller, hardware_decision_provider,
                         research_goal, max_hardware_decisions, human_decision_callback)
    except Exception as exc:
        failure = exc
    except BaseException as exc:  # KeyboardInterrupt, SystemExit: propagate after cleanup
        interrupt = exc
        raise
    finally:
        cleanup_error = _shutdown_host(host, report)
        if cleanup_error is not None and interrupt is not None:
            interrupt.add_note("LUMEN host shutdown FAILED during unwinding: "
                               f"{type(cleanup_error).__name__}")

    if failure is not None:
        report["session_status"] = SESSION_FAILED
        report["reason"] = (f"session did not complete: {type(failure).__name__}; "
                            "nothing was retried")
    if cleanup_error is not None:
        report["session_status"] = SESSION_CLEANUP_FAILED
        report["reason"] = (f"host shutdown FAILED ({type(cleanup_error).__name__})"
                            + (f" after {type(failure).__name__}" if failure else ""))
        raise LumenDriverError(report["reason"], _json_copy(report),
                               cleanup_error) from (failure or cleanup_error)
    if failure is not None:
        raise LumenDriverError(report["reason"], _json_copy(report)) from failure
    return _json_copy(report)
