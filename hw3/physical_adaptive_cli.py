"""Manually authorized, LLM-driven adaptive MCP experiment on the physical iRAYPLE MV-A7200MU130 (HW3.7-D5A).

    python hw3/physical_adaptive_cli.py --physical-adaptive-check --model <model-id>

Unlike physical_mcp_smoke.py (a fixed six-tool script), every MCP tool call here
is chosen by a language model through the existing AdaptiveMCPExecutor, one
decision at a time, after it has seen the results of the earlier calls:

    LLM decision -> trusted first-run check -> authorized MCP server -> SessionGate
    -> IRaypleCamera -> measurement -> executor history -> next LLM decision

Trust chain:
    1. --physical-adaptive-check, an explicit --model, an interactive terminal and
       the typed phrase BEGIN ADAPTIVE HARDWARE CHECK. Without all of them nothing
       is loaded, opened or requested. This does NOT approve hardware access.
    2. preload_vendor_sdk_once() on the main thread, before any event loop exists.
    3. A fresh SessionEnvelope (physical_mcp_smoke.build_envelope: secrets-based
       session_id, 1000-5000 us, 2 measurements, 7 tool calls, 180 s, four camera
       tools) and ONE PhysicalApprovedSessionHost holding exactly that envelope.
    4. host.start() pauses the real LangGraph approval graph; the pending request
       is shown and the human must type exactly "APPROVE <session_id>".
       host.resume() activates only if the graph AND a fresh SessionGate accept.
    5. Paid-API consent: the human must type exactly "AUTHORIZE PAID API
       <session_id>". Only then is the OpenAI client created (by the existing
       create_live_openai_client: max_retries=0, 30 s timeout; the SDK reads its own
       credentials, this module never reads or prints them).
    6. asyncio.run(AdaptiveMCPExecutor(host, FirstRunDecisionGuard(
       BoundedDecisionProvider(LumenOpenAIModelCaller(client, model=...))), ...).run()).
    7. host.shutdown() is always reached: by the executor itself and, if the
       executor never ran or did not get that far, by the finally block here.

FirstRunDecisionGuard is trusted local code between the executor and the model.
It never invents or edits a decision. It forwards the model's reply unchanged
when it passes the first-run protocol, and otherwise raises ProtocolRefusal,
which the executor records as a provider error and stops BEFORE any MCP call:
    - only the envelope's tools (camera_info is refused);
    - camera_open first, once; nothing but finish after camera_close;
    - camera_set_exposure only with exposure_us 1000.0 or 5000.0 (a further
      restriction inside the approved 1000-5000 us envelope), and only when no
      set exposure is waiting to be measured;
    - camera_measure only directly after a new exposure, at most 2 in total.
Hence the second exposure decision can only be made after measurement 1 is in
the executor history. Malformed replies are passed through unchanged so that
the executor's own parse_decision rejects them. SessionGate still enforces the
approved envelope on every call independently.

Verdicts:
    PASS        two validated hardware measurements (frame_id 1, 2), the second
                exposure chosen by a model decision made after measurement 1,
                camera_close by the model, finish, verified shutdown
    INCOMPLETE  the model finished cleanly but before two measurements or
                without camera_close (the trusted shutdown still closed it)
    FAIL        anything else

Nothing is saved: no image, raw data, log or report file. A killed process
cannot restore ExposureTime; then check it in MV Viewer.

Exit codes:
    0    PASS
    1    FAIL (SDK load, approval graph, model client, model, tool, validation)
    2    no --physical-adaptive-check / --model, usage error or first confirmation refused
    3    not an interactive terminal: nothing was loaded
    4    hardware approval refused, malformed or expired: no camera call, no API request
    5    INCOMPLETE
    6    cleanup failure: restoration/close not verified, camera state uncertain
    7    paid-API consent refused or expired: no API request, no camera call
    130  interrupted
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import threading
import time
from typing import Any

import openai_model_caller as omc
import physical_approved_session_host as phys
import physical_mcp_smoke as smoke
from approved_session_host import HOST_SHUT_DOWN
from executor_mcp_loop import (
    ACTION_CALL_TOOL,
    ACTION_FINISH,
    STATUS_COMPLETED,
    STATUS_FINISHED_WITHOUT_MEASUREMENT,
    STATUS_PROVIDER_ERROR,
    AdaptiveMCPExecutor,
    DecisionError,
    parse_decision,
)
from llm_decision_provider import BoundedDecisionProvider, BoundedProviderError
from session_policy import EXPOSURE_TOOL, MEASUREMENT_TOOL, SessionEnvelope, session_approval

CONFIRMATION_PHRASE = "BEGIN ADAPTIVE HARDWARE CHECK"
APPROVE_PREFIX = "APPROVE "
API_CONSENT_PREFIX = "AUTHORIZE PAID API "
DENY = "DENY"
APPROVAL_TIMEOUT_SECONDS = 120.0
API_CONSENT_TIMEOUT_SECONDS = 60.0

# First-run restriction inside the approved envelope: the two physically verified values.
EXPOSURE_CANDIDATES_US = (1000.0, 5000.0)

# Model budget: open + 2 x (set, measure) + close + finish = 7 decisions, no slack.
MAX_MODEL_DECISIONS = 7
MAX_INPUT_CHARS = 16000
MAX_OUTPUT_CHARS = 1000
MAX_RECENT_EVENTS = 14  # 7 decisions + 6 tool calls: the whole history stays visible
MAX_OUTPUT_TOKENS = 1024
REASONING_EFFORT = omc.DEFAULT_REASONING_EFFORT
SHOWN_OUTPUT_CHARS = 160  # how much of a rejected model reply is echoed

RESEARCH_GOAL = (
    "Use a lensless monochrome camera to obtain at most two physical measurements under "
    "constant ambient illumination. Aim for a mean digital signal near 70 DN without "
    "saturation. Choose exposure based on earlier measured statistics. Report raw digital "
    "numbers only."
)

FIRST_RUN_RULES = """
Trusted first-run restrictions (checked locally before any MCP call; a violation ends the run):
- Use only camera_open, camera_set_exposure, camera_measure, camera_close. camera_info is not allowed.
- camera_set_exposure: exposure_us must be exactly 1000.0 or 5000.0; no other value is accepted.
- Order: camera_open once; then for each measurement exactly one camera_set_exposure followed by
  one camera_measure; at most 2 camera_measure calls; then camera_close; then finish.
- Choose each exposure from the measured statistics already received (latest_measurement, history).
- Finishing before two measurements ends the experiment as INCOMPLETE.
"""

EXIT_PASS = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_NOT_INTERACTIVE = 3
EXIT_APPROVAL_REFUSED = 4
EXIT_INCOMPLETE = 5
EXIT_CLEANUP_FAILED = 6
EXIT_API_CONSENT_REFUSED = 7
EXIT_INTERRUPTED = 130

APPROVAL_NOT_REQUESTED = "not_requested"
APPROVAL_APPROVED = "approved"
APPROVAL_REFUSED = "refused"
CONSENT_NOT_REQUESTED = "not_requested"
CONSENT_GIVEN = "given"
CONSENT_REFUSED = "refused"

VERDICT_PASS = "PASS"
VERDICT_INCOMPLETE = "INCOMPLETE"
VERDICT_FAIL = "FAIL"

CHECK_PASSED = "passed"

UNCERTAIN_STATE_ADVICE = (
    "CAMERA STATE UNCERTAIN: close this Python process, open MV Viewer and inspect "
    "ExposureTime manually before any further use. Nothing will be retried."
)
SCOPE_NOTE = (
    "Uncalibrated frames, raw DN only. Not a linearity test and not a calibration. "
    "No image, raw data, log or report was saved."
)

_clock = time.monotonic  # prompt-timeout clock; replaced only by offline tests


# -- trusted first-run protocol ---------------------------------------------------------

class ProtocolRefusal(Exception):
    """A model decision refused by the trusted first-run protocol; nothing was sent to MCP."""


def protocol_state(input_text: str) -> dict:
    """What has really happened so far, from the executor's own event history."""
    try:
        context = json.loads(input_text)
        history = context["history"]
        latest = context["latest_measurement"]
    except (TypeError, ValueError, KeyError) as exc:
        raise ProtocolRefusal(f"executor context is unreadable: {type(exc).__name__}") from exc
    done = [event for event in history
            if isinstance(event, dict) and event.get("type") == "tool_call"
            and event.get("status") == "success"]
    tools = [event.get("tool") for event in done]
    return {
        "opened": "camera_open" in tools,
        "closed": "camera_close" in tools,
        "exposures_set": tools.count(EXPOSURE_TOOL),
        "measurements": tools.count(MEASUREMENT_TOOL),
        "measurement_frame_ids": [event.get("frame_id") for event in done
                                  if event.get("tool") == MEASUREMENT_TOOL],
        "latest_frame_id": latest.get("frame_id") if isinstance(latest, dict) else None,
        "latest_mean_dn": (latest.get("metrics", {}).get("mean_dn")
                           if isinstance(latest, dict) and isinstance(latest.get("metrics"), dict)
                           else None),
    }


def protocol_problem(decision: dict, state: dict, envelope: SessionEnvelope) -> str:
    """Why a parsed decision may not be forwarded ("" if it may)."""
    if decision["action"] == ACTION_FINISH:
        return ""
    tool, arguments = decision["tool"], decision["arguments"]
    if tool not in envelope.allowed_tools:
        return f"{tool} is not in the approved envelope"
    if state["closed"]:
        return f"{tool} after camera_close; only finish is possible"
    if tool == "camera_open":
        return "camera_open was already done" if state["opened"] else ""
    if not state["opened"]:
        return f"{tool} before camera_open"
    if tool == EXPOSURE_TOOL:
        value = arguments["exposure_us"]
        if value not in EXPOSURE_CANDIDATES_US:
            return (f"exposure_us {value!r} is not a first-run candidate "
                    f"{list(EXPOSURE_CANDIDATES_US)}")
        if not envelope.min_exposure_us <= value <= envelope.max_exposure_us:
            return f"exposure_us {value!r} is outside the approved envelope"
        if state["measurements"] >= envelope.max_measurements:
            return "no measurement left for a new exposure"
        if state["exposures_set"] > state["measurements"]:
            return "the previously set exposure has not been measured yet"
        return ""
    if tool == MEASUREMENT_TOOL:
        if state["measurements"] >= envelope.max_measurements:
            return f"measurement {state['measurements'] + 1} exceeds the limit of {envelope.max_measurements}"
        if state["exposures_set"] != state["measurements"] + 1:
            return "camera_measure needs a newly set exposure first"
        return ""
    return ""  # camera_close while open


def _safe_error(exc: BaseException) -> str:
    """Own error texts are shown; third-party (API client) texts are not, only their type."""
    if isinstance(exc, (ProtocolRefusal, BoundedProviderError, omc.ModelCallerError)):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__} (details withheld)"


class FirstRunDecisionGuard:
    """Decision provider for AdaptiveMCPExecutor: the model decides, this only refuses.

    Records, per decision, what the model was shown (measurements so far), what it
    replied and what the trusted check concluded, separately from the executor log.
    """

    def __init__(self, provider: Any, envelope: SessionEnvelope):
        if not callable(getattr(provider, "call", None)):
            raise TypeError("provider must have a call(instructions, input_text) method")
        if not isinstance(envelope, SessionEnvelope):
            raise TypeError("envelope must be a SessionEnvelope")
        self._provider = provider
        self._envelope = envelope
        self.records: list[dict] = []

    def call(self, instructions: str, input_text: str) -> str:
        record = {"decision_number": len(self.records) + 1, "model_output": None,
                  "model_decision": None, "parse_error": None, "error": None,
                  "trusted_check": None}
        self.records.append(record)
        state = protocol_state(input_text)
        record["visible_measurement_frame_ids"] = list(state["measurement_frame_ids"])
        record["latest_frame_id"] = state["latest_frame_id"]
        record["latest_mean_dn"] = state["latest_mean_dn"]
        try:
            raw = self._provider.call(instructions + FIRST_RUN_RULES, input_text)
        except BaseException as exc:
            record["error"] = _safe_error(exc)
            raise
        record["model_output"] = raw
        try:
            decision = parse_decision(raw)
        except DecisionError as exc:
            record["parse_error"] = str(exc)
            record["trusted_check"] = "left to the executor (malformed decision)"
            return raw  # unchanged: parse_decision in the executor rejects it
        record["model_decision"] = decision
        problem = protocol_problem(decision, state, self._envelope)
        if problem:
            record["trusted_check"] = f"REFUSED: {problem}"
            refusal = ProtocolRefusal(f"first-run protocol refused the model decision: {problem}")
            record["error"] = _safe_error(refusal)
            raise refusal
        record["trusted_check"] = CHECK_PASSED
        return raw


# -- report ---------------------------------------------------------------------------

class Report:
    """What this run did and observed. Built only from real host, executor and guard results."""

    def __init__(self, envelope: SessionEnvelope, model: str):
        self.session_id = envelope.session_id
        self.envelope_sha256 = envelope.sha256()
        self.model = model
        self.approval_status = APPROVAL_NOT_REQUESTED
        self.approval_reason = ""
        self.api_consent = CONSENT_NOT_REQUESTED
        self.api_requests: int | None = None
        self.result: dict | None = None
        self.records: list[dict] = []
        self.experiment_error: str | None = None
        self.interrupted = False
        self.session_state_before_shutdown: str | None = None
        self.shutdown_by: str | None = None
        self.shutdown_ok = False
        self.owns_device_after: bool | None = None
        self.cleanup_error: str | None = None

    @property
    def tool_events(self) -> list[dict]:
        events = self.result["events"] if self.result else []
        return [event for event in events if event.get("type") == "tool_call"]

    def successful(self, tool: str) -> list[dict]:
        return [e for e in self.tool_events if e.get("tool") == tool and e.get("status") == "success"]


def validation_problem(report: Report) -> str:
    """Strict re-validation of every successful tool result ("" if all are valid)."""
    exposure: float | None = None
    frame_id = 1
    for event in report.tool_events:
        if event.get("status") != "success":
            continue
        tool, content = event["tool"], event.get("result")
        expected = exposure if tool == MEASUREMENT_TOOL else None
        problem = smoke.step_problem(tool, event["arguments"], content, report.session_id,
                                     expected, frame_id)
        if problem:
            return f"{tool} (event {event['sequence']}): {problem}"
        if tool == EXPOSURE_TOOL:
            exposure = content["exposure_us"]
        elif tool == MEASUREMENT_TOOL:
            frame_id += 1
    if report.result and report.result["provenance"]["sources_observed"] not in ([], ["hardware"]):
        return f"non-hardware sources observed: {report.result['provenance']['sources_observed']}"
    return ""


def feedback_evidence(report: Report) -> tuple[str, str]:
    """(problem, description) proving the model chose the next exposure after measurement 1."""
    if not report.result:
        return "no executor result", ""
    events = report.result["events"]
    measures = report.successful(MEASUREMENT_TOOL)
    if len(measures) < 2:
        return "fewer than two measurements", ""
    first, second = measures[0], measures[1]
    choice = next((e for e in events
                   if e.get("type") == "decision" and e.get("valid") is True
                   and first["sequence"] < e["sequence"] < second["sequence"]
                   and e["decision"].get("action") == ACTION_CALL_TOOL
                   and e["decision"].get("tool") == EXPOSURE_TOOL), None)
    if choice is None:
        return "no model exposure decision between measurement 1 and measurement 2", ""
    number = choice["decision_number"]
    record = next((r for r in report.records if r["decision_number"] == number), None)
    if record is None or record["latest_frame_id"] != first["frame_id"] \
            or type(record["latest_mean_dn"]) not in (int, float) \
            or not math.isfinite(record["latest_mean_dn"]) \
            or record["trusted_check"] != CHECK_PASSED:
        return f"decision {number} was not shown measurement {first['frame_id']}", ""
    return "", (f"decision {number} (event {choice['sequence']}) was requested after measurement "
                f"frame_id {first['frame_id']} (event {first['sequence']}, mean "
                f"{record['latest_mean_dn']:.4f} DN) was in its input and chose "
                f"camera_set_exposure({choice['decision']['arguments']['exposure_us']!r}); "
                f"measurement frame_id {second['frame_id']} followed (event {second['sequence']})")


def verdict(report: Report) -> tuple[str, str]:
    """(PASS | INCOMPLETE | FAIL, reason)."""
    if report.cleanup_error is not None or not report.shutdown_ok or report.owns_device_after is not False:
        return VERDICT_FAIL, "trusted shutdown not verified"
    if report.interrupted:
        return VERDICT_FAIL, "interrupted"
    if report.approval_status != APPROVAL_APPROVED:
        return VERDICT_FAIL, "hardware session not approved"
    if report.api_consent != CONSENT_GIVEN:
        return VERDICT_FAIL, "paid-API use not authorized"
    if report.experiment_error:
        return VERDICT_FAIL, report.experiment_error
    if report.result is None:
        return VERDICT_FAIL, "the executor did not run"
    status = report.result["status"]
    if status not in (STATUS_COMPLETED, STATUS_FINISHED_WITHOUT_MEASUREMENT):
        return VERDICT_FAIL, f"executor status {status}"
    problem = validation_problem(report)
    if problem:
        return VERDICT_FAIL, f"VALIDATION_FAILED: {problem}"
    if len(report.successful(MEASUREMENT_TOOL)) < 2:
        return VERDICT_INCOMPLETE, "the model finished before two measurements"
    problem, _ = feedback_evidence(report)
    if problem:
        return VERDICT_FAIL, f"adaptivity not demonstrated: {problem}"
    if not report.successful("camera_close"):
        return VERDICT_INCOMPLETE, "the model finished without camera_close (trusted shutdown closed it)"
    return VERDICT_PASS, "two hardware measurements, feedback-driven second exposure, verified shutdown"


def exit_code_for(report: Report) -> int:
    result, _ = verdict(report)
    if result == VERDICT_PASS:
        return EXIT_PASS
    if report.cleanup_error is not None or not report.shutdown_ok or report.owns_device_after is not False:
        return EXIT_CLEANUP_FAILED
    if report.interrupted:
        return EXIT_INTERRUPTED
    if report.approval_status == APPROVAL_REFUSED:
        return EXIT_APPROVAL_REFUSED
    if report.api_consent == CONSENT_REFUSED:
        return EXIT_API_CONSENT_REFUSED
    if result == VERDICT_INCOMPLETE:
        return EXIT_INCOMPLETE
    return EXIT_FAILED


def _executor_reason(report: Report) -> str:
    """The executor's stop reason; a provider error is shown via the guard's safe text."""
    if report.result["status"] == STATUS_PROVIDER_ERROR:
        errors = [r["error"] for r in report.records if r["error"]]
        return errors[-1] if errors else "provider error (details withheld)"
    return report.result["final_reason"]


def _shown_output(raw: Any) -> str:
    text = raw if isinstance(raw, str) else repr(raw)
    return repr(text[:SHOWN_OUTPUT_CHARS] + ("..." if len(text) > SHOWN_OUTPUT_CHARS else ""))


def format_report(report: Report) -> list[str]:
    lines = [
        "=" * 76,
        "LUMEN PHYSICAL ADAPTIVE CHECK REPORT (model-chosen MCP actions)",
        "=" * 76,
        f"session_id          : {report.session_id}",
        f"envelope SHA-256    : {report.envelope_sha256}",
        f"approval            : {report.approval_status}"
        + (f" ({report.approval_reason})" if report.approval_reason else ""),
        f"paid-API consent    : {report.api_consent}",
        f"model               : {report.model}",
        f"API requests        : {report.api_requests if report.api_requests is not None else 0} "
        f"(limit {MAX_MODEL_DECISIONS}, no retries)",
    ]
    if report.records:
        lines.append("model decisions (model reply | trusted first-run check):")
    for record in report.records:
        seen = record["visible_measurement_frame_ids"]
        if record["model_decision"] is not None:
            decision = record["model_decision"]
            chosen = (f"finish({decision['reason']!r})" if decision["action"] == ACTION_FINISH else
                      f"{decision['tool']}(" + ", ".join(f"{k}={v!r}" for k, v in
                                                         decision["arguments"].items()) + ")")
        elif record["model_output"] is not None:
            chosen = f"MALFORMED {_shown_output(record['model_output'])}: {record['parse_error']}"
        else:
            chosen = f"NO REPLY: {record['error']}"
        lines.append(f"  {record['decision_number']}. saw measurements {seen} -> model: {chosen} "
                     f"| check: {record['trusted_check'] or 'n/a'}")
    events = report.tool_events
    lines.append(f"MCP tool calls      : {len(events)} issued (policy max "
                 f"{smoke.MAX_TOOL_CALLS})")
    for number, event in enumerate(events, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in event["arguments"].items())
        detail = f"  {number}. {event['tool']}({args}) -> {event['status']}"
        if event["tool"] == EXPOSURE_TOOL and event["status"] == "success":
            detail += f"; applied {event['result']['exposure_us']!r} us"
        if event.get("error"):
            detail += f": {event['error']}"
        lines.append(detail)
    measures = report.successful(MEASUREMENT_TOOL)
    lines.append(f"measurements        : {len(measures)} (policy max {smoke.MAX_MEASUREMENTS})")
    for number, event in enumerate(measures, 1):
        m = event["result"]
        metrics = m["metrics"]
        lines.append(
            f"  measurement {number}: source={m['source']} frame_id={m['frame_id']} "
            f"exposure={m['exposure_us']!r} us {m['width']}x{m['height']} Mono{m['bit_depth']} "
            f"min={metrics['min_dn']} max={metrics['max_dn']} DN mean={metrics['mean_dn']:.4f} DN "
            f"saturation_fraction={metrics['saturation_fraction']:.6f}")
    if len(measures) == 2:
        problem, description = feedback_evidence(report)
        lines.append("feedback evidence   : " + (description or f"MISSING ({problem})"))
    if report.result:
        lines.append(f"executor status     : {report.result['status']} ({_executor_reason(report)})")
    lines.append(f"session state       : {report.session_state_before_shutdown or 'not active'} "
                 f"before trusted shutdown")
    if report.shutdown_ok:
        lines.append(f"host shutdown       : OK by {report.shutdown_by} (native Close/DestroyHandle "
                     "not repeated; ExposureTime restored and verified if it was written)")
    else:
        lines.append("host shutdown       : " + ("FAILED" if report.shutdown_by else "NOT ATTEMPTED"))
    lines.append(f"device ownership    : {'released' if report.owns_device_after is False else 'STILL CLAIMED'}")
    if report.experiment_error:
        lines.append(f"EXPERIMENT_ERROR    : {report.experiment_error}")
    if report.interrupted:
        lines.append("INTERRUPTED         : KeyboardInterrupt")
    if report.cleanup_error:
        lines.append(f"CLEANUP_ERROR       : {report.cleanup_error}")
    if report.cleanup_error or not report.shutdown_ok:
        lines.append(UNCERTAIN_STATE_ADVICE)
    if measures:
        lines.append(SCOPE_NOTE)
    result, reason = verdict(report)
    lines.append(f"PHYSICAL_ADAPTIVE_CHECK: {result} ({reason})")
    lines.append("=" * 76)
    return lines


# -- human interaction --------------------------------------------------------------------

def _is_interactive(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _is_interrupt(exc: BaseException) -> bool:
    if isinstance(exc, KeyboardInterrupt):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_interrupt(inner) for inner in exc.exceptions)
    return False


def _timed_answer(timeout: float) -> str | None:
    """One input line, or None on EOF or when it arrives after the timeout."""
    shown_at = _clock()
    try:
        line = input("> ")
    except EOFError:
        print("\nREFUSED: no input (EOF).")
        return None
    elapsed = _clock() - shown_at
    if not (math.isfinite(elapsed) and 0 <= elapsed <= timeout):
        print("REFUSED: the answer arrived after the timeout (expired).")
        return None
    return line


def _request_problem(request: Any, pending: Any, envelope: SessionEnvelope) -> str:
    """The paused graph must be waiting on exactly this envelope."""
    if not isinstance(request, dict) or pending != request:
        return "the approval graph is not paused on the request it returned"
    if request.get("kind") != "hardware_session_approval":
        return "the pending request is not a hardware session approval"
    if request.get("session_id") != envelope.session_id:
        return "the pending request names a different session"
    if request.get("envelope_sha256") != envelope.sha256():
        return "the pending request's SHA-256 does not match the trusted envelope"
    if request.get("envelope") != envelope.canonical_dict():
        return "the pending request's envelope differs from the trusted envelope"
    return ""


def _show_request(request: dict, envelope: SessionEnvelope, model: str) -> None:
    body = request["envelope"]
    print("=" * 76)
    print("LUMEN PHYSICAL HARDWARE SESSION APPROVAL REQUEST (pending in LangGraph)")
    print("=" * 76)
    print(f"session_id          : {request['session_id']}")
    print(f"envelope SHA-256    : {request['envelope_sha256']}")
    print(f"device              : {body['device_label']}")
    print(f"allowed tools       : {', '.join(body['allowed_tools'])}")
    print(f"exposure bounds (us): [{body['min_exposure_us']!r}, {body['max_exposure_us']!r}]")
    print(f"max measurements    : {body['max_measurements']}")
    print(f"max tool calls      : {body['max_tool_calls']}")
    print(f"max duration (s)    : {body['max_duration_seconds']!r}")
    print("canonical envelope JSON (exactly what the SHA-256 covers):")
    print(f"  {envelope.canonical_json()}")
    print("-" * 76)
    print(f"If approved, the language model {model!r} chooses every MCP tool call, one at a")
    print("time, after seeing the earlier results. Trusted local first-run restriction:")
    print(f"exposure_us only {' or '.join(repr(v) for v in EXPOSURE_CANDIDATES_US)}; open -> "
          "(set -> measure) x <= 2 -> close.")
    print(f"Research goal: {RESEARCH_GOAL}")
    print("The camera's original ExposureTime is restored and verified at close.")
    print("-" * 76)


def _ask_approval(request: dict) -> dict:
    """One prompt; approved only if the exact phrase arrives within the timeout."""
    session_id, digest = request["session_id"], request["envelope_sha256"]
    phrase = APPROVE_PREFIX + session_id
    print(f"To approve, type exactly:  {phrase}")
    print(f"To refuse, type:           {DENY}")
    print(f"(answer within {APPROVAL_TIMEOUT_SECONDS:g} s; there is no second prompt)")
    line = _timed_answer(APPROVAL_TIMEOUT_SECONDS)
    approved = line == phrase
    if approved:
        print("Approval phrase received; submitting it to the approval graph.")
    elif line == DENY:
        print("REFUSED by the human operator.")
    elif line is not None:
        print("REFUSED: input did not match the approval phrase (no retry).")
    return session_approval(session_id, digest, approved)


def _ask_api_consent(session_id: str, model: str) -> bool:
    phrase = API_CONSENT_PREFIX + session_id
    print("-" * 76)
    print("PAID MODEL API USE (separate from the hardware approval)")
    print(f"  model             : {model} (OpenAI Responses API; availability not verified)")
    print(f"  requests          : at most {MAX_MODEL_DECISIONS}, no automatic retries, no fallback model")
    print(f"  per request       : <= {MAX_INPUT_CHARS} input characters, <= {MAX_OUTPUT_TOKENS} "
          f"output tokens, reasoning effort {REASONING_EFFORT!r}")
    print("  Billed by the API provider. Credentials are read by the OpenAI SDK and never shown.")
    print("  The approved session clock (180 s) is already running.")
    print(f"To authorize, type exactly:  {phrase}")
    print(f"To refuse, type:             {DENY}")
    print(f"(answer within {API_CONSENT_TIMEOUT_SECONDS:g} s; there is no second prompt)")
    line = _timed_answer(API_CONSENT_TIMEOUT_SECONDS)
    if line == phrase:
        print("Paid-API use authorized for this session.")
        return True
    if line is not None:
        print("REFUSED: paid-API use was not authorized (no retry).")
    return False


# -- orchestration ----------------------------------------------------------------------

def _approve_and_run(host: Any, envelope: SessionEnvelope, report: Report) -> None:
    request = host.start(envelope.session_id)  # the fresh session_id is the approval thread_id
    problem = _request_problem(request, host.pending(), envelope)
    if problem:
        report.experiment_error = f"APPROVAL_REQUEST_INVALID: {problem}"
        return
    _show_request(request, envelope, report.model)
    activation = host.resume(_ask_approval(request))
    report.approval_reason = str(activation.get("reason"))
    if activation.get("activated") is not True:
        report.approval_status = APPROVAL_REFUSED
        return
    report.approval_status = APPROVAL_APPROVED

    if not _ask_api_consent(envelope.session_id, report.model):
        report.api_consent = CONSENT_REFUSED
        return
    report.api_consent = CONSENT_GIVEN

    try:
        client = omc.create_live_openai_client()  # max_retries=0, timeout 30 s; no request yet
    except Exception as exc:
        report.experiment_error = (f"MODEL_CLIENT_UNAVAILABLE: {type(exc).__name__} "
                                   "(details withheld; no API request was made, no camera call)")
        return
    caller = omc.LumenOpenAIModelCaller(client, model=report.model,
                                        reasoning_effort=REASONING_EFFORT,
                                        max_output_tokens=MAX_OUTPUT_TOKENS)
    provider = BoundedDecisionProvider(caller, max_calls=MAX_MODEL_DECISIONS,
                                       max_input_chars=MAX_INPUT_CHARS,
                                       max_output_chars=MAX_OUTPUT_CHARS,
                                       max_recent_events=MAX_RECENT_EVENTS)
    guard = FirstRunDecisionGuard(provider, envelope)
    report.records = guard.records
    executor = AdaptiveMCPExecutor(host, guard, RESEARCH_GOAL, MAX_MODEL_DECISIONS)
    try:
        # One event loop on this (the SDK-preloading) thread; tool handlers run on it.
        report.result = asyncio.run(executor.run())  # ends with host.shutdown()
    finally:
        report.api_requests = caller.calls_made


def _shutdown(host: Any, report: Report) -> None:
    """Trusted shutdown unless the executor already performed it; never a second attempt."""
    try:
        status = host.status()
        report.session_state_before_shutdown = status["session_state"]
        already = status["host_state"] == HOST_SHUT_DOWN
    except Exception:
        already = False
    if already:
        report.shutdown_by = "AdaptiveMCPExecutor"
        executor_shutdown = report.result["host_shutdown"] if report.result else None
        if executor_shutdown and executor_shutdown.get("error"):
            report.cleanup_error = executor_shutdown["error"]
        else:
            report.shutdown_ok = True
    else:
        report.shutdown_by = "trusted CLI"
        try:
            host.shutdown()
        except BaseException as exc:  # recorded, never retried; a second Ctrl+C must not hide it
            if _is_interrupt(exc):
                report.interrupted = True
            report.cleanup_error = f"{type(exc).__name__}: {exc}"
        else:
            report.shutdown_ok = True
    try:
        report.owns_device_after = host.status()["owns_device"]
    except Exception:
        report.owns_device_after = None
    if report.owns_device_after is not False:
        report.shutdown_ok = False
        report.cleanup_error = report.cleanup_error or "device ownership was not released"


# -- entry point --------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physical_adaptive_cli",
        description="Manually approved, LLM-driven adaptive MCP experiment on the physical iRAYPLE camera.")
    parser.add_argument("--physical-adaptive-check", action="store_true",
                        help="required: run the experiment (three interactive confirmations follow)")
    parser.add_argument("--model", default=None,
                        help="required with the flag: the OpenAI model identifier (no default)")
    return parser


def _warning(model: str) -> list[str]:
    return [
        "WARNING: this program controls a PHYSICAL camera (iRAYPLE MV-A7200MU130) through the",
        "authorized MCP server. A language model chooses each tool call; it may temporarily set",
        f"ExposureTime to {' or '.join(repr(v) for v in EXPOSURE_CANDIDATES_US)} us and capture",
        "at most two frames. The original ExposureTime is restored at close.",
        f"It makes PAID requests to the model {model!r} (separate consent follows).",
        "A forcibly killed process cannot restore ExposureTime.",
        "A separate approval of the exact session envelope follows this confirmation.",
    ]


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_REFUSED

    print("LUMEN physical adaptive check (HW3.7-D5A)")
    if not args.physical_adaptive_check:
        print("NO_ACTION: pass --physical-adaptive-check to run. No SDK was loaded, no camera was "
              "accessed and no model was called.")
        return EXIT_REFUSED
    model = args.model
    if not isinstance(model, str) or not model or model != model.strip() or not model.isprintable():
        print("REFUSED: an explicit --model <model-id> is required (there is no default). "
              "No SDK was loaded and no model was called.")
        return EXIT_REFUSED
    if not (_is_interactive(sys.stdin) and _is_interactive(sys.stdout)):
        print("REFUSED: an interactive terminal is required. No SDK was loaded, no camera was "
              "accessed and no model was called.")
        return EXIT_NOT_INTERACTIVE

    for line in _warning(model):
        print(line)
    try:
        response = input(f"Type {CONFIRMATION_PHRASE} to continue to session approval: ")
    except (EOFError, KeyboardInterrupt):
        print("\nREFUSED: no confirmation received. No SDK was loaded and nothing was requested.")
        return EXIT_REFUSED
    if response != CONFIRMATION_PHRASE:
        print("REFUSED: confirmation phrase did not match. No SDK was loaded and nothing was requested.")
        return EXIT_REFUSED
    print("Confirmation recorded. This does NOT approve hardware access or API use.")

    bootstrap_thread = threading.get_ident()
    try:
        sdk = phys.preload_vendor_sdk_once()  # main thread, before any event loop or MCP serving
    except KeyboardInterrupt:
        print("INTERRUPTED while loading the vendor SDK. No camera was opened.")
        return EXIT_INTERRUPTED
    except Exception as exc:
        print(f"PHYSICAL_ADAPTIVE_CHECK: FAIL (SDK_LOAD_FAILED: {type(exc).__name__}: {exc}). "
              "No camera was opened.")
        return EXIT_FAILED

    envelope = smoke.build_envelope(smoke.new_session_id())
    report = Report(envelope, model)
    try:
        host = phys.PhysicalApprovedSessionHost(envelope, sdk)  # no SDK call
    except Exception as exc:
        print(f"PHYSICAL_ADAPTIVE_CHECK: FAIL (HOST_NOT_CREATED: {type(exc).__name__}: {exc}). "
              "No camera was opened.")
        return EXIT_FAILED

    escaped: BaseException | None = None
    try:
        if threading.get_ident() != bootstrap_thread:
            raise RuntimeError("bootstrap thread changed before MCP serving")
        _approve_and_run(host, envelope, report)
    except BaseException as exc:
        if _is_interrupt(exc):
            report.interrupted = True
            report.experiment_error = report.experiment_error or "interrupted before the run completed"
        elif isinstance(exc, Exception):
            report.experiment_error = (report.experiment_error
                                       or f"UNEXPECTED_ERROR: {_safe_error(exc)}")
        else:
            escaped = exc  # SystemExit and the like: shut down first, then let it propagate
    finally:
        _shutdown(host, report)

    for line in format_report(report):
        print(line)
    if escaped is not None:
        raise escaped
    return exit_code_for(report)


if __name__ == "__main__":
    sys.exit(main())
