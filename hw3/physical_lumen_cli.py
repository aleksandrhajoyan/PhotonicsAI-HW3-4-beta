"""Manually authorized four-agent LUMEN run on the physical iRAYPLE MV-A7200MU130 (HW3.7-D6A).

    python hw3/physical_lumen_cli.py --physical-lumen-check --model <model-id>

The ORIGINAL HW1 four-role LangGraph (task6 AgenticSilverboxSystem.build(),
inherited unchanged by hw3 LumenAgentSystem) runs the dialogue:

    orchestrator -> scientist | general | executor -> orchestrator -> ... -> finish

The Orchestrator, Scientist and General are model calls with no tools. Only the
LumenAgentSystem Executor can reach the camera: on its first visit it calls the
existing bridge run_adaptive_experiment() once, which runs AdaptiveMCPExecutor
against the ONE approved PhysicalApprovedSessionHost, with the D5A
FirstRunDecisionGuard(BoundedDecisionProvider(LumenOpenAIModelCaller)) as its
decision provider. Every physical action is therefore

    hardware model decision -> FirstRunDecisionGuard -> authorized MCP server
    -> SessionGate -> IRaypleCamera

A second Executor visit is answered by LumenAgentSystem itself without any
dispatch; this CLI then stops the graph and reports FAIL.

Trust chain:
    1. --physical-lumen-check, an explicit --model, an interactive terminal and
       the typed phrase BEGIN FOUR AGENT HARDWARE CHECK; otherwise nothing is
       loaded, opened or requested.
    2. preload_vendor_sdk_once() on the main thread, before any event loop.
    3. A fresh envelope from physical_mcp_smoke.build_envelope (1000-5000 us,
       2 measurements, 7 tool calls, 180 s, four camera tools) and ONE
       PhysicalApprovedSessionHost holding exactly that envelope.
    4. Real LangGraph HITL approval: "APPROVE <session_id>"; host.resume()
       activates only if the approval graph AND a fresh SessionGate accept.
    5. Paid-API consent: "AUTHORIZE PAID API <session_id>". Only then is the
       OpenAI client created (create_live_openai_client: max_retries=0, 30 s).
    6. graph.stream() of the four-agent graph on this (main) thread; the
       Executor's asyncio.run() therefore also runs on the SDK-preloading thread.
    7. host.shutdown() on every path (by the Executor, the bridge, or the
       finally block here); never a second attempt after a failed cleanup.

Model budgets (separate LumenOpenAIModelCaller instances, same model and client):
    outer agents  OuterCallBudget: at most 16 calls (the HW1 graph allows at most
                  8 orchestrator decisions, each followed by at most one
                  Scientist/General call), bounded input and output size
    hardware      BoundedDecisionProvider: at most 7 calls (D5A limits)
No retries, no fallback model. Call limits are not a monetary spending cap.

Verdicts:
    PASS        physical D5A verdict PASS (two validated hardware frames,
                feedback-driven second exposure, camera_close, verified
                shutdown) AND orchestrator, scientist, general and executor all
                took part AND a Scientist/General conclusion follows the
                completed Executor turn
    INCOMPLETE  everything clean, but a role was not visited, the Executor
                was never dispatched, the hardware model finished early, or no
                post-measurement conclusion was written
    FAIL        anything else (including a requested second Executor dispatch)

Nothing is saved. A killed process cannot restore ExposureTime; then check it in
MV Viewer. Exit codes are those of physical_adaptive_cli.
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Any

import lumen_agent_system as las
import openai_model_caller as omc
import physical_adaptive_cli as adaptive
import physical_approved_session_host as phys
import physical_mcp_smoke as smoke
from approved_session_host import HOST_SHUT_DOWN
from executor_dialogue_bridge import LUMEN_COMPLETED_MARK
from llm_decision_provider import BoundedDecisionProvider
from lumen_session_driver import RECURSION_LIMIT
from session_policy import SessionEnvelope

CONFIRMATION_PHRASE = "BEGIN FOUR AGENT HARDWARE CHECK"

LUMEN_TASK = (
    "Coordinate a scientific lensless-camera experiment. Have the relevant scientific and "
    "general-purpose agents assess the objective and safety constraints. The Executor must use "
    "the approved physical MCP tools to obtain at most two real Mono8 measurements and adapt "
    "exposure using the first measurement. Aim for a mean near 70 DN without saturation. "
    "Produce a concise conclusion based only on measured digital numbers."
)
# The hardware decision model gets the D5A goal that was verified on the camera.
HARDWARE_GOAL = adaptive.RESEARCH_GOAL

ROLES = ("orchestrator", "scientist", "general", "executor")
MAX_OUTER_CALLS = 2 * las.MAX_DECISIONS  # 8 orchestrator decisions + <= 8 specialist replies
OUTER_MAX_INPUT_CHARS = 32000  # HW1 format_dialogue keeps the last 28000 characters
OUTER_MAX_OUTPUT_CHARS = 8000
OUTER_MAX_OUTPUT_TOKENS = 2048
MAX_TOTAL_API_CALLS = MAX_OUTER_CALLS + adaptive.MAX_MODEL_DECISIONS

EXIT_PASS = adaptive.EXIT_PASS
EXIT_FAILED = adaptive.EXIT_FAILED
EXIT_REFUSED = adaptive.EXIT_REFUSED
EXIT_NOT_INTERACTIVE = adaptive.EXIT_NOT_INTERACTIVE
EXIT_APPROVAL_REFUSED = adaptive.EXIT_APPROVAL_REFUSED
EXIT_INCOMPLETE = adaptive.EXIT_INCOMPLETE
EXIT_CLEANUP_FAILED = adaptive.EXIT_CLEANUP_FAILED
EXIT_API_CONSENT_REFUSED = adaptive.EXIT_API_CONSENT_REFUSED
EXIT_INTERRUPTED = adaptive.EXIT_INTERRUPTED

VERDICT_PASS = adaptive.VERDICT_PASS
VERDICT_INCOMPLETE = adaptive.VERDICT_INCOMPLETE
VERDICT_FAIL = adaptive.VERDICT_FAIL

_ROLE_OF_INSTRUCTIONS = {
    las.ORCHESTRATOR_INSTRUCTIONS: "orchestrator",
    las.SCIENTIST_INSTRUCTIONS: "scientist",
    las.GENERAL_INSTRUCTIONS: "general",
}


# -- outer-agent call budget ----------------------------------------------------------

class OuterCallError(Exception):
    """An outer-agent model call refused by OuterCallBudget."""


class OuterCallBudget:
    """ModelCaller for the outer roles: a hard call cap and bounded request/response size."""

    def __init__(self, model_caller: Any, *, max_calls: int, max_input_chars: int,
                 max_output_chars: int):
        if not callable(getattr(model_caller, "call", None)):
            raise TypeError("model_caller must have a call(instructions, input_text) method")
        self._caller = model_caller
        self._max_calls = max_calls
        self._max_input_chars = max_input_chars
        self._max_output_chars = max_output_chars
        self.calls: list[str] = []  # role of every attempted call, in order

    def call(self, instructions: str, input_text: str) -> str:
        if len(self.calls) >= self._max_calls:
            raise OuterCallError(f"outer-agent call limit of {self._max_calls} reached")
        size = len(instructions) + len(input_text)
        if size > self._max_input_chars:
            raise OuterCallError(f"outer request has {size} characters, limit is {self._max_input_chars}")
        self.calls.append(_ROLE_OF_INSTRUCTIONS.get(instructions, "unknown"))
        output = self._caller.call(instructions, input_text)  # exactly once; errors propagate
        if not isinstance(output, str) or len(output) > self._max_output_chars:
            raise OuterCallError(f"outer reply is not a str of at most {self._max_output_chars} characters")
        return output


# -- report ---------------------------------------------------------------------------

class LumenReport:
    """The outer four-agent record plus the D5A physical report, built only from real results."""

    def __init__(self, envelope: SessionEnvelope, model: str):
        self.physical = adaptive.Report(envelope, model)
        self.dialogue: list[dict] = []
        self.outer_calls: list[str] = []
        self.dispatches: list[str] = []  # LumenAgentSystem experiment_records outcomes
        self.duplicate_dispatch = False
        self.graph_error: str | None = None

    @property
    def visits(self) -> list[str]:
        return [turn["role"] for turn in self.dialogue if turn.get("role") != "user"]

    def missing_roles(self) -> list[str]:
        visited = set(self.visits)
        missing = [role for role in ROLES if role not in visited]
        if "result" not in self.dispatches and "executor" not in missing:
            missing.append("executor (no physical dispatch)")
        return missing

    def conclusion(self) -> dict | None:
        """The first Scientist/General turn after the completed Executor turn, if any."""
        done = next((i for i, turn in enumerate(self.dialogue) if turn.get("role") == "executor"
                     and turn.get("text", "").startswith(LUMEN_COMPLETED_MARK)), None)
        if done is None:
            return None
        return next((turn for turn in self.dialogue[done + 1:]
                     if turn.get("role") in ("scientist", "general")), None)


def cites_measurements(report: LumenReport, text: str) -> bool:
    """Informational heuristic only: does the text quote a measured mean DN value?"""
    for event in report.physical.successful("camera_measure"):
        mean = event["result"]["metrics"]["mean_dn"]
        if any(form in text for form in (f"{mean:.1f}", f"{mean:.2f}", f"{mean:.4f}")):
            return True
    return False


def verdict(report: LumenReport) -> tuple[str, str]:
    physical = report.physical
    dispatched = bool(report.dispatches)
    if physical.cleanup_error is not None or not physical.shutdown_ok \
            or physical.owns_device_after is not False:
        return VERDICT_FAIL, "trusted shutdown not verified"
    if physical.interrupted:
        return VERDICT_FAIL, "interrupted"
    if physical.approval_status != adaptive.APPROVAL_APPROVED:
        return VERDICT_FAIL, "hardware session not approved"
    if physical.api_consent != adaptive.CONSENT_GIVEN:
        return VERDICT_FAIL, "paid-API use not authorized"
    if report.duplicate_dispatch:
        return VERDICT_FAIL, "a second Executor dispatch was requested and rejected; run stopped"
    if report.graph_error:
        return VERDICT_FAIL, report.graph_error
    if physical.experiment_error:
        return VERDICT_FAIL, physical.experiment_error
    if not dispatched:
        return VERDICT_INCOMPLETE, "the orchestrator never dispatched the Executor"
    result, reason = adaptive.verdict(physical)
    if result != VERDICT_PASS:
        return result, f"physical experiment {result}: {reason}"
    missing = report.missing_roles()
    if missing:
        return VERDICT_INCOMPLETE, f"four-agent demonstration incomplete: not visited {missing}"
    if report.conclusion() is None:
        return VERDICT_INCOMPLETE, "no Scientist/General conclusion after the measurements"
    return VERDICT_PASS, ("four roles took part; two hardware measurements with a "
                          "feedback-driven second exposure; verified shutdown")


def exit_code_for(report: LumenReport) -> int:
    result, _ = verdict(report)
    physical = report.physical
    if result == VERDICT_PASS:
        return EXIT_PASS
    if physical.cleanup_error is not None or not physical.shutdown_ok \
            or physical.owns_device_after is not False:
        return EXIT_CLEANUP_FAILED
    if physical.interrupted:
        return EXIT_INTERRUPTED
    if physical.approval_status == adaptive.APPROVAL_REFUSED:
        return EXIT_APPROVAL_REFUSED
    if physical.api_consent == adaptive.CONSENT_REFUSED:
        return EXIT_API_CONSENT_REFUSED
    if result == VERDICT_INCOMPLETE:
        return EXIT_INCOMPLETE
    return EXIT_FAILED


def format_report(report: LumenReport) -> list[str]:
    physical = report.physical
    lines = [
        "=" * 76,
        "LUMEN FOUR-AGENT PHYSICAL CHECK REPORT",
        "=" * 76,
        f"session_id          : {physical.session_id}",
        f"approval            : {physical.approval_status}",
        f"paid-API consent    : {physical.api_consent}",
        f"model               : {physical.model}",
        f"outer API requests  : {len(report.outer_calls)} (limit {MAX_OUTER_CALLS}): "
        + (", ".join(report.outer_calls) or "none"),
        f"role visits (order) : {' -> '.join(report.visits) or 'none'}",
        f"orchestrator turns  : {report.visits.count('orchestrator')} (limit {las.MAX_DECISIONS})",
        f"executor dispatches : {report.dispatches or 'none'}",
    ]
    missing = report.missing_roles()
    lines.append("four roles          : " + ("all took part" if not missing else f"MISSING {missing}"))
    conclusion = report.conclusion()
    if conclusion is not None:
        cited = "quotes a measured mean" if cites_measurements(report, conclusion["text"]) \
            else "does not quote a measured mean"
        lines.append(f"conclusion          : by {conclusion['role']} after the Executor turn "
                     f"({cited}; heuristic, informational)")
    else:
        lines.append("conclusion          : none after a completed Executor turn")
    if report.duplicate_dispatch:
        lines.append("DUPLICATE_DISPATCH  : a second Executor visit was rejected without device "
                     "access; the graph was stopped")
    if report.graph_error:
        lines.append(f"GRAPH_ERROR         : {report.graph_error}")
    lines.append("-" * 76)
    lines.append("PHYSICAL EXECUTOR (hardware model decisions; separate from the outer dialogue):")
    if report.dispatches:
        lines.extend(adaptive.format_report(physical)[3:-1])
    else:
        lines.append("  not dispatched")
        lines.append(f"host shutdown       : {'OK' if physical.shutdown_ok else 'FAILED'} "
                     f"by {physical.shutdown_by}")
        lines.append("device ownership    : "
                     + ("released" if physical.owns_device_after is False else "STILL CLAIMED"))
        if physical.cleanup_error:
            lines.append(f"CLEANUP_ERROR       : {physical.cleanup_error}")
            lines.append(adaptive.UNCERTAIN_STATE_ADVICE)
    result, reason = verdict(report)
    lines.append("=" * 76)
    lines.append(f"PHYSICAL_LUMEN_CHECK: {result} ({reason})")
    lines.append("=" * 76)
    return lines


# -- human interaction ----------------------------------------------------------------

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
    print(f"If approved, the four-agent LUMEN graph runs with the model {model!r}.")
    print("Only the Executor can reach the camera, once, through the authorized MCP server;")
    print("each tool call is chosen by the model after seeing the earlier results.")
    print(f"Trusted first-run restriction: exposure_us only "
          f"{' or '.join(repr(v) for v in adaptive.EXPOSURE_CANDIDATES_US)}; "
          "open -> (set -> measure) x <= 2 -> close.")
    print(f"Task         : {LUMEN_TASK}")
    print(f"Executor goal: {HARDWARE_GOAL}")
    print("The camera's original ExposureTime is restored and verified at close.")
    print("-" * 76)


def _ask_api_consent(session_id: str, model: str) -> bool:
    phrase = adaptive.API_CONSENT_PREFIX + session_id
    print("-" * 76)
    print("PAID MODEL API USE (separate from the hardware approval)")
    print(f"  model             : {model} (OpenAI Responses API; availability not verified)")
    print(f"  outer agents      : at most {MAX_OUTER_CALLS} requests, <= {OUTER_MAX_OUTPUT_TOKENS} "
          "output tokens each")
    print(f"  hardware Executor : at most {adaptive.MAX_MODEL_DECISIONS} requests, <= "
          f"{adaptive.MAX_OUTPUT_TOKENS} output tokens each")
    print(f"  total             : at most {MAX_TOTAL_API_CALLS} requests; no automatic retries, "
          "no fallback model; not a monetary spending cap")
    print("  Billed by the API provider. Credentials are read by the OpenAI SDK and never shown.")
    print("  The approved session clock (180 s) is already running.")
    print(f"To authorize, type exactly:  {phrase}")
    print(f"To refuse, type:             {adaptive.DENY}")
    print(f"(answer within {adaptive.API_CONSENT_TIMEOUT_SECONDS:g} s; there is no second prompt)")
    line = adaptive._timed_answer(adaptive.API_CONSENT_TIMEOUT_SECONDS)
    if line == phrase:
        print("Paid-API use authorized for this session.")
        return True
    if line is not None:
        print("REFUSED: paid-API use was not authorized (no retry).")
    return False


# -- orchestration --------------------------------------------------------------------

def _run_graph(system: las.LumenAgentSystem, report: LumenReport) -> None:
    graph = system.build()  # the inherited HW1 topology, unchanged
    state = {"dialogue": [{"role": "user", "text": LUMEN_TASK}]}
    try:
        for values in graph.stream(state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="values"):
            report.dialogue = list(values["dialogue"])
            report.dispatches = [record["outcome"] for record in system.experiment_records]
            if "host_consumed" in report.dispatches:
                report.duplicate_dispatch = True  # rejected by LumenAgentSystem; stop here
                return
    finally:
        report.dispatches = [record["outcome"] for record in system.experiment_records]
        _record_experiment(system, report)


def _record_experiment(system: las.LumenAgentSystem, report: LumenReport) -> None:
    physical = report.physical
    physical.result = system.latest_experiment_result
    for record in system.experiment_records:
        if record["outcome"] in ("bridge_error", "format_error"):
            error = record["error"]
            cause = error.__cause__ if error is not None and error.__cause__ else error
            physical.experiment_error = physical.experiment_error or (
                f"EXECUTOR_{record['outcome'].upper()}: {type(cause).__name__} (details withheld)")


def _approve_and_run(host: Any, envelope: SessionEnvelope, report: LumenReport) -> None:
    physical = report.physical
    request = host.start(envelope.session_id)  # the fresh session_id is the approval thread_id
    problem = adaptive._request_problem(request, host.pending(), envelope)
    if problem:
        physical.experiment_error = f"APPROVAL_REQUEST_INVALID: {problem}"
        return
    _show_request(request, envelope, physical.model)
    activation = host.resume(adaptive._ask_approval(request))
    physical.approval_reason = str(activation.get("reason"))
    if activation.get("activated") is not True:
        physical.approval_status = adaptive.APPROVAL_REFUSED
        return
    physical.approval_status = adaptive.APPROVAL_APPROVED

    if not _ask_api_consent(envelope.session_id, physical.model):
        physical.api_consent = adaptive.CONSENT_REFUSED
        return
    physical.api_consent = adaptive.CONSENT_GIVEN

    try:
        client = omc.create_live_openai_client()  # max_retries=0, timeout 30 s; no request yet
    except Exception as exc:
        physical.experiment_error = (f"MODEL_CLIENT_UNAVAILABLE: {type(exc).__name__} "
                                     "(details withheld; no API request was made, no camera call)")
        return
    outer_caller = omc.LumenOpenAIModelCaller(client, model=physical.model,
                                              reasoning_effort=adaptive.REASONING_EFFORT,
                                              max_output_tokens=OUTER_MAX_OUTPUT_TOKENS)
    outer = OuterCallBudget(outer_caller, max_calls=MAX_OUTER_CALLS,
                            max_input_chars=OUTER_MAX_INPUT_CHARS,
                            max_output_chars=OUTER_MAX_OUTPUT_CHARS)
    report.outer_calls = outer.calls
    hardware_caller = omc.LumenOpenAIModelCaller(client, model=physical.model,
                                                 reasoning_effort=adaptive.REASONING_EFFORT,
                                                 max_output_tokens=adaptive.MAX_OUTPUT_TOKENS)
    provider = BoundedDecisionProvider(hardware_caller, max_calls=adaptive.MAX_MODEL_DECISIONS,
                                       max_input_chars=adaptive.MAX_INPUT_CHARS,
                                       max_output_chars=adaptive.MAX_OUTPUT_CHARS,
                                       max_recent_events=adaptive.MAX_RECENT_EVENTS)
    guard = adaptive.FirstRunDecisionGuard(provider, envelope)
    physical.records = guard.records
    system = las.LumenAgentSystem(outer, experiment_host=host, decision_provider=guard,
                                  research_goal=HARDWARE_GOAL,
                                  max_hardware_decisions=adaptive.MAX_MODEL_DECISIONS)
    try:
        _run_graph(system, report)
    except las.RoutingError as exc:
        report.graph_error = f"ROUTING_ERROR: {str(exc)[:200]}"
    except OuterCallError as exc:
        report.graph_error = f"OUTER_MODEL_REFUSED: {exc}"
    except omc.ModelCallerError as exc:
        report.graph_error = f"OUTER_MODEL_ERROR: {type(exc).__name__}: {exc}"
    finally:
        physical.api_requests = hardware_caller.calls_made


def _shutdown(host: Any, report: LumenReport) -> None:
    """Trusted shutdown unless the Executor/bridge already performed it; never a second attempt."""
    physical = report.physical
    try:
        status = host.status()
        physical.session_state_before_shutdown = status["session_state"]
        already = status["host_state"] == HOST_SHUT_DOWN
    except Exception:
        already = False
    if already:
        physical.shutdown_by = "AdaptiveMCPExecutor" if physical.result else "executor bridge"
        executor_shutdown = physical.result["host_shutdown"] if physical.result else None
        if executor_shutdown and executor_shutdown.get("error"):
            physical.cleanup_error = executor_shutdown["error"]
        else:
            physical.shutdown_ok = True
    else:
        physical.shutdown_by = "trusted CLI"
        try:
            host.shutdown()
        except BaseException as exc:  # recorded, never retried
            if adaptive._is_interrupt(exc):
                physical.interrupted = True
            physical.cleanup_error = f"{type(exc).__name__}: {exc}"
        else:
            physical.shutdown_ok = True
    try:
        physical.owns_device_after = host.status()["owns_device"]
    except Exception:
        physical.owns_device_after = None
    if physical.owns_device_after is not False:
        physical.shutdown_ok = False
        physical.cleanup_error = physical.cleanup_error or "device ownership was not released"


# -- entry point ------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physical_lumen_cli",
        description="Manually approved four-agent LUMEN run on the physical iRAYPLE camera.")
    parser.add_argument("--physical-lumen-check", action="store_true",
                        help="required: run the four-agent check (three interactive confirmations follow)")
    parser.add_argument("--model", default=None,
                        help="required with the flag: the OpenAI model identifier (no default)")
    return parser


def _warning(model: str) -> list[str]:
    return [
        "WARNING: this program runs the four-agent LUMEN graph against a PHYSICAL camera",
        "(iRAYPLE MV-A7200MU130). Only its Executor can reach the camera, through the authorized",
        "MCP server; a language model chooses each tool call. ExposureTime may be set to",
        f"{' or '.join(repr(v) for v in adaptive.EXPOSURE_CANDIDATES_US)} us for at most two frames "
        "and is restored at close.",
        f"It makes PAID requests to the model {model!r} (separate consent follows).",
        "A forcibly killed process cannot restore ExposureTime.",
        "A separate approval of the exact session envelope follows this confirmation.",
    ]


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_REFUSED

    print("LUMEN four-agent physical check (HW3.7-D6A)")
    if not args.physical_lumen_check:
        print("NO_ACTION: pass --physical-lumen-check to run. No SDK was loaded, no camera was "
              "accessed and no model was called.")
        return EXIT_REFUSED
    model = args.model
    if not isinstance(model, str) or not model or model != model.strip() or not model.isprintable():
        print("REFUSED: an explicit --model <model-id> is required (there is no default). "
              "No SDK was loaded and no model was called.")
        return EXIT_REFUSED
    if not (adaptive._is_interactive(sys.stdin) and adaptive._is_interactive(sys.stdout)):
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
        print(f"PHYSICAL_LUMEN_CHECK: FAIL (SDK_LOAD_FAILED: {type(exc).__name__}: {exc}). "
              "No camera was opened.")
        return EXIT_FAILED

    envelope = smoke.build_envelope(smoke.new_session_id())
    report = LumenReport(envelope, model)
    try:
        host = phys.PhysicalApprovedSessionHost(envelope, sdk)  # no SDK call
    except Exception as exc:
        print(f"PHYSICAL_LUMEN_CHECK: FAIL (HOST_NOT_CREATED: {type(exc).__name__}: {exc}). "
              "No camera was opened.")
        return EXIT_FAILED

    escaped: BaseException | None = None
    try:
        if threading.get_ident() != bootstrap_thread:
            raise RuntimeError("bootstrap thread changed before MCP serving")
        _approve_and_run(host, envelope, report)
    except BaseException as exc:
        physical = report.physical
        if adaptive._is_interrupt(exc):
            physical.interrupted = True
            physical.experiment_error = physical.experiment_error or "interrupted before the run completed"
        elif isinstance(exc, Exception):
            physical.experiment_error = (physical.experiment_error
                                         or f"UNEXPECTED_ERROR: {adaptive._safe_error(exc)}")
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
