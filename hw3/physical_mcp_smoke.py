"""Manually authorized, fixed six-tool MCP diagnostic on the physical iRAYPLE MV-A7200MU130 (HW3.7-D4).

    python hw3/physical_mcp_smoke.py --physical-mcp-check

This is NOT an LLM-driven experiment. A fixed, transparent script issues exactly

    camera_open
    camera_set_exposure(exposure_us=1000.0)
    camera_measure
    camera_set_exposure(exposure_us=5000.0)
    camera_measure
    camera_close

through the official MCP Client against the authorized MCP server of an approved
PhysicalApprovedSessionHost. The CLI never calls IRaypleCamera or
AuthorizedCameraRuntime itself; every camera action goes Client -> MCP server ->
runtime.execute() -> SessionGate.begin_action() -> IRaypleCamera.

Trust chain:
    1. --physical-mcp-check, an interactive terminal and the typed phrase
       BEGIN PHYSICAL MCP CHECK. Without all three nothing is loaded or opened.
       This first confirmation does NOT approve hardware access.
    2. preload_vendor_sdk_once(): the vendor SDK is loaded once, on the main
       thread, before any event loop exists or any MCP tool is served.
    3. A fresh SessionEnvelope (secrets-based session_id, 1000-5000 us, 2
       measurements, 7 tool calls, 180 s, four camera tools; camera_info is not
       allowed) and a PhysicalApprovedSessionHost (no SDK call).
    4. host.start(session_id) pauses the real LangGraph approval graph. The
       pending request (canonical envelope JSON, session_id, SHA-256) is shown
       and the human must type exactly "APPROVE <session_id>" within
       APPROVAL_TIMEOUT_SECONDS. Anything else, EOF or a late answer is sent to
       host.resume() as a refusal. The decision's approved flag is only ever the
       result of comparing the typed line with that phrase.
    5. host.resume() activates only if the LangGraph thread AND a fresh
       execution-side SessionGate both accept the decision.
    6. asyncio.run() drives the six MCP calls. Each result must have
       is_error False and pass strict validation; the first failure stops the
       sequence (no retries, no fallback, no new session).
    7. host.shutdown() always runs in a finally block (success, refusal,
       tool/validation failure, expiry, KeyboardInterrupt, early return).

Execution context (checked offline by the tests): authorized_mcp tool handlers
are coroutines, which the MCP SDK awaits directly (only synchronous handlers go
to anyio.to_thread), and the in-memory Client runs the server in its own task
group. Every SDK call therefore happens on the thread that called asyncio.run(),
i.e. the main thread that preloaded the SDK, one call at a time. No thread is
started here and os.chdir() is never called while tools are served.

Nothing is saved: no image, raw data, log or report file. Two uncalibrated
frames are not evidence of linearity and no calibration is claimed. A killed
process cannot restore ExposureTime; then check it in MV Viewer.

Exit codes:
    0    PASS: six validated calls, verified shutdown
    1    experiment failure (SDK load, approval graph, tool, validation)
    2    no --physical-mcp-check, usage error or first confirmation refused
    3    not an interactive terminal: nothing was loaded
    4    hardware approval refused, malformed or expired: no camera call
    6    cleanup failure: restoration/close not verified, camera state uncertain
    130  interrupted
"""

from __future__ import annotations

import argparse
import asyncio
import math
import secrets
import sys
import threading
import time
from typing import Any

from mcp import Client

import physical_approved_session_host as phys
from session_policy import SessionEnvelope, session_approval

CONFIRMATION_PHRASE = "BEGIN PHYSICAL MCP CHECK"
APPROVE_PREFIX = "APPROVE "
DENY = "DENY"
APPROVAL_TIMEOUT_SECONDS = 120.0
CLIENT_READ_TIMEOUT_SECONDS = 30

SESSION_ID_PREFIX = "lumen-phys-"
SESSION_ID_RANDOM_BYTES = 16  # 128 bits from secrets.token_hex
DEVICE_LABEL = "iRAYPLE MV-A7200MU130 (physical camera, uncalibrated raw DN)"
ALLOWED_TOOLS = ("camera_open", "camera_set_exposure", "camera_measure", "camera_close")
MIN_EXPOSURE_US = 1000.0
MAX_EXPOSURE_US = 5000.0
MAX_MEASUREMENTS = 2
MAX_TOOL_CALLS = 7
MAX_DURATION_SECONDS = 180.0

SEQUENCE = (
    ("camera_open", {}),
    ("camera_set_exposure", {"exposure_us": 1000.0}),
    ("camera_measure", {}),
    ("camera_set_exposure", {"exposure_us": 5000.0}),
    ("camera_measure", {}),
    ("camera_close", {}),
)
TOOL_ARGUMENTS = {
    "camera_open": (),
    "camera_set_exposure": ("exposure_us",),
    "camera_measure": (),
    "camera_close": (),
}

SOURCE_HARDWARE = "hardware"
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1200
EXPECTED_BIT_DEPTH = 8
FULL_SCALE_DN = 255

EXIT_PASS = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_NOT_INTERACTIVE = 3
EXIT_APPROVAL_REFUSED = 4
EXIT_CLEANUP_FAILED = 6
EXIT_INTERRUPTED = 130

APPROVAL_NOT_REQUESTED = "not_requested"
APPROVAL_APPROVED = "approved"
APPROVAL_REFUSED = "refused"

STEP_SUCCESS = "success"
STEP_MCP_ERROR = "mcp_error"
STEP_TRANSPORT_ERROR = "transport_error"
STEP_INVALID = "invalid_response"

UNCERTAIN_STATE_ADVICE = (
    "CAMERA STATE UNCERTAIN: close this Python process, open MV Viewer and inspect "
    "ExposureTime manually before any further use. Nothing will be retried."
)
SCOPE_NOTE = (
    "Two uncalibrated frames, raw DN only. Not a linearity test and not a calibration. "
    "No image, raw data, log or report was saved."
)

_clock = time.monotonic  # approval-timeout clock; replaced only by offline tests


class Report:
    """What this run did and observed. Built only from real host and MCP results."""

    def __init__(self, envelope: SessionEnvelope):
        self.session_id = envelope.session_id
        self.envelope_sha256 = envelope.sha256()
        self.approval_status = APPROVAL_NOT_REQUESTED
        self.approval_reason = ""
        self.steps: list[dict] = []
        self.measurements: list[dict] = []
        self.experiment_error: str | None = None
        self.interrupted = False
        self.session_state_before_shutdown: str | None = None
        self.shutdown_attempted = False
        self.shutdown_ok = False
        self.final_session_state: str | None = None
        self.owns_device_after: bool | None = None
        self.cleanup_error: str | None = None

    @property
    def sequence_ok(self) -> bool:
        return (len(self.steps) == len(SEQUENCE)
                and all(step["status"] == STEP_SUCCESS for step in self.steps)
                and len(self.measurements) == MAX_MEASUREMENTS)

    @property
    def passed(self) -> bool:
        return (self.approval_status == APPROVAL_APPROVED
                and self.sequence_ok
                and self.experiment_error is None
                and not self.interrupted
                and self.shutdown_ok
                and self.cleanup_error is None
                and self.owns_device_after is False)


# -- trusted inputs -------------------------------------------------------------------

def new_session_id() -> str:
    """A fresh, unpredictable session identifier (128 random bits)."""
    return SESSION_ID_PREFIX + secrets.token_hex(SESSION_ID_RANDOM_BYTES)


def build_envelope(session_id: str) -> SessionEnvelope:
    """The fixed trusted envelope; nothing here comes from MCP or model text."""
    return SessionEnvelope(
        session_id=session_id,
        device_label=DEVICE_LABEL,
        allowed_tools=ALLOWED_TOOLS,
        min_exposure_us=MIN_EXPOSURE_US,
        max_exposure_us=MAX_EXPOSURE_US,
        max_measurements=MAX_MEASUREMENTS,
        max_tool_calls=MAX_TOOL_CALLS,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )


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


# -- human approval -------------------------------------------------------------------

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


def _show_request(request: dict, envelope: SessionEnvelope) -> None:
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
    print("If approved, this fixed script (not an AI) will issue these MCP tool calls:")
    for number, (tool, arguments) in enumerate(SEQUENCE, 1):
        args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
        print(f"  {number}. {tool}({args})")
    print("The camera's original ExposureTime is restored and verified at close.")
    print("-" * 76)


def _ask_decision(request: dict) -> dict:
    """One prompt; approved only if the exact phrase arrives within the timeout."""
    session_id = request["session_id"]
    digest = request["envelope_sha256"]
    phrase = APPROVE_PREFIX + session_id
    print(f"To approve, type exactly:  {phrase}")
    print(f"To refuse, type:           {DENY}")
    print(f"(answer within {APPROVAL_TIMEOUT_SECONDS:g} s; there is no second prompt)")
    shown_at = _clock()
    try:
        line = input("> ")
    except EOFError:
        print("\nREFUSED: no input (EOF). Nothing was approved.")
        return session_approval(session_id, digest, False)
    elapsed = _clock() - shown_at
    if not (math.isfinite(elapsed) and 0 <= elapsed <= APPROVAL_TIMEOUT_SECONDS):
        print("REFUSED: the approval answer arrived after the timeout (expired). Nothing was approved.")
        return session_approval(session_id, digest, False)
    approved = line == phrase
    if approved:
        print("Approval phrase received; submitting it to the approval graph.")
    elif line == DENY:
        print("REFUSED by the human operator.")
    else:
        print("REFUSED: input did not match the approval phrase (no retry).")
    return session_approval(session_id, digest, approved)


# -- MCP result validation ----------------------------------------------------------------

def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _common_problem(tool: str, content: Any, session_id: str) -> str:
    if not isinstance(content, dict):
        return "structured result is not an object"
    if content.get("tool") != tool:
        return f"result names tool {content.get('tool')!r}"
    if content.get("session_id") != session_id:
        return f"result names session {content.get('session_id')!r}"
    if type(content.get("action_id")) is not str or not content["action_id"]:
        return "result has no action_id"
    return ""


def measurement_problem(content: dict, expected_exposure_us: float, expected_frame_id: int) -> str:
    """Strict checks of one camera_measure result against the existing runtime contract."""
    if content.get("source") != SOURCE_HARDWARE or content.get("synthetic") is not False:
        return (f"provenance is source={content.get('source')!r}, synthetic={content.get('synthetic')!r}; "
                "only hardware is accepted")
    frame_id = content.get("frame_id")
    if type(frame_id) is not int or frame_id != expected_frame_id:
        return f"frame_id {frame_id!r}, expected {expected_frame_id}"
    exposure = content.get("exposure_us")
    if type(exposure) is not float or exposure != expected_exposure_us:
        return f"frame exposure_us {exposure!r}, expected {expected_exposure_us!r}"
    for key, expected in (("bit_depth", EXPECTED_BIT_DEPTH), ("width", EXPECTED_WIDTH),
                          ("height", EXPECTED_HEIGHT)):
        if type(content.get(key)) is not int or content[key] != expected:
            return f"{key} {content.get(key)!r}, expected {expected}"
    metrics = content.get("metrics")
    if not isinstance(metrics, dict):
        return "measurement has no metrics"
    for key in ("width", "height", "bit_depth"):
        if metrics.get(key) != content[key]:
            return f"metrics {key} {metrics.get(key)!r} disagrees with the result"
    if metrics.get("full_scale_dn") != FULL_SCALE_DN:
        return f"full_scale_dn {metrics.get('full_scale_dn')!r}, expected {FULL_SCALE_DN}"
    low, high, mean = metrics.get("min_dn"), metrics.get("max_dn"), metrics.get("mean_dn")
    if type(low) is not int or type(high) is not int or not 0 <= low <= high <= FULL_SCALE_DN:
        return f"invalid min/max DN {low!r}/{high!r}"
    if not _finite(mean) or not low <= mean <= high:
        return f"mean DN {mean!r} not within [{low}, {high}]"
    saturation = metrics.get("saturation_fraction")
    if not _finite(saturation) or not 0.0 <= saturation <= 1.0:
        return f"invalid saturation_fraction {saturation!r}"
    if (saturation > 0) != (high == FULL_SCALE_DN):
        return "saturation_fraction disagrees with max DN"
    return ""


def step_problem(tool: str, arguments: dict, content: Any, session_id: str,
                 expected_exposure_us: float | None, expected_frame_id: int) -> str:
    problem = _common_problem(tool, content, session_id)
    if problem:
        return problem
    if tool == "camera_open":
        if content.get("ok") is not True or content.get("is_open") is not True:
            return "camera_open did not report an open camera"
    elif tool == "camera_set_exposure":
        applied = content.get("exposure_us")
        if content.get("ok") is not True or type(applied) is not float \
                or applied != arguments["exposure_us"]:
            return f"applied exposure {applied!r} us, requested {arguments['exposure_us']!r} us"
    elif tool == "camera_measure":
        if expected_exposure_us is None:
            return "camera_measure before any verified exposure was set"
        return measurement_problem(content, expected_exposure_us, expected_frame_id)
    elif tool == "camera_close":
        if content.get("ok") is not True or content.get("is_open") is not False:
            return "camera_close did not report a closed camera"
    else:
        return f"unexpected tool {tool!r}"
    return ""


def _error_text(result: Any) -> str:
    return " ".join(getattr(block, "text", "") for block in (result.content or []))


async def _schema_problem(client: Client) -> str:
    """Read the server's actual tool schemas (a protocol request, not a camera tool call)."""
    listed = {tool.name: tool for tool in (await client.list_tools()).tools}
    for name, expected in TOOL_ARGUMENTS.items():
        if name not in listed:
            return f"authorized server does not offer {name}"
        schema = listed[name].input_schema
        properties = schema.get("properties", {}) if isinstance(schema, dict) else None
        if not isinstance(properties, dict) or sorted(properties) != sorted(expected):
            return f"server schema for {name} does not match the expected arguments {expected}"
    return ""


async def run_sequence(server: Any, session_id: str, report: Report) -> None:
    """Issue SEQUENCE through the official MCP Client; stop at the first failure."""
    async with Client(server, mode="legacy", read_timeout_seconds=CLIENT_READ_TIMEOUT_SECONDS) as client:
        problem = await _schema_problem(client)
        if problem:
            report.experiment_error = f"SCHEMA_MISMATCH: {problem}; no camera tool was called"
            return
        exposure: float | None = None
        for tool, arguments in SEQUENCE:
            step = {"tool": tool, "arguments": dict(arguments), "status": None, "result": None, "error": None}
            report.steps.append(step)
            try:
                result = await client.call_tool(tool, dict(arguments))
            except Exception as exc:
                step["status"], step["error"] = STEP_TRANSPORT_ERROR, f"{type(exc).__name__}: {exc}"
                report.experiment_error = f"MCP_CLIENT_ERROR in {tool}: {step['error']} (not retried)"
                return
            if result.is_error is not False:
                step["status"], step["error"] = STEP_MCP_ERROR, _error_text(result)
                report.experiment_error = f"TOOL_ERROR in {tool}: {step['error']} (not retried)"
                return
            content = result.structured_content
            problem = step_problem(tool, arguments, content, session_id, exposure,
                                   len(report.measurements) + 1)
            if problem:
                step["status"], step["error"] = STEP_INVALID, problem
                step["result"] = content if isinstance(content, dict) else None
                report.experiment_error = f"VALIDATION_FAILED in {tool}: {problem}"
                return
            step["status"], step["result"] = STEP_SUCCESS, content
            if tool == "camera_set_exposure":
                exposure = content["exposure_us"]
            elif tool == "camera_measure":
                report.measurements.append(content)


# -- orchestration ----------------------------------------------------------------------

def _approve_and_run(host: Any, envelope: SessionEnvelope, report: Report) -> None:
    request = host.start(envelope.session_id)  # the fresh session_id is the approval thread_id
    problem = _request_problem(request, host.pending(), envelope)
    if problem:
        report.experiment_error = f"APPROVAL_REQUEST_INVALID: {problem}"
        return
    _show_request(request, envelope)
    decision = _ask_decision(request)
    activation = host.resume(decision)
    if activation.get("activated") is not True:
        report.approval_status = APPROVAL_REFUSED
        report.approval_reason = str(activation.get("reason"))
        return
    report.approval_status = APPROVAL_APPROVED
    report.approval_reason = str(activation.get("reason"))
    server = host.get_mcp_server()
    # One event loop on this (the SDK-preloading) thread; tool handlers run on it.
    asyncio.run(run_sequence(server, envelope.session_id, report))


def _shutdown(host: Any, report: Report) -> None:
    report.shutdown_attempted = True
    try:
        report.session_state_before_shutdown = host.status()["session_state"]
    except Exception:
        pass
    try:
        result = host.shutdown()
    except BaseException as exc:  # recorded, never retried; a second Ctrl+C must not hide it
        if _is_interrupt(exc):
            report.interrupted = True
        report.cleanup_error = f"{type(exc).__name__}: {exc}"
    else:
        report.shutdown_ok = True
        runtime = result.get("runtime") if isinstance(result, dict) else None
        report.final_session_state = runtime.get("session_state") if isinstance(runtime, dict) else None
    try:
        report.owns_device_after = host.status()["owns_device"]
    except Exception:
        report.owns_device_after = None


# -- report ---------------------------------------------------------------------------------

def _restoration_line(report: Report) -> str:
    if not report.shutdown_attempted:
        return "NOT ATTEMPTED; ExposureTime unverified"
    if report.cleanup_error is not None:
        return "NOT VERIFIED (trusted shutdown failed; see CLEANUP_ERROR)"
    wrote = any(step["tool"] == "camera_set_exposure" and step["status"] is not None
                for step in report.steps)
    if not wrote:
        return "NOT REQUIRED (no exposure write was requested)"
    if any(step["tool"] == "camera_close" and step["status"] == STEP_SUCCESS for step in report.steps):
        return ("RESTORED AND VERIFIED by IRaypleCamera.close() during camera_close "
                "(exact readback; a mismatch would have failed the call)")
    return ("RESTORED AND VERIFIED or not required, by IRaypleCamera.close() during the "
            "runtime's abort cleanup or trusted host shutdown (failure would have raised)")


def format_report(report: Report) -> list[str]:
    lines = [
        "=" * 76,
        "LUMEN PHYSICAL MCP CHECK REPORT (fixed script, not AI decisions)",
        "=" * 76,
        f"session_id          : {report.session_id}",
        f"envelope SHA-256    : {report.envelope_sha256}",
        f"approval            : {report.approval_status}"
        + (f" ({report.approval_reason})" if report.approval_reason else ""),
    ]
    issued = report.steps
    lines.append(f"MCP tool calls      : {len(issued)} issued (fixed plan {len(SEQUENCE)}, "
                 f"policy max {MAX_TOOL_CALLS})")
    for number, step in enumerate(issued, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in step["arguments"].items())
        detail = f"  {number}. {step['tool']}({args}) -> {step['status']}"
        if step["tool"] == "camera_set_exposure" and step["status"] == STEP_SUCCESS:
            detail += f"; applied {step['result']['exposure_us']!r} us"
        if step["error"]:
            detail += f": {step['error']}"
        lines.append(detail)
    lines.append(f"measurements        : {len(report.measurements)} validated (policy max {MAX_MEASUREMENTS})")
    for number, m in enumerate(report.measurements, 1):
        metrics = m["metrics"]
        lines.append(
            f"  measurement {number}: source={m['source']} frame_id={m['frame_id']} "
            f"exposure={m['exposure_us']!r} us {m['width']}x{m['height']} Mono{m['bit_depth']} "
            f"min={metrics['min_dn']} max={metrics['max_dn']} DN mean={metrics['mean_dn']:.4f} DN "
            f"saturation_fraction={metrics['saturation_fraction']:.6f}")
    if len(report.measurements) == 2:
        first = report.measurements[0]["metrics"]["mean_dn"]
        second = report.measurements[1]["metrics"]["mean_dn"]
        lines.append(f"mean DN difference  : {second - first:+.4f} DN (measurement 2 - measurement 1)")
        ratio = f"{second / first:.4f}" if first > 0 else "undefined (measurement 1 mean is 0 DN)"
        lines.append(f"mean DN ratio       : {ratio} (exposure ratio "
                     f"{report.measurements[1]['exposure_us'] / report.measurements[0]['exposure_us']:g})")
    lines.append(f"session state       : {report.session_state_before_shutdown or 'not active'} before "
                 f"shutdown, {report.final_session_state or 'n/a'} after")
    lines.append(f"exposure restoration: {_restoration_line(report)}")
    if report.shutdown_ok:
        lines.append("host shutdown       : OK (native Close/DestroyHandle not repeated)")
    else:
        lines.append("host shutdown       : " + ("FAILED" if report.shutdown_attempted else "NOT ATTEMPTED"))
    lines.append(f"device ownership    : {'released' if report.owns_device_after is False else 'STILL CLAIMED'}")
    if report.experiment_error:
        lines.append(f"EXPERIMENT_ERROR    : {report.experiment_error}")
    if report.interrupted:
        lines.append("INTERRUPTED         : KeyboardInterrupt")
    if report.cleanup_error:
        lines.append(f"CLEANUP_ERROR       : {report.cleanup_error}")
    if report.cleanup_error or (report.interrupted and not report.shutdown_ok):
        lines.append(UNCERTAIN_STATE_ADVICE)
    if report.measurements:
        lines.append(SCOPE_NOTE)
    lines.append("PHYSICAL_MCP_CHECK: " + ("PASS" if report.passed else "FAIL"))
    lines.append("=" * 76)
    return lines


def exit_code_for(report: Report) -> int:
    if report.passed:
        return EXIT_PASS
    if report.cleanup_error is not None or not report.shutdown_ok:
        return EXIT_CLEANUP_FAILED
    if report.interrupted:
        return EXIT_INTERRUPTED
    if report.approval_status == APPROVAL_REFUSED:
        return EXIT_APPROVAL_REFUSED
    return EXIT_FAILED


# -- entry point --------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physical_mcp_smoke",
        description="Manually approved six-tool MCP diagnostic on the physical iRAYPLE camera.")
    parser.add_argument("--physical-mcp-check", action="store_true",
                        help="required: run the diagnostic (two interactive confirmations follow)")
    return parser


def _warning() -> list[str]:
    return [
        "WARNING: this program controls a PHYSICAL camera (iRAYPLE MV-A7200MU130) through the",
        "authorized MCP server. It temporarily changes ExposureTime to 1000 us and 5000 us,",
        "captures two frames, and restores the original ExposureTime at close.",
        "This is a fixed diagnostic script, not an AI experiment. No API, no network, no files.",
        "A forcibly killed process cannot restore ExposureTime.",
        "A second, separate approval of the exact session envelope follows this confirmation.",
    ]


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_REFUSED

    print("LUMEN physical MCP check (HW3.7-D4)")
    if not args.physical_mcp_check:
        print("NO_ACTION: pass --physical-mcp-check to run. No SDK was loaded and no camera was accessed.")
        return EXIT_REFUSED
    if not (_is_interactive(sys.stdin) and _is_interactive(sys.stdout)):
        print("REFUSED: an interactive terminal is required. No SDK was loaded and no camera was accessed.")
        return EXIT_NOT_INTERACTIVE

    for line in _warning():
        print(line)
    try:
        response = input(f"Type {CONFIRMATION_PHRASE} to continue to session approval: ")
    except (EOFError, KeyboardInterrupt):
        print("\nREFUSED: no confirmation received. No SDK was loaded and no camera was accessed.")
        return EXIT_REFUSED
    if response != CONFIRMATION_PHRASE:
        print("REFUSED: confirmation phrase did not match. No SDK was loaded and no camera was accessed.")
        return EXIT_REFUSED
    print("Confirmation recorded. This does NOT approve hardware access.")

    bootstrap_thread = threading.get_ident()
    try:
        sdk = phys.preload_vendor_sdk_once()  # main thread, before any event loop or MCP serving
    except KeyboardInterrupt:
        print("INTERRUPTED while loading the vendor SDK. No camera was opened.")
        return EXIT_INTERRUPTED
    except Exception as exc:
        print(f"PHYSICAL_MCP_CHECK: FAIL (SDK_LOAD_FAILED: {type(exc).__name__}: {exc}). No camera was opened.")
        return EXIT_FAILED

    envelope = build_envelope(new_session_id())
    report = Report(envelope)
    try:
        host = phys.PhysicalApprovedSessionHost(envelope, sdk)  # no SDK call
    except Exception as exc:
        print(f"PHYSICAL_MCP_CHECK: FAIL (HOST_NOT_CREATED: {type(exc).__name__}: {exc}). No camera was opened.")
        return EXIT_FAILED

    escaped: BaseException | None = None
    try:
        if threading.get_ident() != bootstrap_thread:
            raise RuntimeError("bootstrap thread changed before MCP serving")
        _approve_and_run(host, envelope, report)
    except BaseException as exc:
        if _is_interrupt(exc):
            report.interrupted = True
            report.experiment_error = report.experiment_error or "interrupted before the sequence completed"
        elif isinstance(exc, Exception):
            report.experiment_error = report.experiment_error or f"UNEXPECTED_ERROR: {type(exc).__name__}: {exc}"
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
