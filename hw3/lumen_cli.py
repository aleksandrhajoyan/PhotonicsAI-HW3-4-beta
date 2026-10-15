"""Manually operated, OFFLINE-only command line for one LUMEN session (HW3.6-A).

    python hw3/lumen_cli.py [--mode offline]

One run = one call of the trusted run_lumen_session() driver, which owns the
fresh session identity, the LangGraph human approval, the four-agent graph,
the authorized MCP camera tools on FakeCamera and the mandatory host shutdown.
This module adds no orchestration, approval validation or camera logic; it only
supplies the trusted inputs of that driver and prints a short summary:

    trusted envelope   offline_envelope_template(): the five camera MCP tools,
                       FakeCamera's own exposure range and small fixed budgets;
                       the driver replaces its placeholder session_id
    human approval     TerminalApproval: shows the genuine pending request and
                       builds an approving decision only after the human types
                       exactly "APPROVE <session_id>"; "DENY", any other text,
                       EOF or a second call never approve and never re-prompt
    scripted models    ScriptedOuterCaller and ScriptedHardwareCaller are
                       deterministic SCRIPTS (not AI reasoning) implementing
                       call(instructions, input_text) -> str

Offline only: there is no live mode, no API key, no LLM API client, no vendor
SDK and no physical camera. Every measurement is FakeCamera synthetic data and
is never presented as a physical camera measurement. Nothing is written to
disk and nothing is retried. Importing this module starts nothing.

Exit codes:
    0    experiment completed and the host shut down cleanly
    1    driver failure, cleanup failure or unexpected error
    2    command-line usage error (argparse)
    3    not an interactive terminal: nothing was started
    4    human refused (DENY, wrong phrase, EOF): no camera call was made
    5    session ended without a completed experiment
    130  interrupted (KeyboardInterrupt) after the driver's cleanup attempt
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from executor_mcp_loop import DECISION_INSTRUCTIONS
from fake_camera import MAX_EXPOSURE_US, MIN_EXPOSURE_US
from lumen_agent_system import ORCHESTRATOR_INSTRUCTIONS, SCIENTIST_INSTRUCTIONS
from lumen_session_driver import (
    SESSION_EXPERIMENT_COMPLETED,
    SESSION_REFUSED,
    LumenDriverError,
    run_lumen_session,
)
from session_policy import CAMERA_TOOLS, SessionEnvelope, session_approval

MODE_OFFLINE = "offline"

EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NOT_INTERACTIVE = 3
EXIT_REFUSED = 4
EXIT_NOT_COMPLETED = 5
EXIT_INTERRUPTED = 130

# Trusted, fixed session limits (never taken from model output).
TEMPLATE_SESSION_ID = "lumen-cli-template-replaced-by-driver"
DEVICE_LABEL = "FakeCamera (offline synthetic test double; NOT a physical camera)"
MAX_MEASUREMENTS = 2
MAX_TOOL_CALLS = 6
MAX_DURATION_SECONDS = 120.0
MAX_HARDWARE_DECISIONS = 6  # five tool calls + finish

FIRST_EXPOSURE_US = 500.0
SECOND_EXPOSURE_US = 2000.0

RESEARCH_GOAL = ("Compare the mean brightness and saturation of the synthetic FakeCamera "
                 f"image at {FIRST_EXPOSURE_US:g} us and {SECOND_EXPOSURE_US:g} us exposure.")

SCRIPTED_LABEL = "SCRIPTED offline demonstration (deterministic, not AI reasoning)"
UNAVAILABLE = "unavailable"


def offline_envelope_template() -> SessionEnvelope:
    """The trusted template; run_lumen_session() gives every run a fresh session_id."""
    return SessionEnvelope(
        session_id=TEMPLATE_SESSION_ID,
        device_label=DEVICE_LABEL,
        allowed_tools=tuple(CAMERA_TOOLS),
        min_exposure_us=MIN_EXPOSURE_US,
        max_exposure_us=MAX_EXPOSURE_US,
        max_measurements=MAX_MEASUREMENTS,
        max_tool_calls=MAX_TOOL_CALLS,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )


# -- Part C: manual human approval ----------------------------------------------------

class TerminalApproval:
    """Trusted human_decision_callback for run_lumen_session(); single use.

    This is a local manual interaction, not remote identity authentication.
    """

    def __init__(self, input_fn=None, out=None):
        self._input = input_fn
        self._out = out
        self.calls = 0
        self.approved = False

    def _print(self, text: str = "") -> None:
        print(text, file=self._out if self._out is not None else sys.stdout, flush=True)

    def __call__(self, request: Any) -> dict | None:
        self.calls += 1
        if self.calls != 1:
            self._print("REFUSED: approval was already requested once in this run.")
            return None
        if not isinstance(request, dict) or not isinstance(request.get("envelope"), dict) \
                or type(request.get("session_id")) is not str \
                or type(request.get("envelope_sha256")) is not str \
                or request["envelope"].get("session_id") != request["session_id"]:
            self._print("REFUSED: the pending approval request is malformed.")
            return None
        session_id = request["session_id"]
        digest = request["envelope_sha256"]
        envelope = request["envelope"]
        self._show(request, envelope)
        phrase = f"APPROVE {session_id}"
        self._print(f"To approve, type exactly:  {phrase}")
        self._print("To refuse, type:           DENY")
        try:
            line = (self._input or input)("> ")
        except EOFError:
            self._print("\nREFUSED: no input (EOF). Nothing was approved.")
            return session_approval(session_id, digest, False)
        if line == phrase:
            self.approved = True
            self._print("Approval recorded for this session only.")
            return session_approval(session_id, digest, True)
        if line == "DENY":
            self._print("REFUSED by the human operator.")
        else:
            self._print("REFUSED: input did not match the approval phrase (no retry).")
        return session_approval(session_id, digest, False)

    def _show(self, request: dict, envelope: dict) -> None:
        tools = envelope.get("allowed_tools")
        self._print("=" * 72)
        self._print("LUMEN HARDWARE SESSION APPROVAL REQUEST (pending in LangGraph)")
        self._print("=" * 72)
        self._print(f"request kind        : {request.get('kind', UNAVAILABLE)}")
        self._print(f"session_id          : {request['session_id']}")
        self._print(f"envelope SHA-256    : {request['envelope_sha256']}")
        self._print(f"device              : {envelope.get('device_label', UNAVAILABLE)}")
        self._print("allowed tools       : "
                    + (", ".join(tools) if isinstance(tools, list) else UNAVAILABLE))
        self._print(f"exposure bounds (us): [{envelope.get('min_exposure_us', UNAVAILABLE)}, "
                    f"{envelope.get('max_exposure_us', UNAVAILABLE)}]")
        self._print(f"max measurements    : {envelope.get('max_measurements', UNAVAILABLE)}")
        self._print(f"max tool calls      : {envelope.get('max_tool_calls', UNAVAILABLE)}")
        self._print(f"max duration (s)    : {envelope.get('max_duration_seconds', UNAVAILABLE)}")
        self._print("data source         : synthetic FakeCamera only (offline mode)")
        self._print("-" * 72)


# -- Part D: deterministic scripted callers -----------------------------------------

class ScriptError(RuntimeError):
    """The scripted caller could not identify the expected context; it fails closed."""


def _route(next_role: str, instruction: str, reason: str) -> str:
    return json.dumps({"next_role": next_role, "instruction": instruction, "reason": reason})


class ScriptedOuterCaller:
    """SCRIPTED outer model: Orchestrator -> Scientist -> Orchestrator -> Executor -> finish."""

    HYPOTHESIS = (
        "[SCRIPTED, not AI reasoning] Hypothesis: on the synthetic FakeCamera, mean "
        "brightness increases with exposure and the saturated fraction grows once pixels "
        f"clip. Goal: measure at {FIRST_EXPOSURE_US:g} us, then at {SECOND_EXPOSURE_US:g} us, "
        "and compare mean_dn and saturation_fraction. Synthetic data only.")

    def __init__(self):
        self.calls: list[str] = []

    def call(self, instructions: str, input_text: str) -> str:
        if instructions == SCIENTIST_INSTRUCTIONS:
            self.calls.append("scientist")
            return self.HYPOTHESIS
        if instructions != ORCHESTRATOR_INSTRUCTIONS:
            raise ScriptError("scripted outer caller received unexpected instructions")
        self.calls.append("orchestrator")
        lines = input_text.splitlines()
        if "[SCIENTIST]" not in lines:
            return _route("scientist", "State a hypothesis and an exposure research goal.",
                          "Scripted step: a research goal is needed first.")
        if "[EXECUTOR]" not in lines:
            return _route("executor", "Run the approved two-exposure synthetic measurement.",
                          "Scripted step: the scientist has stated the goal.")
        return _route("finish", "Stop.", "Scripted step: the executor has reported.")


class ScriptedHardwareCaller:
    """SCRIPTED Executor decisions; each step is checked against the latest context."""

    PLAN = (
        ("camera_open", {}),
        ("camera_set_exposure", {"exposure_us": FIRST_EXPOSURE_US}),
        ("camera_measure", {}),
        ("camera_set_exposure", {"exposure_us": SECOND_EXPOSURE_US}),
        ("camera_measure", {}),
    )

    def __init__(self):
        self.contexts: list[dict] = []

    def call(self, instructions: str, input_text: str) -> str:
        if instructions != DECISION_INSTRUCTIONS:
            raise ScriptError("scripted hardware caller received unexpected instructions")
        context = json.loads(input_text)
        self.contexts.append(context)
        history = context.get("history")
        if not isinstance(history, list):
            raise ScriptError("context has no history")
        done = [e for e in history if isinstance(e, dict) and e.get("type") == "tool_call"]
        for index, event in enumerate(done):
            if index >= len(self.PLAN) or event.get("status") != "success" \
                    or (event.get("tool"), event.get("arguments")) != self.PLAN[index]:
                raise ScriptError(f"tool call history does not match the script at step {index}")
        step = len(done)
        latest = context.get("latest_measurement")
        if step in (3, 5):
            exposure = FIRST_EXPOSURE_US if step == 3 else SECOND_EXPOSURE_US
            if not isinstance(latest, dict) or latest != done[step - 1].get("result") \
                    or latest.get("exposure_us") != exposure \
                    or type(latest.get("frame_id")) is not int:
                raise ScriptError(f"expected measurement at {exposure!r} us is not in the context")
        if step == len(self.PLAN):
            first, second = done[2]["result"], done[4]["result"]
            if first["frame_id"] == second["frame_id"]:
                raise ScriptError("the two measurements do not have distinct frame_ids")
            return json.dumps({"action": "finish",
                               "reason": "Scripted: both exposures were measured."})
        tool, arguments = self.PLAN[step]
        return json.dumps({"action": "call_tool", "tool": tool, "arguments": arguments})


# -- Part E: terminal report --------------------------------------------------------

def _value(mapping: Any, key: str) -> Any:
    if isinstance(mapping, dict) and key in mapping and mapping[key] is not None:
        return mapping[key]
    return UNAVAILABLE


def _successful_events(result: Any, tool: str) -> list[dict]:
    events = result.get("events") if isinstance(result, dict) else None
    if not isinstance(events, list):
        return []
    return [e for e in events if isinstance(e, dict) and e.get("type") == "tool_call"
            and e.get("tool") == tool and e.get("status") == "success"
            and isinstance(e.get("result"), dict)]


def summarize(report: dict) -> list[str]:
    """Concise summary from the driver report only; missing fields read 'unavailable'."""
    result = report.get("experiment_result")
    dialogue = report.get("dialogue")
    roles = ([turn.get("role", UNAVAILABLE) for turn in dialogue if isinstance(turn, dict)]
             if isinstance(dialogue, list) else None)
    exposures = [e["result"].get("exposure_us", UNAVAILABLE)
                 for e in _successful_events(result, "camera_set_exposure")]
    measurements = [e["result"] for e in _successful_events(result, "camera_measure")]
    status = report.get("session_status")
    lines = [
        "=" * 72,
        f"LUMEN OFFLINE SESSION SUMMARY ({SCRIPTED_LABEL})",
        "=" * 72,
        f"session_id          : {_value(report, 'session_id')}",
        f"session status      : {_value(report, 'session_status')}",
        f"approval status     : {_value(report, 'approval_status')}",
        "outer roles visited : " + (" -> ".join(roles) if roles else UNAVAILABLE),
        f"experiment status   : {_value(report, 'experiment_status')}",
        f"tool calls          : {_value(result, 'tool_call_count')}",
        f"measurements        : {len(measurements)} successful camera_measure result(s)",
        "exposures applied   : " + (", ".join(f"{x} us" for x in exposures)
                                    if exposures else UNAVAILABLE),
    ]
    for number, measurement in enumerate(measurements, 1):
        metrics = measurement.get("metrics")
        lines.append(f"  measurement {number}: frame_id={_value(measurement, 'frame_id')} "
                     f"exposure_us={_value(measurement, 'exposure_us')} "
                     f"mean_dn={_value(metrics, 'mean_dn')} "
                     f"saturation_fraction={_value(metrics, 'saturation_fraction')} "
                     f"source={_value(measurement, 'source')}")
    lines.append(f"evidence provenance : {_value(report, 'evidence_provenance')}")
    if report.get("evidence_provenance") == "synthetic" or any(
            m.get("source") == "synthetic" for m in measurements):
        lines.append("provenance note     : synthetic = simulated FakeCamera data; "
                     "NOT a physical camera measurement")
    shutdown = ("succeeded" if report.get("shutdown_succeeded") is True
                else "FAILED" if report.get("shutdown_attempted") is True else "not attempted")
    lines.append(f"host shutdown       : {shutdown}")
    if status == SESSION_EXPERIMENT_COMPLETED and report.get("shutdown_succeeded") is True:
        outcome = "COMPLETED"
    elif status == SESSION_REFUSED:
        outcome = "REFUSED (no camera call was made)"
    else:
        outcome = "NOT COMPLETED"
    lines.append(f"run outcome         : {outcome}")
    if report.get("reason"):
        lines.append(f"reason              : {report['reason']}")
    lines.append("=" * 72)
    return lines


# -- Part A: entry point ------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumen_cli",
        description="Run one manually approved, OFFLINE LUMEN session on the synthetic "
                    "FakeCamera with scripted (non-AI) models.")
    parser.add_argument("--mode", choices=[MODE_OFFLINE], default=MODE_OFFLINE,
                        help="only 'offline' is supported (default)")
    return parser


def _exit_code(report: dict) -> int:
    if report.get("shutdown_succeeded") is not True:
        return EXIT_FAILED
    status = report.get("session_status")
    if status == SESSION_EXPERIMENT_COMPLETED:
        return EXIT_COMPLETED
    if status == SESSION_REFUSED:
        return EXIT_REFUSED
    return EXIT_NOT_COMPLETED


def main(argv: list[str] | None = None) -> int:
    try:
        _parser().parse_args(argv)
    except SystemExit as exc:  # argparse usage error (2) or --help (0)
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("lumen_cli: an interactive terminal is required for human approval; "
              "nothing was started.", file=sys.stderr)
        return EXIT_NOT_INTERACTIVE

    print(f"LUMEN offline mode: {SCRIPTED_LABEL}.")
    print("Camera: FakeCamera synthetic data only. No API key, no network, no physical camera.")
    approval = TerminalApproval()
    try:
        report = run_lumen_session(offline_envelope_template(), ScriptedOuterCaller(),
                                   ScriptedHardwareCaller(), RESEARCH_GOAL,
                                   MAX_HARDWARE_DECISIONS, approval)
    except LumenDriverError as exc:
        print(f"lumen_cli: session FAILED: {exc}", file=sys.stderr)
        if isinstance(exc.report, dict):
            print("\n".join(summarize(exc.report)))
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\nlumen_cli: interrupted; the driver attempted host shutdown. "
              "Nothing was completed.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:
        print(f"lumen_cli: unexpected failure ({type(exc).__name__}); nothing is reported "
              "as completed.", file=sys.stderr)
        return EXIT_FAILED
    print("\n".join(summarize(report)))
    return _exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
