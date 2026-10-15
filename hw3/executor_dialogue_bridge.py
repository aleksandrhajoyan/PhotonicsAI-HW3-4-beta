"""Synchronous bridge: AdaptiveMCPExecutor -> one shared-dialogue executor turn.

Two pieces for a future synchronous LangGraph Executor node:

    result = run_adaptive_experiment(host, provider, research_goal, max_decisions)
    turn = format_executor_turn(result)       # {"role": "executor", "text": ...}

run_adaptive_experiment() runs the existing asynchronous AdaptiveMCPExecutor
once, from synchronous code, against ONE host that a human has already
approved elsewhere. It never creates, starts, resumes or approves a host and
never builds an LLM client; it only wires the supplied dependencies into the
Executor, runs it with asyncio.run() and returns the Executor's own result
object unchanged (no retry, no normalization). The Executor itself always
ends with the trusted host.shutdown().

If an event loop is already running in the calling thread, nothing is
dispatched: no Executor and no coroutine are created, no second loop is
started, the trusted host.shutdown() is attempted and BridgeError is raised,
naming any cleanup failure. The same fail-closed handling applies when the
Executor constructor rejects its arguments or the run raises instead of
returning a result: an approved host never stays active.

format_executor_turn() is deterministic and read-only. It renders only known
structured fields of the Executor result, so the turn can never be mistaken
for an HW1 Silverbox code execution:
    - no line starts with "exit_code=", "TEST_RMSE=", "TRAIN_RMSE=" or "file=";
    - no "--- stdout ---" / "--- stderr ---" sections, no raw model output,
      no tracebacks, no histograms or pixels;
    - untrusted strings (reasons, errors) are put on one line after a fixed
      key, with control characters and line separators replaced and length
      capped, so embedded newlines cannot start a new record.
The first line is LUMEN_COMPLETED_MARK only if the result shows status
"completed", at least one successful measurement, a clean host shutdown and
consistent source provenance; otherwise LUMEN_NOT_COMPLETED_MARK.
evidence_type is derived only from the observed source labels:
synthetic (FakeCamera), hardware, mixed, or unknown (none or inconsistent).
The turn is at most MAX_TURN_CHARS; older tool-call lines are dropped first
and their number is stated. The full evidence stays in the Executor result.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

from approved_session_host import OfflineApprovedSessionHost
from executor_mcp_loop import RESULT_SCHEMA, STATUS_COMPLETED, AdaptiveMCPExecutor

LUMEN_COMPLETED_MARK = "LUMEN_EXPERIMENT_COMPLETED"
LUMEN_NOT_COMPLETED_MARK = "LUMEN_EXPERIMENT_NOT_COMPLETED"
MAX_TURN_CHARS = 6000

EXECUTOR_ROLE = "executor"
EVIDENCE_SYNTHETIC = "synthetic"
EVIDENCE_HARDWARE = "hardware"
EVIDENCE_MIXED = "mixed"
EVIDENCE_UNKNOWN = "unknown"

_MEASUREMENT_TOOL = "camera_measure"
_KNOWN_SOURCES = ("synthetic", "hardware")
_LATEST_METRICS = ("mean_dn", "mean_fraction_of_full_scale", "saturation_fraction")

_REASON_CHARS = 600
_ERROR_CHARS = 200
_SHORT_CHARS = 80
_ARGUMENTS_CHARS = 160

# Line starts that HW1 (task6/agentic_system.py) parses as code-execution records.
_FORBIDDEN_LINE_STARTS = ("exit_code=", "TEST_RMSE", "TRAIN_RMSE", "file=")
_FORBIDDEN_SUBSTRINGS = ("--- stdout ---", "--- stderr ---", "--- traceback ---")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")


class BridgeError(Exception):
    """The experiment was not dispatched (or the turn could not be rendered safely)."""


# -- Part A: synchronous execution bridge ----------------------------------------

def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _fail_closed(host: Any, problem: str, cause: BaseException | None) -> None:
    """Attempt the trusted host.shutdown() (idempotent), then always raise BridgeError.

    The original error stays the __cause__; a cleanup failure is named in the
    message (and becomes the cause only when there was no original error).
    """
    if not isinstance(host, OfflineApprovedSessionHost):
        raise BridgeError(f"{problem}; host is not an OfflineApprovedSessionHost, "
                          f"no shutdown attempted") from cause
    try:
        host.shutdown()
    except Exception as exc:
        raise BridgeError(f"{problem}; host shutdown FAILED (cleanup failed): "
                          f"{type(exc).__name__}: {exc}") from (cause or exc)
    raise BridgeError(f"{problem}; host was shut down") from cause


def run_adaptive_experiment(host: OfflineApprovedSessionHost, decision_provider: Any,
                            research_goal: str, max_decisions: int) -> dict:
    """Run AdaptiveMCPExecutor once on an already approved host; return its result unchanged.

    Fails closed: if a loop is already running, the Executor cannot be
    constructed, or the run raises instead of returning a result, the host is
    shut down and BridgeError is raised (original error as __cause__; only its
    type is named in the message). Nothing is retried.
    """
    # Checked before any Executor or coroutine exists.
    if _running_loop() is not None:
        _fail_closed(host, "an event loop is already running in this thread; "
                           "experiment not dispatched", None)
    try:
        executor = AdaptiveMCPExecutor(host, decision_provider, research_goal, max_decisions)
    except Exception as exc:
        _fail_closed(host, f"executor startup failed ({type(exc).__name__}); "
                           f"experiment not dispatched", exc)
    try:
        return asyncio.run(executor.run())
    except Exception as exc:
        _fail_closed(host, f"executor run failed unexpectedly ({type(exc).__name__}); "
                           f"no result, not retried", exc)


# -- Part B: dialogue formatter ---------------------------------------------------

def _clean(value: Any, limit: int) -> str:
    """One safe line: no control characters or line separators, no record markers, bounded."""
    text = value if isinstance(value, str) else repr(value)
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"-{3,}", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = f"{text[:limit]}...[{len(text) - limit} more chars omitted]"
    return text


def _quoted(value: Any, limit: int) -> str:
    return json.dumps(_clean(value, limit), ensure_ascii=False)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _int_text(value: Any) -> str:
    return str(value) if _is_int(value) else "unknown"


def _number_text(value: Any) -> str:
    return repr(value) if _is_number(value) else "unknown"


def _tool_events(result: dict) -> list[dict]:
    events = result.get("events")
    if not isinstance(events, list):
        return []
    return [e for e in events if isinstance(e, dict) and e.get("type") == "tool_call"]


def evidence_type(result: dict) -> str:
    """synthetic | hardware | mixed | unknown, from observed source labels only.

    unknown when nothing was observed, a label is unrecognized, or the event
    log, latest measurement and provenance summary disagree.
    """
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        return EVIDENCE_UNKNOWN
    summary = provenance.get("sources_observed")
    if not isinstance(summary, list) or not all(isinstance(s, str) for s in summary):
        return EVIDENCE_UNKNOWN
    observed = set(summary)
    from_events = {e.get("source") for e in _tool_events(result)
                   if e.get("status") == "success" and e.get("source") is not None}
    latest = result.get("latest_measurement")
    if isinstance(latest, dict):
        if latest.get("source") not in _KNOWN_SOURCES:
            return EVIDENCE_UNKNOWN
        if latest.get("synthetic") is not (latest["source"] == "synthetic"):
            return EVIDENCE_UNKNOWN
        from_events.add(latest["source"])
    elif latest is not None:
        return EVIDENCE_UNKNOWN
    if not observed or observed != from_events or not observed <= set(_KNOWN_SOURCES):
        return EVIDENCE_UNKNOWN
    if provenance.get("synthetic_only") is not (observed == {"synthetic"}):
        return EVIDENCE_UNKNOWN
    if provenance.get("hardware_evidence") is not ("hardware" in observed):
        return EVIDENCE_UNKNOWN
    if observed == {"synthetic"}:
        return EVIDENCE_SYNTHETIC
    if observed == {"hardware"}:
        return EVIDENCE_HARDWARE
    return EVIDENCE_MIXED


def is_completed(result: dict) -> bool:
    """True only for a finished run with a measurement, a clean shutdown and consistent provenance."""
    shutdown = result.get("host_shutdown")
    count = result.get("successful_measurement_count")
    return (result.get("schema") == RESULT_SCHEMA
            and result.get("status") == STATUS_COMPLETED
            and _is_int(count) and count >= 1
            and isinstance(result.get("latest_measurement"), dict)
            and result.get("host_shut_down") is True
            and isinstance(shutdown, dict) and shutdown.get("succeeded") is True
            and evidence_type(result) != EVIDENCE_UNKNOWN)


def _tool_line(event: dict) -> str:
    arguments = event.get("arguments")
    try:
        args_text = json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        args_text = "unrenderable"
    parts = [f"  tool_call #{_int_text(event.get('sequence'))}",
             f"decision={_int_text(event.get('decision_number'))}",
             f"tool={_clean(event.get('tool'), _SHORT_CHARS)}",
             f"arguments={_clean(args_text, _ARGUMENTS_CHARS)}",
             f"status={_clean(event.get('status'), _SHORT_CHARS)}"]
    if "frame_id" in event and event["frame_id"] is not None:
        parts.append(f"frame_id={_int_text(event['frame_id'])}")
    if "source" in event and event["source"] is not None:
        parts.append(f"source={_clean(event['source'], _SHORT_CHARS)}")
    result = event.get("result")
    if event.get("tool") == "camera_set_exposure" and isinstance(result, dict) \
            and "exposure_us" in result:
        parts.append(f"applied_exposure_us={_number_text(result['exposure_us'])}")
    if event.get("error"):
        parts.append(f"error={_quoted(event['error'], _ERROR_CHARS)}")
    return " ".join(parts)


def _header(result: dict, completed: bool, evidence: str) -> list[str]:
    tool_count = result.get("tool_call_count")
    measured = result.get("successful_measurement_count")
    provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
    sources = provenance.get("sources_observed")
    sources = sources if isinstance(sources, list) else []
    known = sorted({s for s in sources if s in _KNOWN_SOURCES})
    unknown = len([s for s in sources if s not in _KNOWN_SOURCES])
    sources_text = ",".join(known) or "none"
    if unknown:
        sources_text += f" (+{unknown} unrecognized)"
    if evidence == EVIDENCE_SYNTHETIC:
        note = "synthetic = simulated FakeCamera data; NOT a physical hardware measurement"
    elif evidence == EVIDENCE_HARDWARE:
        note = "hardware = a real backend reported hardware provenance"
    elif evidence == EVIDENCE_MIXED:
        note = "mixed = both synthetic and hardware sources; do not treat as pure hardware evidence"
    else:
        note = "unknown = no source observed or provenance is inconsistent; no evidence claim"
    lines = [
        LUMEN_COMPLETED_MARK if completed else LUMEN_NOT_COMPLETED_MARK,
        "turn_kind=lumen_hw3_mcp_camera_experiment (not a Silverbox script run; "
        "no script, no exit code, no RMSE)",
        f"run_status={_clean(result.get('status'), _SHORT_CHARS)}",
        f"final_reason={_quoted(result.get('final_reason'), _REASON_CHARS)}",
        f"decisions_used={_int_text(result.get('decision_count'))} "
        f"max_decisions={_int_text(result.get('max_decisions'))} (LUMEN executor budget)",
        f"tool_call_count={_int_text(tool_count)}",
        f"successful_measurement_count={_int_text(measured)}",
        f"evidence_type={evidence}",
        f"sources_observed={sources_text}",
        f"provenance_note={note}",
    ]
    if not (_is_int(measured) and measured >= 1):
        lines.append("measurement_note=no successful measurement exists in this run")
    return lines


def _footer(result: dict) -> list[str]:
    frames = [e.get("frame_id") for e in _tool_events(result)
              if e.get("tool") == _MEASUREMENT_TOOL and e.get("status") == "success"
              and _is_int(e.get("frame_id"))]
    latest = result.get("latest_measurement")
    lines = [f"first_frame_id={frames[0] if frames else 'none'}"]
    if isinstance(latest, dict):
        lines.append(f"latest_frame_id={_int_text(latest.get('frame_id'))}")
        metrics = latest.get("metrics") if isinstance(latest.get("metrics"), dict) else {}
        parts = [f"latest_measurement: exposure_us={_number_text(latest.get('exposure_us'))}"]
        parts += [f"{key}={_number_text(metrics[key])}" for key in _LATEST_METRICS if key in metrics]
        parts.append(f"source={_clean(latest.get('source'), _SHORT_CHARS)}")
        lines.append(" ".join(parts))
    else:
        lines += ["latest_frame_id=none", "latest_measurement=none"]
    shutdown = result.get("host_shutdown")
    if not isinstance(shutdown, dict):
        lines.append("host_shutdown=unknown")
    elif shutdown.get("succeeded") is True and result.get("host_shut_down") is True:
        lines.append("host_shutdown=succeeded")
    elif shutdown.get("succeeded") is True or result.get("host_shut_down") is True:
        lines.append("host_shutdown=inconsistent (shutdown flags disagree)")
    elif shutdown.get("attempted") is True:
        lines.append(f"host_shutdown=FAILED error={_quoted(shutdown.get('error'), _ERROR_CHARS)}")
    else:
        lines.append("host_shutdown=not_attempted")
    return lines


def _check_safe(text: str) -> None:
    for line in text.splitlines():
        if line.lstrip().startswith(_FORBIDDEN_LINE_STARTS):
            raise BridgeError("formatted turn would contain an HW1 execution record line")
    if any(marker in text for marker in _FORBIDDEN_SUBSTRINGS):
        raise BridgeError("formatted turn would contain an HW1 output section marker")
    if len(text) > MAX_TURN_CHARS:
        raise BridgeError(f"formatted turn has {len(text)} characters, limit is {MAX_TURN_CHARS}")


def format_executor_turn(result: dict) -> dict:
    """Deterministic {"role": "executor", "text": ...} for one Executor result; input not modified."""
    if not isinstance(result, dict):
        raise TypeError(f"result must be a dict, got {type(result).__name__}")
    evidence = evidence_type(result)
    head = _header(result, is_completed(result), evidence)
    tail = _footer(result)
    tool_lines = [_tool_line(event) for event in _tool_events(result)]

    def render(shown: list[str], omitted: int) -> str:
        log = [f"tool_call_log (chronological, {len(shown)} of {len(tool_lines)} shown):"]
        if omitted:
            log.append(f"  tool_events_omitted={omitted} (oldest first; full evidence "
                       f"remains in the Executor result)")
        return "\n".join(head + log + shown + tail)

    text = render(tool_lines, 0)
    if len(text) > MAX_TURN_CHARS:
        # Keep the most recent tool-call lines that fit; state how many were dropped.
        kept: list[str] = []
        for line in reversed(tool_lines):
            if len(render([line] + kept, len(tool_lines) - len(kept) - 1)) > MAX_TURN_CHARS:
                break
            kept.insert(0, line)
        text = render(kept, len(tool_lines) - len(kept))
    _check_safe(text)
    return {"role": EXECUTOR_ROLE, "text": text}
