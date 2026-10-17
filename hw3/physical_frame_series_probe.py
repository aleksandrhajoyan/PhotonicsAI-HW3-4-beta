"""Manual, operator-authorized diagnostic: one bounded frame series from the physical iRAYPLE camera.

This is a MANUAL hardware diagnostic, not an AI-approved MCP session: no language model,
MCP server, SessionGate or network is involved, and nothing is approved automatically.

Nothing is loaded from the vendor SDK on import. main() touches the camera only with
--physical-diagnostic, valid limits, an interactive terminal and the exact typed phrase
(which contains the frame count). Then: irayple_camera.preload_vendor_sdk() once, one
IRaypleCamera instance, and through its public methods only: open (exactly one USB3
MV-A7200MU130, ambiguous selection refused by the backend), get_info, a check that the
already verified ExposureTime lies in [1000, 5000] us, exactly one capture_series(), and
close() in a mandatory finally block. No device setting is written: exposure, gain, frame
rate, ROI and trigger mode are left exactly as found. Nothing is retried or reopened.

After close(), the series is summarized and passed to temporal_analysis on HOST
timestamps only. Raw device timestamps are reported as unscaled ticks of unknown unit;
no device clock rate or device FPS is derived. A completed acquisition is not a
successful frequency recovery: the analyzer status and warnings are reported unchanged.

With --output PATH, per-frame scalar samples and diagnostic metadata are written to a new
local JSON file (never overwritten; no images, pixels, serial numbers or identities).
Export happens only after close() and never affects cleanup.

If the process is killed, the finally block does not run; check the camera in MV Viewer.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import irayple_camera as ic
import irayple_discovery as discovery
import irayple_first_frame as first_frame
import temporal_analysis as ta
from frame_series_contract import SOURCE_HARDWARE, STATUS_COMPLETED, FrameSeries

EXPECTED_MODEL = ic.EXPECTED_MODEL
EXPECTED_BACKEND = "IRaypleCamera"
DEFAULT_FRAMES = 64
MIN_FRAMES = ta.MIN_FRAMES  # fewer frames than this get no timing statistics from the analyzer
MAX_FRAMES = 384  # manual CLI cap only; the backend's MAX_SERIES_FRAMES and 5 s ceiling stay authoritative
DEFAULT_MAX_DURATION_S = ic.MAX_SERIES_DURATION_S
MAX_DURATION_S = ic.MAX_SERIES_DURATION_S
DEFAULT_FRAME_TIMEOUT_MS = 1000
MAX_FRAME_TIMEOUT_MS = ic.MAX_SERIES_FRAME_TIMEOUT_MS
MIN_EXPOSURE_US = 1000.0
MAX_EXPOSURE_US = 5000.0
EXPORT_SCHEMA = "lumen.frame_series_probe.v1"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_CLEANUP_FAILED = 4
EXIT_PRECONDITION = 5
EXIT_EXPORT_FAILED = 7
EXIT_INTERRUPTED = 130

STATUS_ACQUIRED = "SERIES_ACQUIRED"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
PRECONDITION_STATUSES = ("UNEXPECTED_BACKEND_INFO", "EXPOSURE_OUT_OF_RANGE")

TIMESTAMP_NOTE = ("device timestamps are raw ticks of UNKNOWN unit; no device clock rate, "
                  "device FPS or physical device time is reported")


def confirmation_phrase(n_frames):
    return f"CAPTURE {n_frames} FRAMES"


class ProbeError(Exception):
    """Controlled refusal or validation failure."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class ProbeResult:
    def __init__(self, n_frames, frame_timeout_ms, max_duration_s):
        self.n_frames = n_frames
        self.frame_timeout_ms = frame_timeout_ms
        self.max_duration_s = max_duration_s
        self.status = STATUS_UNEXPECTED_ERROR
        self.message = ""
        self.model = None
        self.exposure_us = None
        self.capture_calls = 0
        self.series = None
        self.restore_required = None
        self.close_attempted = False
        self.close_error = None
        self.interrupted = False

    @property
    def close_ok(self):
        return self.close_attempted and self.close_error is None

    @property
    def ok(self):
        return (self.status == STATUS_ACQUIRED and not self.interrupted and self.series is not None
                and self.capture_calls == 1 and self.close_ok)


# -- hardware procedure ----------------------------------------------------------

def _verify_device(info):
    expected = {"backend": EXPECTED_BACKEND, "source": SOURCE_HARDWARE, "synthetic": False,
                "model": EXPECTED_MODEL, "is_open": True}
    for key, value in expected.items():
        if info.get(key) != value or type(info.get(key)) is not type(value):
            raise ProbeError("UNEXPECTED_BACKEND_INFO",
                             f"get_info()[{key!r}] is {info.get(key)!r}, expected {value!r}")


def _verify_exposure(exposure_us):
    if (isinstance(exposure_us, bool) or not isinstance(exposure_us, float) or not math.isfinite(exposure_us)
            or not MIN_EXPOSURE_US <= exposure_us <= MAX_EXPOSURE_US):
        raise ProbeError("EXPOSURE_OUT_OF_RANGE",
                         f"verified ExposureTime {exposure_us!r} us is not in [{MIN_EXPOSURE_US}, "
                         f"{MAX_EXPOSURE_US}] us; nothing was written or acquired")


def validate_series(series, n_frames, exposure_us):
    """Accept only a completed hardware FrameSeries of the requested length and verified exposure."""
    if not isinstance(series, FrameSeries):
        raise ProbeError("INVALID_SERIES", f"capture_series returned {type(series).__name__}, not FrameSeries")
    checks = (
        ("source", series.source, SOURCE_HARDWARE),
        ("status", series.status, STATUS_COMPLETED),
        ("requested_frames", series.requested_frames, n_frames),
        ("acquired_frames", series.acquired_frames, n_frames),
        ("exposure_us", series.exposure_us, exposure_us),
        ("synthetic_device_tick_hz", series.synthetic_device_tick_hz, None),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise ProbeError("INVALID_SERIES", f"series {name} is {actual!r}, expected {expected!r}")
    if not (series.has_block_ids and series.has_device_timestamps):
        raise ProbeError("INVALID_SERIES", "series lacks native block ids or raw device timestamps")


def _run(camera, result):
    camera.open()
    info = camera.get_info()
    result.model = info.get("model")
    result.exposure_us = info.get("exposure_us")
    _verify_device(info)
    _verify_exposure(result.exposure_us)
    result.capture_calls += 1
    series = camera.capture_series(result.n_frames, frame_timeout_ms=result.frame_timeout_ms,
                                   max_duration_s=result.max_duration_s)
    validate_series(series, result.n_frames, result.exposure_us)
    result.series = series
    result.status = STATUS_ACQUIRED


def run_probe(camera, n_frames, frame_timeout_ms, max_duration_s):
    """Run the single-series procedure on one backend instance. close() is always attempted."""
    result = ProbeResult(n_frames, frame_timeout_ms, max_duration_s)
    try:
        _run(camera, result)
    except ProbeError as exc:
        result.status, result.message = exc.status, str(exc)
    except ic.IRaypleCameraError as exc:
        result.status, result.message = exc.status, str(exc)
    except KeyboardInterrupt:
        result.interrupted = True
        result.status, result.message = STATUS_INTERRUPTED, "interrupted before the series completed"
    except Exception as exc:
        result.status, result.message = STATUS_UNEXPECTED_ERROR, repr(exc)
    finally:
        try:
            result.restore_required = camera.get_info().get("exposure_restore_required")
        except Exception:  # get_info() makes no SDK call; a failure here must not skip close()
            result.restore_required = None
        result.close_attempted = True
        try:
            camera.close()
        except KeyboardInterrupt:
            result.interrupted = True
            result.close_error = "interrupted by KeyboardInterrupt during close(); camera state unknown"
        except Exception as exc:
            result.close_error = exc
    return result


# -- diagnostics (pure; after close) -----------------------------------------------

def _int_stats(values):
    if len(values) == 0:
        return None
    return {"min": int(min(values)), "median": float(np.median(values)), "max": int(max(values))}


def summarize(series, analysis):
    """Scalar diagnostics of a completed series. Host timing comes from temporal_analysis."""
    samples = series.samples
    block_ids = [s.native_block_id for s in samples]
    ticks = [s.device_timestamp_raw for s in samples]
    host = [s.host_t_ns for s in samples]
    means = [s.mean_dn for s in samples]
    acquisition, timing = analysis["acquisition"], analysis["timing"]
    return {
        "frames": series.acquired_frames,
        "requested_frames": series.requested_frames,
        "source": series.source,
        "completion_status": series.status,
        "exposure_us": series.exposure_us,
        "width": series.width,
        "height": series.height,
        "bit_depth": series.bit_depth,
        "first_block_id": block_ids[0],
        "last_block_id": block_ids[-1],
        "missing_block_ids": series.dropped_frames_by_block_id,
        "first_device_timestamp_raw": ticks[0],
        "last_device_timestamp_raw": ticks[-1],
        "device_timestamp_span_raw": series.device_tick_span,
        "device_timestamp_delta_raw": _int_stats([b - a for a, b in zip(ticks, ticks[1:])]),
        "device_timestamp_unit": "unknown",
        "first_host_t_ns": host[0],
        "last_host_t_ns": host[-1],
        "host_span_ns": series.host_span_ns,
        "host_fps": acquisition["fps_host"],
        "host_median_interval_s": timing["median_interval_s"],
        "host_interval_std_s": timing["interval_std_s"],
        "host_jitter_rel": timing["jitter_rel"],
        "host_min_interval_s": timing["min_interval_s"],
        "host_max_interval_s": timing["max_interval_s"],
        "mean_dn_min": min(means),
        "mean_dn_max": max(means),
        "mean_dn_average": float(np.mean(means)),
        "saturation_fraction_max": max(s.saturation_fraction for s in samples),
    }


def analyze(series):
    """temporal_analysis on host timestamps; raw device ticks are passed without any tick rate."""
    return ta.analyze_temporal_series(**series.analyzer_inputs())


def _fmt(value, spec=""):
    return "n/a" if value is None else format(value, spec)


def format_diagnostics(d, analysis):
    deltas = d["device_timestamp_delta_raw"]
    delta_text = "n/a" if deltas is None else f"min {deltas['min']}, median {deltas['median']:g}, max {deltas['max']}"
    lines = [
        f"Frames:                 {d['frames']} of {d['requested_frames']} requested",
        f"Source provenance:      {d['source']}",
        f"Completion status:      {d['completion_status']}",
        f"Verified exposure:      {d['exposure_us']!r} us (unchanged by this diagnostic)",
        f"Frame size:             {d['width']} x {d['height']}, {d['bit_depth']}-bit",
        f"Native block IDs:       first {d['first_block_id']}, last {d['last_block_id']}, "
        f"missing {d['missing_block_ids']}",
        f"Raw device timestamps:  first {d['first_device_timestamp_raw']}, last {d['last_device_timestamp_raw']} "
        f"(unit UNKNOWN)",
        f"Raw timestamp deltas:   {delta_text} ticks of unknown unit",
        f"Host timestamps (ns):   first {d['first_host_t_ns']}, last {d['last_host_t_ns']} (perf_counter_ns, "
        "arrival time)",
        f"Host-observed FPS:      {_fmt(d['host_fps'], '.4f')}",
        f"Host median interval:   {_fmt(None if d['host_median_interval_s'] is None else d['host_median_interval_s'] * 1e3, '.4f')} ms",
        f"Host timing jitter:     std {_fmt(None if d['host_interval_std_s'] is None else d['host_interval_std_s'] * 1e3, '.4f')} ms, "
        f"std/median {_fmt(d['host_jitter_rel'], '.4%')}",
        f"Mean DN:                min {d['mean_dn_min']:.4f}, max {d['mean_dn_max']:.4f}, "
        f"average {d['mean_dn_average']:.4f}",
        f"Max saturation:         {d['saturation_fraction_max']:.6f} (pixels == 255 DN)",
        f"Note:                   {TIMESTAMP_NOTE}",
        f"TEMPORAL_ANALYSIS_STATUS: {analysis['status']} (frequency recovery is separate from acquisition)",
    ]
    f0 = analysis["estimate"]["f0_hz"]
    if f0 is not None:
        lines.append(f"Host-timed f0 estimate: {f0:.4f} Hz +/- {analysis['estimate']['tolerance_hz']:.4f} Hz")
    for warning in analysis["warnings"]:
        lines.append(f"  warning {warning['code']}: {warning['message']}")
    return lines


def build_export(result, diagnostics, analysis):
    """JSON-compatible export: scalar samples and metadata only. No serial, identity or pixels."""
    return {
        "schema": EXPORT_SCHEMA,
        "kind": "manual_physical_diagnostic",
        "ai_approved_session": False,
        "model": result.model,
        "limits": {"n_frames": result.n_frames, "frame_timeout_ms": result.frame_timeout_ms,
                   "max_duration_s": result.max_duration_s},
        "cleanup": "ok" if result.close_ok else "failed",
        "device_timestamp_note": TIMESTAMP_NOTE,
        "diagnostics": diagnostics,
        "temporal_analysis": analysis,
        "samples": [
            {"frame_index": s.frame_index, "native_block_id": s.native_block_id,
             "device_timestamp_raw": s.device_timestamp_raw, "host_t_ns": s.host_t_ns,
             "mean_dn": s.mean_dn, "saturation_fraction": s.saturation_fraction}
            for s in result.series.samples
        ],
    }


def write_export(path, document):
    """Create a NEW file; an existing file is never overwritten (exclusive-create mode)."""
    text = json.dumps(document, indent=2, allow_nan=False)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text + "\n")


# -- console --------------------------------------------------------------------

def _close_line(result):
    if not result.close_attempted:
        return "NOT ATTEMPTED"
    error = result.close_error
    if error is None:
        return "OK (backend close() confirmed)"
    if isinstance(error, (ic.IRaypleCloseError, str)):
        return f"FAILED: {error}"
    return f"FAILED: {error!r}"


def format_plan(n_frames, frame_timeout_ms, max_duration_s, output):
    return [
        "MANUAL PHYSICAL DIAGNOSTIC. This is NOT an AI-approved MCP session; no language model,",
        "MCP server or session gate is involved, and nothing is approved automatically.",
        f"Device required:   exactly one USB3 {EXPECTED_MODEL}, via the existing IRaypleCamera backend",
        "Planned action:    preload the vendor SDK once, open the camera (TriggerMode, ExposureAuto and",
        "                   GainAuto must read 'Off'), then ONE frame series with capture_series()",
        f"Frame count:       {n_frames} frames (hard CLI maximum {MAX_FRAMES})",
        f"Duration ceiling:  {max_duration_s!r} s for the whole series (hard maximum {MAX_DURATION_S!r} s)",
        f"Frame timeout:     {frame_timeout_ms} ms per frame, further limited by the remaining time",
        f"Exposure:          the current verified ExposureTime must already be in [{MIN_EXPOSURE_US}, "
        f"{MAX_EXPOSURE_US}] us; otherwise the diagnostic aborts",
        "Settings:          NO device setting is written (exposure, gain, frame rate, ROI, trigger mode)",
        "Possible effects:  the camera streams for up to the duration ceiling; another application",
        "                   using the camera (e.g. MV Viewer) must be closed; close() always follows",
        "No retries and no automatic reopening. A forcibly killed process cannot run close().",
        f"Export:            {'new JSON file ' + str(output) if output else 'none (console output only)'}",
    ]


def exit_code_for(result):
    if result.ok:
        return EXIT_OK
    if result.interrupted:
        return EXIT_INTERRUPTED
    if not result.close_ok:
        return EXIT_CLEANUP_FAILED
    if result.status in PRECONDITION_STATUSES:
        return EXIT_PRECONDITION
    return EXIT_FAILED


def _check_request(args):
    """Pure preliminary checks; touches no SDK, camera or file content."""
    if not MIN_FRAMES <= args.frames <= MAX_FRAMES:
        return f"--frames must be in [{MIN_FRAMES}, {MAX_FRAMES}], got {args.frames}"
    if not (math.isfinite(args.max_duration_s) and 0 < args.max_duration_s <= MAX_DURATION_S):
        return f"--max-duration-s must be in (0, {MAX_DURATION_S}], got {args.max_duration_s}"
    if not 1 <= args.frame_timeout_ms <= MAX_FRAME_TIMEOUT_MS:
        return f"--frame-timeout-ms must be in [1, {MAX_FRAME_TIMEOUT_MS}], got {args.frame_timeout_ms}"
    if args.output is not None:
        path = Path(args.output)
        if path.suffix.lower() != ".json":
            return f"--output must name a .json file, got {args.output!r}"
        if path.exists():
            return f"--output {args.output!r} already exists; it is never overwritten"
        if not path.parent.is_dir():
            return f"--output directory {str(path.parent)!r} does not exist; it is not created"
    return None


def main(argv=None, input_fn=input, stdin=None):
    parser = argparse.ArgumentParser(
        description="Manual one-series frame diagnostic on the iRAYPLE MV-A7200MU130 (operator authorized)."
    )
    parser.add_argument("--physical-diagnostic", action="store_true",
                        help="required: run the single physical frame series (interactive confirmation follows)")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help=f"frames to acquire, {MIN_FRAMES}..{MAX_FRAMES} (default {DEFAULT_FRAMES})")
    parser.add_argument("--max-duration-s", type=float, default=DEFAULT_MAX_DURATION_S,
                        help=f"hard series duration ceiling, (0, {MAX_DURATION_S}] s")
    parser.add_argument("--frame-timeout-ms", type=int, default=DEFAULT_FRAME_TIMEOUT_MS,
                        help=f"per-frame SDK GetFrame timeout, 1..{MAX_FRAME_TIMEOUT_MS} ms")
    parser.add_argument("--output", default=None,
                        help="optional NEW local .json file for scalar samples and diagnostics")
    args = parser.parse_args(argv)

    print("LUMEN physical frame-series diagnostic (one camera, one series, manual authorization)")
    if not args.physical_diagnostic:
        print("NO_ACTION: pass --physical-diagnostic to run. No SDK was loaded and no camera was accessed.")
        return EXIT_REFUSED
    problem = _check_request(args)
    if problem is not None:
        print(f"REFUSED: {problem}. No SDK was loaded and no camera was accessed.")
        return EXIT_REFUSED
    stdin = sys.stdin if stdin is None else stdin
    if not first_frame._is_interactive(stdin):
        print("REFUSED: an interactive terminal is required for confirmation. No camera was accessed.")
        return EXIT_REFUSED

    for line in format_plan(args.frames, args.frame_timeout_ms, args.max_duration_s, args.output):
        print(line)
    phrase = confirmation_phrase(args.frames)
    try:
        response = input_fn(f"Type {phrase} to authorize this single diagnostic: ")
    except (EOFError, KeyboardInterrupt):
        print("\nREFUSED: no confirmation received. No camera was accessed.")
        return EXIT_REFUSED
    if response != phrase:
        print("REFUSED: confirmation phrase did not match. No camera was accessed.")
        return EXIT_REFUSED

    try:
        sdk = ic.preload_vendor_sdk()
    except discovery.DiscoveryError as exc:
        print(f"PROBE_FAILURE: SDK_LOAD_FAILED: {exc}")
        print("DIAGNOSTIC_STATUS: FAILED")
        return EXIT_FAILED
    camera = ic.IRaypleCamera(sdk)
    try:
        result = run_probe(camera, args.frames, args.frame_timeout_ms, args.max_duration_s)
    except KeyboardInterrupt:
        print("PROBE_FAILURE: INTERRUPTED during close(); camera state unknown.")
        print("DIAGNOSTIC_STATUS: FAILED")
        return EXIT_INTERRUPTED

    # The camera is closed (or its close failure recorded) before anything below runs.
    print(f"Model:                  {result.model or 'not reached'}")
    print(f"capture_series calls:   {result.capture_calls}")
    analysis = diagnostics = None
    if result.series is not None:
        try:
            analysis = analyze(result.series)
            diagnostics = summarize(result.series, analysis)
            lines = format_diagnostics(diagnostics, analysis)
        except Exception as exc:  # pure post-processing; the camera is already closed
            print(f"PROBE_FAILURE: ANALYSIS_FAILED: {exc!r}")
            diagnostics = None
        else:
            for line in lines:
                print(line)
    print(f"Exposure writes:        {'none' if result.restore_required is False else 'UNVERIFIED'}")
    print(f"Cleanup status:         {_close_line(result)}")
    if result.status != STATUS_ACQUIRED:
        print(f"PROBE_FAILURE: {result.status}: {result.message}")
    print(f"ACQUISITION_STATUS:     {'COMPLETED' if result.ok else 'FAILED'}")

    code = exit_code_for(result)
    if code == EXIT_OK and diagnostics is None:
        code = EXIT_FAILED
    if code == EXIT_OK and args.output is not None:
        try:
            write_export(args.output, build_export(result, diagnostics, analysis))
            print(f"Export:                 wrote new file {args.output}")
        except (OSError, ValueError, TypeError) as exc:
            print(f"Export:                 FAILED ({exc!r}); camera cleanup was not affected")
            code = EXIT_EXPORT_FAILED
    print(f"DIAGNOSTIC_STATUS: {'PASS' if code == EXIT_OK else 'FAILED'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
