"""Offline converter: one Frame Lab physical diagnostic (lumen.frame_series_probe.v1) -> one lumen.flicker.v1 record.

Pure bookkeeping on JSON documents: this module never touches a camera, a vendor SDK, MCP, the network or a
language model, imports only the standard library, never re-runs the frequency analysis and never changes its
input. The contract authority is the Event Lab comparator (agentic_ode/dualvision/comparison.py, assess()) and
its baseline records; every field it requires is produced with the type it checks, except that Frame Lab records
use the frame-count request representation (requested_s null, DV-05C), which the comparator must accept.

Entry points:

    to_flicker_record(document, *, local_time, reference_hz=None, distance_cm=None, trial_id=None) -> dict
    main(argv=None)   CLI: python -B -m hw3.frame_flicker_export --input IN.json --output NEW.json
                           --local-time ISO8601 (--reference-hz HZ | --no-reference)
                           [--distance-cm CM] [--trial-id ID]

Field mapping (input -> lumen.flicker.v1):
    schema, lab                 "lumen.flicker.v1", "frame"
    run_id, recording_id        "frame-" + first 16 hex digits of the SHA-256 of the canonical input JSON.
                                One probe invocation acquires exactly one series, so run_id == recording_id.
                                Stable and non-sensitive; no serial number or file path is used.
    local_time                  OPERATOR INPUT, ISO 8601 with a UTC offset. The input has no wall-clock
                                time (host_t_ns is perf_counter_ns); file metadata is never used.
    hardware                    {"model": model, "sensor": null, "real": diagnostics.source == "hardware"}.
                                The input has no sensor name and no serial; no serial key is written.
    source                      type "LED_PWM" (the shared Dual-Vision stimulus); reference_hz ONLY from the
                                operator, else null (never inferred from f0_hz); duty null (not recorded);
                                distance_cm from the operator or null; set_by "manual" with a reference,
                                else "not_supplied".
    acquisition.request_mode    "frame_count": a Frame Lab measurement requests N FRAMES, not a duration.
    acquisition.requested_frames diagnostics.requested_frames, verified equal to limits.n_frames and, for a
                                completed series, to diagnostics.frames (== acquisition.samples).
    acquisition.requested_s     null. No duration was requested; none is derived from the measured fps, and
                                limits.max_duration_s (a safety deadline) is never used. Basis recorded in
                                acquisition.requested_s_basis.
    acquisition.actual_s        temporal_analysis.acquisition.duration_s: host arrival time first -> last frame
    acquisition.samples         diagnostics.frames (delivered frames)
    acquisition.sample_kind     "frames"
    acquisition.fps             temporal_analysis.acquisition.fps (measured host fps, == diagnostics.host_fps)
    acquisition.exposure_us     diagnostics.exposure_us (verified readback, unchanged by the probe)
    acquisition.early_stop      false (only completed series are accepted)
    estimate.f0_hz              temporal_analysis.estimate.f0_hz, or null (never 0)
    estimate.resolution_hz      temporal_analysis.estimate.resolution_hz, or null
    estimate.snr                temporal_analysis.estimate.snr unchanged: the SPECTRAL POWER ratio of the
                                strongest significant peak (which may be a harmonic, not f0), or null
    status, warnings            temporal_analysis.status and its structured {"code", "message"} warnings,
                                copied unchanged and in order
    analysis                    plain-text description of where the record comes from

Extra keys (kept for provenance): acquisition.requested_s_basis, acquisition.timestamp_source, estimate.method, estimate.snr_definition,
estimate.snr_basis, estimate.tolerance_hz, estimate.peaks, estimate.harmonics_hz, source.reference_provenance,
and a top-level "provenance" object (real vs synthetic, manual diagnostic vs AI session, original acquisition
vs offline export, input hash). trial_id is written only when the operator supplies one; it is never generated.

Rejected inputs (ExportError): anything that is not a complete, internally consistent probe export of a
completed, cleanly closed acquisition: wrong schema or kind, an AI-approved session flag, failed cleanup,
partial or failed series, frame-count mismatches, non-finite or out-of-range numbers, an f0 on a non-"ok"
status (or none on "ok"), malformed warnings, or timing that disagrees with the per-frame samples.
"""

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

SCHEMA = "lumen.flicker.v1"
INPUT_SCHEMA = "lumen.frame_series_probe.v1"
INPUT_KIND = "manual_physical_diagnostic"
LAB = "frame"
SAMPLE_KIND = "frames"
SOURCE_TYPE = "LED_PWM"
TIMESTAMP_SOURCE = "host_monotonic"
DATA_SOURCES = ("hardware", "synthetic")
ANALYZER_STATUSES = ("ok", "ambiguous", "no_flicker_detected", "insufficient_data", "irregular_sampling",
                     "invalid_input")
WARNING_KEYS = frozenset({"code", "message"})
SNR_DEFINITION = "spectral power ratio, Frame Lab DV-01"  # the comparator's wording for frame snr
SNR_BASIS = ("power of the strongest significant spectral peak divided by the estimated mean noise power; "
             "the strongest peak may be a harmonic, not necessarily f0")
REQUEST_MODE = "frame_count"
REQUESTED_S_BASIS = ("the probe requests a frame count, not a duration, so requested_s is null; actual_s is the "
                     "observed host duration. limits.max_duration_s is a safety deadline, not a requested "
                     "duration")
ANALYSIS_TEXT = ("offline export of the original DV-01 temporal analysis recorded by a manual Frame Lab "
                 "diagnostic (lumen.frame_series_probe.v1); host arrival timestamps; the analysis was not "
                 "re-run and reference_hz played no part in it")
REFERENCE_PROVENANCE = ("PWM generator setting entered by the operator; not a calibrated optical measurement "
                        "and not inferred from the estimate")
TIMING_REL_TOL = 1e-9
TIMING_ABS_TOL_S = 1e-9

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_INVALID_INPUT = 3


class ExportError(ValueError):
    """The input or an operator value cannot be converted truthfully."""


# -- small validators -------------------------------------------------------------

def _get(document, path):
    value = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ExportError(f"{path} is missing")
        value = value[key]
    return value


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if not _is_number(value) or not math.isfinite(value) or value <= 0:
        raise ExportError(f"{name} must be a finite number > 0, got {value!r}")
    return value


def _non_negative(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if not _is_number(value) or not math.isfinite(value) or value < 0:
        raise ExportError(f"{name} must be a finite number >= 0, got {value!r}")
    return value


def _count(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExportError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"{name} must be a non-empty string")
    return value


def _warnings(raw):
    if not isinstance(raw, list):
        raise ExportError("temporal_analysis.warnings must be a list")
    for i, item in enumerate(raw):
        if not (isinstance(item, dict) and set(item) == WARNING_KEYS
                and all(isinstance(item[k], str) and item[k].strip() for k in WARNING_KEYS)):
            raise ExportError(f"temporal_analysis.warnings[{i}] is not a {{code, message}} object")
    return copy.deepcopy(raw)


def input_digest(document):
    """SHA-256 of the canonical JSON form of the input (key order and whitespace do not matter)."""
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- input validation --------------------------------------------------------------

def _validate_samples(samples, frames, duration_s, fps):
    if not isinstance(samples, list) or len(samples) != frames:
        raise ExportError(f"samples must be a list of {frames} frames")
    host = []
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ExportError(f"samples[{i}] is not an object")
        if _get(sample, "frame_index") != i or isinstance(sample["frame_index"], bool):
            raise ExportError(f"samples[{i}].frame_index is not {i}")
        t = _get(sample, "host_t_ns")
        if not isinstance(t, int) or isinstance(t, bool) or t < 0:
            raise ExportError(f"samples[{i}].host_t_ns must be an integer >= 0")
        if host and t <= host[-1]:
            raise ExportError(f"samples[{i}].host_t_ns is not strictly increasing")
        host.append(t)
        _non_negative(_get(sample, "mean_dn"), f"samples[{i}].mean_dn")
        fraction = _non_negative(_get(sample, "saturation_fraction"), f"samples[{i}].saturation_fraction")
        if fraction > 1:
            raise ExportError(f"samples[{i}].saturation_fraction exceeds 1")
    span_s = (host[-1] - host[0]) * 1e-9
    if not math.isclose(span_s, duration_s, rel_tol=TIMING_REL_TOL, abs_tol=TIMING_ABS_TOL_S):
        raise ExportError(f"duration_s {duration_s!r} does not match the sample host span {span_s!r} s")
    if not math.isclose(fps, (frames - 1) / duration_s, rel_tol=TIMING_REL_TOL):
        raise ExportError(f"fps {fps!r} does not match (frames - 1) / duration_s")


def validate_probe_document(document):
    """Check a lumen.frame_series_probe.v1 document; return the values the record needs. Raises ExportError."""
    if not isinstance(document, dict):
        raise ExportError("the input is not a JSON object")
    if document.get("schema") != INPUT_SCHEMA:
        raise ExportError(f"schema is not {INPUT_SCHEMA}")
    if document.get("kind") != INPUT_KIND:
        raise ExportError(f"kind is not {INPUT_KIND}")
    if document.get("ai_approved_session") is not False:
        raise ExportError("ai_approved_session must be false for a manual diagnostic")
    if document.get("cleanup") != "ok":
        raise ExportError("cleanup is not 'ok': the acquisition did not end cleanly")
    model = _text(document.get("model"), "model")

    limit_frames = _count(_get(document, "limits.n_frames"), "limits.n_frames")
    d = _get(document, "diagnostics")
    if not isinstance(d, dict):
        raise ExportError("diagnostics is not an object")
    if _get(d, "completion_status") != "completed":
        raise ExportError("diagnostics.completion_status is not 'completed': partial or failed acquisition")
    frames = _count(_get(d, "frames"), "diagnostics.frames")
    requested = _count(_get(d, "requested_frames"), "diagnostics.requested_frames")
    if not frames == requested == limit_frames:
        raise ExportError(f"frame counts disagree: acquired {frames}, requested {requested}, limit {limit_frames}")
    data_source = _get(d, "source")
    if data_source not in DATA_SOURCES:
        raise ExportError(f"diagnostics.source must be one of {DATA_SOURCES}")
    exposure_us = _positive(_get(d, "exposure_us"), "diagnostics.exposure_us")
    host_fps = _positive(_get(d, "host_fps"), "diagnostics.host_fps")
    missing = _get(d, "missing_block_ids")
    if missing is not None and (not isinstance(missing, int) or isinstance(missing, bool) or missing < 0):
        raise ExportError("diagnostics.missing_block_ids must be null or an integer >= 0")

    ta = _get(document, "temporal_analysis")
    if not isinstance(ta, dict):
        raise ExportError("temporal_analysis is not an object")
    status = _get(ta, "status")
    if status not in ANALYZER_STATUSES:
        raise ExportError(f"temporal_analysis.status must be one of {ANALYZER_STATUSES}")
    warnings = _warnings(_get(ta, "warnings"))
    if _get(ta, "acquisition.sample_kind") != SAMPLE_KIND:
        raise ExportError("temporal_analysis.acquisition.sample_kind is not 'frames'")
    if _get(ta, "acquisition.n_frames") != frames:
        raise ExportError("temporal_analysis.acquisition.n_frames does not match diagnostics.frames")
    if _get(ta, "acquisition.timestamp_source") != TIMESTAMP_SOURCE:
        raise ExportError(f"temporal_analysis.acquisition.timestamp_source is not {TIMESTAMP_SOURCE!r}")
    fps = _positive(_get(ta, "acquisition.fps"), "temporal_analysis.acquisition.fps")
    duration_s = _positive(_get(ta, "acquisition.duration_s"), "temporal_analysis.acquisition.duration_s")
    if fps != host_fps or _get(ta, "acquisition.fps_host") != host_fps:
        raise ExportError("temporal_analysis fps does not equal diagnostics.host_fps")

    f0 = _positive(_get(ta, "estimate.f0_hz"), "temporal_analysis.estimate.f0_hz", nullable=True)
    resolution = _positive(_get(ta, "estimate.resolution_hz"), "temporal_analysis.estimate.resolution_hz",
                           nullable=True)
    tolerance = _positive(_get(ta, "estimate.tolerance_hz"), "temporal_analysis.estimate.tolerance_hz",
                          nullable=True)
    snr = _non_negative(_get(ta, "estimate.snr"), "temporal_analysis.estimate.snr", nullable=True)
    if status == "ok" and (f0 is None or resolution is None):
        raise ExportError("status 'ok' requires f0_hz and resolution_hz")
    if status != "ok" and f0 is not None:
        raise ExportError(f"status {status!r} must have f0_hz null")
    peaks = _get(ta, "estimate.peaks")
    if not isinstance(peaks, list):
        raise ExportError("temporal_analysis.estimate.peaks must be a list")
    for i, peak in enumerate(peaks):
        if not isinstance(peak, dict):
            raise ExportError(f"temporal_analysis.estimate.peaks[{i}] is not an object")
        _positive(_get(peak, "frequency_hz"), f"temporal_analysis.estimate.peaks[{i}].frequency_hz")
        _non_negative(_get(peak, "snr"), f"temporal_analysis.estimate.peaks[{i}].snr")
    method = ta["estimate"].get("method")
    if method is not None and not isinstance(method, str):
        raise ExportError("temporal_analysis.estimate.method must be a string or null")

    _validate_samples(_get(document, "samples"), frames, duration_s, fps)
    return {
        "model": model, "frames": frames, "requested_frames": requested, "data_source": data_source, "exposure_us": exposure_us,
        "fps": fps, "duration_s": duration_s, "status": status, "warnings": warnings, "f0": f0,
        "resolution": resolution, "tolerance": tolerance, "snr": snr, "peaks": copy.deepcopy(peaks),
        "method": method,
    }


# -- operator values -----------------------------------------------------------------

def _local_time(value):
    _text(value, "local_time")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ExportError(f"local_time {value!r} is not ISO 8601") from None
    if parsed.tzinfo is None:
        raise ExportError("local_time must include a UTC offset, e.g. 2026-10-16T18:06:00+04:00")
    return value


def _optional_positive(value, name):
    return None if value is None else _positive(value, name)


def _trial_id(value):
    if value is None:
        return None
    return _text(value, "trial_id")


# -- conversion ------------------------------------------------------------------------

def to_flicker_record(document, *, local_time, reference_hz=None, distance_cm=None, trial_id=None):
    """Convert one probe export into a new lumen.flicker.v1 dict. The input is never modified."""
    v = validate_probe_document(document)
    local_time = _local_time(local_time)
    reference_hz = _optional_positive(reference_hz, "reference_hz")
    distance_cm = _optional_positive(distance_cm, "distance_cm")
    trial_id = _trial_id(trial_id)
    digest = input_digest(document)
    record_id = f"frame-{digest[:16]}"
    real = v["data_source"] == "hardware"
    if v["requested_frames"] != v["frames"]:  # a null requested_s must never hide an incomplete capture
        raise ExportError(f"requested {v['requested_frames']} frames but {v['frames']} were delivered")

    harmonics = None
    if v["status"] == "ok":  # every other significant peak was explained as a harmonic of f0
        harmonics = sorted(p["frequency_hz"] for p in v["peaks"] if p["frequency_hz"] != v["f0"])

    record = {
        "schema": SCHEMA,
        "lab": LAB,
        "run_id": record_id,
        "recording_id": record_id,
        "local_time": local_time,
        "hardware": {"model": v["model"], "sensor": None, "real": real},
        "source": {
            "type": SOURCE_TYPE,
            "reference_hz": reference_hz,
            "duty": None,
            "distance_cm": distance_cm,
            "set_by": "manual" if reference_hz is not None else "not_supplied",
            "reference_provenance": REFERENCE_PROVENANCE if reference_hz is not None else None,
        },
        "acquisition": {
            "request_mode": REQUEST_MODE,
            "requested_frames": v["requested_frames"],
            "requested_s": None,
            "actual_s": v["duration_s"],
            "samples": v["frames"],
            "sample_kind": SAMPLE_KIND,
            "fps": v["fps"],
            "exposure_us": v["exposure_us"],
            "early_stop": False,
            "requested_s_basis": REQUESTED_S_BASIS,
            "timestamp_source": TIMESTAMP_SOURCE,
        },
        "estimate": {
            "f0_hz": v["f0"],
            "resolution_hz": v["resolution"],
            "snr": v["snr"],
            "snr_definition": SNR_DEFINITION,
            "snr_basis": SNR_BASIS,
            "tolerance_hz": v["tolerance"],
            "method": v["method"],
            "peaks": v["peaks"],
            "harmonics_hz": harmonics,
        },
        "status": v["status"],
        "warnings": v["warnings"],
        "analysis": ANALYSIS_TEXT,
        "provenance": {
            "exporter": "hw3/frame_flicker_export.py (DV-05C)",
            "input_schema": INPUT_SCHEMA,
            "input_kind": INPUT_KIND,
            "input_sha256": digest,
            "data_origin": "real_hardware" if real else "synthetic",
            "acquisition_mode": "manual_operator_diagnostic",
            "ai_approved_session": False,
            "record_origin": "offline_export_of_original_acquisition",
            "analysis_rerun": False,
            "local_time_source": "operator",
            "reference_source": "operator" if reference_hz is not None else None,
            "distance_source": "operator" if distance_cm is not None else None,
            "trial_id_source": "operator" if trial_id is not None else None,
            "device_timestamp_unit": "unknown",
        },
    }
    if trial_id is not None:
        record["trial_id"] = trial_id
    return record


# -- files -----------------------------------------------------------------------------

def _reject_constant(name):
    raise ExportError(f"the input contains the non-standard JSON number {name}")


def load_document(path):
    """Read a JSON document; NaN and Infinity literals are rejected."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def write_record(path, record):
    """Create a NEW file; an existing file is never overwritten (exclusive-create mode)."""
    text = json.dumps(record, indent=2, allow_nan=False, ensure_ascii=False)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _check_paths(input_path, output_path):
    if not input_path.is_file():
        return f"--input {str(input_path)!r} is not an existing file"
    if output_path.suffix.lower() != ".json":
        return f"--output must name a .json file, got {str(output_path)!r}"
    if output_path.exists():
        return f"--output {str(output_path)!r} already exists; it is never overwritten"
    if not output_path.parent.is_dir():
        return f"--output directory {str(output_path.parent)!r} does not exist; it is not created"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert one Frame Lab diagnostic (lumen.frame_series_probe.v1) into a NEW lumen.flicker.v1 "
                    "file. Offline only: no camera, SDK, MCP, network or model.")
    parser.add_argument("--input", required=True, help="existing lumen.frame_series_probe.v1 JSON file")
    parser.add_argument("--output", required=True, help="NEW .json file to create (never overwritten)")
    parser.add_argument("--local-time", required=True,
                        help="operator-supplied ISO 8601 local time of the acquisition, with UTC offset")
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-hz", type=float, help="PWM generator setting entered by the operator")
    reference.add_argument("--no-reference", action="store_true",
                           help="no operator reference (the comparator then rates the record invalid)")
    parser.add_argument("--distance-cm", type=float, default=None, help="operator-reported LED distance, cm")
    parser.add_argument("--trial-id", default=None, help="explicit, externally assigned trial id (optional)")
    args = parser.parse_args(argv)

    input_path, output_path = Path(args.input), Path(args.output)
    problem = _check_paths(input_path, output_path)
    if problem is not None:
        print(f"REFUSED: {problem}. Nothing was written.")
        return EXIT_REFUSED
    try:
        record = to_flicker_record(load_document(input_path), local_time=args.local_time,
                                   reference_hz=args.reference_hz, distance_cm=args.distance_cm,
                                   trial_id=args.trial_id)
    except ExportError as exc:
        print(f"INVALID: {exc}. Nothing was written.")
        return EXIT_INVALID_INPUT
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: cannot read {str(input_path)!r}: {exc}. Nothing was written.")
        return EXIT_INVALID_INPUT
    try:
        write_record(output_path, record)
    except FileExistsError:
        print(f"REFUSED: --output {str(output_path)!r} appeared during the export; it was not overwritten.")
        return EXIT_REFUSED
    print(f"Recording id:   {record['recording_id']}")
    print(f"Status:         {record['status']}")
    print(f"f0_hz:          {record['estimate']['f0_hz']}")
    print(f"reference_hz:   {record['source']['reference_hz']} (operator)")
    print(f"trial_id:       {record.get('trial_id', 'none (unpaired)')}")
    print(f"Wrote new file: {output_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
