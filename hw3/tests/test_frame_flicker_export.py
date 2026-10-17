"""Offline tests for the Frame Lab lumen.flicker.v1 exporter (DV-05).

Input fixtures are produced by the real probe export path (physical_frame_series_probe.analyze, summarize and
build_export) from deterministic synthetic per-frame samples shaped like the DV-04 recordings (128 frames at
about 164 fps, a 10 Hz two-level brightness signal, 5000 us exposure, different brightness baselines). No real
evidence file, camera, SDK, MCP, network or language model is used; files are written only to a temporary
directory outside the repository.
"""

import ast
import copy
import io
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HW3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = HW3_DIR.parent
sys.path.insert(0, str(HW3_DIR))

import frame_flicker_export as fx  # noqa: E402
import physical_frame_series_probe as probe  # noqa: E402
from frame_series_contract import FrameSample, FrameSeries  # noqa: E402

TEST_MODEL = "TEST-CAMERA-MODEL"  # no real hardware identifier
LOCAL_TIME = "2026-10-16T18:06:00+04:00"
FLICKER_FIELDS = ("schema", "lab", "run_id", "recording_id", "local_time", "hardware", "source", "acquisition",
                  "estimate", "status", "warnings", "analysis")


def make_document(*, fps=164.08, frequency_hz=10.0, low_dn=60.0, high_dn=112.0, n_frames=128, seed=1,
                  exposure_us=5000.0, model=TEST_MODEL):
    """A lumen.frame_series_probe.v1 document built by the probe's own export code, as plain JSON types."""
    rng = random.Random(seed)
    samples = []
    host_start, tick_start = 19_000_000_000_000, 1_400_000_000_000
    for k in range(n_frames):
        t = k / fps
        level = high_dn if (frequency_hz * t) % 1.0 < 0.5 else low_dn
        samples.append(FrameSample(
            frame_index=k,
            native_block_id=k + 1,
            device_timestamp_raw=tick_start + k * 6_094_440,
            host_t_ns=host_start + round(t * 1e9) + rng.randint(-20_000, 20_000),
            mean_dn=level + rng.gauss(0.0, 0.5),
            saturation_fraction=0.0,
        ))
    series = FrameSeries(samples=tuple(samples), requested_frames=n_frames, acquired_frames=n_frames,
                         exposure_us=exposure_us, width=1920, height=1200, bit_depth=8, source="hardware",
                         status="completed")
    analysis = probe.analyze(series)
    diagnostics = probe.summarize(series, analysis)
    result = probe.ProbeResult(n_frames, 1000, 5.0)
    result.model, result.series, result.close_attempted = model, series, True
    document = probe.build_export(result, diagnostics, analysis)
    return json.loads(json.dumps(document, allow_nan=False))


RUN1_LIKE = make_document(seed=1, low_dn=60.0, high_dn=112.0, fps=164.076)
RUN2_LIKE = make_document(seed=2, low_dn=24.0, high_dn=80.0, fps=164.085)


def convert(document, **kwargs):
    kwargs.setdefault("local_time", LOCAL_TIME)
    return fx.to_flicker_record(document, **kwargs)


def make_ambiguous(document):
    """The DV-03c outcome: analysis ambiguous, no fundamental. Only analyzer fields change."""
    doc = copy.deepcopy(document)
    estimate = doc["temporal_analysis"]["estimate"]
    estimate["candidate_f0_hz"], estimate["f0_hz"] = estimate["f0_hz"], None
    estimate["harmonic_number_of_strongest"] = None
    doc["temporal_analysis"]["status"] = "ambiguous"
    doc["temporal_analysis"]["warnings"].append(
        {"code": "HARMONIC_AMBIGUITY", "message": "peaks are not harmonics of the lowest peak"})
    return doc


class FixtureSanityTest(unittest.TestCase):
    def test_fixtures_have_the_dv04_shape(self):
        for doc in (RUN1_LIKE, RUN2_LIKE):
            self.assertEqual(doc["schema"], "lumen.frame_series_probe.v1")
            self.assertEqual(doc["temporal_analysis"]["status"], "ok")
            self.assertEqual(doc["diagnostics"]["frames"], 128)
            self.assertEqual(doc["diagnostics"]["missing_block_ids"], 0)
            estimate = doc["temporal_analysis"]["estimate"]
            self.assertLessEqual(abs(estimate["f0_hz"] - 10.0), estimate["resolution_hz"])


class RealShapeConversionTest(unittest.TestCase):
    def test_both_dv04_shapes_convert_to_the_event_lab_contract(self):
        for doc in (RUN1_LIKE, RUN2_LIKE):
            record = convert(doc, reference_hz=10.0)
            for key in FLICKER_FIELDS:
                self.assertIn(key, record)
            self.assertEqual(record["schema"], "lumen.flicker.v1")
            self.assertEqual(record["lab"], "frame")
            self.assertEqual(record["acquisition"]["sample_kind"], "frames")
            self.assertIs(record["acquisition"]["early_stop"], False)
            self.assertIsInstance(record["acquisition"]["samples"], int)
            self.assertEqual(record["acquisition"]["request_mode"], "frame_count")
            self.assertIsNone(record["acquisition"]["requested_s"])
            for path in ("actual_s", "fps", "exposure_us"):
                value = record["acquisition"][path]
                self.assertTrue(isinstance(value, float) and math.isfinite(value) and value > 0, path)
            for name in ("run_id", "recording_id", "local_time", "status", "analysis"):
                self.assertIsInstance(record[name], str)
            self.assertIsInstance(record["source"]["type"], str)
            self.assertIsInstance(record["source"]["set_by"], str)

    def test_values_are_copied_from_the_measurement(self):
        record = convert(RUN1_LIKE, reference_hz=10.0)
        ta, d = RUN1_LIKE["temporal_analysis"], RUN1_LIKE["diagnostics"]
        self.assertEqual(record["acquisition"]["fps"], d["host_fps"])
        self.assertEqual(record["acquisition"]["fps"], ta["acquisition"]["fps"])
        self.assertEqual(record["acquisition"]["exposure_us"], d["exposure_us"])
        self.assertEqual(record["acquisition"]["samples"], d["frames"])
        self.assertEqual(record["acquisition"]["actual_s"], ta["acquisition"]["duration_s"])
        self.assertEqual(record["estimate"]["f0_hz"], ta["estimate"]["f0_hz"])
        self.assertEqual(record["estimate"]["resolution_hz"], ta["estimate"]["resolution_hz"])
        self.assertEqual(record["status"], "ok")

    def test_requested_duration_is_not_the_safety_ceiling(self):
        record = convert(RUN1_LIKE, reference_hz=10.0)
        self.assertEqual(RUN1_LIKE["limits"]["max_duration_s"], 5.0)
        self.assertNotEqual(record["acquisition"]["requested_s"], 5.0)
        self.assertIsNone(record["acquisition"]["requested_s"])
        self.assertEqual(record["acquisition"]["requested_frames"], 128)
        self.assertIn("frame count", record["acquisition"]["requested_s_basis"])
        self.assertIn("safety deadline", record["acquisition"]["requested_s_basis"])
        self.assertNotIn(5.0, [v for k, v in record["acquisition"].items() if k != "exposure_us"])

    def test_hardware_block_has_no_serial(self):
        record = convert(RUN1_LIKE, reference_hz=10.0)
        self.assertEqual(record["hardware"], {"model": TEST_MODEL, "sensor": None, "real": True})
        self.assertNotIn("serial", json.dumps(record).lower())


class FrameCountRequestTest(unittest.TestCase):
    """DV-05C: a Frame Lab measurement requests N frames, never a duration."""

    def test_frame_count_request_fields(self):
        for doc in (RUN1_LIKE, RUN2_LIKE):
            acq = convert(doc, reference_hz=10.0)["acquisition"]
            self.assertEqual(acq["request_mode"], "frame_count")
            self.assertEqual(acq["requested_frames"], doc["diagnostics"]["requested_frames"])
            self.assertEqual(acq["requested_frames"], doc["limits"]["n_frames"])
            self.assertEqual((acq["requested_frames"], acq["samples"]), (128, 128))
            self.assertIs(type(acq["requested_frames"]), int)
            self.assertIsNone(acq["requested_s"])
            self.assertEqual(acq["actual_s"], doc["temporal_analysis"]["acquisition"]["duration_s"])
            self.assertIs(acq["early_stop"], False)

    def test_requested_s_is_never_copied_or_derived(self):
        acq = convert(RUN1_LIKE)["acquisition"]
        duration = RUN1_LIKE["temporal_analysis"]["acquisition"]["duration_s"]
        derived = acq["requested_frames"] / acq["fps"]
        self.assertNotEqual(acq["requested_s"], duration)
        self.assertNotEqual(acq["requested_s"], derived)
        self.assertNotIn("requested_s", {k for k, v in acq.items() if v in (duration, derived, 5.0)})

    def test_real_timestamps_are_not_reinterpreted(self):
        acq = convert(RUN1_LIKE)["acquisition"]
        hosts = [s["host_t_ns"] for s in RUN1_LIKE["samples"]]
        self.assertTrue(math.isclose(acq["actual_s"], (hosts[-1] - hosts[0]) * 1e-9, rel_tol=1e-9))
        self.assertEqual(acq["timestamp_source"], "host_monotonic")

    def test_requested_and_delivered_counts_must_match(self):
        short = copy.deepcopy(RUN1_LIKE)
        short["diagnostics"]["frames"] = 127
        short["temporal_analysis"]["acquisition"]["n_frames"] = 127
        short["samples"] = short["samples"][:-1]
        mismatched = copy.deepcopy(RUN1_LIKE)
        mismatched["diagnostics"]["requested_frames"] = 256
        limit_only = copy.deepcopy(RUN1_LIKE)
        limit_only["limits"]["n_frames"] = 64
        for doc in (short, mismatched, limit_only):
            with self.assertRaises(fx.ExportError):
                convert(doc, reference_hz=10.0)

    def test_boolean_and_malformed_requested_counts_are_rejected(self):
        for path in ("diagnostics.requested_frames", "limits.n_frames"):
            for bad in (True, False, 128.0, "128", 0, -128, None, [128]):
                doc = copy.deepcopy(RUN1_LIKE)
                section, key = path.split(".")
                doc[section][key] = bad
                with self.subTest(path=path, value=bad), self.assertRaises(fx.ExportError):
                    convert(doc, reference_hz=10.0)
            doc = copy.deepcopy(RUN1_LIKE)
            section, key = path.split(".")
            del doc[section][key]
            with self.subTest(path=path, missing=True), self.assertRaises(fx.ExportError):
                convert(doc)

    def test_unsuccessful_capture_cannot_hide_behind_null_requested_s(self):
        for status in ("partial", "failed", "timeout", None):
            doc = copy.deepcopy(RUN1_LIKE)
            doc["diagnostics"]["completion_status"] = status
            with self.subTest(status=status), self.assertRaises(fx.ExportError):
                convert(doc, reference_hz=10.0)

    def test_schema_and_reference_are_unchanged_by_the_new_semantics(self):
        record = convert(RUN1_LIKE)
        self.assertEqual(record["schema"], "lumen.flicker.v1")
        self.assertIsNone(record["source"]["reference_hz"])
        self.assertEqual(record["source"]["set_by"], "not_supplied")
        self.assertEqual(record["warnings"], RUN1_LIKE["temporal_analysis"]["warnings"])
        self.assertEqual(record["provenance"]["input_sha256"], fx.input_digest(RUN1_LIKE))


class ReferenceTest(unittest.TestCase):
    def test_operator_reference_is_used_as_given(self):
        record = convert(RUN2_LIKE, reference_hz=10.0)
        self.assertEqual(record["source"]["reference_hz"], 10.0)
        self.assertEqual(record["source"]["set_by"], "manual")
        self.assertEqual(record["provenance"]["reference_source"], "operator")
        self.assertIn("not a calibrated", record["source"]["reference_provenance"])

    def test_reference_is_never_inferred_from_f0(self):
        record = convert(RUN1_LIKE)
        self.assertIsNotNone(record["estimate"]["f0_hz"])
        self.assertIsNone(record["source"]["reference_hz"])
        self.assertEqual(record["source"]["set_by"], "not_supplied")
        self.assertIsNone(record["provenance"]["reference_source"])

    def test_reference_does_not_change_the_estimate(self):
        with_ref = convert(RUN1_LIKE, reference_hz=12.5)
        without = convert(RUN1_LIKE)
        self.assertEqual(with_ref["estimate"], without["estimate"])
        self.assertEqual(with_ref["status"], without["status"])
        self.assertEqual(with_ref["warnings"], without["warnings"])

    def test_invalid_operator_values_are_rejected(self):
        for bad in (0, -10.0, float("nan"), float("inf"), True, "10"):
            with self.assertRaises(fx.ExportError):
                convert(RUN1_LIKE, reference_hz=bad)
        for bad in (0, -1.0, float("nan")):
            with self.assertRaises(fx.ExportError):
                convert(RUN1_LIKE, distance_cm=bad)

    def test_distance_is_operator_metadata_only(self):
        record = convert(RUN1_LIKE, reference_hz=10.0, distance_cm=15.82)
        self.assertEqual(record["source"]["distance_cm"], 15.82)
        self.assertEqual(record["provenance"]["distance_source"], "operator")
        self.assertIsNone(convert(RUN1_LIKE)["source"]["distance_cm"])
        self.assertIsNone(record["source"]["duty"])


class WarningAndSnrTest(unittest.TestCase):
    def test_warnings_are_copied_unchanged_and_in_order(self):
        record = convert(RUN1_LIKE, reference_hz=10.0)
        self.assertEqual(record["warnings"], RUN1_LIKE["temporal_analysis"]["warnings"])
        codes = [w["code"] for w in record["warnings"]]
        self.assertIn("ALIASING_NOT_EXCLUDED", codes)
        self.assertIn("HOST_TIMESTAMPS", codes)
        for warning in record["warnings"]:
            self.assertEqual(set(warning), {"code", "message"})

    def test_malformed_warnings_are_rejected(self):
        for bad in ("a string", {"code": "X"}, {"code": "", "message": "m"},
                    {"code": "X", "message": "m", "extra": 1}):
            doc = copy.deepcopy(RUN1_LIKE)
            doc["temporal_analysis"]["warnings"].append(bad)
            with self.assertRaises(fx.ExportError):
                convert(doc)

    def test_snr_is_the_original_power_ratio_with_its_definition(self):
        record = convert(RUN1_LIKE, reference_hz=10.0)
        self.assertEqual(record["estimate"]["snr"], RUN1_LIKE["temporal_analysis"]["estimate"]["snr"])
        self.assertEqual(record["estimate"]["snr_definition"], "spectral power ratio, Frame Lab DV-01")
        self.assertIn("strongest", record["estimate"]["snr_basis"])
        self.assertEqual(record["estimate"]["peaks"], RUN1_LIKE["temporal_analysis"]["estimate"]["peaks"])

    def test_harmonics_list_excludes_the_fundamental(self):
        record = convert(RUN1_LIKE)
        f0 = record["estimate"]["f0_hz"]
        self.assertNotIn(f0, record["estimate"]["harmonics_hz"])
        self.assertEqual(len(record["estimate"]["harmonics_hz"]), len(record["estimate"]["peaks"]) - 1)


class ProvenanceTest(unittest.TestCase):
    def test_real_manual_diagnostic_is_not_an_ai_session(self):
        p = convert(RUN1_LIKE, reference_hz=10.0)["provenance"]
        self.assertEqual(p["data_origin"], "real_hardware")
        self.assertEqual(p["acquisition_mode"], "manual_operator_diagnostic")
        self.assertIs(p["ai_approved_session"], False)
        self.assertEqual(p["record_origin"], "offline_export_of_original_acquisition")
        self.assertIs(p["analysis_rerun"], False)
        self.assertEqual(p["device_timestamp_unit"], "unknown")
        self.assertEqual(p["local_time_source"], "operator")

    def test_synthetic_data_cannot_become_real_hardware(self):
        doc = copy.deepcopy(RUN1_LIKE)
        doc["diagnostics"]["source"] = "synthetic"
        record = convert(doc, reference_hz=10.0)
        self.assertIs(record["hardware"]["real"], False)
        self.assertEqual(record["provenance"]["data_origin"], "synthetic")
        doc["diagnostics"]["source"] = "simulated"
        with self.assertRaises(fx.ExportError):
            convert(doc)

    def test_an_ai_session_flag_is_rejected(self):
        for value in (True, None, "false", 0):
            doc = copy.deepcopy(RUN1_LIKE)
            doc["ai_approved_session"] = value
            with self.assertRaises(fx.ExportError):
                convert(doc)

    def test_identifiers_are_stable_and_content_derived(self):
        a, b = convert(RUN1_LIKE), convert(copy.deepcopy(RUN1_LIKE))
        self.assertEqual(a["recording_id"], b["recording_id"])
        self.assertEqual(a["run_id"], a["recording_id"])
        self.assertRegex(a["recording_id"], r"^frame-[0-9a-f]{16}$")
        self.assertNotEqual(a["recording_id"], convert(RUN2_LIKE)["recording_id"])
        reordered = json.loads(json.dumps(RUN1_LIKE, sort_keys=True))
        self.assertEqual(convert(reordered)["recording_id"], a["recording_id"])

    def test_local_time_is_operator_input_with_offset(self):
        self.assertEqual(convert(RUN1_LIKE)["local_time"], LOCAL_TIME)
        for bad in (None, "", "yesterday", "2026-10-16T18:06:00", 1760623560):
            with self.assertRaises(fx.ExportError):
                convert(RUN1_LIKE, local_time=bad)


class AmbiguousAndFailedTest(unittest.TestCase):
    def test_ambiguous_analysis_keeps_f0_null(self):
        record = convert(make_ambiguous(RUN1_LIKE), reference_hz=10.0)
        self.assertEqual(record["status"], "ambiguous")
        self.assertIsNone(record["estimate"]["f0_hz"])
        self.assertIsNone(record["estimate"]["harmonics_hz"])
        self.assertIn("HARMONIC_AMBIGUITY", [w["code"] for w in record["warnings"]])

    def test_f0_on_a_non_ok_status_is_rejected(self):
        doc = make_ambiguous(RUN1_LIKE)
        doc["temporal_analysis"]["estimate"]["f0_hz"] = 10.0
        with self.assertRaises(fx.ExportError):
            convert(doc)

    def test_ok_without_f0_is_rejected_not_zeroed(self):
        doc = copy.deepcopy(RUN1_LIKE)
        doc["temporal_analysis"]["estimate"]["f0_hz"] = None
        with self.assertRaises(fx.ExportError):
            convert(doc)
        doc["temporal_analysis"]["estimate"]["f0_hz"] = 0
        with self.assertRaises(fx.ExportError):
            convert(doc)

    def test_partial_or_failed_acquisition_is_rejected(self):
        cases = []
        doc = copy.deepcopy(RUN1_LIKE)
        doc["diagnostics"]["completion_status"] = "partial"
        cases.append(doc)
        doc = copy.deepcopy(RUN1_LIKE)
        doc["cleanup"] = "failed"
        cases.append(doc)
        doc = copy.deepcopy(RUN1_LIKE)
        doc["diagnostics"]["frames"] = 100
        cases.append(doc)
        doc = copy.deepcopy(RUN1_LIKE)
        doc["samples"] = doc["samples"][:-1]
        cases.append(doc)
        for doc in cases:
            with self.assertRaises(fx.ExportError):
                convert(doc, reference_hz=10.0)


class MalformedInputTest(unittest.TestCase):
    def mutate(self, path, value, delete=False):
        doc = copy.deepcopy(RUN1_LIKE)
        *parents, last = path.split(".")
        target = doc
        for key in parents:
            target = target[key]
        if delete:
            del target[last]
        else:
            target[last] = value
        return doc

    def test_missing_fields_are_rejected(self):
        for path in ("schema", "kind", "model", "limits", "diagnostics", "temporal_analysis", "samples",
                     "diagnostics.exposure_us", "diagnostics.host_fps", "temporal_analysis.estimate",
                     "temporal_analysis.acquisition.duration_s", "temporal_analysis.estimate.resolution_hz"):
            with self.subTest(path=path), self.assertRaises(fx.ExportError):
                convert(self.mutate(path, None, delete=True))

    def test_wrong_types_and_values_are_rejected(self):
        cases = [
            ("schema", "lumen.flicker.v1"), ("kind", "ai_session"), ("model", ""),
            ("diagnostics.exposure_us", -5000.0), ("diagnostics.exposure_us", "5000"),
            ("diagnostics.frames", 128.0), ("diagnostics.frames", True), ("limits.n_frames", 0),
            ("temporal_analysis.status", "great"), ("temporal_analysis.acquisition.sample_kind", "events"),
            ("temporal_analysis.acquisition.timestamp_source", "device_verified"),
            ("temporal_analysis.acquisition.duration_s", -0.77), ("temporal_analysis.acquisition.duration_s", 0.5),
            ("temporal_analysis.acquisition.fps", 200.0), ("temporal_analysis.estimate.peaks", "none"),
            ("temporal_analysis.estimate.snr", -1.0), ("diagnostics.missing_block_ids", -1),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value), self.assertRaises(fx.ExportError):
                convert(self.mutate(path, value))

    def test_non_object_input_is_rejected(self):
        for bad in (None, [], "text", 1):
            with self.assertRaises(fx.ExportError):
                convert(bad)

    def test_inconsistent_samples_are_rejected(self):
        doc = copy.deepcopy(RUN1_LIKE)
        doc["samples"][5]["host_t_ns"] = doc["samples"][4]["host_t_ns"]
        with self.assertRaises(fx.ExportError):
            convert(doc)
        doc = copy.deepcopy(RUN1_LIKE)
        doc["samples"][3]["frame_index"] = 7
        with self.assertRaises(fx.ExportError):
            convert(doc)
        doc = copy.deepcopy(RUN1_LIKE)
        doc["samples"][-1]["host_t_ns"] += 50_000_000
        with self.assertRaises(fx.ExportError):
            convert(doc)


class NonFiniteTest(unittest.TestCase):
    def test_nonfinite_numbers_are_rejected(self):
        for path in ("diagnostics.exposure_us", "diagnostics.host_fps", "temporal_analysis.acquisition.fps",
                     "temporal_analysis.estimate.f0_hz", "temporal_analysis.estimate.resolution_hz",
                     "temporal_analysis.estimate.snr"):
            for value in (float("nan"), float("inf"), float("-inf")):
                doc = copy.deepcopy(RUN1_LIKE)
                target = doc
                *parents, last = path.split(".")
                for key in parents:
                    target = target[key]
                target[last] = value
                with self.subTest(path=path, value=value), self.assertRaises(fx.ExportError):
                    convert(doc)

    def test_nan_literal_in_an_input_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.json"
            text = json.dumps(RUN1_LIKE).replace('"snr": ' + json.dumps(
                RUN1_LIKE["temporal_analysis"]["estimate"]["snr"]), '"snr": NaN', 1)
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(fx.ExportError):
                fx.load_document(path)


class TrialIdTest(unittest.TestCase):
    def test_trial_id_only_when_supplied(self):
        record = convert(RUN1_LIKE, reference_hz=10.0, trial_id="T-2026-10-18-A")
        self.assertEqual(record["trial_id"], "T-2026-10-18-A")
        self.assertEqual(record["provenance"]["trial_id_source"], "operator")
        for bad in ("", "   ", 7):
            with self.assertRaises(fx.ExportError):
                convert(RUN1_LIKE, trial_id=bad)

    def test_no_automatic_pairing(self):
        a, b = convert(RUN1_LIKE, reference_hz=10.0), convert(RUN2_LIKE, reference_hz=10.0)
        for record in (a, b):
            self.assertNotIn("trial_id", record)
            self.assertIsNone(record["provenance"]["trial_id_source"])
        self.assertNotEqual(a["run_id"], b["run_id"])


class JsonAndFilesTest(unittest.TestCase):
    def test_json_roundtrip(self):
        record = convert(RUN1_LIKE, reference_hz=10.0, distance_cm=15.82, trial_id="T1")
        text = json.dumps(record, allow_nan=False)
        self.assertEqual(json.loads(text), record)

    def test_input_is_not_modified_and_output_shares_no_objects(self):
        original = copy.deepcopy(RUN1_LIKE)
        record = convert(RUN1_LIKE, reference_hz=10.0)
        self.assertEqual(RUN1_LIKE, original)
        record["warnings"].append({"code": "X", "message": "y"})
        record["estimate"]["peaks"][0]["snr"] = -1
        self.assertEqual(RUN1_LIKE, original)

    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = fx.main(list(argv))
        return code, out.getvalue()

    def test_cli_creates_a_new_file_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "probe.json"
            source.write_text(json.dumps(RUN1_LIKE, indent=2), encoding="utf-8")
            before = source.read_bytes()
            target = Path(tmp) / "flicker.json"
            args = ["--input", str(source), "--output", str(target), "--local-time", LOCAL_TIME,
                    "--reference-hz", "10", "--distance-cm", "15.82"]
            code, output = self.run_cli(*args)
            self.assertEqual(code, fx.EXIT_OK, output)
            record = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(record["source"]["reference_hz"], 10.0)
            self.assertNotIn("trial_id", record)
            self.assertEqual(record["acquisition"]["request_mode"], "frame_count")
            self.assertIsNone(record["acquisition"]["requested_s"])
            written = target.read_bytes()
            self.assertFalse(written.startswith(b"\xef\xbb\xbf"), "output must be UTF-8 without BOM")
            written.decode("utf-8")

            code, output = self.run_cli(*args)
            self.assertEqual(code, fx.EXIT_REFUSED)
            self.assertIn("already exists", output)
            self.assertEqual(target.read_bytes(), written)
            self.assertEqual(source.read_bytes(), before)

    def test_cli_refusals_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "probe.json"
            source.write_text(json.dumps(RUN1_LIKE), encoding="utf-8")
            bad = tmp / "bad.json"
            bad.write_text(json.dumps(make_ambiguous(RUN1_LIKE) | {"cleanup": "failed"}), encoding="utf-8")
            cases = [
                (["--input", str(tmp / "missing.json"), "--output", str(tmp / "o1.json")], fx.EXIT_REFUSED),
                (["--input", str(source), "--output", str(tmp / "o2.txt")], fx.EXIT_REFUSED),
                (["--input", str(source), "--output", str(tmp / "no_dir" / "o3.json")], fx.EXIT_REFUSED),
                (["--input", str(source), "--output", str(source)], fx.EXIT_REFUSED),
                (["--input", str(bad), "--output", str(tmp / "o4.json")], fx.EXIT_INVALID_INPUT),
            ]
            for argv, expected in cases:
                code, _ = self.run_cli(*argv, "--local-time", LOCAL_TIME, "--reference-hz", "10")
                self.assertEqual(code, expected, argv)
            self.assertEqual(sorted(p.name for p in tmp.iterdir()), ["bad.json", "probe.json"])

    def test_cli_requires_an_explicit_reference_choice(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            fx.main(["--input", "x.json", "--output", "y.json", "--local-time", LOCAL_TIME])

    def test_cli_without_reference_writes_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "p.json", Path(tmp) / "f.json"
            source.write_text(json.dumps(RUN2_LIKE), encoding="utf-8")
            code, _ = self.run_cli("--input", str(source), "--output", str(target), "--local-time", LOCAL_TIME,
                                   "--no-reference")
            self.assertEqual(code, fx.EXIT_OK)
            self.assertIsNone(json.loads(target.read_text(encoding="utf-8"))["source"]["reference_hz"])

    def test_module_entry_point_runs_from_the_repository_root(self):
        completed = subprocess.run([sys.executable, "-B", "-m", "hw3.frame_flicker_export", "--help"],
                                   cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--reference-hz", completed.stdout)


class IsolationTest(unittest.TestCase):
    SOURCE = (HW3_DIR / "frame_flicker_export.py").read_text(encoding="utf-8")

    def test_only_standard_library_imports(self):
        imported = set()
        for node in ast.walk(ast.parse(self.SOURCE)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported, {"argparse", "copy", "hashlib", "json", "math", "sys", "datetime", "pathlib"})

    def test_no_hardware_sdk_mcp_model_network_or_clock_names_in_code(self):
        names = set()
        for node in ast.walk(ast.parse(self.SOURCE)):
            if isinstance(node, ast.Name):
                names.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                names.add(node.attr.lower())
        for word in ("irayple", "imvapi", "load_vendor_sdk", "preload_vendor_sdk", "fastmcp", "openai",
                     "anthropic", "langgraph", "socket", "urlopen", "perf_counter_ns", "time_ns", "now",
                     "getmtime", "st_mtime", "stat"):
            self.assertNotIn(word, names)


if __name__ == "__main__":
    unittest.main()
