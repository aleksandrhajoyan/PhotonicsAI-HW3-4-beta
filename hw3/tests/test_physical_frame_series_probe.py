"""Offline tests for the manual physical frame-series diagnostic CLI.

The real IRaypleCamera runs against the fake SDK binding of test_irayple_camera (with the
per-frame timestamps of test_frame_series_capture) and a fake monotonic clock. Terminal
input is simulated. No vendor SDK, camera, MCP, network or language model is touched.
"""

import ast
import builtins
import importlib
import io
import json
import math
import os
import re
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import irayple_camera as ic  # noqa: E402
import physical_frame_series_probe as probe  # noqa: E402
import temporal_analysis as ta  # noqa: E402
import test_frame_series_capture as series_base  # noqa: E402  (SeriesSdk, FakeClock)
import test_irayple_camera as base  # noqa: E402  (fake SDK binding shared with the backend tests)

PHRASE_64 = "CAPTURE 64 FRAMES"
BACKEND_METHODS = ("open", "get_info", "capture_series", "close")
SUCCESS_EVENTS = ["input", "preload", "construct", "open", "get_info", "capture_series", "get_info", "close"]
CLOCK_STEP_NS = 5_000_000  # two clock reads per frame -> 10 ms host intervals -> 100 fps


class FakeTty:
    def __init__(self, interactive):
        self.interactive = interactive

    def isatty(self):
        return self.interactive


class ProbeTestCase(base.OfflineTestCase):
    """Runs probe.main() with a recorded preload, a recorded real IRaypleCamera, a fake SDK and clock."""

    def run_main(self, argv=("--physical-diagnostic",), response=PHRASE_64, tty=True, fail=None,
                 clock_step_ns=CLOCK_STEP_NS, **fake_kwargs):
        self.fake = series_base.SeriesSdk(**fake_kwargs)
        self.events = []
        self.constructed = []
        self.series_calls = []
        self.returned = []
        fail = dict(fail or {})
        real_backend = ic.IRaypleCamera

        def record(name, method):
            def wrapper(*args, **kw):
                self.events.append(name)
                if name == "capture_series":
                    self.series_calls.append((args, kw))
                if name in fail:
                    raise fail.pop(name)
                value = method(*args, **kw)
                if name == "capture_series":
                    self.returned.append(value)
                return value
            return wrapper

        def construct(sdk):
            self.events.append("construct")
            camera = real_backend(sdk)
            self.constructed.append((sdk, camera))
            for name in BACKEND_METHODS:
                setattr(camera, name, record(name, getattr(camera, name)))
            return camera

        def preload():
            self.events.append("preload")
            return self.fake.ns

        self.prompt = None

        def input_fn(prompt):
            self.events.append("input")
            self.prompt = prompt
            if isinstance(response, BaseException):
                raise response
            return response

        clock = series_base.FakeClock(step_ns=clock_step_ns)
        out = io.StringIO()
        with mock.patch.object(ic, "preload_vendor_sdk", side_effect=preload) as self.preload, \
                mock.patch.object(ic, "IRaypleCamera", side_effect=construct), \
                mock.patch.object(ic, "time", SimpleNamespace(perf_counter_ns=clock.perf_counter_ns)), \
                redirect_stdout(out):
            code = probe.main(list(argv), input_fn=input_fn, stdin=FakeTty(tty))
        self.output = out.getvalue()
        return code

    def assert_refused_without_hardware(self, code):
        self.assertEqual(code, probe.EXIT_REFUSED)
        self.preload.assert_not_called()
        self.assertEqual(self.constructed, [])
        self.assertEqual(self.fake.calls, [])
        self.assertNotIn("DIAGNOSTIC_STATUS: PASS", self.output)
        self.assertIn("no camera was accessed", self.output.lower())

    def assert_closed_once(self):
        self.assertEqual(self.events.count("close"), 1)
        self.assertEqual(self.events[-1], "close")
        self.assert_shut_down(self.fake)
        self.assertEqual(self.fake.counts.get("IMV_Close", 0), 1)
        self.assertEqual(self.fake.counts.get("IMV_DestroyHandle", 0), 1)

    def assert_failed(self, code, expected_code=probe.EXIT_FAILED):
        self.assertEqual(code, expected_code, self.output)
        self.assertIn("DIAGNOSTIC_STATUS: FAILED", self.output)
        self.assertNotIn("DIAGNOSTIC_STATUS: PASS", self.output)

    def assert_passed(self, code):
        self.assertEqual(code, probe.EXIT_OK, self.output)
        self.assertIn("ACQUISITION_STATUS:     COMPLETED", self.output)
        self.assertIn("DIAGNOSTIC_STATUS: PASS", self.output)


class TempDirMixin:
    def temp_dir(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)


# 5. Import safety.
class ImportSafetyTests(unittest.TestCase):
    def test_import_touches_no_hardware(self):
        cwd = os.getcwd()
        with mock.patch.object(ic.discovery, "load_vendor_sdk") as loader, \
                mock.patch.object(ic, "preload_vendor_sdk") as preload, \
                mock.patch.object(ic, "IRaypleCamera") as backend, \
                mock.patch("ctypes.WinDLL", create=True) as win_dll, \
                mock.patch("ctypes.CDLL") as cdll, \
                mock.patch("os.chdir") as chdir, \
                mock.patch.object(builtins, "input") as prompt, \
                redirect_stdout(io.StringIO()) as out:
            importlib.reload(probe)
        for patched in (loader, preload, backend, win_dll, cdll, chdir, prompt):
            patched.assert_not_called()
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(os.getcwd(), cwd)
        self.assertNotIn("IMVApi", sys.modules)


# 1-8. Refusal before any SDK activity.
class RefusalTests(ProbeTestCase, TempDirMixin):
    def test_missing_physical_flag_refuses(self):
        for argv in ((), ("--frames", "64"), ("--output", "x.json")):
            with self.subTest(argv=argv):
                code = self.run_main(argv=argv)
                self.assert_refused_without_hardware(code)
                self.assertEqual(self.events, [])
                self.assertIn("NO_ACTION", self.output)

    def test_missing_confirmation_refuses(self):
        for response in (EOFError(), KeyboardInterrupt(), ""):
            with self.subTest(response=response):
                code = self.run_main(response=response)
                self.assert_refused_without_hardware(code)
                self.assertEqual(self.events, ["input"])

    def test_wrong_confirmation_refuses(self):
        for response in ("yes", "capture 64 frames", "CAPTURE 64 FRAMES ", " CAPTURE 64 FRAMES",
                         "CAPTURE 65 FRAMES", "CAPTURE 128 FRAMES", "CAPTURE FRAMES", "TEST IRAYPLE BACKEND"):
            with self.subTest(response=response):
                code = self.run_main(response=response)
                self.assert_refused_without_hardware(code)
                self.assertIn("did not match", self.output)

    def test_phrase_contains_requested_frame_count(self):
        code = self.run_main(argv=("--physical-diagnostic", "--frames", "100"), response=PHRASE_64)
        self.assert_refused_without_hardware(code)
        self.assertIn("CAPTURE 100 FRAMES", self.prompt)
        self.assertEqual(probe.confirmation_phrase(32), "CAPTURE 32 FRAMES")

    def test_noninteractive_refuses(self):
        code = self.run_main(tty=False)
        self.assert_refused_without_hardware(code)
        self.assertEqual(self.events, [])
        with mock.patch.object(ic, "preload_vendor_sdk") as preload, redirect_stdout(io.StringIO()):
            code = probe.main(["--physical-diagnostic"], input_fn=lambda prompt: PHRASE_64,
                              stdin=SimpleNamespace())
        self.assertEqual(code, probe.EXIT_REFUSED)
        preload.assert_not_called()

    def test_environment_never_approves(self):
        with mock.patch.dict(os.environ, {"LUMEN_APPROVE": "1", "CAPTURE_64_FRAMES": "1", "CI": "true"}):
            code = self.run_main(response="")
        self.assert_refused_without_hardware(code)

    def test_frame_count_limits(self):
        for frames in ("385", "386", "512", "1000", "31", "0", "-1", "-5"):
            with self.subTest(frames=frames):
                code = self.run_main(argv=("--physical-diagnostic", "--frames", frames),
                                     response=f"CAPTURE {frames} FRAMES")
                self.assert_refused_without_hardware(code)
                self.assertEqual(self.events, [])  # refused before the prompt
                self.assertIn("--frames must be in [32, 384]", self.output)

    def test_cli_cap_constants(self):
        self.assertEqual((probe.MIN_FRAMES, probe.DEFAULT_FRAMES, probe.MAX_FRAMES), (32, 64, 384))
        self.assertLessEqual(probe.MAX_FRAMES, ic.MAX_SERIES_FRAMES)  # backend limit stays authoritative
        self.assertEqual(probe.MAX_DURATION_S, 5.0)
        self.assertEqual(ic.MAX_SERIES_DURATION_S, 5.0)

    def test_boolean_and_noninteger_frame_counts_rejected_before_sdk(self):
        for frames in ("True", "False", "128.0", "64.5", "1e2", "abc", ""):
            with self.subTest(frames=frames), mock.patch("sys.stderr", io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.run_main(argv=("--physical-diagnostic", "--frames", frames))
            self.preload.assert_not_called()
            self.assertEqual(self.constructed, [])
            self.assertEqual(self.events, [])
            self.assertEqual(self.fake.calls, [])

    def test_384_phrase_and_help(self):
        self.assertEqual(probe.confirmation_phrase(384), "CAPTURE 384 FRAMES")
        code = self.run_main(argv=("--physical-diagnostic", "--frames", "384"), response="no")
        self.assert_refused_without_hardware(code)
        self.assertIn("Type CAPTURE 384 FRAMES to authorize", self.prompt)
        self.assertIn("384 frames (hard CLI maximum 384)", self.output)
        help_out = io.StringIO()
        with redirect_stdout(help_out), self.assertRaises(SystemExit):
            probe.main(["--help"], input_fn=lambda prompt: "", stdin=FakeTty(True))
        self.assertIn("32..384", help_out.getvalue())

    def test_wrong_confirmation_for_384_refuses(self):
        for response in ("CAPTURE 128 FRAMES", "CAPTURE 64 FRAMES", "CAPTURE 385 FRAMES", "capture 384 frames",
                         "CAPTURE 384 FRAMES ", "CAPTURE 384 FRAME", "384"):
            with self.subTest(response=response):
                code = self.run_main(argv=("--physical-diagnostic", "--frames", "384"), response=response)
                self.assert_refused_without_hardware(code)
                self.assertIn("did not match", self.output)

    def test_missing_flag_and_noninteractive_refuse_at_384(self):
        code = self.run_main(argv=("--frames", "384"), response="CAPTURE 384 FRAMES")
        self.assert_refused_without_hardware(code)
        self.assertIn("NO_ACTION", self.output)
        self.assertEqual(self.events, [])
        code = self.run_main(argv=("--physical-diagnostic", "--frames", "384"), response="CAPTURE 384 FRAMES",
                             tty=False)
        self.assert_refused_without_hardware(code)
        self.assertEqual(self.events, [])

    def test_duration_limits(self):
        for duration in ("5.001", "6", "0", "-1", "nan", "inf"):
            with self.subTest(duration=duration):
                code = self.run_main(argv=("--physical-diagnostic", "--max-duration-s", duration))
                self.assert_refused_without_hardware(code)
                self.assertIn("--max-duration-s must be in (0, 5.0]", self.output)

    def test_frame_timeout_limits(self):
        for timeout in ("0", "3001", "-1"):
            with self.subTest(timeout=timeout):
                code = self.run_main(argv=("--physical-diagnostic", "--frame-timeout-ms", timeout))
                self.assert_refused_without_hardware(code)

    def test_non_numeric_limits_rejected_by_parser(self):
        for argv in (("--frames", "64.5"), ("--frame-timeout-ms", "1e3"), ("--max-duration-s", "five")):
            with self.subTest(argv=argv), redirect_stdout(io.StringIO()), \
                    mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                probe.main(["--physical-diagnostic", *argv], input_fn=lambda prompt: PHRASE_64,
                           stdin=FakeTty(True))

    def test_plan_shown_before_prompt(self):
        self.run_main(response="no")
        for text in ("NOT an AI-approved MCP session", "MV-A7200MU130", "64 frames (hard CLI maximum 384)",
                     "5.0 s for the whole series", "NO device setting is written", "Possible effects",
                     "[1000.0, 5000.0] us", "console output only"):
            self.assertIn(text, self.output)
        self.assertLess(self.output.index("Planned action"), len(self.output))
        self.assertIn(PHRASE_64, self.prompt)

    def test_existing_output_file_refused_before_hardware(self):
        path = self.temp_dir() / "existing.json"
        path.write_text("keep me", encoding="utf-8")
        code = self.run_main(argv=("--physical-diagnostic", "--output", str(path)))
        self.assert_refused_without_hardware(code)
        self.assertIn("never overwritten", self.output)
        self.assertEqual(path.read_text(encoding="utf-8"), "keep me")

    def test_invalid_output_paths_refused(self):
        directory = self.temp_dir()
        for target in (directory / "diag.txt", directory / "missing" / "diag.json"):
            with self.subTest(target=target):
                code = self.run_main(argv=("--physical-diagnostic", "--output", str(target)))
                self.assert_refused_without_hardware(code)
                self.assertFalse(target.exists())
                self.assertFalse((directory / "missing").exists())


# 8. Exposure precondition.
class ExposureTests(ProbeTestCase):
    def test_exposure_outside_range_aborts_without_acquisition(self):
        for current in (999.0, 5000.5, 16.0, 20000.0):
            with self.subTest(current=current):
                code = self.run_main(exposure=(current, 16.0, 1000000.0))
                self.assert_failed(code, probe.EXIT_PRECONDITION)
                self.assertIn("EXPOSURE_OUT_OF_RANGE", self.output)
                self.assertNotIn("capture_series", self.events)
                self.assertNotIn("IMV_StartGrabbing", self.fake.names())
                self.assertEqual(self.fake.writes(), [])
                self.assertIn("Exposure writes:        none", self.output)
                self.assert_closed_once()

    def test_boundary_exposures_accepted(self):
        for current in (1000.0, 5000.0):
            with self.subTest(current=current):
                code = self.run_main(exposure=(current, 16.0, 1000000.0))
                self.assert_passed(code)
                self.assertIn(f"Verified exposure:      {current!r} us", self.output)

    def test_unexpected_backend_info_aborts(self):
        real_get_info = ic.IRaypleCamera.get_info
        with mock.patch.object(ic.IRaypleCamera, "get_info", lambda camera: dict(real_get_info(camera),
                                                                                 synthetic=True)):
            code = self.run_main()
        self.assert_failed(code, probe.EXIT_PRECONDITION)
        self.assertIn("UNEXPECTED_BACKEND_INFO", self.output)
        self.assertNotIn("capture_series", self.events)
        self.assert_closed_once()


# 9-11. Successful diagnostic.
class SuccessTests(ProbeTestCase):
    def test_exact_confirmation_runs_single_series(self):
        threads_before = threading.active_count()
        code = self.run_main()
        self.assert_passed(code)
        self.assertEqual(self.events, SUCCESS_EVENTS)
        self.preload.assert_called_once_with()
        self.assertEqual(len(self.constructed), 1)
        self.assertIs(self.constructed[0][0], self.fake.ns)
        self.assertEqual(threading.active_count(), threads_before)

    def test_exactly_one_capture_series(self):
        self.run_main()
        self.assertEqual(self.events.count("capture_series"), 1)
        self.assertEqual(self.fake.counts["IMV_StartGrabbing"], 1)
        self.assertEqual(self.fake.counts["IMV_StopGrabbing"], 1)
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 64)
        self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)
        self.assertEqual(self.fake.counts["IMV_Open"], 1)
        self.assertIn("capture_series calls:   1", self.output)

    def test_driver_arguments(self):
        self.run_main()
        self.assertEqual(self.series_calls, [((64,), {"frame_timeout_ms": 1000, "max_duration_s": 5.0})])
        self.run_main(argv=("--physical-diagnostic", "--frames", "100", "--frame-timeout-ms", "500",
                            "--max-duration-s", "2.5"), response="CAPTURE 100 FRAMES")
        self.assertEqual(self.series_calls, [((100,), {"frame_timeout_ms": 500, "max_duration_s": 2.5})])
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 100)

    def test_no_device_setting_written(self):
        self.run_main()
        self.assertEqual(self.fake.writes(), [])
        self.assertIn("Exposure writes:        none", self.output)
        for name in base.FORBIDDEN_CALLS:
            self.assertNotIn(name, self.fake.names())

    def test_closed_after_success(self):
        self.run_main()
        self.assert_closed_once()
        self.assertIn("Cleanup status:         OK (backend close() confirmed)", self.output)

    def test_diagnostics_report(self):
        code = self.run_main()
        self.assert_passed(code)
        for text in ("Frames:                 64 of 64 requested", "Source provenance:      hardware",
                     "Completion status:      completed", "Verified exposure:      5000.0 us",
                     "Frame size:             4 x 2, 8-bit",
                     "Native block IDs:       first 1000, last 1063, missing 0",
                     "Raw device timestamps:  first 5000000, last 5272223 (unit UNKNOWN)",
                     "Raw timestamp deltas:   min 4321, median 4321, max 4321 ticks of unknown unit",
                     "Mean DN:                min 4.5000, max 157.5000",
                     "Max saturation:         0.250000"):
            self.assertIn(text, self.output)
        series = self.returned[0]
        self.assertIn(f"first {series.samples[0].host_t_ns}, last {series.samples[-1].host_t_ns}", self.output)
        average = sum(s.mean_dn for s in series.samples) / 64
        self.assertIn(f"average {average:.4f}", self.output)


# DV-06: the manual CLI accepts 32..384 frames; one series, same limits, same cleanup.
class ExtendedFrameCountTests(ProbeTestCase, TempDirMixin):
    def run_frames(self, n, **kwargs):
        argv = ("--physical-diagnostic", "--frames", str(n), *kwargs.pop("extra", ()))
        return self.run_main(argv=argv, response=f"CAPTURE {n} FRAMES", **kwargs)

    def test_accepted_counts_forward_exactly_one_series(self):
        for n in (32, 128, 256, 384):
            with self.subTest(frames=n):
                code = self.run_frames(n)
                self.assert_passed(code)
                self.assertEqual(self.events, SUCCESS_EVENTS)
                self.assertEqual(self.series_calls, [((n,), {"frame_timeout_ms": 1000, "max_duration_s": 5.0})])
                self.assertEqual(self.fake.counts["IMV_GetFrame"], n)
                self.assertEqual(self.fake.counts["IMV_StartGrabbing"], 1)
                self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)
                self.assertEqual(self.fake.counts["IMV_Open"], 1)
                self.assertIn(f"Frames:                 {n} of {n} requested", self.output)
                self.assertIn("capture_series calls:   1", self.output)
                self.assert_closed_once()
                self.assertEqual(self.fake.writes(), [])
                for name in base.FORBIDDEN_CALLS:
                    self.assertNotIn(name, self.fake.names())
                self.assertIn("Exposure writes:        none", self.output)

    def test_default_is_still_64_frames(self):
        self.assert_passed(self.run_main())
        self.assertEqual(self.series_calls, [((64,), {"frame_timeout_ms": 1000, "max_duration_s": 5.0})])

    def test_384_frames_still_bounded_by_5_s_ceiling(self):
        code = self.run_frames(384, clock_step_ns=7_000_000)  # 14 ms per frame -> 5.4 s > 5 s
        self.assert_failed(code)
        self.assertIn("SERIES_DEADLINE_EXCEEDED", self.output)
        self.assertEqual(self.series_calls[0][1]["max_duration_s"], 5.0)
        self.assertEqual(self.events.count("capture_series"), 1)
        self.assert_closed_once()
        code = self.run_frames(384, extra=("--max-duration-s", "5.5"))
        self.assert_refused_without_hardware(code)

    def test_384_frames_closed_after_sdk_failures(self):
        for kwargs, status in (({"codes": {"IMV_GetFrame": [0] * 300 + [-119]}}, "GET_FRAME_FAILED"),
                               ({"raises": {"IMV_GetFrame": [None] * 200 + [KeyboardInterrupt()]}}, "INTERRUPTED")):
            with self.subTest(status=status):
                code = self.run_frames(384, **kwargs)
                self.assertNotEqual(code, probe.EXIT_OK)
                self.assertIn(status, self.output)
                self.assertIn("DIAGNOSTIC_STATUS: FAILED", self.output)
                self.assertEqual(self.events.count("capture_series"), 1)
                self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)
                self.assert_closed_once()
        code = self.run_frames(384, codes={"IMV_Close": -101})
        self.assert_failed(code, probe.EXIT_CLEANUP_FAILED)
        self.assertEqual(self.events.count("close"), 1)

    def test_384_frame_export_provenance_unchanged(self):
        path = self.temp_dir() / "diag384.json"
        code = self.run_frames(384, extra=("--output", str(path)))
        self.assert_passed(code)
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], "lumen.frame_series_probe.v1")
        self.assertEqual(document["kind"], "manual_physical_diagnostic")
        self.assertIs(document["ai_approved_session"], False)
        self.assertEqual(document["cleanup"], "ok")
        self.assertEqual(document["limits"], {"n_frames": 384, "frame_timeout_ms": 1000, "max_duration_s": 5.0})
        self.assertEqual(document["diagnostics"]["source"], "hardware")
        self.assertEqual(document["diagnostics"]["requested_frames"], 384)
        self.assertEqual(document["diagnostics"]["frames"], 384)
        self.assertEqual(document["diagnostics"]["device_timestamp_unit"], "unknown")
        self.assertEqual(len(document["samples"]), 384)
        self.assertNotIn("serial", path.read_text(encoding="utf-8"))

    def test_384_frame_export_never_overwrites(self):
        path = self.temp_dir() / "diag384.json"
        path.write_text("keep me", encoding="utf-8")
        code = self.run_frames(384, extra=("--output", str(path)))
        self.assert_refused_without_hardware(code)
        self.assertEqual(path.read_text(encoding="utf-8"), "keep me")


# 14-17. Integrity of the reported values.
class IntegrityTests(ProbeTestCase):
    def test_host_fps_interval_and_jitter(self):
        code = self.run_main()
        self.assert_passed(code)
        self.assertIn("Host-observed FPS:      100.0000", self.output)
        self.assertIn("Host median interval:   10.0000 ms", self.output)
        self.assertIn("Host timing jitter:     std 0.0000 ms, std/median 0.0000%", self.output)
        series = self.returned[0]
        expected = (64 - 1) / ((series.samples[-1].host_t_ns - series.samples[0].host_t_ns) * 1e-9)
        self.assertTrue(math.isclose(expected, 100.0))

    def test_block_id_gaps_preserved_and_analyzer_status_kept(self):
        ids = list(range(500, 530)) + list(range(533, 567))  # 3 missing ids: 530, 531, 532
        code = self.run_main(block_ids=ids)
        self.assert_passed(code)  # acquisition succeeded although frequency recovery is refused
        self.assertIn("Native block IDs:       first 500, last 566, missing 3", self.output)
        self.assertIn("TEMPORAL_ANALYSIS_STATUS: irregular_sampling", self.output)
        self.assertIn("warning DROPPED_FRAMES: 3 frame(s) missing", self.output)
        self.assertIn("warning SPECTRUM_NOT_COMPUTED", self.output)
        self.assertNotIn("f0 estimate", self.output)

    def test_raw_timestamp_units_remain_unknown(self):
        calls = []
        real = ta.analyze_temporal_series

        def spy(**kwargs):
            calls.append(kwargs)
            return real(**kwargs)

        with mock.patch.object(probe.ta, "analyze_temporal_series", side_effect=spy):
            code = self.run_main()
        self.assert_passed(code)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("device_tick_hz", calls[0])
        self.assertNotIn("reference_hz", calls[0])
        self.assertIn("device_ts", calls[0])
        self.assertIn("unit UNKNOWN", self.output)
        self.assertIn("no device clock rate, device FPS or physical device time is reported", self.output)
        # Only the analyzer's own DEVICE_TICK_UNVERIFIED warning may mention a tick rate.
        own_lines = "\n".join(line for line in self.output.lower().splitlines() if "warning " not in line)
        self.assertNotRegex(own_lines, r"device fps:|tick[_ ]hz|ticks per second|tick rate:|fps_device")

    def test_analyzer_warnings_retained(self):
        code = self.run_main()
        self.assert_passed(code)
        self.assertIn("warning HOST_TIMESTAMPS", self.output)
        self.assertIn("warning DEVICE_TICK_UNVERIFIED", self.output)
        self.assertIn("frequency recovery is separate from acquisition", self.output)

    def test_large_raw_timestamps_reported_exactly(self):
        stamps = [2**64 - 1000 + 10 * k for k in range(64)]
        code = self.run_main(timestamps=stamps)
        self.assert_passed(code)
        self.assertIn(f"first {stamps[0]}, last {stamps[-1]} (unit UNKNOWN)", self.output)


# 12, 13. Failures and cleanup.
class FailureTests(ProbeTestCase, TempDirMixin):
    def test_closed_after_failed_acquisition(self):
        code = self.run_main(codes={"IMV_GetFrame": [0, 0, 0, 0, -119]})
        self.assert_failed(code)
        self.assertIn("GET_FRAME_FAILED", self.output)
        self.assertIn("ACQUISITION_STATUS:     FAILED", self.output)
        self.assertNotIn("Frames:", self.output)
        self.assertNotIn("TEMPORAL_ANALYSIS_STATUS", self.output)
        self.assertEqual(self.events.count("capture_series"), 1)  # never repeated
        self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)  # never reopened
        self.assert_closed_once()

    def test_closed_after_deadline_failure(self):
        code = self.run_main(clock_step_ns=100_000_000)  # 200 ms per frame cannot fit 64 frames in 5 s
        self.assert_failed(code)
        self.assertIn("SERIES_DEADLINE_EXCEEDED", self.output)
        self.assert_closed_once()

    def test_closed_after_open_refusal(self):
        code = self.run_main(enums={"ExposureAuto": b"Continuous"})
        self.assert_failed(code)
        self.assertIn("EXPOSURE_AUTO_NOT_OFF", self.output)
        self.assertEqual(self.events[-1], "close")
        self.assert_shut_down(self.fake)

    def test_ambiguous_camera_refused(self):
        code = self.run_main(devices=[base._device(), base._device(serial=b"DA0000001111")])
        self.assert_failed(code)
        self.assertIn("MULTIPLE_DEVICES", self.output)
        self.assertNotIn("capture_series", self.events)
        self.assertEqual(self.fake.handles_alive, 0)

    def test_contract_violation_reported(self):
        code = self.run_main(block_ids=[1] * 64)  # what single-frame captures reported
        self.assert_failed(code)
        self.assertIn("SERIES_CONTRACT_VIOLATION", self.output)
        self.assert_closed_once()

    def test_keyboard_interrupt_during_series_closes(self):
        code = self.run_main(raises={"IMV_GetFrame": [None, None, KeyboardInterrupt()]})
        self.assert_failed(code, probe.EXIT_INTERRUPTED)
        self.assertIn("INTERRUPTED", self.output)
        self.assert_closed_once()

    def test_cleanup_failure_never_hidden(self):
        code = self.run_main(codes={"IMV_Close": -101})
        self.assert_failed(code, probe.EXIT_CLEANUP_FAILED)
        self.assertIn("Cleanup status:         FAILED: CLEANUP_FAILED", self.output)
        self.assertIn("IMV_Close FAILED with code -101", self.output)
        self.assertIn("ACQUISITION_STATUS:     FAILED", self.output)
        self.assertEqual(self.events.count("close"), 1)
        self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)

    def test_unexpected_close_exception_reported(self):
        code = self.run_main(fail={"close": OSError("native crash")})
        self.assert_failed(code, probe.EXIT_CLEANUP_FAILED)
        self.assertIn("Cleanup status:         FAILED: OSError('native crash')", self.output)

    def test_keyboard_interrupt_during_close_not_success(self):
        code = self.run_main(fail={"close": KeyboardInterrupt()})
        self.assert_failed(code, probe.EXIT_INTERRUPTED)
        self.assertIn("camera state unknown", self.output)

    def test_close_failure_prevents_export(self):
        path = self.temp_dir() / "diag.json"
        code = self.run_main(argv=("--physical-diagnostic", "--output", str(path)), codes={"IMV_Close": -101})
        self.assert_failed(code, probe.EXIT_CLEANUP_FAILED)
        self.assertFalse(path.exists())

    def test_analysis_failure_after_close(self):
        with mock.patch.object(probe.ta, "analyze_temporal_series", side_effect=ValueError("analysis broke")):
            code = self.run_main()
        self.assert_failed(code)
        self.assertIn("ANALYSIS_FAILED", self.output)
        self.assertIn("Cleanup status:         OK", self.output)
        self.assert_closed_once()

    def test_sdk_load_failure_reported(self):
        def failing_preload():
            raise ic.discovery.DiscoveryError("HuaRay SDK file not found")

        with mock.patch.object(ic, "preload_vendor_sdk", side_effect=failing_preload) as preload, \
                mock.patch.object(ic, "IRaypleCamera") as backend, redirect_stdout(io.StringIO()) as out:
            code = probe.main(["--physical-diagnostic"], input_fn=lambda prompt: PHRASE_64, stdin=FakeTty(True))
        self.assertEqual(code, probe.EXIT_FAILED)
        preload.assert_called_once_with()
        backend.assert_not_called()
        self.assertIn("SDK_LOAD_FAILED", out.getvalue())


# 18, 19. Privacy and export.
class ExportTests(ProbeTestCase, TempDirMixin):
    def test_console_contains_no_serial_or_pixels(self):
        self.run_main()
        for secret in (base.FULL_SERIAL, base.FULL_CAMERA_KEY, "****7890", "7890"):
            self.assertNotIn(secret, self.output)
        for raw in ("[[", "array(", "dtype", "uint8"):  # no pixel arrays printed
            self.assertNotIn(raw, self.output)

    def test_console_only_run_writes_no_file(self):
        directory = self.temp_dir()
        with mock.patch.object(builtins, "open", side_effect=AssertionError("file opened")):
            code = self.run_main()
        self.assert_passed(code)
        self.assertEqual(list(directory.iterdir()), [])
        self.assertNotIn("Export:                 wrote", self.output)

    def test_export_contents(self):
        path = self.temp_dir() / "diag.json"
        code = self.run_main(argv=("--physical-diagnostic", "--output", str(path)))
        self.assert_passed(code)
        self.assertIn(f"Export:                 wrote new file {path}", self.output)
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], probe.EXPORT_SCHEMA)
        self.assertFalse(document["ai_approved_session"])
        self.assertEqual(document["cleanup"], "ok")
        self.assertEqual(document["model"], "MV-A7200MU130")
        self.assertEqual(len(document["samples"]), 64)
        self.assertEqual(set(document["samples"][0]), {"frame_index", "native_block_id", "device_timestamp_raw",
                                                       "host_t_ns", "mean_dn", "saturation_fraction"})
        diagnostics = document["diagnostics"]
        self.assertEqual(diagnostics["device_timestamp_unit"], "unknown")
        self.assertEqual(diagnostics["missing_block_ids"], 0)
        self.assertAlmostEqual(diagnostics["host_fps"], 100.0)
        self.assertEqual(document["temporal_analysis"]["status"], self.output.split(
            "TEMPORAL_ANALYSIS_STATUS: ")[1].split(" ")[0])
        self.assertFalse(document["temporal_analysis"]["timing"]["device_timestamps"]["used"])
        text = path.read_text(encoding="utf-8")
        for secret in (base.FULL_SERIAL, "7890", "serial", "pixels", "camera_key", "user"):
            self.assertNotIn(secret, text)

    def test_export_never_overwrites_file_created_during_run(self):
        path = self.temp_dir() / "diag.json"

        def create_competitor():
            if not path.exists():
                path.write_text("competitor", encoding="utf-8")

        code = self.run_main(argv=("--physical-diagnostic", "--output", str(path)), on_get_frame=create_competitor)
        self.assertEqual(code, probe.EXIT_EXPORT_FAILED, self.output)
        self.assertEqual(path.read_text(encoding="utf-8"), "competitor")
        self.assertIn("Export:                 FAILED", self.output)
        self.assertIn("camera cleanup was not affected", self.output)
        self.assertIn("Cleanup status:         OK", self.output)
        self.assertIn("ACQUISITION_STATUS:     COMPLETED", self.output)
        self.assertIn("DIAGNOSTIC_STATUS: FAILED", self.output)
        self.assert_closed_once()

    def test_export_happens_after_close(self):
        path = self.temp_dir() / "diag.json"
        order = []
        real_write = probe.write_export

        def write(target, document):
            order.append(("export", list(self.events)))
            return real_write(target, document)

        with mock.patch.object(probe, "write_export", side_effect=write):
            self.run_main(argv=("--physical-diagnostic", "--output", str(path)))
        self.assertEqual(order[0][1][-1], "close")


# 20. Isolation.
class IsolationTests(ProbeTestCase):
    SOURCE = (HW3_DIR / "physical_frame_series_probe.py").read_text(encoding="utf-8")

    def test_only_backend_public_api_and_one_series(self):
        self.assertEqual(re.findall(r"IMV_\w+", self.SOURCE), [])
        self.assertEqual(re.findall(r"\._(?:sdk|camera|lock|frame)\b", self.SOURCE), [])
        self.assertEqual(re.findall(r"(?<!pre)load_vendor_sdk", self.SOURCE), [])
        self.assertEqual(len(re.findall(r"camera\.capture_series\(", self.SOURCE)), 1)
        self.assertEqual(len(re.findall(r"ic\.preload_vendor_sdk\(\)", self.SOURCE)), 1)
        self.assertEqual(len(re.findall(r"ic\.IRaypleCamera\(", self.SOURCE)), 1)
        self.assertEqual(re.findall(r"set_exposure_us|capture_frame", self.SOURCE), [])
        for forbidden in ("ctypes", "chdir", "add_dll_directory", "WinDLL", "import_module", "retry",
                          "Thread(", "while ", "asyncio", "subprocess", "getenv", "environ", "np.save",
                          "tofile", "imwrite", '"w"', "'w'"):
            self.assertNotIn(forbidden, self.SOURCE)
        tree = ast.parse(self.SOURCE)
        self.assertEqual([node for node in ast.walk(tree) if isinstance(node, (ast.While, ast.For))
                          and "capture" in ast.unparse(node)], [])

    def test_imports_limited_to_stdlib_numpy_and_hw3_modules(self):
        imported = set()
        for node in ast.walk(ast.parse(self.SOURCE)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"argparse", "json", "math", "sys", "pathlib", "numpy", "irayple_camera",
                                    "irayple_discovery", "irayple_first_frame", "temporal_analysis",
                                    "frame_series_contract"})
        self.assertNotRegex(self.SOURCE.lower(), r"openai|anthropic|langgraph|api_key|socket|urllib|requests|http")

    def test_no_mcp_network_model_or_real_sdk_at_runtime(self):
        before = set(sys.modules)
        cwd = os.getcwd()
        with mock.patch("socket.socket", side_effect=AssertionError("network")), \
                mock.patch("os.getenv", side_effect=AssertionError("environment read")):
            code = self.run_main()
        self.assert_passed(code)
        for name in set(sys.modules) - before:
            self.assertNotRegex(name, r"openai|mcp|llm|anthropic|langgraph")
        ic.discovery.load_vendor_sdk.assert_not_called()
        self.assertEqual(os.getcwd(), cwd)
        self.assertNotIn("IMVApi", sys.modules)
        self.assertIs(self.constructed[0][0], self.fake.ns)


if __name__ == "__main__":
    unittest.main()
