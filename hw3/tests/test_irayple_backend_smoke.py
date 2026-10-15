"""Offline tests for the IRaypleCamera smoke-test CLI. The real backend runs against a fake SDK binding;
no vendor SDK, camera, network or LLM is touched."""

import ast
import builtins
import ctypes
import importlib
import io
import os
import re
import sys
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import irayple_backend_smoke as smoke  # noqa: E402
import irayple_camera as ic  # noqa: E402
import test_irayple_camera as base  # noqa: E402  (fake SDK binding shared with the backend tests)
from camera_contract import CameraFrame  # noqa: E402
from measurements import analyze_frame  # noqa: E402

WIDTH, HEIGHT = 1920, 1200
# Every DN 0..255 appears equally often: min 0, max 255, mean 127.5, saturation 9000/2304000.
PAYLOAD = (np.arange(WIDTH * HEIGHT) % 256).astype(np.uint8).tobytes()
DIM_PAYLOAD = bytes([7]) * (WIDTH * HEIGHT - 1) + bytes([200])
WRITE = base.WRITE
BACKEND_METHODS = ("open", "get_info", "get_exposure_limits_us", "set_exposure_us", "capture_frame", "close")
SUCCESS_EVENTS = [
    "input", "preload", "construct", "open", "get_info", "get_exposure_limits_us", "set_exposure_us",
    "capture_frame", "get_info", "get_info", "close",
]


class FakeTty:
    def __init__(self, interactive):
        self.interactive = interactive

    def isatty(self):
        return self.interactive


class SmokeTestCase(base.OfflineTestCase):
    """Runs smoke.main() with a recorded preload, a recorded real IRaypleCamera and a fake SDK."""

    def run_main(self, argv=("--check-once",), response=smoke.CONFIRMATION_PHRASE, tty=True, fail=None,
                 **fake_kwargs):
        kwargs = {"payloads": (PAYLOAD,), "width": WIDTH, "height": HEIGHT}
        kwargs.update(fake_kwargs)
        self.fake = base.FakeSdk(**kwargs)
        self.events = []
        self.constructed = []
        fail = dict(fail or {})
        real_backend = ic.IRaypleCamera

        def record(name, method):
            def wrapper(*args, **kw):
                self.events.append(name)
                if name in fail:
                    raise fail.pop(name)
                return method(*args, **kw)
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

        def input_fn(prompt):
            self.events.append("input")
            self.prompt = prompt
            if isinstance(response, BaseException):
                raise response
            return response

        out = io.StringIO()
        with mock.patch.object(ic, "preload_vendor_sdk", side_effect=preload) as self.preload, \
                mock.patch.object(ic, "IRaypleCamera", side_effect=construct), redirect_stdout(out):
            code = smoke.main(list(argv), input_fn=input_fn, stdin=FakeTty(tty))
        self.output = out.getvalue()
        return code

    def assert_refused_without_hardware(self, code):
        self.assertEqual(code, smoke.EXIT_REFUSED)
        self.preload.assert_not_called()
        self.assertEqual(self.constructed, [])
        self.assertEqual(self.fake.calls, [])
        self.assertNotIn("PASS", self.output)

    def assert_closed_once(self):
        self.assertEqual(self.events.count("close"), 1)
        self.assertEqual(self.events[-1], "close")
        self.assert_shut_down(self.fake)
        self.assertEqual(self.fake.counts.get("IMV_Close", 0), 1)
        self.assertEqual(self.fake.counts.get("IMV_DestroyHandle", 0), 1)

    def assert_failed(self, code, expected_code=smoke.EXIT_FAILED):
        self.assertEqual(code, expected_code)
        self.assertIn("SMOKE_TEST_STATUS: FAILED", self.output)
        self.assertNotIn("SMOKE_TEST_STATUS: PASS", self.output)

    def assert_no_more_than_one_acquisition(self):
        self.assertLessEqual(self.fake.counts.get("IMV_GetFrame", 0), 1)
        self.assertLessEqual(self.fake.counts.get("IMV_StartGrabbing", 0), 1)
        self.assertEqual(self.fake.counts.get("IMV_EnumDevices", 0), 1)
        self.assertLessEqual(self.events.count("capture_frame"), 1)
        self.assertEqual(self.events.count("open"), 1)


# 1. Import safety.
class ImportSafetyTests(unittest.TestCase):
    def test_import_touches_no_hardware(self):
        cwd = os.getcwd()
        with mock.patch.object(ic.discovery, "load_vendor_sdk") as loader, \
                mock.patch.object(ic, "preload_vendor_sdk") as preload, \
                mock.patch.object(ic, "IRaypleCamera") as backend, \
                mock.patch("ctypes.WinDLL", create=True) as win_dll, \
                mock.patch("ctypes.CDLL") as cdll, \
                mock.patch("os.add_dll_directory", create=True) as add_dir, \
                mock.patch("os.chdir") as chdir, \
                mock.patch.object(builtins, "input") as prompt, \
                redirect_stdout(io.StringIO()) as out:
            importlib.reload(smoke)
        for patched in (loader, preload, backend, win_dll, cdll, add_dir, chdir, prompt):
            patched.assert_not_called()
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(os.getcwd(), cwd)
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)


# 2-4. Refusal before any SDK activity.
class RefusalTests(SmokeTestCase):
    def test_missing_check_once_refuses(self):
        code = self.run_main(argv=())
        self.assert_refused_without_hardware(code)
        self.assertEqual(self.events, [])
        self.assertIn("NO_ACTION", self.output)

    def test_noninteractive_refuses(self):
        code = self.run_main(tty=False)
        self.assert_refused_without_hardware(code)
        self.assertEqual(self.events, [])  # no prompt was even shown

    def test_stdin_without_isatty_refuses(self):
        with mock.patch.object(ic, "preload_vendor_sdk") as preload, redirect_stdout(io.StringIO()):
            code = smoke.main(["--check-once"], input_fn=lambda prompt: smoke.CONFIRMATION_PHRASE,
                              stdin=SimpleNamespace())
        self.assertEqual(code, smoke.EXIT_REFUSED)
        preload.assert_not_called()

    def test_incorrect_confirmation_refuses(self):
        for response in ("", "yes", "test irayple backend", "TEST IRAYPLE BACKEND ", " TEST IRAYPLE BACKEND",
                         "RUN EXPOSURE SWEEP", "CAPTURE ONE FRAME", EOFError(), KeyboardInterrupt()):
            with self.subTest(response=response):
                code = self.run_main(response=response)
                self.assert_refused_without_hardware(code)
                self.assertEqual(self.events, ["input"])
                self.assertIn("REFUSED", self.output)

    def test_no_environment_approval(self):
        with mock.patch.dict(os.environ, {"LUMEN_APPROVE": "1", "TEST_IRAYPLE_BACKEND": "1", "CI": "true"}):
            code = self.run_main(response="")
        self.assert_refused_without_hardware(code)

    def test_plan_shown_before_prompt(self):
        self.run_main(response="no")
        self.assertIn(smoke.CONFIRMATION_PHRASE, self.prompt)
        self.assertIn("capture_frame() exactly once", self.output)
        self.assertIn("forcibly killed process cannot restore", self.output)


# 5-11, 14-17, 21. Successful workflow.
class SuccessTests(SmokeTestCase):
    def test_exact_confirmation_runs_backend_sequence(self):
        threads_before = threading.active_count()
        code = self.run_main()
        self.assertEqual(code, smoke.EXIT_OK, self.output)
        self.assertEqual(self.events, SUCCESS_EVENTS)
        self.preload.assert_called_once_with()
        self.assertEqual(len(self.constructed), 1)
        self.assertIs(self.constructed[0][0], self.fake.ns)  # constructed with the preloaded binding
        self.assertEqual(threading.active_count(), threads_before)
        self.assertIn("SMOKE_TEST_STATUS: PASS", self.output)

    def test_backend_sdk_sequence(self):
        self.run_main()
        names = self.fake.names()
        self.assertEqual(names[:len(base.OPEN_SEQUENCE)], base.OPEN_SEQUENCE)
        self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)
        self.assertEqual(self.fake.counts["IMV_CreateHandle"], 1)
        self.assertEqual(self.fake.counts["IMV_Open"], 1)
        self.assertEqual(self.fake.counts["IMV_StartGrabbing"], 1)
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)
        self.assertEqual(self.fake.counts["IMV_ReleaseFrame"], 1)
        self.assertEqual(names[-4:], [WRITE, base.READ_VALUE, "IMV_Close", "IMV_DestroyHandle"])

    def test_limits_reported_from_backend(self):
        code = self.run_main(exposure=(5000.0, 20.5, 200000.0))
        self.assertEqual(code, smoke.EXIT_OK, self.output)
        self.assertIn("SDK exposure limits: (20.5, 200000.0) us", self.output)
        self.assertIn("Original exposure:   5000.0 us", self.output)

    def test_target_applied_and_read_back(self):
        self.run_main()
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
        self.assertEqual(self.fake.exposure_at_frame, [1000.0])
        self.assertEqual(self.fake.current, 5000.0)
        self.assertIn("Requested exposure:  1000.0 us", self.output)
        self.assertIn("Applied exposure:    1000.0 us", self.output)

    def test_exactly_one_capture(self):
        self.run_main()
        self.assertEqual(self.events.count("capture_frame"), 1)
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)
        self.assertIn("capture_frame calls: 1", self.output)

    def test_measurements_from_returned_pixels(self):
        captured = []
        with mock.patch.object(smoke, "analyze_frame", side_effect=lambda *a: captured.append(a) or
                               analyze_frame(*a)):
            code = self.run_main()
        self.assertEqual(code, smoke.EXIT_OK, self.output)
        self.assertEqual(len(captured), 1)
        pixels, bit_depth = captured[0]
        self.assertEqual(bit_depth, 8)
        self.assertEqual(pixels.tobytes(), PAYLOAD)
        self.assertEqual(pixels.shape, (HEIGHT, WIDTH))
        self.assertIn("min DN 0, max DN 255, mean DN 127.5000, saturation fraction 0.003906", self.output)
        self.assertIn(f"Frame:               {WIDTH} x {HEIGHT}, dtype uint8, source=hardware", self.output)

    def test_measurements_track_pixel_content(self):
        code = self.run_main(payloads=(DIM_PAYLOAD,))
        self.assertEqual(code, smoke.EXIT_OK, self.output)
        mean = (7 * (WIDTH * HEIGHT - 1) + 200) / (WIDTH * HEIGHT)
        self.assertIn(f"min DN 7, max DN 200, mean DN {mean:.4f}, saturation fraction 0.000000", self.output)

    def test_closed_and_restored_after_success(self):
        self.run_main()
        self.assert_closed_once()
        self.assertIn("Exposure restoration: RESTORED_AND_VERIFIED by backend close() (original 5000.0 us)",
                      self.output)
        self.assertIn("Backend close:       OK", self.output)

    def test_report_masks_identifiers(self):
        self.run_main()
        self.assertIn("Model:               MV-A7200MU130", self.output)
        self.assertIn("Serial (masked):     ****7890", self.output)
        self.assertNotIn(base.FULL_SERIAL, self.output)
        self.assertNotIn(base.FULL_CAMERA_KEY, self.output)
        self.assertIn("no image, raw data or report was saved", self.output)


# 12, 13. Preconditions prevent writing.
class PreconditionTests(SmokeTestCase):
    def assert_no_write(self, code):
        self.assert_failed(code, smoke.EXIT_PRECONDITION)
        self.assertEqual(self.fake.writes(), [])
        self.assertNotIn("set_exposure_us", self.events)
        self.assertNotIn("IMV_GetFrame", self.fake.names())
        self.assert_closed_once()
        self.assertIn("NOT_REQUIRED (no exposure write was issued)", self.output)

    def test_wrong_initial_exposure_prevents_write(self):
        for current in (4000.0, 5000.001, 1000.0):
            with self.subTest(current=current):
                code = self.run_main(exposure=(current, 16.0, 1000000.0))
                self.assert_no_write(code)
                self.assertIn("ORIGINAL_EXPOSURE_UNEXPECTED", self.output)

    def test_out_of_range_target_prevents_write(self):
        code = self.run_main(exposure=(5000.0, 2000.0, 1000000.0))
        self.assert_no_write(code)
        self.assertIn("TARGET_EXPOSURE_OUT_OF_RANGE", self.output)

    def test_open_refusal_reported_and_close_attempted(self):
        code = self.run_main(enums={"ExposureAuto": b"Continuous"})
        self.assert_failed(code)
        self.assertIn("EXPOSURE_AUTO_NOT_OFF", self.output)
        self.assertEqual(self.events[-1], "close")
        self.assertEqual(self.fake.writes(), [])
        self.assert_shut_down(self.fake)

    def test_unexpected_backend_info_prevents_write(self):
        real_get_info = ic.IRaypleCamera.get_info

        def synthetic_info(camera):
            return dict(real_get_info(camera), synthetic=True)

        with mock.patch.object(ic.IRaypleCamera, "get_info", synthetic_info):
            code = self.run_main()
        self.assert_no_write(code)
        self.assertIn("UNEXPECTED_BACKEND_INFO", self.output)

    def test_get_info_error_still_closes(self):
        code = self.run_main(fail={"get_info": RuntimeError("info failed")})
        self.assert_failed(code)
        self.assertEqual(self.fake.writes(), [])
        self.assert_closed_once()


# 18-20. Frame validation.
class FrameValidationTests(SmokeTestCase):
    def frame(self, **overrides):
        fields = {"pixels": np.zeros((HEIGHT, WIDTH), np.uint8), "bit_depth": 8, "exposure_us": 1000.0,
                  "source": "hardware", "frame_id": 1}
        fields.update(overrides)
        return CameraFrame(**fields)

    def test_valid_frame_accepted(self):
        smoke.validate_frame(self.frame(), 1000.0)

    def test_invalid_frames_refused(self):
        cases = (
            ({"source": "synthetic"}, "NOT_HARDWARE_FRAME"),
            ({"source": "HARDWARE"}, "NOT_HARDWARE_FRAME"),
            ({"pixels": np.zeros((HEIGHT, WIDTH), np.uint16)}, "INVALID_FRAME"),
            ({"pixels": np.zeros((HEIGHT, WIDTH), np.int8)}, "INVALID_FRAME"),
            ({"pixels": np.zeros((HEIGHT, WIDTH), np.float32)}, "INVALID_FRAME"),
            ({"pixels": np.zeros((WIDTH, HEIGHT), np.uint8)}, "INVALID_FRAME"),
            ({"pixels": np.zeros((2, 4), np.uint8)}, "INVALID_FRAME"),
            ({"pixels": np.zeros((HEIGHT, WIDTH, 3), np.uint8)}, "INVALID_FRAME"),
            ({"pixels": [[0]]}, "INVALID_FRAME"),
            ({"bit_depth": 10}, "INVALID_FRAME"),
            ({"bit_depth": True}, "INVALID_FRAME"),
            ({"exposure_us": 5000.0}, "INVALID_FRAME"),
            ({"frame_id": 2}, "INVALID_FRAME"),
            ({"frame_id": True}, "INVALID_FRAME"),
        )
        for overrides, status in cases:
            with self.subTest(overrides=list(overrides)), self.assertRaises(smoke.SmokeError) as ctx:
                smoke.validate_frame(self.frame(**overrides), 1000.0)
            self.assertEqual(ctx.exception.status, status)
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_frame(SimpleNamespace(**self.frame().__dict__), 1000.0)

    def test_synthetic_provenance_refused_end_to_end(self):
        with mock.patch.object(ic, "CameraFrame", side_effect=lambda **f: CameraFrame(**dict(f, source="synthetic"))), \
                mock.patch.object(smoke, "analyze_frame") as analyze:
            code = self.run_main()
        self.assert_failed(code)
        self.assertIn("NOT_HARDWARE_FRAME", self.output)
        analyze.assert_not_called()
        self.assert_closed_once()

    def test_wrong_dtype_refused_end_to_end(self):
        def widen(**fields):
            return CameraFrame(**dict(fields, pixels=fields["pixels"].astype(np.uint16)))

        with mock.patch.object(ic, "CameraFrame", side_effect=widen):
            code = self.run_main()
        self.assert_failed(code)
        self.assertIn("pixel dtype uint16", self.output)
        self.assert_closed_once()

    def test_wrong_native_shape_refused_end_to_end(self):
        code = self.run_main(payloads=(base.PAYLOAD_1,), width=4, height=2)
        self.assert_failed(code)
        self.assertIn("expected (1200, 1920)", self.output)
        self.assertNotIn("min DN", self.output)
        self.assert_closed_once()
        self.assertIn("RESTORED_AND_VERIFIED", self.output)

    def test_native_block_id_not_substituted_for_frame_id(self):
        code = self.run_main(block_ids=[77])
        self.assertEqual(code, smoke.EXIT_OK, self.output)
        self.assertIn("frame_id:            1 (per-backend acquisition counter)", self.output)
        self.assertIn("Native SDK blockId:  77 (camera-reported; distinct from frame_id)", self.output)


# 22-26. Errors fail closed without retries.
class FailureTests(SmokeTestCase):
    def test_closed_after_exposure_write_error(self):
        code = self.run_main(codes={WRITE: [-111]})
        self.assert_failed(code)
        self.assertIn("EXPOSURE_WRITE_FAILED", self.output)
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])  # one write, then backend restoration
        self.assertNotIn("capture_frame", self.events)
        self.assert_closed_once()
        self.assertIn("RESTORED_AND_VERIFIED", self.output)

    def test_readback_mismatch_not_accepted(self):
        code = self.run_main(apply_set=lambda value, index: 998.4 if index == 0 else value)
        self.assert_failed(code)
        self.assertIn("EXPOSURE_READBACK_MISMATCH", self.output)
        self.assertNotIn("capture_frame", self.events)
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
        self.assert_closed_once()

    def test_closed_after_capture_error(self):
        code = self.run_main(codes={"IMV_GetFrame": -119})
        self.assert_failed(code)
        self.assertIn("GET_FRAME_FAILED", self.output)
        self.assert_no_more_than_one_acquisition()
        self.assert_closed_once()
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])

    def test_closed_after_analysis_error(self):
        with mock.patch.object(smoke, "analyze_frame", side_effect=ValueError("analysis failed")):
            code = self.run_main()
        self.assert_failed(code)
        self.assertIn("analysis failed", self.output)
        self.assertNotIn("min DN", self.output)
        self.assert_closed_once()

    def test_keyboard_interrupt_during_capture_closes_and_fails(self):
        code = self.run_main(raises={"IMV_GetFrame": KeyboardInterrupt()})
        self.assert_failed(code, smoke.EXIT_INTERRUPTED)
        self.assertIn("INTERRUPTED", self.output)
        self.assert_no_more_than_one_acquisition()
        self.assert_closed_once()
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])

    def test_close_failure_prevents_pass(self):
        code = self.run_main(codes={"IMV_Close": -101})
        self.assert_failed(code, smoke.EXIT_CLEANUP_FAILED)
        self.assertIn("Backend close:       FAILED: CLEANUP_FAILED", self.output)
        self.assertIn("IMV_Close FAILED with code -101", self.output)
        self.assertEqual(self.events.count("close"), 1)  # never retried
        self.assertEqual(self.fake.counts["IMV_Close"], 1)
        self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)  # never reopened

    def test_restore_failure_prevents_pass(self):
        for kwargs, status in (({"codes": {WRITE: [0, -101]}}, ic.RESTORE_WRITE_FAILED),
                               ({"apply_set": lambda value, index: 4999.0 if index == 1 else value},
                                ic.RESTORE_MISMATCH)):
            with self.subTest(status=status):
                code = self.run_main(**kwargs)
                self.assert_failed(code, smoke.EXIT_RESTORE_FAILED)
                self.assertIn(f"Exposure restoration: FAILED: {status}", self.output)
                self.assertNotIn("RESTORED_AND_VERIFIED", self.output)
                self.assertEqual(self.events.count("close"), 1)
                self.assertEqual(self.fake.counts[WRITE], 2)  # target + one restoration attempt

    def test_unexpected_close_exception_reported_unverified(self):
        code = self.run_main(fail={"close": OSError("native crash")})
        self.assert_failed(code, smoke.EXIT_RESTORE_FAILED)
        self.assertIn("Exposure restoration: UNVERIFIED", self.output)
        self.assertIn("Backend close:       FAILED: OSError('native crash')", self.output)

    def test_keyboard_interrupt_during_close_not_success(self):
        code = self.run_main(fail={"close": KeyboardInterrupt()})
        self.assert_failed(code, smoke.EXIT_INTERRUPTED)
        self.assertIn("UNVERIFIED", self.output)

    def test_sdk_load_failure_reported(self):
        def failing_preload():
            raise ic.discovery.DiscoveryError("HuaRay SDK file not found")

        with mock.patch.object(ic, "preload_vendor_sdk", side_effect=failing_preload) as preload, \
                mock.patch.object(ic, "IRaypleCamera") as backend, redirect_stdout(io.StringIO()) as out:
            code = smoke.main(["--check-once"], input_fn=lambda prompt: smoke.CONFIRMATION_PHRASE,
                              stdin=FakeTty(True))
        self.assertEqual(code, smoke.EXIT_FAILED)
        preload.assert_called_once_with()
        backend.assert_not_called()
        self.assertIn("SDK_LOAD_FAILED", out.getvalue())

    def test_no_retry_on_any_failure(self):
        for kwargs in ({"codes": {"IMV_GetFrame": -119}}, {"codes": {"IMV_StartGrabbing": -101}},
                       {"codes": {WRITE: [-111]}}, {"status": 3}, {"codes": {"IMV_ReleaseFrame": [-101]}}):
            with self.subTest(kwargs=kwargs):
                code = self.run_main(**kwargs)
                self.assertNotEqual(code, smoke.EXIT_OK)
                self.assert_no_more_than_one_acquisition()
                self.assertEqual(self.events.count("set_exposure_us"), 1)
                self.assertEqual(self.events.count("close"), 1)
                self.assertEqual(self.preload.call_count, 1)
                self.fake.frames_outstanding.clear()  # the fake keeps a frame whose release "failed"


# 27-30. Isolation.
class IsolationTests(SmokeTestCase):
    SOURCE = (HW3_DIR / "irayple_backend_smoke.py").read_text(encoding="utf-8")

    def test_no_direct_sdk_calls_in_cli(self):
        self.assertEqual(re.findall(r"IMV_\w+", self.SOURCE), [])
        self.assertEqual(re.findall(r"\._(?:sdk|camera)\b", self.SOURCE), [])  # no backend internals
        self.assertEqual(re.findall(r"(?<![\w.])open\(", self.SOURCE), [])  # no file writes
        self.assertEqual(re.findall(r"(?<!pre)load_vendor_sdk", self.SOURCE), [])  # only the backend's preload
        for forbidden in ("ctypes", "chdir", "add_dll_directory", "WinDLL", "import_module",
                          "retry", "Thread(", "asyncio", "getenv", "environ", "np.save", "tofile", "imwrite"):
            self.assertNotIn(forbidden, self.SOURCE)
        tree = ast.parse(self.SOURCE)
        self.assertEqual([node for node in ast.walk(tree) if isinstance(node, (ast.While, ast.For))
                          and "capture" in ast.unparse(node)], [])
        self.assertEqual(len(re.findall(r"camera\.capture_frame\(\)", self.SOURCE)), 1)
        self.assertEqual(len(re.findall(r"ic\.preload_vendor_sdk\(\)", self.SOURCE)), 1)

    def test_imports_limited_to_existing_hw3_modules(self):
        imported = set()
        for node in ast.walk(ast.parse(self.SOURCE)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"argparse", "math", "sys", "numpy", "irayple_camera", "irayple_discovery",
                                    "irayple_exposure_sweep", "irayple_first_frame", "camera_contract",
                                    "measurements"})

    def test_no_mcp_langgraph_openai_network_or_files(self):
        self.assertNotRegex(self.SOURCE.lower(), r"openai|langgraph|mcp|anthropic|api_key|socket")
        before = set(sys.modules)
        with mock.patch.object(builtins, "open", side_effect=AssertionError("file opened")), \
                mock.patch("socket.socket", side_effect=AssertionError("network")), \
                mock.patch("os.getenv", side_effect=AssertionError("environment read")):
            code = self.run_main()
        self.assertEqual(code, smoke.EXIT_OK, self.output)
        for name in set(sys.modules) - before:
            self.assertNotRegex(name, r"openai|mcp|llm|anthropic|langgraph")

    def test_no_real_sdk_or_working_directory_change(self):
        cwd = os.getcwd()
        self.run_main()  # setUp guards WinDLL, CDLL, add_dll_directory, chdir and load_vendor_sdk
        ic.discovery.load_vendor_sdk.assert_not_called()
        self.assertEqual(os.getcwd(), cwd)
        self.assertNotIn("IMVApi", sys.modules)
        self.assertEqual(len(self.constructed), 1)
        self.assertIs(self.constructed[0][0], self.fake.ns)  # the fake binding, never a vendor module
        self.assertNotIsInstance(self.fake.ns, type(ctypes))


if __name__ == "__main__":
    unittest.main()
