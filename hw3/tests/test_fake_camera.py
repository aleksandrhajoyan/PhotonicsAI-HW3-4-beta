import ast
import inspect
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

from camera_contract import CameraBackend, CameraFrame  # noqa: E402
from fake_camera import FakeCamera  # noqa: E402
from measurements import analyze_frame  # noqa: E402

CONTRACT_METHODS = (
    "open",
    "close",
    "get_info",
    "get_exposure_limits_us",
    "set_exposure_us",
    "capture_frame",
)


def _open_camera():
    camera = FakeCamera()
    camera.open()
    return camera


def _capture_at(exposure_us):
    camera = _open_camera()
    camera.set_exposure_us(exposure_us)
    return camera.capture_frame()


class ContractTests(unittest.TestCase):
    def test_satisfies_backend_protocol(self):
        camera = FakeCamera()
        self.assertIsInstance(camera, CameraBackend)
        for name in CONTRACT_METHODS:
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(camera, name)))
        self.assertEqual(
            list(inspect.signature(camera.set_exposure_us).parameters), ["value"]
        )

    def test_exposure_limits(self):
        self.assertEqual(FakeCamera().get_exposure_limits_us(), (100.0, 10000.0))

    def test_info_states_synthetic(self):
        info = FakeCamera().get_info()
        self.assertEqual(info["source"], "synthetic")
        self.assertIs(info["synthetic"], True)
        self.assertEqual(info["bit_depth"], 8)
        self.assertEqual(json.loads(json.dumps(info)), info)


class OpenCloseTests(unittest.TestCase):
    def test_starts_closed(self):
        camera = FakeCamera()
        self.assertFalse(camera.is_open)
        self.assertFalse(camera.get_info()["is_open"])

    def test_open_and_close(self):
        camera = FakeCamera()
        camera.open()
        self.assertTrue(camera.is_open)
        camera.close()
        self.assertFalse(camera.is_open)

    def test_close_when_closed_is_safe(self):
        camera = FakeCamera()
        camera.close()
        camera.open()
        camera.close()
        camera.close()
        self.assertFalse(camera.is_open)

    def test_capture_before_open_rejected(self):
        with self.assertRaises(RuntimeError):
            FakeCamera().capture_frame()

    def test_capture_after_close_rejected(self):
        camera = _open_camera()
        camera.close()
        with self.assertRaises(RuntimeError):
            camera.capture_frame()

    def test_exposure_change_before_open_rejected(self):
        camera = FakeCamera()
        with self.assertRaises(RuntimeError):
            camera.set_exposure_us(1000.0)
        self.assertEqual(camera.get_info()["exposure_us"], 1000.0)


class ExposureTests(unittest.TestCase):
    def test_default_exposure(self):
        self.assertEqual(_open_camera().capture_frame().exposure_us, 1000.0)

    def test_valid_exposure_accepted(self):
        camera = _open_camera()
        for value, expected in ((100, 100.0), (2500.5, 2500.5), (10000.0, 10000.0)):
            with self.subTest(value=value):
                applied = camera.set_exposure_us(value)
                self.assertEqual(applied, expected)
                self.assertIsInstance(applied, float)
                self.assertEqual(camera.capture_frame().exposure_us, expected)

    def test_out_of_range_rejected(self):
        camera = _open_camera()
        camera.set_exposure_us(2000.0)
        for value in (99.999, 0, -1.0, 10000.001, 1e9):
            with self.subTest(value=value), self.assertRaises(ValueError):
                camera.set_exposure_us(value)
        self.assertEqual(camera.capture_frame().exposure_us, 2000.0)

    def test_nan_and_infinity_rejected(self):
        camera = _open_camera()
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                camera.set_exposure_us(value)

    def test_bool_and_non_numeric_rejected(self):
        camera = _open_camera()
        for value in (True, False, "1000", None, [1000]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                camera.set_exposure_us(value)


class FrameTests(unittest.TestCase):
    def test_shape_and_pixel_format(self):
        frame = _open_camera().capture_frame()
        self.assertIsInstance(frame, CameraFrame)
        self.assertEqual(frame.pixels.shape, (8, 8))
        self.assertEqual(frame.pixels.ndim, 2)
        self.assertEqual(frame.pixels.dtype, np.uint8)
        self.assertEqual(frame.bit_depth, 8)

    def test_source_is_synthetic(self):
        self.assertEqual(_open_camera().capture_frame().source, "synthetic")

    def test_deterministic_default_pattern(self):
        # At the 1000 us reference exposure the frame equals the fixed pattern.
        expected = (30 + 2 * np.arange(64)).reshape(8, 8).astype(np.uint8)
        first = _open_camera().capture_frame().pixels
        second = _open_camera().capture_frame().pixels
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)

    def test_higher_exposure_increases_mean_before_saturation(self):
        means = []
        for exposure in (100.0, 500.0, 1000.0, 1500.0):
            pixels = _capture_at(exposure).pixels
            self.assertLess(int(pixels.max()), 255)
            means.append(float(pixels.mean()))
        self.assertEqual(means, sorted(means))
        self.assertEqual(len(set(means)), len(means))

    def test_saturation_limited_to_255(self):
        # 2000 us: pixel k = 60 + 4k, which exceeds 255 for k >= 49 (15 pixels).
        partial = _capture_at(2000.0).pixels
        self.assertEqual(int(partial.max()), 255)
        self.assertEqual(int(np.count_nonzero(partial == 255)), 15)
        self.assertEqual(int(partial.flat[48]), 252)
        full = _capture_at(10000.0).pixels
        self.assertTrue(np.all(full == 255))

    def test_frame_ids_increase(self):
        camera = _open_camera()
        ids = [camera.capture_frame().frame_id for _ in range(3)]
        camera.close()
        camera.open()
        ids.append(camera.capture_frame().frame_id)
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_frames_do_not_share_storage(self):
        camera = _open_camera()
        first = camera.capture_frame()
        second = camera.capture_frame()
        self.assertFalse(np.shares_memory(first.pixels, second.pixels))
        self.assertTrue(first.pixels.flags.owndata)
        self.assertTrue(first.pixels.flags.writeable)
        first.pixels[:] = 0
        third = camera.capture_frame()
        self.assertEqual(int(second.pixels[0, 0]), 30)
        self.assertEqual(int(third.pixels[0, 0]), 30)


class MeasurementIntegrationTests(unittest.TestCase):
    def test_analyze_frame_accepts_default_frame(self):
        frame = _open_camera().capture_frame()
        self.assertEqual(frame.source, "synthetic")
        result = analyze_frame(frame.pixels, frame.bit_depth)
        self.assertEqual((result["width"], result["height"]), (8, 8))
        self.assertEqual(result["min_dn"], 30)
        self.assertEqual(result["max_dn"], 156)
        self.assertAlmostEqual(result["mean_dn"], 93.0)
        self.assertAlmostEqual(result["mean_fraction_of_full_scale"], 93.0 / 255)
        self.assertEqual(result["saturation_fraction"], 0.0)
        self.assertEqual(result["zero_fraction"], 0.0)
        self.assertEqual(sum(result["histogram"]["counts"]), 64)

    def test_analyze_frame_reports_full_saturation(self):
        frame = _capture_at(10000.0)
        result = analyze_frame(frame.pixels, frame.bit_depth)
        self.assertEqual(result["saturation_fraction"], 1.0)
        self.assertEqual(result["mean_dn"], 255.0)


class NoVendorSdkTests(unittest.TestCase):
    ALLOWED_IMPORTS = {"dataclasses", "typing", "numpy", "math", "numbers", "camera_contract"}

    def test_new_modules_import_only_allowed_modules(self):
        for name in ("camera_contract.py", "fake_camera.py"):
            tree = ast.parse((HW3_DIR / name).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add((node.module or "").split(".")[0])
            with self.subTest(module=name):
                self.assertTrue(
                    imported <= self.ALLOWED_IMPORTS,
                    f"unexpected imports: {imported - self.ALLOWED_IMPORTS}",
                )
                self.assertNotIn("ctypes", imported)
                self.assertNotIn("sdk_probe", imported)


if __name__ == "__main__":
    unittest.main()
