import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measurements import analyze_frame, analyze_temporal_pair  # noqa: E402


class AnalyzeFrameTests(unittest.TestCase):
    def setUp(self):
        # 2x4 8-bit frame: sum = 1020, mean = 127.5
        self.frame8 = np.array([[0, 0, 255, 255], [100, 155, 0, 255]], dtype=np.uint8)

    def test_mean_brightness_8bit(self):
        result = analyze_frame(self.frame8, 8)
        self.assertAlmostEqual(result["mean_dn"], 127.5)
        self.assertEqual(result["width"], 4)
        self.assertEqual(result["height"], 2)
        self.assertEqual(result["min_dn"], 0)
        self.assertEqual(result["max_dn"], 255)

    def test_mean_fraction_of_full_scale(self):
        result = analyze_frame(self.frame8, 8)
        self.assertAlmostEqual(result["mean_fraction_of_full_scale"], 127.5 / 255)

    def test_saturation_and_zero_fraction(self):
        result = analyze_frame(self.frame8, 8)
        self.assertAlmostEqual(result["saturation_fraction"], 3 / 8)
        self.assertAlmostEqual(result["zero_fraction"], 3 / 8)

    def test_histogram_total_and_range(self):
        result = analyze_frame(self.frame8, 8, bins=16)
        hist = result["histogram"]
        self.assertEqual(sum(hist["counts"]), 8)
        self.assertEqual(len(hist["counts"]), 16)
        self.assertEqual(len(hist["bin_edges_dn"]), 17)
        self.assertEqual(hist["range_dn"], [0, 256])

    def test_10bit_in_uint16(self):
        frame = np.array([[0, 1023], [512, 1023]], dtype=np.uint16)
        result = analyze_frame(frame, 10)
        self.assertEqual(result["full_scale_dn"], 1023)
        self.assertAlmostEqual(result["mean_dn"], 2558 / 4)
        self.assertAlmostEqual(result["mean_fraction_of_full_scale"], (2558 / 4) / 1023)
        self.assertAlmostEqual(result["saturation_fraction"], 0.5)
        self.assertEqual(result["histogram"]["range_dn"], [0, 1024])

    def test_rejects_value_above_bit_depth(self):
        frame = np.array([[0, 1024]], dtype=np.uint16)
        with self.assertRaises(ValueError):
            analyze_frame(frame, 10)

    def test_rejects_rgb(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            analyze_frame(frame, 8)

    def test_rejects_empty(self):
        frame = np.zeros((0, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            analyze_frame(frame, 8)

    def test_rejects_float(self):
        frame = np.array([[0.0, 0.5]], dtype=np.float32)
        with self.assertRaises(TypeError):
            analyze_frame(frame, 8)

    def test_rejects_unsupported_bit_depth(self):
        frame = np.zeros((2, 2), dtype=np.uint16)
        for depth in (0, 9, 11, 32):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                analyze_frame(frame, depth)

    def test_rejects_dtype_bit_depth_mismatch(self):
        with self.assertRaises(TypeError):
            analyze_frame(np.zeros((2, 2), dtype=np.uint8), 10)
        with self.assertRaises(TypeError):
            analyze_frame(np.zeros((2, 2), dtype=np.uint16), 8)

    def test_output_is_json_compatible(self):
        result = analyze_frame(self.frame8, 8)
        self.assertEqual(json.loads(json.dumps(result)), result)


class AnalyzeTemporalPairTests(unittest.TestCase):
    def test_rejects_mismatched_shapes(self):
        a = np.zeros((2, 2), dtype=np.uint8)
        b = np.zeros((2, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            analyze_temporal_pair(a, b, 8)

    def test_identical_frames_zero_noise(self):
        a = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        result = analyze_temporal_pair(a, a.copy(), 8)
        self.assertEqual(result["mean_difference_dn"], 0.0)
        self.assertEqual(result["temporal_noise_proxy_dn"], 0.0)

    def test_known_temporal_noise_proxy(self):
        # d = [0, 2, 0, 2]: mean 1, population variance 1 -> proxy sqrt(1/2)
        a = np.array([[10, 12], [10, 12]], dtype=np.uint16)
        b = np.array([[10, 10], [10, 10]], dtype=np.uint16)
        result = analyze_temporal_pair(a, b, 12)
        self.assertAlmostEqual(result["mean_difference_dn"], 1.0)
        self.assertAlmostEqual(result["difference_spatial_std_dn"], 1.0)
        self.assertAlmostEqual(result["temporal_noise_proxy_dn"], math.sqrt(0.5))

    def test_mean_brightness_drift(self):
        # b brighter than a by 5 DN everywhere; unsigned subtraction must not wrap
        a = np.array([[0, 10], [20, 30]], dtype=np.uint8)
        b = a + np.uint8(5)
        result = analyze_temporal_pair(a, b, 8)
        self.assertAlmostEqual(result["mean_difference_dn"], -5.0)
        self.assertAlmostEqual(result["temporal_noise_proxy_dn"], 0.0)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_inputs_not_mutated(self):
        a = np.array([[1, 2], [3, 255]], dtype=np.uint8)
        b = np.array([[4, 0], [3, 250]], dtype=np.uint8)
        a_copy, b_copy = a.copy(), b.copy()
        analyze_frame(a, 8)
        analyze_temporal_pair(a, b, 8)
        np.testing.assert_array_equal(a, a_copy)
        np.testing.assert_array_equal(b, b_copy)
        self.assertEqual(a.dtype, np.uint8)
        self.assertEqual(b.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
