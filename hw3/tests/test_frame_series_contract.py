"""Offline tests for hw3/frame_series_contract.py. Hand-built samples only; no camera, SDK or MCP."""

import ast
import dataclasses
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import camera_contract  # noqa: E402
import frame_series_contract as fsc  # noqa: E402
from frame_series_contract import FrameSample, FrameSeries  # noqa: E402


def sample(i, **overrides):
    values = dict(frame_index=i, native_block_id=100 + i, device_timestamp_raw=5000 + 37 * i,
                  host_t_ns=10**12 + 10_000_000 * i, mean_dn=100.0 + i, saturation_fraction=0.0)
    values.update(overrides)
    return FrameSample(**values)


def series(n=4, requested=None, **overrides):
    values = dict(samples=tuple(sample(i) for i in range(n)),
                  requested_frames=n if requested is None else requested, acquired_frames=n,
                  exposure_us=1000.0, width=640, height=480, bit_depth=8,
                  source=fsc.SOURCE_SYNTHETIC, status=fsc.STATUS_COMPLETED)
    values.update(overrides)
    return FrameSeries(**values)


# 1. Contract creation and validation.
class CreationTests(unittest.TestCase):
    def test_valid_sample_round_trips(self):
        s = sample(3)
        self.assertEqual((s.frame_index, s.native_block_id, s.device_timestamp_raw, s.host_t_ns),
                         (3, 103, 5111, 10**12 + 30_000_000))
        self.assertEqual((s.mean_dn, s.saturation_fraction), (103.0, 0.0))

    def test_numpy_scalars_are_normalized_to_python_types(self):
        s = sample(0, native_block_id=np.uint64(7), host_t_ns=np.int64(5),
                   mean_dn=np.float32(1.5), saturation_fraction=np.float64(0.25))
        self.assertIs(type(s.native_block_id), int)
        self.assertIs(type(s.host_t_ns), int)
        self.assertIs(type(s.mean_dn), float)
        self.assertIs(type(s.saturation_fraction), float)

    def test_optional_block_id_and_device_timestamp(self):
        s = sample(0, native_block_id=None, device_timestamp_raw=None)
        self.assertIsNone(s.native_block_id)
        self.assertIsNone(s.device_timestamp_raw)
        fs = series(3, samples=tuple(sample(i, native_block_id=None, device_timestamp_raw=None)
                                     for i in range(3)))
        self.assertFalse(fs.has_block_ids)
        self.assertFalse(fs.has_device_timestamps)
        self.assertIsNone(fs.dropped_frames_by_block_id)
        self.assertIsNone(fs.device_tick_span)

    def test_valid_series_metadata(self):
        fs = series(4)
        self.assertTrue(fs.is_complete)
        self.assertEqual(fs.acquired_frames, 4)
        self.assertEqual(fs.host_span_ns, 30_000_000)
        self.assertAlmostEqual(fs.host_span_s, 0.03)
        self.assertEqual(fs.device_tick_span, 111)
        self.assertIsNone(fs.synthetic_device_span_s)
        self.assertEqual(fs.dropped_frames_by_block_id, 0)

    def test_list_of_samples_is_converted_to_tuple(self):
        fs = series(2, samples=[sample(0), sample(1)])
        self.assertIsInstance(fs.samples, tuple)

    def test_non_sample_elements_rejected(self):
        with self.assertRaises(TypeError):
            series(2, samples=(sample(0), {"frame_index": 1}))
        with self.assertRaises(TypeError):
            series(2, samples=np.zeros(2))

    def test_unknown_source_and_status_rejected(self):
        with self.assertRaises(ValueError):
            series(source="simulated")
        with self.assertRaises(ValueError):
            series(status="ok")

    def test_source_constants_match_camera_contract(self):
        self.assertEqual(fsc.SOURCE_HARDWARE, camera_contract.SOURCE_HARDWARE)
        self.assertEqual(fsc.SOURCE_SYNTHETIC, camera_contract.SOURCE_SYNTHETIC)

    def test_frame_index_must_match_position(self):
        with self.assertRaises(ValueError):
            series(3, samples=(sample(0), sample(2), sample(1)))
        with self.assertRaises(ValueError):
            series(2, samples=(sample(1), sample(2)))

    def test_host_timestamps_strictly_increasing(self):
        for host in (10**12, 10**12 - 1):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    series(2, samples=(sample(0), sample(1, host_t_ns=host)))

    def test_block_ids_and_device_timestamps_strictly_increasing(self):
        for field, value in (("native_block_id", 100), ("native_block_id", 50),
                             ("device_timestamp_raw", 5000), ("device_timestamp_raw", 1)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    series(2, samples=(sample(0), sample(1, **{field: value})))

    def test_optional_fields_present_for_all_or_none(self):
        for field in ("native_block_id", "device_timestamp_raw"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    series(2, samples=(sample(0), sample(1, **{field: None})))

    def test_device_timestamp_unit_is_never_assumed(self):
        fs = series(4)
        self.assertIsNone(fs.synthetic_device_span_s)
        self.assertNotIn("device_tick_hz", fs.analyzer_inputs())
        self.assertEqual(fs.analyzer_inputs()["device_ts"].tolist(), [5000, 5037, 5074, 5111])

    def test_hardware_series_rejects_tick_frequency(self):
        hardware = series(4, source=fsc.SOURCE_HARDWARE)
        self.assertIsNone(hardware.synthetic_device_tick_hz)
        with self.assertRaises(ValueError):
            series(4, source=fsc.SOURCE_HARDWARE, synthetic_device_tick_hz=1e6)
        with self.assertRaises(ValueError):
            hardware.analyzer_inputs(use_synthetic_tick_hz=True)

    def test_synthetic_tick_frequency_validation(self):
        fs = series(4, synthetic_device_tick_hz=3700)
        self.assertEqual(fs.synthetic_device_tick_hz, 3700.0)
        self.assertAlmostEqual(fs.synthetic_device_span_s, 111 / 3700)
        for bad in (0, -1.0, math.nan, math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    series(4, synthetic_device_tick_hz=bad)
        with self.assertRaises(ValueError):
            series(2, samples=tuple(sample(i, device_timestamp_raw=None) for i in range(2)),
                   synthetic_device_tick_hz=1e6)


# 2. Immutability.
class ImmutabilityTests(unittest.TestCase):
    def test_sample_is_frozen(self):
        s = sample(0)
        for field in dataclasses.fields(FrameSample):
            with self.subTest(field=field.name):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(s, field.name, 1)

    def test_series_is_frozen(self):
        fs = series(3)
        for field in dataclasses.fields(FrameSeries):
            with self.subTest(field=field.name):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(fs, field.name, 1)
        with self.assertRaises(AttributeError):
            fs.samples.append(sample(3))

    def test_input_list_mutation_does_not_affect_series(self):
        samples = [sample(0), sample(1)]
        fs = series(2, samples=samples)
        samples.append(sample(2))
        samples[0] = sample(0, mean_dn=1.0)
        self.assertEqual(len(fs.samples), 2)
        self.assertEqual(fs.samples[0].mean_dn, 100.0)

    def test_analyzer_inputs_are_independent_copies(self):
        fs = series(4)
        inputs = fs.analyzer_inputs()
        inputs["mean_dn"][:] = 0.0
        inputs["host_t_ns"][:] = 0
        self.assertEqual(fs.samples[0].mean_dn, 100.0)
        self.assertEqual(fs.analyzer_inputs()["mean_dn"][0], 100.0)

    def test_equal_series_compare_equal_and_hash(self):
        self.assertEqual(series(3), series(3))
        self.assertEqual(hash(series(3)), hash(series(3)))


# 3. Count consistency and completion status.
class CountTests(unittest.TestCase):
    def test_acquired_must_equal_number_of_samples(self):
        with self.assertRaises(ValueError):
            series(3, acquired_frames=4, requested=4)
        with self.assertRaises(ValueError):
            series(3, acquired_frames=2)

    def test_acquired_cannot_exceed_requested(self):
        with self.assertRaises(ValueError):
            series(4, requested=3)

    def test_requested_must_be_positive(self):
        with self.assertRaises(ValueError):
            series(0, requested=0, status=fsc.STATUS_FAILED, stop_reason="x")

    def test_completed_requires_all_requested_frames(self):
        with self.assertRaises(ValueError):
            series(3, requested=5)

    def test_completed_rejects_stop_reason(self):
        with self.assertRaises(ValueError):
            series(3, stop_reason="spurious")

    def test_partial_series_is_explicit(self):
        fs = series(3, requested=5, status=fsc.STATUS_PARTIAL, stop_reason="timeout on frame 3")
        self.assertFalse(fs.is_complete)
        self.assertEqual((fs.requested_frames, fs.acquired_frames), (5, 3))

    def test_partial_and_failed_require_reason(self):
        for status in (fsc.STATUS_PARTIAL, fsc.STATUS_FAILED):
            for reason in (None, "", "   ", 3):
                with self.subTest(status=status, reason=reason):
                    with self.assertRaises(ValueError):
                        series(2, requested=5, status=status, stop_reason=reason)

    def test_partial_or_failed_cannot_hide_a_full_series(self):
        for status in (fsc.STATUS_PARTIAL, fsc.STATUS_FAILED):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    series(3, status=status, stop_reason="x")

    def test_partial_requires_a_frame_failed_may_be_empty(self):
        with self.assertRaises(ValueError):
            series(0, requested=3, status=fsc.STATUS_PARTIAL, stop_reason="x")
        empty = series(0, requested=3, status=fsc.STATUS_FAILED, stop_reason="no frames")
        self.assertEqual(empty.samples, ())
        self.assertIsNone(empty.host_span_ns)
        self.assertFalse(empty.has_block_ids)

    def test_single_frame_metadata(self):
        fs = series(1)
        self.assertIsNone(fs.host_span_ns)
        self.assertIsNone(fs.device_tick_span)
        self.assertEqual(fs.dropped_frames_by_block_id, 0)

    def test_dropped_frames_from_block_ids(self):
        samples = tuple(sample(i, native_block_id=b) for i, b in enumerate((10, 11, 14, 15, 17)))
        self.assertEqual(series(5, samples=samples).dropped_frames_by_block_id, 3)


# 4. Numeric bounds, types and non-finite inputs.
class NumericTests(unittest.TestCase):
    def test_nonfinite_floats_rejected(self):
        for field in ("mean_dn", "saturation_fraction"):
            for bad in (math.nan, math.inf, -math.inf, np.float64("nan")):
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(ValueError):
                        sample(0, **{field: bad})
        for bad in (math.nan, math.inf):
            with self.subTest(exposure=bad):
                with self.assertRaises(ValueError):
                    series(exposure_us=bad)

    def test_integer_fields_reject_float_bool_and_str(self):
        for field in ("frame_index", "native_block_id", "device_timestamp_raw", "host_t_ns"):
            for bad in (1.0, True, "1", np.float64(2.0)):
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(TypeError):
                        sample(0, **{field: bad})

    def test_real_fields_reject_bool_and_str(self):
        for field in ("mean_dn", "saturation_fraction"):
            for bad in (True, "1.0", None):
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(TypeError):
                        sample(0, **{field: bad})

    def test_integer_ranges(self):
        for field in ("frame_index", "native_block_id", "device_timestamp_raw", "host_t_ns"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    sample(0, **{field: -1})
        with self.assertRaises(ValueError):
            sample(0, host_t_ns=2**63)
        with self.assertRaises(ValueError):
            sample(0, device_timestamp_raw=2**64)
        self.assertEqual(sample(0, device_timestamp_raw=2**64 - 1).device_timestamp_raw, 2**64 - 1)

    def test_mean_dn_bounds(self):
        with self.assertRaises(ValueError):
            sample(0, mean_dn=-0.001)
        self.assertEqual(series(1, samples=(sample(0, mean_dn=255.0),)).samples[0].mean_dn, 255.0)
        with self.assertRaises(ValueError):
            series(1, samples=(sample(0, mean_dn=255.5),))
        twelve = series(1, bit_depth=12, samples=(sample(0, mean_dn=4095.0),))
        self.assertEqual(twelve.bit_depth, 12)

    def test_saturation_fraction_bounds(self):
        for good in (0, 0.5, 1):
            self.assertEqual(sample(0, saturation_fraction=good).saturation_fraction, float(good))
        for bad in (-1e-9, 1.000001):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    sample(0, saturation_fraction=bad)

    def test_series_geometry_and_exposure_bounds(self):
        for overrides in ({"width": 0}, {"height": 0}, {"bit_depth": 7}, {"bit_depth": 17},
                          {"exposure_us": 0.0}, {"exposure_us": -5.0}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    series(**overrides)
        for overrides in ({"width": 640.0}, {"bit_depth": True}, {"exposure_us": "1000"}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(TypeError):
                    series(**overrides)


# 13. No raw pixel storage.
class NoPixelStorageTests(unittest.TestCase):
    def test_fields_are_scalar_metadata_only(self):
        self.assertEqual([f.name for f in dataclasses.fields(FrameSample)],
                         ["frame_index", "native_block_id", "device_timestamp_raw", "host_t_ns",
                          "mean_dn", "saturation_fraction"])
        names = {f.name for f in dataclasses.fields(FrameSeries)}
        self.assertFalse({"pixels", "image", "frames", "data"} & names)

    def test_no_array_values_retained(self):
        fs = series(4)
        for value in vars(fs).values():
            self.assertNotIsInstance(value, np.ndarray)
        for s in fs.samples:
            for value in vars(s).values():
                self.assertIsInstance(value, (int, float, type(None)))


# 14. No physical hardware imports.
class ImportTests(unittest.TestCase):
    def test_only_stdlib_and_numpy_imports(self):
        tree = ast.parse((HW3_DIR / "frame_series_contract.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported, {"dataclasses", "math", "numpy"})
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for name in names:
            for forbidden in ("imv", "ctypes", "irayple", "mcp", "langgraph", "openai", "socket",
                              "subprocess", "__import__", "import_module"):
                self.assertNotIn(forbidden, name.lower(), name)


if __name__ == "__main__":
    unittest.main()
