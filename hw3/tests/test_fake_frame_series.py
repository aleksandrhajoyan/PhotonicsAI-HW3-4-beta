"""Offline tests for hw3/fake_frame_series.py and its integration with the DV-01 analyzer.

Synthetic, seeded data only; no camera, SDK, MCP, network or model calls.
Frequency estimates must land within half of the analyzer's tolerance_hz
(one native bin, about 1 / record length), as in test_temporal_analysis.py.
"""

import ast
import dataclasses
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import fake_frame_series as ffs  # noqa: E402
import frame_series_contract as fsc  # noqa: E402
import temporal_analysis as ta  # noqa: E402
from fake_frame_series import FakeFrameSeries  # noqa: E402
from frame_series_contract import FrameSeries  # noqa: E402
from temporal_analysis import analyze_temporal_series  # noqa: E402

SYNTHETIC_TICK_HZ = 3_141_593.0  # arbitrary synthetic clock; not any camera's tick rate


def flicker(waveform="sine", hz=10.0, fps=100.0, n=300, **kwargs):
    kwargs.setdefault("amplitude_dn", 20.0)
    kwargs.setdefault("noise_dn", 0.5)
    kwargs.setdefault("phase_rad", 0.3)
    return FakeFrameSeries(n_frames=n, fps=fps, waveform=waveform, flicker_hz=hz, **kwargs)


def analyze(fs, **kwargs):
    use_tick = kwargs.pop("use_synthetic_tick_hz", False)
    return analyze_temporal_series(**fs.analyzer_inputs(use_synthetic_tick_hz=use_tick), **kwargs)


def codes(result):
    return {w["code"] for w in result["warnings"]}


class AnalyzerTestCase(unittest.TestCase):
    def assert_f0(self, result, hz):
        self.assertEqual(result["status"], ta.STATUS_OK, result["warnings"])
        estimate = result["estimate"]
        self.assertLessEqual(abs(estimate["f0_hz"] - hz), 0.5 * estimate["resolution_hz"])


class GeneratorTests(unittest.TestCase):
    def test_returns_completed_series_with_requested_frames(self):
        fs = flicker().acquire()
        self.assertIsInstance(fs, FrameSeries)
        self.assertEqual(fs.status, fsc.STATUS_COMPLETED)
        self.assertEqual((fs.requested_frames, fs.acquired_frames, len(fs.samples)), (300, 300, 300))
        self.assertIsNone(fs.stop_reason)

    def test_deterministic_for_same_seed(self):
        self.assertEqual(flicker(seed=7).acquire(), flicker(seed=7).acquire())
        generator = flicker(seed=7)
        self.assertEqual(generator.acquire(), generator.acquire())
        self.assertNotEqual(flicker(seed=7).acquire(), flicker(seed=8).acquire())

    def test_timestamps_at_configured_fps(self):
        fs = FakeFrameSeries(n_frames=5, fps=200.0, host_start_ns=1000).acquire()
        self.assertEqual([s.host_t_ns for s in fs.samples],
                         [1000, 5_001_000, 10_001_000, 15_001_000, 20_001_000])
        self.assertEqual(fs.host_span_ns, 20_000_000)

    def test_block_ids_consecutive_from_start(self):
        fs = FakeFrameSeries(n_frames=6, fps=50.0, block_id_start=40).acquire()
        self.assertEqual([s.native_block_id for s in fs.samples], [40, 41, 42, 43, 44, 45])
        self.assertEqual([s.frame_index for s in fs.samples], list(range(6)))
        self.assertEqual(fs.dropped_frames_by_block_id, 0)

    def test_sine_values_match_model(self):
        fs = flicker(hz=10.0, fps=100.0, n=20, noise_dn=0.0, phase_rad=0.0, base_dn=100.0).acquire()
        t = np.arange(20) / 100.0
        np.testing.assert_allclose([s.mean_dn for s in fs.samples],
                                   100.0 + 20.0 * np.sin(2 * np.pi * 10.0 * t), atol=1e-9)

    def test_square_values_and_duty(self):
        fs = flicker("square", hz=10.0, fps=100.0, n=100, noise_dn=0.0,
                     phase_rad=2 * math.pi * 0.05, duty=0.3, base_dn=100.0).acquire()
        values = np.array([s.mean_dn for s in fs.samples])
        self.assertEqual(set(values.tolist()), {80.0, 120.0})
        self.assertAlmostEqual(np.mean(values == 120.0), 0.3)

    def test_clipping_and_saturation_fraction(self):
        fs = FakeFrameSeries(n_frames=4, fps=10.0, base_dn=400.0).acquire()
        self.assertTrue(all(s.mean_dn == 255.0 and s.saturation_fraction == 1.0 for s in fs.samples))
        dark = FakeFrameSeries(n_frames=4, fps=10.0, base_dn=-5.0).acquire()
        self.assertTrue(all(s.mean_dn == 0.0 and s.saturation_fraction == 0.0 for s in dark.samples))
        deep = FakeFrameSeries(n_frames=4, fps=10.0, base_dn=400.0, bit_depth=12).acquire()
        self.assertTrue(all(s.mean_dn == 400.0 and s.saturation_fraction == 0.0 for s in deep.samples))

    def test_metadata_passed_through(self):
        fs = FakeFrameSeries(n_frames=3, fps=20.0, exposure_us=2500.0, width=1280, height=1024,
                             bit_depth=10).acquire()
        self.assertEqual((fs.exposure_us, fs.width, fs.height, fs.bit_depth),
                         (2500.0, 1280, 1024, 10))

    def test_partial_acquisition_is_never_reported_complete(self):
        fs = flicker(n=100, stop_after=40).acquire()
        self.assertEqual(fs.status, fsc.STATUS_PARTIAL)
        self.assertFalse(fs.is_complete)
        self.assertEqual((fs.requested_frames, fs.acquired_frames), (100, 40))
        self.assertIn("40 of 100", fs.stop_reason)
        failed = flicker(n=100, stop_after=0).acquire()
        self.assertEqual((failed.status, failed.acquired_frames), (fsc.STATUS_FAILED, 0))
        self.assertTrue(failed.stop_reason)
        self.assertEqual(flicker(n=10, stop_after=50).acquire().status, fsc.STATUS_COMPLETED)

    def test_invalid_configuration_rejected(self):
        bad = [
            (ValueError, dict(n_frames=0, fps=10.0)),
            (ValueError, dict(n_frames=10, fps=0.0)),
            (ValueError, dict(n_frames=10, fps=math.nan)),
            (TypeError, dict(n_frames=10.0, fps=10.0)),
            (ValueError, dict(n_frames=10, fps=10.0, waveform="triangle", flicker_hz=1.0)),
            (ValueError, dict(n_frames=10, fps=10.0, flicker_hz=1.0)),
            (TypeError, dict(n_frames=10, fps=10.0, waveform="sine")),
            (ValueError, dict(n_frames=10, fps=10.0, waveform="sine", flicker_hz=-1.0)),
            (ValueError, dict(n_frames=10, fps=10.0, amplitude_dn=-1.0)),
            (ValueError, dict(n_frames=10, fps=10.0, noise_dn=-1.0)),
            (ValueError, dict(n_frames=10, fps=10.0, base_dn=math.inf)),
            (ValueError, dict(n_frames=10, fps=10.0, duty=1.0)),
            (ValueError, dict(n_frames=10, fps=100.0, exposure_us=20000.0)),
            (ValueError, dict(n_frames=10, fps=10.0, bit_depth=17)),
            (ValueError, dict(n_frames=10, fps=10.0, drop_frames=(-1,))),
            (ValueError, dict(n_frames=10, fps=100.0, host_jitter_ns=5_000_000)),
            (ValueError, dict(n_frames=10, fps=10.0, synthetic_device_tick_hz=0.0)),
            (ValueError, dict(n_frames=10, fps=10.0, stop_after=-1)),
            (TypeError, dict(n_frames=10, fps=10.0, seed=1.5)),
        ]
        for error, kwargs in bad:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error):
                    FakeFrameSeries(**kwargs)


# 5-8. Flicker and steady scenes through the DV-01 analyzer.
class SpectralCaseTests(AnalyzerTestCase):
    def test_10hz_sine(self):
        result = analyze(flicker(hz=10.0, fps=100.0, n=300).acquire(), reference_hz=10.0)
        self.assert_f0(result, 10.0)
        self.assertEqual(result["reference_validation"]["verdict"], "match")

    def test_50hz_sine_with_adequate_fps(self):
        fs = flicker(hz=50.0, fps=200.0, n=400, exposure_us=1000.0).acquire()
        result = analyze(fs, reference_hz=50.0)
        self.assert_f0(result, 50.0)
        self.assertGreater(result["acquisition"]["nyquist_hz"], 50.0)
        self.assertEqual(result["reference_validation"]["verdict"], "match")

    def test_50hz_at_inadequate_fps_is_not_claimed(self):
        result = analyze(flicker(hz=50.0, fps=60.0, n=300).acquire(), reference_hz=50.0)
        self.assertEqual(result["reference_validation"]["verdict"], "not_recoverable")
        self.assertIn("REFERENCE_AT_OR_ABOVE_NYQUIST", codes(result))

    def test_steady_scene(self):
        noiseless = analyze(FakeFrameSeries(n_frames=200, fps=100.0).acquire())
        self.assertEqual(noiseless["status"], ta.STATUS_NO_FLICKER)
        self.assertIn("NO_VARIATION_AFTER_DETREND", codes(noiseless))
        noisy = analyze(FakeFrameSeries(n_frames=200, fps=100.0, noise_dn=0.5, seed=3).acquire())
        self.assertEqual(noisy["status"], ta.STATUS_NO_FLICKER, noisy["warnings"])
        self.assertIsNone(noisy["estimate"]["f0_hz"])

    def test_square_wave(self):
        fs = flicker("square", hz=10.0, fps=200.0, n=400, phase_rad=0.05).acquire()
        result = analyze(fs, reference_hz=10.0)
        self.assert_f0(result, 10.0)
        self.assertGreaterEqual(len(result["estimate"]["peaks"]), 2)  # odd harmonics present
        self.assertEqual(result["reference_validation"]["verdict"], "match")


# 9. Dropped frames.
class DroppedFrameTests(unittest.TestCase):
    def test_drops_leave_block_id_gaps_and_still_deliver_requested_count(self):
        fs = flicker(n=200, drop_frames=(50, 51, 120)).acquire()
        self.assertEqual(fs.status, fsc.STATUS_COMPLETED)
        self.assertEqual(fs.acquired_frames, 200)
        self.assertEqual(fs.dropped_frames_by_block_id, 3)
        ids = [s.native_block_id for s in fs.samples]
        self.assertNotIn(ffs.DEFAULT_BLOCK_ID_START + 50, ids)
        self.assertEqual(ids[50], ffs.DEFAULT_BLOCK_ID_START + 52)

    def test_analyzer_reports_drops_and_refuses_spectrum(self):
        result = analyze(flicker(n=200, drop_frames=(50, 51, 120)).acquire())
        self.assertEqual(result["status"], ta.STATUS_IRREGULAR)
        self.assertEqual(result["timing"]["dropped_frames"], 3)
        self.assertIn("DROPPED_FRAMES", codes(result))
        self.assertIsNone(result["estimate"]["f0_hz"])


# 10. Timestamp jitter.
class JitterTests(AnalyzerTestCase):
    def test_jitter_is_bounded_and_seeded(self):
        fs = flicker(fps=100.0, n=300, host_jitter_ns=2_000_000, seed=5).acquire()
        ideal = ffs.DEFAULT_HOST_START_NS + np.round(np.arange(300) / 100.0 * 1e9)
        offsets = np.array([s.host_t_ns for s in fs.samples]) - ideal
        self.assertLessEqual(np.max(np.abs(offsets)), 2_000_000)
        self.assertGreater(np.std(offsets), 500_000)
        self.assertEqual(fs, flicker(fps=100.0, n=300, host_jitter_ns=2_000_000, seed=5).acquire())

    def test_small_jitter_is_tolerated(self):
        fs = flicker(hz=10.0, fps=100.0, n=300, host_jitter_ns=100_000).acquire()
        self.assert_f0(analyze(fs), 10.0)

    def test_large_host_jitter_is_flagged(self):
        fs = flicker(hz=10.0, fps=100.0, n=300, host_jitter_ns=4_500_000).acquire()
        result = analyze(fs)
        self.assertEqual(result["status"], ta.STATUS_IRREGULAR)
        self.assertIn("EXCESSIVE_JITTER", codes(result))

    def test_declared_synthetic_clock_bypasses_host_jitter(self):
        fs = flicker(hz=10.0, fps=100.0, n=300, host_jitter_ns=4_500_000,
                     synthetic_device_tick_hz=SYNTHETIC_TICK_HZ).acquire()
        result = analyze(fs, use_synthetic_tick_hz=True)
        self.assert_f0(result, 10.0)
        self.assertEqual(result["acquisition"]["timestamp_source"], ta.SOURCE_DEVICE)


# 11. Synthetic provenance.
class ProvenanceTests(unittest.TestCase):
    def test_every_series_is_synthetic(self):
        configs = [dict(), dict(waveform="sine", flicker_hz=5.0), dict(stop_after=2),
                   dict(stop_after=0), dict(synthetic_device_tick_hz=1e6), dict(drop_frames=(1,))]
        for config in configs:
            with self.subTest(config=config):
                fs = FakeFrameSeries(n_frames=8, fps=50.0, **config).acquire()
                self.assertEqual(fs.source, fsc.SOURCE_SYNTHETIC)

    def test_source_is_not_configurable(self):
        with self.assertRaises(TypeError):
            FakeFrameSeries(n_frames=8, fps=50.0, source=fsc.SOURCE_HARDWARE)

    def test_no_device_clock_unless_explicitly_declared(self):
        fs = FakeFrameSeries(n_frames=8, fps=50.0).acquire()
        self.assertFalse(fs.has_device_timestamps)
        self.assertIsNone(fs.synthetic_device_tick_hz)
        self.assertIsNone(FakeFrameSeries(n_frames=8, fps=50.0).synthetic_device_tick_hz)

    def test_declared_clock_produces_matching_ticks(self):
        fs = FakeFrameSeries(n_frames=5, fps=50.0, synthetic_device_tick_hz=1000.0,
                             device_tick_start=7).acquire()
        self.assertEqual([s.device_timestamp_raw for s in fs.samples], [7, 27, 47, 67, 87])
        self.assertAlmostEqual(fs.synthetic_device_span_s, 0.08)

    def test_synthetic_tick_rate_not_used_by_analyzer_unless_opted_in(self):
        fs = flicker(synthetic_device_tick_hz=SYNTHETIC_TICK_HZ).acquire()
        self.assertNotIn("device_tick_hz", fs.analyzer_inputs())
        result = analyze(fs)
        self.assertIn("DEVICE_TICK_UNVERIFIED", codes(result))
        self.assertEqual(result["acquisition"]["timestamp_source"], ta.SOURCE_HOST)

    def test_synthetic_tick_rate_cannot_be_relabelled_hardware(self):
        fs = flicker(synthetic_device_tick_hz=SYNTHETIC_TICK_HZ).acquire()
        with self.assertRaises(ValueError):
            dataclasses.replace(fs, source=fsc.SOURCE_HARDWARE)


# 12. Integration with DV-01 (analyze_temporal_series unchanged).
class IntegrationTests(AnalyzerTestCase):
    def test_analyzer_inputs_match_analyzer_signature(self):
        fs = flicker(synthetic_device_tick_hz=SYNTHETIC_TICK_HZ).acquire()
        inputs = fs.analyzer_inputs(use_synthetic_tick_hz=True)
        self.assertEqual(set(inputs), {"host_t_ns", "mean_dn", "block_ids", "device_ts",
                                       "device_tick_hz"})
        self.assertEqual(inputs["host_t_ns"].dtype, np.int64)
        self.assertEqual(inputs["mean_dn"].dtype, np.float64)
        self.assertEqual(inputs["block_ids"].dtype, np.uint64)
        self.assertEqual(inputs["device_ts"].dtype, np.uint64)

    def test_end_to_end_report(self):
        fs = flicker(hz=10.0, fps=100.0, n=300, synthetic_device_tick_hz=SYNTHETIC_TICK_HZ).acquire()
        result = analyze(fs, use_synthetic_tick_hz=True, reference_hz=10.0)
        self.assert_f0(result, 10.0)
        acquisition, timing = result["acquisition"], result["timing"]
        self.assertEqual(acquisition["n_frames"], fs.acquired_frames)
        self.assertAlmostEqual(acquisition["fps"], 100.0, delta=1e-5)  # tick rounding
        self.assertAlmostEqual(acquisition["duration_s"], fs.synthetic_device_span_s, places=9)
        self.assertTrue(timing["block_ids_available"])
        self.assertEqual(timing["dropped_frames"], 0)
        self.assertTrue(timing["device_timestamps"]["scale_consistent"])
        self.assertEqual(result["reference_validation"]["verdict"], "match")

    def test_host_only_path(self):
        fs = flicker(hz=10.0, fps=100.0, n=300).acquire()
        inputs = fs.analyzer_inputs()
        self.assertNotIn("device_ts", inputs)
        result = analyze(fs)
        self.assert_f0(result, 10.0)
        self.assertAlmostEqual(result["acquisition"]["duration_s"], fs.host_span_s, places=9)

    def test_too_short_partial_series_is_insufficient(self):
        fs = flicker(n=300, stop_after=10).acquire()
        self.assertEqual(analyze(fs)["status"], ta.STATUS_INSUFFICIENT)


# 13-14. No pixel storage, no hardware imports.
class IsolationTests(unittest.TestCase):
    def test_no_pixel_arrays_in_output(self):
        fs = flicker(n=50).acquire()
        for value in vars(fs).values():
            self.assertNotIsInstance(value, np.ndarray)
        for s in fs.samples:
            for value in vars(s).values():
                self.assertIsInstance(value, (int, float, type(None)))

    def test_only_stdlib_numpy_and_contract_imports(self):
        tree = ast.parse((HW3_DIR / "fake_frame_series.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported, {"math", "numpy", "frame_series_contract"})
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for name in names:
            for forbidden in ("imv", "ctypes", "irayple", "mcp", "langgraph", "openai", "socket",
                              "subprocess", "camera", "__import__", "import_module"):
                self.assertNotIn(forbidden, name.lower(), name)


if __name__ == "__main__":
    unittest.main()
