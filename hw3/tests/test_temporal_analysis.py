"""Offline tests for hw3/temporal_analysis.py. Synthetic, deterministic data only; no camera, SDK or MCP.

Estimator tolerance: a record of T seconds cannot resolve components closer
than 1 / T, so the module reports tolerance_hz = resolution_hz = fs / N
(= 1 / (N * T_s), about 1 / T). The clean synthetic cases below must land
within half of that; every reference "match" uses the full tolerance.
"""

import ast
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import temporal_analysis as ta  # noqa: E402
from temporal_analysis import analyze_temporal_series  # noqa: E402

HOST_OFFSET_NS = 10**12
DEVICE_TICK_HZ = 100_000_000.0  # synthetic 100 MHz camera clock
DEVICE_OFFSET = 2**40


def times(fps, n):
    return np.arange(n, dtype=np.float64) / fps


def host_ns(t_s):
    return np.round(t_s * 1e9).astype(np.int64) + HOST_OFFSET_NS


def sine(t_s, hz, amplitude=20.0, base=100.0, phase=0.3):
    return base + amplitude * np.sin(2 * np.pi * hz * t_s + phase)


def square(t_s, hz, amplitude=20.0, base=100.0):
    # Samples are taken half a sample after each phase origin, so no sample lies on an edge.
    return base + amplitude * np.where(np.sin(2 * np.pi * hz * t_s + 0.05) >= 0, 1.0, -1.0)


def exposed_pwm(t_s, hz, exposure_s, duty=0.5, low=70.0, high=145.0):
    """Mean level of a PWM light integrated over a rectangular exposure starting at each t_s (exact integral)."""
    def on_cycles(u):  # on-time, in cycles, of the PWM between cycle 0 and cycle u
        return np.floor(u) * duty + np.minimum(u - np.floor(u), duty)
    phase = 0.137
    on_fraction = (on_cycles(hz * (t_s + exposure_s) + phase) - on_cycles(hz * t_s + phase)) / (hz * exposure_s)
    return low + (high - low) * on_fraction


def noisy(values, sigma=0.5, seed=1):
    return values + np.random.default_rng(seed).normal(0.0, sigma, len(values))


def codes(result):
    return {w["code"] for w in result["warnings"]}


class EstimateTestCase(unittest.TestCase):
    def assert_f0(self, result, hz, fraction=0.5):
        self.assertEqual(result["status"], ta.STATUS_OK, result["warnings"])
        estimate = result["estimate"]
        self.assertLessEqual(abs(estimate["f0_hz"] - hz), fraction * estimate["resolution_hz"])
        self.assertEqual(estimate["tolerance_hz"], estimate["resolution_hz"])


# 1-2. Sinusoidal flicker at sufficient frame rates.
class SinusoidTests(EstimateTestCase):
    def test_10hz_sine_at_200fps(self):
        t = times(200.0, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(sine(t, 10.0)),
                                         block_ids=np.arange(5, 405))
        self.assert_f0(result, 10.0)
        acq = result["acquisition"]
        self.assertEqual(acq["sample_kind"], "frames")
        self.assertEqual(acq["n_frames"], 400)
        self.assertAlmostEqual(acq["fps"], 200.0, places=6)
        self.assertAlmostEqual(acq["fps_host"], 200.0, places=6)
        self.assertIsNone(acq["fps_device"])
        self.assertAlmostEqual(acq["nyquist_hz"], 100.0, places=6)
        self.assertAlmostEqual(acq["duration_s"], 399 / 200.0, places=9)
        self.assertNotIn("exposure_us", acq)
        self.assertAlmostEqual(result["estimate"]["resolution_hz"], 0.5, places=6)
        self.assertGreater(result["estimate"]["snr"], result["estimate"]["snr_threshold"])
        self.assertEqual(result["timing"]["dropped_frames"], 0)
        self.assertIn("ALIASING_NOT_EXCLUDED", codes(result))
        self.assertIn("HOST_TIMESTAMPS", codes(result))
        self.assertEqual(result["estimate"]["harmonic_number_of_strongest"], 1)

    def test_50hz_sine_at_500fps(self):
        t = times(500.0, 1000)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(sine(t, 50.0), seed=2))
        self.assert_f0(result, 50.0)
        self.assertAlmostEqual(result["estimate"]["resolution_hz"], 0.5, places=6)

    def test_off_bin_frequency(self):
        t = times(200.0, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(sine(t, 13.37), seed=3))
        self.assert_f0(result, 13.37)

    def test_result_is_plain_json_compatible(self):
        import json
        t = times(200.0, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=sine(t, 10.0),
                                         block_ids=np.arange(400), reference_hz=10.0)
        json.dumps(result, allow_nan=False)


# 3. Square waves and harmonic ambiguity.
class HarmonicTests(EstimateTestCase):
    def test_10hz_square_wave_odd_harmonics(self):
        t = times(500.0, 1000)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(square(t, 10.0)))
        self.assert_f0(result, 10.0)
        peaks = [p["frequency_hz"] for p in result["estimate"]["peaks"]]
        for harmonic in (30.0, 50.0, 70.0):
            self.assertTrue(any(abs(p - harmonic) < 0.5 for p in peaks), (harmonic, peaks))
        self.assertEqual(result["estimate"]["harmonic_number_of_strongest"], 1)

    def test_harmonic_stronger_than_fundamental_is_not_reported_as_f0(self):
        # The Event Lab 10 Hz / 70 Hz failure mode: the strongest peak is a harmonic.
        t = times(500.0, 1000)
        values = (100.0 + 3.0 * np.sin(2 * np.pi * 10 * t) + 10.0 * np.sin(2 * np.pi * 30 * t + 0.3)
                  + 2.0 * np.sin(2 * np.pi * 50 * t + 1.0))
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(values, 0.2))
        self.assert_f0(result, 10.0)
        self.assertAlmostEqual(result["estimate"]["strongest_peak_hz"], 30.0, delta=0.25)
        self.assertEqual(result["estimate"]["harmonic_number_of_strongest"], 3)
        self.assertIn("STRONGEST_PEAK_IS_HARMONIC", codes(result))

    def test_missing_fundamental_is_ambiguous(self):
        t = times(500.0, 1000)
        values = (100.0 + 10.0 * np.sin(2 * np.pi * 30 * t) + 6.0 * np.sin(2 * np.pi * 50 * t + 0.4)
                  + 4.0 * np.sin(2 * np.pi * 70 * t + 0.8))
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(values, 0.2))
        self.assertEqual(result["status"], ta.STATUS_AMBIGUOUS)
        self.assertIsNone(result["estimate"]["f0_hz"])
        self.assertAlmostEqual(result["estimate"]["candidate_f0_hz"], 30.0, delta=0.25)
        self.assertIn("HARMONIC_AMBIGUITY", codes(result))

    def test_two_unrelated_sources_are_ambiguous(self):
        t = times(500.0, 1000)
        values = 100.0 + 10.0 * np.sin(2 * np.pi * 10 * t) + 8.0 * np.sin(2 * np.pi * 23 * t)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(values, 0.2))
        self.assertEqual(result["status"], ta.STATUS_AMBIGUOUS)
        self.assertIsNone(result["estimate"]["f0_hz"])


# 3b. A harmonic aliased below the fundamental (Dual-Vision 50 Hz run, 2026-10-17: a 50 % PWM LED at
# ~164 fps with 5 ms exposure gave peaks at |fs - 3 f| and f, and the lowest-peak rule returned ambiguous).
class AliasedHarmonicTests(EstimateTestCase):
    N = 384
    EXPOSURE_S = 5e-3

    def pwm_result(self, hz, fps, extra=None):
        t = times(fps, self.N)
        values = exposed_pwm(t, hz, self.EXPOSURE_S)
        if extra is not None:
            values = values + extra(t)
        return analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(values, 0.3))

    def peak_near(self, result, hz):
        tolerance = ta.HARMONIC_TOL_BINS * result["estimate"]["resolution_hz"]
        return [p for p in result["estimate"]["peaks"] if abs(p["frequency_hz"] - hz) <= tolerance]

    def test_pwm_with_third_harmonic_aliased_below_f0(self):
        hz, fps = 50.3, 164.16
        result = self.pwm_result(hz, fps)
        self.assertTrue(self.peak_near(result, abs(fps - 3 * hz)), result["estimate"]["peaks"])
        self.assert_f0(result, hz)
        self.assertIn("LOWER_PEAK_IS_ALIASED_HARMONIC", codes(result))
        self.assertIn("ALIASING_NOT_EXCLUDED", codes(result))
        self.assertIn("HOST_TIMESTAMPS", codes(result))
        self.assertEqual(result["estimate"]["harmonic_number_of_strongest"], 1)

    def test_alias_position_moves_with_the_sampling_rate(self):
        # Rates where the weak alias does not by chance explain f as one of ITS harmonics; where it does
        # (e.g. 158 fps), see test_coincidental_lowest_peak_explanations_are_ambiguous.
        hz = 50.3
        for fps in (145.0, 160.0, 164.16, 170.0, 180.0):
            with self.subTest(fps=fps):
                result = self.pwm_result(hz, fps)
                alias = abs(fps - 3 * hz)
                self.assertLess(alias, hz)
                self.assertTrue(self.peak_near(result, alias), (alias, result["estimate"]["peaks"]))
                self.assert_f0(result, hz)
                self.assertIn("LOWER_PEAK_IS_ALIASED_HARMONIC", codes(result))

    def test_unrelated_second_frequency_stays_ambiguous(self):
        # 31 Hz is no direct or aliased harmonic (orders 1..10) of 50.3 Hz at 164.16 fps.
        result = self.pwm_result(50.3, 164.16, extra=lambda t: 10.0 * np.sin(2 * np.pi * 31.0 * t))
        self.assertTrue(self.peak_near(result, 31.0), result["estimate"]["peaks"])
        self.assertEqual(result["status"], ta.STATUS_AMBIGUOUS)
        self.assertIsNone(result["estimate"]["f0_hz"])
        self.assertIn("HARMONIC_AMBIGUITY", codes(result))

    def test_competing_fundamentals_stay_ambiguous(self):
        # At 160 fps, 30 Hz and 70 Hz each explain the other (alias of 3 x 30 = 70, alias of 5 x 70 = 30)
        # and the weaker 20 Hz peak (6 x 30 and 2 x 70); 20 Hz explains neither. 70 Hz is the strongest, so
        # both candidates also pass the strength rule (every lower peak is weaker than the candidate).
        t = times(160.0, self.N)
        values = (100.0 + 9.0 * np.sin(2 * np.pi * 30 * t) + 10.0 * np.sin(2 * np.pi * 70 * t + 0.5)
                  + 2.0 * np.sin(2 * np.pi * 20 * t + 1.1))
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(values, 0.2))
        self.assertEqual(result["status"], ta.STATUS_AMBIGUOUS)
        self.assertIsNone(result["estimate"]["f0_hz"])
        self.assertIn("COMPETING_FUNDAMENTALS", codes(result))

    def test_alias_stronger_than_candidate_stays_ambiguous(self):
        # A lower peak exactly at the alias of 3 f, but stronger than f: not an aliased harmonic of f.
        hz, fps = 50.3, 164.16
        t = times(fps, self.N)
        values = (100.0 + 4.0 * np.sin(2 * np.pi * hz * t)
                  + 12.0 * np.sin(2 * np.pi * abs(fps - 3 * hz) * t + 0.7))
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(values, 0.2))
        self.assertEqual(result["status"], ta.STATUS_AMBIGUOUS)
        self.assertIsNone(result["estimate"]["f0_hz"])
        self.assertIn("ALIAS_STRONGER_THAN_CANDIDATE", codes(result))

    def test_coincidental_lowest_peak_explanations_are_ambiguous(self):
        # Rates from the 105-199 fps sweep (DV-FRAME-ALIAS-FIX-01) where the weak alias of 3 f by chance explains
        # f = 50.3 Hz as one of its own harmonics (158 fps: 7 x 7.11 = 49.8 Hz); the lowest-peak rule alone
        # returned that alias as f0 with status ok. Both peaks explain everything, so nothing is determined.
        for fps in (109.0, 123.0, 141.75, 158.0, 171.0, 191.0):
            with self.subTest(fps=fps):
                result = self.pwm_result(50.3, fps)
                self.assertTrue(self.peak_near(result, abs(fps - 3 * 50.3)), result["estimate"]["peaks"])
                self.assertEqual(result["status"], ta.STATUS_AMBIGUOUS)
                self.assertIsNone(result["estimate"]["f0_hz"])
                self.assertIn("COMPETING_FUNDAMENTALS", codes(result))
                self.assertIn("ALIASING_NOT_EXCLUDED", codes(result))
                self.assertIn("HOST_TIMESTAMPS", codes(result))

    def test_competing_explanations_ignore_the_reference(self):
        t = times(158.0, self.N)
        values = noisy(exposed_pwm(t, 50.3, self.EXPOSURE_S), 0.3)
        for reference in (None, 50.3, abs(158.0 - 3 * 50.3)):
            with self.subTest(reference=reference):
                result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=values, reference_hz=reference)
                self.assertEqual(result["status"], ta.STATUS_AMBIGUOUS)
                self.assertIsNone(result["estimate"]["f0_hz"])

    def test_fundamental_above_nyquist_is_never_reported(self):
        # A 100 Hz PWM at 164.16 fps is itself aliased; any estimate stays below Nyquist.
        result = self.pwm_result(100.0, 164.16)
        nyquist = result["acquisition"]["nyquist_hz"]
        self.assertTrue(all(p["frequency_hz"] <= nyquist for p in result["estimate"]["peaks"]))
        if result["estimate"]["f0_hz"] is not None:
            self.assertLessEqual(result["estimate"]["f0_hz"], nyquist)
            self.assertNotAlmostEqual(result["estimate"]["f0_hz"], 100.0, delta=1.0)
        self.assertIn("ALIASING_NOT_EXCLUDED", codes(result))

    def test_valid_lowest_peak_keeps_the_existing_path(self):
        # Clean 10 Hz square wave: the lowest peak is the fundamental; the new rule is not used.
        t = times(500.0, 1000)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(square(t, 10.0)))
        self.assert_f0(result, 10.0)
        self.assertNotIn("LOWER_PEAK_IS_ALIASED_HARMONIC", codes(result))

    def test_reference_does_not_select_the_candidate(self):
        hz, fps = 50.3, 164.16
        t = times(fps, self.N)
        values = noisy(exposed_pwm(t, hz, self.EXPOSURE_S), 0.3)
        plain = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=values)
        for reference in (abs(fps - 3 * hz), 50.0, 60.0):
            with self.subTest(reference=reference):
                checked = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=values, reference_hz=reference)
                self.assertEqual(checked["status"], plain["status"])
                self.assertEqual(checked["estimate"], plain["estimate"])


# 4. No flicker.
class NoFlickerTests(unittest.TestCase):
    def test_constant_signal(self):
        t = times(200.0, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=np.full(400, 87, dtype=np.uint8))
        self.assertEqual(result["status"], ta.STATUS_NO_FLICKER)
        self.assertIsNone(result["estimate"]["f0_hz"])
        self.assertIn("NO_VARIATION_AFTER_DETREND", codes(result))

    def test_linear_drift_only(self):
        t = times(200.0, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=100.0 + 3.0 * t)
        self.assertEqual(result["status"], ta.STATUS_NO_FLICKER)

    def test_white_noise_does_not_produce_a_frequency(self):
        t = times(200.0, 400)
        for seed in range(10):
            result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(np.full(400, 100.0), 1.0, seed))
            self.assertEqual(result["status"], ta.STATUS_NO_FLICKER, seed)
            self.assertIsNone(result["estimate"]["f0_hz"])
            self.assertLess(result["estimate"]["snr"], result["estimate"]["snr_threshold"])


# 5-7. Insufficient and invalid data.
class InvalidDataTests(unittest.TestCase):
    def test_insufficient_sample_count(self):
        t = times(200.0, 10)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=sine(t, 10.0))
        self.assertEqual(result["status"], ta.STATUS_INSUFFICIENT)
        self.assertIn("TOO_FEW_FRAMES", codes(result))
        self.assertIsNone(result["estimate"]["f0_hz"])

    def test_empty_series(self):
        result = analyze_temporal_series(host_t_ns=np.array([], dtype=np.int64), mean_dn=[])
        self.assertEqual(result["status"], ta.STATUS_INSUFFICIENT)

    def test_nonmonotonic_timestamps(self):
        t = times(200.0, 400)
        stamps = host_ns(t)
        stamps[[100, 101]] = stamps[[101, 100]]
        result = analyze_temporal_series(host_t_ns=stamps, mean_dn=sine(t, 10.0))
        self.assertEqual(result["status"], ta.STATUS_INVALID)
        self.assertIn("NONMONOTONIC_TIMESTAMPS", codes(result))
        self.assertIsNone(result["estimate"]["f0_hz"])

    def test_duplicate_timestamps(self):
        t = times(200.0, 400)
        stamps = host_ns(t)
        stamps[200] = stamps[199]
        result = analyze_temporal_series(host_t_ns=stamps, mean_dn=sine(t, 10.0))
        self.assertEqual(result["status"], ta.STATUS_INVALID)
        self.assertIn("DUPLICATE_TIMESTAMPS", codes(result))

    def test_nonfinite_values(self):
        t = times(200.0, 400)
        values = sine(t, 10.0)
        values[50] = np.nan
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=values)
        self.assertEqual(result["status"], ta.STATUS_INVALID)
        self.assertIn("NONFINITE_VALUES", codes(result))
        stamps = host_ns(t).astype(np.float64)
        stamps[3] = np.inf
        result = analyze_temporal_series(host_t_ns=stamps, mean_dn=sine(t, 10.0))
        self.assertEqual(result["status"], ta.STATUS_INVALID)

    def test_block_id_reset_is_invalid(self):
        t = times(200.0, 400)
        blocks = np.concatenate([np.arange(1, 201), np.arange(1, 201)])
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=sine(t, 10.0), block_ids=blocks)
        self.assertEqual(result["status"], ta.STATUS_INVALID)
        self.assertIn("NONMONOTONIC_BLOCK_IDS", codes(result))

    def test_api_misuse_raises(self):
        t = times(200.0, 64)
        good = {"host_t_ns": host_ns(t), "mean_dn": sine(t, 10.0)}
        with self.assertRaises(TypeError):
            analyze_temporal_series(host_ns(t), sine(t, 10.0))  # keyword-only
        with self.assertRaises(ValueError):
            analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=sine(t, 10.0)[:-1])
        with self.assertRaises(ValueError):
            analyze_temporal_series(**good, block_ids=np.arange(10))
        with self.assertRaises(TypeError):
            analyze_temporal_series(**good, block_ids=np.arange(64, dtype=np.float64))
        with self.assertRaises(ValueError):
            analyze_temporal_series(**good, device_ts=np.arange(64), device_tick_hz=0.0)
        with self.assertRaises(ValueError):
            analyze_temporal_series(**good, device_tick_hz=1e6)  # no device_ts
        with self.assertRaises(ValueError):
            analyze_temporal_series(**good, reference_hz=-10.0)
        with self.assertRaises(TypeError):
            analyze_temporal_series(**good, reference_hz=True)
        with self.assertRaises(TypeError):
            analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=np.zeros(64, dtype=bool))
        with self.assertRaises(ValueError):
            analyze_temporal_series(host_t_ns=host_ns(t).reshape(8, 8), mean_dn=sine(t, 10.0).reshape(8, 8))


# 8-10. Native block ids, dropped frames and jitter.
class SamplingIntegrityTests(EstimateTestCase):
    def test_missing_block_ids_are_reported(self):
        t = times(200.0, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(sine(t, 10.0)))
        self.assert_f0(result, 10.0)
        self.assertIn("BLOCK_IDS_UNAVAILABLE", codes(result))
        self.assertFalse(result["timing"]["block_ids_available"])
        self.assertIsNone(result["timing"]["dropped_frames"])

    def test_dropped_frames_block_spectrum(self):
        t = times(200.0, 403)
        keep = np.setdiff1d(np.arange(403), [100, 101, 102])
        result = analyze_temporal_series(host_t_ns=host_ns(t[keep]), mean_dn=sine(t[keep], 10.0),
                                         block_ids=keep + 1)
        self.assertEqual(result["status"], ta.STATUS_IRREGULAR)
        self.assertEqual(result["timing"]["dropped_frames"], 3)
        self.assertIn("DROPPED_FRAMES", codes(result))
        self.assertIn("SPECTRUM_NOT_COMPUTED", codes(result))
        self.assertIsNone(result["estimate"]["f0_hz"])

    def test_dropped_frames_detected_without_block_ids(self):
        t = times(200.0, 403)
        keep = np.setdiff1d(np.arange(403), [200])
        result = analyze_temporal_series(host_t_ns=host_ns(t[keep]), mean_dn=sine(t[keep], 10.0))
        self.assertEqual(result["status"], ta.STATUS_IRREGULAR)
        self.assertEqual(result["timing"]["interval_gap_count"], 1)
        self.assertIn("INTERVAL_GAPS", codes(result))

    def test_block_id_gap_with_uniform_timestamps_is_still_irregular(self):
        t = times(200.0, 400)
        blocks = np.arange(400)
        blocks[300:] += 1
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=sine(t, 10.0), block_ids=blocks)
        self.assertEqual(result["status"], ta.STATUS_IRREGULAR)
        self.assertEqual(result["timing"]["dropped_frames"], 1)

    def test_material_jitter_blocks_spectrum(self):
        t = times(200.0, 400) + np.random.default_rng(4).uniform(-1.5e-3, 1.5e-3, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=sine(t, 10.0))
        self.assertEqual(result["status"], ta.STATUS_IRREGULAR)
        self.assertGreater(result["timing"]["jitter_rel"], ta.MAX_JITTER_REL)
        self.assertIn("EXCESSIVE_JITTER", codes(result))
        self.assertIsNone(result["estimate"]["f0_hz"])

    def test_small_jitter_is_tolerated(self):
        t = times(200.0, 400) + np.random.default_rng(5).uniform(-0.1e-3, 0.1e-3, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(sine(t, 10.0)))
        self.assert_f0(result, 10.0)
        self.assertLess(result["timing"]["jitter_rel"], 0.05)


# 11. Device timestamp scale.
class DeviceTimestampTests(EstimateTestCase):
    def setUp(self):
        self.t = times(200.0, 400)
        self.ticks = np.round(self.t * DEVICE_TICK_HZ).astype(np.uint64) + np.uint64(DEVICE_OFFSET)
        self.host = host_ns(self.t + np.random.default_rng(6).uniform(-0.2e-3, 0.2e-3, 400))
        self.values = noisy(sine(self.t, 10.0))

    def test_unverified_scale_is_never_used(self):
        result = analyze_temporal_series(host_t_ns=self.host, mean_dn=self.values, device_ts=self.ticks)
        self.assert_f0(result, 10.0)
        self.assertEqual(result["acquisition"]["timestamp_source"], ta.SOURCE_HOST)
        self.assertIsNone(result["acquisition"]["fps_device"])
        self.assertIn("DEVICE_TICK_UNVERIFIED", codes(result))
        device = result["timing"]["device_timestamps"]
        self.assertTrue(device["provided"])
        self.assertFalse(device["used"])
        self.assertIsNone(device["declared_tick_hz"])
        self.assertAlmostEqual(device["implied_tick_hz"] / DEVICE_TICK_HZ, 1.0, delta=0.01)

    def test_verified_scale_is_used(self):
        result = analyze_temporal_series(host_t_ns=self.host, mean_dn=self.values, device_ts=self.ticks,
                                         device_tick_hz=DEVICE_TICK_HZ)
        self.assert_f0(result, 10.0)
        self.assertEqual(result["acquisition"]["timestamp_source"], ta.SOURCE_DEVICE)
        self.assertAlmostEqual(result["acquisition"]["fps_device"], 200.0, places=6)
        self.assertEqual(result["acquisition"]["fps"], result["acquisition"]["fps_device"])
        self.assertTrue(result["timing"]["device_timestamps"]["used"])
        self.assertLess(result["timing"]["jitter_rel"], 1e-3)  # device clock, not host arrival jitter
        self.assertNotIn("HOST_TIMESTAMPS", codes(result))

    def test_incorrect_scale_falls_back_to_host(self):
        # A microsecond assumption applied to a 100 MHz counter.
        result = analyze_temporal_series(host_t_ns=self.host, mean_dn=self.values, device_ts=self.ticks,
                                         device_tick_hz=1e6)
        self.assert_f0(result, 10.0)
        self.assertEqual(result["acquisition"]["timestamp_source"], ta.SOURCE_HOST)
        self.assertIsNone(result["acquisition"]["fps_device"])
        self.assertFalse(result["timing"]["device_timestamps"]["scale_consistent"])
        self.assertIn("DEVICE_TICK_SCALE_MISMATCH", codes(result))


# 12-13. Reference frequency: validation only, never an input to the estimate.
class ReferenceTests(EstimateTestCase):
    def test_200hz_reference_above_nyquist(self):
        t = times(150.0, 300)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=noisy(sine(t, 200.0)),
                                         reference_hz=200.0)
        check = result["reference_validation"]
        self.assertEqual(check["verdict"], "not_recoverable")
        self.assertFalse(check["recoverable"])
        self.assertAlmostEqual(check["expected_alias_hz"], 50.0, places=6)
        self.assertIn("REFERENCE_AT_OR_ABOVE_NYQUIST", codes(result))
        self.assertIn("ALIASING_NOT_EXCLUDED", codes(result))
        # The data only show the alias; 200 Hz is never reported.
        self.assertAlmostEqual(result["estimate"]["f0_hz"], 50.0, delta=result["estimate"]["resolution_hz"])
        self.assertGreater(abs(result["estimate"]["f0_hz"] - 200.0), 100.0)

    def test_reference_exactly_at_nyquist_is_not_recoverable(self):
        t = times(200.0, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=sine(t, 10.0), reference_hz=100.0)
        self.assertEqual(result["reference_validation"]["verdict"], "not_recoverable")

    def test_reference_has_no_influence_on_the_estimate(self):
        t = times(500.0, 1000)
        values = noisy(square(t, 10.0))
        stamps = host_ns(t)
        baseline = analyze_temporal_series(host_t_ns=stamps, mean_dn=values)
        self.assertIsNone(baseline["reference_validation"])
        for reference in (10.0, 12.0, 30.0, 70.0, 249.0, 400.0):
            result = analyze_temporal_series(host_t_ns=stamps, mean_dn=values, reference_hz=reference)
            for key in ("status", "estimate", "acquisition", "timing"):
                self.assertEqual(result[key], baseline[key], (reference, key))
            self.assertFalse(result["reference_validation"]["used_for_estimation"])
        self.assertEqual(analyze_temporal_series(host_t_ns=stamps, mean_dn=values, reference_hz=10.0)
                         ["reference_validation"]["verdict"], "match")
        self.assertEqual(analyze_temporal_series(host_t_ns=stamps, mean_dn=values, reference_hz=12.0)
                         ["reference_validation"]["verdict"], "mismatch")
        self.assertEqual(analyze_temporal_series(host_t_ns=stamps, mean_dn=values, reference_hz=70.0)
                         ["reference_validation"]["verdict"], "mismatch")

    def test_reference_without_estimate(self):
        t = times(200.0, 400)
        result = analyze_temporal_series(host_t_ns=host_ns(t), mean_dn=np.full(400, 50.0), reference_hz=10.0)
        self.assertEqual(result["status"], ta.STATUS_NO_FLICKER)
        self.assertEqual(result["reference_validation"]["verdict"], "no_estimate")


# 14. Isolation: no hardware, SDK, MCP or network dependency.
class IsolationTests(unittest.TestCase):
    def test_only_numpy_and_stdlib_imports(self):
        source = Path(ta.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported, {"math", "numpy"})
        # Code identifiers only (the docstring legitimately names IMV_FrameInfo and MCP).
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for name in names:
            for forbidden in ("imv", "ctypes", "irayple", "mcp", "langgraph", "openai", "socket",
                              "subprocess", "camera", "session", "__import__", "import_module"):
                self.assertNotIn(forbidden, name.lower(), name)
        self.assertNotIn("open", names)

    def test_tolerance_constants_are_consistent(self):
        # A 50 % square wave's 9th harmonic (power 1/81) stays above the relative peak floor,
        # and HARMONIC_LIMIT covers it.
        self.assertGreater(1 / 81, ta.PEAK_REL_MIN)
        self.assertGreaterEqual(ta.HARMONIC_LIMIT, 9)
        self.assertAlmostEqual(math.exp(-(math.pi * ta.MAX_JITTER_REL) ** 2 / 2), 0.952, places=3)


if __name__ == "__main__":
    unittest.main()
