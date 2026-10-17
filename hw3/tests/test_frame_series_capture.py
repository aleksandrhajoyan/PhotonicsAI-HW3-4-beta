"""Offline tests for IRaypleCamera.capture_series. Every SDK call goes to the fake binding of
test_irayple_camera (extended here with per-frame timestamps); no hardware, network or LLM.
"""

import dataclasses
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import irayple_camera as ic  # noqa: E402
import test_irayple_camera as base  # noqa: E402  (fake SDK binding shared with the backend tests)
from frame_series_contract import FrameSeries  # noqa: E402

TIMEOUT_MS = 1000
DURATION_S = 2.0
# Mean DN and saturation fraction of base.PAYLOAD_1/2/3 (4 x 2 Mono8, 255 = saturated).
PAYLOAD_STATS = ((35.0, 0.0), (157.5, 0.25), (4.5, 0.0))


class SeriesSdk(base.FakeSdk):
    """FakeSdk that also reports IMV_FrameInfo.timeStamp and per-frame metadata overrides.

    block_ids / timestamps default to 1000 + k and 5_000_000 + 4321 * k for the k-th delivered frame.
    frame_overrides maps a frame index to IMV_FrameInfo fields to overwrite for that frame.
    """

    def __init__(self, timestamps=None, frame_overrides=None, **kwargs):
        self.timestamp_list = timestamps
        self.frame_overrides = frame_overrides or {}
        self.get_frame_timeouts = []
        super().__init__(**kwargs)

    def _on_IMV_GetFrame(self, frame, timeout_ms):
        index = self.frames_served
        self.get_frame_timeouts.append(timeout_ms)
        if self.block_ids is None:
            self.block_ids = _Sequence(lambda k: 1000 + k)
        super()._on_IMV_GetFrame(frame, timeout_ms)
        info = frame.frameInfo
        info.timeStamp = (5_000_000 + 4321 * index if self.timestamp_list is None
                          else self.timestamp_list[index])
        for name, value in self.frame_overrides.get(index, {}).items():
            setattr(info, name, value)


class _Sequence:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, index):
        return self.function(index)


class FakeClock:
    """Replaces irayple_camera.time: perf_counter_ns advances by step_ns on every read."""

    def __init__(self, start_ns=10**12, step_ns=1_000_000):
        self.now = start_ns
        self.step_ns = step_ns
        self.reads = []

    def perf_counter_ns(self):
        value = self.now
        self.now += self.step_ns
        self.reads.append(value)
        return value


def series_sequence(n):
    return ["IMV_StartGrabbing", *["IMV_GetFrame", "memmove", "IMV_ReleaseFrame"] * n, "IMV_StopGrabbing"]


class SeriesTestCase(base.OfflineTestCase):
    def make(self, **kwargs):
        self.fake = SeriesSdk(**kwargs)
        return self.fake, ic.IRaypleCamera(self.fake.ns)

    def capture(self, camera, n=10, **kwargs):
        kwargs.setdefault("frame_timeout_ms", TIMEOUT_MS)
        kwargs.setdefault("max_duration_s", DURATION_S)
        return camera.capture_series(n, **kwargs)

    def with_clock(self, clock):
        patcher = mock.patch.object(ic, "time", SimpleNamespace(perf_counter_ns=clock.perf_counter_ns))
        patcher.start()
        self.addCleanup(patcher.stop)
        return clock

    def assert_failed_cleanly(self, fake, camera, status, call=None):
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            (call or (lambda: self.capture(camera)))()
        self.assertEqual(ctx.exception.status, status)
        self.assertEqual(fake.counts["IMV_StartGrabbing"], 1)
        self.assertEqual(fake.counts["IMV_StopGrabbing"], 1)
        self.assertFalse(fake.is_grabbing)
        self.assertEqual(fake.frames_outstanding, {})
        self.assertEqual(fake.counts.get("IMV_ReleaseFrame", 0), len(fake.released_buffers))
        self.assertTrue(camera.is_open)
        return ctx.exception


# 1-6. Successful series and SDK call accounting.
class SuccessTests(SeriesTestCase):
    def test_ten_frame_series(self):
        fake, camera = self.opened()
        mark = len(fake.calls)
        series = self.capture(camera, 10)
        self.assertIsInstance(series, FrameSeries)
        self.assertEqual(self.calls_since(fake, mark), series_sequence(10))
        self.assertEqual((series.source, series.status, series.stop_reason), ("hardware", "completed", None))
        self.assertEqual((series.requested_frames, series.acquired_frames, len(series.samples)), (10, 10, 10))
        self.assertIsNone(series.synthetic_device_tick_hz)
        self.assertTrue(series.is_complete)
        self.assertEqual((series.width, series.height, series.bit_depth), (4, 2, 8))

    def test_longer_series_up_to_limit(self):
        for n in (100, ic.MAX_SERIES_FRAMES):
            with self.subTest(n=n):
                fake, camera = self.opened()
                series = self.capture(camera, n)
                self.assertEqual(series.acquired_frames, n)
                self.assertEqual(fake.counts["IMV_GetFrame"], n)
                self.assertEqual(series.dropped_frames_by_block_id, 0)

    def test_single_frame_series(self):
        fake, camera = self.opened()
        series = self.capture(camera, 1)
        self.assertEqual(series.acquired_frames, 1)
        self.assertEqual(fake.counts["IMV_GetFrame"], 1)

    def test_one_start_and_one_stop_per_series(self):
        fake, camera = self.opened()
        self.capture(camera, 5)
        self.capture(camera, 7)
        self.assertEqual(fake.counts["IMV_StartGrabbing"], 2)
        self.assertEqual(fake.counts["IMV_StopGrabbing"], 2)
        names = fake.names()
        first_stop = names.index("IMV_StopGrabbing")
        self.assertEqual(names[:first_stop].count("IMV_GetFrame"), 5)  # grabbing continuous within a series

    def test_one_get_and_one_release_per_frame(self):
        fake, camera = self.opened()
        self.capture(camera, 12)
        self.assertEqual(fake.counts["IMV_GetFrame"], 12)
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 12)
        self.assertEqual(len(fake.released_buffers), 12)
        self.assertEqual(fake.frames_outstanding, {})
        for name in base.FORBIDDEN_CALLS:
            self.assertNotIn(name, fake.names())

    def test_get_frame_timeout_is_bounded(self):
        fake, camera = self.opened()
        self.capture(camera, 4, frame_timeout_ms=250)
        self.assertEqual(fake.get_frame_timeouts, [250] * 4)
        self.assertTrue(all(isinstance(t, int) for t in fake.get_frame_timeouts))


# 7-11. Sample content.
class SampleContentTests(SeriesTestCase):
    def test_mean_dn_and_saturation(self):
        _, camera = self.opened()
        series = self.capture(camera, 6)
        observed = [(s.mean_dn, s.saturation_fraction) for s in series.samples]
        self.assertEqual(observed, list(PAYLOAD_STATS) * 2)

    def test_statistics_computed_from_copy_not_released_sdk_memory(self):
        fake, camera = self.opened(payloads=(bytes([255] * 8),))
        series = self.capture(camera, 3)
        for buffer in fake.released_buffers:
            self.assertEqual(bytes(buffer), b"\xee" * 8)  # the SDK reused its buffer after release
        self.assertEqual([(s.mean_dn, s.saturation_fraction) for s in series.samples], [(255.0, 1.0)] * 3)

    def test_native_block_ids_preserved_with_gaps(self):
        ids = [77, 78, 80, 81, 90]
        _, camera = self.opened(block_ids=ids)
        series = self.capture(camera, 5)
        self.assertEqual([s.native_block_id for s in series.samples], ids)
        self.assertEqual([s.frame_index for s in series.samples], list(range(5)))
        self.assertEqual(series.dropped_frames_by_block_id, 9)  # 79 and 82..89

    def test_raw_device_timestamps_preserved_unscaled(self):
        stamps = [2**64 - 5, 2**64 - 4, 2**64 - 2]
        _, camera = self.opened(timestamps=stamps)
        series = self.capture(camera, 3)
        self.assertEqual([s.device_timestamp_raw for s in series.samples], stamps)
        self.assertEqual(series.device_tick_span, 3)
        self.assertIsNone(series.synthetic_device_span_s)
        self.assertNotIn("device_tick_hz", series.analyzer_inputs())

    def test_host_timestamps_read_right_after_get_frame(self):
        clock = self.with_clock(FakeClock(start_ns=10**12, step_ns=1000))
        stamps_at_get = []
        fake, camera = self.opened(on_get_frame=lambda: stamps_at_get.append(clock.now))
        series = self.capture(camera, 4)
        self.assertEqual([s.host_t_ns for s in series.samples], stamps_at_get)

    def test_host_timestamps_monotonic_with_real_clock(self):
        _, camera = self.opened()
        before = ic.time.perf_counter_ns()
        series = self.capture(camera, 20)
        after = ic.time.perf_counter_ns()
        stamps = [s.host_t_ns for s in series.samples]
        self.assertEqual(stamps, sorted(set(stamps)))
        self.assertTrue(before <= stamps[0] and stamps[-1] <= after)

    def test_exposure_metadata(self):
        _, camera = self.opened()
        self.assertEqual(self.capture(camera, 2).exposure_us, 5000.0)
        camera.set_exposure_us(1000.0)
        self.assertEqual(self.capture(camera, 2).exposure_us, 1000.0)

    def test_unverified_exposure_refuses_without_sdk_calls(self):
        fake, camera = self.opened(apply_set=lambda value, index: 998.4)
        with self.assertRaises(ic.IRaypleCameraError):
            camera.set_exposure_us(1000.0)
        mark = len(fake.calls)
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            self.capture(camera)
        self.assertEqual(ctx.exception.status, "EXPOSURE_UNVERIFIED")
        self.assertEqual(len(fake.calls), mark)


# 12, 24, 25. Camera state, close, and capture_frame unchanged.
class StateTests(SeriesTestCase):
    def test_camera_state_preserved(self):
        fake, camera = self.opened()
        info_before = camera.get_info()
        self.capture(camera, 8)
        self.assertEqual(camera.get_info(), info_before)
        self.assertEqual(fake.writes(), [])
        self.assertEqual(fake.current, 5000.0)
        self.assertFalse(fake.is_grabbing)
        self.assertTrue(camera.is_open)

    def test_close_after_series_restores_exposure(self):
        fake, camera = self.opened()
        camera.set_exposure_us(1000.0)
        self.capture(camera, 5)
        mark = len(fake.calls)
        camera.close()
        self.assertEqual(self.calls_since(fake, mark),
                         [base.WRITE, base.READ_VALUE, "IMV_Close", "IMV_DestroyHandle"])
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assert_shut_down(fake)

    def test_close_after_failed_series_restores_exposure(self):
        fake, camera = self.opened(codes={"IMV_GetFrame": [0, 0, -119]})
        camera.set_exposure_us(1000.0)
        with self.assertRaises(ic.IRaypleCameraError):
            self.capture(camera)
        camera.close()
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assert_shut_down(fake)

    def test_capture_frame_unchanged_after_series(self):
        fake, camera = self.opened()
        self.capture(camera, 3)
        mark = len(fake.calls)
        frame = camera.capture_frame()
        self.assertEqual(self.calls_since(fake, mark), base.CAPTURE_SEQUENCE)
        self.assertEqual(fake.get_frame_timeouts[-1], ic.GET_FRAME_TIMEOUT_MS)
        self.assertEqual(frame.frame_id, 1)
        self.assertEqual(frame.pixels.shape, (2, 4))
        self.assertEqual(camera.get_info()["last_native_block_id"], 1003)

    def test_closed_camera_refuses_without_sdk_calls(self):
        fake, camera = self.make()
        with self.assertRaises(RuntimeError):
            self.capture(camera)
        self.assertEqual(fake.calls, [])

    def test_reentrant_calls_rejected(self):
        rejected = []

        def reenter():
            for call in (lambda: self.capture(camera), camera.capture_frame, camera.close):
                try:
                    call()
                except RuntimeError as exc:
                    rejected.append(str(exc))

        fake, camera = self.opened(on_get_frame=reenter)
        self.capture(camera, 2)
        self.assertEqual(len(rejected), 6)
        self.assertTrue(all("sequential use only" in message for message in rejected))
        self.assertEqual(fake.counts["IMV_StartGrabbing"], 1)


# 13, 14. Frame validation.
class ValidationTests(SeriesTestCase):
    def test_every_frame_validated_like_capture_frame(self):
        cases = (
            ({"status": 1}, "INVALID_FRAME"),
            ({"pixel_format": base.MONO10, "size_delta": 8}, "NOT_MONO8"),
            ({"pixel_format": base.MONO10}, "INVALID_FRAME"),
            ({"padding": (2, 0)}, "INVALID_LAYOUT"),
            ({"padding": (0, 1)}, "INVALID_LAYOUT"),
            ({"size_delta": 4}, "INVALID_LAYOUT"),
            ({"size_delta": -1}, "INVALID_FRAME"),
            ({"width": 0}, "INVALID_FRAME"),
        )
        for kwargs, status in cases:
            with self.subTest(kwargs=kwargs):
                fake, camera = self.opened(**kwargs)
                mark = len(fake.calls)
                self.assert_failed_cleanly(fake, camera, status)
                self.assertEqual(self.calls_since(fake, mark),
                                 ["IMV_StartGrabbing", "IMV_GetFrame", "IMV_ReleaseFrame", "IMV_StopGrabbing"])

    def test_malformed_frame_mid_series(self):
        cases = (
            ({"status": 3}, "INVALID_FRAME"),
            ({"pixelFormat": base.MONO10, "size": 16}, "NOT_MONO8"),
            ({"paddingX": 1}, "INVALID_LAYOUT"),
            ({"size": 12}, "INVALID_LAYOUT"),
            ({"width": 2, "height": 4}, "INVALID_LAYOUT"),  # same payload size, changed geometry
        )
        for overrides, status in cases:
            with self.subTest(overrides=overrides):
                fake, camera = self.opened(frame_overrides={3: overrides})
                self.assert_failed_cleanly(fake, camera, status)
                self.assertEqual(fake.counts["IMV_GetFrame"], 4)
                self.assertEqual(fake.counts["IMV_ReleaseFrame"], 4)
                self.assertEqual(fake.names().count("memmove"), 3)

    def test_null_data_pointer_rejected(self):
        fake, camera = self.opened()
        original = fake._on_IMV_GetFrame

        def null_data(frame, timeout_ms):
            original(frame, timeout_ms)
            frame.pData = None

        fake._on_IMV_GetFrame = null_data
        self.assert_failed_cleanly(fake, camera, "INVALID_FRAME")
        self.assertNotIn("memmove", fake.names())

    def test_repeated_block_id_is_contract_violation(self):
        fake, camera = self.opened(block_ids=[1] * 10)  # what single-frame captures reported
        error = self.assert_failed_cleanly(fake, camera, "SERIES_CONTRACT_VIOLATION")
        self.assertIn("native_block_id must be strictly increasing", str(error))
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 10)

    def test_non_increasing_device_timestamp_is_contract_violation(self):
        for stamps in ([0] * 10, [5, 6, 7, 7, 8, 9, 10, 11, 12, 13], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]):
            with self.subTest(stamps=stamps):
                fake, camera = self.opened(timestamps=stamps)
                error = self.assert_failed_cleanly(fake, camera, "SERIES_CONTRACT_VIOLATION")
                self.assertIn("device_timestamp_raw", str(error))
                self.assertIsInstance(error.__cause__, ValueError)


# 15-17. Limits.
class LimitTests(SeriesTestCase):
    def test_argument_limits_rejected_without_sdk_calls(self):
        fake, camera = self.opened()
        mark = len(fake.calls)
        bad = (
            (dict(n=0), ValueError), (dict(n=ic.MAX_SERIES_FRAMES + 1), ValueError), (dict(n=-1), ValueError),
            (dict(n=True), TypeError), (dict(n=10.0), TypeError), (dict(n="10"), TypeError),
            (dict(frame_timeout_ms=0), ValueError),
            (dict(frame_timeout_ms=ic.MAX_SERIES_FRAME_TIMEOUT_MS + 1), ValueError),
            (dict(frame_timeout_ms=100.0), TypeError), (dict(frame_timeout_ms=None), TypeError),
            (dict(max_duration_s=0), ValueError), (dict(max_duration_s=-1.0), ValueError),
            (dict(max_duration_s=5.001), ValueError), (dict(max_duration_s=float("nan")), ValueError),
            (dict(max_duration_s=float("inf")), ValueError), (dict(max_duration_s=True), TypeError),
            (dict(max_duration_s="1"), TypeError),
        )
        for kwargs, error in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                self.capture(camera, **kwargs)
        self.assertEqual(len(fake.calls), mark)

    def test_limits_are_keyword_only_and_required(self):
        _, camera = self.opened()
        with self.assertRaises(TypeError):
            camera.capture_series(5, 1000, 1.0)
        with self.assertRaises(TypeError):
            camera.capture_series(5)

    def test_boundary_values_accepted(self):
        _, camera = self.opened()
        series = camera.capture_series(np.int64(3), frame_timeout_ms=ic.MAX_SERIES_FRAME_TIMEOUT_MS,
                                       max_duration_s=ic.MAX_SERIES_DURATION_S)
        self.assertEqual(series.requested_frames, 3)

    def test_duration_limit_fails_before_next_frame(self):
        self.with_clock(FakeClock(step_ns=100_000_000))  # every clock read advances 100 ms
        fake, camera = self.opened()
        error = self.assert_failed_cleanly(fake, camera, "SERIES_DEADLINE_EXCEEDED",
                                           lambda: self.capture(camera, 10, max_duration_s=0.5))
        served = fake.counts["IMV_GetFrame"]
        self.assertLess(served, 10)
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], served)
        self.assertIn(f"{served} frame(s) collected", str(error))

    def test_frame_arriving_after_deadline_is_released_and_rejected(self):
        clock = FakeClock(step_ns=1000)
        self.with_clock(clock)

        def stall():
            if clock.now > 10**12 + 5000:
                clock.now += 2_000_000_000

        fake, camera = self.opened(on_get_frame=stall)
        error = self.assert_failed_cleanly(fake, camera, "SERIES_DEADLINE_EXCEEDED",
                                           lambda: self.capture(camera, 10, max_duration_s=1.0))
        self.assertIn("arrived after the series deadline", str(error))
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], fake.counts["IMV_GetFrame"])

    def test_get_frame_timeout_limited_by_remaining_time(self):
        self.with_clock(FakeClock(step_ns=100_000_000))
        fake, camera = self.opened()
        with self.assertRaises(ic.IRaypleCameraError):
            self.capture(camera, 10, frame_timeout_ms=3000, max_duration_s=1.0)
        timeouts = fake.get_frame_timeouts
        self.assertEqual(timeouts[0], 900)  # 1 s deadline less one 100 ms clock step
        self.assertEqual(timeouts, sorted(timeouts, reverse=True))
        self.assertTrue(all(1 <= t <= 900 for t in timeouts))

    def test_frame_timeout(self):
        fake, camera = self.opened(codes={"IMV_GetFrame": [0, 0, 0, -119]})
        error = self.assert_failed_cleanly(fake, camera, "GET_FRAME_FAILED")
        self.assertEqual(error.sdk_code_name, "IMV_TIMEOUT")
        self.assertIn("frame 4 of 10", str(error))
        self.assertIn("3 frame(s) collected", str(error))
        self.assertEqual(fake.counts["IMV_GetFrame"], 4)  # never repeated
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 3)


# 18-22. SDK failures and no false completion.
class FailureTests(SeriesTestCase):
    def test_get_frame_failure_code(self):
        fake, camera = self.opened(codes={"IMV_GetFrame": [0, -101]})
        mark = len(fake.calls)
        error = self.assert_failed_cleanly(fake, camera, "GET_FRAME_FAILED")
        self.assertEqual(error.sdk_code, -101)
        self.assertEqual(self.calls_since(fake, mark),
                         ["IMV_StartGrabbing", "IMV_GetFrame", "memmove", "IMV_ReleaseFrame", "IMV_GetFrame",
                          "IMV_StopGrabbing"])

    def test_get_frame_exception_still_stops(self):
        for exc in (OSError("access violation"), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__):
                fake, camera = self.opened(raises={"IMV_GetFrame": [None, exc]})
                with self.assertRaises(type(exc)):
                    self.capture(camera)
                self.assertEqual(fake.counts["IMV_StopGrabbing"], 1)
                self.assertEqual(fake.counts["IMV_ReleaseFrame"], 1)
                self.assertFalse(fake.is_grabbing)

    def test_start_grabbing_failure(self):
        fake, camera = self.opened(codes={"IMV_StartGrabbing": [-101]})
        mark = len(fake.calls)
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            self.capture(camera)
        self.assertEqual(ctx.exception.status, "START_GRABBING_FAILED")
        self.assertEqual(self.calls_since(fake, mark), ["IMV_StartGrabbing"])
        self.assertEqual(self.capture(camera, 2).acquired_frames, 2)  # camera remains usable

    def test_release_failure_stops_series(self):
        fake, camera = self.opened(codes={"IMV_ReleaseFrame": [0, 0, -101]})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            self.capture(camera)
        self.assertEqual(ctx.exception.status, "FRAME_LIFECYCLE_FAILED")
        self.assertIn("frame 3 of 10", str(ctx.exception))
        self.assertIn("IMV_ReleaseFrame FAILED with code -101", str(ctx.exception))
        self.assertEqual(fake.counts["IMV_GetFrame"], 3)
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 3)  # released exactly once, never repeated
        self.assertEqual(fake.counts["IMV_StopGrabbing"], 1)
        self.assertFalse(fake.is_grabbing)
        fake.frames_outstanding.clear()  # the fake kept the frame because its release "failed"
        camera.close()
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 3)

    def test_release_exception_stops_series(self):
        fake, camera = self.opened(raises={"IMV_ReleaseFrame": [None, OSError("boom")]})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            self.capture(camera)
        self.assertEqual(ctx.exception.status, "FRAME_LIFECYCLE_FAILED")
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 2)
        self.assertEqual(fake.counts["IMV_StopGrabbing"], 1)
        fake.frames_outstanding.clear()

    def test_stop_failure_discards_series_and_blocks_use(self):
        fake, camera = self.opened(codes={"IMV_StopGrabbing": [-101]})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            self.capture(camera)
        self.assertEqual(ctx.exception.status, "FRAME_LIFECYCLE_FAILED")
        self.assertIn("series discarded", str(ctx.exception))
        self.assertEqual(fake.counts["IMV_GetFrame"], 10)
        mark = len(fake.calls)
        for call in (lambda: self.capture(camera), camera.capture_frame, lambda: camera.set_exposure_us(1000.0)):
            with self.assertRaises(ic.IRaypleCameraError) as blocked:
                call()
            self.assertEqual(blocked.exception.status, "ACQUISITION_ACTIVE")
        self.assertEqual(len(fake.calls), mark)
        camera.close()
        self.assertEqual(fake.names()[mark:], ["IMV_StopGrabbing", "IMV_Close", "IMV_DestroyHandle"])
        self.assert_shut_down(fake)

    def test_stop_failure_after_frame_failure_is_noted(self):
        fake, camera = self.opened(codes={"IMV_GetFrame": [0, -119], "IMV_StopGrabbing": [-101]})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            self.capture(camera)
        self.assertEqual(ctx.exception.status, "GET_FRAME_FAILED")
        self.assertIn("IMV_StopGrabbing FAILED with code -101", " ".join(ctx.exception.__notes__))
        with self.assertRaises(ic.IRaypleCameraError) as blocked:
            self.capture(camera)
        self.assertEqual(blocked.exception.status, "ACQUISITION_ACTIVE")
        camera.close()
        self.assert_shut_down(fake)

    def test_unresolved_cleanup_still_refuses_reopen(self):
        fake, camera = self.opened(codes={"IMV_StopGrabbing": [-101, -101]})
        with self.assertRaises(ic.IRaypleCameraError):
            self.capture(camera)
        with self.assertRaises(ic.IRaypleCloseError):
            camera.close()
        with self.assertRaises(ic.IRaypleCameraError) as reopen:
            camera.open()
        self.assertEqual(reopen.exception.status, "UNRESOLVED_CLEANUP_FAILURE")

    def test_statistics_failure_still_releases_and_stops(self):
        fake, camera = self.opened()
        with mock.patch.object(ic.np, "count_nonzero", side_effect=MemoryError("stats failed")):
            with self.assertRaises(MemoryError):
                self.capture(camera)
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 1)
        self.assertEqual(fake.counts["IMV_StopGrabbing"], 1)
        self.assertEqual(fake.frames_outstanding, {})

    def test_copy_failure_still_releases_and_stops(self):
        fake, camera = self.opened()
        with mock.patch.object(ic.ctypes, "memmove", side_effect=MemoryError("copy failed")):
            with self.assertRaises(MemoryError):
                self.capture(camera)
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 1)
        self.assertEqual(fake.counts["IMV_StopGrabbing"], 1)

    def test_no_series_returned_on_any_failure(self):
        scenarios = (
            {"codes": {"IMV_GetFrame": [0, 0, 0, 0, -119]}},
            {"codes": {"IMV_ReleaseFrame": [0, -101]}},
            {"codes": {"IMV_StopGrabbing": [-101]}},
            {"frame_overrides": {9: {"status": 1}}},
            {"block_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 13]},
        )
        for kwargs in scenarios:
            with self.subTest(kwargs=kwargs):
                fake, camera = self.opened(**kwargs)
                with mock.patch.object(ic, "FrameSeries", wraps=FrameSeries) as constructor:
                    with self.assertRaises(ic.IRaypleCameraError):
                        self.capture(camera)
                built = [call.kwargs for call in constructor.call_args_list]
                self.assertTrue(all(kw["status"] == "completed" for kw in built))
                self.assertTrue(all(kw["acquired_frames"] == 10 for kw in built))
                fake.frames_outstanding.clear()


# 23. No persistent raw frames.
class MemoryTests(SeriesTestCase):
    def test_one_reused_buffer_and_no_retained_arrays(self):
        destinations = []
        real_memmove = base.ctypes.memmove

        def record(dst, src, size):
            destinations.append(dst)
            return real_memmove(dst, src, size)

        fake, camera = self.opened()
        with mock.patch.object(ic.ctypes, "memmove", side_effect=record):
            series = self.capture(camera, 25)
        self.assertEqual(len(destinations), 25)
        self.assertEqual(len(set(destinations)), 1)
        for value in vars(camera).values():
            self.assertNotIsInstance(value, np.ndarray)
        for value in vars(series).values():
            self.assertNotIsInstance(value, np.ndarray)
        for sample in series.samples:
            for field in dataclasses.fields(sample):
                self.assertIsInstance(getattr(sample, field.name), (int, float))

    def test_source_respects_scanner_tokens(self):
        source = (HW3_DIR / "irayple_camera.py").read_text(encoding="utf-8")
        for token in ("while ", "retry", "Thread(", "StartGrabbingEx", "AttachGrabbing", "CloneFrame"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
