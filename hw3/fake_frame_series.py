"""Deterministic, seeded synthetic frame-series source producing FrameSeries.

This is a test double for the temporal-analysis pipeline, NOT a validated
optical, sensor or camera-timing model. It uses no vendor SDK, no hardware, no
network and no LLM. Every series it returns is labelled source="synthetic".

Model (frame k is the k-th EXPOSED frame, k = 0, 1, 2, ...):
    t_k                 = k / fps                         ideal exposure time, s
    brightness(t)       = base_dn                                     "steady"
                        = base_dn + amplitude_dn * sin(2 pi f t + phase_rad)  "sine"
                        = base_dn + amplitude_dn * (+1 if frac(f t + phase_rad / 2 pi) < duty
                                                   else -1)                    "square"
    mean_dn             = clip(brightness(t_k) + N(0, noise_dn), 0, 2**bit_depth - 1)
    saturation_fraction = 1.0 if the unclipped value reaches 2**bit_depth - 1, else 0.0
                          (a scalar stand-in; no pixel distribution is modelled)
    native_block_id     = block_id_start + k
    host_t_ns           = host_start_ns + round(t_k * 1e9) + U(-host_jitter_ns, +host_jitter_ns)
    device_timestamp_raw= device_tick_start + round(t_k * synthetic_device_tick_hz), or None
                          when no synthetic tick rate is declared (the default)

Brightness is sampled instantaneously at t_k; integration over the exposure
window is not modelled. Host jitter perturbs arrival times only, never the
sampled brightness or the synthetic device clock.

Exposed frames whose index k is in drop_frames are never delivered, leaving a
gap in native_block_id. Exposure continues until n_frames frames are
delivered, unless stop_after limits delivery, in which case the series is
"partial" (or "failed" with zero frames) and says why.

The synthetic device clock rate is an explicit, arbitrary test parameter. It
does not establish, approximate or imply the tick frequency of any real camera.
"""

import math

import numpy as np

from frame_series_contract import (
    MAX_BIT_DEPTH,
    MIN_BIT_DEPTH,
    SOURCE_SYNTHETIC,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    FrameSample,
    FrameSeries,
)

WAVEFORM_STEADY = "steady"
WAVEFORM_SINE = "sine"
WAVEFORM_SQUARE = "square"
WAVEFORMS = (WAVEFORM_STEADY, WAVEFORM_SINE, WAVEFORM_SQUARE)

DEFAULT_HOST_START_NS = 10**12
DEFAULT_BLOCK_ID_START = 1
DEFAULT_DEVICE_TICK_START = 2**40


class FakeFrameSeries:
    """Synthetic frame-series source. Equal configurations return equal series."""

    def __init__(self, *, n_frames, fps, waveform=WAVEFORM_STEADY, flicker_hz=None,
                 base_dn=100.0, amplitude_dn=0.0, phase_rad=0.0, duty=0.5, noise_dn=0.0,
                 seed=0, exposure_us=1000.0, width=640, height=480, bit_depth=8,
                 drop_frames=(), host_jitter_ns=0, synthetic_device_tick_hz=None,
                 stop_after=None, host_start_ns=DEFAULT_HOST_START_NS,
                 block_id_start=DEFAULT_BLOCK_ID_START,
                 device_tick_start=DEFAULT_DEVICE_TICK_START):
        self.n_frames = _count("n_frames", n_frames, minimum=1)
        self.fps = _positive("fps", fps)
        if waveform not in WAVEFORMS:
            raise ValueError(f"waveform must be one of {WAVEFORMS}, got {waveform!r}")
        self.waveform = waveform
        if waveform == WAVEFORM_STEADY:
            if flicker_hz is not None:
                raise ValueError("a steady waveform takes no flicker_hz")
            self.flicker_hz = None
        else:
            self.flicker_hz = _positive("flicker_hz", flicker_hz)
        self.base_dn = _real("base_dn", base_dn)
        self.amplitude_dn = _real("amplitude_dn", amplitude_dn)
        if self.amplitude_dn < 0:
            raise ValueError(f"amplitude_dn must be >= 0, got {self.amplitude_dn!r}")
        self.phase_rad = _real("phase_rad", phase_rad)
        self.duty = _real("duty", duty)
        if not 0.0 < self.duty < 1.0:
            raise ValueError(f"duty must be in (0, 1), got {self.duty!r}")
        self.noise_dn = _real("noise_dn", noise_dn)
        if self.noise_dn < 0:
            raise ValueError(f"noise_dn must be >= 0, got {self.noise_dn!r}")
        self.seed = _count("seed", seed, minimum=0)
        self.exposure_us = _positive("exposure_us", exposure_us)
        if self.exposure_us > 1e6 / self.fps:
            raise ValueError(f"exposure_us {self.exposure_us!r} exceeds the frame period "
                             f"{1e6 / self.fps!r} us at {self.fps!r} fps")
        self.width = _count("width", width, minimum=1)
        self.height = _count("height", height, minimum=1)
        self.bit_depth = _count("bit_depth", bit_depth, minimum=MIN_BIT_DEPTH)
        if self.bit_depth > MAX_BIT_DEPTH:
            raise ValueError(f"bit_depth must be <= {MAX_BIT_DEPTH}, got {self.bit_depth}")
        self.drop_frames = frozenset(_count("drop_frames entry", k, minimum=0) for k in drop_frames)
        self.host_jitter_ns = _count("host_jitter_ns", host_jitter_ns, minimum=0)
        if self.host_jitter_ns >= 0.5e9 / self.fps:
            raise ValueError("host_jitter_ns must be below half the frame period so host "
                             "timestamps stay strictly increasing")
        self.synthetic_device_tick_hz = (None if synthetic_device_tick_hz is None
                                         else _positive("synthetic_device_tick_hz",
                                                        synthetic_device_tick_hz))
        self.stop_after = None if stop_after is None else _count("stop_after", stop_after, minimum=0)
        self.host_start_ns = _count("host_start_ns", host_start_ns, minimum=0)
        self.block_id_start = _count("block_id_start", block_id_start, minimum=0)
        self.device_tick_start = _count("device_tick_start", device_tick_start, minimum=0)

    def acquire(self):
        """Generate the configured series. Deterministic for a given configuration and seed."""
        delivered = self.n_frames
        if self.stop_after is not None:
            delivered = min(delivered, self.stop_after)
        exposed = []
        k = 0
        while len(exposed) < delivered:
            if k not in self.drop_frames:
                exposed.append(k)
            k += 1
        exposed = np.array(exposed, dtype=np.int64)
        t_s = exposed / self.fps

        rng = np.random.default_rng(self.seed)
        max_dn = float(2**self.bit_depth - 1)
        raw = self._brightness(t_s) + rng.normal(0.0, self.noise_dn, len(t_s))
        jitter = rng.integers(-self.host_jitter_ns, self.host_jitter_ns, len(t_s), endpoint=True)
        host = self.host_start_ns + np.round(t_s * 1e9).astype(np.int64) + jitter

        samples = []
        for i, k in enumerate(exposed):
            ticks = None
            if self.synthetic_device_tick_hz is not None:
                ticks = self.device_tick_start + int(round(int(k) * self.synthetic_device_tick_hz
                                                           / self.fps))
            samples.append(FrameSample(
                frame_index=i,
                native_block_id=self.block_id_start + int(k),
                device_timestamp_raw=ticks,
                host_t_ns=int(host[i]),
                mean_dn=float(min(max(raw[i], 0.0), max_dn)),
                saturation_fraction=1.0 if raw[i] >= max_dn else 0.0,
            ))

        status, reason = STATUS_COMPLETED, None
        if delivered < self.n_frames:
            status = STATUS_PARTIAL if delivered else STATUS_FAILED
            reason = (f"synthetic acquisition stopped after {delivered} of "
                      f"{self.n_frames} requested frames (stop_after)")
        return FrameSeries(
            samples=tuple(samples),
            requested_frames=self.n_frames,
            acquired_frames=len(samples),
            exposure_us=self.exposure_us,
            width=self.width,
            height=self.height,
            bit_depth=self.bit_depth,
            source=SOURCE_SYNTHETIC,
            status=status,
            stop_reason=reason,
            synthetic_device_tick_hz=self.synthetic_device_tick_hz,
        )

    def _brightness(self, t_s):
        if self.waveform == WAVEFORM_STEADY:
            return np.full(len(t_s), self.base_dn)
        cycles = self.flicker_hz * t_s + self.phase_rad / (2 * math.pi)
        if self.waveform == WAVEFORM_SINE:
            shape = np.sin(2 * math.pi * cycles)
        else:
            shape = np.where(np.mod(cycles, 1.0) < self.duty, 1.0, -1.0)
        return self.base_dn + self.amplitude_dn * shape


def _count(name, value, *, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _positive(name, value):
    value = _real(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value
