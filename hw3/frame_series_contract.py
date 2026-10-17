"""Hardware-independent contract for a timestamped series of per-frame brightness samples.

A FrameSeries is what a frame-series acquisition backend (synthetic now,
hardware later) hands to the offline temporal analyzer (temporal_analysis.py).
It carries per-frame scalars only; raw pixel arrays are never stored.

This module is pure validation and bookkeeping: it never touches a camera, a
vendor SDK, MCP, the network or an LLM.

Invariants (enforced at construction; violations raise TypeError/ValueError):
    FrameSample
        frame_index           int >= 0, position in the delivered series
        native_block_id       int >= 0 or None; the backend's native frame id.
                              A gap between consecutive ids is a dropped frame.
        device_timestamp_raw  int >= 0 or None; raw device clock value of
                              UNKNOWN physical unit. It is never converted to
                              seconds here.
        host_t_ns             int >= 0, monotonic host arrival time, ns
        mean_dn               finite float >= 0, mean digital number
        saturation_fraction   finite float in [0, 1]
    FrameSeries
        samples               tuple of FrameSample (lists are converted)
        frame_index           equals 0, 1, ..., acquired_frames - 1
        host_t_ns             strictly increasing
        native_block_id,      each either present in every sample or in none;
        device_timestamp_raw  when present, strictly increasing
        mean_dn               <= 2**bit_depth - 1
        acquired_frames       == len(samples) <= requested_frames
        status                "completed" iff acquired_frames == requested_frames;
                              "partial" (>= 1 frame) or "failed" otherwise. A
                              short series can never be labelled completed.
        stop_reason           None for "completed", a non-empty str otherwise
        source                "hardware" or "synthetic"
        synthetic_device_tick_hz
                              declared tick rate of a SYNTHETIC device clock, or
                              None. It is rejected for hardware series: the real
                              device tick frequency is not established by this
                              contract.
"""

import math
from dataclasses import dataclass

import numpy as np

SOURCE_HARDWARE = "hardware"
SOURCE_SYNTHETIC = "synthetic"
SOURCES = (SOURCE_HARDWARE, SOURCE_SYNTHETIC)

STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUSES = (STATUS_COMPLETED, STATUS_PARTIAL, STATUS_FAILED)

MIN_BIT_DEPTH = 8
MAX_BIT_DEPTH = 16
MAX_INT64 = 2**63 - 1
MAX_UINT64 = 2**64 - 1


@dataclass(frozen=True)
class FrameSample:
    """Scalar summary of one delivered frame. See the module docstring for units and ranges."""

    frame_index: int
    native_block_id: int | None
    device_timestamp_raw: int | None
    host_t_ns: int
    mean_dn: float
    saturation_fraction: float

    def __post_init__(self):
        _set(self, "frame_index", _int("frame_index", self.frame_index, MAX_INT64))
        _set(self, "native_block_id",
             _optional_int("native_block_id", self.native_block_id, MAX_UINT64))
        _set(self, "device_timestamp_raw",
             _optional_int("device_timestamp_raw", self.device_timestamp_raw, MAX_UINT64))
        _set(self, "host_t_ns", _int("host_t_ns", self.host_t_ns, MAX_INT64))
        mean_dn = _real("mean_dn", self.mean_dn)
        if mean_dn < 0:
            raise ValueError(f"mean_dn must be >= 0, got {mean_dn!r}")
        _set(self, "mean_dn", mean_dn)
        fraction = _real("saturation_fraction", self.saturation_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"saturation_fraction must be in [0, 1], got {fraction!r}")
        _set(self, "saturation_fraction", fraction)


@dataclass(frozen=True)
class FrameSeries:
    """Validated, immutable frame series plus its acquisition metadata."""

    samples: tuple
    requested_frames: int
    acquired_frames: int
    exposure_us: float
    width: int
    height: int
    bit_depth: int
    source: str
    status: str
    stop_reason: str | None = None
    synthetic_device_tick_hz: float | None = None

    def __post_init__(self):
        if isinstance(self.samples, (str, bytes)) or not isinstance(self.samples, (tuple, list)):
            raise TypeError(f"samples must be a tuple or list, got {type(self.samples).__name__}")
        samples = tuple(self.samples)
        for position, sample in enumerate(samples):
            if not isinstance(sample, FrameSample):
                raise TypeError(f"samples[{position}] is {type(sample).__name__}, not FrameSample")
        _set(self, "samples", samples)

        requested = _int("requested_frames", self.requested_frames, MAX_INT64)
        acquired = _int("acquired_frames", self.acquired_frames, MAX_INT64)
        if requested < 1:
            raise ValueError(f"requested_frames must be >= 1, got {requested}")
        if acquired != len(samples):
            raise ValueError(f"acquired_frames is {acquired} but {len(samples)} samples were given")
        if acquired > requested:
            raise ValueError(f"acquired_frames {acquired} exceeds requested_frames {requested}")
        _set(self, "requested_frames", requested)
        _set(self, "acquired_frames", acquired)

        exposure = _real("exposure_us", self.exposure_us)
        if exposure <= 0:
            raise ValueError(f"exposure_us must be > 0, got {exposure!r}")
        _set(self, "exposure_us", exposure)
        for name in ("width", "height"):
            value = _int(name, getattr(self, name), MAX_INT64)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
            _set(self, name, value)
        bit_depth = _int("bit_depth", self.bit_depth, MAX_INT64)
        if not MIN_BIT_DEPTH <= bit_depth <= MAX_BIT_DEPTH:
            raise ValueError(f"bit_depth must be in [{MIN_BIT_DEPTH}, {MAX_BIT_DEPTH}], got {bit_depth}")
        _set(self, "bit_depth", bit_depth)

        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        self._validate_status(requested, acquired)
        self._validate_tick_hz()
        self._validate_samples(samples, bit_depth)

    # -- validation ----------------------------------------------------------

    def _validate_status(self, requested, acquired):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.status == STATUS_COMPLETED:
            if acquired != requested:
                raise ValueError(f"status 'completed' requires {requested} frames, got {acquired}")
            if self.stop_reason is not None:
                raise ValueError("a completed series must not carry a stop_reason")
            return
        if acquired == requested:
            raise ValueError(f"all {requested} requested frames were acquired; status must be 'completed'")
        if self.status == STATUS_PARTIAL and acquired < 1:
            raise ValueError("status 'partial' requires at least one frame; use 'failed'")
        if not isinstance(self.stop_reason, str) or not self.stop_reason.strip():
            raise ValueError(f"status {self.status!r} requires a non-empty stop_reason")

    def _validate_tick_hz(self):
        if self.synthetic_device_tick_hz is None:
            return
        if self.source != SOURCE_SYNTHETIC:
            raise ValueError("synthetic_device_tick_hz is only allowed for synthetic series; the "
                             "hardware device tick frequency is not established by this contract")
        tick_hz = _real("synthetic_device_tick_hz", self.synthetic_device_tick_hz)
        if tick_hz <= 0:
            raise ValueError(f"synthetic_device_tick_hz must be > 0, got {tick_hz!r}")
        if self.samples and self.samples[0].device_timestamp_raw is None:
            raise ValueError("synthetic_device_tick_hz requires device_timestamp_raw values")
        _set(self, "synthetic_device_tick_hz", tick_hz)

    @staticmethod
    def _validate_samples(samples, bit_depth):
        max_dn = 2**bit_depth - 1
        for position, sample in enumerate(samples):
            if sample.frame_index != position:
                raise ValueError(f"samples[{position}].frame_index is {sample.frame_index}, expected {position}")
            if sample.mean_dn > max_dn:
                raise ValueError(f"samples[{position}].mean_dn {sample.mean_dn!r} exceeds "
                                 f"{max_dn} for bit_depth {bit_depth}")
        for name in ("native_block_id", "device_timestamp_raw"):
            present = {getattr(s, name) is not None for s in samples}
            if len(present) > 1:
                raise ValueError(f"{name} must be given for every sample or for none")
        for name in ("host_t_ns", "native_block_id", "device_timestamp_raw"):
            values = [getattr(s, name) for s in samples]
            if values and values[0] is None:
                continue
            for position in range(1, len(values)):
                if values[position] <= values[position - 1]:
                    raise ValueError(f"{name} must be strictly increasing; samples[{position}] "
                                     f"has {values[position]} after {values[position - 1]}")

    # -- derived metadata ------------------------------------------------------

    @property
    def is_complete(self):
        return self.status == STATUS_COMPLETED

    @property
    def has_block_ids(self):
        return bool(self.samples) and self.samples[0].native_block_id is not None

    @property
    def has_device_timestamps(self):
        return bool(self.samples) and self.samples[0].device_timestamp_raw is not None

    @property
    def host_span_ns(self):
        """Host arrival time from the first to the last delivered frame (n - 1 intervals), or None.

        This is arrival-to-arrival time, including transport latency; it is not
        the time covered by exposures.
        """
        if len(self.samples) < 2:
            return None
        return self.samples[-1].host_t_ns - self.samples[0].host_t_ns

    @property
    def host_span_s(self):
        span = self.host_span_ns
        return None if span is None else span * 1e-9

    @property
    def device_tick_span(self):
        """Raw device clock difference, first to last frame, in ticks of unknown unit, or None."""
        if len(self.samples) < 2 or not self.has_device_timestamps:
            return None
        return self.samples[-1].device_timestamp_raw - self.samples[0].device_timestamp_raw

    @property
    def synthetic_device_span_s(self):
        """device_tick_span in seconds, only for a synthetic clock with a declared tick rate."""
        span = self.device_tick_span
        if span is None or self.synthetic_device_tick_hz is None:
            return None
        return span / self.synthetic_device_tick_hz

    @property
    def dropped_frames_by_block_id(self):
        """Frames missing between consecutive native block ids, or None without block ids."""
        if not self.has_block_ids:
            return None
        if len(self.samples) < 2:
            return 0
        return (self.samples[-1].native_block_id - self.samples[0].native_block_id
                - (len(self.samples) - 1))

    def analyzer_inputs(self, *, use_synthetic_tick_hz=False):
        """Keyword arguments for temporal_analysis.analyze_temporal_series (new arrays each call).

        Device timestamps are passed as unscaled ticks. A tick rate is passed
        only when use_synthetic_tick_hz is True, which requires a synthetic
        series with a declared synthetic_device_tick_hz.
        """
        if use_synthetic_tick_hz and self.synthetic_device_tick_hz is None:
            raise ValueError("use_synthetic_tick_hz requires a synthetic series with "
                             "synthetic_device_tick_hz")
        inputs = {
            "host_t_ns": np.array([s.host_t_ns for s in self.samples], dtype=np.int64),
            "mean_dn": np.array([s.mean_dn for s in self.samples], dtype=np.float64),
        }
        if self.has_block_ids:
            inputs["block_ids"] = np.array([s.native_block_id for s in self.samples], dtype=np.uint64)
        if self.has_device_timestamps:
            inputs["device_ts"] = np.array([s.device_timestamp_raw for s in self.samples],
                                           dtype=np.uint64)
        if use_synthetic_tick_hz:
            inputs["device_tick_hz"] = self.synthetic_device_tick_hz
        return inputs


# -- helpers -------------------------------------------------------------------

def _set(instance, name, value):
    object.__setattr__(instance, name, value)


def _int(name, value, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")
    return value


def _optional_int(name, value, maximum):
    return None if value is None else _int(name, value, maximum)


def _real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value
