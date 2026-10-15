"""Deterministic, offline FakeCamera implementing CameraBackend.

This is an integration test double, NOT a validated optical or sensor model.
It uses no vendor SDK, no hardware, no network and no randomness.

Synthetic image model (8x8, Mono8, bit_depth 8):
    fixed_pattern[r, c] = 30 + 2 * (8 * r + c)          -> 30 .. 156 DN
    signal              = fixed_pattern * (exposure_us / 1000)
    pixels              = clip(rint(signal), 0, 255).astype(uint8)

Quantization rule: numpy.rint, i.e. round to the nearest integer with exact
halves rounded to the nearest even integer (IEEE 754 round-half-to-even),
computed in float64, then clipped to [0, 255]. Consequences:
    100 us   -> 3 .. 16 DN      (dark)
    1000 us  -> 30 .. 156 DN    (default, no saturation)
    10000 us -> 255 everywhere  (fully saturated)
"""

import math
from numbers import Real

import numpy as np

from camera_contract import SOURCE_SYNTHETIC, CameraFrame

WIDTH = 8
HEIGHT = 8
BIT_DEPTH = 8
MIN_EXPOSURE_US = 100.0
MAX_EXPOSURE_US = 10000.0
DEFAULT_EXPOSURE_US = 1000.0
REFERENCE_EXPOSURE_US = 1000.0

_FIXED_PATTERN = (30.0 + 2.0 * np.arange(HEIGHT * WIDTH, dtype=np.float64)).reshape(
    HEIGHT, WIDTH
)
_FIXED_PATTERN.setflags(write=False)


class FakeCamera:
    """Synthetic camera. Every frame is labelled source="synthetic"."""

    def __init__(self) -> None:
        self._is_open = False
        self._exposure_us = DEFAULT_EXPOSURE_US
        self._last_frame_id = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def get_info(self) -> dict:
        return {
            "backend": "FakeCamera",
            "source": SOURCE_SYNTHETIC,
            "synthetic": True,
            "description": (
                "Deterministic synthetic test double. Not a real camera and not "
                "a validated optical model."
            ),
            "width": WIDTH,
            "height": HEIGHT,
            "bit_depth": BIT_DEPTH,
            "pixel_format": "Mono8",
            "exposure_limits_us": [MIN_EXPOSURE_US, MAX_EXPOSURE_US],
            "exposure_us": self._exposure_us,
            "is_open": self._is_open,
        }

    def get_exposure_limits_us(self) -> tuple[float, float]:
        return (MIN_EXPOSURE_US, MAX_EXPOSURE_US)

    def set_exposure_us(self, value: float) -> float:
        self._require_open("set_exposure_us")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"exposure_us must be a real number, got {type(value).__name__}"
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"exposure_us must be finite, got {value}")
        if not MIN_EXPOSURE_US <= value <= MAX_EXPOSURE_US:
            raise ValueError(
                f"exposure_us {value} outside [{MIN_EXPOSURE_US}, {MAX_EXPOSURE_US}] us"
            )
        self._exposure_us = value
        return value

    def capture_frame(self) -> CameraFrame:
        self._require_open("capture_frame")
        signal = _FIXED_PATTERN * (self._exposure_us / REFERENCE_EXPOSURE_US)
        # np.clip/astype allocate a new array, so the frame owns its pixels.
        pixels = np.clip(np.rint(signal), 0, 255).astype(np.uint8)
        self._last_frame_id += 1
        return CameraFrame(
            pixels=pixels,
            bit_depth=BIT_DEPTH,
            exposure_us=self._exposure_us,
            source=SOURCE_SYNTHETIC,
            frame_id=self._last_frame_id,
        )

    def _require_open(self, operation: str) -> None:
        if not self._is_open:
            raise RuntimeError(f"{operation} requires an open camera; call open() first")
