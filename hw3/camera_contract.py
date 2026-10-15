"""Hardware-independent camera backend contract.

A CameraBackend delivers decoded monochrome frames to the rest of LUMEN
(measurement engine, future MCP server, agents) without exposing any vendor
SDK types.

Invariants every backend must uphold:
    - CameraFrame.pixels is a decoded, unpacked, two-dimensional
      (height, width) monochrome image.
    - pixels uses uint8 for 8-bit images and uint16 for supported higher bit
      depths (10/12/14/16-bit samples stored unscaled in uint16).
    - All exposure values are in microseconds (us).
    - CameraFrame owns its pixel data. It must never reference memory that a
      vendor SDK may invalidate or reuse after the frame buffer is released.
    - A synthetic frame has source="synthetic"; a real hardware frame has
      source="hardware". A backend must never label synthetic data as
      hardware data.
    - Acquisition (capture_frame) and parameter changes (set_exposure_us)
      require an open backend; calling them on a closed backend raises
      RuntimeError.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

SOURCE_SYNTHETIC = "synthetic"
SOURCE_HARDWARE = "hardware"


@dataclass(frozen=True)
class CameraFrame:
    """One acquired frame plus the acquisition metadata needed to analyze it.

    pixels       decoded 2-D monochrome array owned by this frame
    bit_depth    significant bits per pixel sample
    exposure_us  exposure actually applied for this frame, microseconds
    source       "synthetic" or "hardware"
    frame_id     per-backend acquisition counter, strictly increasing
    """

    pixels: np.ndarray
    bit_depth: int
    exposure_us: float
    source: str
    frame_id: int


@runtime_checkable
class CameraBackend(Protocol):
    def open(self) -> None:
        """Open the backend so it can acquire and accept parameter changes."""

    def close(self) -> None:
        """Release the backend. Closing an already closed backend is safe."""

    def get_info(self) -> dict:
        """Describe the backend, including whether it is synthetic."""

    def get_exposure_limits_us(self) -> tuple[float, float]:
        """Return the inclusive (minimum, maximum) exposure in microseconds."""

    def set_exposure_us(self, value: float) -> float:
        """Apply an exposure in microseconds and return the applied value."""

    def capture_frame(self) -> CameraFrame:
        """Acquire one frame that owns its pixel data."""
