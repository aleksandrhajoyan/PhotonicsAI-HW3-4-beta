"""MCP server exposing a synthetic CameraBackend through five MCP tools.

Built on the official MCP Python SDK v2 (mcp.server.MCPServer). This
iteration is strictly fake-only: the factory refuses any backend that does not
describe itself as synthetic, and the standalone entry point always uses the
offline FakeCamera. No vendor SDK, hardware, network or LLM is involved.

Tools:
    camera_info          backend description and exposure limits (works closed)
    camera_open          open the backend (idempotent for FakeCamera)
    camera_set_exposure  apply an exposure in microseconds, no clipping
    camera_measure       capture one frame and analyze it with analyze_frame
    camera_close         close the backend (safe when already closed)

Backend failures are raised as ToolError, which the SDK returns to the client
as an MCP tool error (CallToolResult.is_error = True).

Standalone use (stdio transport; stdout is reserved for MCP messages):
    python <path>/hw3/mcp_server.py
"""

from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from camera_contract import SOURCE_SYNTHETIC, CameraBackend
from fake_camera import FakeCamera
from measurements import analyze_frame

SERVER_NAME = "lumen-hw3-fake-camera"
MEASURE_HISTOGRAM_BINS = 16
SYNTHETIC_NOTE = (
    "Synthetic FakeCamera data for integration testing only; not a laboratory "
    "measurement and not a validated optical model."
)
_BACKEND_ERRORS = (RuntimeError, ValueError, TypeError)


def create_server(backend: CameraBackend) -> MCPServer:
    """Create an MCP server whose tools operate on the supplied synthetic backend."""
    if not isinstance(backend, CameraBackend):
        raise TypeError(f"backend must implement CameraBackend, got {type(backend).__name__}")
    if backend.get_info().get("source") != SOURCE_SYNTHETIC:
        raise ValueError("This iteration only serves synthetic backends")

    server = MCPServer(
        name=SERVER_NAME,
        instructions=(
            "Controls a synthetic FakeCamera. All results are synthetic and must "
            "not be reported as laboratory measurements."
        ),
    )

    @server.tool()
    def camera_info() -> dict[str, Any]:
        """Describe the synthetic camera backend and its exposure limits in microseconds.

        Works whether or not the camera is open.
        """
        info = backend.get_info()
        min_us, max_us = backend.get_exposure_limits_us()
        return {
            "source": info["source"],
            "synthetic": info["source"] == SOURCE_SYNTHETIC,
            "backend": info,
            "exposure_limits_us": {"min": float(min_us), "max": float(max_us)},
            "note": SYNTHETIC_NOTE,
        }

    @server.tool()
    def camera_open() -> dict[str, Any]:
        """Open the synthetic camera so it can acquire frames and accept exposure changes."""
        try:
            backend.open()
        except _BACKEND_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        return {"ok": True, "action": "open", "is_open": True, "source": SOURCE_SYNTHETIC}

    @server.tool()
    def camera_set_exposure(
        exposure_us: Annotated[float, Field(strict=True, description="Exposure in microseconds")],
    ) -> dict[str, Any]:
        """Set the exposure in microseconds and return the value actually applied.

        Requires an open camera. Values outside the backend limits are rejected,
        never clipped.
        """
        try:
            applied = backend.set_exposure_us(exposure_us)
        except _BACKEND_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        return {
            "ok": True,
            "action": "set_exposure",
            "exposure_us": float(applied),
            "source": SOURCE_SYNTHETIC,
        }

    @server.tool()
    def camera_measure() -> dict[str, Any]:
        """Capture exactly one synthetic frame and return its measurement metrics.

        Requires an open camera. Metrics come from the Measurement Engine
        (analyze_frame) with a 16-bin histogram; raw pixels are not returned.
        """
        try:
            frame = backend.capture_frame()
        except _BACKEND_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        if frame.source != SOURCE_SYNTHETIC:
            raise ToolError(f"Expected a synthetic frame, got source={frame.source!r}")
        try:
            metrics = analyze_frame(frame.pixels, frame.bit_depth, bins=MEASURE_HISTOGRAM_BINS)
        except _BACKEND_ERRORS as exc:
            raise ToolError(f"Measurement failed: {exc}") from exc
        height, width = frame.pixels.shape
        return {
            "source": frame.source,
            "synthetic": True,
            "frame_id": int(frame.frame_id),
            "exposure_us": float(frame.exposure_us),
            "bit_depth": int(frame.bit_depth),
            "width": int(width),
            "height": int(height),
            "metrics": metrics,
            "note": SYNTHETIC_NOTE,
        }

    @server.tool()
    def camera_close() -> dict[str, Any]:
        """Close the synthetic camera. Closing an already closed camera is safe."""
        try:
            backend.close()
        except _BACKEND_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        return {"ok": True, "action": "close", "is_open": False, "source": SOURCE_SYNTHETIC}

    return server


if __name__ == "__main__":
    create_server(FakeCamera()).run("stdio")
