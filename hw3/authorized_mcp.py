"""MCP server whose five camera tools are delegated to an AuthorizedCameraRuntime.

Built on the official MCP Python SDK v2 (mcp.server.MCPServer). Every tool
handler makes exactly one call:

    runtime.execute(action_id, tool_name, arguments)

so the runtime's SessionGate decides, before any backend action, whether the
call is authorized. This module never touches a CameraBackend, never sees a
SessionGate, never approves a session and never computes measurement
statistics. It imports no vendor SDK, LangGraph or LLM, and has no stdio
entry point: the trusted application that hosts it builds the runtime
(backend + gate approved through its own human-in-the-loop workflow) and
passes it to create_authorized_server().

Tools (names and argument schemas match hw3/mcp_server.py):
    camera_info, camera_open, camera_set_exposure(exposure_us), camera_measure,
    camera_close

Action identity: each server instance owns a private issuer that generates a
fresh action_id per tool invocation ("mcp-<uuid4 hex>-<counter>"). Clients
cannot supply one; an id is consumed even when the call is rejected or fails,
and is passed to the runtime unchanged.

Errors: a refusal by the runtime (PolicyError, TypeError, ValueError raised by
SessionGate.begin_action) and a failed authorized operation
(AuthorizedOperationError, after the runtime has already aborted the session
and attempted cleanup) are both returned as MCP tool errors
(CallToolResult.is_error = True). Nothing is retried and no session state is
changed here. Any other exception is reported by the SDK as an opaque error
without its internal text.

Concurrency: the handlers are coroutines that call the synchronous runtime
directly, so one server processes tool calls one at a time on its event loop
and never runs the (not thread-safe) SessionGate from several worker threads.

Cleanup ownership: runtime.shutdown() is deliberately NOT a tool. The trusted
application hosting this server is responsible for calling runtime.shutdown()
when the server lifecycle ends (normally or after an error). This module adds
no automatic cleanup hook, and cleanup after a process crash is not
guaranteed.
"""

from __future__ import annotations

import itertools
import uuid
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from authorized_camera import AuthorizedCameraRuntime, AuthorizedOperationError
from session_policy import PolicyError

SERVER_NAME = "lumen-hw3-authorized-camera"
_REJECTION_ERRORS = (PolicyError, TypeError, ValueError)


class _ActionIdIssuer:
    """Unique, never-reused action ids for one server instance."""

    def __init__(self) -> None:
        self._prefix = f"mcp-{uuid.uuid4().hex}"
        self._counter = itertools.count(1)

    def next_id(self) -> str:
        return f"{self._prefix}-{next(self._counter)}"


def create_authorized_server(runtime: AuthorizedCameraRuntime) -> MCPServer:
    """Create an MCP server whose camera tools all go through the given runtime.

    The runtime is built by trusted application code; its gate may be in any
    state (an inactive gate makes every tool call fail).
    """
    if not isinstance(runtime, AuthorizedCameraRuntime):
        raise TypeError(
            f"runtime must be an AuthorizedCameraRuntime, got {type(runtime).__name__}"
        )

    action_ids = _ActionIdIssuer()
    server = MCPServer(
        name=SERVER_NAME,
        instructions=(
            "Controls one camera session authorized by a human outside this server. "
            "Every call is checked against the approved session limits and may be "
            "refused. Results carry their own source label; synthetic data must not "
            "be reported as laboratory measurements."
        ),
    )

    def run(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        action_id = action_ids.next_id()
        try:
            return runtime.execute(action_id, tool, arguments)
        except AuthorizedOperationError as exc:
            raise ToolError(
                f"{exc} (session {exc.session_state}; the operation was not retried)"
            ) from exc
        except _REJECTION_ERRORS as exc:
            raise ToolError(f"Not authorized: {exc}") from exc

    @server.tool()
    async def camera_info() -> dict[str, Any]:
        """Describe the camera backend, its exposure limits and the approved exposure range.

        Requires an authorized session; counts against the tool-call budget.
        """
        return run("camera_info", {})

    @server.tool()
    async def camera_open() -> dict[str, Any]:
        """Open the camera within the authorized session."""
        return run("camera_open", {})

    @server.tool()
    async def camera_set_exposure(
        exposure_us: Annotated[float, Field(strict=True, description="Exposure in microseconds")],
    ) -> dict[str, Any]:
        """Set the exposure in microseconds and return the value actually applied.

        Values outside the approved session range are rejected, never clipped.
        """
        return run("camera_set_exposure", {"exposure_us": exposure_us})

    @server.tool()
    async def camera_measure() -> dict[str, Any]:
        """Capture exactly one frame and return its Measurement Engine metrics.

        Counts against the session's measurement budget; raw pixels are not returned.
        """
        return run("camera_measure", {})

    @server.tool()
    async def camera_close() -> dict[str, Any]:
        """Close the camera. This is an ordinary tool call subject to the session policy."""
        return run("camera_close", {})

    return server
