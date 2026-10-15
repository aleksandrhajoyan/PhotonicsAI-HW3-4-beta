"""Trusted host for ONE physical iRAYPLE session: LangGraph approval -> SessionGate -> authorized MCP server.

PhysicalApprovedSessionHost is OfflineApprovedSessionHost with exactly one
difference: the backend is an IRaypleCamera built on an already loaded vendor
SDK binding instead of FakeCamera. start(), pending(), resume() and
get_mcp_server() are inherited unchanged, so activation still requires BOTH
    1. this host's own LangGraph thread (hitl_approval) finishing "approved" for
       exactly this envelope's session_id and SHA-256, and
    2. a fresh execution-side SessionGate accepting the very same decision
       through submit_decision();
only then are AuthorizedCameraRuntime(IRaypleCamera, gate) and
create_authorized_server(runtime) built. There is no approved=True parameter,
no default approval and no way to pass an outcome or token in as authority.
Every camera tool call afterwards goes through runtime.execute(), i.e. through
SessionGate.begin_action(); a refused call never reaches the backend.

It is NOT an MCP tool. The trusted application bootstrap must:

    sdk = preload_vendor_sdk_once()          # once per process, before any MCP serving
    with PhysicalApprovedSessionHost(envelope, sdk) as host:   # no SDK call
        host.start(thread_id)                # the human sees host.pending()
        host.resume(decision)                # the human's structured decision
        server = host.get_mcp_server()       # only after valid approval
        ...                                  # serve tools on ONE event loop
    # __exit__ always calls shutdown(), also on errors, KeyboardInterrupt and cancellation

Vendor SDK: preload_vendor_sdk_once() is the only route to
irayple_camera.preload_vendor_sdk() (which briefly changes the working
directory). It loads at most once per process and refuses while a physical
host exists, so it can never run while tools are being served. The host only
receives the loaded binding; no tool handler loads or reloads anything.

Physical limits: the envelope (shown to the human unchanged by the existing
HITL flow) must lie within PHYSICAL_LIMITS, otherwise construction raises
ValueError before any camera object exists. The backend's wider SDK range
never widens the approved range: SessionGate checks the envelope only.

camera_info: AuthorizedCameraRuntime calls get_exposure_limits_us(), which on
IRaypleCamera requires an open camera. camera_info before camera_open
therefore fails without touching the device; the runtime treats the failure as
ambiguous, aborts the session and releases the (unopened) backend. Call
camera_open first.

Device ownership: at most one PhysicalApprovedSessionHost (hence one
IRaypleCamera) exists per process. Ownership is released only by a shutdown()
whose backend release succeeded; after a failed cleanup the device stays
claimed and the process must be restarted.

Sequential use: tool handlers of the authorized MCP server run one at a time
on its event loop; IRaypleCamera additionally rejects overlapping calls. No
thread-safety of the vendor SDK is assumed. This host is not thread-safe and
does not survive a process crash; a killed process cannot restore ExposureTime.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import irayple_camera as ic
from approved_session_host import HOST_NEW, HOST_SHUT_DOWN, HostError, OfflineApprovedSessionHost
from hitl_approval import build_approval_graph
from session_policy import SessionEnvelope

PHYSICAL_LIMITS = {
    "min_exposure_us": 1000.0,
    "max_exposure_us": 5000.0,
    "max_measurements": 2,
    "max_tool_calls": 7,
    "max_duration_seconds": 300.0,
}

_ownership_lock = threading.Lock()
_device_owner: PhysicalApprovedSessionHost | None = None
_preloaded_sdk: Any = None


def preload_vendor_sdk_once() -> Any:
    """Trusted bootstrap only: load the vendor binding once per process and return it.

    Refuses while a physical host exists, so it never runs during MCP serving.
    Raises irayple_discovery.DiscoveryError when the SDK cannot be loaded.
    """
    global _preloaded_sdk
    with _ownership_lock:
        if _device_owner is not None:
            raise HostError("refusing to load the vendor SDK while a physical session host exists")
        if _preloaded_sdk is None:
            _preloaded_sdk = ic.preload_vendor_sdk()
        return _preloaded_sdk


def check_physical_envelope(envelope: SessionEnvelope) -> None:
    """Reject an envelope broader than PHYSICAL_LIMITS. Never corrects or rebuilds it."""
    if not isinstance(envelope, SessionEnvelope):
        raise TypeError(f"envelope must be a SessionEnvelope, got {type(envelope).__name__}")
    limits = PHYSICAL_LIMITS
    problems = []
    if envelope.min_exposure_us < limits["min_exposure_us"]:
        problems.append(f"min_exposure_us {envelope.min_exposure_us!r} < {limits['min_exposure_us']!r}")
    if envelope.max_exposure_us > limits["max_exposure_us"]:
        problems.append(f"max_exposure_us {envelope.max_exposure_us!r} > {limits['max_exposure_us']!r}")
    for name in ("max_measurements", "max_tool_calls", "max_duration_seconds"):
        if getattr(envelope, name) > limits[name]:
            problems.append(f"{name} {getattr(envelope, name)!r} > {limits[name]!r}")
    if problems:
        raise ValueError("envelope exceeds the physical session limits: " + "; ".join(problems))


class PhysicalApprovedSessionHost(OfflineApprovedSessionHost):
    """Owns one envelope, one approval thread, one IRaypleCamera and, once approved, one MCP server."""

    def __init__(self, envelope: SessionEnvelope, sdk: Any, *,
                 clock: Callable[[], float] = time.monotonic):
        global _device_owner
        check_physical_envelope(envelope)
        if not callable(clock):
            raise TypeError("clock must be callable")
        with _ownership_lock:
            if _device_owner is not None:
                raise HostError("another physical session host already owns the camera")
            # IRaypleCamera validates the binding and makes no SDK call here.
            camera = ic.IRaypleCamera(sdk)
            _device_owner = self
        # Deliberately not calling OfflineApprovedSessionHost.__init__, which would build a
        # FakeCamera. These are the fields its inherited methods use, set the same way.
        self._envelope = envelope
        self._envelope_sha256 = envelope.sha256()
        self._clock = clock
        self._graph = build_approval_graph()
        self._camera = camera
        self._state = HOST_NEW
        self._thread_id = None
        self._runtime = None
        self._server = None

    def shutdown(self) -> dict:
        """Refuse all further host use, end the gate and release the camera via runtime.shutdown().

        Idempotent. IRaypleCamera.close() restores and verifies the original
        ExposureTime if any write was attempted and never repeats native
        Close/DestroyHandle. A failure raises authorized_camera.CleanupError and
        keeps the device claimed; calling shutdown() again reports it again.
        """
        result = super().shutdown()
        self._release_ownership()
        return result

    def status(self) -> dict:
        """Read-only host and session counters. Not an authorization and not an experiment verdict."""
        return {
            "host_state": self._state,
            "session_state": None if self._runtime is None else self._runtime.session_state,
            "owns_device": _device_owner is self,
        }

    def __enter__(self) -> PhysicalApprovedSessionHost:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()  # a CleanupError propagates, chained to any exception from the body

    def _release_ownership(self) -> None:
        global _device_owner
        with _ownership_lock:
            if _device_owner is self and self._state == HOST_SHUT_DOWN:
                _device_owner = None
