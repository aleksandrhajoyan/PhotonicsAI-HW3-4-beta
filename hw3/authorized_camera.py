"""AuthorizedCameraRuntime: every ordinary camera tool call passes through a SessionGate.

Offline component. It uses the CameraBackend contract, the existing
SessionGate and the existing Measurement Engine; it never imports a vendor
SDK, the MCP server, LangGraph or an LLM, and it never approves a session.

Approval is the job of trusted application code (for example the HW3 LangGraph
approval workflow feeding SessionGate.submit_decision). This runtime only
accepts a SessionGate object; a boolean, a string such as "approved" or an
approval outcome dict is not an authorization capability.

Ordinary operations (the five camera tools of hw3/mcp_server.py):

    result = runtime.execute(action_id, tool, arguments)

    1. gate.begin_action(action_id, tool, arguments) reserves budget and checks
       the session state, allowed tools, exposure range and the one-in-flight
       rule. If it refuses, the backend is never contacted.
    2. Exactly one backend operation is dispatched for the authorized tool.
    3. gate.complete_action(action_id, "succeeded") records success.

Fail-closed rules:
    - Any failure after authorization (backend error, provenance mismatch,
      exposure mismatch, measurement error) is treated as an uncertain outcome:
      the action is marked "ambiguous", which aborts the session; nothing is
      retried; shutdown() is attempted; AuthorizedOperationError is raised.
    - If completion bookkeeping itself fails, the session is aborted, shutdown()
      is attempted and AuthorizedOperationError is raised.

Emergency cleanup (not a tool, not reachable through execute):

    runtime.shutdown()

ends authorization locally (gate.close()) and then calls backend.close(). It
works in every gate state, including exhausted, expired and aborted, makes no
hardware call other than backend.close(), never authorizes acquisition or
parameter changes, and is idempotent. A failed backend.close() raises
CleanupError and is never reported as successful cleanup.

Provenance: results carry the frame's own source. A frame must match the
source the backend declares, so synthetic data can never come back labelled
as hardware data.
"""

from __future__ import annotations

import math
from typing import Any

from camera_contract import SOURCE_HARDWARE, SOURCE_SYNTHETIC, CameraBackend
from measurements import analyze_frame
from session_policy import AMBIGUOUS, CAMERA_TOOLS, SUCCEEDED, ActionGrant, SessionGate

MEASURE_HISTOGRAM_BINS = 16  # the same binning as camera_measure in hw3/mcp_server.py
SUPPORTED_TOOLS = CAMERA_TOOLS
_KNOWN_SOURCES = (SOURCE_SYNTHETIC, SOURCE_HARDWARE)


class ProvenanceError(Exception):
    """The backend's declared source and the data it returned do not agree."""


class ExposureMismatchError(Exception):
    """The exposure the backend applied or used differs from what was authorized."""


class CleanupError(Exception):
    """backend.close() failed during shutdown; cleanup did not succeed."""


class AuthorizedOperationError(Exception):
    """An authorized operation failed; the session was aborted and nothing is retried.

    cleanup_error is the CleanupError raised by the automatic shutdown, or
    None if cleanup succeeded.
    """

    def __init__(self, message: str, *, action_id: str, tool: str, session_state: str,
                 cleanup_error: CleanupError | None):
        super().__init__(message)
        self.action_id = action_id
        self.tool = tool
        self.session_state = session_state
        self.cleanup_error = cleanup_error


def _declared_source(info: Any) -> str:
    if not isinstance(info, dict):
        raise ProvenanceError("backend get_info() did not return a dict")
    source = info.get("source")
    if source not in _KNOWN_SOURCES:
        raise ProvenanceError(f"backend declares unknown source {source!r}")
    if "synthetic" in info and info["synthetic"] is not (source == SOURCE_SYNTHETIC):
        raise ProvenanceError(
            f"backend declares source={source!r} but synthetic={info['synthetic']!r}"
        )
    return source


class AuthorizedCameraRuntime:
    """Dispatches the five camera tools to a backend, each guarded by a SessionGate."""

    def __init__(self, backend: CameraBackend, gate: SessionGate):
        if not isinstance(backend, CameraBackend):
            raise TypeError(f"backend must implement CameraBackend, got {type(backend).__name__}")
        if not isinstance(gate, SessionGate):
            raise TypeError(f"gate must be a SessionGate, got {type(gate).__name__}")
        # No backend call here: nothing touches the camera before an authorized action.
        self._backend = backend
        self._gate = gate
        self._backend_released = False
        self._dispatch = {
            "camera_info": self._camera_info,
            "camera_open": self._camera_open,
            "camera_set_exposure": self._camera_set_exposure,
            "camera_measure": self._camera_measure,
            "camera_close": self._camera_close,
        }

    @property
    def session_state(self) -> str:
        return self._gate.state

    # -- ordinary operations -------------------------------------------------

    def execute(self, action_id: str, tool: str, arguments: dict | None = None) -> dict:
        """Run one authorized tool call and return its JSON-compatible result.

        Raises PolicyError/TypeError/ValueError (from SessionGate.begin_action)
        without contacting the backend when the call is not authorized, and
        AuthorizedOperationError when an authorized call fails.
        """
        grant = self._gate.begin_action(action_id, tool, arguments)
        try:
            result = self._dispatch[grant.tool](grant)
        except BaseException as exc:
            # The physical effect is unknown: abort, never retry, clean up.
            try:
                self._gate.complete_action(grant.action_id, AMBIGUOUS)
            except Exception:
                self._gate.abort(f"could not record ambiguous outcome of {grant.action_id!r}")
            cleanup_error = self._attempt_cleanup()
            if not isinstance(exc, Exception):
                raise
            raise self._operation_error(grant, f"{grant.tool} failed: {exc}", cleanup_error) from exc
        try:
            self._gate.complete_action(grant.action_id, SUCCEEDED)
        except BaseException as exc:
            self._gate.abort(f"completion bookkeeping failed for {grant.action_id!r}: {exc}")
            cleanup_error = self._attempt_cleanup()
            if not isinstance(exc, Exception):
                raise
            raise self._operation_error(
                grant, f"completion bookkeeping failed: {exc}", cleanup_error
            ) from exc
        return result

    def _operation_error(self, grant: ActionGrant, message: str,
                         cleanup_error: CleanupError | None) -> AuthorizedOperationError:
        if cleanup_error is not None:
            message = f"{message}; cleanup also failed: {cleanup_error}"
        return AuthorizedOperationError(
            message,
            action_id=grant.action_id,
            tool=grant.tool,
            session_state=self._gate.state,
            cleanup_error=cleanup_error,
        )

    def _result(self, grant: ActionGrant, **fields: Any) -> dict:
        return {
            "tool": grant.tool,
            "action_id": grant.action_id,
            "session_id": grant.session_id,
            **fields,
        }

    def _camera_info(self, grant: ActionGrant) -> dict:
        info = self._backend.get_info()
        source = _declared_source(info)
        min_us, max_us = self._backend.get_exposure_limits_us()
        envelope = self._gate.envelope
        return self._result(
            grant,
            source=source,
            synthetic=source == SOURCE_SYNTHETIC,
            backend=info,
            backend_exposure_limits_us={"min": float(min_us), "max": float(max_us)},
            approved_exposure_limits_us={
                "min": envelope.min_exposure_us,
                "max": envelope.max_exposure_us,
            },
            # Reported, never reconciled: the approved range is not clipped or extended.
            approved_range_within_backend_limits=(
                float(min_us) <= envelope.min_exposure_us
                and envelope.max_exposure_us <= float(max_us)
            ),
        )

    def _camera_open(self, grant: ActionGrant) -> dict:
        self._backend.open()
        return self._result(grant, ok=True, is_open=True)

    def _camera_set_exposure(self, grant: ActionGrant) -> dict:
        requested = grant.exposure_us  # already validated against the envelope by the gate
        applied = self._backend.set_exposure_us(requested)
        if isinstance(applied, bool) or not isinstance(applied, (int, float)) or applied != requested:
            raise ExposureMismatchError(
                f"backend applied {applied!r} us instead of the authorized {requested!r} us"
            )
        return self._result(grant, ok=True, exposure_us=float(applied))

    def _camera_measure(self, grant: ActionGrant) -> dict:
        declared = _declared_source(self._backend.get_info())
        frame = self._backend.capture_frame()
        if frame.source != declared:
            raise ProvenanceError(
                f"backend declares source={declared!r} but the frame has source={frame.source!r}"
            )
        exposure_us = float(frame.exposure_us)
        envelope = self._gate.envelope
        if not (math.isfinite(exposure_us)
                and envelope.min_exposure_us <= exposure_us <= envelope.max_exposure_us):
            raise ExposureMismatchError(
                f"frame was acquired at {frame.exposure_us!r} us, outside the approved range "
                f"[{envelope.min_exposure_us!r}, {envelope.max_exposure_us!r}]"
            )
        metrics = analyze_frame(frame.pixels, frame.bit_depth, bins=MEASURE_HISTOGRAM_BINS)
        return self._result(
            grant,
            source=frame.source,
            synthetic=frame.source == SOURCE_SYNTHETIC,
            frame_id=int(frame.frame_id),
            exposure_us=exposure_us,
            bit_depth=int(frame.bit_depth),
            width=metrics["width"],
            height=metrics["height"],
            metrics=metrics,
        )

    def _camera_close(self, grant: ActionGrant) -> dict:
        self._backend.close()
        return self._result(grant, ok=True, is_open=False)

    # -- emergency cleanup ---------------------------------------------------

    def shutdown(self) -> dict:
        """End authorization and release the backend. Idempotent; not a tool.

        gate.close() runs first (a local transition that never raises), so no
        further action can be authorized. backend.close() is then called once;
        after it succeeds, later calls return without touching the backend.
        Raises CleanupError if backend.close() fails; calling shutdown() again
        retries only backend.close(), never an operation.
        """
        self._gate.close()
        if not self._backend_released:
            try:
                self._backend.close()
            except Exception as exc:
                raise CleanupError(f"backend.close() failed: {exc}") from exc
            self._backend_released = True
        return {"backend_closed": True, "session_state": self._gate.state}

    def _attempt_cleanup(self) -> CleanupError | None:
        try:
            self.shutdown()
        except CleanupError as exc:
            return exc
        return None
