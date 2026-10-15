"""Offline tests for hw3/authorized_camera.py.

Sessions are activated here by a simulated structured human approval passed
to SessionGate.submit_decision(). This is offline authorization testing, not
a real human approving a physical experiment. Only FakeCamera and small spy
or fault-injection subclasses of it are used: no camera, vendor SDK, MCP
server, network or LLM is involved.
"""

import ast
import dataclasses
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import fake_camera  # noqa: E402
from authorized_camera import (  # noqa: E402
    MEASURE_HISTOGRAM_BINS,
    SUPPORTED_TOOLS,
    AuthorizedCameraRuntime,
    AuthorizedOperationError,
    CleanupError,
    ExposureMismatchError,
    ProvenanceError,
)
from camera_contract import SOURCE_HARDWARE  # noqa: E402
from fake_camera import FakeCamera  # noqa: E402
from measurements import analyze_frame  # noqa: E402
from session_policy import (  # noqa: E402
    ABORTED,
    ACTIVE,
    AWAITING_APPROVAL,
    CLOSED,
    EXHAUSTED,
    EXPIRED,
    REFUSED,
    PolicyError,
    SessionEnvelope,
    SessionGate,
    session_approval,
)

ALL_TOOLS = (
    "camera_info",
    "camera_open",
    "camera_set_exposure",
    "camera_measure",
    "camera_close",
)
BACKEND_METHODS = ("open", "close", "get_info", "get_exposure_limits_us",
                   "set_exposure_us", "capture_frame")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SpyCamera(FakeCamera):
    """FakeCamera that records every backend call and can inject faults."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.faults = {}  # method name -> exception raised (after recording the call)
        self.frame_source = None  # overrides the captured frame's source
        self.applied_override = None  # overrides set_exposure_us's return value

    def _enter(self, name):
        self.calls.append(name)
        if name in self.faults:
            raise self.faults[name]

    def open(self):
        self._enter("open")
        super().open()

    def close(self):
        self._enter("close")
        super().close()

    def get_info(self):
        self._enter("get_info")
        return super().get_info()

    def get_exposure_limits_us(self):
        self._enter("get_exposure_limits_us")
        return super().get_exposure_limits_us()

    def set_exposure_us(self, value):
        self._enter("set_exposure_us")
        applied = super().set_exposure_us(value)
        return applied if self.applied_override is None else self.applied_override

    def capture_frame(self):
        self._enter("capture_frame")
        frame = super().capture_frame()
        if self.frame_source is not None:
            frame = dataclasses.replace(frame, source=self.frame_source)
        return frame


def _envelope(**overrides):
    fields = {
        "session_id": "session-001",
        "device_label": "fake-camera",
        "allowed_tools": ALL_TOOLS,
        "min_exposure_us": 100.0,
        "max_exposure_us": 10000.0,
        "max_measurements": 2,
        "max_tool_calls": 8,
        "max_duration_seconds": 60.0,
    }
    fields.update(overrides)
    return SessionEnvelope(**fields)


def _gate(envelope=None, approve=True, clock=None):
    envelope = envelope or _envelope()
    gate = SessionGate(envelope, clock or Clock())
    if approve:
        # Simulated structured human approval through the existing public API.
        assert gate.submit_decision(session_approval(envelope.session_id, envelope.sha256(), True))
    return gate


class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self.camera = SpyCamera()
        self.clock = Clock()
        self.gate = _gate(clock=self.clock)
        self.runtime = AuthorizedCameraRuntime(self.camera, self.gate)
        self._ids = 0

    def run_tool(self, tool, arguments=None, runtime=None):
        self._ids += 1
        return (runtime or self.runtime).execute(f"a-{self._ids}", tool, arguments)


class ConstructionTests(RuntimeTestCase):
    def test_only_a_session_gate_is_accepted(self):
        for fake in (True, "approved", {"status": "approved"}, None):
            with self.subTest(gate=fake), self.assertRaises(TypeError):
                AuthorizedCameraRuntime(FakeCamera(), fake)

    def test_backend_must_implement_contract(self):
        with self.assertRaises(TypeError):
            AuthorizedCameraRuntime(object(), self.gate)

    def test_construction_does_not_touch_backend(self):
        camera = SpyCamera()
        AuthorizedCameraRuntime(camera, _gate(approve=False))
        self.assertEqual(camera.calls, [])

    def test_vocabulary_matches_mcp_server_tools(self):
        tree = ast.parse((HW3_DIR / "mcp_server.py").read_text(encoding="utf-8"))
        tools = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any("tool" in ast.unparse(d) for d in node.decorator_list)
        }
        self.assertEqual(tools, set(SUPPORTED_TOOLS))
        self.assertEqual(set(self.runtime._dispatch), set(SUPPORTED_TOOLS))
        self.assertEqual(MEASURE_HISTOGRAM_BINS, 16)


class ApprovalStateTests(RuntimeTestCase):
    def test_unapproved_gate_blocks_camera_open(self):
        camera = SpyCamera()
        gate = _gate(approve=False)
        runtime = AuthorizedCameraRuntime(camera, gate)
        with self.assertRaises(PolicyError):
            runtime.execute("a-1", "camera_open", {})
        self.assertFalse(camera.is_open)
        self.assertEqual(gate.state, AWAITING_APPROVAL)

    def test_no_backend_call_of_any_kind_before_approval(self):
        camera = SpyCamera()
        runtime = AuthorizedCameraRuntime(camera, _gate(approve=False))
        for i, tool in enumerate(ALL_TOOLS):
            args = {"exposure_us": 500.0} if tool == "camera_set_exposure" else None
            with self.subTest(tool=tool), self.assertRaises(PolicyError):
                runtime.execute(f"a-{i}", tool, args)
        self.assertEqual(camera.calls, [])

    def test_refused_session_blocks_every_tool(self):
        envelope = _envelope()
        gate = _gate(envelope, approve=False)
        self.assertFalse(gate.submit_decision(session_approval("session-001", envelope.sha256(), False)))
        self.assertEqual(gate.state, REFUSED)
        camera = SpyCamera()
        camera.open()  # even an already open backend cannot be used
        camera.calls.clear()
        runtime = AuthorizedCameraRuntime(camera, gate)
        for i, tool in enumerate(ALL_TOOLS):
            args = {"exposure_us": 500.0} if tool == "camera_set_exposure" else None
            with self.subTest(tool=tool), self.assertRaises(PolicyError):
                runtime.execute(f"a-{i}", tool, args)
        self.assertEqual(camera.calls, [])

    def test_active_gate_permits_camera_open(self):
        result = self.run_tool("camera_open")
        self.assertEqual(result, {"tool": "camera_open", "action_id": "a-1",
                                  "session_id": "session-001", "ok": True, "is_open": True})
        self.assertTrue(self.camera.is_open)
        self.assertEqual(self.camera.calls, ["open"])
        self.assertEqual(self.gate.tool_calls_used, 1)
        self.assertEqual(self.gate.action_status("a-1"), "succeeded")
        self.assertIsNone(self.gate.in_flight_action_id)


class ToolPolicyTests(RuntimeTestCase):
    def test_exposure_inside_approved_range_is_applied(self):
        self.run_tool("camera_open")
        result = self.run_tool("camera_set_exposure", {"exposure_us": 2500.0})
        self.assertEqual(result["exposure_us"], 2500.0)
        self.assertEqual(self.camera.get_info()["exposure_us"], 2500.0)

    def test_exposure_outside_approved_range_never_reaches_backend(self):
        gate = _gate(_envelope(min_exposure_us=200.0, max_exposure_us=5000.0))
        runtime = AuthorizedCameraRuntime(self.camera, gate)
        self.run_tool("camera_open", runtime=runtime)
        bad_arguments = ({"exposure_us": 5000.5}, {"exposure_us": 150.0},
                         {"exposure_us": float("nan")}, {"exposure_us": True},
                         {"exposure_us": "1000"}, {}, {"exposure_us": 1000.0, "gain": 2})
        for args in bad_arguments:
            with self.subTest(args=args), self.assertRaises(PolicyError):
                self.run_tool("camera_set_exposure", args, runtime=runtime)
        self.assertNotIn("set_exposure_us", self.camera.calls)
        self.assertEqual(gate.state, ACTIVE)

    def test_unauthorized_tools_rejected(self):
        gate = _gate(_envelope(allowed_tools=("camera_info", "camera_open", "camera_close")))
        runtime = AuthorizedCameraRuntime(self.camera, gate)
        for tool, args in (("camera_measure", None), ("camera_set_exposure", {"exposure_us": 500.0}),
                           ("shutdown", None), ("camera_reset", None), ("open", None)):
            with self.subTest(tool=tool), self.assertRaises(PolicyError):
                self.run_tool(tool, args, runtime=runtime)
        self.assertEqual(self.camera.calls, [])

    def test_arguments_to_argumentless_tools_rejected(self):
        with self.assertRaises(PolicyError):
            self.run_tool("camera_open", {"force": True})
        self.assertEqual(self.camera.calls, [])

    def test_camera_info_reports_backend_and_approved_limits(self):
        gate = _gate(_envelope(max_exposure_us=20000.0))
        result = self.run_tool("camera_info", runtime=AuthorizedCameraRuntime(self.camera, gate))
        self.assertEqual(result["source"], "synthetic")
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["backend_exposure_limits_us"], {"min": 100.0, "max": 10000.0})
        self.assertEqual(result["approved_exposure_limits_us"], {"min": 100.0, "max": 20000.0})
        self.assertFalse(result["approved_range_within_backend_limits"])
        self.assertEqual(json.loads(json.dumps(result)), result)


class MeasurementTests(RuntimeTestCase):
    def test_measure_returns_measurement_engine_metrics(self):
        self.run_tool("camera_open")
        self.run_tool("camera_set_exposure", {"exposure_us": 2000.0})
        result = self.run_tool("camera_measure")
        reference = FakeCamera()
        reference.open()
        reference.set_exposure_us(2000.0)
        frame = reference.capture_frame()
        self.assertEqual(result["metrics"], analyze_frame(frame.pixels, 8, bins=16))
        self.assertEqual(result["metrics"]["histogram"]["bins"], 16)
        self.assertEqual(self.camera.calls.count("capture_frame"), 1)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_measure_is_labelled_synthetic(self):
        self.run_tool("camera_open")
        result = self.run_tool("camera_measure")
        self.assertEqual(result["source"], "synthetic")
        self.assertIs(result["synthetic"], True)

    def test_frame_metadata_is_the_backends_not_fabricated(self):
        self.run_tool("camera_open")
        self.run_tool("camera_set_exposure", {"exposure_us": 3000.0})
        first = self.run_tool("camera_measure")
        second = self.run_tool("camera_measure")
        self.assertEqual((first["frame_id"], second["frame_id"]), (1, 2))
        self.assertEqual(first["exposure_us"], 3000.0)
        self.assertEqual((first["bit_depth"], first["width"], first["height"]),
                         (fake_camera.BIT_DEPTH, fake_camera.WIDTH, fake_camera.HEIGHT))

    def test_frame_outside_approved_exposure_is_rejected(self):
        # FakeCamera starts at 1000 us; the approved range starts at 2000 us.
        gate = _gate(_envelope(min_exposure_us=2000.0))
        runtime = AuthorizedCameraRuntime(self.camera, gate)
        self.run_tool("camera_open", runtime=runtime)
        with self.assertRaises(AuthorizedOperationError) as ctx:
            self.run_tool("camera_measure", runtime=runtime)
        self.assertIsInstance(ctx.exception.__cause__, ExposureMismatchError)
        self.assertEqual(gate.state, ABORTED)

    def test_backend_that_alters_applied_exposure_aborts(self):
        self.camera.applied_override = 1234.0
        self.run_tool("camera_open")
        with self.assertRaises(AuthorizedOperationError) as ctx:
            self.run_tool("camera_set_exposure", {"exposure_us": 2000.0})
        self.assertIsInstance(ctx.exception.__cause__, ExposureMismatchError)
        self.assertEqual(self.gate.state, ABORTED)


class BudgetTests(RuntimeTestCase):
    def test_measurement_budget_enforced(self):
        self.run_tool("camera_open")
        self.run_tool("camera_measure")
        self.run_tool("camera_measure")
        with self.assertRaises(PolicyError):
            self.run_tool("camera_measure")
        self.assertEqual(self.camera.calls.count("capture_frame"), 2)
        self.assertEqual(self.gate.measurements_used, 2)

    def test_tool_call_budget_enforced(self):
        gate = _gate(_envelope(max_tool_calls=3, max_measurements=1))
        runtime = AuthorizedCameraRuntime(self.camera, gate)
        self.run_tool("camera_open", runtime=runtime)
        self.run_tool("camera_info", runtime=runtime)
        self.run_tool("camera_measure", runtime=runtime)
        self.assertEqual(gate.state, EXHAUSTED)
        calls_before = list(self.camera.calls)
        for tool in ALL_TOOLS:
            args = {"exposure_us": 500.0} if tool == "camera_set_exposure" else None
            with self.subTest(tool=tool), self.assertRaises(PolicyError):
                self.run_tool(tool, args, runtime=runtime)
        self.assertEqual(self.camera.calls, calls_before)

    def test_expired_session_cannot_acquire(self):
        self.run_tool("camera_open")
        self.clock.now = 60.0
        with self.assertRaises(PolicyError):
            self.run_tool("camera_measure")
        self.assertEqual(self.gate.state, EXPIRED)
        self.assertNotIn("capture_frame", self.camera.calls)

    def test_repeated_action_id_cannot_execute_again(self):
        self.runtime.execute("same", "camera_open")
        with self.assertRaises(PolicyError):
            self.runtime.execute("same", "camera_open")
        with self.assertRaises(PolicyError):
            self.runtime.execute("same", "camera_measure")
        self.assertEqual(self.camera.calls, ["open"])

    def test_rejected_action_id_is_also_spent(self):
        with self.assertRaises(PolicyError):
            self.runtime.execute("x", "camera_set_exposure", {"exposure_us": 1e9})
        with self.assertRaises(PolicyError):
            self.runtime.execute("x", "camera_open")
        self.assertEqual(self.camera.calls, [])

    def test_one_action_in_flight_at_a_time(self):
        inner_errors = []
        runtime = self.runtime

        class ReentrantCamera(SpyCamera):
            def open(self):
                super().open()
                try:
                    runtime.execute("inner", "camera_measure")
                except PolicyError as exc:
                    inner_errors.append(exc)

        self.camera.__class__ = ReentrantCamera
        self.run_tool("camera_open")
        self.assertEqual(len(inner_errors), 1)
        self.assertIn("in flight", str(inner_errors[0]))
        self.assertNotIn("capture_frame", self.camera.calls)


class FailureTests(RuntimeTestCase):
    def test_ambiguous_backend_failure_aborts_and_cleans_up(self):
        self.run_tool("camera_open")
        self.camera.faults["capture_frame"] = RuntimeError("transport timeout")
        with self.assertRaises(AuthorizedOperationError) as ctx:
            self.run_tool("camera_measure")
        err = ctx.exception
        self.assertEqual((err.action_id, err.tool, err.session_state), ("a-2", "camera_measure", ABORTED))
        self.assertIsNone(err.cleanup_error)
        self.assertEqual(self.gate.state, ABORTED)
        self.assertEqual(self.gate.action_status("a-2"), "ambiguous")
        self.assertEqual(self.camera.calls[-1], "close")
        self.assertFalse(self.camera.is_open)

    def test_failed_operation_is_not_retried(self):
        self.camera.faults["open"] = RuntimeError("device busy")
        with self.assertRaises(AuthorizedOperationError):
            self.run_tool("camera_open")
        self.assertEqual(self.camera.calls.count("open"), 1)
        del self.camera.faults["open"]
        with self.assertRaises(PolicyError):
            self.run_tool("camera_open")
        self.assertEqual(self.camera.calls.count("open"), 1)

    def test_completion_accounting_failure_fails_closed(self):
        self.run_tool("camera_open")
        with mock.patch.object(self.gate, "complete_action", side_effect=RuntimeError("ledger down")):
            with self.assertRaises(AuthorizedOperationError) as ctx:
                self.run_tool("camera_measure")
        self.assertIn("completion bookkeeping failed", str(ctx.exception))
        self.assertEqual(self.gate.state, ABORTED)
        self.assertEqual(self.camera.calls[-1], "close")
        with self.assertRaises(PolicyError):
            self.run_tool("camera_measure")
        self.assertEqual(self.camera.calls.count("capture_frame"), 1)

    def test_cleanup_failure_is_reported_not_hidden(self):
        self.run_tool("camera_open")
        self.camera.faults["capture_frame"] = RuntimeError("timeout")
        self.camera.faults["close"] = RuntimeError("close failed")
        with self.assertRaises(AuthorizedOperationError) as ctx:
            self.run_tool("camera_measure")
        self.assertIsInstance(ctx.exception.cleanup_error, CleanupError)
        self.assertIn("cleanup also failed", str(ctx.exception))
        with self.assertRaises(CleanupError):
            self.runtime.shutdown()
        del self.camera.faults["close"]
        self.assertEqual(self.runtime.shutdown(), {"backend_closed": True, "session_state": ABORTED})
        self.assertEqual(self.camera.calls.count("capture_frame"), 1)


class ProvenanceTests(RuntimeTestCase):
    def test_synthetic_frame_relabelled_as_hardware_is_rejected(self):
        self.camera.frame_source = SOURCE_HARDWARE
        self.run_tool("camera_open")
        with self.assertRaises(AuthorizedOperationError) as ctx:
            self.run_tool("camera_measure")
        self.assertIsInstance(ctx.exception.__cause__, ProvenanceError)
        self.assertEqual(self.gate.state, ABORTED)

    def test_inconsistent_declared_source_is_rejected(self):
        original = SpyCamera.get_info
        with mock.patch.object(SpyCamera, "get_info", autospec=True,
                               side_effect=lambda cam: {**original(cam), "source": SOURCE_HARDWARE}):
            with self.assertRaises(AuthorizedOperationError) as ctx:
                self.run_tool("camera_info")
        self.assertIsInstance(ctx.exception.__cause__, ProvenanceError)
        self.assertEqual(self.gate.state, ABORTED)

    def test_unknown_frame_source_is_rejected(self):
        self.camera.frame_source = "simulated-hardware"
        self.run_tool("camera_open")
        with self.assertRaises(AuthorizedOperationError) as ctx:
            self.run_tool("camera_measure")
        self.assertIsInstance(ctx.exception.__cause__, ProvenanceError)


class CleanupTests(RuntimeTestCase):
    def test_shutdown_after_budget_exhaustion(self):
        gate = _gate(_envelope(max_tool_calls=2, max_measurements=1))
        runtime = AuthorizedCameraRuntime(self.camera, gate)
        self.run_tool("camera_open", runtime=runtime)
        self.run_tool("camera_measure", runtime=runtime)
        self.assertEqual(gate.state, EXHAUSTED)
        with self.assertRaises(PolicyError):
            self.run_tool("camera_close", runtime=runtime)
        self.assertTrue(self.camera.is_open)
        self.assertEqual(runtime.shutdown(), {"backend_closed": True, "session_state": CLOSED})
        self.assertFalse(self.camera.is_open)

    def test_shutdown_after_expiry(self):
        self.run_tool("camera_open")
        self.clock.now = 120.0
        with self.assertRaises(PolicyError):
            self.run_tool("camera_info")
        self.assertEqual(self.gate.state, EXPIRED)
        self.runtime.shutdown()
        self.assertFalse(self.camera.is_open)

    def test_shutdown_after_abort(self):
        self.run_tool("camera_open")
        self.gate.abort("operator abort")
        self.assertEqual(self.runtime.shutdown(), {"backend_closed": True, "session_state": ABORTED})
        self.assertFalse(self.camera.is_open)

    def test_explicit_camera_close_follows_policy(self):
        self.run_tool("camera_open")
        result = self.run_tool("camera_close")
        self.assertEqual(result["is_open"], False)
        self.assertFalse(self.camera.is_open)
        self.assertEqual(self.gate.tool_calls_used, 2)
        self.assertEqual(self.gate.action_status("a-2"), "succeeded")
        self.assertEqual(self.gate.state, ACTIVE)

    def test_camera_close_not_allowed_by_envelope_is_rejected(self):
        gate = _gate(_envelope(allowed_tools=("camera_open",)))
        runtime = AuthorizedCameraRuntime(self.camera, gate)
        self.run_tool("camera_open", runtime=runtime)
        with self.assertRaises(PolicyError):
            self.run_tool("camera_close", runtime=runtime)
        self.assertNotIn("close", self.camera.calls)

    def test_shutdown_is_idempotent_and_ends_authorization(self):
        self.run_tool("camera_open")
        first = self.runtime.shutdown()
        second = self.runtime.shutdown()
        self.assertEqual(first, second)
        self.assertEqual(self.camera.calls.count("close"), 1)
        self.assertEqual(self.gate.state, CLOSED)
        with self.assertRaises(PolicyError):
            self.run_tool("camera_open")
        self.assertEqual(self.camera.calls, ["open", "close"])

    def test_shutdown_only_calls_backend_close(self):
        self.runtime.shutdown()
        self.assertEqual(self.camera.calls, ["close"])

    def test_shutdown_is_not_a_tool(self):
        with self.assertRaises(PolicyError):
            self.runtime.execute("a-x", "shutdown")
        self.assertEqual(self.camera.calls, [])


class IsolationTests(unittest.TestCase):
    ALLOWED_IMPORTS = {"__future__", "math", "typing", "camera_contract", "measurements",
                       "session_policy"}

    def test_module_imports_no_sdk_mcp_langgraph_or_llm(self):
        tree = ast.parse((HW3_DIR / "authorized_camera.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertLessEqual(imported, self.ALLOWED_IMPORTS)

    def test_no_own_approval_or_budget_logic(self):
        tree = ast.parse((HW3_DIR / "authorized_camera.py").read_text(encoding="utf-8"))
        used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        used |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for name in ("submit_decision", "_approval_problem", "hashlib", "max_measurements",
                     "max_tool_calls", "tool_calls_used", "measurements_used"):
            self.assertNotIn(name, used)
        self.assertIn("begin_action", used)
        self.assertIn("complete_action", used)

    def test_backend_calls_stay_inside_camera_contract(self):
        camera = SpyCamera()
        runtime = AuthorizedCameraRuntime(camera, _gate())
        for i, (tool, args) in enumerate((("camera_info", None), ("camera_open", None),
                                          ("camera_set_exposure", {"exposure_us": 800.0}),
                                          ("camera_measure", None), ("camera_close", None))):
            runtime.execute(f"a-{i}", tool, args)
        runtime.shutdown()
        self.assertLessEqual(set(camera.calls), set(BACKEND_METHODS))


if __name__ == "__main__":
    unittest.main()
