"""Offline protocol tests for hw3/authorized_mcp.py.

Every camera call goes through the official MCP SDK Client against an
in-process server. Sessions are activated by trusted fixture code that passes
a simulated structured human approval to SessionGate.submit_decision(); that
approval is never reachable through MCP. Only FakeCamera and a small spy /
fault-injection subclass of it are used: no camera, vendor SDK, network or
LLM is involved, and the injected clock means nothing ever waits.
"""

import ast
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

from mcp import Client

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

from authorized_camera import AuthorizedCameraRuntime  # noqa: E402
from authorized_mcp import SERVER_NAME, create_authorized_server  # noqa: E402
from fake_camera import FakeCamera  # noqa: E402
from mcp_server import create_server  # noqa: E402
from measurements import analyze_frame  # noqa: E402
from session_policy import (  # noqa: E402
    ABORTED,
    ACTIVE,
    AWAITING_APPROVAL,
    CLOSED,
    EXHAUSTED,
    EXPIRED,
    REFUSED,
    SessionEnvelope,
    SessionGate,
    session_approval,
)

MODULE_PATH = HW3_DIR / "authorized_mcp.py"
ALL_TOOLS = (
    "camera_info",
    "camera_open",
    "camera_set_exposure",
    "camera_measure",
    "camera_close",
)


def _error_text(result):
    return " ".join(getattr(block, "text", "") for block in result.content)


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
        return super().set_exposure_us(value)

    def capture_frame(self):
        self._enter("capture_frame")
        return super().capture_frame()


def _envelope(**overrides):
    fields = {
        "session_id": "session-mcp-001",
        "device_label": "fake-camera",
        "allowed_tools": ALL_TOOLS,
        "min_exposure_us": 100.0,
        "max_exposure_us": 10000.0,
        "max_measurements": 3,
        "max_tool_calls": 12,
        "max_duration_seconds": 60.0,
    }
    fields.update(overrides)
    return SessionEnvelope(**fields)


def _gate(envelope=None, approve=True, clock=None):
    """Trusted fixture code: the simulated human decision happens here, never over MCP."""
    envelope = envelope or _envelope()
    gate = SessionGate(envelope, clock or Clock())
    if approve:
        assert gate.submit_decision(session_approval(envelope.session_id, envelope.sha256(), True))
    return gate


class AuthorizedMcpTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Fresh camera, clock, gate, runtime and server per test.
        self.camera = SpyCamera()
        self.clock = Clock()
        self.gate = _gate(clock=self.clock)
        self.runtime = AuthorizedCameraRuntime(self.camera, self.gate)
        self.server = create_authorized_server(self.runtime)

    def use(self, gate=None, camera=None):
        """Replace the default fixture with a differently configured session."""
        self.camera = camera or SpyCamera()
        self.gate = gate or _gate(clock=self.clock)
        self.runtime = AuthorizedCameraRuntime(self.camera, self.gate)
        self.server = create_authorized_server(self.runtime)

    @asynccontextmanager
    async def connect(self, server=None):
        async with Client(server or self.server, mode="legacy", read_timeout_seconds=30) as client:
            yield client

    async def call_ok(self, client, name, arguments=None):
        result = await client.call_tool(name, arguments or {})
        self.assertFalse(result.is_error, f"{name} failed: {_error_text(result)}")
        self.assertIsInstance(result.structured_content, dict)
        return result.structured_content

    async def call_error(self, client, name, arguments=None):
        result = await client.call_tool(name, arguments or {})
        self.assertTrue(result.is_error, f"{name} unexpectedly succeeded")
        self.assertIsNone(result.structured_content)
        return _error_text(result)


class FactoryTests(unittest.TestCase):
    def test_accepts_authorized_runtime(self):
        runtime = AuthorizedCameraRuntime(FakeCamera(), _gate(approve=False))
        server = create_authorized_server(runtime)
        self.assertEqual(server.name, SERVER_NAME)

    def test_rejects_other_objects(self):
        camera = FakeCamera()
        gate = _gate()
        for candidate in (camera, gate, None, "approved", True, {"approved": True}, object()):
            with self.subTest(candidate=type(candidate).__name__), self.assertRaises(TypeError):
                create_authorized_server(candidate)

    def test_factory_does_not_touch_backend_or_gate(self):
        camera = SpyCamera()
        gate = _gate(approve=False)
        create_authorized_server(AuthorizedCameraRuntime(camera, gate))
        self.assertEqual(camera.calls, [])
        self.assertEqual(gate.state, AWAITING_APPROVAL)


class ProtocolTests(AuthorizedMcpTestCase):
    async def test_client_initializes(self):
        async with self.connect() as client:
            self.assertEqual(client.server_info.name, SERVER_NAME)
            self.assertIsNotNone(client.server_capabilities.tools)
            self.assertIn("synthetic", client.instructions)
        self.assertEqual(self.camera.calls, [])

    async def test_lists_exactly_the_five_camera_tools(self):
        async with self.connect() as client:
            tools = (await client.list_tools()).tools
        names = [tool.name for tool in tools]
        self.assertEqual(len(names), 5)
        self.assertEqual(set(names), set(ALL_TOOLS))
        for tool in tools:
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.description)
                expected = ["exposure_us"] if tool.name == "camera_set_exposure" else []
                self.assertEqual(sorted(tool.input_schema.get("properties", {})), expected)

    async def test_schemas_match_original_synthetic_server(self):
        async with self.connect() as client:
            ours = {t.name: t.input_schema for t in (await client.list_tools()).tools}
        async with self.connect(create_server(FakeCamera())) as client:
            original = {t.name: t.input_schema for t in (await client.list_tools()).tools}
        self.assertEqual(ours, original)
        self.assertEqual(ours["camera_set_exposure"]["required"], ["exposure_us"])
        self.assertEqual(ours["camera_set_exposure"]["properties"]["exposure_us"]["type"], "number")

    async def test_no_approval_or_session_management_tools(self):
        async with self.connect() as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            for forbidden in ("approve", "authoriz", "session", "gate", "shutdown", "decision",
                              "token", "fingerprint", "sha256"):
                with self.subTest(fragment=forbidden):
                    self.assertFalse([n for n in names if forbidden in n])
            for tool in ("approve_session", "submit_decision", "shutdown", "session_gate"):
                with self.subTest(tool=tool):
                    await self.call_error(client, tool)
        self.assertEqual(self.camera.calls, [])


class ApprovalStateTests(AuthorizedMcpTestCase):
    async def test_unapproved_gate_blocks_camera_open(self):
        self.use(gate=_gate(approve=False))
        async with self.connect() as client:
            text = await self.call_error(client, "camera_open")
        self.assertIn("Not authorized", text)
        self.assertIn(AWAITING_APPROVAL, text)
        self.assertFalse(self.camera.is_open)
        self.assertEqual(self.gate.state, AWAITING_APPROVAL)

    async def test_unapproved_requests_never_reach_fake_camera(self):
        self.use(gate=_gate(approve=False))
        async with self.connect() as client:
            for tool in ALL_TOOLS:
                args = {"exposure_us": 500.0} if tool == "camera_set_exposure" else None
                with self.subTest(tool=tool):
                    await self.call_error(client, tool, args)
        self.assertEqual(self.camera.calls, [])
        self.assertEqual(self.gate.state, AWAITING_APPROVAL)

    async def test_client_cannot_approve_by_arguments(self):
        self.use(gate=_gate(approve=False))
        envelope = self.gate.envelope
        smuggled = {"approved": True, "session_id": envelope.session_id,
                    "envelope_sha256": envelope.sha256(), "approval_token": "yes"}
        async with self.connect() as client:
            await self.call_error(client, "camera_open", smuggled)
            await self.call_error(client, "camera_measure", smuggled)
        self.assertEqual(self.camera.calls, [])
        self.assertEqual(self.gate.state, AWAITING_APPROVAL)

    async def test_refused_gate_blocks_acquisition(self):
        envelope = _envelope()
        gate = _gate(envelope, approve=False)
        self.assertFalse(gate.submit_decision(session_approval(envelope.session_id, envelope.sha256(), False)))
        camera = SpyCamera()
        camera.open()  # even an already open backend cannot be used
        camera.calls.clear()
        self.use(gate=gate, camera=camera)
        async with self.connect() as client:
            text = await self.call_error(client, "camera_measure")
            await self.call_error(client, "camera_set_exposure", {"exposure_us": 500.0})
        self.assertIn(REFUSED, text)
        self.assertEqual(camera.calls, [])
        self.assertEqual(gate.state, REFUSED)

    async def test_active_gate_permits_camera_open(self):
        async with self.connect() as client:
            result = await self.call_ok(client, "camera_open")
        self.assertIs(result["ok"], True)
        self.assertIs(result["is_open"], True)
        self.assertEqual(result["tool"], "camera_open")
        self.assertEqual(result["session_id"], "session-mcp-001")
        self.assertTrue(self.camera.is_open)
        self.assertEqual(self.camera.calls, ["open"])
        self.assertEqual(self.gate.tool_calls_used, 1)
        self.assertEqual(self.gate.action_status(result["action_id"]), "succeeded")

    async def test_aborted_gate_blocks_subsequent_acquisition(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            self.gate.abort("operator abort")  # trusted code, not an MCP tool
            text = await self.call_error(client, "camera_measure")
            await self.call_error(client, "camera_set_exposure", {"exposure_us": 500.0})
        self.assertIn(ABORTED, text)
        self.assertEqual(self.camera.calls, ["open"])
        self.assertEqual(self.gate.state, ABORTED)

    async def test_expired_session_blocks_acquisition(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            self.clock.now = 60.0
            await self.call_error(client, "camera_measure")
        self.assertEqual(self.gate.state, EXPIRED)
        self.assertNotIn("capture_frame", self.camera.calls)


class ToolTests(AuthorizedMcpTestCase):
    async def test_camera_info_goes_through_runtime(self):
        with mock.patch.object(self.runtime, "execute", wraps=self.runtime.execute) as spy:
            async with self.connect() as client:
                info = await self.call_ok(client, "camera_info")
        self.assertEqual(spy.call_count, 1)
        action_id, tool, arguments = spy.call_args.args
        self.assertEqual((tool, arguments), ("camera_info", {}))
        self.assertEqual(info["action_id"], action_id)
        self.assertEqual(info["source"], "synthetic")
        self.assertIs(info["synthetic"], True)
        self.assertEqual(info["backend_exposure_limits_us"], {"min": 100.0, "max": 10000.0})
        self.assertEqual(info["approved_exposure_limits_us"], {"min": 100.0, "max": 10000.0})
        self.assertEqual(self.gate.tool_calls_used, 1)
        self.assertEqual(self.gate.action_status(action_id), "succeeded")

    async def test_valid_exposure_change_succeeds(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            result = await self.call_ok(client, "camera_set_exposure", {"exposure_us": 2500.5})
        self.assertEqual(result["exposure_us"], 2500.5)
        self.assertEqual(self.camera.get_info()["exposure_us"], 2500.5)

    async def test_invalid_exposure_rejected(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            for value in ("1000", True, None, [1000.0], {"value": 1000.0}):
                with self.subTest(value=value):
                    await self.call_error(client, "camera_set_exposure", {"exposure_us": value})
            await self.call_error(client, "camera_set_exposure", {})
        self.assertNotIn("set_exposure_us", self.camera.calls)
        self.assertEqual(self.camera.get_info()["exposure_us"], 1000.0)
        self.assertEqual(self.gate.state, ACTIVE)

    async def test_exposure_outside_approved_range_rejected(self):
        self.use(gate=_gate(_envelope(min_exposure_us=200.0, max_exposure_us=5000.0)))
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            # 150 and 6000 are inside FakeCamera's limits but outside the approval.
            for value in (150.0, 199.999, 5000.5, 6000.0, 0, -1.0, 1e9):
                with self.subTest(value=value):
                    text = await self.call_error(client, "camera_set_exposure", {"exposure_us": value})
                    self.assertIn("Not authorized", text)
        self.assertNotIn("set_exposure_us", self.camera.calls)
        self.assertEqual(self.gate.state, ACTIVE)

    async def test_failed_exposure_change_keeps_state(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_set_exposure", {"exposure_us": 2000.0})
            await self.call_error(client, "camera_set_exposure", {"exposure_us": 20000.0})
            measurement = await self.call_ok(client, "camera_measure")
        self.assertEqual(self.camera.get_info()["exposure_us"], 2000.0)
        self.assertEqual(measurement["exposure_us"], 2000.0)
        self.assertEqual(self.camera.calls.count("set_exposure_us"), 1)
        self.assertEqual(self.gate.state, ACTIVE)

    async def test_measure_returns_synthetic_fake_camera_output(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            result = await self.call_ok(client, "camera_measure")
        self.assertEqual(result["source"], "synthetic")
        self.assertIs(result["synthetic"], True)
        self.assertEqual(result["frame_id"], 1)
        self.assertEqual(result["exposure_us"], 1000.0)
        self.assertEqual((result["bit_depth"], result["width"], result["height"]), (8, 8, 8))
        self.assertNotIn("pixels", result)
        self.assertEqual(self.camera.calls.count("capture_frame"), 1)

    async def test_metrics_match_measurement_engine(self):
        reference = FakeCamera()
        reference.open()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            for exposure in (1000.0, 2000.0):
                with self.subTest(exposure=exposure):
                    await self.call_ok(client, "camera_set_exposure", {"exposure_us": exposure})
                    result = await self.call_ok(client, "camera_measure")
                    reference.set_exposure_us(exposure)
                    frame = reference.capture_frame()
                    self.assertEqual(result["metrics"], analyze_frame(frame.pixels, frame.bit_depth, bins=16))
        self.assertEqual(result["metrics"]["histogram"]["bins"], 16)

    async def test_frame_ids_increase(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            ids = [(await self.call_ok(client, "camera_measure"))["frame_id"] for _ in range(3)]
        self.assertEqual(ids, [1, 2, 3])


class BudgetTests(AuthorizedMcpTestCase):
    async def test_measurement_budget_enforced(self):
        self.use(gate=_gate(_envelope(max_measurements=2)))
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_measure")
            await self.call_ok(client, "camera_measure")
            text = await self.call_error(client, "camera_measure")
        self.assertIn("measurement budget exhausted", text)
        self.assertEqual(self.camera.calls.count("capture_frame"), 2)
        self.assertEqual(self.gate.measurements_used, 2)

    async def test_tool_call_budget_enforced(self):
        self.use(gate=_gate(_envelope(max_tool_calls=3, max_measurements=1)))
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_info")
            await self.call_ok(client, "camera_measure")
            self.assertEqual(self.gate.state, EXHAUSTED)
            calls_before = list(self.camera.calls)
            for tool in ALL_TOOLS:
                args = {"exposure_us": 500.0} if tool == "camera_set_exposure" else None
                with self.subTest(tool=tool):
                    await self.call_error(client, tool, args)
        self.assertEqual(self.camera.calls, calls_before)
        self.assertEqual(self.gate.tool_calls_used, 3)


class ActionIdentityTests(AuthorizedMcpTestCase):
    async def test_client_supplied_action_ids_are_never_used(self):
        with mock.patch.object(self.runtime, "execute", wraps=self.runtime.execute) as spy:
            async with self.connect() as client:
                # The SDK drops unknown arguments, so each call runs with a fresh server id.
                first = await self.call_ok(client, "camera_open", {"action_id": "dup"})
                second = await self.call_ok(client, "camera_open", {"action_id": "dup"})
                await self.call_ok(client, "camera_set_exposure",
                                   {"exposure_us": 2000.0, "action_id": "dup"})
        self.assertEqual(spy.call_count, 3)
        self.assertNotEqual(first["action_id"], second["action_id"])
        for call in spy.call_args_list:
            action_id, tool, arguments = call.args
            self.assertNotEqual(action_id, "dup")
            self.assertNotIn("action_id", arguments)
        self.assertIsNone(self.gate.action_status("dup"))
        # A replayed client id cannot make a second open count as the "same" action.
        self.assertEqual(self.camera.calls.count("open"), spy.call_args_list.count(
            mock.call(mock.ANY, "camera_open", {})))

    async def test_server_action_ids_are_distinct_and_survive_errors(self):
        self.use(gate=_gate(_envelope(min_exposure_us=200.0)))
        with mock.patch.object(self.runtime, "execute", wraps=self.runtime.execute) as spy:
            async with self.connect() as client:
                await self.call_ok(client, "camera_open")
                await self.call_error(client, "camera_set_exposure", {"exposure_us": 150.0})
                await self.call_ok(client, "camera_set_exposure", {"exposure_us": 300.0})
                await self.call_ok(client, "camera_measure")
                await self.call_ok(client, "camera_info")
        ids = [call.args[0] for call in spy.call_args_list]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)
        for action_id in ids:
            self.assertTrue(action_id.startswith("mcp-"))
            self.assertLessEqual(len(action_id), 128)
        self.assertEqual(self.gate.action_status(ids[1]), None)  # rejected: never reserved
        self.assertEqual(self.gate.state, ACTIVE)

    async def test_independent_servers_do_not_share_identity_state(self):
        other_camera = SpyCamera()
        other_gate = _gate()
        other_runtime = AuthorizedCameraRuntime(other_camera, other_gate)
        other_server = create_authorized_server(other_runtime)
        async with self.connect() as client:
            first = await self.call_ok(client, "camera_open")
        async with self.connect(other_server) as client:
            second = await self.call_ok(client, "camera_open")
        prefix = lambda action_id: action_id.rsplit("-", 1)[0]  # noqa: E731
        self.assertNotEqual(prefix(first["action_id"]), prefix(second["action_id"]))
        self.assertEqual(self.gate.tool_calls_used, 1)
        self.assertEqual(other_gate.tool_calls_used, 1)
        self.assertEqual((self.camera.calls, other_camera.calls), (["open"], ["open"]))


class FailureTests(AuthorizedMcpTestCase):
    async def test_backend_failure_is_mcp_error_without_retry(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            self.camera.faults["capture_frame"] = RuntimeError("transport timeout")
            text = await self.call_error(client, "camera_measure")
            del self.camera.faults["capture_frame"]
            after = await self.call_error(client, "camera_measure")
        self.assertIn("transport timeout", text)
        self.assertIn("not retried", text)
        self.assertIn(ABORTED, text)
        self.assertNotIn("Traceback", text)
        self.assertNotIn(str(HW3_DIR), text)
        self.assertIn(ABORTED, after)
        self.assertEqual(self.camera.calls.count("capture_frame"), 1)
        self.assertEqual(self.gate.state, ABORTED)
        self.assertEqual(self.camera.calls[-1], "close")  # runtime's own fail-closed cleanup
        self.assertFalse(self.camera.is_open)

    async def test_cleanup_failure_is_not_concealed(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            self.camera.faults["capture_frame"] = RuntimeError("timeout")
            self.camera.faults["close"] = RuntimeError("close failed")
            text = await self.call_error(client, "camera_measure")
        self.assertIn("cleanup also failed", text)
        self.assertEqual(self.camera.calls.count("capture_frame"), 1)
        self.assertEqual(self.gate.state, ABORTED)

    async def test_unexpected_error_does_not_leak_internal_text(self):
        with mock.patch.object(self.runtime, "execute", side_effect=KeyError(r"C:\secret\path")):
            async with self.connect() as client:
                text = await self.call_error(client, "camera_open")
        self.assertNotIn("secret", text)
        self.assertEqual(self.camera.calls, [])


class CleanupTests(AuthorizedMcpTestCase):
    async def test_camera_close_follows_session_gate(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            result = await self.call_ok(client, "camera_close")
        self.assertIs(result["is_open"], False)
        self.assertFalse(self.camera.is_open)
        self.assertEqual(self.gate.tool_calls_used, 2)
        self.assertEqual(self.gate.action_status(result["action_id"]), "succeeded")
        self.assertEqual(self.gate.state, ACTIVE)

    async def test_camera_close_not_in_envelope_is_rejected(self):
        self.use(gate=_gate(_envelope(allowed_tools=("camera_open", "camera_measure"))))
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_error(client, "camera_close")
        self.assertTrue(self.camera.is_open)
        self.assertNotIn("close", self.camera.calls)

    async def test_camera_close_is_no_bypass_when_unapproved(self):
        camera = SpyCamera()
        camera.open()
        camera.calls.clear()
        self.use(gate=_gate(approve=False), camera=camera)
        async with self.connect() as client:
            await self.call_error(client, "camera_close")
        self.assertTrue(camera.is_open)
        self.assertEqual(camera.calls, [])

    async def test_trusted_shutdown_available_after_exhaustion(self):
        self.use(gate=_gate(_envelope(max_tool_calls=2, max_measurements=1)))
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_measure")
            self.assertEqual(self.gate.state, EXHAUSTED)
            await self.call_error(client, "camera_close")
            self.assertTrue(self.camera.is_open)
            # Trusted host code, not an MCP tool.
            self.assertEqual(self.runtime.shutdown(), {"backend_closed": True, "session_state": CLOSED})
            self.assertFalse(self.camera.is_open)
            await self.call_error(client, "camera_open")
        self.assertEqual(self.camera.calls.count("open"), 1)


class IsolationTests(unittest.TestCase):
    ALLOWED_IMPORTS = {"__future__", "itertools", "uuid", "typing", "mcp", "pydantic",
                       "authorized_camera", "session_policy"}

    def setUp(self):
        self.source = MODULE_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_imports_no_vendor_sdk_backend_langgraph_or_llm(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertLessEqual(imported, self.ALLOWED_IMPORTS)

    def test_no_vendor_references(self):
        for token in ("MVSDK", "sdk_probe", "ctypes", "iRAYPLE", ".dll"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_no_backend_gate_or_approval_access(self):
        used = {node.attr for node in ast.walk(self.tree) if isinstance(node, ast.Attribute)}
        used |= {node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)}
        for name in ("capture_frame", "set_exposure_us", "get_info", "get_exposure_limits_us",
                     "_backend", "_gate", "_dispatch", "submit_decision", "begin_action",
                     "complete_action", "abort", "shutdown", "analyze_frame", "SessionGate",
                     "FakeCamera", "session_approval"):
            with self.subTest(name=name):
                self.assertNotIn(name, used)
        self.assertIn("execute", used)

    def test_no_global_mutable_runtime(self):
        for node in self.tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                self.assertNotIn("runtime", ast.unparse(node).lower())
        self.assertFalse([n for n in ast.walk(self.tree) if isinstance(n, ast.Global)])

    def test_sdk_probe_not_loaded(self):
        self.assertNotIn("sdk_probe", sys.modules)


if __name__ == "__main__":
    unittest.main()
