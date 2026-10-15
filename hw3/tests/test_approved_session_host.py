"""Offline tests for hw3/approved_session_host.py.

The "human" decisions here are SIMULATED by this trusted test harness and
passed to host.resume(). They drive the real LangGraph interrupt/resume path
and the real SessionGate, but they are NOT real user authentication and NOT a
real human approving a physical experiment. Camera calls go through the
official MCP SDK Client against the in-process authorized server. The only
backend is FakeCamera (replaced by a recording subclass of it per test); no
camera, vendor SDK, network or LLM is involved, and the injected clock means
nothing ever waits.
"""

import ast
import json
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

from mcp import Client

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import approved_session_host as ash  # noqa: E402
import authorized_mcp  # noqa: E402
from approved_session_host import HostError, OfflineApprovedSessionHost  # noqa: E402
from authorized_camera import AuthorizedCameraRuntime  # noqa: E402
from fake_camera import FakeCamera  # noqa: E402
from hitl_approval import (  # noqa: E402
    APPROVAL_NODE,
    STATUS_APPROVED,
    STATUS_REFUSED,
    ApprovalError,
    build_approval_graph,
    resume_approval,
    start_approval,
    thread_config,
)
from measurements import analyze_frame  # noqa: E402
from session_policy import CLOSED, SessionEnvelope, SessionGate, session_approval  # noqa: E402

MODULE_PATH = HW3_DIR / "approved_session_host.py"
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
    """FakeCamera that records every backend call (installed as the host's FakeCamera)."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def open(self):
        self.calls.append("open")
        super().open()

    def close(self):
        self.calls.append("close")
        super().close()

    def get_info(self):
        self.calls.append("get_info")
        return super().get_info()

    def get_exposure_limits_us(self):
        self.calls.append("get_exposure_limits_us")
        return super().get_exposure_limits_us()

    def set_exposure_us(self, value):
        self.calls.append("set_exposure_us")
        return super().set_exposure_us(value)

    def capture_frame(self):
        self.calls.append("capture_frame")
        return super().capture_frame()


def _envelope(**overrides):
    fields = {
        "session_id": "session-host-001",
        "device_label": "fake-camera",
        "allowed_tools": ALL_TOOLS,
        "min_exposure_us": 200.0,
        "max_exposure_us": 5000.0,
        "max_measurements": 2,
        "max_tool_calls": 10,
        "max_duration_seconds": 60.0,
    }
    fields.update(overrides)
    return SessionEnvelope(**fields)


def _approve(envelope):
    """The simulated human's structured approval (test harness only)."""
    return session_approval(envelope.session_id, envelope.sha256(), True)


class HostTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Independent fixture per test: the host's FakeCamera is a fresh SpyCamera.
        patcher = mock.patch.object(ash, "FakeCamera", SpyCamera)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.envelope = _envelope()
        self.clock = Clock()
        self.host = self.new_host(self.envelope)

    def new_host(self, envelope=None):
        host = OfflineApprovedSessionHost(envelope or self.envelope, clock=self.clock)
        self.assertIsInstance(host._camera, SpyCamera)
        return host

    def camera(self, host=None):
        return (host or self.host)._camera

    def approve(self, host=None, thread_id="t-1"):
        host = host or self.host
        host.start(thread_id)
        result = host.resume(_approve(host._envelope))
        self.assertTrue(result["activated"], result)
        return result

    def assert_inactive(self, host=None):
        host = host or self.host
        with self.assertRaises(HostError):
            host.get_mcp_server()
        self.assertIsNone(host._runtime)
        self.assertIsNone(host._server)
        self.assertEqual(self.camera(host).calls, [])
        self.assertFalse(self.camera(host).is_open)

    def assert_refused(self, decision, host=None, thread_id="t-1"):
        host = host or self.host
        host.start(thread_id)
        result = host.resume(decision)
        self.assertIs(result["activated"], False)
        self.assertEqual(host._state, ash.HOST_REFUSED)
        self.assert_inactive(host)
        return result

    @asynccontextmanager
    async def connect(self, host=None):
        server = (host or self.host).get_mcp_server()
        async with Client(server, mode="legacy", read_timeout_seconds=30) as client:
            yield client

    async def call_ok(self, client, name, arguments=None):
        result = await client.call_tool(name, arguments or {})
        self.assertFalse(result.is_error, f"{name} failed: {_error_text(result)}")
        self.assertIsInstance(result.structured_content, dict)
        return result.structured_content

    async def call_error(self, client, name, arguments=None):
        result = await client.call_tool(name, arguments or {})
        self.assertTrue(result.is_error, f"{name} unexpectedly succeeded")
        return _error_text(result)


class ConstructionTests(HostTestCase):
    def test_constructs_without_touching_fake_camera(self):
        self.assertEqual(self.camera().calls, [])
        self.assertFalse(self.camera().is_open)
        self.assertEqual(self.host._state, ash.HOST_NEW)
        self.assertIsNone(self.host.pending())

    def test_only_a_session_envelope_is_accepted(self):
        for candidate in (self.envelope.canonical_dict(), None, "approved", {"approved": True}):
            with self.subTest(candidate=type(candidate).__name__), self.assertRaises(TypeError):
                OfflineApprovedSessionHost(candidate)

    def test_no_backend_or_hardware_mode_parameter(self):
        for kwargs in ({"backend": FakeCamera()}, {"camera": FakeCamera()}, {"hardware": True}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(TypeError):
                OfflineApprovedSessionHost(self.envelope, **kwargs)

    def test_public_api_is_minimal(self):
        public = {name for name in dir(OfflineApprovedSessionHost) if not name.startswith("_")}
        self.assertEqual(public, {"start", "pending", "resume", "get_mcp_server", "shutdown"})


class InterruptTests(HostTestCase):
    def test_start_pauses_at_real_langgraph_interrupt(self):
        request = self.host.start("t-1")
        snapshot = self.host._graph.get_state(thread_config("t-1"))
        self.assertEqual(snapshot.next, (APPROVAL_NODE,))
        self.assertEqual(len(snapshot.interrupts), 1)
        self.assertEqual(snapshot.interrupts[0].value, request)
        self.assertEqual(request, SessionGate(self.envelope).approval_request())
        self.assertEqual(request["envelope_sha256"], self.envelope.sha256())

    def test_pending_request_available_for_display(self):
        request = self.host.start("t-1")
        self.assertEqual(self.host.pending(), request)
        self.assertEqual(json.loads(json.dumps(request)), request)

    def test_no_mcp_server_before_approval(self):
        with self.assertRaises(HostError):
            self.host.get_mcp_server()
        self.host.start("t-1")
        with self.assertRaises(HostError):
            self.host.get_mcp_server()

    def test_no_backend_runtime_or_server_before_approval(self):
        with mock.patch.object(ash, "AuthorizedCameraRuntime") as runtime_cls, \
                mock.patch.object(ash, "create_authorized_server") as factory:
            self.host.start("t-1")
            self.host.pending()
        runtime_cls.assert_not_called()
        factory.assert_not_called()
        self.assert_inactive()


class ActivationTests(HostTestCase):
    def test_structured_approval_activates_host(self):
        self.host.start("t-1")
        self.assertEqual(self.camera().calls, [])
        result = self.host.resume(_approve(self.envelope))
        self.assertIs(result["activated"], True)
        self.assertEqual(result["approval_outcome"]["status"], STATUS_APPROVED)
        self.assertEqual(result["approval_outcome"]["envelope_sha256"], self.envelope.sha256())
        self.assertIsNone(self.host.pending())
        self.assertIs(self.host.get_mcp_server(), self.host._server)
        self.assertEqual(self.host._runtime.session_state, "active")
        self.assertEqual(self.camera().calls, [])  # activation itself makes no backend call

    def test_same_decision_reaches_execution_gate(self):
        decision = _approve(self.envelope)
        original = SessionGate.submit_decision
        with mock.patch.object(SessionGate, "submit_decision", autospec=True,
                               side_effect=original) as spy:
            self.host.start("t-1")
            spy.assert_not_called()
            self.host.resume(decision)
        # Once inside the graph's approval node, once on the execution-side gate.
        self.assertEqual(spy.call_count, 2)
        graph_gate, exec_gate = (call.args[0] for call in spy.call_args_list)
        self.assertIsNot(graph_gate, exec_gate)
        for call in spy.call_args_list:
            self.assertEqual(call.args[1], decision)
        self.assertIs(self.host._runtime._gate, exec_gate)

    def test_envelope_is_not_mutated(self):
        before = self.envelope.canonical_json()
        self.approve()
        self.host.shutdown()
        self.assertEqual(self.envelope.canonical_json(), before)
        self.assertIs(self.host._envelope, self.envelope)


class McpSessionTests(HostTestCase):
    async def test_approved_tool_sequence_over_real_mcp_client(self):
        self.approve()
        async with self.connect() as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            info = await self.call_ok(client, "camera_info")
            opened = await self.call_ok(client, "camera_open")
            exposure = await self.call_ok(client, "camera_set_exposure", {"exposure_us": 2000.0})
            measurement = await self.call_ok(client, "camera_measure")
        self.assertEqual(tools, set(ALL_TOOLS))

        self.assertEqual(info["source"], "synthetic")
        self.assertIs(info["synthetic"], True)
        self.assertEqual(info["backend"]["backend"], "FakeCamera")
        self.assertEqual(info["approved_exposure_limits_us"], {"min": 200.0, "max": 5000.0})
        self.assertEqual(info["session_id"], "session-host-001")

        self.assertIs(opened["is_open"], True)
        self.assertEqual(exposure["exposure_us"], 2000.0)

        reference = FakeCamera()
        reference.open()
        reference.set_exposure_us(2000.0)
        frame = reference.capture_frame()
        self.assertEqual(measurement["source"], "synthetic")
        self.assertIs(measurement["synthetic"], True)
        self.assertEqual(measurement["exposure_us"], 2000.0)
        self.assertEqual(measurement["frame_id"], 1)
        self.assertEqual(measurement["metrics"], analyze_frame(frame.pixels, frame.bit_depth, bins=16))
        self.assertEqual(self.camera().calls.count("capture_frame"), 1)
        self.assertTrue(self.camera().is_open)

    async def test_every_tool_call_goes_through_runtime(self):
        self.approve()
        runtime = self.host._runtime
        with mock.patch.object(runtime, "execute", wraps=runtime.execute) as spy:
            async with self.connect() as client:
                await self.call_ok(client, "camera_open")
                await self.call_ok(client, "camera_measure")
        self.assertEqual([call.args[1] for call in spy.call_args_list],
                         ["camera_open", "camera_measure"])

    async def test_exposure_outside_approval_rejected(self):
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            text = await self.call_error(client, "camera_set_exposure", {"exposure_us": 6000.0})
        self.assertIn("Not authorized", text)
        self.assertNotIn("set_exposure_us", self.camera().calls)

    async def test_out_of_budget_measurement_rejected(self):
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_measure")
            await self.call_ok(client, "camera_measure")
            text = await self.call_error(client, "camera_measure")
        self.assertIn("measurement budget exhausted", text)
        self.assertEqual(self.camera().calls.count("capture_frame"), 2)

    async def test_mcp_client_has_no_approval_tools(self):
        self.approve()
        async with self.connect() as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            for fragment in ("approve", "authoriz", "session", "gate", "resume", "start",
                             "shutdown", "decision", "host", "token", "sha256"):
                with self.subTest(fragment=fragment):
                    self.assertFalse([n for n in names if fragment in n])
            for tool in ("resume", "approve_session", "submit_decision", "shutdown"):
                with self.subTest(tool=tool):
                    await self.call_error(client, tool, {"decision": _approve(self.envelope)})


class RefusalTests(HostTestCase):
    def test_explicit_refusal_never_touches_backend(self):
        with mock.patch.object(ash, "AuthorizedCameraRuntime") as runtime_cls, \
                mock.patch.object(ash, "create_authorized_server") as factory:
            result = self.assert_refused(
                session_approval(self.envelope.session_id, self.envelope.sha256(), False))
        runtime_cls.assert_not_called()
        factory.assert_not_called()
        self.assertEqual(result["approval_outcome"]["status"], STATUS_REFUSED)
        self.assertEqual(result["reason"], "human refused")

    def test_refusal_is_terminal(self):
        self.assert_refused(session_approval(self.envelope.session_id, self.envelope.sha256(), False))
        with self.assertRaises(HostError):
            self.host.resume(_approve(self.envelope))
        with self.assertRaises(HostError):
            self.host.start("t-2")
        self.assert_inactive()

    def test_free_text_cannot_activate(self):
        texts = ("approve", "yes", "approved: True", json.dumps(_approve(self.envelope)))
        for i, text in enumerate(texts):
            with self.subTest(text=text):
                self.assert_refused(text, host=self.new_host(), thread_id=f"t-text-{i}")

    def test_wrong_session_id_cannot_activate(self):
        result = self.assert_refused(session_approval("session-other", self.envelope.sha256(), True))
        self.assertIn("session-other", result["reason"])

    def test_wrong_envelope_hash_cannot_activate(self):
        other = _envelope(max_measurements=3).sha256()
        for i, digest in enumerate((other, self.envelope.sha256().upper(), "", None)):
            with self.subTest(digest=digest):
                result = self.assert_refused(
                    session_approval(self.envelope.session_id, digest, True),
                    host=self.new_host(), thread_id=f"t-hash-{i}")
                self.assertIn("SHA-256", result["reason"])

    def test_missing_or_malformed_fields_fail_closed(self):
        full = _approve(self.envelope)
        decisions = [{k: v for k, v in full.items() if k != key} for key in full]
        decisions += [{}, None, [], 42, True, ("approved", True)]
        decisions += [dict(full, approved=value) for value in (1, 1.0, "true", "yes", [True], None)]
        decisions += [dict(full, session_id=value) for value in (None, 1, ["session-host-001"])]
        for i, decision in enumerate(decisions):
            with self.subTest(decision=repr(decision)):
                self.assert_refused(decision, host=self.new_host(), thread_id=f"t-bad-{i}")

    def test_forged_outcome_or_status_is_not_authority(self):
        forged_outcome = {
            "status": STATUS_APPROVED,
            "session_id": self.envelope.session_id,
            "envelope_sha256": self.envelope.sha256(),
            "reason": "forged",
        }
        for i, forged in enumerate((forged_outcome, "approved", {"approved": "approved"},
                                    {"approval_token": self.envelope.sha256()})):
            with self.subTest(forged=repr(forged)):
                self.assert_refused(forged, host=self.new_host(), thread_id=f"t-forged-{i}")

    def test_mismatched_graph_outcome_does_not_activate(self):
        good = _approve(self.envelope)
        approved = {"status": STATUS_APPROVED, "session_id": self.envelope.session_id,
                    "envelope_sha256": self.envelope.sha256(), "reason": "x"}
        bad_outcomes = (
            dict(approved, session_id="session-other"),
            dict(approved, envelope_sha256=_envelope(max_measurements=3).sha256()),
            dict(approved, status="APPROVED"),
            None,
        )
        for i, outcome in enumerate(bad_outcomes):
            with self.subTest(outcome=repr(outcome)), \
                    mock.patch.object(ash, "resume_approval", return_value=outcome):
                self.assert_refused(good, host=self.new_host(), thread_id=f"t-mm-{i}")

    def test_execution_gate_independently_rejects_bad_decision(self):
        # Even if the graph reported approval, the execution-side gate re-validates the decision.
        approved = {"status": STATUS_APPROVED, "session_id": self.envelope.session_id,
                    "envelope_sha256": self.envelope.sha256(), "reason": "x"}
        with mock.patch.object(ash, "resume_approval", return_value=approved):
            result = self.assert_refused("approve")
        self.assertIn("execution gate refused", result["reason"])


class ReplayAndIsolationTests(HostTestCase):
    def test_completed_approval_cannot_be_replayed(self):
        self.approve()
        with self.assertRaises(HostError):
            self.host.resume(_approve(self.envelope))
        with self.assertRaises(HostError):
            self.host.start("t-1")
        with self.assertRaises(HostError):
            self.host.start("t-2")
        self.assertIsNotNone(self.host.get_mcp_server())

    def test_repeated_start_rejected(self):
        self.host.start("t-1")
        with self.assertRaises(HostError):
            self.host.start("t-1")
        with self.assertRaises(HostError):
            self.host.start("t-2")
        self.assertIsNotNone(self.host.pending())

    def test_resume_before_start_rejected(self):
        with self.assertRaises(HostError):
            self.host.resume(_approve(self.envelope))
        self.assert_inactive()

    def test_decision_for_another_thread_cannot_activate(self):
        env_b = _envelope(session_id="session-host-B", max_exposure_us=1000.0)
        host_b = self.new_host(env_b)
        self.host.start("t-a")
        host_b.start("t-b")
        # Host B's human decision delivered to host A is refused.
        result = self.host.resume(_approve(env_b))
        self.assertIs(result["activated"], False)
        self.assert_inactive(self.host)
        # Host B is still pending, and its own decision still works.
        self.assertIsNotNone(host_b.pending())
        self.assertIs(host_b.resume(_approve(env_b))["activated"], True)
        self.assert_inactive(self.host)

    def test_approval_on_a_foreign_graph_thread_does_not_activate(self):
        self.host.start("t-1")
        foreign = build_approval_graph()
        start_approval(foreign, "t-1", self.envelope)
        self.assertEqual(resume_approval(foreign, "t-1", _approve(self.envelope))["status"],
                         STATUS_APPROVED)
        with self.assertRaises(ApprovalError):
            resume_approval(self.host._graph, "t-other", _approve(self.envelope))
        with self.assertRaises(HostError):
            self.host.get_mcp_server()
        self.assertIsNotNone(self.host.pending())

    def test_separate_hosts_do_not_share_approval_state(self):
        other = self.new_host()
        self.approve(thread_id="t-shared")
        self.assertIsNot(other._graph, self.host._graph)
        self.assertIsNot(other._camera, self.host._camera)
        self.assertIsNone(other.pending())
        with self.assertRaises(HostError):
            other.get_mcp_server()
        other.start("t-shared")  # same thread id, separate checkpointer
        self.assertIsNotNone(other.pending())
        with self.assertRaises(HostError):
            other.get_mcp_server()
        self.assertEqual(self.camera(other).calls, [])


class ShutdownTests(HostTestCase):
    async def test_shutdown_closes_backend_and_blocks_host(self):
        self.approve()
        server = self.host.get_mcp_server()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
        original = AuthorizedCameraRuntime.shutdown
        with mock.patch.object(AuthorizedCameraRuntime, "shutdown", autospec=True,
                               side_effect=original) as spy:
            result = self.host.shutdown()
        spy.assert_called_once()
        self.assertEqual(result["runtime"], {"backend_closed": True, "session_state": CLOSED})
        self.assertFalse(self.camera().is_open)
        self.assertEqual(self.camera().calls[-1], "close")
        for operation, args in (("get_mcp_server", ()), ("start", ("t-2",)),
                                ("resume", (_approve(self.envelope),)), ("pending", ())):
            with self.subTest(operation=operation), self.assertRaises(HostError):
                getattr(self.host, operation)(*args)
        # A client still holding the old server object is refused by the closed gate.
        calls_before = list(self.camera().calls)
        async with Client(server, mode="legacy", read_timeout_seconds=30) as client:
            await self.call_error(client, "camera_open")
            await self.call_error(client, "camera_measure")
        self.assertEqual(self.camera().calls, calls_before)

    def test_shutdown_is_idempotent(self):
        self.approve()
        self.host.shutdown()
        self.host.shutdown()
        self.assertEqual(self.camera().calls, ["close"])

    def test_shutdown_before_or_after_refusal_touches_nothing(self):
        self.host.start("t-1")
        self.assertEqual(self.host.shutdown(), {"host_state": ash.HOST_SHUT_DOWN, "runtime": None})
        self.assert_inactive()
        refused = self.new_host()
        self.assert_refused("no", host=refused)
        refused.shutdown()
        self.assert_inactive(refused)


class IsolationTests(unittest.TestCase):
    ALLOWED_IMPORTS = {"__future__", "time", "collections", "typing", "authorized_camera",
                       "authorized_mcp", "fake_camera", "hitl_approval", "session_policy"}

    def setUp(self):
        self.source = MODULE_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_imports_no_sdk_network_llm_or_mcp_transport(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertLessEqual(imported, self.ALLOWED_IMPORTS)

    def test_no_vendor_network_or_launcher_references(self):
        for token in ("MVSDK", "sdk_probe", "ctypes", "iRAYPLE", ".dll", "socket", "http",
                      "stdio", "anthropic", "openai", "__main__", ".run("):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_sdk_probe_not_loaded(self):
        self.assertNotIn("sdk_probe", sys.modules)

    def test_authorized_server_module_unchanged_in_role(self):
        # The host passes only an AuthorizedCameraRuntime to the existing factory.
        self.assertIs(ash.create_authorized_server, authorized_mcp.create_authorized_server)


if __name__ == "__main__":
    unittest.main()
