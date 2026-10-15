"""Offline tests for hw3/physical_approved_session_host.py.

The "human" decisions are SIMULATED by this trusted test harness and passed to
host.resume(); they drive the real LangGraph interrupt/resume path and the real
SessionGate but are not real user authentication. Camera calls go through the
official MCP SDK Client against the in-process authorized server, into the real
AuthorizedCameraRuntime and the real IRaypleCamera, whose SDK binding is the
fake binding of test_irayple_camera. No camera, vendor SDK, network or LLM is
involved, and the injected clock means nothing ever waits.
"""

import ast
import importlib
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

from mcp import Client

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import approved_session_host as ash  # noqa: E402
import irayple_camera as ic  # noqa: E402
import physical_approved_session_host as phys  # noqa: E402
import test_irayple_camera as base  # noqa: E402  (fake SDK binding shared with the backend tests)
from approved_session_host import HostError, OfflineApprovedSessionHost  # noqa: E402
from authorized_camera import CleanupError  # noqa: E402
from hitl_approval import STATUS_APPROVED  # noqa: E402
from measurements import analyze_frame  # noqa: E402
from session_policy import SessionEnvelope, SessionGate, session_approval  # noqa: E402

import numpy as np  # noqa: E402

MODULE_PATH = HW3_DIR / "physical_approved_session_host.py"
ALL_TOOLS = ("camera_info", "camera_open", "camera_set_exposure", "camera_measure", "camera_close")
WRITE = base.WRITE
NATIVE_LIFECYCLE = ("IMV_Close", "IMV_DestroyHandle")


def _envelope(**overrides):
    fields = {
        "session_id": "physical-session-001",
        "device_label": "irayple-MV-A7200MU130",
        "allowed_tools": ALL_TOOLS,
        "min_exposure_us": 1000.0,
        "max_exposure_us": 5000.0,
        "max_measurements": 2,
        "max_tool_calls": 7,
        "max_duration_seconds": 120.0,
    }
    fields.update(overrides)
    return SessionEnvelope(**fields)


def _approve(envelope):
    """The simulated human's structured approval (test harness only)."""
    return session_approval(envelope.session_id, envelope.sha256(), True)


def _error_text(result):
    return " ".join(getattr(block, "text", "") for block in result.content)


def _pixels(payload):
    return np.frombuffer(payload, np.uint8).reshape(2, 4)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PhysicalHostTestCase(unittest.IsolatedAsyncioTestCase):
    """Real SDK loading, DLL loading, directory changes and FakeCamera construction all fail loudly."""

    def setUp(self):
        guard = AssertionError("real SDK access attempted in an offline test")
        for target in (
            mock.patch("ctypes.WinDLL", side_effect=guard, create=True),
            mock.patch("ctypes.CDLL", side_effect=guard),
            mock.patch("os.add_dll_directory", side_effect=guard, create=True),
            mock.patch("os.chdir", side_effect=AssertionError("working directory changed")),
            mock.patch.object(ic.discovery, "load_vendor_sdk", side_effect=guard),
            mock.patch.object(ic, "preload_vendor_sdk", side_effect=guard),
            mock.patch.object(ash, "FakeCamera", side_effect=AssertionError("FakeCamera used")),
        ):
            target.start()
            self.addCleanup(target.stop)
        self.addCleanup(self._release_process_state)
        self.clock = Clock()
        self.envelope = _envelope()
        self.fake = None

    def _release_process_state(self):
        # Tests that deliberately leave a failed cleanup keep the device claimed; reset per test.
        phys._device_owner = None
        phys._preloaded_sdk = None

    def tearDown(self):
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)
        if self.fake is not None:
            self.assertEqual(self.fake.violations, [])

    def make_host(self, envelope=None, **fake_kwargs):
        self.fake = base.FakeSdk(**fake_kwargs)
        self.host = phys.PhysicalApprovedSessionHost(envelope or self.envelope, self.fake.ns, clock=self.clock)
        return self.host

    def approve(self, host=None, thread_id="t-1"):
        host = host or self.host
        host.start(thread_id)
        result = host.resume(_approve(host._envelope))
        self.assertTrue(result["activated"], result)
        return result

    def assert_refused(self, decision, thread_id="t-1"):
        host = self.make_host()
        host.start(thread_id)
        result = host.resume(decision)
        self.assertIs(result["activated"], False)
        with self.assertRaises(HostError):
            host.get_mcp_server()
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(host.shutdown(), {"host_state": ash.HOST_SHUT_DOWN, "runtime": None})
        self.assertEqual(self.fake.calls, [])
        self.assertIsNone(phys._device_owner)
        return result

    def assert_released(self):
        self.assertFalse(self.fake.is_open)
        self.assertFalse(self.fake.is_grabbing)
        self.assertEqual(self.fake.handles_alive, 0)
        self.assertEqual(self.fake.frames_outstanding, {})
        for name in NATIVE_LIFECYCLE:
            self.assertEqual(self.fake.counts.get(name, 0), 1, name)  # never repeated

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


# 1. Import safety.
class ImportSafetyTests(unittest.TestCase):
    def test_import_loads_no_sdk(self):
        cwd = os.getcwd()
        with mock.patch.object(ic.discovery, "load_vendor_sdk") as loader, \
                mock.patch.object(ic, "preload_vendor_sdk") as preload, \
                mock.patch.object(ic, "IRaypleCamera") as backend, \
                mock.patch("ctypes.WinDLL", create=True) as win_dll, \
                mock.patch("os.add_dll_directory", create=True) as add_dir, \
                mock.patch("os.chdir") as chdir:
            importlib.reload(phys)
        for patched in (loader, preload, backend, win_dll, add_dir, chdir):
            patched.assert_not_called()
        self.assertIsNone(phys._preloaded_sdk)
        self.assertIsNone(phys._device_owner)
        self.assertEqual(os.getcwd(), cwd)
        self.assertNotIn("IMVApi", sys.modules)


# 2, 7, 25. Construction, envelope limits and single ownership.
class ConstructionTests(PhysicalHostTestCase):
    def test_creation_does_not_open_camera(self):
        host = self.make_host()
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.fake.handles_alive, 0)
        self.assertEqual(host.status(), {"host_state": ash.HOST_NEW, "session_state": None, "owns_device": True})
        self.assertIsNone(host.pending())
        self.assertIsInstance(host, OfflineApprovedSessionHost)  # same trusted host interface

    def test_envelope_broader_than_physical_limits_rejected(self):
        for overrides in ({"min_exposure_us": 999.0}, {"max_exposure_us": 5000.5}, {"max_exposure_us": 1e6},
                          {"max_measurements": 3, "max_tool_calls": 7}, {"max_tool_calls": 8},
                          {"max_duration_seconds": 301.0}):
            with self.subTest(overrides=overrides):
                with mock.patch.object(ic, "IRaypleCamera") as backend, self.assertRaises(ValueError):
                    self.make_host(envelope=_envelope(**overrides))
                backend.assert_not_called()
                self.assertIsNone(phys._device_owner)
        self.make_host(envelope=_envelope(min_exposure_us=2000.0, max_exposure_us=3000.0, max_measurements=1,
                                          max_tool_calls=3, max_duration_seconds=10.0))

    def test_constructor_rejects_non_envelopes_and_bad_bindings(self):
        for envelope in (self.envelope.canonical_dict(), None, "approved"):
            with self.subTest(envelope=type(envelope).__name__), self.assertRaises(TypeError):
                phys.PhysicalApprovedSessionHost(envelope, base.FakeSdk().ns)
        for sdk in (None, object(), "IMVApi"):
            with self.subTest(sdk=sdk), self.assertRaises(TypeError):
                phys.PhysicalApprovedSessionHost(self.envelope, sdk)
        self.assertIsNone(phys._device_owner)

    def test_no_approval_parameter(self):
        for kwargs in ({"approved": True}, {"approval": _approve(self.envelope)}, {"decision": True},
                       {"backend": object()}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(TypeError):
                phys.PhysicalApprovedSessionHost(self.envelope, base.FakeSdk().ns, **kwargs)

    def test_single_physical_camera_owner(self):
        first = self.make_host()
        with mock.patch.object(ic, "IRaypleCamera") as backend, self.assertRaises(HostError):
            phys.PhysicalApprovedSessionHost(_envelope(session_id="other"), base.FakeSdk().ns)
        backend.assert_not_called()
        self.assertIs(phys._device_owner, first)
        first.shutdown()
        self.assertFalse(first.status()["owns_device"])
        self.make_host()
        self.assertTrue(self.host.status()["owns_device"])


# 8. Vendor SDK preload.
class PreloadTests(PhysicalHostTestCase):
    def test_preload_once_through_existing_loader(self):
        binding = base.FakeSdk().ns
        with mock.patch.object(ic, "preload_vendor_sdk", return_value=binding) as preload:
            self.assertIs(phys.preload_vendor_sdk_once(), binding)
            self.assertIs(phys.preload_vendor_sdk_once(), binding)
        preload.assert_called_once_with()

    def test_preload_refused_while_host_exists(self):
        self.make_host()
        with mock.patch.object(ic, "preload_vendor_sdk") as preload, self.assertRaises(HostError):
            phys.preload_vendor_sdk_once()
        preload.assert_not_called()

    async def test_tools_never_load_sdk(self):
        host = self.make_host()
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_info")
            await self.call_ok(client, "camera_set_exposure", {"exposure_us": 1000.0})
            await self.call_ok(client, "camera_measure")
            await self.call_ok(client, "camera_close")
        host.shutdown()
        ic.preload_vendor_sdk.assert_not_called()  # guarded in setUp
        ic.discovery.load_vendor_sdk.assert_not_called()
        self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)  # the injected binding was used


# 3-6. Approval boundary.
class ApprovalTests(PhysicalHostTestCase):
    def test_no_approval_prevents_all_camera_operations(self):
        host = self.make_host()
        with self.assertRaises(HostError):
            host.get_mcp_server()
        host.start("t-1")
        with self.assertRaises(HostError):
            host.get_mcp_server()
        with self.assertRaises(HostError):
            host.start("t-2")
        self.assertEqual(self.fake.calls, [])
        host.shutdown()
        self.assertEqual(self.fake.calls, [])

    def test_rejected_approval_never_opens(self):
        result = self.assert_refused(session_approval(self.envelope.session_id, self.envelope.sha256(), False))
        self.assertEqual(result["reason"], "human refused")

    def test_malformed_or_forged_decisions_fail_closed(self):
        full = _approve(self.envelope)
        forged_outcome = {"status": STATUS_APPROVED, "session_id": self.envelope.session_id,
                          "envelope_sha256": self.envelope.sha256(), "reason": "forged"}
        decisions = [{k: v for k, v in full.items() if k != key} for key in full]
        decisions += [{}, None, [], 42, True, "approve", "yes", "approved", forged_outcome]
        decisions += [dict(full, approved=value) for value in (1, "true", [True])]
        decisions += [session_approval("other-session", self.envelope.sha256(), True),
                      session_approval(self.envelope.session_id, _envelope(max_measurements=1).sha256(), True),
                      session_approval(self.envelope.session_id, self.envelope.sha256().upper(), True)]
        for i, decision in enumerate(decisions):
            with self.subTest(decision=repr(decision)):
                self.assert_refused(decision, thread_id=f"t-bad-{i}")

    def test_existing_hitl_graph_and_execution_gate_both_decide(self):
        host = self.make_host()
        decision = _approve(self.envelope)
        original = SessionGate.submit_decision
        with mock.patch.object(SessionGate, "submit_decision", autospec=True, side_effect=original) as spy:
            request = host.start("t-1")
            spy.assert_not_called()
            self.assertEqual(host.pending(), request)
            result = host.resume(decision)
        self.assertIs(result["activated"], True)
        self.assertEqual(spy.call_count, 2)  # inside the LangGraph node, then the execution gate
        self.assertIsNot(spy.call_args_list[0].args[0], spy.call_args_list[1].args[0])
        self.assertEqual(self.fake.calls, [])  # activation itself makes no SDK call

    def test_graph_outcome_alone_is_not_authority(self):
        approved = {"status": STATUS_APPROVED, "session_id": self.envelope.session_id,
                    "envelope_sha256": self.envelope.sha256(), "reason": "x"}
        with mock.patch.object(ash, "resume_approval", return_value=approved):
            result = self.assert_refused("approve")
        self.assertIn("execution gate refused", result["reason"])

    def test_resume_before_start_and_replay_rejected(self):
        host = self.make_host()
        with self.assertRaises(HostError):
            host.resume(_approve(self.envelope))
        self.assertEqual(self.fake.calls, [])

    def test_approved_envelope_identity_preserved(self):
        host = self.make_host()
        before = self.envelope.canonical_json()
        request = host.start("t-1")
        self.assertEqual(request["envelope"], self.envelope.canonical_dict())
        self.assertEqual(request["envelope_sha256"], self.envelope.sha256())
        result = host.resume(_approve(self.envelope))
        self.assertEqual(result["approval_outcome"]["envelope_sha256"], self.envelope.sha256())
        self.assertIs(host._runtime._gate.envelope, self.envelope)
        with self.assertRaises(HostError):
            host.resume(_approve(_envelope(max_exposure_us=4000.0)))
        host.shutdown()
        self.assertEqual(self.envelope.canonical_json(), before)


# 9-20. Authorized MCP tool lifecycle.
class ToolLifecycleTests(PhysicalHostTestCase):
    async def test_planned_physical_sequence(self):
        host = self.make_host()
        self.approve()
        self.assertEqual(self.fake.calls, [])
        async with self.connect() as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            opened = await self.call_ok(client, "camera_open")
            self.assertEqual(self.fake.names(), base.OPEN_SEQUENCE)  # opened only now
            first_exposure = await self.call_ok(client, "camera_set_exposure", {"exposure_us": 1000.0})
            first = await self.call_ok(client, "camera_measure")
            second_exposure = await self.call_ok(client, "camera_set_exposure", {"exposure_us": 5000.0})
            second = await self.call_ok(client, "camera_measure")
            closed = await self.call_ok(client, "camera_close")
        self.assertEqual(tools, set(ALL_TOOLS))
        self.assertIs(opened["is_open"], True)
        self.assertEqual((first_exposure["exposure_us"], second_exposure["exposure_us"]), (1000.0, 5000.0))
        for result, exposure, payload in ((first, 1000.0, base.PAYLOAD_1), (second, 5000.0, base.PAYLOAD_2)):
            self.assertEqual(result["source"], "hardware")
            self.assertIs(result["synthetic"], False)
            self.assertEqual(result["exposure_us"], exposure)
            self.assertEqual((result["bit_depth"], result["width"], result["height"]), (8, 4, 2))
            self.assertEqual(result["metrics"], analyze_frame(_pixels(payload), 8, bins=16))
        self.assertEqual([first["frame_id"], second["frame_id"]], [1, 2])  # native blockId is 1 both times
        self.assertEqual(self.fake.exposure_at_frame, [1000.0, 5000.0])
        self.assertIs(closed["is_open"], False)
        # camera_close restored the original 5000 us and verified it by readback.
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0, 5000.0])
        self.assertEqual(self.fake.names()[-4:], [WRITE, base.READ_VALUE, *NATIVE_LIFECYCLE])
        mark = len(self.fake.calls)
        self.assertEqual(host.shutdown()["runtime"], {"backend_closed": True, "session_state": "closed"})
        self.assertEqual(len(self.fake.calls), mark)  # no duplicate Close/DestroyHandle
        self.assert_released()
        self.assertIsNone(phys._device_owner)

    async def test_camera_info_after_open_reports_both_ranges(self):
        self.make_host()
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            info = await self.call_ok(client, "camera_info")
        self.assertEqual(info["source"], "hardware")
        self.assertEqual(info["backend"]["backend"], "IRaypleCamera")
        self.assertEqual(info["backend"]["serial"], "****7890")
        self.assertNotIn(base.FULL_SERIAL, repr(info))
        self.assertEqual(info["backend_exposure_limits_us"], {"min": 16.0, "max": 1000000.0})
        self.assertEqual(info["approved_exposure_limits_us"], {"min": 1000.0, "max": 5000.0})  # not widened
        self.host.shutdown()

    async def test_camera_info_before_open_does_not_initialize_device(self):
        host = self.make_host()
        self.approve()
        async with self.connect() as client:
            text = await self.call_error(client, "camera_info")
            self.assertIn("requires an open camera", text)
            self.assertIn("call open() first", text)
            self.assertEqual(self.fake.calls, [])
            self.assertEqual(host.status()["session_state"], "aborted")
            await self.call_error(client, "camera_open")
        self.assertEqual(self.fake.calls, [])
        host.shutdown()
        self.assertEqual(self.fake.calls, [])

    async def test_exposure_outside_approved_range_never_written(self):
        host = self.make_host()
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            mark = len(self.fake.calls)
            for value in (999.0, 5000.5, 16.0, 1000000.0, -1000.0):
                with self.subTest(value=value):
                    text = await self.call_error(client, "camera_set_exposure", {"exposure_us": value})
                    self.assertIn("Not authorized", text)
            await self.call_error(client, "camera_set_exposure", {"exposure_us": "1000"})
            self.assertEqual(self.fake.calls[mark:], [])  # rejected before any SDK call
            self.assertEqual(host.status()["session_state"], "active")
        host.shutdown()
        self.assertEqual(self.fake.writes(), [])
        self.assert_released()

    async def test_unapproved_tool_rejected(self):
        host = self.make_host(envelope=_envelope(allowed_tools=("camera_open", "camera_close")))
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            mark = len(self.fake.calls)
            for name, args in (("camera_measure", {}), ("camera_set_exposure", {"exposure_us": 1000.0}),
                               ("camera_info", {})):
                with self.subTest(name=name):
                    self.assertIn("not authorized", await self.call_error(client, name, args))
            for name in ("camera_reset", "approve_session", "resume", "shutdown", "load_sdk"):
                with self.subTest(name=name):
                    await self.call_error(client, name, {"decision": _approve(self.envelope)})
            self.assertEqual(self.fake.calls[mark:], [])
        host.shutdown()
        self.assertNotIn("IMV_GetFrame", self.fake.names())

    async def test_measurement_budget_enforced(self):
        host = self.make_host()
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_measure")
            await self.call_ok(client, "camera_measure")
            self.assertIn("measurement budget exhausted", await self.call_error(client, "camera_measure"))
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 2)
        host.shutdown()
        self.assert_released()

    async def test_tool_call_budget_enforced(self):
        host = self.make_host()
        self.approve()
        async with self.connect() as client:
            for name, args in (("camera_open", {}), ("camera_set_exposure", {"exposure_us": 1000.0}),
                               ("camera_measure", {}), ("camera_set_exposure", {"exposure_us": 5000.0}),
                               ("camera_measure", {}), ("camera_close", {}), ("camera_open", {})):
                await self.call_ok(client, name, args)
            self.assertEqual(host.status()["session_state"], "exhausted")
            mark = len(self.fake.calls)
            text = await self.call_error(client, "camera_close")
            self.assertIn("exhausted", text)
            self.assertEqual(self.fake.calls[mark:], [])
        # The 7th call reopened the camera; the budget cannot block the trusted shutdown.
        self.assertTrue(self.fake.is_open)
        host.shutdown()
        self.assertFalse(self.fake.is_open)
        self.assertEqual(self.fake.handles_alive, 0)

    async def test_session_expiry_enforced(self):
        host = self.make_host()
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            self.clock.now = 120.0
            self.assertIn("expired", await self.call_error(client, "camera_set_exposure", {"exposure_us": 1000.0}))
            await self.call_error(client, "camera_measure")
        self.assertEqual(host.status()["session_state"], "expired")
        self.assertEqual(self.fake.writes(), [])
        self.assertNotIn("IMV_GetFrame", self.fake.names())
        host.shutdown()
        self.assert_released()

    async def test_every_call_goes_through_runtime_gate(self):
        host = self.make_host()
        self.approve()
        runtime = host._runtime
        with mock.patch.object(runtime._gate, "begin_action", wraps=runtime._gate.begin_action) as gate_spy:
            async with self.connect() as client:
                await self.call_ok(client, "camera_open")
                await self.call_error(client, "camera_set_exposure", {"exposure_us": 6000.0})
                await self.call_ok(client, "camera_measure")
        self.assertEqual([call.args[1] for call in gate_spy.call_args_list],
                         ["camera_open", "camera_set_exposure", "camera_measure"])
        host.shutdown()


# 22-24. Failure handling, shutdown and cleanup visibility.
class FailureTests(PhysicalHostTestCase):
    async def test_failed_capture_aborts_without_synthetic_fallback(self):
        host = self.make_host(codes={"IMV_GetFrame": -119})
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_set_exposure", {"exposure_us": 1000.0})
            text = await self.call_error(client, "camera_measure")
            self.assertIn("GET_FRAME_FAILED", text)
            self.assertIn("not retried", text)
            self.assertEqual(host.status()["session_state"], "aborted")
            self.assertIn("aborted", await self.call_error(client, "camera_measure"))
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)
        # The runtime's automatic shutdown already restored 5000 us and closed the device.
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
        self.assert_released()
        mark = len(self.fake.calls)
        self.assertEqual(host.shutdown()["runtime"]["backend_closed"], True)
        self.assertEqual(len(self.fake.calls), mark)

    async def test_context_manager_shuts_down_on_error_and_interrupt(self):
        for exc in (RuntimeError("model loop failed"), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__):
                with self.assertRaises(type(exc)):
                    with self.make_host() as host:
                        self.approve()
                        async with self.connect() as client:
                            await self.call_ok(client, "camera_open")
                            await self.call_ok(client, "camera_set_exposure", {"exposure_us": 1000.0})
                        raise exc
                self.assertEqual(host.status()["host_state"], ash.HOST_SHUT_DOWN)
                self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
                self.assert_released()
                self.assertIsNone(phys._device_owner)

    async def test_early_finish_and_rejected_host_shut_down_by_context(self):
        with self.make_host() as host:
            self.approve()
            async with self.connect() as client:
                await self.call_ok(client, "camera_open")
        self.assert_released()
        with self.make_host() as host:
            host.start("t-1")
            host.resume("no")
        self.assertEqual(self.fake.calls, [])
        self.assertIsNone(phys._device_owner)

    async def test_close_failure_visible_and_never_passes(self):
        host = self.make_host(codes={"IMV_Close": -101})
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_set_exposure", {"exposure_us": 1000.0})
            text = await self.call_error(client, "camera_close")
        self.assertIn("IMV_Close FAILED with code -101", text)
        self.assertIn("cleanup also failed", text)
        for _ in range(2):
            with self.assertRaises(CleanupError) as ctx:
                host.shutdown()
            self.assertIn("CLEANUP_FAILED", str(ctx.exception))
        self.assertEqual(self.fake.counts["IMV_Close"], 1)  # native shutdown never repeated
        self.assertEqual(self.fake.counts["IMV_DestroyHandle"], 1)
        self.assertTrue(host.status()["owns_device"])  # device state unknown: stays claimed
        with self.assertRaises(HostError):
            phys.PhysicalApprovedSessionHost(_envelope(session_id="next"), base.FakeSdk().ns)

    async def test_restore_failure_visible_at_host_shutdown(self):
        host = self.make_host(codes={WRITE: [0, -101]})
        self.approve()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_set_exposure", {"exposure_us": 1000.0})
            await self.call_ok(client, "camera_measure")
        with self.assertRaises(CleanupError) as ctx:
            host.shutdown()
        self.assertIn("RESTORE_FAILED", str(ctx.exception))
        self.assertIn(ic.RESTORE_WRITE_FAILED, str(ctx.exception))
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])  # one restoration attempt, not retried
        self.assertEqual(self.fake.handles_alive, 0)
        self.assertTrue(host.status()["owns_device"])

    async def test_open_failure_is_not_retried(self):
        host = self.make_host(enums={"ExposureAuto": b"Continuous"})
        self.approve()
        async with self.connect() as client:
            self.assertIn("EXPOSURE_AUTO_NOT_OFF", await self.call_error(client, "camera_open"))
            await self.call_error(client, "camera_open")
        self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)
        self.assertEqual(self.fake.writes(), [])
        host.shutdown()
        self.assertEqual(self.fake.handles_alive, 0)


# 26, 27. Isolation.
class IsolationTests(unittest.TestCase):
    def test_imports_limited_to_existing_components(self):
        imported = set()
        for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"__future__", "threading", "time", "collections", "typing", "irayple_camera",
                                    "approved_session_host", "hitl_approval", "session_policy"})

    def test_public_api(self):
        public = {name for name in dir(phys.PhysicalApprovedSessionHost) if not name.startswith("_")}
        self.assertEqual(public, {"start", "pending", "resume", "get_mcp_server", "shutdown", "status"})
        for inherited in ("start", "pending", "resume", "get_mcp_server"):
            self.assertIs(getattr(phys.PhysicalApprovedSessionHost, inherited),
                          getattr(OfflineApprovedSessionHost, inherited))


if __name__ == "__main__":
    unittest.main()
