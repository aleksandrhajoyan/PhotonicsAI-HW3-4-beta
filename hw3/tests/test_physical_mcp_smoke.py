"""Offline tests for hw3/physical_mcp_smoke.py.

main() runs end to end against the REAL LangGraph approval graph, the REAL
SessionGate, the REAL PhysicalApprovedSessionHost, AuthorizedCameraRuntime,
authorized MCP server and official MCP Client, and the REAL IRaypleCamera; only
the vendor SDK binding is the fake of test_irayple_camera, injected through the
patched irayple_camera.preload_vendor_sdk. Terminal input is SIMULATED by this
harness (the "operator" reads the session_id from the printed request). No
camera, vendor DLL, network, API key or LLM is involved.
"""

import ast
import asyncio
import importlib
import io
import os
import re
import socket
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import anyio.to_thread
import numpy as np
from mcp import Client

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import approved_session_host as ash  # noqa: E402
import authorized_camera as ac  # noqa: E402
import irayple_camera as ic  # noqa: E402
import physical_approved_session_host as phys  # noqa: E402
import physical_mcp_smoke as smoke  # noqa: E402
import test_irayple_camera as base  # noqa: E402
from session_policy import SessionGate, session_approval  # noqa: E402

MODULE_PATH = HW3_DIR / "physical_mcp_smoke.py"
FLAG = "--physical-mcp-check"
N_PIXELS = smoke.EXPECTED_WIDTH * smoke.EXPECTED_HEIGHT
# Full-size Mono8 frames. Frame 2 contains 255 DN pixels, so its saturation fraction is > 0.
FRAME_1 = (np.arange(N_PIXELS) % 41).astype(np.uint8).tobytes()
FRAME_2 = (np.arange(N_PIXELS) % 256).astype(np.uint8).tobytes()
EXPECTED_TOOLS = [tool for tool, _ in smoke.SEQUENCE]
WRITE = base.WRITE
SESSION_RE = re.compile(r"session_id\s+:\s+(\S+)")


class TTY(io.StringIO):
    def isatty(self):
        return True


class RecordingFakeSdk(base.FakeSdk):
    """The shared fake binding, additionally recording the thread and loop of every SDK call."""

    def __init__(self, **kwargs):
        self.contexts = []
        kwargs.setdefault("width", smoke.EXPECTED_WIDTH)
        kwargs.setdefault("height", smoke.EXPECTED_HEIGHT)
        kwargs.setdefault("payloads", (FRAME_1, FRAME_2))
        super().__init__(**kwargs)

    def _call(self, name, *args):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        self.contexts.append((name, threading.get_ident(), loop))
        return super()._call(name, *args)


class Operator:
    """Simulated terminal operator: answers prompts in order, may read the screen."""

    def __init__(self, test, answers):
        self.test = test
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        self.test.events.append(("input", len(self.prompts)))
        if not self.answers:
            raise EOFError
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer(self.test.output()) if callable(answer) else answer


def approve_shown_session(screen):
    """What a careful human types: APPROVE plus the session_id displayed in the request."""
    return f"APPROVE {SESSION_RE.findall(screen)[-1]}"


class SmokeTestCase(unittest.TestCase):
    def setUp(self):
        guard = AssertionError("real SDK, DLL or directory access attempted in an offline test")
        for target in (
            mock.patch("ctypes.WinDLL", side_effect=guard, create=True),
            mock.patch("ctypes.CDLL", side_effect=guard),
            mock.patch("os.add_dll_directory", side_effect=guard, create=True),
            mock.patch("os.chdir", side_effect=AssertionError("working directory changed")),
            mock.patch.object(ic.discovery, "load_vendor_sdk", side_effect=guard),
            mock.patch.object(ash, "FakeCamera", side_effect=AssertionError("FakeCamera (synthetic) used")),
            # Name resolution and outbound connections fail loudly; asyncio's own loopback
            # self-pipe (socket.socketpair) is not a network request and stays usable.
            mock.patch.object(socket, "getaddrinfo", side_effect=AssertionError("network access")),
            mock.patch.object(socket, "create_connection", side_effect=AssertionError("network access")),
            mock.patch.object(anyio.to_thread, "run_sync",
                              side_effect=AssertionError("MCP dispatched to a worker thread")),
        ):
            target.start()
            self.addCleanup(target.stop)
        self.addCleanup(self._reset_process_state)
        self._reset_process_state()
        self.events = []
        self.fake = None
        self.preload = mock.patch.object(ic, "preload_vendor_sdk", side_effect=self._preload)
        self.preload_mock = self.preload.start()
        self.addCleanup(self.preload.stop)
        self.stdout = TTY()

    def _reset_process_state(self):
        phys._device_owner = None
        phys._preloaded_sdk = None

    def tearDown(self):
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)
        if self.fake is not None:
            self.assertEqual(self.fake.violations, [])

    def make_fake(self, **kwargs):
        self.fake = RecordingFakeSdk(**kwargs)
        return self.fake

    def _preload(self):
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False
        self.events.append(("preload", running, threading.get_ident()))
        if self.fake is None:
            self.make_fake()
        return self.fake.ns

    def output(self):
        return self.stdout.getvalue()

    def run_cli(self, answers=(smoke.CONFIRMATION_PHRASE, approve_shown_session), argv=(FLAG,),
                stdin_tty=True, stdout_tty=True):
        self.operator = Operator(self, answers)
        stdin = TTY() if stdin_tty else io.StringIO()
        if not stdout_tty:
            self.stdout = io.StringIO()
        with mock.patch("sys.stdin", stdin), mock.patch("sys.stdout", self.stdout), \
                mock.patch("builtins.input", side_effect=self.operator):
            return smoke.main(list(argv))

    def session_id(self):
        return SESSION_RE.findall(self.output())[-1]

    def spy_execute(self):
        original = ac.AuthorizedCameraRuntime.execute
        spy = mock.patch.object(ac.AuthorizedCameraRuntime, "execute", autospec=True, side_effect=original)
        mocked = spy.start()
        self.addCleanup(spy.stop)
        return mocked

    def executed_tools(self, spy):
        return [call.args[2] for call in spy.call_args_list]

    def assert_no_camera_access(self):
        self.assertEqual(self.fake.calls if self.fake else [], [])

    def assert_released(self):
        self.assertFalse(self.fake.is_open)
        self.assertFalse(self.fake.is_grabbing)
        self.assertEqual(self.fake.handles_alive, 0)
        self.assertEqual(self.fake.frames_outstanding, {})
        for name in ("IMV_Close", "IMV_DestroyHandle"):
            self.assertEqual(self.fake.counts.get(name, 0), 1, name)  # never duplicated
        self.assertIsNone(phys._device_owner)


# 1, 32, 33. Import safety and isolation.
class ImportAndIsolationTests(unittest.TestCase):
    def test_import_loads_no_sdk_and_touches_no_camera(self):
        cwd = os.getcwd()
        with mock.patch.object(ic.discovery, "load_vendor_sdk") as loader, \
                mock.patch.object(ic, "preload_vendor_sdk") as preload, \
                mock.patch.object(ic, "IRaypleCamera") as backend, \
                mock.patch.object(phys, "PhysicalApprovedSessionHost") as host, \
                mock.patch("ctypes.WinDLL", create=True) as win_dll, \
                mock.patch("os.add_dll_directory", create=True) as add_dir, \
                mock.patch("os.chdir") as chdir:
            importlib.reload(smoke)
        for patched in (loader, preload, backend, host, win_dll, add_dir, chdir):
            patched.assert_not_called()
        self.assertEqual(os.getcwd(), cwd)
        self.assertNotIn("IMVApi", sys.modules)

    def test_imports_limited_no_llm_network_or_vendor(self):
        imported = set()
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"__future__", "argparse", "asyncio", "math", "secrets", "sys",
                                    "threading", "time", "typing", "mcp", "physical_approved_session_host",
                                    "session_policy"})
        source = MODULE_PATH.read_text(encoding="utf-8").lower()
        for forbidden in ("openai", "api_key", "environ", "getenv", "fake_camera", "llm_decision"):
            self.assertNotIn(forbidden, source)
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for forbidden in ("chdir", "Thread", "to_thread", "run_in_executor", "preload_vendor_sdk",
                          "FakeCamera", "environ"):
            self.assertNotIn(forbidden, names)

    def test_cli_never_calls_backend_or_runtime_directly(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for name in ("execute", "capture_frame", "set_exposure_us", "IRaypleCamera", "AuthorizedCameraRuntime",
                     "create_authorized_server", "submit_decision", "begin_action", "_runtime", "_camera"):
            self.assertNotIn(name, attributes)


# 2-6. Entry-point refusals and the first confirmation.
class EntryPointTests(SmokeTestCase):
    def test_missing_flag_refuses_without_sdk(self):
        for argv in ((), ("--check-once",)):
            with self.subTest(argv=argv):
                code = self.run_cli(argv=argv)
                self.assertEqual(code, smoke.EXIT_REFUSED)
                self.assertEqual(self.operator.prompts, [])
        self.preload_mock.assert_not_called()
        self.assertIsNone(phys._device_owner)

    def test_noninteractive_refuses_without_sdk(self):
        for stdin_tty, stdout_tty in ((False, True), (True, False), (False, False)):
            with self.subTest(stdin_tty=stdin_tty, stdout_tty=stdout_tty):
                code = self.run_cli(stdin_tty=stdin_tty, stdout_tty=stdout_tty)
                self.assertEqual(code, smoke.EXIT_NOT_INTERACTIVE)
                self.assertEqual(self.operator.prompts, [])
        self.preload_mock.assert_not_called()

    def test_environment_cannot_bypass_confirmation(self):
        env = {"LUMEN_APPROVE": "1", "PHYSICAL_MCP_CHECK": smoke.CONFIRMATION_PHRASE, "CI": "1"}
        with mock.patch.dict(os.environ, env):
            code = self.run_cli(stdin_tty=False)
            self.assertEqual(code, smoke.EXIT_NOT_INTERACTIVE)
            code = self.run_cli(answers=())
        self.assertEqual(code, smoke.EXIT_REFUSED)
        self.preload_mock.assert_not_called()

    def test_wrong_first_confirmation_eof_or_interrupt_refuses(self):
        for answer in ("begin physical mcp check", "BEGIN PHYSICAL MCP CHECK ", "yes", "",
                       EOFError(), KeyboardInterrupt()):
            with self.subTest(answer=repr(answer)):
                code = self.run_cli(answers=(answer,))
                self.assertEqual(code, smoke.EXIT_REFUSED)
                self.assertIn("REFUSED", self.output())
                self.assertEqual(len(self.operator.prompts), 1)
        self.preload_mock.assert_not_called()
        self.assertIsNone(self.fake)

    def test_warning_shown_before_first_confirmation(self):
        self.run_cli(answers=("no",))
        self.assertIn("PHYSICAL camera", self.output())
        self.assertIn("temporarily changes ExposureTime", self.output())

    def test_first_confirmation_alone_does_not_approve_hardware(self):
        code = self.run_cli(answers=(smoke.CONFIRMATION_PHRASE,))  # EOF at the approval prompt
        self.assertEqual(code, smoke.EXIT_APPROVAL_REFUSED)
        self.assertEqual(len(self.operator.prompts), 2)
        self.assert_no_camera_access()
        self.assertIn("approval            : refused", self.output())
        self.assertIsNone(phys._device_owner)

    def test_sdk_preloaded_once_only_after_first_confirmation_and_before_event_loop(self):
        real_run = asyncio.run

        def run(coro, *args, **kwargs):
            self.events.append(("event_loop", threading.get_ident()))
            return real_run(coro, *args, **kwargs)

        with mock.patch.object(smoke.asyncio, "run", side_effect=run):
            code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        kinds = [event[0] for event in self.events]
        self.assertEqual(kinds, ["input", "preload", "input", "event_loop"])
        preload, loop = self.events[1], self.events[3]
        self.assertIs(preload[1], False)  # no running event loop during preload
        self.assertEqual(preload[2], loop[1])  # same (main) thread
        self.preload_mock.assert_called_once_with()

    def test_sdk_load_failure_fails_without_camera(self):
        self.preload_mock.side_effect = ic.discovery.DiscoveryError("SDK_NOT_FOUND", "missing")
        code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_FAILED)
        self.assertIn("SDK_LOAD_FAILED", self.output())
        self.assertIsNone(phys._device_owner)


# 8-15. Envelope and human approval.
class ApprovalTests(SmokeTestCase):
    def test_trusted_envelope_has_exact_limits(self):
        envelope = smoke.build_envelope("lumen-phys-test")
        self.assertEqual(envelope.allowed_tools,
                         ("camera_open", "camera_set_exposure", "camera_measure", "camera_close"))
        self.assertNotIn("camera_info", envelope.allowed_tools)
        self.assertEqual((envelope.min_exposure_us, envelope.max_exposure_us), (1000.0, 5000.0))
        self.assertEqual((envelope.max_measurements, envelope.max_tool_calls), (2, 7))
        self.assertEqual(envelope.max_duration_seconds, 180.0)
        phys.check_physical_envelope(envelope)  # within the host's physical limits
        for tool, arguments in smoke.SEQUENCE:
            self.assertIn(tool, envelope.allowed_tools)
            if "exposure_us" in arguments:
                self.assertTrue(envelope.min_exposure_us <= arguments["exposure_us"] <= envelope.max_exposure_us)

    def test_fresh_unpredictable_session_ids(self):
        ids = {smoke.new_session_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        for value in ids:
            self.assertRegex(value, r"^lumen-phys-[0-9a-f]{32}$")
        with mock.patch.object(smoke.secrets, "token_hex", return_value="ab" * 16) as token:
            self.assertEqual(smoke.new_session_id(), "lumen-phys-" + "ab" * 16)
        token.assert_called_once_with(16)

    def test_each_run_uses_a_new_session(self):
        self.run_cli(answers=(smoke.CONFIRMATION_PHRASE, "DENY"))
        first = self.session_id()
        self.stdout = TTY()
        self.run_cli(answers=(smoke.CONFIRMATION_PHRASE, "DENY"))
        self.assertNotEqual(first, self.session_id())

    def test_complete_envelope_and_hash_shown_before_approval(self):
        screens = []

        def capture(screen):
            screens.append(screen)
            self.assertEqual(self.fake.calls, [])  # nothing touched the camera before approval
            return approve_shown_session(screen)

        code = self.run_cli(answers=(smoke.CONFIRMATION_PHRASE, capture))
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        envelope = smoke.build_envelope(self.session_id())
        shown = screens[0]
        self.assertIn(envelope.canonical_json(), shown)
        self.assertIn(envelope.sha256(), shown)
        self.assertIn(f"APPROVE {envelope.session_id}", shown)
        for value in ("camera_open, camera_set_exposure, camera_measure, camera_close",
                      "[1000.0, 5000.0]", "max measurements    : 2", "max tool calls      : 7",
                      "max duration (s)    : 180.0"):
            self.assertIn(value, shown)
        self.assertIn("pending in LangGraph", shown)

    def test_hitl_requires_second_human_interaction(self):
        code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        self.assertEqual(len(self.operator.prompts), 2)
        self.assertEqual(self.operator.prompts[1], "> ")

    def assert_refused_without_camera(self, answer):
        code = self.run_cli(answers=(smoke.CONFIRMATION_PHRASE, answer))
        self.assertEqual(code, smoke.EXIT_APPROVAL_REFUSED, self.output())
        self.assert_no_camera_access()
        self.assertIn("approval            : refused", self.output())
        self.assertIn("PHYSICAL_MCP_CHECK: FAIL", self.output())
        self.assertIsNone(phys._device_owner)  # the refused host was still shut down

    def test_wrong_session_id_fails_closed(self):
        self.assert_refused_without_camera("APPROVE lumen-phys-" + "0" * 32)
        self.assert_refused_without_camera(lambda screen: approve_shown_session(screen)[:-1])

    def test_rejection_prevents_camera_opening(self):
        self.assert_refused_without_camera("DENY")

    def test_malformed_approval_prevents_camera_opening(self):
        for answer in ("APPROVE", "approve", "yes", "", lambda s: approve_shown_session(s).lower(),
                       lambda s: approve_shown_session(s) + " ", lambda s: " " + approve_shown_session(s),
                       lambda s: approve_shown_session(s).replace("APPROVE ", "APPROVE  "),
                       lambda s: approve_shown_session(s) + "\nDENY"):
            with self.subTest(answer=answer if isinstance(answer, str) else "derived"):
                self.assert_refused_without_camera(answer)

    def test_expired_approval_prevents_camera_opening(self):
        ticks = iter([100.0, 100.0 + smoke.APPROVAL_TIMEOUT_SECONDS + 0.5])
        with mock.patch.object(smoke, "_clock", side_effect=lambda: next(ticks)):
            self.assert_refused_without_camera(approve_shown_session)
        self.assertIn("expired", self.output())

    def test_interrupt_at_approval_prompt_shuts_down_without_camera(self):
        code = self.run_cli(answers=(smoke.CONFIRMATION_PHRASE, KeyboardInterrupt()))
        self.assertEqual(code, smoke.EXIT_INTERRUPTED)
        self.assert_no_camera_access()
        self.assertIsNone(phys._device_owner)

    def test_graph_and_execution_gate_both_validate_the_decision(self):
        original = SessionGate.submit_decision
        with mock.patch.object(SessionGate, "submit_decision", autospec=True, side_effect=original) as spy:
            code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        self.assertEqual(spy.call_count, 2)  # LangGraph node, then the execution-side gate
        self.assertIsNot(spy.call_args_list[0].args[0], spy.call_args_list[1].args[0])
        decision = spy.call_args_list[0].args[1]
        self.assertEqual(decision, session_approval(self.session_id(), smoke.build_envelope(
            self.session_id()).sha256(), True))

    def test_cli_cannot_forge_a_decision_for_another_session(self):
        # Even a correctly typed phrase is refused if the submitted decision names another session.
        forged = lambda sid, digest, approved: session_approval("lumen-phys-other", digest, approved)  # noqa: E731
        with mock.patch.object(smoke, "session_approval", side_effect=forged):
            self.assert_refused_without_camera(approve_shown_session)
        self.assertIn("approval names session", self.output())


# 16-22, 29-31. The six-tool MCP sequence on the fake SDK.
class SequenceTests(SmokeTestCase):
    def test_pass_through_protected_mcp_server(self):
        execute = self.spy_execute()
        servers = []
        real_get = phys.PhysicalApprovedSessionHost.get_mcp_server
        clients = []
        real_client = Client

        def get_server(host):
            servers.append(real_get(host))
            return servers[-1]

        def client(server, **kwargs):
            clients.append((server, kwargs))
            return real_client(server, **kwargs)

        with mock.patch.object(phys.PhysicalApprovedSessionHost, "get_mcp_server", autospec=True,
                               side_effect=get_server), \
                mock.patch.object(smoke, "Client", side_effect=client):
            code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        self.assertEqual(len(servers), 1)
        self.assertEqual([c[0] for c in clients], servers)  # the Client talks to the authorized server
        self.assertEqual(clients[0][1]["mode"], "legacy")
        self.assertEqual(self.executed_tools(execute), EXPECTED_TOOLS)  # exactly six, in order
        self.assertNotIn("camera_info", self.executed_tools(execute))
        self.assertEqual([call.args[3] for call in execute.call_args_list],
                         [{}, {"exposure_us": 1000.0}, {}, {"exposure_us": 5000.0}, {}, {}])
        self.assertIn("PHYSICAL_MCP_CHECK: PASS", self.output())
        self.assertIn("MCP tool calls      : 6 issued", self.output())

    def test_two_hardware_measurements_with_backend_frame_ids(self):
        for block_ids in (None, [1, 1], [7, 7]):  # the native blockId repeats
            with self.subTest(block_ids=block_ids):
                self._reset_process_state()
                self.make_fake(block_ids=block_ids)
                self.stdout = TTY()
                code = self.run_cli()
                self.assertEqual(code, smoke.EXIT_PASS, self.output())
                self.assertEqual(self.fake.counts["IMV_GetFrame"], 2)
                self.assertEqual(self.fake.exposure_at_frame, [1000.0, 5000.0])
                out = self.output()
                self.assertIn("measurement 1: source=hardware frame_id=1 exposure=1000.0 us 1920x1200 Mono8", out)
                self.assertIn("measurement 2: source=hardware frame_id=2 exposure=5000.0 us 1920x1200 Mono8", out)

    def test_reported_statistics_come_from_the_frames(self):
        code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        first = np.frombuffer(FRAME_1, np.uint8)
        second = np.frombuffer(FRAME_2, np.uint8)
        out = self.output()
        self.assertIn(f"min={first.min()} max={first.max()} DN mean={first.mean():.4f} DN "
                      f"saturation_fraction={0:.6f}", out)
        self.assertIn(f"min={second.min()} max={second.max()} DN mean={second.mean():.4f} DN "
                      f"saturation_fraction={np.count_nonzero(second == 255) / N_PIXELS:.6f}", out)
        self.assertIn(f"mean DN difference  : {second.mean() - first.mean():+.4f} DN", out)
        self.assertIn(f"mean DN ratio       : {second.mean() / first.mean():.4f} (exposure ratio 5)", out)
        self.assertIn("Not a linearity test and not a calibration", out)

    def test_original_exposure_restored_and_verified_through_fake_sdk(self):
        self.make_fake(exposure=(3000.0, 16.0, 1000000.0))
        code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0, 3000.0])
        self.assertEqual(self.fake.current, 3000.0)
        self.assertEqual(self.fake.names()[-4:], [WRITE, base.READ_VALUE, "IMV_Close", "IMV_DestroyHandle"])
        self.assertIn("RESTORED AND VERIFIED by IRaypleCamera.close() during camera_close", self.output())
        self.assert_released()

    def test_native_close_and_destroy_not_duplicated(self):
        code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        self.assert_released()
        self.assertIn("session state       : active before shutdown, closed after", self.output())

    def test_tool_handlers_run_on_the_preloading_thread_inside_the_event_loop(self):
        code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        preload_thread = self.events[1][2]
        self.assertEqual(preload_thread, threading.main_thread().ident)
        self.assertTrue(self.fake.contexts)
        for name, thread, loop in self.fake.contexts:
            self.assertEqual(thread, preload_thread, name)  # no worker-thread SDK call
            self.assertIsNotNone(loop, name)  # inside the single asyncio.run() loop
        self.assertEqual(len({id(loop) for _, _, loop in self.fake.contexts}), 1)
        anyio.to_thread.run_sync.assert_not_called()

    def test_os_chdir_never_called_while_serving(self):
        code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_PASS, self.output())
        os.chdir.assert_not_called()

    def test_out_of_policy_exposure_denied_by_real_gate(self):
        for bad in (999.0, 5000.5, 16.0):
            with self.subTest(exposure=bad):
                self._reset_process_state()
                self.make_fake()
                self.stdout = TTY()
                sequence = list(smoke.SEQUENCE)
                sequence[1] = ("camera_set_exposure", {"exposure_us": bad})
                with mock.patch.object(smoke, "SEQUENCE", tuple(sequence)):
                    code = self.run_cli()
                self.assertEqual(code, smoke.EXIT_FAILED, self.output())
                self.assertIn("Not authorized", self.output())
                self.assertEqual(self.fake.writes(), [])
                self.assertNotIn("IMV_GetFrame", self.fake.names())
                self.assert_released()

    def test_measurement_budget_enforced_by_real_gate(self):
        sequence = list(smoke.SEQUENCE)
        sequence.insert(5, ("camera_measure", {}))
        with mock.patch.object(smoke, "SEQUENCE", tuple(sequence)):
            code = self.run_cli()
        self.assertEqual(code, smoke.EXIT_FAILED)
        self.assertIn("measurement budget exhausted", self.output())
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 2)
        self.assert_released()


# 23-28. Failures, provenance and shutdown on every exit path.
class FailureTests(SmokeTestCase):
    def assert_failed_and_shut_down(self, code, expected_code=smoke.EXIT_FAILED):
        self.assertEqual(code, expected_code, self.output())
        self.assertIn("PHYSICAL_MCP_CHECK: FAIL", self.output())
        self.assertNotIn("PHYSICAL_MCP_CHECK: PASS", self.output())

    def test_mcp_error_payload_is_a_failure_and_aborts_the_rest(self):
        execute = self.spy_execute()
        self.make_fake(codes={"IMV_GetFrame": -119})
        code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertEqual(self.executed_tools(execute), EXPECTED_TOOLS[:3])
        self.assertIn("TOOL_ERROR in camera_measure", self.output())
        self.assertIn("GET_FRAME_FAILED", self.output())
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)  # never retried
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])  # restored by abort cleanup
        self.assert_released()

    def test_is_error_flag_must_be_exactly_false(self):
        real_call = Client.call_tool

        async def call_tool(client, name, arguments=None, *args, **kwargs):
            result = await real_call(client, name, arguments, *args, **kwargs)
            if name == "camera_set_exposure":
                result.is_error = None
            return result

        with mock.patch.object(Client, "call_tool", autospec=True, side_effect=call_tool):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertIn("TOOL_ERROR in camera_set_exposure", self.output())
        self.assertEqual(self.fake.counts.get("IMV_GetFrame", 0), 0)
        self.assert_released()

    def test_open_failure_aborts_without_retry(self):
        execute = self.spy_execute()
        self.make_fake(enums={"ExposureAuto": b"Continuous"})
        code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertEqual(self.executed_tools(execute), ["camera_open"])
        self.assertEqual(self.fake.counts["IMV_EnumDevices"], 1)
        self.assertEqual(self.fake.writes(), [])
        self.assertEqual(self.fake.handles_alive, 0)

    def test_client_exception_triggers_shutdown(self):
        execute = self.spy_execute()
        real_call = Client.call_tool

        async def call_tool(client, name, arguments=None, *args, **kwargs):
            if len(execute.call_args_list) == 2:
                raise ConnectionError("transport lost")
            return await real_call(client, name, arguments, *args, **kwargs)

        with mock.patch.object(Client, "call_tool", autospec=True, side_effect=call_tool):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertIn("MCP_CLIENT_ERROR in camera_measure", self.output())
        self.assertEqual(self.executed_tools(execute), EXPECTED_TOOLS[:2])
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])  # trusted host shutdown restored
        self.assert_released()

    def test_keyboard_interrupt_triggers_shutdown(self):
        execute = self.spy_execute()
        real_call = Client.call_tool

        async def call_tool(client, name, arguments=None, *args, **kwargs):
            if len(execute.call_args_list) == 3:
                raise KeyboardInterrupt
            return await real_call(client, name, arguments, *args, **kwargs)

        with mock.patch.object(Client, "call_tool", autospec=True, side_effect=call_tool):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code, smoke.EXIT_INTERRUPTED)
        self.assertIn("INTERRUPTED", self.output())
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
        self.assert_released()

    def test_invalid_provenance_triggers_shutdown_without_fallback(self):
        real = ac.AuthorizedCameraRuntime._camera_measure

        def lying_measure(runtime, grant):
            result = real(runtime, grant)
            result.update(source="synthetic", synthetic=True)
            return result

        execute = self.spy_execute()
        with mock.patch.object(ac.AuthorizedCameraRuntime, "_camera_measure", autospec=True,
                               side_effect=lying_measure):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertIn("VALIDATION_FAILED in camera_measure: provenance is source='synthetic'", self.output())
        self.assertEqual(self.executed_tools(execute), EXPECTED_TOOLS[:3])
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
        self.assert_released()

    def test_backend_provenance_mismatch_rejected_by_runtime(self):
        with mock.patch.object(ic, "SOURCE_HARDWARE", "synthetic"):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertIn("TOOL_ERROR in camera_measure", self.output())
        self.assert_released()

    def test_measurement_validation_rejects_bad_fields(self):
        good = {"source": "hardware", "synthetic": False, "frame_id": 1, "exposure_us": 1000.0,
                "bit_depth": 8, "width": 1920, "height": 1200,
                "metrics": {"width": 1920, "height": 1200, "bit_depth": 8, "full_scale_dn": 255,
                            "min_dn": 0, "max_dn": 40, "mean_dn": 20.0, "saturation_fraction": 0.0}}
        self.assertEqual(smoke.measurement_problem(good, 1000.0, 1), "")
        bad = [dict(good, frame_id=2), dict(good, frame_id=True), dict(good, exposure_us=5000.0),
               dict(good, source="synthetic"), dict(good, synthetic=True), dict(good, width=4),
               dict(good, metrics=dict(good["metrics"], mean_dn=float("nan"))),
               dict(good, metrics=dict(good["metrics"], mean_dn=99.0)),
               dict(good, metrics=dict(good["metrics"], saturation_fraction=0.5)),
               dict(good, metrics=dict(good["metrics"], min_dn=50)), dict(good, metrics=None)]
        for content in bad:
            with self.subTest(content=content):
                self.assertNotEqual(smoke.measurement_problem(content, 1000.0, 1), "")

    def test_wrong_frame_id_fails(self):
        real = ac.AuthorizedCameraRuntime._camera_measure

        def renumbered(runtime, grant):
            return dict(real(runtime, grant), frame_id=1)  # e.g. a native blockId leaking through

        with mock.patch.object(ac.AuthorizedCameraRuntime, "_camera_measure", autospec=True,
                               side_effect=renumbered):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertIn("frame_id 1, expected 2", self.output())
        self.assert_released()

    def test_session_time_expiry_stops_and_shuts_down(self):
        clock = [0.0]
        real_host = phys.PhysicalApprovedSessionHost

        def host_with_clock(envelope, sdk):
            return real_host(envelope, sdk, clock=lambda: clock[0])

        self.make_fake(on_get_frame=lambda: clock.__setitem__(0, 181.0))
        with mock.patch.object(smoke.phys, "PhysicalApprovedSessionHost", side_effect=host_with_clock):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertIn("expired", self.output())
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])  # second write is the restoration
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)
        self.assert_released()

    def test_native_close_failure_prevents_pass(self):
        self.make_fake(codes={"IMV_Close": -101})
        code = self.run_cli()
        self.assert_failed_and_shut_down(code, smoke.EXIT_CLEANUP_FAILED)
        out = self.output()
        self.assertIn("TOOL_ERROR in camera_close", out)
        self.assertIn("CLEANUP_ERROR", out)
        self.assertIn("MV Viewer", out)
        self.assertIn("ExposureTime", out)
        self.assertIn("device ownership    : STILL CLAIMED", out)
        self.assertEqual(self.fake.counts["IMV_Close"], 1)  # not retried
        self.assertEqual(self.fake.counts["IMV_DestroyHandle"], 1)

    def test_restoration_failure_prevents_pass(self):
        self.make_fake(codes={WRITE: [0, 0, -101]})
        code = self.run_cli()
        self.assert_failed_and_shut_down(code, smoke.EXIT_CLEANUP_FAILED)
        out = self.output()
        self.assertIn("RESTORE_FAILED", out)
        self.assertIn("exposure restoration: NOT VERIFIED", out)
        self.assertIn("CAMERA STATE UNCERTAIN", out)
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0, 5000.0])  # one restoration attempt

    def test_shutdown_failure_after_sequence_reported_distinctly(self):
        real_shutdown = phys.PhysicalApprovedSessionHost.shutdown

        def failing_shutdown(host):
            real_shutdown(host)
            raise ac.CleanupError("simulated host shutdown failure")

        with mock.patch.object(phys.PhysicalApprovedSessionHost, "shutdown", autospec=True,
                               side_effect=failing_shutdown):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code, smoke.EXIT_CLEANUP_FAILED)
        out = self.output()
        self.assertNotIn("EXPERIMENT_ERROR", out)  # the experiment itself succeeded
        self.assertIn("CLEANUP_ERROR       : CleanupError: simulated host shutdown failure", out)

    def test_unexpected_error_still_shuts_down(self):
        with mock.patch.object(smoke, "run_sequence", side_effect=RuntimeError("boom")):
            code = self.run_cli()
        self.assert_failed_and_shut_down(code)
        self.assertIn("UNEXPECTED_ERROR: RuntimeError: boom", self.output())
        self.assertIsNone(phys._device_owner)


if __name__ == "__main__":
    unittest.main()
