"""Offline tests for hw3/lumen_cli.py (HW3.6-A).

The human here is SIMULATED by mocking builtins.input: it reads the session_id
the CLI displayed and types a line. That is a trusted test harness, not a real
human and not authentication. The scripted callers are the CLI's own
deterministic scripts (not AI reasoning). The camera backend is FakeCamera (a
recording subclass from the driver tests); camera tools are reached through the
real LangGraph approval, the official MCP SDK Client and the authorized
in-process server. No LLM API, vendor SDK, physical camera or network is used.
"""

import ast
import contextlib
import importlib
import io
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import approved_session_host as ash  # noqa: E402
import executor_mcp_loop as eml  # noqa: E402
import lumen_cli  # noqa: E402
import lumen_session_driver as lsd  # noqa: E402
from approved_session_host import OfflineApprovedSessionHost  # noqa: E402
from executor_mcp_loop import DECISION_INSTRUCTIONS  # noqa: E402
from lumen_session_driver import LumenDriverError  # noqa: E402
from session_policy import CAMERA_TOOLS, SessionGate  # noqa: E402
from test_lumen_session_driver import SpyCamera, openai_sdk_guard  # noqa: E402

MODULE_PATH = HW3_DIR / "lumen_cli.py"
SESSION_LINE = re.compile(r"^session_id\s+:\s+(lumen-[0-9a-f]{32})$", re.MULTILINE)


class TTY(io.StringIO):
    """A captured stream that claims to be an interactive terminal."""

    def isatty(self):
        return True


class RecordingEnviron(dict):
    """os.environ stand-in that records every key read."""

    def __init__(self, data, reads):
        super().__init__(data)
        self.reads = reads

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.reads.append(key)
        return super().__contains__(key)


def typed(kind):
    """SIMULATED human typing, based only on what the CLI printed."""

    def respond(stdout):
        if kind == "eof":
            raise EOFError
        if kind == "interrupt":
            raise KeyboardInterrupt
        if kind == "approve":
            return f"APPROVE {SESSION_LINE.search(stdout.getvalue()).group(1)}"
        if kind == "wrong_id":
            return "APPROVE lumen-" + "0" * 32
        return {"deny": "DENY", "blank": ""}.get(kind, kind)

    return respond


class CLITestCase(unittest.TestCase):
    def setUp(self):
        self.cameras = []
        self.fail_close = False
        self.pending_results = []
        self.reports = []
        self.hw_callers = []
        real_driver = lsd.run_lumen_session
        test = self

        def capturing_driver(*args):
            report = real_driver(*args)
            test.reports.append(report)
            return report

        class RecordingHardwareCaller(lumen_cli.ScriptedHardwareCaller):
            def __init__(self):
                super().__init__()
                test.hw_callers.append(self)

        self.driver_spy = self.patch(mock.patch.object(lumen_cli, "run_lumen_session",
                                                       side_effect=capturing_driver))
        self.patch(mock.patch.object(lumen_cli, "ScriptedHardwareCaller",
                                     RecordingHardwareCaller))
        self.patch(mock.patch.object(ash, "FakeCamera", self._new_camera))
        self.patch(mock.patch("socket.create_connection",
                              side_effect=AssertionError("network connection")))
        self.spies = {}
        for name, store in (("start", None), ("pending", self.pending_results),
                            ("resume", None), ("shutdown", None)):
            self.spies[name] = self.patch(self._wrap(OfflineApprovedSessionHost, name, store))
        # call_tool is a coroutine function: the AsyncMock spy awaits the real method.
        self.call_tool = self.patch(mock.patch.object(eml.Client, "call_tool", autospec=True,
                                                      side_effect=eml.Client.call_tool))

    def patch(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    @staticmethod
    def _wrap(cls, name, store=None):
        original = getattr(cls, name)

        def side_effect(*args, **kwargs):
            result = original(*args, **kwargs)
            if store is not None:
                store.append(result)
            return result

        return mock.patch.object(cls, name, autospec=True, side_effect=side_effect)

    def _new_camera(self):
        camera = SpyCamera(fail_close=self.fail_close)
        self.cameras.append(camera)
        return camera

    def run_cli(self, kind="approve", argv=(), stdin_tty=True, stdout_tty=True):
        stdout = TTY() if stdout_tty else io.StringIO()
        stdin = TTY() if stdin_tty else io.StringIO()
        self.stderr = io.StringIO()
        respond = typed(kind)
        self.input = mock.Mock(side_effect=lambda prompt="": respond(stdout))
        with mock.patch("sys.stdin", stdin), mock.patch("sys.stdout", stdout), \
                mock.patch("sys.stderr", self.stderr), mock.patch("builtins.input", self.input):
            code = lumen_cli.main(list(argv))
        self.out = stdout.getvalue()
        return code

    @property
    def report(self):
        (report,) = self.reports
        return report

    @property
    def host(self):
        return self.spies["start"].call_args.args[0]

    @property
    def camera(self):
        return self.cameras[-1]

    def events(self, tool):
        return [e for e in self.report["experiment_result"]["events"]
                if e["type"] == "tool_call" and e["tool"] == tool and e["status"] == "success"]

    def assert_nothing_started(self):
        self.driver_spy.assert_not_called()
        self.input.assert_not_called()
        self.spies["start"].assert_not_called()
        self.assertEqual(self.cameras, [])

    def assert_refused_without_camera(self, code):
        self.assertEqual(code, lumen_cli.EXIT_REFUSED)
        self.assertEqual(self.report["session_status"], lsd.SESSION_REFUSED)
        self.assertEqual(self.report["approval_status"], lsd.APPROVAL_REFUSED)
        self.assertEqual(self.report["outer_graph_status"], lsd.GRAPH_NOT_RUN)
        self.assertIsNone(self.report["experiment_result"])
        self.assertEqual(self.camera.calls, [])
        self.call_tool.assert_not_called()
        self.assertEqual(self.hw_callers[0].contexts, [])
        self.assertTrue(self.report["shutdown_succeeded"])
        self.assertEqual(self.host._state, ash.HOST_SHUT_DOWN)
        self.assertIn("REFUSED (no camera call was made)", self.out)
        self.assertNotIn("run outcome         : COMPLETED", self.out)


# 1-4: entry point ------------------------------------------------------------------

class ImportTests(unittest.TestCase):
    def test_import_has_no_side_effects(self):
        out = io.StringIO()
        with mock.patch.object(lsd, "run_lumen_session",
                               side_effect=AssertionError("driver run on import")) as driver, \
                mock.patch.object(ash, "FakeCamera",
                                  side_effect=AssertionError("camera on import")) as camera, \
                mock.patch("builtins.input", side_effect=AssertionError("prompt on import")), \
                mock.patch("sys.stdout", out):
            module = importlib.reload(lumen_cli)
        importlib.reload(lumen_cli)  # rebind the real driver
        driver.assert_not_called()
        camera.assert_not_called()
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(callable(module.main))
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        for node in tree.body:
            self.assertIsInstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                         ast.ClassDef, ast.Assign, ast.Expr, ast.If))
            if isinstance(node, ast.Expr):
                self.assertIsInstance(node.value, ast.Constant)  # docstring only
            if isinstance(node, ast.If):
                self.assertEqual(ast.unparse(node.test), "__name__ == '__main__'")


class EntryPointTests(CLITestCase):
    def test_offline_is_default_mode(self):
        self.assertEqual(lumen_cli._parser().parse_args([]).mode, "offline")
        self.assertEqual(lumen_cli._parser().parse_args(["--mode", "offline"]).mode, "offline")

    def test_unknown_and_live_arguments_refused(self):
        for argv in (["--mode", "live"], ["--live"], ["--approve"], ["extra"],
                     ["--mode", "offline", "--auto-approve"], ["--mode"]):
            with self.subTest(argv=argv):
                code = self.run_cli(argv=argv)
                self.assertEqual(code, lumen_cli.EXIT_USAGE)
                self.assert_nothing_started()

    def test_noninteractive_execution_fails_closed(self):
        for stdin_tty, stdout_tty in ((False, True), (True, False), (False, False)):
            with self.subTest(stdin_tty=stdin_tty, stdout_tty=stdout_tty):
                code = self.run_cli(stdin_tty=stdin_tty, stdout_tty=stdout_tty)
                self.assertEqual(code, lumen_cli.EXIT_NOT_INTERACTIVE)
                self.assertIn("interactive terminal is required", self.stderr.getvalue())
                self.assertNotIn("APPROVE", self.out)
                self.assert_nothing_started()

    def test_trusted_envelope_template(self):
        template = lumen_cli.offline_envelope_template()
        self.assertEqual(template.allowed_tools, CAMERA_TOOLS)
        self.assertEqual((template.min_exposure_us, template.max_exposure_us), (100.0, 10000.0))
        self.assertEqual(template.max_measurements, 2)
        self.assertEqual(template.max_tool_calls, 6)
        self.assertEqual(template.max_duration_seconds, 120.0)
        self.assertIn("FakeCamera", template.device_label)
        self.assertIn("NOT a physical camera", template.device_label)
        self.assertEqual(template.session_id, lumen_cli.TEMPLATE_SESSION_ID)


# 5-11: manual human approval ---------------------------------------------------------

class ApprovalTests(CLITestCase):
    def test_displays_genuine_pending_session_details(self):
        self.run_cli("deny")
        (pending,) = self.pending_results
        envelope = pending["envelope"]
        self.assertEqual(SESSION_LINE.search(self.out).group(1), pending["session_id"])
        self.assertEqual(pending["session_id"], self.report["session_id"])
        self.assertNotEqual(pending["session_id"], lumen_cli.TEMPLATE_SESSION_ID)
        self.assertIn(f"envelope SHA-256    : {pending['envelope_sha256']}", self.out)
        self.assertIn("allowed tools       : " + ", ".join(envelope["allowed_tools"]), self.out)
        self.assertIn("exposure bounds (us): [100.0, 10000.0]", self.out)
        self.assertIn("max measurements    : 2", self.out)
        self.assertIn("max tool calls      : 6", self.out)
        self.assertIn("max duration (s)    : 120.0", self.out)
        self.assertIn(envelope["device_label"], self.out)
        self.assertIn(f"type exactly:  APPROVE {pending['session_id']}", self.out)
        self.assertEqual(pending, SessionGate(self.host._envelope).approval_request())

    def test_exact_approval_activates_session(self):
        code = self.run_cli("approve")
        self.assertEqual(code, lumen_cli.EXIT_COMPLETED)
        self.assertEqual(self.report["approval_status"], lsd.APPROVAL_APPROVED)
        self.assertEqual(self.report["session_status"], lsd.SESSION_EXPERIMENT_COMPLETED)
        (resume_call,) = self.spies["resume"].call_args_list
        self.assertTrue(resume_call.args[1]["approved"])
        self.assertIn("run outcome         : COMPLETED", self.out)

    def test_refusals(self):
        for kind in ("wrong_id", "deny", "blank", "eof", "approve_lower", "APPROVE",
                     "Deny", " DENY", "yes"):
            with self.subTest(kind=kind):
                self.reports.clear()
                self.hw_callers.clear()
                self.call_tool.reset_mock()
                if kind == "approve_lower":
                    respond = typed("approve")
                    code = self._run_with(lambda out: respond(out).lower())
                else:
                    code = self.run_cli(kind)
                self.assert_refused_without_camera(code)
                self.assertEqual(self.input.call_count, 1)  # no repeat prompt

    def _run_with(self, responder):
        stdout = TTY()
        self.stderr = io.StringIO()
        self.input = mock.Mock(side_effect=lambda prompt="": responder(stdout))
        with mock.patch("sys.stdin", TTY()), mock.patch("sys.stdout", stdout), \
                mock.patch("sys.stderr", self.stderr), mock.patch("builtins.input", self.input):
            code = lumen_cli.main([])
        self.out = stdout.getvalue()
        return code

    def test_eof_does_not_approve(self):
        code = self.run_cli("eof")
        self.assert_refused_without_camera(code)
        self.assertIn("REFUSED: no input (EOF)", self.out)
        self.assertFalse(self.spies["resume"].call_args.args[1]["approved"])

    def test_callback_invoked_exactly_once(self):
        self.run_cli("approve")
        self.assertEqual(self.input.call_count, 1)
        self.spies["pending"].assert_called_once()
        self.spies["resume"].assert_called_once()
        self.assertEqual(self.out.count("LUMEN HARDWARE SESSION APPROVAL REQUEST"), 1)

    def test_callback_is_single_use_and_never_approves_again(self):
        self.run_cli("deny")
        (pending,) = self.pending_results
        approval = lumen_cli.TerminalApproval(input_fn=lambda prompt: (
            f"APPROVE {pending['session_id']}"), out=io.StringIO())
        first = approval(pending)
        self.assertIs(first["approved"], True)
        self.assertIsNone(approval(pending))  # a second call never prompts or approves
        self.assertEqual(approval.calls, 2)

    def test_malformed_request_never_prompts(self):
        prompt = mock.Mock(side_effect=AssertionError("prompted"))
        for request in (None, {}, {"session_id": "lumen-x", "envelope_sha256": "a"},
                        {"session_id": "lumen-x", "envelope_sha256": "a",
                         "envelope": {"session_id": "lumen-y"}}):
            with self.subTest(request=request):
                approval = lumen_cli.TerminalApproval(input_fn=prompt, out=io.StringIO())
                self.assertIsNone(approval(request))
        prompt.assert_not_called()

    def test_interrupted_input_does_not_approve(self):
        code = self.run_cli("interrupt")
        self.assertEqual(code, lumen_cli.EXIT_INTERRUPTED)
        self.assertEqual(self.reports, [])
        self.spies["resume"].assert_not_called()
        self.assertEqual(self.camera.calls, [])
        self.assertEqual(self.host._state, ash.HOST_SHUT_DOWN)
        self.assertNotIn("COMPLETED", self.out)

    def test_no_api_key_required_or_accessed(self):
        import os
        reads = []
        with mock.patch.object(os, "environ", RecordingEnviron(os.environ, reads)):
            code = self.run_cli("approve")
        self.assertEqual(code, lumen_cli.EXIT_COMPLETED)
        self.assertFalse([key for key in reads
                          if "KEY" in str(key).upper() or "OPENAI" in str(key).upper()])
        self.assertEqual(self.input.call_count, 1)  # only the approval phrase is requested
        self.assertNotIn("api key", self.out.lower().replace("no api key", ""))
        source = MODULE_PATH.read_text(encoding="utf-8")
        for token in ("environ", "getenv", "API_KEY", "openai", "OpenAI", "getpass",
                      "socket", "http", "MVSDK", "sdk_probe", "ctypes", "subprocess"):
            self.assertNotIn(token, source)


# 12-24: full scripted offline run ----------------------------------------------------

class OfflineRunTests(CLITestCase):
    def setUp(self):
        super().setUp()
        self.code = self.run_cli("approve")

    def test_real_langgraph_and_mcp_used(self):
        from langgraph.pregel import Pregel
        self.assertEqual(self.code, lumen_cli.EXIT_COMPLETED)
        self.assertIsInstance(self.host._graph, Pregel)  # real LangGraph approval graph
        self.assertEqual(self.report["outer_graph_status"], lsd.GRAPH_COMPLETED)
        self.assertEqual([c.args[1] for c in self.call_tool.call_args_list],
                         ["camera_open", "camera_set_exposure", "camera_measure",
                          "camera_set_exposure", "camera_measure"])
        self.assertIsInstance(self.call_tool.call_args.args[0], eml.Client)
        self.assertEqual(self.report["experiment_result"]["provenance"]["transport"],
                         "mcp-client (in-process, official SDK)")

    def test_scientist_participates_in_shared_dialogue(self):
        roles = [turn["role"] for turn in self.report["dialogue"]]
        self.assertEqual(roles, ["user", "orchestrator", "scientist", "orchestrator",
                                 "executor", "orchestrator"])
        self.assertIn({"role": "scientist", "text": lumen_cli.ScriptedOuterCaller.HYPOTHESIS},
                      self.report["dialogue"])
        self.assertIn("outer roles visited : user -> orchestrator -> scientist -> orchestrator "
                      "-> executor -> orchestrator", self.out)

    def test_two_distinct_exposures_applied(self):
        applied = [e["result"]["exposure_us"] for e in self.events("camera_set_exposure")]
        self.assertEqual(applied, [500.0, 2000.0])
        self.assertEqual(self.camera.calls.count("set_exposure_us"), 2)
        self.assertIn("exposures applied   : 500.0 us, 2000.0 us", self.out)

    def test_two_real_fake_camera_frames(self):
        measures = [e["result"] for e in self.events("camera_measure")]
        self.assertEqual(self.camera.calls.count("capture_frame"), 2)
        self.assertEqual([m["frame_id"] for m in measures], [1, 2])
        self.assertEqual([m["exposure_us"] for m in measures], [500.0, 2000.0])
        self.assertEqual(self.report["experiment_result"]["successful_measurement_count"], 2)

    def test_second_decision_sees_first_measurement(self):
        (caller,) = self.hw_callers
        self.assertEqual(len(caller.contexts), 6)
        before_second_exposure = caller.contexts[3]
        latest = before_second_exposure["latest_measurement"]
        self.assertEqual((latest["frame_id"], latest["exposure_us"]), (1, 500.0))
        decisions = [e for e in self.report["experiment_result"]["events"]
                     if e["type"] == "decision"]
        self.assertEqual(decisions[3]["decision"]["arguments"], {"exposure_us": 2000.0})
        self.assertEqual(caller.contexts[5]["latest_measurement"]["frame_id"], 2)
        self.assertEqual(decisions[5]["decision"]["action"], "finish")

    def test_summary_metrics_match_evidence(self):
        measures = [e["result"] for e in self.events("camera_measure")]
        for number, m in enumerate(measures, 1):
            self.assertIn(f"  measurement {number}: frame_id={m['frame_id']} "
                          f"exposure_us={m['exposure_us']} mean_dn={m['metrics']['mean_dn']} "
                          f"saturation_fraction={m['metrics']['saturation_fraction']} "
                          "source=synthetic", self.out)
        self.assertEqual(measures[0]["metrics"]["saturation_fraction"], 0.0)
        self.assertGreater(measures[1]["metrics"]["saturation_fraction"], 0.0)
        self.assertGreater(measures[1]["metrics"]["mean_dn"], measures[0]["metrics"]["mean_dn"])

    def test_synthetic_never_described_as_physical_evidence(self):
        self.assertEqual(self.report["evidence_provenance"], "synthetic")
        self.assertIn("evidence provenance : synthetic", self.out)
        self.assertIn("NOT a physical camera measurement", self.out)
        self.assertNotIn("source=hardware", self.out)
        self.assertFalse(self.report["experiment_result"]["provenance"]["hardware_evidence"])
        self.assertIn("SCRIPTED", self.out)

    def test_tool_budgets_not_exceeded(self):
        gate = self.host._runtime._gate
        self.assertEqual(gate.tool_calls_used, 5)
        self.assertLessEqual(gate.tool_calls_used, gate.envelope.max_tool_calls)
        self.assertEqual(gate.measurements_used, 2)
        self.assertLessEqual(gate.measurements_used, gate.envelope.max_measurements)
        self.assertEqual(self.report["experiment_result"]["decision_count"], 6)
        self.assertLessEqual(self.report["experiment_result"]["decision_count"],
                             lumen_cli.MAX_HARDWARE_DECISIONS)

    def test_trusted_driver_closes_host(self):
        self.assertTrue(self.report["shutdown_succeeded"])
        self.assertEqual(self.host._state, ash.HOST_SHUT_DOWN)
        driver_calls = [c for c in self.spies["shutdown"].call_args_list if c.args[0] is self.host]
        self.assertGreaterEqual(len(driver_calls), 1)
        self.assertEqual(self.camera.calls[-1], "close")
        self.assertIn("host shutdown       : succeeded", self.out)

    def test_no_raw_event_log_or_decision_printed(self):
        self.assertNotIn('"approved"', self.out)
        self.assertNotIn("histogram", self.out)
        self.assertNotIn("raw_output", self.out)


class FailureTests(CLITestCase):
    def test_refusal_produces_no_camera_calls(self):
        build = self.patch(mock.patch.object(lsd.LumenAgentSystem, "build",
                                             side_effect=AssertionError("graph built")))
        code = self.run_cli("deny")
        self.assert_refused_without_camera(code)
        build.assert_not_called()

    def test_driver_exception_gives_nonzero_exit(self):
        for error in (LumenDriverError("simulated", {"session_status": "failed",
                                                     "shutdown_attempted": True,
                                                     "shutdown_succeeded": True}),
                      RuntimeError("unexpected")):
            with self.subTest(error=type(error).__name__):
                self.driver_spy.side_effect = error
                code = self.run_cli("approve")
                self.assertEqual(code, lumen_cli.EXIT_FAILED)
                self.assertNotIn("COMPLETED", self.out.replace("NOT COMPLETED", ""))

    def test_cleanup_failure_visible_and_not_success(self):
        self.fail_close = True
        code = self.run_cli("approve")
        self.assertEqual(code, lumen_cli.EXIT_FAILED)
        self.assertIn("session FAILED", self.stderr.getvalue())
        self.assertIn("host shutdown       : FAILED", self.out)
        self.assertIn("run outcome         : NOT COMPLETED", self.out)

    def test_hardware_script_fails_closed_on_missing_measurement(self):
        caller = lumen_cli.ScriptedHardwareCaller()

        def context(history, latest):
            return json.dumps({"history": history, "latest_measurement": latest})

        def ok(tool, arguments, result=None):
            return {"type": "tool_call", "tool": tool, "arguments": arguments,
                    "status": "success", "result": result or {}}

        measurement = {"frame_id": 1, "exposure_us": 500.0, "source": "synthetic"}
        history = [ok("camera_open", {}), ok("camera_set_exposure", {"exposure_us": 500.0}),
                   ok("camera_measure", {}, measurement)]
        for latest in (None, {}, {**measurement, "exposure_us": 600.0},
                       {**measurement, "frame_id": "1"}):
            with self.subTest(latest=latest), self.assertRaises(lumen_cli.ScriptError):
                caller.call(DECISION_INSTRUCTIONS, context(history, latest))
        failed = history[:2] + [{**history[2], "status": "mcp_error"}]
        with self.assertRaises(lumen_cli.ScriptError):
            caller.call(DECISION_INSTRUCTIONS, context(failed, measurement))
        with self.assertRaises(lumen_cli.ScriptError):
            caller.call("other instructions", context([], None))
        decision = json.loads(caller.call(DECISION_INSTRUCTIONS, context(history, measurement)))
        self.assertEqual(decision["arguments"], {"exposure_us": 2000.0})

    def test_no_openai_client_or_request(self):
        with openai_sdk_guard(self):
            code = self.run_cli("approve")
        self.assertEqual(code, lumen_cli.EXIT_COMPLETED)
        self.assertEqual(self.camera.calls.count("capture_frame"), 2)


class SampleOutputTests(CLITestCase):
    """Prints the actual summary of one test run (for the iteration report)."""

    def test_print_sample_summary(self):
        self.assertEqual(self.run_cli("approve"), lumen_cli.EXIT_COMPLETED)
        summary = self.out[self.out.index("LUMEN OFFLINE SESSION SUMMARY") - 73:]
        with contextlib.redirect_stdout(sys.__stdout__):
            print("\n[sample CLI summary from this test run]\n" + summary)


if __name__ == "__main__":
    unittest.main()
