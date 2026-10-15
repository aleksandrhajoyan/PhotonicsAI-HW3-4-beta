"""Offline tests for hw3/physical_adaptive_cli.py.

main() runs end to end against the REAL LangGraph approval graph, SessionGate,
PhysicalApprovedSessionHost, AuthorizedCameraRuntime, authorized MCP server,
official MCP Client, AdaptiveMCPExecutor, BoundedDecisionProvider,
LumenOpenAIModelCaller, create_live_openai_client and IRaypleCamera. Only these
are fakes:
    - the vendor SDK binding (test_irayple_camera.FakeSdk via test_physical_mcp_smoke),
    - the ``openai`` module, placed in sys.modules for one run; its OpenAI client
      answers responses.create() with a FAKE model that decides only from the
      input text it receives,
    - terminal input (a simulated operator reading the printed session_id).
No camera, vendor DLL, network, API key, paid API or real LLM is involved.
"""

import ast
import importlib
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorized_camera as ac  # noqa: E402
import executor_mcp_loop as eml  # noqa: E402
import irayple_camera as ic  # noqa: E402
import llm_decision_provider as ldp  # noqa: E402
import openai_model_caller as omc  # noqa: E402
import physical_adaptive_cli as cli  # noqa: E402
import physical_approved_session_host as phys  # noqa: E402
import physical_mcp_smoke as smoke  # noqa: E402
import test_physical_mcp_smoke as smoke_tests  # noqa: E402
from session_policy import SessionGate  # noqa: E402

MODULE_PATH = HW3_DIR / "physical_adaptive_cli.py"
FLAG = "--physical-adaptive-check"
MODEL_ID = "test-model-not-real"
ARGV = (FLAG, "--model", MODEL_ID)
WRITE = smoke_tests.WRITE
N_PIXELS = smoke_tests.N_PIXELS
BRIGHT_FRAME = np.full(N_PIXELS, 200, np.uint8).tobytes()  # mean 200 DN, not saturated
FRAME_1_MEAN = float(np.frombuffer(smoke_tests.FRAME_1, np.uint8).mean())
SECRET = "sk-test-secret-do-not-print"


def call(tool, **arguments):
    return json.dumps({"action": "call_tool", "tool": tool, "arguments": arguments})


def finish(reason="goal handled"):
    return json.dumps({"action": "finish", "reason": reason})


def approve_shown_session(screen):
    return f"APPROVE {smoke_tests.SESSION_RE.findall(screen)[-1]}"


def consent_shown_session(screen):
    return f"AUTHORIZE PAID API {smoke_tests.SESSION_RE.findall(screen)[-1]}"


ANSWERS = (cli.CONFIRMATION_PHRASE, approve_shown_session, consent_shown_session)


class AdaptiveModel:
    """FAKE model: decides only from the compact context it is sent (no script of exposures)."""

    def __init__(self, target_dn=70.0):
        self.target_dn = target_dn
        self.seen = []  # latest_measurement frame_id visible in each request
        self.contexts = []

    def __call__(self, request):
        context = json.loads(request["input"])
        self.contexts.append(context)
        latest = context["latest_measurement"]
        self.seen.append(None if latest is None else latest["frame_id"])
        tools = [event["tool"] for event in context["recent_history"]
                 if event.get("type") == "tool_call" and event.get("status") == "success"]
        if "camera_open" not in tools:
            return call("camera_open")
        if "camera_close" in tools:
            return finish("two measurements done")
        sets, measures = tools.count("camera_set_exposure"), tools.count("camera_measure")
        if measures == 2:
            return call("camera_close")
        if sets > measures:
            return call("camera_measure")
        if latest is None:
            return call("camera_set_exposure", exposure_us=1000.0)
        mean = latest["metrics"]["mean_dn"]
        return call("camera_set_exposure", exposure_us=5000.0 if mean < self.target_dn else 1000.0)


class ScriptedModel:
    """FAKE model replaying fixed replies (or raising) - only for refusal and failure paths."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, request):
        latest = json.loads(request["input"])["latest_measurement"]
        self.seen.append(None if latest is None else latest["frame_id"])
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeResponses:
    def __init__(self, model):
        self.model = model
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.model(kwargs)
        return types.SimpleNamespace(status="completed", output_text=text, model=kwargs["model"],
                                     usage=types.SimpleNamespace(input_tokens=10, output_tokens=5,
                                                                 total_tokens=15))


class FakeOpenAIClient:
    def __init__(self, model):
        self.responses = FakeResponses(model)


class AdaptiveTestCase(smoke_tests.SmokeTestCase):
    """Shared offline harness: fake SDK + fake openai module; everything else is real."""

    def run_cli(self, answers=ANSWERS, argv=ARGV, model=None, stdin_tty=True, stdout_tty=True,
                client_error=None):
        self.model = AdaptiveModel() if model is None else model
        self.client = FakeOpenAIClient(self.model)
        self.openai_module = types.ModuleType("openai")
        self.openai_module.OpenAI = mock.Mock(return_value=self.client, side_effect=client_error)
        self.operator = smoke_tests.Operator(self, answers)
        stdin = smoke_tests.TTY() if stdin_tty else smoke_tests.io.StringIO()
        if not stdout_tty:
            self.stdout = smoke_tests.io.StringIO()
        with mock.patch.dict(sys.modules, {"openai": self.openai_module}), \
                mock.patch("sys.stdin", stdin), mock.patch("sys.stdout", self.stdout), \
                mock.patch("builtins.input", side_effect=self.operator):
            return cli.main(list(argv))

    def spy_execute(self):
        """One real-execute spy per test, reset for each subtest run (spies cannot be nested)."""
        if getattr(self, "_execute_spy", None) is None:
            self._execute_spy = super().spy_execute()
        self._execute_spy.reset_mock()
        return self._execute_spy

    def requests(self):
        return self.client.responses.requests

    def assert_no_model_request(self):
        self.assertEqual(self.requests(), [])
        self.openai_module.OpenAI.assert_not_called()

    def assert_verdict(self, code, expected_code, verdict):
        self.assertEqual(code, expected_code, self.output())
        self.assertIn(f"PHYSICAL_ADAPTIVE_CHECK: {verdict}", self.output())
        if verdict != "PASS":
            self.assertNotIn("PHYSICAL_ADAPTIVE_CHECK: PASS", self.output())

    def assert_no_backend_activity(self):
        names = self.fake.names() if self.fake else []
        for native in ("IMV_Open", "IMV_GetFrame", WRITE):
            self.assertNotIn(native, names)


# 1, 32, 33. Import safety, isolation, no real API.
class ImportAndIsolationTests(unittest.TestCase):
    def test_import_loads_no_sdk_no_client_no_camera(self):
        had_openai = "openai" in sys.modules
        with mock.patch.object(ic.discovery, "load_vendor_sdk") as loader, \
                mock.patch.object(ic, "preload_vendor_sdk") as preload, \
                mock.patch.object(ic, "IRaypleCamera") as backend, \
                mock.patch.object(phys, "PhysicalApprovedSessionHost") as host, \
                mock.patch.object(omc, "create_live_openai_client") as factory, \
                mock.patch.object(omc, "LumenOpenAIModelCaller") as caller, \
                mock.patch("ctypes.WinDLL", create=True) as win_dll, \
                mock.patch("os.chdir") as chdir:
            importlib.reload(cli)
        for patched in (loader, preload, backend, host, factory, caller, win_dll, chdir):
            patched.assert_not_called()
        self.assertEqual("openai" in sys.modules, had_openai)
        self.assertNotIn("IMVApi", sys.modules)

    def test_source_reads_no_credentials_and_uses_no_threads_or_backend(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in ("environ", "getenv", "api_key", "OPENAI_API_KEY", "Authorization",
                          "fake_camera", "FakeCamera", "httpx", "urllib"):
            self.assertNotIn(forbidden, source)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"__future__", "argparse", "asyncio", "json", "math", "sys",
                                    "threading", "time", "typing", "openai_model_caller",
                                    "physical_approved_session_host", "physical_mcp_smoke",
                                    "approved_session_host", "executor_mcp_loop",
                                    "llm_decision_provider", "session_policy"})
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for name in ("execute", "capture_frame", "set_exposure_us", "IRaypleCamera",
                     "AuthorizedCameraRuntime", "create_authorized_server", "submit_decision",
                     "begin_action", "_runtime", "_camera", "chdir", "Thread", "to_thread",
                     "run_in_executor", "preload_vendor_sdk", "call_tool"):
            self.assertNotIn(name, attributes | names)


# 2-8. Entry point, confirmations, approval and paid-API consent.
class EntryPointTests(AdaptiveTestCase):
    def test_missing_flag_refuses_without_anything(self):
        for argv in ((), ("--model", MODEL_ID), ("--physical-mcp-check",)):
            with self.subTest(argv=argv):
                code = self.run_cli(argv=argv)
                self.assertEqual(code, cli.EXIT_REFUSED)
                self.assertEqual(self.operator.prompts, [])
                self.assert_no_model_request()
        self.preload_mock.assert_not_called()
        self.assertIsNone(phys._device_owner)

    def test_missing_or_blank_model_refuses_without_anything(self):
        for argv in ((FLAG,), (FLAG, "--model", ""), (FLAG, "--model", " gpt ")):
            with self.subTest(argv=argv):
                code = self.run_cli(argv=argv)
                self.assertEqual(code, cli.EXIT_REFUSED)
                self.assertIn("explicit --model", self.output())
                self.assertEqual(self.operator.prompts, [])
        self.preload_mock.assert_not_called()

    def test_noninteractive_refuses_without_anything(self):
        for stdin_tty, stdout_tty in ((False, True), (True, False), (False, False)):
            with self.subTest(stdin_tty=stdin_tty, stdout_tty=stdout_tty):
                code = self.run_cli(stdin_tty=stdin_tty, stdout_tty=stdout_tty)
                self.assertEqual(code, cli.EXIT_NOT_INTERACTIVE)
                self.assertEqual(self.operator.prompts, [])
                self.assert_no_model_request()
        self.preload_mock.assert_not_called()

    def test_first_confirmation_required(self):
        for answer in ("begin adaptive hardware check", "BEGIN PHYSICAL MCP CHECK", "yes", "",
                       cli.CONFIRMATION_PHRASE + " ", EOFError(), KeyboardInterrupt()):
            with self.subTest(answer=repr(answer)):
                code = self.run_cli(answers=(answer,))
                self.assertEqual(code, cli.EXIT_REFUSED)
                self.assertEqual(len(self.operator.prompts), 1)
                self.assert_no_model_request()
        self.preload_mock.assert_not_called()
        self.assertIsNone(self.fake)

    def test_actual_hitl_approval_required(self):
        code = self.run_cli(answers=(cli.CONFIRMATION_PHRASE,))  # EOF at the approval prompt
        self.assert_verdict(code, cli.EXIT_APPROVAL_REFUSED, "FAIL")
        self.assertIn("approval            : refused", self.output())
        self.assertIn("pending in LangGraph", self.output())
        self.assert_no_model_request()
        self.assertEqual(self.fake.calls, [])
        self.assertIsNone(phys._device_owner)

    def assert_approval_refused(self, answer):
        execute = self.spy_execute()
        code = self.run_cli(answers=(cli.CONFIRMATION_PHRASE, answer, consent_shown_session))
        self.assert_verdict(code, cli.EXIT_APPROVAL_REFUSED, "FAIL")
        self.assertEqual(len(self.operator.prompts), 2)  # never reached the API consent
        self.assert_no_model_request()
        execute.assert_not_called()
        self.assertEqual(self.fake.calls, [])
        self.assertIsNone(phys._device_owner)

    def test_wrong_session_id_refuses(self):
        self.assert_approval_refused("APPROVE lumen-phys-" + "0" * 32)
        self.assert_approval_refused(lambda screen: approve_shown_session(screen)[:-1])

    def test_rejected_or_malformed_approval_makes_zero_model_requests(self):
        for answer in ("DENY", "APPROVE", "yes", lambda s: approve_shown_session(s).lower(),
                       lambda s: approve_shown_session(s) + " ", consent_shown_session):
            with self.subTest(answer=answer if isinstance(answer, str) else "derived"):
                self._reset_process_state()
                self.stdout = smoke_tests.TTY()
                self.assert_approval_refused(answer)

    def test_expired_approval_refuses(self):
        ticks = iter([100.0, 100.0 + cli.APPROVAL_TIMEOUT_SECONDS + 0.5])
        with mock.patch.object(cli, "_clock", side_effect=lambda: next(ticks)):
            self.assert_approval_refused(approve_shown_session)
        self.assertIn("expired", self.output())

    def test_paid_api_confirmation_required(self):
        for answer in ("DENY", "AUTHORIZE PAID API", "yes", EOFError(),
                       lambda s: consent_shown_session(s).lower(),
                       lambda s: consent_shown_session(s)[:-1], approve_shown_session):
            with self.subTest(answer=repr(answer) if not callable(answer) else "derived"):
                self._reset_process_state()
                self.stdout = smoke_tests.TTY()
                execute = self.spy_execute()
                code = self.run_cli(answers=(cli.CONFIRMATION_PHRASE, approve_shown_session, answer))
                self.assert_verdict(code, cli.EXIT_API_CONSENT_REFUSED, "FAIL")
                self.assertIn("approval            : approved", self.output())
                self.assertIn("paid-API consent    : refused", self.output())
                self.assert_no_model_request()
                execute.assert_not_called()
                self.assert_no_backend_activity()
                self.assertIsNone(phys._device_owner)

    def test_expired_paid_api_confirmation_refuses(self):
        ticks = iter([0.0, 1.0, 10.0, 10.0 + cli.API_CONSENT_TIMEOUT_SECONDS + 0.5])
        with mock.patch.object(cli, "_clock", side_effect=lambda: next(ticks)):
            code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_API_CONSENT_REFUSED, "FAIL")
        self.assert_no_model_request()

    def test_consent_prompt_names_session_model_and_limits(self):
        code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        out = self.output()
        sid = self.session_id()
        self.assertIn(f"To authorize, type exactly:  AUTHORIZE PAID API {sid}", out)
        self.assertIn(f"model             : {MODEL_ID}", out)
        self.assertIn(f"at most {cli.MAX_MODEL_DECISIONS}, no automatic retries, no fallback model", out)
        self.assertEqual(len(self.operator.prompts), 3)
        # The paid-API consent comes after the hardware approval and before the first request.
        self.assertLess(out.index("Approval phrase received"), out.index("PAID MODEL API USE"))

    def test_api_key_never_printed_or_passed(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": SECRET}):
            code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assertNotIn(SECRET, self.output())
        self.openai_module.OpenAI.assert_called_once_with(max_retries=0, timeout=30.0)
        for request in self.requests():
            self.assertNotIn(SECRET, json.dumps(request))

    def test_environment_cannot_bypass_confirmations(self):
        env = {"LUMEN_APPROVE": "1", "PHYSICAL_ADAPTIVE_CHECK": cli.CONFIRMATION_PHRASE, "CI": "1"}
        with mock.patch.dict(os.environ, env):
            code = self.run_cli(answers=())
        self.assertEqual(code, cli.EXIT_REFUSED)
        self.preload_mock.assert_not_called()


# 9-15. Wiring of the real components.
class WiringTests(AdaptiveTestCase):
    def test_sdk_preloaded_on_main_thread_before_event_loop(self):
        real_run = cli.asyncio.run

        def run(coro, *args, **kwargs):
            self.events.append(("event_loop", smoke_tests.threading.get_ident()))
            return real_run(coro, *args, **kwargs)

        with mock.patch.object(cli.asyncio, "run", side_effect=run):
            code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assertEqual([event[0] for event in self.events],
                         ["input", "preload", "input", "input", "event_loop"])
        self.assertIs(self.events[1][1], False)  # no running loop during preload
        self.assertEqual(self.events[1][2], self.events[4][1])
        self.preload_mock.assert_called_once_with()
        for name, thread, loop in self.fake.contexts:  # every SDK call: same thread, inside the loop
            self.assertEqual(thread, self.events[1][2], name)
            self.assertIsNotNone(loop, name)
        smoke_tests.anyio.to_thread.run_sync.assert_not_called()

    def test_exactly_one_genuine_physical_host(self):
        real = phys.PhysicalApprovedSessionHost
        created = []

        def construct(*args, **kwargs):
            created.append(real(*args, **kwargs))
            return created[-1]

        with mock.patch.object(cli.phys, "PhysicalApprovedSessionHost", side_effect=construct):
            code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assertEqual(len(created), 1)
        self.assertIs(type(created[0]), real)
        self.assertIs(created[0]._camera.__class__, ic.IRaypleCamera)

    def test_existing_executor_provider_and_caller_reused(self):
        self.assertIs(cli.AdaptiveMCPExecutor, eml.AdaptiveMCPExecutor)
        self.assertIs(cli.BoundedDecisionProvider, ldp.BoundedDecisionProvider)
        self.assertIs(cli.omc, omc)
        spies = {}
        targets = ((eml.AdaptiveMCPExecutor, "run"), (ldp.BoundedDecisionProvider, "call"),
                   (omc.LumenOpenAIModelCaller, "call"))
        for owner, name in targets:
            patcher = mock.patch.object(owner, name, autospec=True, side_effect=getattr(owner, name))
            spies[owner] = patcher.start()
            self.addCleanup(patcher.stop)
        code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assertEqual(spies[eml.AdaptiveMCPExecutor].call_count, 1)
        self.assertEqual(spies[ldp.BoundedDecisionProvider].call_count, 7)
        self.assertEqual(spies[omc.LumenOpenAIModelCaller].call_count, 7)
        self.assertEqual(len(self.requests()), 7)

    def test_openai_caller_integration_matches_its_interface(self):
        code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.openai_module.OpenAI.assert_called_once_with(max_retries=0, timeout=30.0)
        for request in self.requests():
            self.assertEqual(set(request), {"model", "instructions", "input", "reasoning",
                                            "max_output_tokens"})
            self.assertEqual(request["model"], MODEL_ID)
            self.assertEqual(request["reasoning"], {"effort": cli.REASONING_EFFORT})
            self.assertEqual(request["max_output_tokens"], cli.MAX_OUTPUT_TOKENS)
            self.assertTrue(request["instructions"].startswith(eml.DECISION_INSTRUCTIONS))
            self.assertIn("exactly 1000.0 or 5000.0", request["instructions"])
            self.assertLessEqual(len(request["instructions"]) + len(request["input"]),
                                 cli.MAX_INPUT_CHARS)
            context = json.loads(request["input"])
            self.assertEqual(context["context_format"], ldp.CONTEXT_FORMAT)
            self.assertEqual(context["research_goal"], cli.RESEARCH_GOAL)
        self.assertIn(f"API requests        : 7 (limit {cli.MAX_MODEL_DECISIONS}, no retries)",
                      self.output())

    def test_model_call_budget_and_output_size_bounded(self):
        self.assertLessEqual(cli.MAX_MODEL_DECISIONS, 8)
        code = self.run_cli(model=ScriptedModel(["x" * (cli.MAX_OUTPUT_CHARS + 1)]))
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("ProviderOutputError", self.output())
        self.assertEqual(len(self.requests()), 1)

    def test_envelope_never_replaced_after_approval(self):
        envelopes, hosts = [], []
        real_build, real_host = smoke.build_envelope, phys.PhysicalApprovedSessionHost

        def build(session_id):
            envelopes.append(real_build(session_id))
            return envelopes[-1]

        def construct(*args, **kwargs):
            hosts.append(real_host(*args, **kwargs))
            return hosts[-1]

        with mock.patch.object(cli.smoke, "build_envelope", side_effect=build), \
                mock.patch.object(cli.phys, "PhysicalApprovedSessionHost", side_effect=construct):
            code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assertEqual(len(envelopes), 1)
        envelope = envelopes[0]
        self.assertIs(hosts[0]._envelope, envelope)
        self.assertIs(hosts[0]._runtime._gate.envelope, envelope)  # the gate enforcing it
        self.assertIn(envelope.canonical_json(), self.output())
        self.assertIn(envelope.sha256(), self.output())
        self.assertEqual(envelope.allowed_tools,
                         ("camera_open", "camera_set_exposure", "camera_measure", "camera_close"))
        self.assertEqual((envelope.min_exposure_us, envelope.max_exposure_us,
                          envelope.max_measurements, envelope.max_tool_calls,
                          envelope.max_duration_seconds), (1000.0, 5000.0, 2, 7, 180.0))

    def test_graph_and_execution_gate_both_validate_the_decision(self):
        original = SessionGate.submit_decision
        with mock.patch.object(SessionGate, "submit_decision", autospec=True, side_effect=original) as spy:
            code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assertEqual(spy.call_count, 2)


# 16-25. Model-chosen actions, trusted checks and the feedback loop.
class AdaptiveRunTests(AdaptiveTestCase):
    def test_model_chosen_actions_reach_authorized_mcp_and_pass(self):
        execute = self.spy_execute()
        code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_PASS, "PASS")
        self.assertEqual(self.executed_tools(execute),
                         ["camera_open", "camera_set_exposure", "camera_measure",
                          "camera_set_exposure", "camera_measure", "camera_close"])
        self.assertEqual([c.args[3] for c in execute.call_args_list],
                         [{}, {"exposure_us": 1000.0}, {}, {"exposure_us": 5000.0}, {}, {}])
        out = self.output()
        self.assertIn("1. saw measurements [] -> model: camera_open() | check: passed", out)
        self.assertIn("4. saw measurements [1] -> model: camera_set_exposure(exposure_us=5000.0) "
                      "| check: passed", out)
        self.assertIn("7. saw measurements [1, 2] -> model: finish('two measurements done')", out)
        self.assertIn("executor status     : completed", out)
        self.assert_released()

    def test_hardware_measurements_with_frame_ids_1_and_2(self):
        code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        out = self.output()
        self.assertIn("measurement 1: source=hardware frame_id=1 exposure=1000.0 us 1920x1200 Mono8", out)
        self.assertIn("measurement 2: source=hardware frame_id=2 exposure=5000.0 us 1920x1200 Mono8", out)
        self.assertIn(f"mean={FRAME_1_MEAN:.4f} DN", out)
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 2)
        self.assertEqual(self.fake.exposure_at_frame, [1000.0, 5000.0])

    def test_model_sees_measurement_one_before_choosing_next_exposure(self):
        code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assertEqual(self.model.seen, [None, None, None, 1, 1, 2, 2])
        fourth = self.model.contexts[3]  # the request that chose the second exposure
        self.assertEqual(fourth["latest_measurement"]["frame_id"], 1)
        self.assertEqual(fourth["latest_measurement"]["source"], "hardware")
        self.assertAlmostEqual(fourth["latest_measurement"]["metrics"]["mean_dn"], FRAME_1_MEAN)
        self.assertIn("feedback evidence   : decision 4 (event ", self.output())
        self.assertIn("was requested after measurement frame_id 1", self.output())

    def test_different_feedback_leads_to_a_different_model_choice(self):
        self.make_fake(payloads=(BRIGHT_FRAME, smoke_tests.FRAME_2))
        code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assertEqual(self.fake.exposure_at_frame, [1000.0, 1000.0])  # bright first frame
        self.assertIn("4. saw measurements [1] -> model: camera_set_exposure(exposure_us=1000.0)",
                      self.output())

    def test_integer_json_exposures_pass_unchanged(self):
        class IntegerModel(AdaptiveModel):
            def __call__(self, request):
                return super().__call__(request).replace(".0}", "}")

        code = self.run_cli(model=IntegerModel())
        self.assert_verdict(code, cli.EXIT_PASS, "PASS")
        self.assertEqual(self.fake.exposure_at_frame, [1000.0, 5000.0])
        self.assertIn("camera_set_exposure(exposure_us=5000) | check: passed", self.output())

    def test_feedback_evidence_derived_from_executor_history(self):
        report = self._report_after_pass()
        problem, description = cli.feedback_evidence(report)
        self.assertEqual(problem, "")
        self.assertIn("decision 4", description)
        events = report.result["events"]
        measures = [e for e in events if e.get("tool") == "camera_measure"]
        choice = [e for e in events if e["type"] == "decision" and e["decision_number"] == 4][0]
        self.assertLess(measures[0]["sequence"], choice["sequence"])
        self.assertLess(choice["sequence"], measures[1]["sequence"])
        # Without a decision between the two measurements, adaptivity is not claimed.
        report.result["events"] = [e for e in events if e is not choice]
        self.assertNotEqual(cli.feedback_evidence(report)[0], "")
        self.assertEqual(cli.verdict(report)[0], cli.VERDICT_FAIL)

    def _report_after_pass(self):
        captured = []
        real = cli.Report

        def make(*args):
            captured.append(real(*args))
            return captured[-1]

        with mock.patch.object(cli, "Report", side_effect=make):
            code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        return captured[0]

    def test_intermediate_exposure_refused_before_mcp(self):
        for value in (3000.0, 3000, 1000.5, 4999.0):
            with self.subTest(value=value):
                self._reset_process_state()
                self.make_fake()
                self.stdout = smoke_tests.TTY()
                execute = self.spy_execute()
                code = self.run_cli(model=ScriptedModel([
                    call("camera_open"), call("camera_set_exposure", exposure_us=value)]))
                self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
                self.assertEqual(self.executed_tools(execute), ["camera_open"])
                self.assertEqual(self.fake.writes(), [])
                self.assertIn("REFUSED: exposure_us", self.output())
                self.assertIn("not a first-run candidate", self.output())
                self.assert_released()

    def test_both_candidates_pass_validation(self):
        envelope = smoke.build_envelope("lumen-phys-test")
        state = {"opened": True, "closed": False, "exposures_set": 0, "measurements": 0}
        for value in (1000.0, 5000.0, 1000, 5000):
            decision = {"action": "call_tool", "tool": "camera_set_exposure",
                        "arguments": {"exposure_us": value}}
            self.assertEqual(cli.protocol_problem(decision, state, envelope), "")
        for value in (999.0, 3000.0, 5000.5, 6000.0):
            decision = {"action": "call_tool", "tool": "camera_set_exposure",
                        "arguments": {"exposure_us": value}}
            self.assertIn("not a first-run candidate", cli.protocol_problem(decision, state, envelope))

    def test_out_of_envelope_action_refused(self):
        # Locally: camera_info is not in the envelope.
        execute = self.spy_execute()
        code = self.run_cli(model=ScriptedModel([call("camera_open"), call("camera_info")]))
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("camera_info is not in the approved envelope", self.output())
        self.assertEqual(self.executed_tools(execute), ["camera_open"])
        self.assert_released()

    def test_session_gate_still_refuses_if_local_check_is_bypassed(self):
        execute = self.spy_execute()
        with mock.patch.object(cli, "protocol_problem", return_value=""):
            code = self.run_cli(model=ScriptedModel([
                call("camera_open"), call("camera_set_exposure", exposure_us=6000.0)]))
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("Not authorized", self.output())
        self.assertIn("outside the approved range", self.output())
        self.assertEqual(self.executed_tools(execute), ["camera_open", "camera_set_exposure"])
        self.assertEqual(self.fake.writes(), [])
        self.assert_released()

    def test_model_cannot_request_a_third_measurement(self):
        replies = [call("camera_open"), call("camera_set_exposure", exposure_us=1000.0),
                   call("camera_measure"), call("camera_set_exposure", exposure_us=5000.0),
                   call("camera_measure"), call("camera_measure")]
        code = self.run_cli(model=ScriptedModel(replies))
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("measurement 3 exceeds the limit of 2", self.output())
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 2)
        self.assert_released()

    def test_protocol_order_enforced(self):
        envelope = smoke.build_envelope("lumen-phys-test")
        fresh = {"opened": False, "closed": False, "exposures_set": 0, "measurements": 0}
        opened = dict(fresh, opened=True)
        cases = [
            (fresh, call("camera_measure"), "before camera_open"),
            (fresh, call("camera_set_exposure", exposure_us=1000.0), "before camera_open"),
            (opened, call("camera_open"), "already done"),
            (opened, call("camera_measure"), "needs a newly set exposure"),
            (dict(opened, exposures_set=1), call("camera_set_exposure", exposure_us=5000.0),
             "not been measured yet"),
            (dict(opened, exposures_set=1, measurements=1), call("camera_measure"),
             "needs a newly set exposure"),
            (dict(opened, exposures_set=2, measurements=2), call("camera_set_exposure", exposure_us=1000.0),
             "no measurement left"),
            (dict(opened, closed=True), call("camera_open"), "after camera_close"),
        ]
        for state, raw, expected in cases:
            with self.subTest(raw=raw, state=state):
                self.assertIn(expected, cli.protocol_problem(eml.parse_decision(raw), state, envelope))
        self.assertEqual(cli.protocol_problem(eml.parse_decision(finish()), fresh, envelope), "")

    def test_model_cannot_call_unauthorized_tool(self):
        for raw, expected in (
                (json.dumps({"action": "call_tool", "tool": "os_system", "arguments": {}}),
                 "invalid_decision"),
                (json.dumps({"action": "call_tool", "tool": "camera_open", "arguments": {},
                             "session_id": "x"}), "invalid_decision"),
                (call("camera_info"), "provider_error")):
            with self.subTest(raw=raw):
                self._reset_process_state()
                self.make_fake()
                self.stdout = smoke_tests.TTY()
                execute = self.spy_execute()
                code = self.run_cli(model=ScriptedModel([raw]))
                self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
                self.assertIn(f"executor status     : {expected}", self.output())
                execute.assert_not_called()
                self.assertEqual(self.fake.calls, [])

    def test_guard_forwards_model_reply_unchanged(self):
        envelope = smoke.build_envelope("lumen-phys-test")
        raw = '{"arguments": {}, "tool": "camera_open", "action": "call_tool"}'
        inner = mock.Mock()
        inner.call.return_value = raw
        guard = cli.FirstRunDecisionGuard(inner, envelope)
        context = json.dumps({"research_goal": "g", "tools": [], "history": [],
                              "latest_measurement": None, "decision_budget": {}})
        self.assertIs(guard.call("instr", context), raw)
        inner.call.assert_called_once_with("instr" + cli.FIRST_RUN_RULES, context)
        self.assertEqual(guard.records[0]["trusted_check"], cli.CHECK_PASSED)


# 26-31 and further failure paths: every exit shuts down.
class FailureTests(AdaptiveTestCase):
    def test_malformed_json_decision_fails_closed(self):
        for raw in ("not json", '{"action": "call_tool"', "[]", '{"action": "jump"}'):
            with self.subTest(raw=raw):
                self._reset_process_state()
                self.make_fake()
                self.stdout = smoke_tests.TTY()
                code = self.run_cli(model=ScriptedModel([call("camera_open"), raw]))
                self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
                self.assertIn("executor status     : invalid_decision", self.output())
                self.assertIn("MALFORMED", self.output())
                self.assertEqual(len(self.requests()), 2)
                self.assert_released()

    def test_api_error_triggers_trusted_shutdown_without_leaking_text(self):
        model = ScriptedModel([call("camera_open"), call("camera_set_exposure", exposure_us=1000.0),
                               RuntimeError(f"upstream said {SECRET}")])
        code = self.run_cli(model=model)
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        out = self.output()
        self.assertNotIn(SECRET, out)
        self.assertIn("RuntimeError (details withheld)", out)
        self.assertIn("executor status     : provider_error", out)
        self.assertEqual(len(self.requests()), 3)
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])  # restored at trusted shutdown
        self.assertNotIn("IMV_GetFrame", self.fake.names())
        self.assert_released()

    def test_model_status_not_completed_fails_closed(self):
        class Incomplete(FakeResponses):
            def create(self, **kwargs):
                self.requests.append(kwargs)
                return types.SimpleNamespace(status="incomplete", output_text=call("camera_open"))

        with mock.patch(__name__ + ".FakeResponses", Incomplete):
            code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("ModelOutputError", self.output())
        self.assertEqual(self.fake.calls, [])

    def test_missing_key_or_sdk_fails_closed_before_any_request(self):
        for error in (Exception(f"api_key must be set {SECRET}"), ModuleNotFoundError("openai")):
            with self.subTest(error=type(error).__name__):
                self._reset_process_state()
                self.make_fake()
                self.stdout = smoke_tests.TTY()
                code = self.run_cli(client_error=error)
                self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
                self.assertIn("MODEL_CLIENT_UNAVAILABLE", self.output())
                self.assertNotIn(SECRET, self.output())
                self.assertEqual(self.requests(), [])
                self.assert_no_backend_activity()
                self.assertIsNone(phys._device_owner)

    def test_early_finish_is_incomplete(self):
        cases = (
            [finish("nothing to do")],
            [call("camera_open"), call("camera_set_exposure", exposure_us=1000.0),
             call("camera_measure"), finish("one is enough")],
            [call("camera_open"), call("camera_set_exposure", exposure_us=1000.0),
             call("camera_measure"), call("camera_set_exposure", exposure_us=5000.0),
             call("camera_measure"), finish("no close")],
        )
        for replies in cases:
            with self.subTest(replies=len(replies)):
                self._reset_process_state()
                self.make_fake()
                self.stdout = smoke_tests.TTY()
                code = self.run_cli(model=ScriptedModel(replies))
                self.assert_verdict(code, cli.EXIT_INCOMPLETE, "INCOMPLETE")
                self.assertIn("host shutdown       : OK", self.output())
                self.assertIsNone(phys._device_owner)
                if len(replies) > 1:
                    self.assert_released()

    def test_decision_budget_exhaustion_fails_and_shuts_down(self):
        with mock.patch.object(cli, "MAX_MODEL_DECISIONS", 3):
            code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("executor status     : decision_budget_exhausted", self.output())
        self.assertEqual(len(self.requests()), 3)
        self.assert_released()

    def test_no_synthetic_substitution(self):
        real = ac.AuthorizedCameraRuntime._camera_measure

        def lying_measure(runtime, grant):
            result = real(runtime, grant)
            result.update(source="synthetic", synthetic=True)
            return result

        with mock.patch.object(ac.AuthorizedCameraRuntime, "_camera_measure", autospec=True,
                               side_effect=lying_measure):
            code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("VALIDATION_FAILED", self.output())
        self.assertIn("only hardware is accepted", self.output())
        self.assert_released()

    def test_mcp_error_payload_fails_without_retry(self):
        self.make_fake(codes={"IMV_GetFrame": -119})
        code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("executor status     : tool_error", self.output())
        self.assertIn("GET_FRAME_FAILED", self.output())
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)
        self.assertEqual(len(self.requests()), 3)  # no decision after the failed measurement
        self.assert_released()

    def test_interrupt_during_model_call_shuts_down(self):
        model = ScriptedModel([call("camera_open"), call("camera_set_exposure", exposure_us=1000.0),
                               call("camera_measure"), KeyboardInterrupt()])
        code = self.run_cli(model=model)
        self.assert_verdict(code, cli.EXIT_INTERRUPTED, "FAIL")
        self.assertIn("INTERRUPTED", self.output())
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
        self.assert_released()

    def test_session_expiry_stops_and_shuts_down(self):
        clock = [0.0]
        real_host = phys.PhysicalApprovedSessionHost

        def host_with_clock(envelope, sdk):
            return real_host(envelope, sdk, clock=lambda: clock[0])

        self.make_fake(on_get_frame=lambda: clock.__setitem__(0, 181.0))
        with mock.patch.object(cli.phys, "PhysicalApprovedSessionHost", side_effect=host_with_clock):
            code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("expired", self.output())
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)
        self.assert_released()

    def test_restoration_failure_prevents_pass(self):
        self.make_fake(codes={WRITE: [0, 0, -101]})
        code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_CLEANUP_FAILED, "FAIL")
        out = self.output()
        self.assertIn("CLEANUP_ERROR", out)
        self.assertIn("CAMERA STATE UNCERTAIN", out)
        self.assertIn("device ownership    : STILL CLAIMED", out)
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0, 5000.0])  # one restoration attempt

    def test_native_close_failure_prevents_pass_and_is_not_repeated(self):
        self.make_fake(codes={"IMV_Close": -101})
        code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_CLEANUP_FAILED, "FAIL")
        self.assertEqual(self.fake.counts["IMV_Close"], 1)
        self.assertEqual(self.fake.counts["IMV_DestroyHandle"], 1)

    def test_shutdown_failure_after_successful_run_prevents_pass(self):
        real_shutdown = phys.PhysicalApprovedSessionHost.shutdown

        def failing_shutdown(host):
            real_shutdown(host)
            raise ac.CleanupError("simulated host shutdown failure")

        with mock.patch.object(phys.PhysicalApprovedSessionHost, "shutdown", autospec=True,
                               side_effect=failing_shutdown):
            code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_CLEANUP_FAILED, "FAIL")
        self.assertIn("CLEANUP_ERROR       : CleanupError: simulated host shutdown failure",
                      self.output())

    def test_native_close_and_destroy_not_duplicated_on_success(self):
        code = self.run_cli()
        self.assertEqual(code, cli.EXIT_PASS, self.output())
        self.assert_released()
        self.assertIn("host shutdown       : OK by AdaptiveMCPExecutor", self.output())

    def test_unexpected_error_still_shuts_down(self):
        with mock.patch.object(cli, "AdaptiveMCPExecutor", side_effect=RuntimeError("boom")):
            code = self.run_cli()
        self.assert_verdict(code, cli.EXIT_FAILED, "FAIL")
        self.assertIn("UNEXPECTED_ERROR: RuntimeError (details withheld)", self.output())
        self.assertIn("host shutdown       : OK by trusted CLI", self.output())
        self.assertIsNone(phys._device_owner)


if __name__ == "__main__":
    unittest.main()
