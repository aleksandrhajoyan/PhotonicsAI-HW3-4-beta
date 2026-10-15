"""Offline tests for hw3/physical_lumen_cli.py.

main() runs end to end against the REAL HW1 four-role LangGraph (via the REAL
LumenAgentSystem adapter), the REAL executor bridge, AdaptiveMCPExecutor,
FirstRunDecisionGuard, BoundedDecisionProvider, LumenOpenAIModelCaller,
create_live_openai_client, LangGraph HITL approval, SessionGate,
PhysicalApprovedSessionHost, authorized MCP server and IRaypleCamera. Fakes
(from the D5A harness in test_physical_adaptive_cli):
    - the vendor SDK binding,
    - the ``openai`` module in sys.modules; its client answers each request with
      a FAKE model chosen by the request's instructions (orchestrator, scientist,
      general or hardware decision maker), deciding only from the text it is sent,
    - terminal input (a simulated operator).
No camera, vendor DLL, network, API key, paid API or real LLM is involved.
"""

import ast
import importlib
import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorized_camera as ac  # noqa: E402
import executor_dialogue_bridge as edb  # noqa: E402
import executor_mcp_loop as eml  # noqa: E402
import irayple_camera as ic  # noqa: E402
import lumen_agent_system as las  # noqa: E402
import openai_model_caller as omc  # noqa: E402
import physical_adaptive_cli as adaptive  # noqa: E402
import physical_approved_session_host as phys  # noqa: E402
import physical_lumen_cli as lumen  # noqa: E402
import physical_mcp_smoke as smoke  # noqa: E402
import test_physical_adaptive_cli as at  # noqa: E402
import test_physical_mcp_smoke as smoke_tests  # noqa: E402

MODULE_PATH = HW3_DIR / "physical_lumen_cli.py"
FLAG = "--physical-lumen-check"
ARGV = (FLAG, "--model", at.MODEL_ID)
ANSWERS = (lumen.CONFIRMATION_PHRASE, at.approve_shown_session, at.consent_shown_session)
WRITE = smoke_tests.WRITE
call, finish = at.call, at.finish


def route(role, instruction="please proceed", reason="the dialogue needs this role"):
    return json.dumps({"next_role": role, "instruction": instruction, "reason": reason})


def dialogue_of(input_text):
    return input_text.split("DIALOGUE:\n", 1)[1]


class ReasoningOrchestrator:
    """FAKE orchestrator deciding from the dialogue it is shown (no fixed sequence)."""

    def __call__(self, input_text):
        dialogue = dialogue_of(input_text)
        last = {role: dialogue.rfind(f"[{role.upper()}]") for role in ("scientist", "general", "executor")}
        if last["scientist"] < 0:
            return route("scientist", "state a hypothesis about exposure and mean DN")
        if last["general"] < 0:
            return route("general", "assess the safety constraints")
        if last["executor"] < 0:
            return route("executor", "run the approved camera experiment")
        if max(last["scientist"], last["general"]) < last["executor"]:
            return route("scientist", "conclude from the measured digital numbers only")
        return route("finish", "done", "the conclusion is written")


def grounded_scientist(input_text):
    dialogue = dialogue_of(input_text)
    if "[EXECUTOR]" not in dialogue:
        return "Hypothesis: mean DN grows with exposure; aim near 70 DN without saturation."
    means = re.findall(r"mean_dn=([0-9.]+)", dialogue)
    return f"Conclusion: the latest measured mean was {float(means[-1]):.1f} DN (raw, uncalibrated)."


def general_agent(input_text):
    return "General: exposure stays within 1000-5000 us; two frames at most; restore at close."


class RoleModel:
    """FAKE models; the role is identified by the request's instructions, as the real API sees it."""

    def __init__(self, orchestrator=None, scientist=None, general=None, hardware=None):
        self.orchestrator = orchestrator or ReasoningOrchestrator()
        self.scientist = scientist or grounded_scientist
        self.general = general or general_agent
        self.hardware = hardware or at.AdaptiveModel()
        self.roles = []

    def __call__(self, request):
        instructions = request["instructions"]
        if instructions == las.ORCHESTRATOR_INSTRUCTIONS:
            role, reply = "orchestrator", self.orchestrator
        elif instructions == las.SCIENTIST_INSTRUCTIONS:
            role, reply = "scientist", self.scientist
        elif instructions == las.GENERAL_INSTRUCTIONS:
            role, reply = "general", self.general
        else:
            self.roles.append("hardware")
            return self.hardware(request)
        self.roles.append(role)
        out = reply(request["input"])
        if isinstance(out, BaseException):
            raise out
        return out


def scripted(replies):
    replies = list(replies)

    def reply(_input_text):
        return replies.pop(0)
    return reply


class LumenTestCase(at.AdaptiveTestCase):
    def run_cli(self, answers=ANSWERS, argv=ARGV, model=None, stdin_tty=True, stdout_tty=True,
                client_error=None):
        self.model = RoleModel() if model is None else model
        self.client = at.FakeOpenAIClient(self.model)
        self.openai_module = at.types.ModuleType("openai")
        self.openai_module.OpenAI = mock.Mock(return_value=self.client, side_effect=client_error)
        self.operator = smoke_tests.Operator(self, answers)
        stdin = smoke_tests.TTY() if stdin_tty else smoke_tests.io.StringIO()
        if not stdout_tty:
            self.stdout = smoke_tests.io.StringIO()
        with mock.patch.dict(sys.modules, {"openai": self.openai_module}), \
                mock.patch("sys.stdin", stdin), mock.patch("sys.stdout", self.stdout), \
                mock.patch("builtins.input", side_effect=self.operator):
            return lumen.main(list(argv))

    def assert_verdict(self, code, expected_code, verdict):
        self.assertEqual(code, expected_code, self.output())
        self.assertIn(f"PHYSICAL_LUMEN_CHECK: {verdict}", self.output())
        if verdict != "PASS":
            self.assertNotIn("PHYSICAL_LUMEN_CHECK: PASS", self.output())

    def capture(self, name):
        """Run with a constructor of `name` in physical_lumen_cli's view wrapped to keep its instances."""
        owner = {"LumenAgentSystem": las, "PhysicalApprovedSessionHost": phys}[name]
        real, made = getattr(owner, name), []

        def construct(*args, **kwargs):
            made.append(real(*args, **kwargs))
            return made[-1]

        patcher = mock.patch.object(owner, name, side_effect=construct)
        patcher.start()
        self.addCleanup(patcher.stop)
        return made


# 1, 30. Import safety and isolation.
class ImportAndIsolationTests(unittest.TestCase):
    def test_import_has_no_hardware_or_api_effect(self):
        with mock.patch.object(ic.discovery, "load_vendor_sdk") as loader, \
                mock.patch.object(ic, "preload_vendor_sdk") as preload, \
                mock.patch.object(ic, "IRaypleCamera") as backend, \
                mock.patch.object(phys, "PhysicalApprovedSessionHost") as host, \
                mock.patch.object(omc, "create_live_openai_client") as factory, \
                mock.patch.object(omc, "LumenOpenAIModelCaller") as caller, \
                mock.patch("ctypes.WinDLL", create=True) as win_dll, \
                mock.patch("os.chdir") as chdir:
            importlib.reload(lumen)
        for patched in (loader, preload, backend, host, factory, caller, win_dll, chdir):
            patched.assert_not_called()
        self.assertNotIn("IMVApi", sys.modules)

    def test_source_reads_no_credentials_and_never_touches_backend(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in ("environ", "getenv", "api_key", "OPENAI_API_KEY", "FakeCamera",
                          "fake_camera", "run_lumen_session", "OfflineApprovedSessionHost("):
            self.assertNotIn(forbidden, source)
        tree = ast.parse(source)
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for name in ("execute", "capture_frame", "set_exposure_us", "IRaypleCamera",
                     "AuthorizedCameraRuntime", "create_authorized_server", "submit_decision",
                     "begin_action", "_runtime", "_camera", "call_tool", "Thread", "to_thread",
                     "preload_vendor_sdk", "run_adaptive_experiment", "AdaptiveMCPExecutor",
                     "add_node", "add_edge", "StateGraph"):
            self.assertNotIn(name, attributes | names)


# 2-8. Entry point, confirmations, approval, paid-API consent.
class EntryPointTests(LumenTestCase):
    def test_missing_flag_or_model_refuses(self):
        for argv in ((), ("--model", at.MODEL_ID), ("--physical-adaptive-check",), (FLAG,),
                     (FLAG, "--model", "")):
            with self.subTest(argv=argv):
                code = self.run_cli(argv=argv)
                self.assertEqual(code, lumen.EXIT_REFUSED)
                self.assertEqual(self.operator.prompts, [])
                self.assert_no_model_request()
        self.preload_mock.assert_not_called()

    def test_noninteractive_refuses(self):
        for stdin_tty, stdout_tty in ((False, True), (True, False)):
            with self.subTest(stdin_tty=stdin_tty):
                code = self.run_cli(stdin_tty=stdin_tty, stdout_tty=stdout_tty)
                self.assertEqual(code, lumen.EXIT_NOT_INTERACTIVE)
                self.assert_no_model_request()
        self.preload_mock.assert_not_called()

    def test_exact_first_confirmation_required(self):
        for answer in ("BEGIN ADAPTIVE HARDWARE CHECK", "begin four agent hardware check",
                       lumen.CONFIRMATION_PHRASE + " ", "", EOFError(), KeyboardInterrupt()):
            with self.subTest(answer=repr(answer)):
                code = self.run_cli(answers=(answer,))
                self.assertEqual(code, lumen.EXIT_REFUSED)
                self.assert_no_model_request()
        self.preload_mock.assert_not_called()

    def test_real_hitl_approval_required_and_refusal_means_no_camera(self):
        for answer in (EOFError(), "DENY", "APPROVE lumen-phys-" + "0" * 32, at.consent_shown_session,
                       lambda s: at.approve_shown_session(s) + " "):
            with self.subTest(answer=repr(answer) if not callable(answer) else "derived"):
                self._reset_process_state()
                self.make_fake()
                self.stdout = smoke_tests.TTY()
                code = self.run_cli(answers=(lumen.CONFIRMATION_PHRASE, answer, at.consent_shown_session))
                self.assert_verdict(code, lumen.EXIT_APPROVAL_REFUSED, "FAIL")
                self.assertIn("pending in LangGraph", self.output())
                self.assertEqual(len(self.operator.prompts), 2)
                self.assertEqual(self.fake.calls, [])  # zero camera operations
                self.assert_no_model_request()
                self.assertIsNone(phys._device_owner)

    def test_paid_api_consent_required_and_refusal_means_no_model_call(self):
        for answer in ("DENY", "AUTHORIZE PAID API", EOFError(), at.approve_shown_session,
                       lambda s: at.consent_shown_session(s).lower()):
            with self.subTest(answer=repr(answer) if not callable(answer) else "derived"):
                self._reset_process_state()
                self.make_fake()
                self.stdout = smoke_tests.TTY()
                execute = self.spy_execute()
                code = self.run_cli(answers=(lumen.CONFIRMATION_PHRASE, at.approve_shown_session, answer))
                self.assert_verdict(code, lumen.EXIT_API_CONSENT_REFUSED, "FAIL")
                self.assertIn("paid-API consent    : refused", self.output())
                self.assert_no_model_request()
                execute.assert_not_called()
                self.assert_no_backend_activity()
                self.assertIsNone(phys._device_owner)

    def test_consent_states_separate_budgets_without_spending_cap_claim(self):
        code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        out = self.output()
        self.assertIn(f"outer agents      : at most {lumen.MAX_OUTER_CALLS} requests", out)
        self.assertIn(f"hardware Executor : at most {adaptive.MAX_MODEL_DECISIONS} requests", out)
        self.assertIn("not a monetary spending cap", out)
        self.assertLess(out.index("Approval phrase received"), out.index("PAID MODEL API USE"))

    def test_api_key_never_printed(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": at.SECRET}):
            code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertNotIn(at.SECRET, self.output())
        self.openai_module.OpenAI.assert_called_once_with(max_retries=0, timeout=30.0)


# 9-18. Wiring of the original architecture and the physical path.
class ArchitectureTests(LumenTestCase):
    def test_sdk_preloaded_before_event_loop_and_tools_run_on_that_thread(self):
        real_run = edb.asyncio.run

        def run(coro, *args, **kwargs):
            self.events.append(("event_loop", smoke_tests.threading.get_ident()))
            return real_run(coro, *args, **kwargs)

        with mock.patch.object(edb.asyncio, "run", side_effect=run):
            code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertEqual([event[0] for event in self.events],
                         ["input", "preload", "input", "input", "event_loop"])
        preload_thread = self.events[1][2]
        self.assertIs(self.events[1][1], False)
        self.assertEqual(preload_thread, smoke_tests.threading.main_thread().ident)
        self.assertEqual(self.events[4][1], preload_thread)  # LangGraph ran the Executor inline
        self.assertTrue(self.fake.contexts)
        for name, thread, loop in self.fake.contexts:
            self.assertEqual(thread, preload_thread, name)
            self.assertIsNotNone(loop, name)
        smoke_tests.anyio.to_thread.run_sync.assert_not_called()

    def test_original_hw1_class_and_graph_preserved(self):
        hw1 = sys.modules["agentic_system"]
        self.assertIs(las.AgenticSilverboxSystem, hw1.AgenticSilverboxSystem)
        self.assertIs(las.LumenAgentSystem.build, hw1.AgenticSilverboxSystem.build)
        self.assertIs(las.LumenAgentSystem.route_after_specialist,
                      hw1.AgenticSilverboxSystem.route_after_specialist)
        self.assertEqual(las.MAX_DECISIONS, hw1.MAX_DECISIONS)
        real_class = las.LumenAgentSystem
        systems = self.capture("LumenAgentSystem")
        code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertEqual(len(systems), 1)
        self.assertIs(type(systems[0]), real_class)
        self.assertIsInstance(systems[0], hw1.AgenticSilverboxSystem)
        self.assertIsInstance(systems[0].llm, lumen.OuterCallBudget)
        self.assertIsNone(systems[0].coding_tool)

    def test_four_roles_participate_and_conclusion_follows_measurements(self):
        code = self.run_cli()
        self.assert_verdict(code, lumen.EXIT_PASS, "PASS")
        out = self.output()
        self.assertIn("role visits (order) : orchestrator -> scientist -> orchestrator -> general -> "
                      "orchestrator -> executor -> orchestrator -> scientist -> orchestrator", out)
        self.assertIn("four roles          : all took part", out)
        self.assertIn("conclusion          : by scientist after the Executor turn "
                      "(quotes a measured mean", out)
        self.assertIn("outer API requests  : 8 (limit 16)", out)
        self.assertIn("API requests        : 7 (limit 7, no retries)", out)
        self.assertEqual(self.model.roles.count("hardware"), 7)

    def test_orchestrator_limit_enforced(self):
        model = RoleModel(orchestrator=lambda text: route("general"))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_INCOMPLETE, "INCOMPLETE")
        self.assertEqual(model.roles.count("orchestrator"), las.MAX_DECISIONS)
        self.assertEqual(model.roles.count("general"), las.MAX_DECISIONS)
        self.assertIn("orchestrator turns  : 8 (limit 8)", self.output())
        self.assertIn("never dispatched the Executor", self.output())
        self.assertEqual(self.fake.calls, [])
        self.assertIsNone(phys._device_owner)

    def test_scientist_and_general_cannot_invoke_camera(self):
        tool_text = call("camera_open") + "\n" + call("camera_set_exposure", exposure_us=5000.0)
        execute = self.spy_execute()
        model = RoleModel(orchestrator=scripted([route("scientist"), route("general"),
                                                 route("finish")]),
                          scientist=lambda text: tool_text, general=lambda text: tool_text)
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_INCOMPLETE, "INCOMPLETE")
        execute.assert_not_called()
        self.assertEqual(self.fake.calls, [])
        self.assertNotIn("hardware", model.roles)

    def test_executor_uses_existing_bridge_once_with_guarded_provider(self):
        real = las.run_adaptive_experiment
        with mock.patch.object(las, "run_adaptive_experiment", side_effect=real) as bridge:
            hosts = self.capture("PhysicalApprovedSessionHost")
            code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        bridge.assert_called_once()
        host, provider, goal, budget = bridge.call_args.args
        self.assertIs(host, hosts[0])
        self.assertIsInstance(provider, adaptive.FirstRunDecisionGuard)
        self.assertIsInstance(provider._provider, lumen.BoundedDecisionProvider)
        self.assertEqual((goal, budget), (lumen.HARDWARE_GOAL, adaptive.MAX_MODEL_DECISIONS))

    def test_existing_adaptive_executor_reused(self):
        self.assertIs(edb.AdaptiveMCPExecutor, eml.AdaptiveMCPExecutor)
        original = eml.AdaptiveMCPExecutor.run
        with mock.patch.object(eml.AdaptiveMCPExecutor, "run", autospec=True, side_effect=original) as run:
            code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertEqual(run.call_count, 1)

    def test_physical_guard_remains_active(self):
        execute = self.spy_execute()
        model = RoleModel(hardware=at.ScriptedModel([call("camera_open"),
                                                     call("camera_set_exposure", exposure_us=3000.0)]))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertIn("not a first-run candidate", self.output())
        self.assertEqual(self.executed_tools(execute), ["camera_open"])
        self.assertEqual(self.fake.writes(), [])
        self.assert_released()

    def test_approved_envelope_identity_retained(self):
        envelopes = []
        real_build = smoke.build_envelope

        def build(session_id):
            envelopes.append(real_build(session_id))
            return envelopes[-1]

        hosts = self.capture("PhysicalApprovedSessionHost")
        with mock.patch.object(lumen.smoke, "build_envelope", side_effect=build):
            code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertEqual((len(envelopes), len(hosts)), (1, 1))
        self.assertIs(hosts[0]._envelope, envelopes[0])
        self.assertIs(hosts[0]._runtime._gate.envelope, envelopes[0])
        self.assertIn(envelopes[0].canonical_json(), self.output())
        self.assertIn(envelopes[0].sha256(), self.output())

    def test_camera_opens_only_through_authorized_mcp(self):
        opened = []
        real_open = ic.IRaypleCamera.open
        real_dispatch = ac.AuthorizedCameraRuntime._camera_open

        def dispatch(runtime, grant):
            opened.append(("runtime", grant.tool))
            return real_dispatch(runtime, grant)

        def camera_open(camera):
            opened.append(("camera", None))
            return real_open(camera)

        with mock.patch.object(ac.AuthorizedCameraRuntime, "_camera_open", autospec=True,
                               side_effect=dispatch), \
                mock.patch.object(ic.IRaypleCamera, "open", autospec=True, side_effect=camera_open):
            code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertEqual(opened, [("runtime", "camera_open"), ("camera", None)])
        self.assertEqual(self.fake.counts["IMV_Open"], 1)


# 19-22. Measurements and feedback.
class MeasurementTests(LumenTestCase):
    def test_two_hardware_measurements_with_frame_ids_1_and_2(self):
        code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        out = self.output()
        self.assertIn("measurement 1: source=hardware frame_id=1 exposure=1000.0 us 1920x1200 Mono8", out)
        self.assertIn("measurement 2: source=hardware frame_id=2 exposure=5000.0 us 1920x1200 Mono8", out)
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 2)
        self.assertEqual(self.fake.exposure_at_frame, [1000.0, 5000.0])

    def test_measurement_one_feeds_the_second_exposure_decision(self):
        code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertEqual(self.model.hardware.seen, [None, None, None, 1, 1, 2, 2])
        self.assertIn("feedback evidence   : decision 4 (event ", self.output())
        self.assertIn("was requested after measurement frame_id 1", self.output())

    def test_feedback_dependent_choice(self):
        self.make_fake(payloads=(at.BRIGHT_FRAME, smoke_tests.FRAME_2))
        code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertEqual(self.fake.exposure_at_frame, [1000.0, 1000.0])

    def test_mcp_results_validated(self):
        real = ac.AuthorizedCameraRuntime._camera_measure

        def renumbered(runtime, grant):
            return dict(real(runtime, grant), frame_id=1)

        with mock.patch.object(ac.AuthorizedCameraRuntime, "_camera_measure", autospec=True,
                               side_effect=renumbered):
            code = self.run_cli()
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertIn("VALIDATION_FAILED", self.output())
        self.assertIn("frame_id 1, expected 2", self.output())
        self.assert_released()


# 23-29 and further failure paths.
class FailureTests(LumenTestCase):
    def test_hardware_model_finish_before_completion_is_incomplete(self):
        model = RoleModel(hardware=at.ScriptedModel([
            call("camera_open"), call("camera_set_exposure", exposure_us=1000.0),
            call("camera_measure"), finish("one is enough")]))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_INCOMPLETE, "INCOMPLETE")
        self.assertIn("physical experiment INCOMPLETE", self.output())
        self.assert_released()

    def test_missing_role_is_incomplete(self):
        model = RoleModel(orchestrator=scripted([route("scientist"), route("executor"),
                                                 route("scientist"), route("finish")]))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_INCOMPLETE, "INCOMPLETE")
        self.assertIn("not visited ['general']", self.output())
        self.assert_released()

    def test_missing_post_measurement_conclusion_is_incomplete(self):
        model = RoleModel(orchestrator=scripted([route("scientist"), route("general"),
                                                 route("executor"), route("finish")]))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_INCOMPLETE, "INCOMPLETE")
        self.assertIn("no Scientist/General conclusion after the measurements", self.output())

    def test_duplicate_executor_invocation_rejected(self):
        model = RoleModel(orchestrator=scripted([route("scientist"), route("general"),
                                                 route("executor"), route("executor"),
                                                 route("scientist")]))
        with mock.patch.object(las, "run_adaptive_experiment",
                               side_effect=las.run_adaptive_experiment) as bridge:
            code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        bridge.assert_called_once()
        self.assertIn("executor dispatches : ['result', 'host_consumed']", self.output())
        self.assertIn("DUPLICATE_DISPATCH", self.output())
        self.assertEqual(model.roles.count("orchestrator"), 4)  # stopped at once
        self.assertEqual(self.fake.counts["IMV_Open"], 1)  # never reopened
        self.assert_released()

    def test_no_synthetic_fallback(self):
        real = ac.AuthorizedCameraRuntime._camera_measure

        def lying(runtime, grant):
            result = real(runtime, grant)
            result.update(source="synthetic", synthetic=True)
            return result

        with mock.patch.object(ac.AuthorizedCameraRuntime, "_camera_measure", autospec=True,
                               side_effect=lying):
            code = self.run_cli()
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertIn("only hardware is accepted", self.output())
        self.assert_released()

    def test_hardware_model_error_triggers_shutdown(self):
        model = RoleModel(hardware=at.ScriptedModel([
            call("camera_open"), call("camera_set_exposure", exposure_us=1000.0),
            RuntimeError(f"upstream {at.SECRET}")]))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertNotIn(at.SECRET, self.output())
        self.assertIn("executor status     : provider_error", self.output())
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
        self.assert_released()

    def test_outer_model_error_triggers_shutdown_without_camera(self):
        model = RoleModel(orchestrator=lambda text: RuntimeError(f"timeout {at.SECRET}"))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertIn("UNEXPECTED_ERROR: RuntimeError (details withheld)", self.output())
        self.assertNotIn(at.SECRET, self.output())
        self.assertIn("host shutdown       : OK by trusted CLI", self.output())
        self.assertEqual(self.fake.calls, [])
        self.assertIsNone(phys._device_owner)

    def test_malformed_orchestrator_reply_fails_closed(self):
        for raw in ("not json", route("camera"), json.dumps({"next_role": "executor",
                                                              "instruction": "x", "reason": "y",
                                                              "approved": True})):
            with self.subTest(raw=raw):
                self._reset_process_state()
                self.make_fake()
                self.stdout = smoke_tests.TTY()
                code = self.run_cli(model=RoleModel(orchestrator=lambda text: raw))
                self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
                self.assertIn("ROUTING_ERROR", self.output())
                self.assertEqual(self.fake.calls, [])

    def test_outer_call_budget_enforced(self):
        caller = mock.Mock()
        caller.call.return_value = "ok"
        budget = lumen.OuterCallBudget(caller, max_calls=2, max_input_chars=100, max_output_chars=5)
        budget.call(las.SCIENTIST_INSTRUCTIONS[:10], "x")
        budget.call("y", "x")
        with self.assertRaises(lumen.OuterCallError):
            budget.call("y", "x")
        self.assertEqual(caller.call.call_count, 2)
        with self.assertRaises(lumen.OuterCallError):
            lumen.OuterCallBudget(caller, max_calls=5, max_input_chars=3,
                                  max_output_chars=5).call("long", "text")
        caller.call.return_value = "toolong"
        with self.assertRaises(lumen.OuterCallError):
            lumen.OuterCallBudget(caller, max_calls=5, max_input_chars=100,
                                  max_output_chars=5).call("a", "b")

    def test_oversized_outer_reply_fails_closed(self):
        model = RoleModel(scientist=lambda text: "x" * (lumen.OUTER_MAX_OUTPUT_CHARS + 1))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertIn("OUTER_MODEL_REFUSED", self.output())
        self.assertEqual(self.fake.calls, [])

    def test_missing_client_fails_closed_before_any_request(self):
        code = self.run_cli(client_error=Exception(f"api_key must be set {at.SECRET}"))
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertIn("MODEL_CLIENT_UNAVAILABLE", self.output())
        self.assertNotIn(at.SECRET, self.output())
        self.assertEqual(self.requests(), [])
        self.assert_no_backend_activity()

    def test_mcp_tool_error_fails_without_retry(self):
        self.make_fake(codes={"IMV_GetFrame": -119})
        code = self.run_cli()
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertIn("executor status     : tool_error", self.output())
        self.assertEqual(self.fake.counts["IMV_GetFrame"], 1)
        self.assert_released()

    def test_session_expiry_stops_and_shuts_down(self):
        clock = [0.0]
        real_host = phys.PhysicalApprovedSessionHost

        def host_with_clock(envelope, sdk):
            return real_host(envelope, sdk, clock=lambda: clock[0])

        self.make_fake(on_get_frame=lambda: clock.__setitem__(0, 181.0))
        with mock.patch.object(lumen.phys, "PhysicalApprovedSessionHost", side_effect=host_with_clock):
            code = self.run_cli()
        self.assert_verdict(code, lumen.EXIT_FAILED, "FAIL")
        self.assertIn("expired", self.output())
        self.assert_released()

    def test_keyboard_interrupt_shuts_down(self):
        model = RoleModel(hardware=at.ScriptedModel([
            call("camera_open"), call("camera_set_exposure", exposure_us=1000.0),
            call("camera_measure"), KeyboardInterrupt()]))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_INTERRUPTED, "FAIL")
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0])
        self.assert_released()

    def test_cleanup_failure_prevents_pass(self):
        self.make_fake(codes={WRITE: [0, 0, -101]})
        code = self.run_cli()
        self.assert_verdict(code, lumen.EXIT_CLEANUP_FAILED, "FAIL")
        self.assertIn("CAMERA STATE UNCERTAIN", self.output())
        self.assertIn("device ownership    : STILL CLAIMED", self.output())
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0, 5000.0])

    def test_native_close_failure_prevents_pass(self):
        self.make_fake(codes={"IMV_Close": -101})
        code = self.run_cli()
        self.assert_verdict(code, lumen.EXIT_CLEANUP_FAILED, "FAIL")
        self.assertEqual(self.fake.counts["IMV_Close"], 1)
        self.assertEqual(self.fake.counts["IMV_DestroyHandle"], 1)

    def test_original_exposure_restored(self):
        self.make_fake(exposure=(3000.0, 16.0, 1000000.0))
        code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assertEqual(self.fake.writes(), [1000.0, 5000.0, 3000.0])
        self.assertEqual(self.fake.current, 3000.0)

    def test_no_duplicate_native_close_or_destroy(self):
        code = self.run_cli()
        self.assertEqual(code, lumen.EXIT_PASS, self.output())
        self.assert_released()
        self.assertIn("host shutdown       : OK by AdaptiveMCPExecutor", self.output())

    def test_executor_never_dispatched_is_incomplete_and_shut_down(self):
        model = RoleModel(orchestrator=scripted([route("scientist"), route("general"),
                                                 route("finish")]))
        code = self.run_cli(model=model)
        self.assert_verdict(code, lumen.EXIT_INCOMPLETE, "INCOMPLETE")
        self.assertIn("  not dispatched", self.output())
        self.assertIn("host shutdown       : OK by trusted CLI", self.output())
        self.assertEqual(self.fake.calls, [])
        self.assertIsNone(phys._device_owner)


if __name__ == "__main__":
    unittest.main()
