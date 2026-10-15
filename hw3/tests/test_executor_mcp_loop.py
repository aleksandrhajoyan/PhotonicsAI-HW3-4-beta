"""Offline tests for hw3/executor_mcp_loop.py.

The decision providers here are deterministic TEST DOUBLES with the
ModelCaller.call(instructions, input_text) shape; no LLM API is called. The
"human" approval of each host is SIMULATED by this trusted test harness
(host.start + host.resume with a structured decision), never by the provider.
Every camera operation goes through the official MCP SDK Client against the
host's authorized in-process server. The only backend is FakeCamera (a
recording subclass of it per host); all data is synthetic. No camera, vendor
SDK or network is involved.
"""

import ast
import contextlib
import json
import math
import socket
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

from mcp import Client

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import approved_session_host as ash  # noqa: E402
import executor_mcp_loop as eml  # noqa: E402
from approved_session_host import HostError, OfflineApprovedSessionHost  # noqa: E402
from executor_mcp_loop import (  # noqa: E402
    DECISION_INSTRUCTIONS,
    AdaptiveMCPExecutor,
    DecisionError,
    ExecutorError,
    parse_decision,
)
from fake_camera import FakeCamera  # noqa: E402
from measurements import analyze_frame  # noqa: E402
from session_policy import SessionEnvelope, session_approval  # noqa: E402

MODULE_PATH = HW3_DIR / "executor_mcp_loop.py"
ALL_TOOLS = (
    "camera_info",
    "camera_open",
    "camera_set_exposure",
    "camera_measure",
    "camera_close",
)
GOAL = "Measure the synthetic beam and bring its mean brightness near half of full scale."


def call(tool, **arguments):
    return json.dumps({"action": "call_tool", "tool": tool, "arguments": arguments})


def finish(reason="Research objective achieved"):
    return json.dumps({"action": "finish", "reason": reason})


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
        self.faults = {}

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


class ScriptedProvider:
    """Test double: returns pre-written raw replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    @property
    def contexts(self):
        return [json.loads(text) for _, text in self.requests]

    def call(self, instructions, input_text):
        self.requests.append((instructions, input_text))
        if not self.replies:
            raise AssertionError("provider called more often than scripted")
        return self.replies.pop(0)


class ThresholdProvider:
    """Test double with its own feedback rule (synthetic demonstration only).

    open -> optional initial exposure -> measure; then, from the MEASURED
    metrics only: saturated -> halve exposure; mean below the target ->
    scale exposure toward the target; otherwise finish. One correction, one
    re-measurement, then finish.
    """

    TARGET_FRACTION = 0.5

    def __init__(self, initial_exposure_us=None):
        self.initial_exposure_us = initial_exposure_us
        self.requests = []
        self.chosen_exposures = []

    def call(self, instructions, input_text):
        self.requests.append((instructions, input_text))
        context = json.loads(input_text)
        done = [e["tool"] for e in context["history"]
                if e["type"] == "tool_call" and e["status"] == "success"]
        latest = context["latest_measurement"]
        if "camera_open" not in done:
            return call("camera_open")
        if self.initial_exposure_us is not None and "camera_set_exposure" not in done:
            return self.set_exposure(self.initial_exposure_us)
        if latest is None:
            return call("camera_measure")
        if done[-1] == "camera_set_exposure":
            return call("camera_measure")
        corrections = len(self.chosen_exposures) - (self.initial_exposure_us is not None)
        metrics = latest["metrics"]
        if corrections == 0:
            if metrics["saturation_fraction"] > 0:
                return self.set_exposure(round(latest["exposure_us"] / 2, 1))
            if metrics["mean_fraction_of_full_scale"] < self.TARGET_FRACTION - 0.05:
                scale = self.TARGET_FRACTION / metrics["mean_fraction_of_full_scale"]
                return self.set_exposure(round(latest["exposure_us"] * scale, 1))
        return finish(f"frame {latest['frame_id']} mean fraction "
                      f"{metrics['mean_fraction_of_full_scale']:.3f}")

    def set_exposure(self, value):
        self.chosen_exposures.append(value)
        return call("camera_set_exposure", exposure_us=value)


class RaisingProvider:
    def call(self, instructions, input_text):
        raise RuntimeError("provider unavailable")


@contextlib.contextmanager
def openai_sdk_guard(test):
    """Fail on any real OpenAI client construction, API request or network lookup.

    Patches the installed SDK's own entry points (never sys.modules): OpenAI and
    AsyncOpenAI construction, their request() transport and Responses.create(),
    plus DNS resolution and outbound connections. No client is ever constructed
    and no credential is read. Importing the SDK itself is allowed.
    """
    import openai
    from openai.resources.responses import AsyncResponses, Responses

    targets = ((openai.OpenAI, "__init__"), (openai.AsyncOpenAI, "__init__"),
               (openai.OpenAI, "request"), (openai.AsyncOpenAI, "request"),
               (Responses, "create"), (AsyncResponses, "create"),
               (socket, "getaddrinfo"), (socket, "create_connection"))
    with contextlib.ExitStack() as stack:
        spies = [stack.enter_context(mock.patch.object(
                     owner, name, side_effect=AssertionError(f"{owner.__name__}.{name}")))
                 for owner, name in targets]
        test.assertIs(openai.OpenAI.__init__, spies[0])  # the guard is live on the real SDK
        yield
    for (owner, name), spy in zip(targets, spies):
        test.assertEqual(spy.call_count, 0, f"{owner.__name__}.{name} was called")


def _envelope(**overrides):
    fields = {
        # A fresh, distinct session id for every fixture; no approval is shared.
        "session_id": f"session-exec-{uuid.uuid4().hex[:12]}",
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


def _reference_measurement(exposure_us):
    camera = FakeCamera()
    camera.open()
    camera.set_exposure_us(exposure_us)
    frame = camera.capture_frame()
    return analyze_frame(frame.pixels, frame.bit_depth, bins=16)


class ExecutorTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.object(ash, "FakeCamera", SpyCamera)
        patcher.start()
        self.addCleanup(patcher.stop)

    def approved_host(self, **envelope_overrides):
        """A distinct host per call, approved by the simulated human of this harness."""
        envelope = _envelope(**envelope_overrides)
        host = OfflineApprovedSessionHost(envelope, clock=Clock())
        host.start(f"thread-{uuid.uuid4().hex[:12]}")
        result = host.resume(session_approval(envelope.session_id, envelope.sha256(), True))
        self.assertTrue(result["activated"], result)
        return host

    async def run_executor(self, provider, host=None, max_decisions=10, goal=GOAL):
        host = host or self.approved_host()
        executor = AdaptiveMCPExecutor(host, provider, goal, max_decisions)
        result = await executor.run()
        json.dumps(result, allow_nan=False)  # must be JSON-compatible
        return host, result

    def tool_events(self, result):
        return [e for e in result["events"] if e["type"] == "tool_call"]

    def decision_events(self, result):
        return [e for e in result["events"] if e["type"] == "decision"]

    async def assert_rejected_before_dispatch(self, reply, prefix=()):
        provider = ScriptedProvider([*prefix, reply, finish()])
        original = Client.call_tool
        with mock.patch.object(Client, "call_tool", autospec=True, side_effect=original) as spy:
            host, result = await self.run_executor(provider)
        self.assertEqual(result["status"], eml.STATUS_INVALID_DECISION, result["final_reason"])
        self.assertEqual(spy.call_count, len(prefix))
        self.assertEqual(result["tool_call_count"], len(prefix))
        self.assertEqual(len(provider.requests), len(prefix) + 1)  # never asked again
        last = result["events"][-1]
        self.assertEqual(last["type"], "decision")
        self.assertIs(last["valid"], False)
        self.assertTrue(result["host_shut_down"])
        return host, result


# 1, 2 -------------------------------------------------------------------------

class HostTests(ExecutorTestCase):
    async def test_approved_host_supplies_mcp_server(self):
        host = self.approved_host()
        server = host.get_mcp_server()
        async with Client(server, mode="legacy", read_timeout_seconds=30) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
        self.assertEqual(names, set(ALL_TOOLS))
        provider = ScriptedProvider([call("camera_open"), call("camera_measure"), finish()])
        _, result = await self.run_executor(provider, host=host)
        self.assertEqual(result["status"], eml.STATUS_COMPLETED)

    async def test_unapproved_host_cannot_begin(self):
        never_started = OfflineApprovedSessionHost(_envelope(), clock=Clock())
        awaiting = OfflineApprovedSessionHost(_envelope(), clock=Clock())
        awaiting.start("thread-awaiting")
        refused_env = _envelope()
        refused = OfflineApprovedSessionHost(refused_env, clock=Clock())
        refused.start("thread-refused")
        refused.resume(session_approval(refused_env.session_id, refused_env.sha256(), False))
        for name, host in (("new", never_started), ("awaiting", awaiting), ("refused", refused)):
            with self.subTest(host=name):
                provider = ScriptedProvider([call("camera_open")])
                _, result = await self.run_executor(provider, host=host)
                self.assertEqual(result["status"], eml.STATUS_HOST_NOT_ACTIVE)
                self.assertEqual(provider.requests, [])
                self.assertEqual((result["decision_count"], result["tool_call_count"]), (0, 0))
                self.assertEqual(host._camera.calls, [])
                self.assertIsNone(host._runtime)
                self.assertTrue(result["host_shut_down"])

    def test_constructor_validation(self):
        host = OfflineApprovedSessionHost(_envelope(), clock=Clock())
        provider = ScriptedProvider([])
        for bad in (0, -1, True, False, 1.5, 3.0, "3", None, math.inf, math.nan):
            with self.subTest(max_decisions=bad), self.assertRaises((TypeError, ValueError)):
                AdaptiveMCPExecutor(host, provider, GOAL, bad)
        for bad in ("", "   ", None, 42):
            with self.subTest(goal=bad), self.assertRaises((TypeError, ValueError)):
                AdaptiveMCPExecutor(host, provider, bad, 3)
        for bad in (None, object(), lambda i, t: "{}"):
            with self.subTest(provider=bad), self.assertRaises(TypeError):
                AdaptiveMCPExecutor(host, bad, GOAL, 3)
        for bad in (host.__dict__, "host", None, FakeCamera()):
            with self.subTest(host=type(bad).__name__), self.assertRaises(TypeError):
                AdaptiveMCPExecutor(bad, provider, GOAL, 3)

    async def test_executor_is_single_use(self):
        executor = AdaptiveMCPExecutor(self.approved_host(), ScriptedProvider([finish()]), GOAL, 3)
        await executor.run()
        with self.assertRaises(ExecutorError):
            await executor.run()


# 3, 4, 5, 7, 8, 9, 13, 15, 16, 17 ---------------------------------------------

class LoopTests(ExecutorTestCase):
    async def test_provider_receives_research_goal_and_instructions(self):
        provider = ScriptedProvider([finish()])
        await self.run_executor(provider)
        instructions, _ = provider.requests[0]
        self.assertEqual(instructions, DECISION_INSTRUCTIONS)
        self.assertEqual(provider.contexts[0]["research_goal"], GOAL)
        self.assertIsNone(provider.contexts[0]["latest_measurement"])
        self.assertEqual(provider.contexts[0]["history"], [])

    async def test_provider_receives_tool_schemas(self):
        provider = ScriptedProvider([finish()])
        await self.run_executor(provider)
        tools = {tool["name"]: tool for tool in provider.contexts[0]["tools"]}
        self.assertEqual(set(tools), set(ALL_TOOLS))
        for name, tool in tools.items():
            with self.subTest(tool=name):
                self.assertTrue(tool["description"])
                expected = ["exposure_us"] if name == "camera_set_exposure" else []
                self.assertEqual(sorted(tool["input_schema"].get("properties", {})), expected)
        exposure = tools["camera_set_exposure"]["input_schema"]
        self.assertEqual(exposure["properties"]["exposure_us"]["type"], "number")
        self.assertEqual(exposure["required"], ["exposure_us"])

    async def test_valid_decisions_execute_through_real_mcp_client(self):
        provider = ScriptedProvider([
            call("camera_info"), call("camera_open"),
            call("camera_set_exposure", exposure_us=2000.0), call("camera_measure"),
            call("camera_close"), finish(),
        ])
        host = self.approved_host()
        runtime = host._runtime
        original = Client.call_tool
        with mock.patch.object(Client, "call_tool", autospec=True, side_effect=original) as client_spy, \
                mock.patch.object(runtime, "execute", wraps=runtime.execute) as runtime_spy:
            _, result = await self.run_executor(provider, host=host)
        expected = ["camera_info", "camera_open", "camera_set_exposure", "camera_measure",
                    "camera_close"]
        self.assertEqual([c.args[1] for c in client_spy.call_args_list], expected)
        self.assertEqual([c.args[1] for c in runtime_spy.call_args_list], expected)
        self.assertEqual(client_spy.call_args_list[2].args[2], {"exposure_us": 2000.0})
        self.assertEqual(result["status"], eml.STATUS_COMPLETED)
        self.assertEqual(host._camera.calls.count("capture_frame"), 1)

    async def test_provider_called_again_after_measurement_with_feedback(self):
        provider = ScriptedProvider([call("camera_open"), call("camera_measure"), finish()])
        _, result = await self.run_executor(provider)
        self.assertEqual(len(provider.requests), 3)
        after = provider.contexts[2]
        latest = after["latest_measurement"]
        self.assertIsNotNone(latest)
        self.assertEqual(latest["metrics"], _reference_measurement(1000.0))
        self.assertEqual(latest["frame_id"], 1)
        self.assertEqual(latest["source"], "synthetic")
        measured = [e for e in after["history"] if e.get("tool") == "camera_measure"]
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["frame_id"], 1)
        self.assertEqual(measured[0]["result"]["metrics"]["mean_dn"], latest["metrics"]["mean_dn"])
        # The context before the measurement had no measurement yet.
        self.assertIsNone(provider.contexts[1]["latest_measurement"])
        self.assertEqual(result["latest_measurement"]["frame_id"], 1)

    async def test_finish_terminates_loop(self):
        provider = ScriptedProvider([call("camera_open"), call("camera_measure"),
                                     finish("enough data"), call("camera_measure")])
        host, result = await self.run_executor(provider)
        self.assertEqual(result["status"], eml.STATUS_COMPLETED)
        self.assertEqual(result["final_reason"], "enough data")
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(len(provider.replies), 1)  # the trailing measure was never consumed
        self.assertEqual(host._camera.calls.count("capture_frame"), 1)

    async def test_counts_and_ordered_event_log(self):
        replies = [call("camera_open"), call("camera_measure"),
                   call("camera_set_exposure", exposure_us=1500.0), call("camera_measure"),
                   finish()]
        provider = ScriptedProvider(replies)
        _, result = await self.run_executor(provider, max_decisions=7)
        self.assertEqual(result["decision_count"], 5)
        self.assertEqual(result["max_decisions"], 7)
        self.assertEqual(result["tool_call_count"], 4)
        self.assertEqual(result["successful_measurement_count"], 2)
        self.assertEqual([e["sequence"] for e in result["events"]], list(range(1, 10)))
        self.assertEqual([e["type"] for e in result["events"]],
                         ["decision", "tool_call"] * 4 + ["decision"])
        decisions = self.decision_events(result)
        self.assertEqual([d["decision_number"] for d in decisions], [1, 2, 3, 4, 5])
        self.assertEqual([d["raw_output"] for d in decisions], replies)
        self.assertEqual(decisions[2]["decision"], {"action": "call_tool",
                                                   "tool": "camera_set_exposure",
                                                   "arguments": {"exposure_us": 1500.0}})
        tools = self.tool_events(result)
        self.assertEqual([t["tool"] for t in tools],
                         ["camera_open", "camera_measure", "camera_set_exposure", "camera_measure"])
        self.assertEqual([t["decision_number"] for t in tools], [1, 2, 3, 4])
        self.assertTrue(all(t["status"] == "success" for t in tools))
        self.assertEqual([t["frame_id"] for t in tools if t["tool"] == "camera_measure"], [1, 2])
        self.assertEqual(tools[2]["result"]["exposure_us"], 1500.0)
        for event in tools:
            self.assertEqual(event["result"]["tool"], event["tool"])
            self.assertTrue(event["result"]["action_id"].startswith("mcp-"))
            self.assertNotIn("pixels", json.dumps(event))
        budgets = [c["decision_budget"] for c in provider.contexts]
        self.assertEqual([b["remaining"] for b in budgets], [7, 6, 5, 4, 3])
        self.assertEqual([b["used"] for b in budgets], [0, 1, 2, 3, 4])

    async def test_history_given_to_provider_matches_log(self):
        provider = ScriptedProvider([call("camera_open"), call("camera_measure"), finish()])
        _, result = await self.run_executor(provider)
        self.assertEqual(provider.contexts[2]["history"], result["events"][:4])


# 10, 11, 12 --------------------------------------------------------------------

class FeedbackTests(ExecutorTestCase):
    async def test_dim_measurement_makes_provider_raise_exposure(self):
        provider = ThresholdProvider()
        host, result = await self.run_executor(provider)
        self.assertEqual(result["status"], eml.STATUS_COMPLETED)
        first_metrics = _reference_measurement(1000.0)
        expected = round(1000.0 * 0.5 / first_metrics["mean_fraction_of_full_scale"], 1)
        self.assertEqual(provider.chosen_exposures, [expected])
        self.assertNotEqual(expected, 1000.0)
        measures = [t for t in self.tool_events(result) if t["tool"] == "camera_measure"]
        self.assertEqual([m["result"]["exposure_us"] for m in measures], [1000.0, expected])
        self.assertEqual([m["frame_id"] for m in measures], [1, 2])
        self.assertEqual(measures[1]["result"]["metrics"], _reference_measurement(expected))
        self.assertGreater(measures[1]["result"]["metrics"]["mean_dn"],
                           measures[0]["result"]["metrics"]["mean_dn"])
        # The value the provider used came from what it was shown after frame 1.
        shown = [json.loads(t)["latest_measurement"] for _, t in provider.requests]
        self.assertIsNone(shown[1])  # the measure decision itself had no measurement yet
        self.assertEqual(shown[2]["frame_id"], 1)  # the exposure decision saw frame 1
        self.assertEqual(shown[2]["metrics"], first_metrics)
        self.assertEqual(shown[4]["frame_id"], 2)  # the finish decision saw frame 2

    async def test_saturated_measurement_makes_provider_lower_exposure(self):
        provider = ThresholdProvider(initial_exposure_us=3000.0)
        _, result = await self.run_executor(provider)
        self.assertEqual(result["status"], eml.STATUS_COMPLETED)
        self.assertGreater(_reference_measurement(3000.0)["saturation_fraction"], 0)
        self.assertEqual(provider.chosen_exposures, [3000.0, 1500.0])
        measures = [t for t in self.tool_events(result) if t["tool"] == "camera_measure"]
        self.assertEqual([m["result"]["exposure_us"] for m in measures], [3000.0, 1500.0])
        self.assertEqual(measures[1]["result"]["metrics"]["saturation_fraction"], 0.0)

    async def test_executor_never_sets_exposure_by_itself(self):
        provider = ScriptedProvider([call("camera_open"), call("camera_measure"),
                                     call("camera_measure"), finish()])
        host, result = await self.run_executor(provider)
        self.assertNotIn("set_exposure_us", host._camera.calls)
        self.assertEqual([t["tool"] for t in self.tool_events(result)],
                         ["camera_open", "camera_measure", "camera_measure"])
        self.assertEqual(result["latest_measurement"]["exposure_us"], 1000.0)

    async def test_provider_exposure_is_passed_through_unchanged(self):
        for value in (1234.5, 777, 100.0, 10000.0):
            with self.subTest(value=value):
                provider = ScriptedProvider([call("camera_open"),
                                             call("camera_set_exposure", exposure_us=value),
                                             call("camera_measure"), finish()])
                original = Client.call_tool
                with mock.patch.object(Client, "call_tool", autospec=True,
                                       side_effect=original) as spy:
                    _, result = await self.run_executor(provider)
                sent = spy.call_args_list[1].args[2]["exposure_us"]
                self.assertEqual(sent, value)
                self.assertIs(type(sent), type(value))
                self.assertEqual(result["latest_measurement"]["exposure_us"], float(value))

    async def test_out_of_range_exposure_is_not_clipped(self):
        provider = ScriptedProvider([call("camera_open"),
                                     call("camera_set_exposure", exposure_us=20000.0), finish()])
        host = self.approved_host(max_exposure_us=5000.0)
        _, result = await self.run_executor(provider, host=host)
        self.assertEqual(result["status"], eml.STATUS_TOOL_ERROR)
        self.assertIn("Not authorized", self.tool_events(result)[-1]["error"])
        self.assertNotIn("set_exposure_us", host._camera.calls)

    def test_no_exposure_numbers_in_executor_source(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        floats = [n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, float)]
        self.assertEqual(floats, [])
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for forbidden in ("random", "uniform", "randint", "linspace", "arange"):
            self.assertNotIn(forbidden, names)


# 14, 18 ------------------------------------------------------------------------

class CompletionAndProvenanceTests(ExecutorTestCase):
    async def test_finish_before_measurement_is_not_success(self):
        for replies in ([finish()], [call("camera_open"), finish()],
                        [call("camera_info"), call("camera_open"), finish("looks fine")]):
            with self.subTest(replies=len(replies)):
                _, result = await self.run_executor(ScriptedProvider(replies))
                self.assertEqual(result["status"], eml.STATUS_FINISHED_WITHOUT_MEASUREMENT)
                self.assertNotEqual(result["status"], eml.STATUS_COMPLETED)
                self.assertEqual(result["successful_measurement_count"], 0)
                self.assertIsNone(result["latest_measurement"])
                self.assertTrue(result["host_shut_down"])

    async def test_provenance_remains_synthetic(self):
        provider = ScriptedProvider([call("camera_info"), call("camera_open"),
                                     call("camera_measure"), finish()])
        _, result = await self.run_executor(provider)
        provenance = result["provenance"]
        self.assertEqual(provenance["sources_observed"], ["synthetic"])
        self.assertIs(provenance["synthetic_only"], True)
        self.assertIs(provenance["hardware_evidence"], False)
        self.assertIn("ScriptedProvider", provenance["decision_provider"])
        for event in self.tool_events(result):
            if event["tool"] in ("camera_info", "camera_measure"):
                self.assertEqual(event["source"], "synthetic")
                self.assertIs(event["result"]["synthetic"], True)
        self.assertEqual(result["latest_measurement"]["source"], "synthetic")
        self.assertEqual(result["research_goal"], GOAL)

    async def test_no_measurement_means_no_synthetic_claim(self):
        _, result = await self.run_executor(ScriptedProvider([finish()]))
        self.assertEqual(result["provenance"]["sources_observed"], [])
        self.assertIs(result["provenance"]["synthetic_only"], False)


# 19, 20, 21, 22 ---------------------------------------------------------------

class ValidationTests(ExecutorTestCase):
    async def test_unknown_tools_rejected_before_dispatch(self):
        for tool in ("shutdown", "approve_session", "submit_decision", "camera_reset",
                     "Camera_Measure", "camera_measure ", "", None, 3, ["camera_measure"]):
            with self.subTest(tool=tool):
                reply = json.dumps({"action": "call_tool", "tool": tool, "arguments": {}})
                host, _ = await self.assert_rejected_before_dispatch(reply)
                self.assertEqual(host._camera.calls, ["close"])  # only trusted shutdown

    async def test_malformed_output_rejected_before_dispatch(self):
        good = call("camera_measure")
        for reply in ("not json", "", "```json\n" + good + "\n```", good + " extra",
                      "[" + good + "]", '"camera_measure"', "null", "42",
                      None, {"action": "call_tool"}, b"{}", good[:-1],
                      '{"action": "call_tool", "tool": "camera_measure", "arguments": {}, '
                      '"action": "finish"}'):
            with self.subTest(reply=repr(reply)):
                await self.assert_rejected_before_dispatch(reply)

    async def test_wrong_structure_rejected(self):
        for decision in ({}, {"action": "measure"}, {"action": "CALL_TOOL", "tool": "camera_measure",
                                                    "arguments": {}},
                         {"action": "call_tool", "tool": "camera_measure"},
                         {"action": "call_tool", "tool": "camera_measure", "arguments": None},
                         {"action": "call_tool", "tool": "camera_measure", "arguments": []},
                         {"action": "call_tool", "tool": "camera_measure", "arguments": {},
                          "note": "also run code"},
                         {"action": "call_tool", "tool": "camera_open", "arguments": {"x": 1}},
                         {"action": "finish"}, {"action": "finish", "reason": ""},
                         {"action": "finish", "reason": "   "}, {"action": "finish", "reason": 1},
                         {"action": "finish", "reason": "ok", "next": "camera_measure"},
                         {"action": ["finish"], "reason": "ok"}):
            with self.subTest(decision=decision):
                await self.assert_rejected_before_dispatch(json.dumps(decision))

    async def test_invalid_exposure_rejected_before_dispatch(self):
        prefix = (call("camera_open"),)
        base = '{"action": "call_tool", "tool": "camera_set_exposure", "arguments": %s}'
        for arguments in ('{"exposure_us": "1000"}', '{"exposure_us": true}',
                          '{"exposure_us": false}', '{"exposure_us": null}',
                          '{"exposure_us": NaN}', '{"exposure_us": Infinity}',
                          '{"exposure_us": -Infinity}', '{"exposure_us": 1e999}',
                          '{"exposure_us": [1000]}', '{"exposure_us": {"value": 1000}}',
                          '{}', '{"exposure_us": 1000, "gain": 2}', '{"exposure": 1000}',
                          '{"exposure_us": ' + "9" * 400 + '}'):
            with self.subTest(arguments=arguments):
                host, _ = await self.assert_rejected_before_dispatch(base % arguments, prefix)
                self.assertNotIn("set_exposure_us", host._camera.calls)

    async def test_authority_fields_rejected(self):
        authority = {"session_id": "session-x", "action_id": "mcp-forged-1", "approval": True,
                     "approved": True, "approval_token": "yes", "envelope_sha256": "0" * 64,
                     "sha256": "0" * 64}
        for field, value in authority.items():
            with self.subTest(where="top", field=field):
                decision = {"action": "call_tool", "tool": "camera_measure", "arguments": {},
                            field: value}
                _, result = await self.assert_rejected_before_dispatch(json.dumps(decision))
                self.assertIn(field, result["final_reason"])
            with self.subTest(where="arguments", field=field):
                decision = {"action": "call_tool", "tool": "camera_open",
                            "arguments": {field: value}}
                await self.assert_rejected_before_dispatch(json.dumps(decision))
            with self.subTest(where="finish", field=field):
                decision = {"action": "finish", "reason": "done", field: value}
                await self.assert_rejected_before_dispatch(json.dumps(decision))

    def test_parse_decision_returns_exact_values(self):
        self.assertEqual(parse_decision(call("camera_set_exposure", exposure_us=1500)),
                         {"action": "call_tool", "tool": "camera_set_exposure",
                          "arguments": {"exposure_us": 1500}})
        self.assertIs(type(parse_decision(call("camera_set_exposure", exposure_us=1500))
                           ["arguments"]["exposure_us"]), int)
        self.assertEqual(parse_decision(finish("x")), {"action": "finish", "reason": "x"})
        with self.assertRaises(DecisionError):
            parse_decision('{"action": "call_tool", "tool": "camera_set_exposure", '
                           '"arguments": {"exposure_us": NaN}}')


# 23, 24, 25 --------------------------------------------------------------------

class FailureTests(ExecutorTestCase):
    async def test_policy_error_terminates_without_retry(self):
        provider = ScriptedProvider([call("camera_open"),
                                     call("camera_set_exposure", exposure_us=50.0),
                                     call("camera_set_exposure", exposure_us=50.0), finish()])
        original = Client.call_tool
        with mock.patch.object(Client, "call_tool", autospec=True, side_effect=original) as spy:
            host, result = await self.run_executor(provider)
        self.assertEqual(result["status"], eml.STATUS_TOOL_ERROR)
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(len(provider.requests), 2)
        last = self.tool_events(result)[-1]
        self.assertEqual(last["status"], "mcp_error")
        self.assertIsNone(last["result"])
        self.assertIn("Not authorized", last["error"])
        self.assertNotIn("set_exposure_us", host._camera.calls)
        self.assertTrue(result["host_shut_down"])

    async def test_backend_failure_terminates_without_retry(self):
        # camera_measure before camera_open: FakeCamera refuses, the runtime aborts the session.
        provider = ScriptedProvider([call("camera_measure"), call("camera_measure"), finish()])
        host, result = await self.run_executor(provider)
        self.assertEqual(result["status"], eml.STATUS_TOOL_ERROR)
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(host._camera.calls.count("capture_frame"), 1)
        self.assertIn("not retried", self.tool_events(result)[0]["error"])
        self.assertEqual(result["successful_measurement_count"], 0)

    async def test_measurement_budget_exhaustion_terminates_safely(self):
        provider = ScriptedProvider([call("camera_open")] + [call("camera_measure")] * 4)
        host = self.approved_host(max_measurements=2)
        _, result = await self.run_executor(provider, host=host)
        self.assertEqual(result["status"], eml.STATUS_TOOL_ERROR)
        self.assertNotEqual(result["status"], eml.STATUS_COMPLETED)
        self.assertIn("measurement budget exhausted", result["final_reason"])
        self.assertEqual(result["successful_measurement_count"], 2)
        self.assertEqual(result["tool_call_count"], 4)
        self.assertEqual(len(provider.requests), 4)
        self.assertEqual(host._camera.calls.count("capture_frame"), 2)
        self.assertTrue(result["host_shut_down"])
        self.assertFalse(host._camera.is_open)

    async def test_decision_budget_exhaustion_terminates_safely(self):
        provider = ScriptedProvider([call("camera_open")] + [call("camera_measure")] * 5)
        original = Client.call_tool
        with mock.patch.object(Client, "call_tool", autospec=True, side_effect=original) as spy:
            host, result = await self.run_executor(provider, max_decisions=3)
        self.assertEqual(result["status"], eml.STATUS_DECISION_BUDGET_EXHAUSTED)
        self.assertNotEqual(result["status"], eml.STATUS_COMPLETED)
        self.assertEqual(result["decision_count"], 3)
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(spy.call_count, 3)
        self.assertEqual(result["tool_call_count"], 3)
        self.assertEqual(provider.contexts[-1]["decision_budget"]["remaining"], 1)
        self.assertEqual(host._camera.calls.count("capture_frame"), 2)
        self.assertTrue(result["host_shut_down"])

    async def test_single_decision_budget(self):
        provider = ScriptedProvider([call("camera_open"), finish()])
        _, result = await self.run_executor(provider, max_decisions=1)
        self.assertEqual(result["status"], eml.STATUS_DECISION_BUDGET_EXHAUSTED)
        self.assertEqual((result["decision_count"], result["tool_call_count"]), (1, 1))

    async def test_provider_exception_stops(self):
        _, result = await self.run_executor(RaisingProvider())
        self.assertEqual(result["status"], eml.STATUS_PROVIDER_ERROR)
        self.assertEqual(result["decision_count"], 1)
        self.assertEqual(result["tool_call_count"], 0)
        self.assertIn("provider unavailable", result["final_reason"])
        self.assertTrue(result["host_shut_down"])

    async def test_uninterpretable_response_fails_closed(self):
        host = self.approved_host()
        runtime = host._runtime
        fake = {"tool": "camera_measure", "source": "synthetic", "synthetic": True}
        provider = ScriptedProvider([call("camera_measure"), finish()])
        with mock.patch.object(runtime, "execute", return_value=fake):
            _, result = await self.run_executor(provider, host=host)
        self.assertEqual(result["status"], eml.STATUS_INVALID_TOOL_RESPONSE)
        self.assertEqual(result["successful_measurement_count"], 0)
        self.assertEqual(self.tool_events(result)[0]["status"], "invalid_response")
        self.assertEqual(len(provider.requests), 1)

    async def test_mislabelled_provenance_fails_closed(self):
        host = self.approved_host()
        runtime = host._runtime
        bad = {"tool": "camera_measure", "source": "hardware", "synthetic": True, "frame_id": 1,
               "exposure_us": 1000.0, "metrics": {}}
        with mock.patch.object(runtime, "execute", return_value=bad):
            _, result = await self.run_executor(ScriptedProvider([call("camera_measure")]),
                                                host=host)
        self.assertEqual(result["status"], eml.STATUS_INVALID_TOOL_RESPONSE)
        self.assertEqual(result["provenance"]["sources_observed"], [])


# 26, 27 ------------------------------------------------------------------------

class ShutdownTests(ExecutorTestCase):
    async def test_host_shutdown_after_completion(self):
        provider = ScriptedProvider([call("camera_open"), call("camera_measure"), finish()])
        host, result = await self.run_executor(provider)
        self.assertEqual(result["status"], eml.STATUS_COMPLETED)
        self.assertIs(result["host_shut_down"], True)
        self.assertEqual(result["host_shutdown"]["result"]["runtime"]["backend_closed"], True)
        self.assertEqual(host._camera.calls[-1], "close")
        self.assertFalse(host._camera.is_open)
        with self.assertRaises(HostError):
            host.get_mcp_server()

    async def test_host_shutdown_after_errors(self):
        cases = {
            "invalid_decision": [call("camera_open"), "garbage"],
            "tool_error": [call("camera_open"), call("camera_set_exposure", exposure_us=1e6)],
            "budget": [call("camera_open"), call("camera_measure")],
        }
        for name, replies in cases.items():
            with self.subTest(case=name):
                host, result = await self.run_executor(ScriptedProvider(replies), max_decisions=2)
                self.assertNotEqual(result["status"], eml.STATUS_COMPLETED)
                self.assertIs(result["host_shut_down"], True)
                self.assertFalse(host._camera.is_open)
                with self.assertRaises(HostError):
                    host.get_mcp_server()
        host, result = await self.run_executor(RaisingProvider())
        self.assertIs(result["host_shut_down"], True)

    async def test_cleanup_failure_is_reported(self):
        host = self.approved_host()
        host._camera.faults["close"] = RuntimeError("close failed")
        provider = ScriptedProvider([call("camera_open"), call("camera_measure"), finish()])
        _, result = await self.run_executor(provider, host=host)
        self.assertEqual(result["status"], eml.STATUS_CLEANUP_FAILED)
        self.assertIs(result["host_shut_down"], False)
        self.assertIs(result["host_shutdown"]["attempted"], True)
        self.assertIn("close failed", result["host_shutdown"]["error"])
        self.assertIn("completed", result["final_reason"])

    async def test_provider_cannot_reach_shutdown(self):
        reply = json.dumps({"action": "call_tool", "tool": "shutdown", "arguments": {}})
        host, result = await self.assert_rejected_before_dispatch(reply)
        self.assertEqual(result["host_shutdown"]["attempted"], True)  # only the trusted path


# 6, 28 -------------------------------------------------------------------------

class IsolationTests(unittest.TestCase):
    ALLOWED_IMPORTS = {"__future__", "json", "math", "typing", "mcp", "approved_session_host",
                       "session_policy"}

    def setUp(self):
        self.source = MODULE_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_imports_no_backend_sdk_llm_or_langgraph(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertLessEqual(imported, self.ALLOWED_IMPORTS)

    def test_no_direct_backend_runtime_or_approval_calls(self):
        used = {node.attr for node in ast.walk(self.tree) if isinstance(node, ast.Attribute)}
        used |= {node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)}
        for name in ("capture_frame", "set_exposure_us", "get_info", "get_exposure_limits_us",
                     "open", "_camera", "_runtime", "_backend", "_gate", "execute",
                     "FakeCamera", "CameraBackend", "AuthorizedCameraRuntime", "SessionGate",
                     "SessionEnvelope", "submit_decision", "session_approval", "begin_action",
                     "start", "resume", "start_approval", "resume_approval", "pending",
                     "create_authorized_server", "exec", "eval", "compile", "subprocess"):
            with self.subTest(name=name):
                self.assertNotIn(name, used)
        self.assertIn("call_tool", used)
        self.assertIn("get_mcp_server", used)
        self.assertIn("shutdown", used)

    def test_no_vendor_network_llm_or_background_references(self):
        for token in ("MVSDK", "sdk_probe", "ctypes", "iRAYPLE", ".dll", "socket", "http",
                      "openai", "OpenAI", "anthropic", "langgraph", "create_task", "gather",
                      "Thread", "__main__", "open("):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_sdk_probe_and_anthropic_not_loaded(self):
        # Importing the OpenAI SDK elsewhere in the process is legitimate;
        # OfflineProviderTests checks that no client or request happens.
        for module in ("sdk_probe", "anthropic"):
            with self.subTest(module=module):
                self.assertNotIn(module, sys.modules)


class OfflineProviderTests(ExecutorTestCase):
    async def test_adaptive_run_constructs_no_openai_client_and_sends_no_request(self):
        provider = ThresholdProvider()
        with openai_sdk_guard(self):
            host, result = await self.run_executor(provider)
        self.assertEqual(result["status"], eml.STATUS_COMPLETED, result["final_reason"])
        # Every decision came from the injected deterministic test double.
        self.assertEqual(len(provider.requests), result["decision_count"])
        self.assertGreaterEqual(host._camera.calls.count("capture_frame"), 1)  # FakeCamera only
        self.assertTrue(result["host_shut_down"])


if __name__ == "__main__":
    unittest.main()
