"""Offline tests for hw3/llm_decision_provider.py.

Every "model" here is a deterministic FAKE ModelCaller (an object with
call(instructions, input_text) -> str); no LLM API, credential or network is
used. Measurement dictionaries are synthetic. The integration tests drive the
real AdaptiveMCPExecutor through the official MCP SDK Client against an
in-process authorized server backed by FakeCamera; the simulated approval is
given by this trusted test harness, never by the fake model.
"""

import ast
import contextlib
import copy
import json
import socket
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import executor_mcp_loop as eml  # noqa: E402
from approved_session_host import OfflineApprovedSessionHost  # noqa: E402
from executor_mcp_loop import DECISION_INSTRUCTIONS, AdaptiveMCPExecutor  # noqa: E402
from llm_decision_provider import (  # noqa: E402
    CONTEXT_FORMAT,
    BoundedDecisionProvider,
    CallLimitError,
    ContextTooLargeError,
    ProviderInputError,
    ProviderOutputError,
    compact_context,
    dumps,
)
from session_policy import SessionEnvelope, session_approval  # noqa: E402

MODULE_PATH = HW3_DIR / "llm_decision_provider.py"
GOAL = "Bring the synthetic beam's mean brightness near half of full scale."
INSTRUCTIONS = "Return one JSON decision."
FINISH = '{"action": "finish", "reason": "done"}'
ALL_TOOLS = ("camera_info", "camera_open", "camera_set_exposure", "camera_measure", "camera_close")


def _tools():
    tools = []
    for name in ALL_TOOLS:
        schema = {"properties": {}, "type": "object"}
        if name == "camera_set_exposure":
            schema = {"properties": {"exposure_us": {"description": "Exposure in microseconds",
                                                     "type": "number"}},
                      "required": ["exposure_us"], "type": "object"}
        tools.append({"name": name, "description": f"{name} (synthetic test)", "input_schema": schema})
    return tools


def _metrics(mean_dn):
    """Synthetic metrics dictionary with the Measurement Engine's field names."""
    return {
        "width": 8, "height": 8, "bit_depth": 8, "full_scale_dn": 255,
        "min_dn": 30, "max_dn": 156, "mean_dn": mean_dn,
        "mean_fraction_of_full_scale": mean_dn / 255, "spatial_std_dn": 37.0,
        "saturation_fraction": 0.0, "zero_fraction": 0.0,
        "percentiles_dn": {"p1": 31.0, "p50": 93.0, "p99": 155.0},
        "histogram": {"bins": 16, "range_dn": [0, 256],
                      "bin_edges_dn": [16.0 * i for i in range(17)], "counts": [4] * 16},
    }


def _measurement(frame_id, exposure_us, mean_dn):
    return {"tool": "camera_measure", "action_id": f"mcp-test-{frame_id}",
            "session_id": "session-synthetic", "source": "synthetic", "synthetic": True,
            "frame_id": frame_id, "exposure_us": exposure_us, "bit_depth": 8, "width": 8,
            "height": 8, "metrics": _metrics(mean_dn)}


def _history(measure_cycles):
    """decision/tool_call pairs: open, then (measure, set_exposure) cycles. Synthetic."""
    events = []

    def add(tool, arguments, result, **extra):
        n = len(events) // 2 + 1
        decision = {"action": "call_tool", "tool": tool, "arguments": arguments}
        events.append({"sequence": len(events) + 1, "type": "decision", "decision_number": n,
                       "raw_output": json.dumps(decision), "valid": True, "decision": decision})
        events.append({"sequence": len(events) + 1, "type": "tool_call", "decision_number": n,
                       "tool": tool, "arguments": arguments, "status": "success",
                       "result": result, "error": None, **extra})

    add("camera_open", {}, {"tool": "camera_open", "ok": True, "is_open": True})
    latest = None
    for i in range(measure_cycles):
        exposure = 1000.0 + 10.0 * i
        latest = _measurement(i + 1, exposure, 90.0 + i)
        add("camera_measure", {}, latest, source="synthetic", frame_id=i + 1)
        add("camera_set_exposure", {"exposure_us": exposure + 10.0},
            {"tool": "camera_set_exposure", "ok": True, "exposure_us": exposure + 10.0})
    return events, latest


def _context(measure_cycles=2):
    history, latest = _history(measure_cycles)
    used = len(history) // 2
    return {"research_goal": GOAL, "tools": _tools(), "history": history,
            "latest_measurement": latest,
            "decision_budget": {"max": used + 5, "used": used, "remaining": 5}}


class FakeModelCaller:
    """Deterministic fake ModelCaller: replies in order and records every call."""

    def __init__(self, replies=(FINISH,)):
        self.replies = list(replies)
        self.calls = []

    def call(self, instructions, input_text):
        self.calls.append((instructions, input_text))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def sent(self, index=-1):
        return json.loads(self.calls[index][1])


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


def _provider(caller=None, **overrides):
    limits = {"max_calls": 5, "max_input_chars": 20000, "max_output_chars": 500,
              "max_recent_events": 6}
    limits.update(overrides)
    return BoundedDecisionProvider(caller or FakeModelCaller(), **limits)


# 1, 2, 20 ---------------------------------------------------------------------

class InvocationTests(unittest.TestCase):
    def test_successful_invocation_with_correct_signature(self):
        caller = FakeModelCaller()
        provider = _provider(caller)
        output = provider.call(INSTRUCTIONS, json.dumps(_context()))
        self.assertEqual(output, FINISH)
        self.assertEqual(len(caller.calls), 1)
        instructions, input_text = caller.calls[0]
        self.assertEqual(instructions, INSTRUCTIONS)
        self.assertIsInstance(input_text, str)
        self.assertEqual(caller.sent()["context_format"], CONTEXT_FORMAT)

    def test_positional_signature_matches_model_caller(self):
        class StrictCaller:
            def __init__(self):
                self.args = None

            def call(self, instructions, input_text, /):
                self.args = (instructions, input_text)
                return FINISH

        caller = StrictCaller()
        _provider(caller).call(INSTRUCTIONS, json.dumps(_context()))
        self.assertEqual(caller.args[0], INSTRUCTIONS)

    def test_valid_output_returned_unchanged(self):
        for reply in (FINISH, "  " + FINISH + "\n", "not even json", "",
                      '{"action":"call_tool","tool":"camera_measure","arguments":{}}'):
            with self.subTest(reply=reply):
                output = _provider(FakeModelCaller([reply])).call(INSTRUCTIONS, json.dumps(_context()))
                self.assertIs(type(output), str)
                self.assertEqual(output, reply)

    def test_compact_context_is_deterministic(self):
        context = json.dumps(_context(5))
        first, second = FakeModelCaller(), FakeModelCaller()
        _provider(first).call(INSTRUCTIONS, context)
        _provider(second).call(INSTRUCTIONS, context)
        self.assertEqual(first.calls, second.calls)
        text = first.calls[0][1]
        self.assertEqual(text, dumps(json.loads(text)))  # sorted keys, no whitespace


# 3-10 -------------------------------------------------------------------------

class CompactionTests(unittest.TestCase):
    def test_goal_tools_budget_and_latest_measurement_preserved(self):
        context = _context(4)
        caller = FakeModelCaller()
        _provider(caller).call(INSTRUCTIONS, json.dumps(context))
        sent = caller.sent()
        self.assertEqual(sent["research_goal"], GOAL)
        self.assertEqual(sent["tools"], context["tools"])
        self.assertEqual(sent["tools"][2]["input_schema"]["properties"]["exposure_us"]["type"],
                         "number")
        self.assertEqual(sent["decision_budget"], context["decision_budget"])
        self.assertEqual(sent["latest_measurement"], context["latest_measurement"])

    def test_last_frame_exposure_and_metrics_preserved(self):
        context = _context(4)
        caller = FakeModelCaller()
        _provider(caller, max_recent_events=0).call(INSTRUCTIONS, json.dumps(context))
        latest = caller.sent()["latest_measurement"]
        self.assertEqual(latest["frame_id"], 4)
        self.assertEqual(latest["exposure_us"], 1030.0)
        self.assertEqual(latest["metrics"], _metrics(93.0))
        self.assertEqual(latest["metrics"]["saturation_fraction"], 0.0)
        self.assertEqual(latest["source"], "synthetic")

    def test_absent_measurement_is_not_invented(self):
        context = _context(0)
        self.assertIsNone(context["latest_measurement"])
        caller = FakeModelCaller()
        _provider(caller).call(INSTRUCTIONS, json.dumps(context))
        sent = caller.sent()
        self.assertIsNone(sent["latest_measurement"])
        self.assertNotIn("frame_id", json.dumps(sent["recent_history"]))

    def test_recent_events_keep_outcomes_and_measurement_summaries(self):
        context = _context(3)
        caller = FakeModelCaller()
        _provider(caller, max_recent_events=4).call(INSTRUCTIONS, json.dumps(context))
        recent = caller.sent()["recent_history"]
        originals = context["history"][-4:]
        self.assertEqual([e["sequence"] for e in recent], [e["sequence"] for e in originals])
        measure = next(e for e in recent if e.get("tool") == "camera_measure")
        self.assertEqual(measure["status"], "success")
        self.assertEqual(measure["frame_id"], 3)
        self.assertEqual(measure["result"]["exposure_us"], 1020.0)
        self.assertEqual(measure["result"]["metrics"]["mean_dn"], 92.0)
        self.assertIn("saturation_fraction", measure["result"]["metrics"])
        self.assertNotIn("histogram", measure["result"]["metrics"])
        decision = next(e for e in recent if e["type"] == "decision")
        self.assertIn("decision", decision)
        self.assertNotIn("raw_output", decision)

    def test_older_history_truncated_and_summarised(self):
        context = _context(10)  # 42 events
        caller = FakeModelCaller()
        _provider(caller, max_recent_events=6).call(INSTRUCTIONS, json.dumps(context))
        sent = caller.sent()
        recent = sent["recent_history"]
        self.assertEqual(len(recent), 6)
        self.assertEqual([e["sequence"] for e in recent], list(range(37, 43)))
        summary = sent["history_summary"]
        self.assertEqual(summary["omitted_event_count"], 36)
        self.assertEqual(summary["omitted_sequence_range"], [1, 36])
        self.assertEqual(summary["omitted_successful_measurements"], 9)
        self.assertEqual(summary["omitted_frame_id_range"], [1, 9])
        self.assertEqual(summary["omitted_tool_calls"],
                         {"camera_open": 1, "camera_measure": 9, "camera_set_exposure": 8})

    def test_zero_recent_events_keeps_essentials(self):
        caller = FakeModelCaller()
        _provider(caller, max_recent_events=0).call(INSTRUCTIONS, json.dumps(_context(3)))
        sent = caller.sent()
        self.assertEqual(sent["recent_history"], [])
        self.assertEqual(sent["history_summary"]["omitted_event_count"], 14)
        self.assertEqual(sent["latest_measurement"]["frame_id"], 3)

    def test_original_context_not_mutated(self):
        context = _context(6)
        before = copy.deepcopy(context)
        compact_context(context, 3, 5000)
        self.assertEqual(context, before)
        text = json.dumps(context)
        _provider(max_input_chars=6000, max_recent_events=10).call(INSTRUCTIONS, text)
        self.assertEqual(json.loads(text), before)

    def test_large_history_compacted_within_limit(self):
        context = _context(200)  # 802 events
        limit = 7000
        caller = FakeModelCaller()
        provider = _provider(caller, max_input_chars=limit, max_recent_events=50)
        provider.call(INSTRUCTIONS, json.dumps(context))
        instructions, text = caller.calls[0]
        self.assertLessEqual(len(instructions) + len(text), limit)
        self.assertEqual(provider.last_input_chars, len(instructions) + len(text))
        sent = json.loads(text)
        recent = sent["recent_history"]
        self.assertLess(len(recent), 50)  # progressively reduced below max_recent_events
        self.assertGreater(len(recent), 0)
        # The newest events are the ones kept, as one contiguous suffix.
        self.assertEqual([e["sequence"] for e in recent],
                         [e["sequence"] for e in context["history"][-len(recent):]])
        self.assertEqual(sent["history_summary"]["omitted_event_count"] + len(recent), 802)
        self.assertEqual(sent["latest_measurement"], context["latest_measurement"])
        self.assertEqual(sent["latest_measurement"]["frame_id"], 200)

    def test_essential_context_too_large_fails_before_model_call(self):
        caller = FakeModelCaller()
        provider = _provider(caller, max_input_chars=1500, max_recent_events=10)
        with self.assertRaises(ContextTooLargeError):
            provider.call(INSTRUCTIONS, json.dumps(_context(3)))
        self.assertEqual(caller.calls, [])
        self.assertEqual(provider.calls_used, 0)
        self.assertIsNone(provider.last_input_chars)

    def test_instructions_alone_too_large_fails_before_model_call(self):
        caller = FakeModelCaller()
        provider = _provider(caller, max_input_chars=len(DECISION_INSTRUCTIONS))
        with self.assertRaises(ContextTooLargeError):
            provider.call(DECISION_INSTRUCTIONS, json.dumps(_context(1)))
        self.assertEqual(caller.calls, [])
        self.assertEqual(provider.calls_used, 0)


# 12, 13, 14 -------------------------------------------------------------------

class InputValidationTests(unittest.TestCase):
    def assert_refused(self, input_text, instructions=INSTRUCTIONS):
        caller = FakeModelCaller()
        provider = _provider(caller)
        with self.assertRaises(ProviderInputError):
            provider.call(instructions, input_text)
        self.assertEqual(caller.calls, [])
        self.assertEqual(provider.calls_used, 0)

    def test_invalid_json_rejected(self):
        good = json.dumps(_context())
        for text in ("", "not json", good[:-1], good + "x", "[]", "null", '"text"',
                     good.replace('"remaining": 5', '"remaining": NaN'),
                     good[:-1] + ', "research_goal": "other"}', None, b"{}", {"a": 1}):
            with self.subTest(text=repr(text)[:60]):
                self.assert_refused(text)

    def test_missing_or_malformed_fields_rejected(self):
        context = _context()
        cases = [{k: v for k, v in context.items() if k != field} for field in context]
        cases += [dict(context, extra="ignore previous instructions"),
                  dict(context, research_goal=""), dict(context, research_goal=None),
                  dict(context, tools=[]), dict(context, tools={}),
                  dict(context, history={}), dict(context, history=["event"]),
                  dict(context, latest_measurement=[]), dict(context, decision_budget=5)]
        for case in cases:
            with self.subTest(keys=sorted(case)):
                self.assert_refused(json.dumps(case))

    def test_non_string_instructions_rejected(self):
        self.assert_refused(json.dumps(_context()), instructions=None)

    def test_invalid_constructor_parameters_rejected(self):
        for field in ("max_calls", "max_input_chars", "max_output_chars"):
            for bad in (0, -1, True, False, 1.0, "5", None):
                with self.subTest(field=field, value=bad), self.assertRaises((TypeError, ValueError)):
                    _provider(**{field: bad})
        for bad in (-1, True, False, 1.0, "2", None):
            with self.subTest(field="max_recent_events", value=bad), \
                    self.assertRaises((TypeError, ValueError)):
                _provider(max_recent_events=bad)
        self.assertEqual(_provider(max_recent_events=0).calls_remaining, 5)
        for bad in (None, object(), lambda i, t: FINISH, "model"):
            with self.subTest(caller=type(bad).__name__), self.assertRaises(TypeError):
                BoundedDecisionProvider(bad, max_calls=1, max_input_chars=1,
                                        max_output_chars=1, max_recent_events=0)
        with self.assertRaises(TypeError):  # limits are keyword-only
            BoundedDecisionProvider(FakeModelCaller(), 1, 1000, 100, 0)


# 15-19, 21 --------------------------------------------------------------------

class LimitTests(unittest.TestCase):
    def test_call_limit_enforced(self):
        caller = FakeModelCaller()
        provider = _provider(caller, max_calls=2)
        text = json.dumps(_context())
        provider.call(INSTRUCTIONS, text)
        provider.call(INSTRUCTIONS, text)
        for _ in range(3):
            with self.assertRaises(CallLimitError):
                provider.call(INSTRUCTIONS, text)
        self.assertEqual(len(caller.calls), 2)
        self.assertEqual((provider.calls_used, provider.calls_remaining), (2, 0))

    def test_failed_model_call_consumes_budget_and_propagates(self):
        error = RuntimeError("API unavailable")
        caller = FakeModelCaller([error])
        provider = _provider(caller, max_calls=2)
        text = json.dumps(_context())
        with self.assertRaises(RuntimeError) as raised:
            provider.call(INSTRUCTIONS, text)
        self.assertIs(raised.exception, error)
        self.assertEqual((provider.calls_used, provider.calls_remaining), (1, 1))
        self.assertIsNotNone(provider.last_input_chars)
        self.assertIsNone(provider.last_output_chars)

    def test_no_automatic_retry(self):
        caller = FakeModelCaller([RuntimeError("timeout"), FINISH])
        provider = _provider(caller, max_calls=3)
        with self.assertRaises(RuntimeError):
            provider.call(INSTRUCTIONS, json.dumps(_context()))
        self.assertEqual(len(caller.calls), 1)
        self.assertEqual(caller.replies, [FINISH])  # the second reply was never requested
        caller = FakeModelCaller([123])
        with self.assertRaises(ProviderOutputError):
            _provider(caller).call(INSTRUCTIONS, json.dumps(_context()))
        self.assertEqual(len(caller.calls), 1)

    def test_oversized_output_rejected(self):
        provider = _provider(FakeModelCaller(["x" * 101]), max_output_chars=100)
        with self.assertRaises(ProviderOutputError):
            provider.call(INSTRUCTIONS, json.dumps(_context()))
        self.assertEqual(provider.calls_used, 1)
        self.assertEqual(provider.last_output_chars, 101)
        exact = _provider(FakeModelCaller(["y" * 100]), max_output_chars=100)
        self.assertEqual(exact.call(INSTRUCTIONS, json.dumps(_context())), "y" * 100)

    def test_non_string_output_rejected(self):
        for reply in (None, 42, {"action": "finish", "reason": "x"}, b"{}", ["finish"]):
            with self.subTest(reply=repr(reply)):
                provider = _provider(FakeModelCaller([reply]))
                with self.assertRaises(ProviderOutputError):
                    provider.call(INSTRUCTIONS, json.dumps(_context()))
                self.assertEqual(provider.calls_used, 1)
                self.assertIsNone(provider.last_output_chars)

    def test_usage_counters(self):
        caller = FakeModelCaller([FINISH, "ab"])
        provider = _provider(caller, max_calls=3)
        self.assertEqual((provider.calls_used, provider.calls_remaining), (0, 3))
        self.assertIsNone(provider.last_input_chars)
        self.assertIsNone(provider.last_output_chars)
        provider.call(INSTRUCTIONS, json.dumps(_context()))
        self.assertEqual((provider.calls_used, provider.calls_remaining), (1, 2))
        self.assertEqual(provider.last_input_chars, len(INSTRUCTIONS) + len(caller.calls[0][1]))
        self.assertEqual(provider.last_output_chars, len(FINISH))
        provider.call(INSTRUCTIONS, json.dumps(_context(0)))
        self.assertEqual((provider.calls_used, provider.calls_remaining), (2, 1))
        self.assertEqual(provider.last_input_chars, len(INSTRUCTIONS) + len(caller.calls[1][1]))
        self.assertEqual(provider.last_output_chars, 2)
        for name in ("calls_used", "calls_remaining", "last_input_chars", "last_output_chars"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(provider, name, 0)

    def test_refused_inputs_do_not_consume_calls(self):
        caller = FakeModelCaller()
        provider = _provider(caller, max_calls=1, max_input_chars=3000)
        for text in ("bad", json.dumps(_context(30))[:10]):
            with self.assertRaises(ProviderInputError):
                provider.call(INSTRUCTIONS, text)
        self.assertEqual(provider.calls_remaining, 1)
        provider.call(INSTRUCTIONS, json.dumps(_context(1)))
        self.assertEqual(provider.calls_remaining, 0)


# Executor integration (in-process MCP, FakeCamera, fake model) ------------------

class ExecutorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def approved_host(self):
        envelope = SessionEnvelope(
            session_id=f"session-llm-{uuid.uuid4().hex[:12]}", device_label="fake-camera",
            allowed_tools=ALL_TOOLS, min_exposure_us=100.0, max_exposure_us=10000.0,
            max_measurements=3, max_tool_calls=12, max_duration_seconds=60.0)
        host = OfflineApprovedSessionHost(envelope, clock=lambda: 0.0)
        host.start(f"thread-{uuid.uuid4().hex[:12]}")
        self.assertTrue(host.resume(
            session_approval(envelope.session_id, envelope.sha256(), True))["activated"])
        return host

    async def run_executor(self, caller, max_decisions=6, **limits):
        provider = _provider(caller, **limits)
        executor = AdaptiveMCPExecutor(self.approved_host(), provider, GOAL, max_decisions)
        return provider, await executor.run()

    async def test_executor_loop_through_bounded_provider(self):
        replies = ['{"action":"call_tool","tool":"camera_open","arguments":{}}',
                   '{"action":"call_tool","tool":"camera_measure","arguments":{}}', FINISH]
        caller = FakeModelCaller(replies)
        provider, result = await self.run_executor(caller, max_recent_events=2)
        self.assertEqual(result["status"], eml.STATUS_COMPLETED)
        self.assertEqual(provider.calls_used, 3)
        self.assertEqual([c[0] for c in caller.calls], [DECISION_INSTRUCTIONS] * 3)
        last = caller.sent()
        self.assertEqual(last["latest_measurement"], result["latest_measurement"])
        self.assertEqual(last["latest_measurement"]["frame_id"], 1)
        self.assertEqual(len(last["recent_history"]), 2)
        self.assertEqual(last["history_summary"]["omitted_event_count"], 2)
        # The Executor's own log keeps the raw outputs and full results.
        self.assertEqual([e["raw_output"] for e in result["events"] if e["type"] == "decision"],
                         replies)
        self.assertIn("histogram",
                      [e for e in result["events"] if e.get("tool") == "camera_measure"][0]
                      ["result"]["metrics"])

    async def test_call_limit_stops_executor(self):
        caller = FakeModelCaller(['{"action":"call_tool","tool":"camera_open","arguments":{}}'])
        provider, result = await self.run_executor(caller, max_calls=2)
        self.assertEqual(result["status"], eml.STATUS_PROVIDER_ERROR)
        self.assertIn("CallLimitError", result["final_reason"])
        self.assertEqual(len(caller.calls), 2)
        self.assertEqual(result["decision_count"], 3)
        self.assertTrue(result["host_shut_down"])

    async def test_model_failure_stops_executor_without_retry(self):
        caller = FakeModelCaller([RuntimeError("rate limited")])
        provider, result = await self.run_executor(caller)
        self.assertEqual(result["status"], eml.STATUS_PROVIDER_ERROR)
        self.assertIn("rate limited", result["final_reason"])
        self.assertEqual(len(caller.calls), 1)
        self.assertEqual(provider.calls_used, 1)
        self.assertEqual(result["tool_call_count"], 0)
        self.assertTrue(result["host_shut_down"])

    async def test_bounded_provider_run_constructs_no_openai_client_and_sends_no_request(self):
        replies = ['{"action":"call_tool","tool":"camera_open","arguments":{}}',
                   '{"action":"call_tool","tool":"camera_measure","arguments":{}}', FINISH]
        caller = FakeModelCaller(replies)
        with openai_sdk_guard(self):
            provider, result = await self.run_executor(caller)
        self.assertEqual(result["status"], eml.STATUS_COMPLETED)
        # Every model answer came from the injected deterministic fake caller.
        self.assertEqual(len(caller.calls), 3)
        self.assertEqual(provider.calls_used, 3)
        self.assertEqual(result["latest_measurement"]["source"], "synthetic")


# 22 ---------------------------------------------------------------------------

class IsolationTests(unittest.TestCase):
    ALLOWED_IMPORTS = {"__future__", "copy", "json", "typing"}

    def setUp(self):
        self.source = MODULE_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_imports_nothing_external(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertLessEqual(imported, self.ALLOWED_IMPORTS)

    def test_no_llm_credentials_network_camera_or_files(self):
        for token in ("import openai", "OpenAI(", "anthropic", "api_key", "API_KEY", "environ",
                      "getenv", "ModelCaller(", "http", "socket", "from mcp", "FakeCamera",
                      "sdk_probe", "MVSDK", "ctypes", "open(", "write", "print(", "logging",
                      "sleep", "random", "retry("):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_no_tool_choice_or_approval_logic(self):
        names = {n.id for n in ast.walk(self.tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(self.tree) if isinstance(n, ast.Attribute)}
        for name in ("parse_decision", "call_tool", "submit_decision", "session_approval",
                     "get_mcp_server", "shutdown", "execute"):
            with self.subTest(name=name):
                self.assertNotIn(name, names)

    def test_no_anthropic_or_vendor_modules_loaded(self):
        # Importing the OpenAI SDK elsewhere in the process is legitimate;
        # ExecutorIntegrationTests checks that no client or request happens.
        for module in ("anthropic", "sdk_probe"):
            with self.subTest(module=module):
                self.assertNotIn(module, sys.modules)


if __name__ == "__main__":
    unittest.main()
