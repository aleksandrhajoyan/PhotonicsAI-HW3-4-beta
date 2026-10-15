"""Offline tests for hw3/openai_model_caller.py.

Every client here is a FAKE OpenAI-compatible object exposing
responses.create(**kwargs); no OpenAI API, credential or network is used. The
live client factory is exercised only against a fake ``openai`` module placed
in sys.modules for the duration of one test. No camera, MCP server or
LangGraph component is involved.
"""

import ast
import contextlib
import inspect
import json
import logging
import socket
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = HW3_DIR.parent
sys.path.insert(0, str(HW3_DIR))

import openai_model_caller as omc  # noqa: E402
from llm_decision_provider import BoundedDecisionProvider, CallLimitError  # noqa: E402
from openai_model_caller import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    LumenOpenAIModelCaller,
    ModelInputError,
    ModelOutputError,
    ModelUsage,
    create_live_openai_client,
    extract_usage,
)

MODULE_PATH = HW3_DIR / "openai_model_caller.py"
TASK6_PATH = PROJECT_DIR / "task6" / "agentic_system.py"
FAKE_KEY = "sk-test-FAKE-NOT-A-REAL-KEY-0123456789"
INSTRUCTIONS = "Return one JSON decision.\nUse only the listed tools."
INPUT_TEXT = '{"research_goal": "synthetic", "note": "  keep  spacing  "}'
FINISH = '{"action": "finish", "reason": "done"}'
MISSING = object()


def _usage(input_tokens=120, output_tokens=30, total_tokens=150):
    return types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                                 total_tokens=total_tokens)


def _response(text=FINISH, model="gpt-6-luna-2026-09-01", usage=MISSING,
              status="completed"):
    response = types.SimpleNamespace(model=model,
                                     usage=_usage() if usage is MISSING else usage)
    if status is not MISSING:
        response.status = status
    if text is not MISSING:
        response.output_text = text
    return response


class FakeResponses:
    """Records every create() call and replies in order (last reply repeats)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeClient:
    """OpenAI-compatible fake that holds a fake key, like the real client would."""

    def __init__(self, *replies):
        self.api_key = FAKE_KEY
        self.responses = FakeResponses(replies or (_response(),))

    @property
    def calls(self):
        return self.responses.calls


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


def _caller(*replies, **settings):
    client = FakeClient(*replies)
    return client, LumenOpenAIModelCaller(client, **settings)


def _context():
    """Minimal synthetic Executor context accepted by BoundedDecisionProvider."""
    return json.dumps({
        "research_goal": "Bring the synthetic beam near half of full scale.",
        "tools": [{"name": "camera_measure", "description": "synthetic",
                   "input_schema": {"properties": {}, "type": "object"}}],
        "history": [], "latest_measurement": None,
        "decision_budget": {"max": 5, "used": 0, "remaining": 5}})


# 1, 2 ---------------------------------------------------------------------------

class InterfaceTests(unittest.TestCase):
    def test_call_signature_matches_task6_model_caller(self):
        tree = ast.parse(TASK6_PATH.read_text(encoding="utf-8"))
        task6 = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == "ModelCaller")
        call = next(n for n in task6.body if isinstance(n, ast.FunctionDef) and n.name == "call")
        expected = [a.arg for a in call.args.args]
        signature = inspect.signature(LumenOpenAIModelCaller.call)
        self.assertEqual(list(signature.parameters), expected)
        self.assertEqual(expected, ["self", "instructions", "input_text"])
        self.assertEqual(signature.return_annotation, "str")

    def test_defaults(self):
        _, caller = _caller()
        self.assertEqual(DEFAULT_MODEL, "gpt-6-luna")
        self.assertEqual(DEFAULT_REASONING_EFFORT, "low")
        self.assertEqual(DEFAULT_MAX_OUTPUT_TOKENS, 512)
        self.assertEqual(caller.model, "gpt-6-luna")
        self.assertEqual(caller.reasoning_effort, "low")
        self.assertEqual(caller.max_output_tokens, 512)
        self.assertEqual(caller.calls_made, 0)
        self.assertIsNone(caller.last_usage)

    def test_explicit_settings(self):
        _, caller = _caller(model="gpt-test-model", reasoning_effort="medium",
                            max_output_tokens=256)
        self.assertEqual((caller.model, caller.reasoning_effort, caller.max_output_tokens),
                         ("gpt-test-model", "medium", 256))

    def test_construction_makes_no_request(self):
        client, _ = _caller()
        self.assertEqual(client.calls, [])


# 3-8, 13 -----------------------------------------------------------------------

class RequestTests(unittest.TestCase):
    def test_responses_api_arguments(self):
        client, caller = _caller()
        caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertEqual(client.calls, [{
            "model": "gpt-6-luna", "instructions": INSTRUCTIONS, "input": INPUT_TEXT,
            "reasoning": {"effort": "low"}, "max_output_tokens": 512}])

    def test_instructions_and_input_preserved_exactly(self):
        client, caller = _caller()
        instructions = "  Leading and trailing whitespace kept.\n\tUnicode: µs, ±, ½\n"
        input_text = '\n{"goal": "measure",   "unicode": "λ=633 nm"}  \n'
        caller.call(instructions, input_text)
        self.assertIs(client.calls[0]["instructions"], instructions)
        self.assertIs(client.calls[0]["input"], input_text)

    def test_reasoning_effort_and_max_output_tokens_configured(self):
        for effort, tokens in (("low", 512), ("minimal", 64), ("high", 2048)):
            with self.subTest(effort=effort, tokens=tokens):
                client, caller = _caller(reasoning_effort=effort, max_output_tokens=tokens)
                caller.call(INSTRUCTIONS, INPUT_TEXT)
                self.assertEqual(client.calls[0]["reasoning"], {"effort": effort})
                self.assertEqual(client.calls[0]["max_output_tokens"], tokens)

    def test_explicit_model_used(self):
        client, caller = _caller(model="gpt-other")
        caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertEqual(client.calls[0]["model"], "gpt-other")

    def test_text_returned_unchanged(self):
        for text in (FINISH, "  " + FINISH + "\n", "not json at all", "```json\n{}\n```"):
            with self.subTest(text=text):
                _, caller = _caller(_response(text))
                self.assertIs(caller.call(INSTRUCTIONS, INPUT_TEXT), text)

    def test_exactly_one_request_per_call(self):
        client, caller = _caller(_response("a"), _response("b"), _response("c"))
        self.assertEqual([caller.call(INSTRUCTIONS, INPUT_TEXT) for _ in range(3)],
                         ["a", "b", "c"])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(caller.calls_made, 3)

    def test_invalid_call_inputs_rejected_before_request(self):
        client, caller = _caller()
        for instructions, input_text in ((None, INPUT_TEXT), (INSTRUCTIONS, None),
                                         (b"x", INPUT_TEXT), (INSTRUCTIONS, {"a": 1}),
                                         ("", INPUT_TEXT), (INSTRUCTIONS, "  \n")):
            with self.subTest(instructions=instructions, input_text=input_text):
                with self.assertRaises(ModelInputError):
                    caller.call(instructions, input_text)
        self.assertEqual(client.calls, [])
        self.assertEqual(caller.calls_made, 0)


# 9-12 ---------------------------------------------------------------------------

class OutputValidationTests(unittest.TestCase):
    def assert_rejected(self, response):
        client, caller = _caller(response)
        with self.assertRaises(ModelOutputError):
            caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertEqual(len(client.calls), 1)
        return caller

    def test_empty_output_rejected(self):
        for text in ("", " ", "\n\t "):
            with self.subTest(text=text):
                self.assert_rejected(_response(text))

    def test_missing_output_rejected(self):
        self.assert_rejected(_response(MISSING))
        self.assert_rejected(_response(None))
        self.assert_rejected(None)

    def test_non_text_output_rejected(self):
        for text in (123, b"bytes", ["a"], {"text": FINISH}, True):
            with self.subTest(text=text):
                self.assert_rejected(_response(text))

    def test_usage_kept_when_output_rejected(self):
        caller = self.assert_rejected(_response(""))
        self.assertEqual(caller.last_usage.total_tokens, 150)

    def test_api_exception_propagates_without_retry(self):
        error = TimeoutError("synthetic timeout")
        client, caller = _caller(error, _response())
        with self.assertRaises(TimeoutError) as raised:
            caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertIs(raised.exception, error)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(caller.calls_made, 1)
        self.assertIsNone(caller.last_usage)
        # The next call is a new, single request, not a hidden retry.
        self.assertEqual(caller.call(INSTRUCTIONS, INPUT_TEXT), FINISH)
        self.assertEqual(len(client.calls), 2)


# HW3.4-C1: response status validation -------------------------------------------

class ResponseStatusTests(unittest.TestCase):
    def assert_rejected(self, response):
        client, caller = _caller(response)
        with self.assertRaises(ModelOutputError) as raised:
            caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(caller.calls_made, 1)
        self.assertNotIn(FINISH, str(raised.exception))
        self.assertNotIn(FAKE_KEY, str(raised.exception))
        return caller

    def test_completed_response_with_text_succeeds(self):
        client, caller = _caller(_response(FINISH, status="completed"))
        self.assertIs(caller.call(INSTRUCTIONS, INPUT_TEXT), FINISH)
        self.assertEqual(len(client.calls), 1)

    def test_non_completed_status_with_text_rejected(self):
        for status in ("incomplete", "failed", "cancelled", "queued", "in_progress"):
            with self.subTest(status=status):
                self.assert_rejected(_response(FINISH, status=status))

    def test_incomplete_partial_text_never_returned(self):
        partial = '{"action": "call_tool", "tool": "camera_meas'
        caller = self.assert_rejected(_response(partial, status="incomplete"))
        self.assertNotIn(partial, str(caller.last_usage))

    def test_missing_or_unknown_status_rejected(self):
        for status in (MISSING, None, "", "COMPLETED", " completed", "completed\n",
                       "done", "success", 1, True, ["completed"]):
            with self.subTest(status=status):
                self.assert_rejected(_response(FINISH, status=status))

    def test_unknown_status_value_not_echoed(self):
        status = "unexpected-" + FAKE_KEY
        client, caller = _caller(_response(FINISH, status=status))
        with self.assertRaises(ModelOutputError) as raised:
            caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertNotIn(status, str(raised.exception))

    def test_completed_with_missing_or_blank_text_rejected(self):
        for text in (MISSING, None, "", " ", "\n\t "):
            with self.subTest(text=text):
                self.assert_rejected(_response(text, status="completed"))

    def test_usage_kept_for_non_completed_response(self):
        for status in ("incomplete", "failed", "cancelled"):
            with self.subTest(status=status):
                caller = self.assert_rejected(
                    _response(FINISH, status=status, usage=_usage(900, 512, 1412)))
                self.assertEqual(caller.last_usage,
                                 ModelUsage("gpt-6-luna-2026-09-01", 900, 512, 1412))

    def test_missing_usage_not_invented_for_non_completed_response(self):
        caller = self.assert_rejected(_response(FINISH, status="failed", usage=None))
        self.assertEqual(caller.last_usage,
                         ModelUsage("gpt-6-luna-2026-09-01", None, None, None))

    def test_no_retry_or_token_increase_after_incomplete(self):
        client, caller = _caller(_response(FINISH, status="incomplete"), _response())
        with self.assertRaises(ModelOutputError):
            caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(caller.max_output_tokens, 512)
        # The next call is a new, single request with unchanged settings.
        self.assertEqual(caller.call(INSTRUCTIONS, INPUT_TEXT), FINISH)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([c["max_output_tokens"] for c in client.calls], [512, 512])

    def test_non_completed_rejection_reaches_bounded_provider(self):
        client = FakeClient(_response(FINISH, status="incomplete"))
        provider = BoundedDecisionProvider(LumenOpenAIModelCaller(client), max_calls=5,
                                           max_input_chars=12000, max_output_chars=2000,
                                           max_recent_events=6)
        with self.assertRaises(ModelOutputError):
            provider.call(INSTRUCTIONS, _context())
        self.assertEqual((len(client.calls), provider.calls_used), (1, 1))


# 14, 15 -------------------------------------------------------------------------

class UsageTests(unittest.TestCase):
    def test_usage_extracted(self):
        _, caller = _caller(_response(usage=_usage(1000, 200, 1200)))
        caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertEqual(caller.last_usage,
                         ModelUsage("gpt-6-luna-2026-09-01", 1000, 200, 1200))

    def test_usage_is_read_only(self):
        _, caller = _caller()
        caller.call(INSTRUCTIONS, INPUT_TEXT)
        with self.assertRaises(AttributeError):
            caller.last_usage.total_tokens = 0
        with self.assertRaises(AttributeError):
            caller.last_usage = None

    def test_missing_usage_handled(self):
        _, caller = _caller(_response(usage=None))
        self.assertEqual(caller.call(INSTRUCTIONS, INPUT_TEXT), FINISH)
        self.assertEqual(caller.last_usage,
                         ModelUsage("gpt-6-luna-2026-09-01", None, None, None))

    def test_partial_or_malformed_usage_not_invented(self):
        usage = types.SimpleNamespace(input_tokens=10, output_tokens="5", total_tokens=-1)
        self.assertEqual(extract_usage(_response(usage=usage, model=None)),
                         ModelUsage(None, 10, None, None))
        self.assertEqual(extract_usage(types.SimpleNamespace()),
                         ModelUsage(None, None, None, None))
        usage = types.SimpleNamespace(input_tokens=True, output_tokens=2.0)
        self.assertEqual(extract_usage(_response(usage=usage, model="")),
                         ModelUsage(None, None, None, None))

    def test_usage_reset_per_call(self):
        _, caller = _caller(_response(usage=_usage(1, 2, 3)), RuntimeError("synthetic"))
        caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertEqual(caller.last_usage.total_tokens, 3)
        with self.assertRaises(RuntimeError):
            caller.call(INSTRUCTIONS, INPUT_TEXT)
        self.assertIsNone(caller.last_usage)


# 16 ------------------------------------------------------------------------------

class ConstructorValidationTests(unittest.TestCase):
    def test_invalid_client_rejected(self):
        for client in (None, object(), types.SimpleNamespace(responses=None),
                       types.SimpleNamespace(responses=types.SimpleNamespace(create=1))):
            with self.subTest(client=client):
                with self.assertRaises(TypeError):
                    LumenOpenAIModelCaller(client)

    def test_invalid_model_rejected(self):
        for model in ("", "   ", " gpt-6-luna", "gpt-6-luna\n", None, 5):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    LumenOpenAIModelCaller(FakeClient(), model=model)

    def test_invalid_reasoning_effort_rejected(self):
        for effort in ("", "LOW", "extreme", None, 1):
            with self.subTest(effort=effort):
                with self.assertRaises(ValueError):
                    LumenOpenAIModelCaller(FakeClient(), reasoning_effort=effort)

    def test_invalid_max_output_tokens_rejected(self):
        for tokens, error in ((True, TypeError), (512.0, TypeError), ("512", TypeError),
                              (None, TypeError), (0, ValueError), (-1, ValueError),
                              (15, ValueError)):
            with self.subTest(tokens=tokens):
                with self.assertRaises(error):
                    LumenOpenAIModelCaller(FakeClient(), max_output_tokens=tokens)

    def test_settings_are_keyword_only(self):
        with self.assertRaises(TypeError):
            LumenOpenAIModelCaller(FakeClient(), "gpt-6-luna")


# 17, 18 -------------------------------------------------------------------------

class BoundedProviderCompatibilityTests(unittest.TestCase):
    def provider(self, client, max_calls=5):
        return BoundedDecisionProvider(LumenOpenAIModelCaller(client), max_calls=max_calls,
                                       max_input_chars=12000, max_output_chars=2000,
                                       max_recent_events=6)

    def test_provider_accepts_adapter(self):
        client = FakeClient()
        provider = self.provider(client)
        self.assertEqual(provider.call(INSTRUCTIONS, _context()), FINISH)
        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        self.assertEqual(sent["instructions"], INSTRUCTIONS)
        self.assertEqual(json.loads(sent["input"])["research_goal"],
                         "Bring the synthetic beam near half of full scale.")

    def test_provider_limits_calls_through_adapter(self):
        client = FakeClient()
        provider = self.provider(client, max_calls=2)
        provider.call(INSTRUCTIONS, _context())
        provider.call(INSTRUCTIONS, _context())
        with self.assertRaises(CallLimitError):
            provider.call(INSTRUCTIONS, _context())
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(provider.calls_used, 2)

    def test_provider_counts_failed_call_without_retry(self):
        client = FakeClient(RuntimeError("synthetic rate limit"))
        provider = self.provider(client)
        with self.assertRaises(RuntimeError):
            provider.call(INSTRUCTIONS, _context())
        self.assertEqual((len(client.calls), provider.calls_used), (1, 1))

    def test_adapter_output_error_reaches_provider_caller(self):
        client = FakeClient(_response(""))
        with self.assertRaises(ModelOutputError):
            self.provider(client).call(INSTRUCTIONS, _context())


# Part B: live client factory (fake openai module only) ---------------------------

class LiveFactoryTests(unittest.TestCase):
    def test_factory_disables_retries_and_sets_timeout(self):
        sentinel = object()
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = mock.Mock(return_value=sentinel)
        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            self.assertIs(create_live_openai_client(), sentinel)
        fake_openai.OpenAI.assert_called_once_with(max_retries=0, timeout=30.0)
        self.assertNotIn("api_key", fake_openai.OpenAI.call_args.kwargs)

    def test_factory_not_called_on_import_or_construction(self):
        with mock.patch.object(omc, "create_live_openai_client") as factory:
            caller = LumenOpenAIModelCaller(FakeClient())
            caller.call(INSTRUCTIONS, INPUT_TEXT)
        factory.assert_not_called()


# 19 -------------------------------------------------------------------------------

class SecretHandlingTests(unittest.TestCase):
    def test_no_key_in_repr_errors_or_logs(self):
        client, caller = _caller(_response(""), _response(42), _response(MISSING), None)
        self.assertNotIn(FAKE_KEY, repr(caller))
        self.assertNotIn(FAKE_KEY, str(caller))
        with self.assertNoLogs(level=logging.DEBUG):
            for _ in range(4):
                with self.assertRaises(ModelOutputError) as raised:
                    caller.call(INSTRUCTIONS, INPUT_TEXT)
                self.assertNotIn(FAKE_KEY, str(raised.exception))
                self.assertNotIn(FAKE_KEY, repr(raised.exception))
            with self.assertRaises(ModelInputError) as raised:
                caller.call(None, INPUT_TEXT)
            self.assertNotIn(FAKE_KEY, str(raised.exception))
            for bad in (types.SimpleNamespace(api_key=FAKE_KEY), FAKE_KEY):
                with self.assertRaises(TypeError) as raised:
                    LumenOpenAIModelCaller(bad)
                self.assertNotIn(FAKE_KEY, str(raised.exception))

    def test_module_never_touches_credentials_or_output(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for token in ("os.environ", "getenv", "api_key", "import os", "dotenv", ".env\"",
                      "print(", "logging", "logger", "open(", "write", "sk-"):
            with self.subTest(token=token):
                self.assertNotIn(token, source)


# 20, 21 ---------------------------------------------------------------------------

class IsolationTests(unittest.TestCase):
    ALLOWED_TOP_LEVEL_IMPORTS = {"__future__", "dataclasses", "typing"}

    def setUp(self):
        self.source = MODULE_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_top_level_imports_are_stdlib_only(self):
        imported = set()
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertLessEqual(imported, self.ALLOWED_TOP_LEVEL_IMPORTS)

    def test_openai_imported_only_inside_live_factory(self):
        factory = next(n for n in self.tree.body
                       if isinstance(n, ast.FunctionDef) and n.name == "create_live_openai_client")
        inside = {id(n) for n in ast.walk(factory)}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) \
                    else [a.name for a in node.names]
                if any(name.split(".")[0] == "openai" for name in names):
                    self.assertIn(id(node), inside)

    def test_no_network_during_calls(self):
        def refuse(*args, **kwargs):
            raise AssertionError("network access attempted")

        with openai_sdk_guard(self), \
                mock.patch.object(omc, "create_live_openai_client") as factory, \
                mock.patch.object(socket.socket, "connect", refuse), \
                mock.patch.object(socket, "create_connection", refuse):
            client, caller = _caller()
            self.assertEqual(caller.call(INSTRUCTIONS, INPUT_TEXT), FINISH)
        factory.assert_not_called()
        self.assertEqual(len(client.calls), 1)  # the request went to the fake client only

    def test_no_camera_mcp_or_langgraph(self):
        for token in ("import mcp", "from mcp", "camera", "langgraph", "FakeCamera", "sdk_probe", "MVSDK",
                      "ctypes", "socket", "http", "SessionGate", "asyncio", "retry("):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)
        names = {n.id for n in ast.walk(self.tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(self.tree) if isinstance(n, ast.Attribute)}
        for name in ("parse_decision", "call_tool", "session_approval", "json", "loads"):
            with self.subTest(name=name):
                self.assertNotIn(name, names)

    def test_no_anthropic_or_vendor_modules_loaded(self):
        # langgraph and openai are not listed: other HW3 modules may import them
        # legitimately in the same process. test_no_camera_mcp_or_langgraph checks
        # this module's source, and test_no_network_during_calls guards the real SDK.
        for module in ("anthropic", "sdk_probe"):
            with self.subTest(module=module):
                self.assertNotIn(module, sys.modules)


if __name__ == "__main__":
    unittest.main()
