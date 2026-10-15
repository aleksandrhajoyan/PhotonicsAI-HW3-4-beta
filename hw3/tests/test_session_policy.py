import ast
import dataclasses
import hashlib
import json
import sys
import unittest
from pathlib import Path

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import session_policy as sp  # noqa: E402
from session_policy import (  # noqa: E402
    PolicyError,
    SessionEnvelope,
    SessionGate,
    envelope_sha256,
    session_approval,
)

ALL_TOOLS = (
    "camera_info",
    "camera_open",
    "camera_set_exposure",
    "camera_measure",
    "camera_close",
)


def _envelope(**overrides):
    fields = {
        "session_id": "session-001",
        "device_label": "lab-camera-A",
        "allowed_tools": ALL_TOOLS,
        "min_exposure_us": 100.0,
        "max_exposure_us": 10000.0,
        "max_measurements": 2,
        "max_tool_calls": 5,
        "max_duration_seconds": 60.0,
    }
    fields.update(overrides)
    return SessionEnvelope(**fields)


class FakeClock:
    """Deterministic monotonic clock; time moves only when advance() is called."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _gate(clock=None, **overrides):
    return SessionGate(_envelope(**overrides), clock=clock or FakeClock())


def _active_gate(clock=None, **overrides):
    gate = _gate(clock, **overrides)
    approved = gate.submit_decision(
        session_approval(gate.envelope.session_id, gate.envelope_sha256, True)
    )
    assert approved and gate.state == sp.ACTIVE
    return gate


def _run(gate, action_id, tool, arguments=None, outcome=sp.SUCCEEDED):
    grant = gate.begin_action(action_id, tool, arguments)
    gate.complete_action(action_id, outcome)
    return grant


class EnvelopeValidationTests(unittest.TestCase):
    def test_valid_envelope(self):
        env = _envelope()
        self.assertEqual(env.session_id, "session-001")
        self.assertEqual(env.allowed_tools, ALL_TOOLS)
        self.assertEqual(env.max_measurements, 2)
        self.assertEqual(env.max_tool_calls, 5)

    def test_envelope_is_immutable(self):
        env = _envelope()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            env.max_measurements = 100
        with self.assertRaises(dataclasses.FrozenInstanceError):
            env.allowed_tools = ALL_TOOLS + ("python_exec",)

    def test_int_limits_normalized_to_float_without_changing_value(self):
        env = _envelope(min_exposure_us=100, max_exposure_us=10000, max_duration_seconds=60)
        self.assertIsInstance(env.min_exposure_us, float)
        self.assertEqual(env.min_exposure_us, 100.0)
        self.assertEqual(env.sha256(), _envelope().sha256())

    def test_empty_identifiers_rejected(self):
        for field in ("session_id", "device_label"):
            for bad in ("", "   ", " padded", "line\nbreak"):
                with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                    _envelope(**{field: bad})
            with self.subTest(field=field, value=None), self.assertRaises(TypeError):
                _envelope(**{field: None})

    def test_invalid_numeric_parameters_rejected(self):
        nan, inf = float("nan"), float("inf")
        cases = [
            ("min_exposure_us", True, TypeError),
            ("min_exposure_us", "100", TypeError),
            ("min_exposure_us", nan, ValueError),
            ("min_exposure_us", inf, ValueError),
            ("min_exposure_us", 0.0, ValueError),
            ("min_exposure_us", -5.0, ValueError),
            ("max_exposure_us", 99.0, ValueError),
            ("max_exposure_us", inf, ValueError),
            ("max_exposure_us", False, TypeError),
            ("max_measurements", 0, ValueError),
            ("max_measurements", -1, ValueError),
            ("max_measurements", 2.0, TypeError),
            ("max_measurements", True, TypeError),
            ("max_tool_calls", 0, ValueError),
            ("max_tool_calls", 1, ValueError),  # < max_measurements (2)
            ("max_tool_calls", 5.0, TypeError),
            ("max_tool_calls", True, TypeError),
            ("max_duration_seconds", 0.0, ValueError),
            ("max_duration_seconds", -1.0, ValueError),
            ("max_duration_seconds", nan, ValueError),
            ("max_duration_seconds", inf, ValueError),
            ("max_duration_seconds", True, TypeError),
        ]
        for field, value, error in cases:
            with self.subTest(field=field, value=value), self.assertRaises(error):
                _envelope(**{field: value})

    def test_equal_min_and_max_exposure_allowed(self):
        env = _envelope(min_exposure_us=500.0, max_exposure_us=500.0)
        self.assertEqual(env.max_exposure_us, 500.0)

    def test_unknown_tools_rejected(self):
        for bad in ("python_exec", "camera_set_gain", "shell", "CAMERA_MEASURE", ""):
            with self.subTest(tool=bad), self.assertRaises(ValueError):
                _envelope(allowed_tools=("camera_info", bad))

    def test_duplicate_tools_rejected(self):
        with self.assertRaises(ValueError):
            _envelope(allowed_tools=("camera_measure", "camera_measure"))

    def test_malformed_tool_collections_rejected(self):
        with self.assertRaises(TypeError):
            _envelope(allowed_tools=["camera_info"])
        with self.assertRaises(TypeError):
            _envelope(allowed_tools="camera_info")
        with self.assertRaises(TypeError):
            _envelope(allowed_tools=("camera_info", 7))
        with self.assertRaises(ValueError):
            _envelope(allowed_tools=())

    def test_envelope_has_no_command_or_code_fields(self):
        names = {f.name for f in dataclasses.fields(SessionEnvelope)}
        self.assertEqual(
            names,
            {
                "session_id", "device_label", "allowed_tools", "min_exposure_us",
                "max_exposure_us", "max_measurements", "max_tool_calls", "max_duration_seconds",
            },
        )
        with self.assertRaises(TypeError):
            _envelope(device_command="rm -rf /")

    def test_tool_vocabulary_matches_mcp_server(self):
        # Parse (never import) mcp_server.py so no MCP SDK code runs.
        tree = ast.parse((HW3_DIR / "mcp_server.py").read_text(encoding="utf-8"))
        server_tools = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    if (
                        isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "tool"
                    ):
                        server_tools.add(node.name)
        self.assertEqual(server_tools, set(sp.CAMERA_TOOLS))
        self.assertEqual(sp.CAMERA_TOOLS, ALL_TOOLS)


class FingerprintTests(unittest.TestCase):
    def test_identical_envelopes_identical_fingerprints(self):
        a, b = _envelope(), _envelope()
        self.assertIsNot(a, b)
        self.assertEqual(envelope_sha256(a), envelope_sha256(b))
        self.assertEqual(a.sha256(), envelope_sha256(a))
        self.assertRegex(a.sha256(), r"^[0-9a-f]{64}$")

    def test_canonical_json_is_exact_and_stable(self):
        expected = (
            '{"allowed_tools":["camera_info","camera_open","camera_set_exposure",'
            '"camera_measure","camera_close"],"device_label":"lab-camera-A",'
            '"max_duration_seconds":60.0,"max_exposure_us":10000.0,"max_measurements":2,'
            '"max_tool_calls":5,"min_exposure_us":100.0,'
            '"schema":"lumen.hw3.session_envelope.v1","session_id":"session-001"}'
        )
        env = _envelope()
        self.assertEqual(env.canonical_json(), expected)
        self.assertEqual(env.sha256(), hashlib.sha256(expected.encode("utf-8")).hexdigest())
        self.assertEqual(json.loads(env.canonical_json()), env.canonical_dict())

    def test_fingerprint_changes_when_a_limit_changes(self):
        base = _envelope().sha256()
        self.assertNotEqual(base, _envelope(max_measurements=3).sha256())

    def test_fingerprint_changes_for_every_field(self):
        base = _envelope().sha256()
        changes = {
            "session_id": "session-002",
            "device_label": "lab-camera-B",
            "allowed_tools": ("camera_info", "camera_measure"),
            "min_exposure_us": 101.0,
            "max_exposure_us": 9999.0,
            "max_measurements": 1,
            "max_tool_calls": 6,
            "max_duration_seconds": 61.0,
        }
        self.assertEqual(set(changes), {f.name for f in dataclasses.fields(SessionEnvelope)})
        for field, value in changes.items():
            with self.subTest(field=field):
                self.assertNotEqual(base, _envelope(**{field: value}).sha256())

    def test_fingerprint_contains_no_secrets(self):
        payload = _envelope().canonical_dict()
        for key in payload:
            for word in ("secret", "key", "token", "password", "credential"):
                self.assertNotIn(word, key.lower())


class ApprovalTests(unittest.TestCase):
    def test_initial_state_awaiting_approval(self):
        gate = _gate()
        self.assertEqual(gate.state, sp.AWAITING_APPROVAL)
        request = gate.approval_request()
        self.assertEqual(request["session_id"], "session-001")
        self.assertEqual(request["envelope_sha256"], gate.envelope_sha256)
        self.assertEqual(request["envelope"], gate.envelope.canonical_dict())

    def test_actions_rejected_before_approval(self):
        gate = _gate()
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_info")
        self.assertEqual(gate.state, sp.AWAITING_APPROVAL)
        self.assertEqual(gate.tool_calls_used, 0)

    def test_valid_approval_activates(self):
        gate = _gate()
        reply = session_approval("session-001", gate.envelope_sha256, True)
        self.assertTrue(gate.submit_decision(reply))
        self.assertEqual(gate.state, sp.ACTIVE)

    def test_extra_text_field_does_not_block_valid_approval(self):
        gate = _gate()
        reply = {**session_approval("session-001", gate.envelope_sha256, True), "text": "ok"}
        self.assertTrue(gate.submit_decision(reply))

    def _assert_refused(self, reply):
        gate = _gate()
        self.assertFalse(gate.submit_decision(reply))
        self.assertEqual(gate.state, sp.REFUSED)
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_info")
        return gate

    def test_wrong_session_id_fails_closed(self):
        digest = _gate().envelope_sha256
        for bad in ("session-002", "session-001 ", "SESSION-001", None, 1):
            with self.subTest(session_id=bad):
                self._assert_refused(session_approval(bad, digest, True))

    def test_wrong_hash_fails_closed(self):
        digest = _gate().envelope_sha256
        other = _envelope(max_measurements=3).sha256()
        for bad in (other, digest.upper(), digest[:-1], "", None):
            with self.subTest(digest=bad):
                self._assert_refused(session_approval("session-001", bad, True))

    def test_free_text_approval_fails_closed(self):
        digest = _gate().envelope_sha256
        for reply in ("approve", "yes", "approved: true", True, None, ["approve"]):
            with self.subTest(reply=reply):
                self._assert_refused(reply)
        for approved in ("true", 1, "yes", None):
            with self.subTest(approved=approved):
                self._assert_refused(session_approval("session-001", digest, approved))

    def test_missing_fields_fail_closed(self):
        digest = _gate().envelope_sha256
        self._assert_refused({"approved": True, "session_id": "session-001"})
        self._assert_refused({"approved": True, "envelope_sha256": digest})
        self._assert_refused({})

    def test_refusal_is_terminal(self):
        gate = _gate()
        self.assertFalse(gate.submit_decision(session_approval("session-001", gate.envelope_sha256, False)))
        self.assertEqual(gate.state, sp.REFUSED)
        self.assertFalse(gate.submit_decision(session_approval("session-001", gate.envelope_sha256, True)))
        self.assertEqual(gate.state, sp.REFUSED)
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_open")

    def test_second_approval_on_active_gate_changes_nothing(self):
        gate = _active_gate()
        self.assertFalse(gate.submit_decision("refuse"))
        self.assertEqual(gate.state, sp.ACTIVE)


class ActionTests(unittest.TestCase):
    def test_authorized_action_accepted(self):
        gate = _active_gate()
        grant = gate.begin_action("a1", "camera_set_exposure", {"exposure_us": 500.0})
        self.assertEqual(grant.tool, "camera_set_exposure")
        self.assertEqual(grant.exposure_us, 500.0)
        self.assertEqual(grant.session_id, "session-001")
        self.assertEqual(grant.envelope_sha256, gate.envelope_sha256)
        self.assertEqual(gate.action_status("a1"), sp.IN_FLIGHT)
        self.assertEqual(gate.in_flight_action_id, "a1")
        self.assertEqual(gate.complete_action("a1", sp.SUCCEEDED), sp.ACTIVE)
        self.assertEqual(gate.action_status("a1"), sp.SUCCEEDED)
        self.assertIsNone(gate.in_flight_action_id)

    def test_exposure_range_boundaries_inclusive(self):
        gate = _active_gate(max_tool_calls=5)
        _run(gate, "a1", "camera_set_exposure", {"exposure_us": 100.0})
        _run(gate, "a2", "camera_set_exposure", {"exposure_us": 10000})

    def test_unauthorized_tool_rejected(self):
        gate = _active_gate(allowed_tools=("camera_info", "camera_measure"))
        for tool in ("camera_set_exposure", "camera_open", "python_exec", None):
            with self.subTest(tool=tool), self.assertRaises(PolicyError):
                gate.begin_action(f"id-{tool}", tool)
        self.assertEqual(gate.tool_calls_used, 0)
        self.assertEqual(gate.state, sp.ACTIVE)

    def test_out_of_range_exposure_rejected(self):
        gate = _active_gate()
        for i, value in enumerate((99.999, 0.0, -1.0, 10000.001, 1e9)):
            with self.subTest(value=value), self.assertRaises(PolicyError):
                gate.begin_action(f"x{i}", "camera_set_exposure", {"exposure_us": value})
        self.assertEqual(gate.tool_calls_used, 0)

    def test_invalid_numeric_exposure_rejected(self):
        gate = _active_gate()
        bad_args = [
            {"exposure_us": float("nan")},
            {"exposure_us": float("inf")},
            {"exposure_us": float("-inf")},
            {"exposure_us": True},
            {"exposure_us": "500"},
            {"exposure_us": None},
            {},
            None,
            {"exposure_us": 500.0, "gain": 2},
            {"value": 500.0},
        ]
        for i, args in enumerate(bad_args):
            with self.subTest(args=args), self.assertRaises(PolicyError):
                gate.begin_action(f"x{i}", "camera_set_exposure", args)
        self.assertEqual(gate.tool_calls_used, 0)

    def test_arguments_rejected_for_argument_free_tools(self):
        gate = _active_gate()
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_measure", {"exposure_us": 500.0})

    def test_measurement_budget_enforced(self):
        gate = _active_gate(max_measurements=2, max_tool_calls=5)
        _run(gate, "m1", "camera_measure")
        _run(gate, "m2", "camera_measure")
        with self.assertRaises(PolicyError):
            gate.begin_action("m3", "camera_measure")
        self.assertEqual(gate.measurements_used, 2)
        self.assertEqual(gate.tool_calls_used, 2)
        _run(gate, "i1", "camera_info")  # other tools still within budget

    def test_tool_call_budget_enforced(self):
        gate = _active_gate(max_measurements=1, max_tool_calls=3)
        _run(gate, "a1", "camera_open")
        _run(gate, "a2", "camera_set_exposure", {"exposure_us": 500.0})
        self.assertEqual(gate.state, sp.ACTIVE)
        _run(gate, "a3", "camera_measure")
        self.assertEqual(gate.state, sp.EXHAUSTED)
        for i, (tool, args) in enumerate(
            (("camera_measure", None), ("camera_set_exposure", {"exposure_us": 500.0}))
        ):
            with self.subTest(tool=tool), self.assertRaises(PolicyError):
                gate.begin_action(f"b{i}", tool, args)
        self.assertEqual(gate.tool_calls_used, 3)

    def test_budget_reserved_before_execution_and_not_refunded(self):
        gate = _active_gate(max_measurements=1, max_tool_calls=2)
        gate.begin_action("m1", "camera_measure")
        self.assertEqual(gate.measurements_used, 1)
        self.assertEqual(gate.tool_calls_used, 1)
        gate.complete_action("m1", sp.FAILED)
        self.assertEqual(gate.action_status("m1"), sp.FAILED)
        self.assertEqual(gate.state, sp.ACTIVE)
        with self.assertRaises(PolicyError):
            gate.begin_action("m2", "camera_measure")

    def test_only_one_action_in_flight(self):
        gate = _active_gate()
        gate.begin_action("a1", "camera_open")
        with self.assertRaises(PolicyError):
            gate.begin_action("a2", "camera_info")
        self.assertEqual(gate.tool_calls_used, 1)

    def test_session_duration_enforced_with_fake_clock(self):
        clock = FakeClock()
        gate = _active_gate(clock, max_duration_seconds=60.0)
        clock.advance(59.999)
        _run(gate, "a1", "camera_info")
        clock.advance(0.001)
        with self.assertRaises(PolicyError):
            gate.begin_action("a2", "camera_measure")
        self.assertEqual(gate.state, sp.EXPIRED)
        self.assertEqual(gate.measurements_used, 0)
        with self.assertRaises(PolicyError):
            gate.begin_action("a3", "camera_set_exposure", {"exposure_us": 500.0})

    def test_expiry_detected_when_completing_long_action(self):
        clock = FakeClock()
        gate = _active_gate(clock, max_duration_seconds=10.0)
        gate.begin_action("m1", "camera_measure")
        clock.advance(11.0)
        self.assertEqual(gate.complete_action("m1", sp.SUCCEEDED), sp.EXPIRED)
        self.assertEqual(gate.action_status("m1"), sp.SUCCEEDED)

    def test_clock_going_backwards_aborts(self):
        clock = FakeClock()
        gate = _active_gate(clock)
        clock.advance(-1.0)
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_info")
        self.assertEqual(gate.state, sp.ABORTED)

    def test_faulty_clock_aborts(self):
        for value in (float("nan"), "now", None):
            with self.subTest(value=value):
                clock = FakeClock()
                gate = _active_gate(clock)
                clock.now = value
                with self.assertRaises(PolicyError):
                    gate.begin_action("a1", "camera_info")
                self.assertEqual(gate.state, sp.ABORTED)

    def test_duplicate_action_id_rejected(self):
        gate = _active_gate()
        gate.begin_action("a1", "camera_open")
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_open")
        gate.complete_action("a1", sp.SUCCEEDED)
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_info")

    def test_rejected_action_id_cannot_be_reused(self):
        gate = _active_gate()
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_set_exposure", {"exposure_us": 1.0})
        with self.assertRaises(PolicyError):
            gate.begin_action("a1", "camera_set_exposure", {"exposure_us": 500.0})

    def test_malformed_action_id_rejected(self):
        gate = _active_gate()
        with self.assertRaises(ValueError):
            gate.begin_action("", "camera_info")
        with self.assertRaises(TypeError):
            gate.begin_action(7, "camera_info")
        self.assertEqual(gate.tool_calls_used, 0)

    def test_ambiguous_failure_aborts_session(self):
        gate = _active_gate()
        gate.begin_action("m1", "camera_measure")
        self.assertEqual(gate.complete_action("m1", sp.AMBIGUOUS), sp.ABORTED)
        self.assertEqual(gate.action_status("m1"), sp.AMBIGUOUS)
        for i, (tool, args) in enumerate(
            (("camera_measure", None), ("camera_set_exposure", {"exposure_us": 500.0}))
        ):
            with self.subTest(tool=tool), self.assertRaises(PolicyError):
                gate.begin_action(f"retry{i}", tool, args)
        with self.assertRaises(PolicyError):
            gate.begin_action("m1", "camera_measure")

    def test_unrecognized_outcome_treated_as_ambiguous(self):
        gate = _active_gate()
        gate.begin_action("m1", "camera_measure")
        self.assertEqual(gate.complete_action("m1", "timeout"), sp.ABORTED)

    def test_completed_actions_cannot_be_replayed(self):
        gate = _active_gate()
        _run(gate, "m1", "camera_measure")
        with self.assertRaises(PolicyError):
            gate.complete_action("m1", sp.SUCCEEDED)
        with self.assertRaises(PolicyError):
            gate.begin_action("m1", "camera_measure")
        with self.assertRaises(PolicyError):
            gate.complete_action("never-begun", sp.SUCCEEDED)
        self.assertEqual(gate.measurements_used, 1)


class TerminationTests(unittest.TestCase):
    def test_closed_session_cannot_be_reused(self):
        gate = _active_gate()
        _run(gate, "a1", "camera_open")
        self.assertEqual(gate.close(), sp.CLOSED)
        for i, (tool, args) in enumerate(
            (("camera_measure", None), ("camera_set_exposure", {"exposure_us": 500.0}),
             ("camera_info", None))
        ):
            with self.subTest(tool=tool), self.assertRaises(PolicyError):
                gate.begin_action(f"b{i}", tool, args)
        self.assertFalse(gate.submit_decision(session_approval("session-001", gate.envelope_sha256, True)))
        self.assertEqual(gate.state, sp.CLOSED)

    def test_close_with_in_flight_action_aborts(self):
        gate = _active_gate()
        gate.begin_action("m1", "camera_measure")
        self.assertEqual(gate.close(), sp.ABORTED)
        self.assertEqual(gate.action_status("m1"), sp.AMBIGUOUS)
        self.assertIsNone(gate.in_flight_action_id)

    def test_close_and_abort_available_after_errors(self):
        # After every kind of failure, close() and abort() never raise and the
        # gate stays terminal.
        clock = FakeClock()
        scenarios = {}

        refused = _gate()
        refused.submit_decision("yes")
        scenarios["refused"] = refused

        aborted = _active_gate()
        aborted.begin_action("m1", "camera_measure")
        aborted.complete_action("m1", sp.AMBIGUOUS)
        scenarios["aborted"] = aborted

        exhausted = _active_gate(max_measurements=1, max_tool_calls=1)
        _run(exhausted, "m1", "camera_measure")
        scenarios["exhausted"] = exhausted

        expired = _active_gate(clock, max_duration_seconds=1.0)
        clock.advance(2.0)
        with self.assertRaises(PolicyError):
            expired.begin_action("a1", "camera_info")
        scenarios["expired"] = expired

        rejected = _active_gate()
        with self.assertRaises(PolicyError):
            rejected.begin_action("a1", "python_exec")
        scenarios["rejected-then-active"] = rejected

        for name, gate in scenarios.items():
            with self.subTest(scenario=name):
                first = gate.abort("operator abort")
                second = gate.close()
                third = gate.abort()
                self.assertIn(first, sp.TERMINAL_STATES)
                self.assertIn(second, sp.TERMINAL_STATES)
                self.assertEqual(second, third)
                with self.assertRaises(PolicyError):
                    gate.begin_action("after", "camera_measure")

        self.assertEqual(scenarios["refused"].state, sp.REFUSED)
        self.assertEqual(scenarios["exhausted"].state, sp.ABORTED)
        self.assertEqual(scenarios["expired"].state, sp.ABORTED)

    def test_exhausted_and_expired_can_close(self):
        exhausted = _active_gate(max_measurements=1, max_tool_calls=1)
        _run(exhausted, "m1", "camera_measure")
        self.assertEqual(exhausted.state, sp.EXHAUSTED)
        self.assertEqual(exhausted.close(), sp.CLOSED)

    def test_awaiting_approval_can_be_closed(self):
        gate = _gate()
        self.assertEqual(gate.close(), sp.CLOSED)
        self.assertFalse(gate.submit_decision(session_approval("session-001", gate.envelope_sha256, True)))

    def test_state_vocabulary(self):
        self.assertEqual(
            sp.STATES,
            ("awaiting_approval", "active", "refused", "exhausted", "expired", "aborted", "closed"),
        )


class IsolationTests(unittest.TestCase):
    ALLOWED_IMPORTS = {
        "__future__", "hashlib", "json", "math", "time", "collections.abc",
        "dataclasses", "typing",
    }

    def test_module_imports_only_stdlib(self):
        tree = ast.parse((HW3_DIR / "session_policy.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported - self.ALLOWED_IMPORTS, set())

    def test_no_sdk_mcp_langgraph_or_code_execution(self):
        source = (HW3_DIR / "session_policy.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for forbidden in ("exec", "eval", "compile", "__import__", "open"):
            self.assertNotIn(forbidden, called)
        for token in ("ctypes", "subprocess", "socket", "langgraph", "MCPServer", "MvCamera", "sleep("):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
