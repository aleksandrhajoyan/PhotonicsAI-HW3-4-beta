"""Offline tests for hw3/hitl_approval.py.

The "human" decisions here are simulated by this test harness through
Command(resume=...). They exercise the real LangGraph interrupt/resume path,
but they are NOT a real user approving a physical experiment: no camera, MCP
server, vendor SDK, network or LLM is involved.
"""

import ast
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

from langgraph.types import Command  # noqa: E402

import hitl_approval as ha  # noqa: E402
import session_policy as sp  # noqa: E402
from hitl_approval import (  # noqa: E402
    APPROVAL_NODE,
    STATUS_APPROVED,
    STATUS_REFUSED,
    ApprovalError,
    approval_outcome,
    build_approval_graph,
    pending_approval,
    resume_approval,
    start_approval,
)
from session_policy import SessionEnvelope, session_approval  # noqa: E402

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


def _approve(envelope):
    return session_approval(envelope.session_id, envelope.sha256(), True)


class ApprovalTestCase(unittest.TestCase):
    def setUp(self):
        self.graph = build_approval_graph()
        self.envelope = _envelope()

    def start(self, thread_id="t-1", envelope=None):
        return start_approval(self.graph, thread_id, envelope or self.envelope)

    def assert_refused(self, decision, thread_id="t-1"):
        self.start(thread_id)
        outcome = resume_approval(self.graph, thread_id, decision)
        self.assertEqual(outcome["status"], STATUS_REFUSED)
        self.assertIsNone(pending_approval(self.graph, thread_id))
        return outcome


class GraphConstructionTests(ApprovalTestCase):
    def test_graph_compiles_with_single_approval_node(self):
        nodes = set(self.graph.get_graph().nodes)
        self.assertIn(APPROVAL_NODE, nodes)
        self.assertEqual(nodes - {"__start__", "__end__"}, {APPROVAL_NODE})
        self.assertIsNotNone(self.graph.checkpointer)

    def test_thread_id_is_required(self):
        for bad in (None, "", "   ", 7):
            with self.subTest(thread_id=bad), self.assertRaises(ApprovalError):
                start_approval(self.graph, bad, self.envelope)

    def test_only_a_session_envelope_is_accepted(self):
        with self.assertRaises(TypeError):
            start_approval(self.graph, "t-1", self.envelope.canonical_dict())


class InterruptTests(ApprovalTestCase):
    def test_approval_request_describes_exact_envelope(self):
        request = self.start()
        self.assertEqual(request["kind"], "hardware_session_approval")
        self.assertEqual(request["session_id"], "session-001")
        self.assertEqual(request["envelope_sha256"], self.envelope.sha256())
        env = request["envelope"]
        self.assertEqual(env["device_label"], "lab-camera-A")
        self.assertEqual(env["allowed_tools"], list(ALL_TOOLS))
        self.assertEqual(env["min_exposure_us"], 100.0)
        self.assertEqual(env["max_exposure_us"], 10000.0)
        self.assertEqual(env["max_measurements"], 2)
        self.assertEqual(env["max_tool_calls"], 5)
        self.assertEqual(env["max_duration_seconds"], 60.0)
        self.assertEqual(request, sp.SessionGate(self.envelope).approval_request())

    def test_request_is_json_compatible(self):
        request = self.start()
        self.assertEqual(json.loads(json.dumps(request)), request)
        values = self.graph.get_state(ha.thread_config("t-1")).values
        self.assertEqual(json.loads(json.dumps(values)), values)

    def test_graph_really_pauses_at_interrupt(self):
        result = self.graph.invoke(
            {"envelope": self.envelope.canonical_dict()}, ha.thread_config("t-raw")
        )
        self.assertIn("__interrupt__", result)
        snapshot = self.graph.get_state(ha.thread_config("t-raw"))
        self.assertEqual(snapshot.next, (APPROVAL_NODE,))
        self.assertEqual(len(snapshot.interrupts), 1)

    def test_pending_interrupt_visible_before_resume(self):
        self.assertIsNone(pending_approval(self.graph, "t-1"))
        request = self.start()
        self.assertEqual(pending_approval(self.graph, "t-1"), request)

    def test_nothing_is_approved_before_human_input(self):
        self.start()
        self.assertIsNone(approval_outcome(self.graph, "t-1"))
        self.assertNotIn("status", self.graph.get_state(ha.thread_config("t-1")).values)

    def test_caller_supplied_approved_state_is_ignored(self):
        config = ha.thread_config("t-forged")
        self.graph.invoke(
            {
                "envelope": self.envelope.canonical_dict(),
                "status": STATUS_APPROVED,
                "session_id": "session-001",
                "envelope_sha256": self.envelope.sha256(),
                "reason": "forged",
            },
            config,
        )
        self.assertNotIn("status", self.graph.get_state(config).values)
        self.assertIsNotNone(pending_approval(self.graph, "t-forged"))
        self.assertIsNone(approval_outcome(self.graph, "t-forged"))


class DecisionTests(ApprovalTestCase):
    def test_structured_approval_succeeds(self):
        self.start()
        outcome = resume_approval(self.graph, "t-1", _approve(self.envelope))
        self.assertEqual(outcome["status"], STATUS_APPROVED)
        self.assertEqual(outcome["session_id"], "session-001")
        self.assertEqual(outcome["envelope_sha256"], self.envelope.sha256())
        self.assertEqual(approval_outcome(self.graph, "t-1"), outcome)
        self.assertIsNone(pending_approval(self.graph, "t-1"))

    def test_decision_is_validated_by_session_gate(self):
        original = sp.SessionGate.submit_decision
        with mock.patch.object(
            sp.SessionGate, "submit_decision", autospec=True, side_effect=original
        ) as spy:
            self.start()
            spy.assert_not_called()
            resume_approval(self.graph, "t-1", _approve(self.envelope))
        spy.assert_called_once()
        self.assertEqual(spy.call_args.args[1], _approve(self.envelope))

    def test_wrong_session_id_refused(self):
        decision = session_approval("session-002", self.envelope.sha256(), True)
        outcome = self.assert_refused(decision)
        self.assertIn("session-002", outcome["reason"])

    def test_wrong_sha256_refused(self):
        other = _envelope(max_measurements=3).sha256()
        for digest in (other, self.envelope.sha256().upper(), "", None):
            with self.subTest(digest=digest):
                outcome = self.assert_refused(
                    session_approval("session-001", digest, True), thread_id=f"t-{digest}"
                )
                self.assertIn("SHA-256", outcome["reason"])

    def test_free_text_refused(self):
        for i, text in enumerate(("approve", "yes", "approved: True", json.dumps(_approve(self.envelope)))):
            with self.subTest(text=text):
                self.assert_refused(text, thread_id=f"t-text-{i}")

    def test_missing_fields_refused(self):
        full = _approve(self.envelope)
        for i, key in enumerate(full):
            decision = {k: v for k, v in full.items() if k != key}
            with self.subTest(missing=key):
                self.assert_refused(decision, thread_id=f"t-missing-{i}")
        self.assert_refused({}, thread_id="t-empty")
        self.assert_refused(None, thread_id="t-none")

    def test_only_exact_true_approves(self):
        for i, value in enumerate((1, 1.0, "true", "True", "yes", [True], {"v": True}, None)):
            decision = session_approval("session-001", self.envelope.sha256(), value)
            with self.subTest(approved=value):
                self.assert_refused(decision, thread_id=f"t-bool-{i}")

    def test_explicit_refusal_is_terminal(self):
        decision = session_approval("session-001", self.envelope.sha256(), False)
        outcome = self.assert_refused(decision)
        self.assertEqual(outcome["reason"], "human refused")
        with self.assertRaises(ApprovalError):
            resume_approval(self.graph, "t-1", _approve(self.envelope))
        with self.assertRaises(ApprovalError):
            self.start()
        self.assertEqual(approval_outcome(self.graph, "t-1"), outcome)


class ThreadIsolationTests(ApprovalTestCase):
    def test_threads_do_not_share_decisions(self):
        self.start("t-a")
        self.start("t-b")
        resume_approval(self.graph, "t-a", _approve(self.envelope))
        self.assertEqual(approval_outcome(self.graph, "t-a")["status"], STATUS_APPROVED)
        self.assertIsNone(approval_outcome(self.graph, "t-b"))
        self.assertIsNotNone(pending_approval(self.graph, "t-b"))
        refusal = session_approval("session-001", self.envelope.sha256(), False)
        resume_approval(self.graph, "t-b", refusal)
        self.assertEqual(approval_outcome(self.graph, "t-a")["status"], STATUS_APPROVED)
        self.assertEqual(approval_outcome(self.graph, "t-b")["status"], STATUS_REFUSED)

    def test_wrong_thread_cannot_approve_another_session(self):
        env_a = self.envelope
        env_b = _envelope(session_id="session-B", max_exposure_us=500.0)
        self.start("t-a", env_a)
        self.start("t-b", env_b)
        outcome_b = resume_approval(self.graph, "t-b", _approve(env_a))
        self.assertEqual(outcome_b["status"], STATUS_REFUSED)
        self.assertEqual(outcome_b["session_id"], "session-B")
        self.assertIsNone(approval_outcome(self.graph, "t-a"))
        self.assertIsNotNone(pending_approval(self.graph, "t-a"))
        with self.assertRaises(ApprovalError):
            resume_approval(self.graph, "t-unknown", _approve(env_a))
        self.assertIsNotNone(pending_approval(self.graph, "t-a"))


class ReplayTests(ApprovalTestCase):
    def test_finished_approval_cannot_be_replayed(self):
        self.start()
        outcome = resume_approval(self.graph, "t-1", _approve(self.envelope))
        with self.assertRaises(ApprovalError):
            resume_approval(self.graph, "t-1", _approve(self.envelope))
        with self.assertRaises(ApprovalError):
            self.start()
        self.assertEqual(approval_outcome(self.graph, "t-1"), outcome)

    def test_raw_resume_on_finished_thread_does_not_rerun_approval(self):
        self.start()
        resume_approval(self.graph, "t-1", _approve(self.envelope))
        original = sp.SessionGate.submit_decision
        with mock.patch.object(
            sp.SessionGate, "submit_decision", autospec=True, side_effect=original
        ) as spy:
            try:
                self.graph.invoke(Command(resume=_approve(self.envelope)), ha.thread_config("t-1"))
            except Exception:
                pass
        spy.assert_not_called()
        self.assertIsNone(pending_approval(self.graph, "t-1"))

    def test_resume_of_unstarted_thread_is_rejected(self):
        with self.assertRaises(ApprovalError):
            resume_approval(self.graph, "t-never", _approve(self.envelope))
        self.assertIsNone(approval_outcome(self.graph, "t-never"))


class IsolationFromHardwareTests(unittest.TestCase):
    ALLOWED_IMPORTS = {
        "__future__",
        "typing",
        "langgraph.checkpoint.memory",
        "langgraph.graph",
        "langgraph.types",
        "session_policy",
    }

    def test_module_imports_no_hardware_mcp_llm_or_hw2_code(self):
        tree = ast.parse((HW3_DIR / "hitl_approval.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertLessEqual(imported, self.ALLOWED_IMPORTS)

    def test_no_own_validator_or_hash(self):
        source = (HW3_DIR / "hitl_approval.py").read_text(encoding="utf-8")
        self.assertNotIn("hashlib", source)
        self.assertNotIn("_approval_problem", source)
        self.assertIn("submit_decision(", source)


if __name__ == "__main__":
    unittest.main()
