"""
Unit tests for multi-turn session tracking and friction evaluation.
"""

import time
import pytest
from src.session import CustomerSessionState, CustomerSessionTracker


class TestCustomerSessionState:
    """Tests for per-thread session state."""

    def test_compose_model_input_cold_start(self):
        """Cold start returns the raw text with no prefix."""
        session = CustomerSessionState("t1")
        assert session.compose_model_input("hello") == "hello"

    def test_compose_model_input_with_history(self):
        """Prior turns are prefixed in causal order."""
        session = CustomerSessionState("t1")
        session.add_turn("first question", "ORDER_TRACKING_AND_DELAY", False, "R", "first reply")
        composed = session.compose_model_input("second question")
        assert "CUSTOMER: first question" in composed
        assert "AGENT: first reply" in composed
        assert composed.endswith("CURRENT CUSTOMER: second question")

    def test_compose_model_input_caps_prior_turns(self):
        """Only the last N prior turns are injected."""
        session = CustomerSessionState("t1")
        for i in range(5):
            session.add_turn(f"q{i}", "FEEDBACK_AND_GENERAL", False, "R", f"a{i}")
        composed = session.compose_model_input("current")
        assert "q0" not in composed
        assert "q4" in composed

    def test_add_turn_records_escalation(self):
        """Escalating a turn marks the session as escalated."""
        session = CustomerSessionState("t1")
        session.add_turn("help", "ACCOUNT_AND_SECURITY", True, "ACCOUNT_SECURITY_OR_LOCKOUT", "dm us")
        assert session.escalated_to_human is True
        assert session.escalation_reason == "ACCOUNT_SECURITY_OR_LOCKOUT"
        assert len(session.turns) == 1

    def test_pleasantry_suppresses_escalation(self):
        """Short gratitude should not escalate."""
        session = CustomerSessionState("t1")
        session.add_turn("my package is late", "ORDER_TRACKING_AND_DELAY", False, "R", "reply")
        result = session.evaluate_multi_turn_friction("thanks!", current_intent="FEEDBACK_AND_GENERAL")
        assert result["override_escalate"] is False
        assert result["reason"] == "CONVERSATIONAL_CLOSURE_OR_GRATITUDE"

    def test_repetition_pattern_escalates(self):
        """'already tried' style friction escalates immediately."""
        session = CustomerSessionState("t1")
        result = session.evaluate_multi_turn_friction("I already tried that twice")
        assert result["override_escalate"] is True
        assert result["friction_score"] >= 0.9

    def test_public_turn_sla_escalates(self):
        """Third public turn must deflect to a private channel."""
        session = CustomerSessionState("t1")
        session.add_turn("q1", "FEEDBACK_AND_GENERAL", False, "R", "a1")
        session.add_turn("q2", "FEEDBACK_AND_GENERAL", False, "R", "a2")
        result = session.evaluate_multi_turn_friction("q3")
        assert result["override_escalate"] is True
        assert result["reason"] == "MULTI_TURN_TWITTER_SLA_EXCEEDED_MAX_TURNS"

    def test_same_intent_repeat_escalates(self):
        """The same unresolved primary inquiry on turn 2 escalates."""
        session = CustomerSessionState("t1")
        session.add_turn("where is my order", "ORDER_TRACKING_AND_DELAY", False, "R", "a1")
        result = session.evaluate_multi_turn_friction(
            "still nothing", current_intent="ORDER_TRACKING_AND_DELAY"
        )
        assert result["override_escalate"] is True
        assert result["reason"] == "MULTI_TURN_UNRESOLVED_PRIMARY_INQUIRY"

    def test_different_intent_does_not_escalate(self):
        """A different secondary question should not trigger the repeat rule."""
        session = CustomerSessionState("t1")
        session.add_turn("where is my order", "ORDER_TRACKING_AND_DELAY", False, "R", "a1")
        result = session.evaluate_multi_turn_friction(
            "also how do I update my address?", current_intent="FEEDBACK_AND_GENERAL"
        )
        assert result["override_escalate"] is False
        assert result["reason"] == "MULTI_TURN_FLOW_WITHIN_NORMAL_BOUNDS"

    def test_missing_intent_does_not_trigger_repeat_rule(self):
        """Backward-compatible call without an intent must not misfire."""
        session = CustomerSessionState("t1")
        session.add_turn("where is my order", "ORDER_TRACKING_AND_DELAY", False, "R", "a1")
        result = session.evaluate_multi_turn_friction("still nothing")
        assert result["override_escalate"] is False


class TestCustomerSessionTracker:
    """Tests for the bounded session manager."""

    def test_get_or_create_reuses(self):
        """Same ID returns the same session object."""
        tracker = CustomerSessionTracker()
        first = tracker.get_or_create("s1")
        second = tracker.get_or_create("s1")
        assert first is second

    def test_lru_eviction_bounds_memory(self):
        """Tracker never exceeds max_sessions entries."""
        tracker = CustomerSessionTracker(max_sessions=3, ttl_seconds=0)
        for i in range(10):
            tracker.get_or_create(f"s{i}")
        assert len(tracker.sessions) == 3

    def test_lru_keeps_most_recent(self):
        """The most recently used session survives eviction."""
        tracker = CustomerSessionTracker(max_sessions=2, ttl_seconds=0)
        tracker.get_or_create("s1")
        tracker.get_or_create("s2")
        # Touch s1 so it becomes most recent, then insert s3 to evict s2
        tracker.get_or_create("s1").add_turn("q", "FEEDBACK_AND_GENERAL", False, "R", "a")
        tracker.get_or_create("s3")
        assert "s1" in tracker.sessions
        assert "s2" not in tracker.sessions

    def test_ttl_expiry(self):
        """Idle sessions beyond TTL are pruned on access."""
        tracker = CustomerSessionTracker(max_sessions=10, ttl_seconds=0.05)
        session = tracker.get_or_create("s1")
        session.last_seen_at = time.time() - 10
        tracker.get_or_create("s2")
        assert "s1" not in tracker.sessions
        assert "s2" in tracker.sessions

    def test_active_count(self):
        """active_count reports live sessions."""
        tracker = CustomerSessionTracker()
        tracker.get_or_create("a")
        tracker.get_or_create("b")
        assert tracker.active_count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])