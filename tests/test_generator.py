"""
Unit tests for response generation.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.generator import (
    ProductionGroundedGenerator,
    BaselineRetrievalOnlyGenerator,
    BaselineGenerativeGenerator,
    TrivialReplyBaseline,
    RetrievalReplayBaseline,
)


class TestProductionGroundedGenerator:
    """Tests for the ProductionGroundedGenerator class."""

    def setup_method(self):
        self.generator = ProductionGroundedGenerator(mode="offline")

    def test_act_selection_escalation(self):
        """Escalation should select DM_WITH_ORDER_ID."""
        exemplars = [{"resolution_act": "LINK_ORDERS"}]
        act = self.generator.select_act(exemplars, "ORDER_TRACKING_AND_DELAY", True)
        assert act == "DM_WITH_ORDER_ID"

    def test_act_selection_by_intent(self):
        """Act selection should match intent defaults."""
        tests = [
            ("ORDER_TRACKING_AND_DELAY", "LINK_ORDERS"),
            ("REFUND_AND_RETURN", "LINK_RETURNS"),
            ("DAMAGED_OR_WRONG_ITEM", "LINK_RETURNS"),
            ("PRODUCT_AND_TECHNICAL", "TROUBLESHOOT_DEVICE"),
            ("ACCOUNT_AND_SECURITY", "DM_WITH_ORDER_ID"),
        ]
        for intent, expected_act in tests:
            act = self.generator.select_act([], intent, False)
            assert act == expected_act, f"Failed for intent: {intent}"

    def test_offline_rewrite_templates(self):
        """Offline mode should use templates."""
        tests = [
            ("LINK_ORDERS", "amazon.com/orders"),
            ("LINK_RETURNS", "amazon.com/returns"),
            ("DM_WITH_ORDER_ID", "twitter.com/messages/compose"),
        ]
        for act, expected_link in tests:
            reply = self.generator._offline_rewrite("test query", "INTENT", act, "")
            assert expected_link in reply

    def test_escalation_templates(self):
        """Escalation reasons should have specific templates."""
        tests = [
            ("ACCOUNT_SECURITY_OR_LOCKOUT", "twitter.com/messages/compose"),
            ("DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING", "twitter.com/messages/compose"),
            ("FINANCIAL_DISPUTE_OR_OVERDUE_REFUND", "twitter.com/messages/compose"),
            ("SEVERE_DAMAGE_OR_MISSING_CONTENTS", "twitter.com/messages/compose"),
            ("HIGH_HOSTILITY_OR_LEGAL_RISK", "twitter.com/messages/compose"),
        ]
        for reason, expected_link in tests:
            reply = self.generator._offline_rewrite("test", "INTENT", "DM_WITH_ORDER_ID", reason)
            assert expected_link in reply

    def test_generate_with_metadata(self):
        """generate_with_metadata should return reply and metadata."""
        exemplars = [{"resolution_act": "LINK_ORDERS", "reply_text": "Test reply"}]
        reply, metadata = self.generator.generate_with_metadata(
            "Where is my package?",
            "ORDER_TRACKING_AND_DELAY",
            exemplars,
            escalate_decision={"should_escalate": False},
        )
        assert isinstance(reply, str)
        assert len(reply) > 0
        assert "resolution_act" in metadata
        assert "backend" in metadata


class TestBaselineGenerators:
    """Tests for baseline generators."""

    def test_trivial_reply_baseline(self):
        """TrivialReplyBaseline should return canned response."""
        gen = TrivialReplyBaseline()
        reply = gen.generate()
        assert "Amazon Support" in reply

    def test_retrieval_replay_baseline(self):
        """RetrievalReplayBaseline should return top exemplar."""
        gen = RetrievalReplayBaseline()
        exemplars = [{"reply_text": "Historical reply"}]
        reply = gen.generate(exemplars=exemplars)
        assert reply == "Historical reply"

    def test_baseline_retrieval_only(self):
        """BaselineRetrievalOnlyGenerator should clean agent signatures."""
        gen = BaselineRetrievalOnlyGenerator()
        exemplars = [{"reply_text": "^AB Test reply"}]
        reply = gen.generate(exemplars, "@customer")
        assert reply == "Test reply"
        assert "^AB" not in reply

    def test_baseline_generative(self):
        """BaselineGenerativeGenerator should produce template response."""
        gen = BaselineGenerativeGenerator()
        reply = gen.generate("Test query", "ORDER_TRACKING_AND_DELAY", "@customer")
        assert "order tracking and delay" in reply.lower()
        assert "@customer" in reply


class TestActRealization:
    """Tests for act realization validation."""

    def setup_method(self):
        self.generator = ProductionGroundedGenerator(mode="offline")

    def test_link_orders_realization(self):
        """LINK_ORDERS should mention amazon.com/orders or 'your orders'."""
        reply = self.generator._offline_rewrite("test", "INTENT", "LINK_ORDERS", "")
        assert self.generator._realizes_act(reply, "LINK_ORDERS") is True

    def test_link_returns_realization(self):
        """LINK_RETURNS should mention amazon.com/returns or return/replacement."""
        reply = self.generator._offline_rewrite("test", "INTENT", "LINK_RETURNS", "")
        assert self.generator._realizes_act(reply, "LINK_RETURNS") is True

    def test_dm_with_order_id_realization(self):
        """DM_WITH_ORDER_ID should mention DM/direct message."""
        reply = self.generator._offline_rewrite("test", "INTENT", "DM_WITH_ORDER_ID", "")
        assert self.generator._realizes_act(reply, "DM_WITH_ORDER_ID") is True

    def test_troubleshoot_device_realization(self):
        """TROUBLESHOOT_DEVICE should mention troubleshooting steps."""
        reply = self.generator._offline_rewrite("test", "INTENT", "TROUBLESHOOT_DEVICE", "")
        assert self.generator._realizes_act(reply, "TROUBLESHOOT_DEVICE") is True

    def test_general_support_realization(self):
        """GENERAL_SUPPORT should ask for more details."""
        reply = self.generator._offline_rewrite("test", "INTENT", "GENERAL_SUPPORT", "")
        assert self.generator._realizes_act(reply, "GENERAL_SUPPORT") is True


class TestGenerationMode:
    """Tests for different generation modes."""

    def test_offline_mode(self):
        """Offline mode should not require model."""
        gen = ProductionGroundedGenerator(mode="offline")
        assert gen.local_model is None

    @patch("src.generator.LocalInstructionModel")
    def test_local_mode_initialization(self, mock_llm):
        """Local mode should initialize model on first use."""
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance
        mock_instance.generate.return_value = "Generated reply"
        
        gen = ProductionGroundedGenerator(mode="local")
        assert gen.local_model is None  # Not initialized yet
        
        exemplars = [{"resolution_act": "LINK_ORDERS"}]
        gen.generate("test", "INTENT", exemplars)
        assert gen.local_model is not None
        mock_llm.assert_called_once()


class TestEdgeCases:
    """Edge case tests for generation."""

    def setup_method(self):
        self.generator = ProductionGroundedGenerator(mode="offline")

    def test_empty_exemplars(self):
        """Should handle empty exemplars."""
        reply = self.generator.generate("test", "INTENT", [])
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_unknown_intent(self):
        """Should handle unknown intent gracefully."""
        reply = self.generator.generate("test", "UNKNOWN_INTENT", [])
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_all_acts_have_templates(self):
        """All resolution acts should have offline templates."""
        acts = [
            "ASK_TRACKING_STATUS", "ASK_CARRIER", "DM_WITH_ORDER_ID",
            "LINK_ORDERS", "LINK_RETURNS", "TROUBLESHOOT_DEVICE",
            "APOLOGIZE_WAIT", "GENERAL_SUPPORT"
        ]
        for act in acts:
            reply = self.generator._offline_rewrite("test", "INTENT", act, "")
            assert isinstance(reply, str)
            assert len(reply) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])