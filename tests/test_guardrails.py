"""
Unit tests for pre-dispatch policy guardrails.
"""

import pytest
from src.guardrails import ProductionPolicyGuard, PolicyVerificationResult


class TestProductionPolicyGuard:
    """Tests for the ProductionPolicyGuard class."""

    def setup_method(self):
        self.guard = ProductionPolicyGuard()

    def test_valid_reply_passes(self):
        """Valid reply should pass all guardrails."""
        reply = "I'm sorry for the delay. Check your order status at amazon.com/orders"
        customer = "Where is my package?"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is True
        assert result.sanitized_reply == reply
        assert result.violations == []

    def test_email_redaction(self):
        """Customer email in reply should be redacted."""
        reply = "Contact us at john@example.com for help"
        customer = "My email is john@example.com"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "PII_ECHO_DETECTED" in result.violations
        assert "john@example.com" not in result.sanitized_reply
        assert "[email protected]" in result.sanitized_reply

    def test_phone_redaction(self):
        """Customer phone in reply should be redacted."""
        reply = "Call us at 555-123-4567"
        customer = "My number is 555-123-4567"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "PII_ECHO_DETECTED" in result.violations
        assert "555-123-4567" not in result.sanitized_reply
        assert "[phone redacted]" in result.sanitized_reply

    def test_order_id_redaction(self):
        """Customer order ID in reply should be redacted."""
        reply = "Your order 123-4567890-1234567 is delayed"
        customer = "Order 123-4567890-1234567 missing"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "PII_ORDER_ID_ECHO_DETECTED" in result.violations
        assert "123-4567890-1234567" not in result.sanitized_reply
        assert "[order ID redacted]" in result.sanitized_reply

    def test_prohibited_credential_request(self):
        """Requesting password should be blocked."""
        reply = "Please tweet your password"
        customer = "Need help with account"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "PROHIBITED_CREDENTIAL_REQUEST" in result.violations
        assert "don't share account" in result.sanitized_reply.lower()

    def test_prohibited_card_request(self):
        """Requesting card number should be blocked."""
        reply = "Reply with your card number"
        customer = "Payment issue"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "PROHIBITED_FINANCIAL_CARD_REQUEST" in result.violations

    def test_prohibited_cvv_request(self):
        """Requesting CVV should be blocked."""
        reply = "Send your CVV for verification"
        customer = "Payment issue"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "PROHIBITED_CVV_REQUEST" in result.violations

    def test_prohibited_public_pii_request(self):
        """Requesting PII publicly should be blocked."""
        reply = "Please provide your order number"
        customer = "Need help"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "PROHIBITED_PUBLIC_PII_REQUEST" in result.violations

    def test_allowed_pii_request_with_dm(self):
        """Requesting PII with DM mention should be allowed."""
        reply = "Please DM your order number"
        customer = "Need help"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is True

    def test_hallucinated_refund_promise(self):
        """Hallucinated refund promises should be neutralized."""
        reply = "We have processed your refund of $50"
        customer = "Where is my refund?"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "UNAUTHORIZED_REFUND_PROMISE" in result.violations
        assert "amazon.com/orders" in result.sanitized_reply

    def test_hallucinated_refund_amount(self):
        """Specific refund amount promises should be neutralized."""
        reply = "We have refunded $100 to your account"
        customer = "Where is my refund?"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "UNAUTHORIZED_REFUND_AMOUNT_PROMISE" in result.violations

    def test_hallucinated_order_cancellation(self):
        """Hallucinated order cancellation should be neutralized."""
        reply = "We have cancelled your order"
        customer = "Cancel my order"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "UNAUTHORIZED_ORDER_CANCELLATION" in result.violations

    def test_hallucinated_driver_dispatch(self):
        """Hallucinated real-time driver dispatch should be neutralized."""
        reply = "Your driver will arrive in 5 minutes"
        customer = "Where is my delivery?"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "UNAUTHORIZED_REALTIME_DRIVER_DISPATCH" in result.violations

    def test_hallucinated_compensation(self):
        """Hallucinated gift card/compensation should be neutralized."""
        reply = "Here is a $25 gift card for the trouble"
        customer = "Bad experience"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "UNAUTHORIZED_COMPENSATION_ISSUANCE" in result.violations

    def test_unverified_url_replacement(self):
        """Unverified URLs should be replaced."""
        reply = "Check example.com for info"
        customer = "Need help"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert "UNVERIFIED_EXTERNAL_URL" in result.violations
        assert "amazon.com/help" in result.sanitized_reply

    def test_allowed_urls_pass(self):
        """Whitelisted Amazon URLs should pass."""
        urls = [
            "amazon.com/orders",
            "https://amazon.com/returns",
            "https://www.amazon.com/help",
            "amazon.com/gp/help",
            "amzn.to/abc123",
            "twitter.com/messages/compose?recipient_id=AmazonHelp",
        ]
        for url in urls:
            reply = f"Check {url} for info"
            customer = "Need help"
            result = self.guard.verify_and_sanitize(reply, customer_text=customer)
            assert result.is_valid is True, f"Failed for URL: {url}"

    def test_character_limit_enforcement(self):
        """Replies over 280 chars should be truncated."""
        long_reply = "A" * 300
        customer = "Test"
        result = self.guard.verify_and_sanitize(long_reply, customer_text=customer)
        assert result.is_valid is False
        assert "TWITTER_CHARACTER_LIMIT_EXCEEDED" in result.violations
        assert len(result.sanitized_reply) <= 280

    def test_character_limit_preserves_urls(self):
        """Truncation should not cut URLs."""
        long_reply = "Check amazon.com/orders for your order status " + "A" * 250
        customer = "Test"
        result = self.guard.verify_and_sanitize(long_reply, customer_text=customer)
        assert len(result.sanitized_reply) <= 280
        assert "amazon.com/orders" in result.sanitized_reply

    def test_multiple_violations_handled(self):
        """Multiple violations should all be caught."""
        reply = "We refunded $50 to john@example.com check example.com"
        customer = "My email is john@example.com"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer)
        assert result.is_valid is False
        assert len(result.violations) >= 2

    def test_escalation_mode_affects_pii_check(self):
        """Escalation mode should affect PII handling."""
        reply = "DM us with your order details"
        customer = "Need help"
        result = self.guard.verify_and_sanitize(reply, customer_text=customer, should_escalate=True)
        # In escalation mode, DM request is expected
        assert result.is_valid is True


class TestPolicyVerificationResult:
    """Tests for PolicyVerificationResult dataclass."""

    def test_to_dict(self):
        """to_dict should return proper structure."""
        result = PolicyVerificationResult(
            is_valid=True,
            sanitized_reply="Test reply",
            violations=[],
            adjustments_made=[]
        )
        d = result.to_dict()
        assert d["is_valid"] is True
        assert d["sanitized_reply"] == "Test reply"
        assert d["violations"] == []
        assert d["adjustments_made"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])