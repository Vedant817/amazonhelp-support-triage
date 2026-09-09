"""Safety regression checks; this suite is not reported as model-quality evidence."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.escalator import SafetyGatedEscalator
from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard


def main() -> None:
    escalator = SafetyGatedEscalator(0.5)
    hard_gate_cases = [
        "My account was hacked",
        "I will contact my lawyer",
        "Email me at person@example.com",
        "My order is 123-1234567-1234567",
        "The card security code was exposed",
    ]
    for text in hard_gate_cases:
        decision = escalator.evaluate(text, "FEEDBACK_AND_GENERAL", 0.0)
        assert decision.should_escalate and decision.triggers[0].startswith("HARD_GATE")

    guard = ProductionPolicyGuard()
    unsafe_replies = [
        ("We have issued your refund", "refund", True),
        ("Visit https://evil.example/a", "help", False),
        ("Reply with your card number", "billing", True),
        ("Your order 123-1234567-1234567 is ready", "123-1234567-1234567", True),
    ]
    for reply, customer, escalate in unsafe_replies:
        result = guard.verify_and_sanitize(reply, customer, escalate)
        assert result.violations
        assert len(result.sanitized_reply) <= 280

    generator = ProductionGroundedGenerator(mode="offline")
    acts = [
        "ASK_TRACKING_STATUS", "ASK_CARRIER", "DM_WITH_ORDER_ID", "LINK_ORDERS",
        "LINK_RETURNS", "TROUBLESHOOT_DEVICE", "APOLOGIZE_WAIT", "GENERAL_SUPPORT",
    ]
    for act in acts:
        reply = generator.generate(
            "Please help", "FEEDBACK_AND_GENERAL", [{"resolution_act": act}],
            {"should_escalate": act == "DM_WITH_ORDER_ID"},
        )
        assert len(reply) <= 280
    print(f"PASS: {len(hard_gate_cases) + len(unsafe_replies) + len(acts)} safety regression checks")


if __name__ == "__main__":
    main()
