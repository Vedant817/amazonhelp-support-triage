"""Escalation baselines plus non-vetoable safety gates for the learned head."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EscalationDecision:
    should_escalate: bool
    stated_reason: str
    risk_score: float
    triggers: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "should_escalate": self.should_escalate,
            "stated_reason": self.stated_reason,
            "risk_score": self.risk_score,
            "triggers": self.triggers,
        }


class TrivialEscalationBaseline:
    def evaluate(self, text: str, **kwargs) -> EscalationDecision:
        return EscalationDecision(True, "UNCONDITIONAL_POLICY_ESCALATION", 1.0, ["ALWAYS_ESCALATE"])


class SimpleKeywordEscalationBaseline:
    def __init__(self):
        self.keywords = ["stolen", "police", "lawyer", "hacked", "fraud", "scam", "damaged", "refund", "cancel"]

    def evaluate(self, text: str, **kwargs) -> EscalationDecision:
        matched = [word for word in self.keywords if word in text.lower()]
        return EscalationDecision(
            bool(matched),
            f"KEYWORD_TRIGGER_{matched[0].upper()}" if matched else "AUTO_HANDLED_NO_FLAGGED_KEYWORDS",
            0.7 if matched else 0.1,
            matched,
        )


ESCALATE_REASON_BY_INTENT = {
    "ORDER_TRACKING_AND_DELAY": "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING",
    "REFUND_AND_RETURN": "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND",
    "DAMAGED_OR_WRONG_ITEM": "SEVERE_DAMAGE_OR_MISSING_CONTENTS",
    "ACCOUNT_AND_SECURITY": "ACCOUNT_SECURITY_OR_LOCKOUT",
    "PRODUCT_AND_TECHNICAL": "TECHNICAL_HARDWARE_FAILURE",
    "FEEDBACK_AND_GENERAL": "HIGH_HOSTILITY_OR_LEGAL_RISK",
}

AUTO_REASON_BY_INTENT = {
    "ORDER_TRACKING_AND_DELAY": "ROUTINE_TRACKING_IN_TRANSIT",
    "REFUND_AND_RETURN": "ROUTINE_RETURN_POLICY_OR_PORTAL",
    "DAMAGED_OR_WRONG_ITEM": "ROUTINE_REPLACEMENT_ELIGIBLE",
    "ACCOUNT_AND_SECURITY": "GENERAL_ACCOUNT_SETTINGS_SELF_SERVE",
    "PRODUCT_AND_TECHNICAL": "DEVICE_TROUBLESHOOTING_SELF_SERVE",
    "FEEDBACK_AND_GENERAL": "POSITIVE_BRAND_PRAISE",
}


class SafetyGatedEscalator:
    """Public PII, takeover, and legal gates run before and cannot be vetoed by the model."""

    def __init__(self, learned_threshold: float = 0.5):
        self.learned_threshold = learned_threshold
        self._email = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        self._order_id = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
        self._payment = re.compile(r"\b(?:cvv|security code|card number|credit card|\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4})\b", re.I)
        self._takeover = re.compile(r"\b(?:hacked|account takeover|taken over|unauthorized access|someone .* logged in|stolen account|phishing)\b", re.I)
        self._legal = re.compile(r"\b(?:lawyer|attorney|sue|legal action|small claims|ombudsman|trading standards|consumer protection|bbb)\b", re.I)

    def hard_gate(self, text: str) -> Optional[EscalationDecision]:
        if self._payment.search(text) or self._email.search(text) or self._order_id.search(text):
            return EscalationDecision(True, "PII_PUBLIC_DEFLECTION_TO_PRIVATE_CHANNEL", 1.0, ["HARD_GATE_PUBLIC_PII"])
        if self._takeover.search(text):
            return EscalationDecision(True, "ACCOUNT_SECURITY_OR_LOCKOUT", 1.0, ["HARD_GATE_ACCOUNT_TAKEOVER"])
        if self._legal.search(text):
            return EscalationDecision(True, "HIGH_HOSTILITY_OR_LEGAL_RISK", 1.0, ["HARD_GATE_LEGAL_THREAT"])
        return None

    def evaluate(
        self,
        text: str,
        intent: str,
        learned_risk_score: float,
        is_uncertain: bool = False,
        learned_threshold: Optional[float] = None,
    ) -> EscalationDecision:
        hard = self.hard_gate(text)
        if hard:
            return hard
        threshold = self.learned_threshold if learned_threshold is None else learned_threshold
        if is_uncertain or learned_risk_score >= threshold:
            return EscalationDecision(
                True,
                ESCALATE_REASON_BY_INTENT.get(intent, "HIGH_HOSTILITY_OR_LEGAL_RISK"),
                float(max(learned_risk_score, threshold if is_uncertain else learned_risk_score)),
                ["INTENT_ABSTENTION" if is_uncertain else "LEARNED_ESCALATION_HEAD"],
            )
        return EscalationDecision(
            False,
            AUTO_REASON_BY_INTENT.get(intent, "POSITIVE_BRAND_PRAISE"),
            float(learned_risk_score),
            ["LEARNED_AUTO_HANDLE_HEAD"],
        )
