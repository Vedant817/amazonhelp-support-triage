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
        self._delivery_dispute = re.compile(
            r"\b(?:marked|says?|shows?)\s+delivered\b.*\b(?:missing|never (?:arrived|received)|not (?:here|received)|wrong (?:house|address)|neighbor)\b|"
            r"\bdelivered\b.*\b(?:wrong (?:house|address)|neighbor refuses|never (?:arrived|received))\b|"
            r"\bfailed delivery\b.*\b(?:twice|three times|multiple|again|investigat)", re.I,
        )
        self._severe_damage = re.compile(
            r"\b(?:cracked|shattered|smashed|fire|smoke|sparks?|hazardous|empty (?:box|package)|missing contents)\b",
            re.I,
        )
        self._financial_dispute = re.compile(
            r"\b(?:charged twice|double charged|unauthorized charge|refund (?:is )?(?:missing|overdue|never arrived)|still no refund)\b",
            re.I,
        )
        self._prompt_attack = re.compile(
            r"\b(?:ignore\b.{0,30}\b(?:rules|instructions)|reveal (?:the )?(?:hidden|system) prompt|override (?:the )?(?:policy|rules))\b",
            re.I,
        )
        self._routine_self_service = re.compile(
            r"\b(?:where|how) (?:can|do) i (?:see|find|check|track|start|make|initiate)\b.*\b(?:tracking|track|return)\b|"
            r"\b(?:latest tracking update|standard return|return (?:policy|portal)|troubleshooting steps)\b|"
            r"\bonly need to know where (?:the |my )?(?:package|order) is\b|"
            r"\bsent (?:me )?size\b.*\binstead of\b",
            re.I,
        )

    def hard_gate(self, text: str) -> Optional[EscalationDecision]:
        if self._payment.search(text) or self._email.search(text) or self._order_id.search(text):
            return EscalationDecision(True, "PII_PUBLIC_DEFLECTION_TO_PRIVATE_CHANNEL", 1.0, ["HARD_GATE_PUBLIC_PII"])
        if self._takeover.search(text):
            return EscalationDecision(True, "ACCOUNT_SECURITY_OR_LOCKOUT", 1.0, ["HARD_GATE_ACCOUNT_TAKEOVER"])
        if self._legal.search(text):
            return EscalationDecision(True, "HIGH_HOSTILITY_OR_LEGAL_RISK", 1.0, ["HARD_GATE_LEGAL_THREAT"])
        if self._prompt_attack.search(text):
            return EscalationDecision(True, "HIGH_HOSTILITY_OR_LEGAL_RISK", 1.0, ["HARD_GATE_PROMPT_MANIPULATION"])
        if self._delivery_dispute.search(text):
            return EscalationDecision(True, "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING", 1.0, ["HARD_GATE_DELIVERY_DISPUTE"])
        if self._severe_damage.search(text):
            return EscalationDecision(True, "SEVERE_DAMAGE_OR_MISSING_CONTENTS", 1.0, ["HARD_GATE_SEVERE_DAMAGE"])
        if self._financial_dispute.search(text):
            return EscalationDecision(True, "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND", 1.0, ["HARD_GATE_FINANCIAL_DISPUTE"])
        return None

    def evaluate(
        self,
        text: str,
        intent: str,
        learned_risk_score: float,
        is_uncertain: bool = False,
        learned_threshold: Optional[float] = None,
        high_precision_threshold: Optional[float] = None,
        resolution_act: Optional[str] = None,
    ) -> EscalationDecision:
        hard = self.hard_gate(text)
        if hard:
            return hard
        threshold = self.learned_threshold if learned_threshold is None else learned_threshold
        if is_uncertain:
            return EscalationDecision(
                True,
                ESCALATE_REASON_BY_INTENT.get(intent, "HIGH_HOSTILITY_OR_LEGAL_RISK"),
                float(max(learned_risk_score, threshold)),
                ["INTENT_ABSTENTION"],
            )
        if high_precision_threshold is not None and learned_risk_score >= high_precision_threshold:
            return EscalationDecision(
                True,
                ESCALATE_REASON_BY_INTENT.get(intent, "HIGH_HOSTILITY_OR_LEGAL_RISK"),
                float(learned_risk_score),
                ["HIGH_CONFIDENCE_ESCALATION_HEAD"],
            )
        self_service_acts = {
            "ASK_TRACKING_STATUS", "ASK_CARRIER", "LINK_ORDERS", "LINK_RETURNS",
            "TROUBLESHOOT_DEVICE", "APOLOGIZE_WAIT",
        }
        if resolution_act in self_service_acts and self._routine_self_service.search(text):
            return EscalationDecision(
                False,
                AUTO_REASON_BY_INTENT.get(intent, "POSITIVE_BRAND_PRAISE"),
                float(learned_risk_score),
                ["RETRIEVAL_SUPPORTED_SELF_SERVICE"],
            )
        if learned_risk_score >= threshold:
            return EscalationDecision(
                True,
                ESCALATE_REASON_BY_INTENT.get(intent, "HIGH_HOSTILITY_OR_LEGAL_RISK"),
                float(learned_risk_score),
                ["LEARNED_ESCALATION_HEAD"],
            )
        return EscalationDecision(
            False,
            AUTO_REASON_BY_INTENT.get(intent, "POSITIVE_BRAND_PRAISE"),
            float(learned_risk_score),
            ["LEARNED_AUTO_HANDLE_HEAD"],
        )
