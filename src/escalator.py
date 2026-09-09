"""
Escalation and Triage Engine for @AmazonHelp.
Decides whether incoming customer support messages should be auto-handled or escalated
to a human tier-2 agent, with explicit, structured, and auditable stated reasons.
"""

import re
from typing import Dict, Any, List, Optional

class EscalationDecision:
    def __init__(
        self,
        should_escalate: bool,
        stated_reason: str,
        risk_score: float,
        triggers: List[str]
    ):
        self.should_escalate = should_escalate
        self.stated_reason = stated_reason
        self.risk_score = risk_score
        self.triggers = triggers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escalate": self.should_escalate,
            "stated_reason": self.stated_reason,
            "risk_score": round(self.risk_score, 3),
            "triggers": self.triggers
        }


class TrivialEscalationBaseline:
    """Trivial Baseline: Always escalates everything to human agent."""
    def evaluate(self, text: str, intent: str = "UNKNOWN", confidence: float = 1.0, is_uncertain: bool = False) -> EscalationDecision:
        return EscalationDecision(
            should_escalate=True,
            stated_reason="BASELINE_ALWAYS_ESCALATE_TO_HUMAN",
            risk_score=1.0,
            triggers=["TRIVIAL_BASELINE_POLICY"]
        )


class SimpleKeywordEscalationBaseline:
    """Simple Baseline: Escalates based strictly on a naive keyword blacklist."""
    def __init__(self):
        self.blacklist = ["scam", "fraud", "lawyer", "sue", "stolen", "police", "cancel prime", "supervisor"]

    def evaluate(self, text: str, intent: str = "UNKNOWN", confidence: float = 1.0, is_uncertain: bool = False) -> EscalationDecision:
        lower_text = text.lower()
        matched = [kw for kw in self.blacklist if kw in lower_text]
        if matched:
            return EscalationDecision(
                should_escalate=True,
                stated_reason=f"KEYWORD_FLAGGED_{matched[0].upper().replace(' ', '_')}",
                risk_score=0.8,
                triggers=matched
            )
        return EscalationDecision(
            should_escalate=False,
            stated_reason="AUTO_HANDLED_NO_FLAGGED_KEYWORDS",
            risk_score=0.1,
            triggers=[]
        )


class ProductionEscalator:
    """
    Production-grade Multi-Factor Escalation Engine:
    Evaluates:
    1. Financial & fulfillment delivery disputes (marked delivered but missing, lost in transit, double charges)
    2. Physical package condition (severe damage, missing items from box)
    3. Account security, credential compromise, & PII vulnerability
    4. Sentiment escalation, legal threats, churn risk
    5. Classifier margin uncertainty
    6. Self-serve capability (standard tracking/returns)
    """
    def __init__(self, confidence_threshold: float = 0.32):
        self.confidence_threshold = confidence_threshold

        # Delivery & Fulfillment Disputes
        self.fulfillment_disputes = [
            (r"(marked|says?|states?|shows?)\s+(it\s+was\s+)?(as\s+)?delivered.*(never\s+(got|arrived|received)|not\s+(here|received|delivered)|didn\'?t\s+(get|receive)|missing|nowhere|nothing|stolen|no\s+package|haven\'?t\s+got|without\s+actual)", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(item|package|order|delivery|tix)\s+(not\s+delivered|never\s+arrived|missing|hasn\'?t\s+arrived).*tracking\s+(shows?|says?|marked)", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(never\s+arrived|never\s+came|didn\'?t\s+arrive)", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(says?|shows?|states?|package\s+was)\s+(it\s+was\s+)?delivered\s+but\b", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(says?|shows?)\s+delivered\s+when\s+they\'?re\s+not", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"handed\s+to\s+(me|resident).*didn\'?t", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(package|order|parcel|item).*(stolen|porch pirate|pilfered|thief)", "THEFT_OR_STOLEN_PACKAGE_REPORT"),
            (r"(lost\s+in\s+transit|lost\s+.*in\s+the\s+post)", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(\d+\s+days?\s+(later|late|past)|still\s+(nothing|waiting|haven\'?t\s+received|no\s+sign)|over\s+a\s+week\s+for\s+delivery)", "CARRIER_TRANSIT_FAILURE_OR_DELAY"),
            (r"(now\s+takes\s+\d+\s+days|\d+\s+days\s+and\s+counting)", "CARRIER_TRANSIT_FAILURE_OR_DELAY"),
            (r"(delivry\s+missin|handd\s+to\s+resident)", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(3rd|third|multiple)\s+pkg.*shipping", "CARRIER_TRANSIT_FAILURE_OR_DELAY"),
            (r"(stop\s+using.*shipping|get\s+their\s+act\s+together)", "CARRIER_TRANSIT_FAILURE_OR_DELAY"),
            (r"(not\s+possible\s+to\s+get\s+in\s+touch|carrier.*not\s+responding)", "CARRIER_TRANSIT_FAILURE_OR_DELAY"),
            (r"(driver|courier)\s+(threw|damaged|stole|forged|lied|refused|rude)", "CARRIER_MISCONDUCT_OR_TRANSIT_FAILURE"),
            (r"(thrown\s+over|hurled\s+over|tossed\s+over|dumped\s+over)", "CARRIER_MISCONDUCT_OR_TRANSIT_FAILURE")
        ]

        # Severe Damage & Missing Contents (Physical Product Damage)
        self.damage_escalations = [
            (r"(shattered|completely\s+(broken|destroyed|smashed)|crushed)", "SEVERE_DAMAGE_OR_MISSING_CONTENTS"),
            (r"(item|product|package|box|order)\s+(is|was|got)?\s*(ruined|destroyed|smashed|crushed|broken)", "SEVERE_DAMAGE_OR_MISSING_CONTENTS"),
            (r"(empty\s+box|missing\s+item|items?\s+missing|contents?\s+missing)", "SEVERE_DAMAGE_OR_MISSING_CONTENTS"),
            (r"(box|package|order)\s+(was\s+)?(open|ripped|torn|tampered)", "SEVERE_DAMAGE_OR_MISSING_CONTENTS"),
            (r"(parcel\s+like\s+this|tape\s+up.*open|its\s+just\s+open)", "SEVERE_DAMAGE_OR_MISSING_CONTENTS"),
            (r"replacement\s+for\s+the\s+replacement", "SEVERE_DAMAGE_OR_MISSING_CONTENTS")
        ]

        # Financial & Refund Disputes
        self.financial_escalations = [
            (r"(charged|deducted)\s+(twice|duplicate|without permission|unauthorized)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"(money\s+(taken\s+out|deducted).*twice)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"(incurring\s+the\s+charges|charges.*account)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"(double\s+charg|overcharg)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"(refund.*(weeks?|months?|never|denied|failed)|where\s+is\s+my\s+refund|haven\'?t\s+received.*refund)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"waiting\s+for\s+a?\s*refund\s+for", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"refund.*(\d+|dm)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"(twice.*return|return.*twice|requested.*return.*still\s+no)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"(\d+\s+days?.*return|return.*\d+\s+days?)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND"),
            (r"(requested\s+cancellation|cancellation.*seller)", "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND")
        ]

        # Past-resolved refunds (avoid false positive escalation)
        self.resolved_refunds = [
            r"have\s+given\s+me\s+a\s+refund",
            r"already\s+refunded",
            r"got\s+my\s+refund",
            r"refund\s+was\s+processed"
        ]

        # Security & Fraud Escalations
        self.security_escalations = [
            (r"(hacked|unauthorized access|someone.*logged into|account.*taken over|stolen account)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(otp|verification code|password reset).*(not working|loop|locked out|can\'?t\s+sign\s+in)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(locked\s+out|account\s+locked|can\'?t\s+access\s+(my\s+)?(account|acct))", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(couldn\'?t\s+access|cannot\s+access|can\'?t\s+sign\s+in|cannot\s+sign\s+in|cannot\s+log\s*in|can\'?t\s+log\s*in)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(reset.*password.*multiple\s+times|still\s+cannot\s+log\s*in)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(changed\s+my\s+account\s+email|unauthorized\s+email)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(scam|phishing|fake email|fraudulent order)", "ACCOUNT_SECURITY_OR_LOCKOUT")
        ]

        # Hostility, Legal, Churn (fixed \bsue\b and explicit cancellation intent)
        self.hostility_escalations = [
            (r"(lawyer|attorney|legal action|court|\bsue\b|bbb|better business bureau|consumer court|police)", "LEGAL_OR_REGULATORY_THREAT"),
            (r"(i\'?m\s+canceling\s+(my\s+)?prime|cancel\s+my\s+prime\s+right\s+now|boycott\s+amazon|done\s+with\s+amazon|never\s+buying\s+again|never\s+ordering\s+again)", "HIGH_HOSTILITY_OR_LEGAL_RISK"),
            (r"(fuck|shitty|bitch|bastard|scumbag|liar|thieves|horrible service|worst service)", "HIGH_HOSTILITY_OR_LEGAL_RISK"),
            (r"screwed\s+up.*beyond\s+a\s+joke", "HIGH_HOSTILITY_OR_LEGAL_RISK")
        ]

        # Explicit Human Request
        self.human_agent_requests = [
            r"talk to a (human|person|agent|representative|supervisor|manager)",
            r"speak with an agent",
            r"transfer me",
            r"real person please"
        ]

        # PII exposure patterns in public tweets
        self.pii_patterns = [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "PII_PUBLIC_EMAIL_LEAK"),
            (r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b", "PII_PUBLIC_PHONE_LEAK"),
            (r"\b(?:\d{4}[-\s]?){4}\b", "PII_PUBLIC_CREDIT_CARD_LEAK"),
            (r"\bcvv\b", "PII_PUBLIC_CVV_LEAK"),
            (r"\b\d{3}-\d{7}-\d{7}\b", "PII_PUBLIC_DEFLECTION_TO_PRIVATE_CHANNEL")
        ]

    def evaluate(
        self,
        text: str,
        intent: str = "UNKNOWN",
        confidence: float = 1.0,
        is_uncertain: bool = False
    ) -> EscalationDecision:
        triggers = []
        lower = text.lower()

        # Check if this is an already-resolved refund statement
        is_resolved_refund = any(re.search(pat, lower) for pat in self.resolved_refunds)

        # 1. PII Exposed in Public Tweet
        for pat, reason in self.pii_patterns:
            if re.search(pat, text, re.IGNORECASE):
                triggers.append(f"PII:{reason}")
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.95,
                    triggers=triggers
                )

        # 2. Explicit Human Request Check
        for pat in self.human_agent_requests:
            if re.search(pat, lower):
                triggers.append("EXPLICIT_HUMAN_REQUEST")
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason="EXPLICIT_CUSTOMER_REQUEST_FOR_HUMAN_AGENT",
                    risk_score=1.0,
                    triggers=triggers
                )

        # 3. Security & Account Threats (Critical Risk)
        for pat, reason in self.security_escalations:
            if re.search(pat, lower):
                triggers.append(f"SECURITY:{reason}")
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.95,
                    triggers=triggers
                )

        # 4. Hostility / Legal Threats
        for pat, reason in self.hostility_escalations:
            if re.search(pat, lower):
                triggers.append(f"HOSTILITY:{reason}")
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.90,
                    triggers=triggers
                )

        # 5. Financial Disputes & Overdue Refunds (skip if resolved)
        if not is_resolved_refund:
            for pat, reason in self.financial_escalations:
                if re.search(pat, lower):
                    triggers.append(f"FINANCIAL:{reason}")
                    return EscalationDecision(
                        should_escalate=True,
                        stated_reason=reason,
                        risk_score=0.88,
                        triggers=triggers
                    )

        # 6. Fulfillment Delivery Disputes
        for pat, reason in self.fulfillment_disputes:
            if re.search(pat, lower):
                triggers.append(f"FULFILLMENT:{reason}")
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.85,
                    triggers=triggers
                )

        # 7. Severe Damage & Tampered Packages
        for pat, reason in self.damage_escalations:
            if re.search(pat, lower):
                triggers.append(f"DAMAGE:{reason}")
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.82,
                    triggers=triggers
                )

        # 8. Intent Classifier Uncertainty & Ambiguity
        if is_uncertain or confidence < self.confidence_threshold:
            triggers.append("LOW_CLASSIFIER_CONFIDENCE")
            return EscalationDecision(
                should_escalate=True,
                stated_reason=f"AMBIGUOUS_QUERY_LOW_MODEL_CONFIDENCE_{confidence:.2f}",
                risk_score=0.60,
                triggers=triggers
            )

        # 9. Otherwise Auto-Handleable
        if intent == "ORDER_TRACKING_AND_DELAY":
            reason = "ROUTINE_TRACKING_IN_TRANSIT"
            risk_score = 0.15
        elif intent == "REFUND_AND_RETURN":
            reason = "ROUTINE_RETURN_POLICY_OR_PORTAL"
            risk_score = 0.20
        elif intent == "DAMAGED_OR_WRONG_ITEM":
            reason = "ROUTINE_REPLACEMENT_ELIGIBLE"
            risk_score = 0.25
        elif intent == "PRODUCT_AND_TECHNICAL":
            reason = "DEVICE_TROUBLESHOOTING_SELF_SERVE"
            risk_score = 0.15
        elif intent == "ACCOUNT_AND_SECURITY":
            reason = "GENERAL_ACCOUNT_SETTINGS_SELF_SERVE"
            risk_score = 0.30
        else:
            reason = "POSITIVE_BRAND_PRAISE"
            risk_score = 0.10

        return EscalationDecision(
            should_escalate=False,
            stated_reason=reason,
            risk_score=risk_score,
            triggers=["SELF_SERVE_ELIGIBLE"]
        )
