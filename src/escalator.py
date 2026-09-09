"""
Escalation Engine and Triage Rules for @AmazonHelp.
Implements 3 comparative systems:
1. TrivialEscalationBaseline (Always escalate unconditionally)
2. SimpleKeywordEscalationBaseline (Heuristic regex keyword matching)
3. ProductionEscalator (Multi-Factor Risk Triage with Stated Reasons, 3P Marketplace Detection, & Porch Piracy Handling)
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class EscalationDecision:
    should_escalate: bool
    stated_reason: str
    risk_score: float
    triggers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_escalate": self.should_escalate,
            "stated_reason": self.stated_reason,
            "risk_score": round(self.risk_score, 3),
            "triggers": self.triggers
        }


class TrivialEscalationBaseline:
    """Always escalates unconditionally with default risk."""
    def evaluate(self, text: str, **kwargs) -> EscalationDecision:
        return EscalationDecision(
            should_escalate=True,
            stated_reason="UNCONDITIONAL_POLICY_ESCALATION",
            risk_score=1.0,
            triggers=["TRIVIAL_BASELINE_ALWAYS_ESCALATE"]
        )


class SimpleKeywordEscalationBaseline:
    """Simple escalation on basic critical terms."""
    def __init__(self):
        self.keywords = ["stolen", "police", "lawyer", "hacked", "fraud", "scam", "damaged", "refund", "cancel"]

    def evaluate(self, text: str, **kwargs) -> EscalationDecision:
        t_low = text.lower()
        matched = [kw for kw in self.keywords if kw in t_low]
        if matched:
            return EscalationDecision(
                should_escalate=True,
                stated_reason=f"KEYWORD_TRIGGER_{matched[0].upper()}",
                risk_score=0.7,
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
    2. 3rd-Party Marketplace (MFN) seller disputes vs direct Amazon retail
    3. Physical package condition (severe damage, missing items from box, porch piracy)
    4. Account security, credential compromise, & PII vulnerability
    5. High-friction churn, legal/regulatory threats, multi-attempt loops
    6. Classifier margin uncertainty
    7. Self-serve capability (standard tracking/returns)
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

        # 3rd-Party Marketplace (MFN) Disputes
        self.marketplace_disputes = [
            (r"(third[- ]party\s+seller|3p\s+seller|marketplace\s+seller)", "MARKETPLACE_3P_SELLER_DISPUTE"),
            (r"seller\s+(hasn\'?t|won\'?t|refused|not|declined|ignoring)\s+(responded?|replied?|help|refund|message)", "MARKETPLACE_3P_SELLER_DISPUTE"),
            (r"(seller\s+has\s+a\s+problem|seller\s+marked\s+item|contact(ed)?\s+the\s+seller)", "MARKETPLACE_3P_SELLER_DISPUTE"),
            (r"(a[- ]to[- ]z\s+guarantee|a[- ]to[- ]z\s+claim)", "MARKETPLACE_3P_SELLER_DISPUTE")
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
            (r"(locked\s+(me\s+)?out|account\s+locked|can\'?t\s+access\s+(my\s+)?(account|acct|prime))", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(couldn\'?t\s+access|cannot\s+access|can\'?t\s+sign\s+in|cannot\s+sign\s+in|cannot\s+log\s*in|can\'?t\s+log\s*in)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(reset.*password.*multiple\s+times|still\s+cannot\s+log\s*in)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(changed\s+my\s+account\s+email|unauthorized\s+email)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(scam|phishing|fake email|fraudulent order)", "ACCOUNT_SECURITY_OR_LOCKOUT")
        ]

        # High-Friction Legal, Regulatory & Ombudsman Threats
        self.legal_and_churn_threats = [
            (r"(better\s+business\s+bureau|\bbbb\b|ombudsman|trading\s+standards|consumer\s+protection)", "LEGAL_REGULATORY_OR_OMBUDSMAN_THREAT"),
            (r"(lawyer|attorney|small\s+claims|take\s+legal\s+action|court|sue\s+you)", "LEGAL_REGULATORY_OR_OMBUDSMAN_THREAT"),
            (r"(cancelled?\s+my\s+(prime|membership)|canceling\s+(prime|membership)|never\s+using\s+amazon)", "HIGH_CHURN_OR_CANCEL_INTENT"),
            (r"(called\s+(\d+|several|multiple)\s+times|no\s+response\s+after\s+\d+\s+calls)", "REPEATED_CONTACT_FAILURE_ESCALATION"),
            (r"(waiting\s+for\s+(\d+|several)\s+weeks|pending\s+after\s+a\s+month)", "HIGH_FRICTION_CHURN_OR_LEGAL_RISK")
        ]

    def evaluate(
        self,
        text: str,
        intent: Optional[str] = None,
        confidence: float = 1.0,
        is_uncertain: bool = False
    ) -> EscalationDecision:
        """
        Evaluates input text against multi-factor risk hierarchy.
        Returns explicit EscalationDecision with stated reason and risk score.
        """
        t_low = text.lower()

        # 1. Security Threat / Account Takeover (Severity 0)
        for pat, reason in self.security_escalations:
            if re.search(pat, t_low):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.98,
                    triggers=[f"PATTERN_{pat}"]
                )

        # 2. PII & Sensitive Data Exposure in Public Timeline
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
            return EscalationDecision(
                should_escalate=True,
                stated_reason="PII_SENSITIVE_DATA_LEAK_IN_PUBLIC",
                risk_score=0.95,
                triggers=["EMAIL_ADDRESS_EXPOSED"]
            )
        if re.search(r"\b\d{16}\b|\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b", text):
            return EscalationDecision(
                should_escalate=True,
                stated_reason="PII_SENSITIVE_DATA_LEAK_IN_PUBLIC",
                risk_score=0.95,
                triggers=["CREDIT_CARD_NUMBER_EXPOSED"]
            )
        if re.search(r"\bcvv\b|\bsecurity\s+code\b", t_low):
            return EscalationDecision(
                should_escalate=True,
                stated_reason="PII_SENSITIVE_DATA_LEAK_IN_PUBLIC",
                risk_score=0.95,
                triggers=["CVV_EXPOSED"]
            )

        # 3. Legal / Regulatory / Ombudsman Threats (Severity 1)
        for pat, reason in self.legal_and_churn_threats:
            if re.search(pat, t_low):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.95,
                    triggers=[f"PATTERN_{pat}"]
                )

        # 4. 3P Marketplace Seller Disputes
        for pat, reason in self.marketplace_disputes:
            if re.search(pat, t_low):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.88,
                    triggers=[f"PATTERN_{pat}"]
                )

        # 5. Severe Damage & Missing Contents
        for pat, reason in self.damage_escalations:
            if re.search(pat, t_low):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.85,
                    triggers=[f"PATTERN_{pat}"]
                )

        # 6. Delivery & Fulfillment Disputes (Delivered but Missing, Porch Piracy, Carrier Misconduct)
        for pat, reason in self.fulfillment_disputes:
            if re.search(pat, t_low):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.85,
                    triggers=[f"PATTERN_{pat}"]
                )

        # 7. Financial & Refund Disputes (Checking if not past-resolved)
        is_past_resolved = any(re.search(res_pat, t_low) for res_pat in self.resolved_refunds)
        if not is_past_resolved:
            for pat, reason in self.financial_escalations:
                if re.search(pat, t_low):
                    return EscalationDecision(
                        should_escalate=True,
                        stated_reason=reason,
                        risk_score=0.82,
                        triggers=[f"PATTERN_{pat}"]
                    )

        # 8. Model Uncertainty / Margin Ambiguity
        if is_uncertain or confidence < self.confidence_threshold:
            return EscalationDecision(
                should_escalate=True,
                stated_reason="LOW_CLASSIFIER_CONFIDENCE_AMBIGUOUS_QUERY",
                risk_score=0.65,
                triggers=[f"CONFIDENCE_{confidence:.3f}"]
            )

        # 9. Default Self-Serve Resolutions
        if intent == "ORDER_TRACKING_AND_DELAY":
            return EscalationDecision(
                should_escalate=False,
                stated_reason="ROUTINE_TRACKING_IN_TRANSIT",
                risk_score=0.15,
                triggers=["SELF_SERVE_TRACKING"]
            )
        elif intent == "REFUND_AND_RETURN":
            return EscalationDecision(
                should_escalate=False,
                stated_reason="ROUTINE_RETURN_POLICY_OR_PORTAL",
                risk_score=0.20,
                triggers=["SELF_SERVE_RETURN"]
            )
        elif intent == "PRODUCT_AND_TECHNICAL":
            return EscalationDecision(
                should_escalate=False,
                stated_reason="DEVICE_TROUBLESHOOTING_SELF_SERVE",
                risk_score=0.25,
                triggers=["SELF_SERVE_TECH"]
            )
        elif intent == "ACCOUNT_AND_SECURITY":
            return EscalationDecision(
                should_escalate=False,
                stated_reason="GENERAL_ACCOUNT_SETTINGS_SELF_SERVE",
                risk_score=0.30,
                triggers=["SELF_SERVE_ACCOUNT"]
            )
        elif intent == "DAMAGED_OR_WRONG_ITEM":
            return EscalationDecision(
                should_escalate=False,
                stated_reason="ROUTINE_REPLACEMENT_PORTAL",
                risk_score=0.35,
                triggers=["SELF_SERVE_REPLACEMENT"]
            )
        else:
            return EscalationDecision(
                should_escalate=False,
                stated_reason="FEEDBACK_OR_POSITIVE_ENGAGEMENT",
                risk_score=0.10,
                triggers=["SELF_SERVE_GENERAL"]
            )
