"""
Escalation Logic Engine for @AmazonHelp.
Evaluates incoming tweets across risk dimensions:
1. Financial, fulfillment, and delivery disputes (porch piracy, marked delivered but missing)
2. 3rd-party marketplace (MFN) seller disputes vs Amazon direct fulfillment
3. Physical package condition (severe damage, driver misconduct)
4. Account security & PII exposure
5. Legal/regulatory threats & high-friction churn
6. Statistical conformal risk & model margin uncertainty
"""

import re
from typing import List, Dict, Any, Optional
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
            "risk_score": self.risk_score,
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
    - Pre-compiled regular expressions for sub-millisecond execution.
    - Delivery and fulfillment disputes (porch piracy, courier exceptions).
    - 3P Marketplace seller disputes vs Amazon Prime.
    - Account security, credential compromise, & PII exposure.
    - Conformal Risk Control threshold integration.
    """
    def __init__(self, confidence_threshold: float = 0.32):
        self.confidence_threshold = confidence_threshold

        # Pre-compile PII patterns
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.cc_pattern = re.compile(r"\b\d{16}\b|\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b")
        self.cvv_pattern = re.compile(r"\bcvv\b|\bsecurity\s+code\b", re.IGNORECASE)

        # Delivery & Fulfillment Disputes
        raw_fulfillment = [
            (r"(marked|says?|states?|shows?)\s+(it\s+was\s+)?(as\s+)?delivered.*(never\s+(got|arrived|received)|not\s+(here|received|delivered)|didn\'?t\s+(get|receive)|missing|nowhere|nothing|stolen|no\s+package|haven\'?t\s+got|without\s+actual)", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(item|package|order|delivery|tix)\s+(not\s+delivered|never\s+arrived|missing|hasn\'?t\s+arrived).*tracking\s+(shows?|says?|marked)", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(never\s+arrived|never\s+came|didn\'?t\s+arrive)", "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING"),
            (r"(porch\s+pirat|stolen\s+from|package\s+(was\s+)?stolen|stolen\s+off)", "THEFT_OR_PORCH_PIRACY_ESCALATION"),
            (r"(handed\s+to\s+resident|delivered\s+to\s+resident).*no\s+one\s+(was\s+)?home", "FALSE_DELIVERY_CONFIRMATION_DISPUTE"),
            (r"(courier|carrier|dsp|driver)\s+(lied|fake|forged|didn\'?t\s+knock)", "FALSE_DELIVERY_CONFIRMATION_DISPUTE"),
            (r"(wrong\s+address|delivered\s+to\s+wrong|neighbour|neighbor)", "MISDELIVERED_LOCATION_EXCEPTION"),
            (r"(stuck\s+in\s+transit|no\s+update.*(days|week)|tracking\s+hasn\'?t\s+updated)", "EXTENDED_TRANSIT_STALL"),
            (r"(driver|courier)\s+(threw|damaged|stole|forged|lied|refused|rude)", "CARRIER_MISCONDUCT_OR_TRANSIT_FAILURE"),
            (r"(thrown\s+over|hurled\s+over|tossed\s+over|dumped\s+over)", "CARRIER_MISCONDUCT_OR_TRANSIT_FAILURE")
        ]
        self.fulfillment_disputes = [(re.compile(p, re.IGNORECASE), r) for p, r in raw_fulfillment]

        # 3rd-Party Marketplace (MFN) Disputes
        raw_marketplace = [
            (r"(third[- ]party\s+seller|3p\s+seller|marketplace\s+seller)", "MARKETPLACE_3P_SELLER_DISPUTE"),
            (r"(seller\s+won\'?t\s+reply|seller\s+refuses|contacted\s+the\s+seller)", "MARKETPLACE_3P_SELLER_DISPUTE"),
            (r"(a[- ]to[- ]z\s+guarantee|a-z\s+claim)", "MARKETPLACE_3P_SELLER_DISPUTE")
        ]
        self.marketplace_disputes = [(re.compile(p, re.IGNORECASE), r) for p, r in raw_marketplace]

        # Severe Damage & Broken Goods
        raw_damage = [
            (r"(broken\s+into\s+pieces|smashed|shattered|completely\s+destroyed|crushed\s+box)", "SEVERE_DAMAGE_OR_DEFECTIVE_GOODS"),
            (r"(empty\s+box|missing\s+the\s+item\s+inside|box\s+was\s+open(ed)?)", "SEVERE_DAMAGE_OR_DEFECTIVE_GOODS"),
            (r"(leaking|liquid\s+everywhere|torn\s+open)", "SEVERE_DAMAGE_OR_DEFECTIVE_GOODS")
        ]
        self.damage_escalations = [(re.compile(p, re.IGNORECASE), r) for p, r in raw_damage]

        # Financial Disputes
        raw_financial = [
            (r"(charged\s+twice|double\s+charg(ed|ing)|unauthorized\s+charge)", "FINANCIAL_DISPUTE_OR_DOUBLE_CHARGE"),
            (r"(where\s+is\s+my\s+refund|haven\'?t\s+received\s+refund|refund\s+not\s+(credited|received))", "FINANCIAL_DISPUTE_OVERDUE_REFUND"),
            (r"(chargeback|dispute\s+with\s+(my\s+)?bank)", "FINANCIAL_DISPUTE_CHARGEBACK_OR_BANK")
        ]
        self.financial_escalations = [(re.compile(p, re.IGNORECASE), r) for p, r in raw_financial]

        raw_resolved = [
            r"refund\s+(was\s+)?(issued|processed|given|received)",
            r"got\s+(my\s+)?(refund|money\s+back)"
        ]
        self.resolved_refunds = [re.compile(p, re.IGNORECASE) for p in raw_resolved]

        # Account Security & Compromise
        raw_security = [
            (r"(hacked|unauthorized access|someone.*logged into|account.*taken over|stolen account)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(otp|verification code|password reset).*(not working|loop|locked out|can\'?t\s+sign\s+in)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(locked\s+(me\s+)?out|account\s+locked|can\'?t\s+access\s+(my\s+)?(account|acct|prime))", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(suspended\s+account|account\s+on\s+hold|closed\s+my\s+account)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(changed\s+my\s+account\s+email|unauthorized\s+email)", "ACCOUNT_SECURITY_OR_LOCKOUT"),
            (r"(scam|phishing|fake email|fraudulent order)", "ACCOUNT_SECURITY_OR_LOCKOUT")
        ]
        self.security_escalations = [(re.compile(p, re.IGNORECASE), r) for p, r in raw_security]

        # High-Friction Legal, Regulatory & Ombudsman Threats
        raw_legal = [
            (r"(better\s+business\s+bureau|\bbbb\b|ombudsman|trading\s+standards|consumer\s+protection)", "LEGAL_REGULATORY_OR_OMBUDSMAN_THREAT"),
            (r"(lawyer|attorney|small\s+claims|take\s+legal\s+action|court|sue\s+you)", "LEGAL_REGULATORY_OR_OMBUDSMAN_THREAT"),
            (r"(cancelled?\s+my\s+(prime|membership)|canceling\s+(prime|membership)|never\s+using\s+amazon)", "HIGH_CHURN_OR_CANCEL_INTENT"),
            (r"(called\s+(\d+|several|multiple)\s+times|no\s+response\s+after\s+\d+\s+calls)", "REPEATED_CONTACT_FAILURE_ESCALATION"),
            (r"(waiting\s+for\s+(\d+|several)\s+weeks|pending\s+after\s+a\s+month)", "HIGH_FRICTION_CHURN_OR_LEGAL_RISK")
        ]
        self.legal_and_churn_threats = [(re.compile(p, re.IGNORECASE), r) for p, r in raw_legal]

    def evaluate(
        self,
        text: str,
        intent: Optional[str] = None,
        confidence: float = 1.0,
        is_uncertain: bool = False,
        conformal_risk_threshold: Optional[float] = None
    ) -> EscalationDecision:
        """
        Evaluates input text against multi-factor risk hierarchy.
        Returns explicit EscalationDecision with stated reason and risk score.
        """
        effective_threshold = conformal_risk_threshold if conformal_risk_threshold is not None else self.confidence_threshold

        # 1. Security Threat / Account Takeover (Severity 0)
        for pat_re, reason in self.security_escalations:
            if pat_re.search(text):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.98,
                    triggers=[f"PATTERN_{pat_re.pattern}"]
                )

        # 2. PII & Sensitive Data Exposure in Public Timeline
        if self.email_pattern.search(text):
            return EscalationDecision(
                should_escalate=True,
                stated_reason="PII_SENSITIVE_DATA_LEAK_IN_PUBLIC",
                risk_score=0.95,
                triggers=["EMAIL_ADDRESS_EXPOSED"]
            )
        if self.cc_pattern.search(text):
            return EscalationDecision(
                should_escalate=True,
                stated_reason="PII_SENSITIVE_DATA_LEAK_IN_PUBLIC",
                risk_score=0.95,
                triggers=["CREDIT_CARD_NUMBER_EXPOSED"]
            )
        if self.cvv_pattern.search(text):
            return EscalationDecision(
                should_escalate=True,
                stated_reason="PII_SENSITIVE_DATA_LEAK_IN_PUBLIC",
                risk_score=0.95,
                triggers=["CVV_SECURITY_CODE_EXPOSED"]
            )

        # 3. Legal / Regulatory / Ombudsman Threats (Severity 1)
        for pat_re, reason in self.legal_and_churn_threats:
            if pat_re.search(text):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.95,
                    triggers=[f"PATTERN_{pat_re.pattern}"]
                )

        # 4. 3P Marketplace Seller Disputes
        for pat_re, reason in self.marketplace_disputes:
            if pat_re.search(text):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.88,
                    triggers=[f"PATTERN_{pat_re.pattern}"]
                )

        # 5. Severe Damage & Missing Contents
        for pat_re, reason in self.damage_escalations:
            if pat_re.search(text):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.85,
                    triggers=[f"PATTERN_{pat_re.pattern}"]
                )

        # 6. Delivery & Fulfillment Disputes
        for pat_re, reason in self.fulfillment_disputes:
            if pat_re.search(text):
                return EscalationDecision(
                    should_escalate=True,
                    stated_reason=reason,
                    risk_score=0.85,
                    triggers=[f"PATTERN_{pat_re.pattern}"]
                )

        # 7. Financial & Refund Disputes
        is_past_resolved = any(res_re.search(text) for res_re in self.resolved_refunds)
        if not is_past_resolved:
            for pat_re, reason in self.financial_escalations:
                if pat_re.search(text):
                    return EscalationDecision(
                        should_escalate=True,
                        stated_reason=reason,
                        risk_score=0.82,
                        triggers=[f"PATTERN_{pat_re.pattern}"]
                    )

        # 8. Model Uncertainty / Margin Ambiguity / CRC Threshold
        if is_uncertain or confidence < effective_threshold:
            return EscalationDecision(
                should_escalate=True,
                stated_reason="LOW_CLASSIFIER_CONFIDENCE_AMBIGUOUS_QUERY",
                risk_score=0.75,
                triggers=["CONFIDENCE_MARGIN_BREACH"]
            )

        # 9. Low-Risk Routine Self-Serve Resolution
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
                stated_reason="ROUTINE_RETURNS_SELF_SERVE",
                risk_score=0.20,
                triggers=["SELF_SERVE_RETURN"]
            )
        elif intent == "PRODUCT_AND_TECHNICAL":
            return EscalationDecision(
                should_escalate=False,
                stated_reason="ROUTINE_DEVICE_TROUBLESHOOTING",
                risk_score=0.25,
                triggers=["SELF_SERVE_TECH"]
            )
        elif intent == "ACCOUNT_AND_SECURITY":
            return EscalationDecision(
                should_escalate=False,
                stated_reason="ACCOUNT_SETTINGS_SELF_SERVE",
                risk_score=0.30,
                triggers=["SELF_SERVE_SECURITY"]
            )
        elif intent == "DAMAGED_OR_WRONG_ITEM":
            return EscalationDecision(
                should_escalate=False,
                stated_reason="ROUTINE_ITEM_REPLACEMENT_PORTAL",
                risk_score=0.30,
                triggers=["SELF_SERVE_REPLACEMENT"]
            )

        return EscalationDecision(
            should_escalate=False,
            stated_reason="GENERAL_FEEDBACK_OR_RESOLVED",
            risk_score=0.10,
            triggers=["FEEDBACK_ACKNOWLEDGEMENT"]
        )
