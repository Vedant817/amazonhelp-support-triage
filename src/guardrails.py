"""
Grounded Policy Guard & Safety Verification Engine for @AmazonHelp.
Enforces pre-dispatch deterministic guardrails:
1. Strict Whitelist URL Verification (blocks hallucinated links and phishing redirects)
2. False Financial Commitment Interception (prevents hallucinating refunds or autonomous credits)
3. Zero-PII Public Echo Invariant (ensures customer PII is never mirrored back publicly)
4. Twitter 280-Character Hard Constraint Validation
5. Secure Channel Redirection for Escalated Tickets
"""

import re
from typing import Dict, Any, List, Tuple

class PolicyVerificationResult:
    def __init__(
        self,
        is_valid: bool,
        sanitized_reply: str,
        violations: List[str],
        adjustments_made: List[str]
    ):
        self.is_valid = is_valid
        self.sanitized_reply = sanitized_reply
        self.violations = violations
        self.adjustments_made = adjustments_made

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "sanitized_reply": self.sanitized_reply,
            "violations": self.violations,
            "adjustments_made": self.adjustments_made
        }


class ProductionPolicyGuard:
    """
    Enterprise-grade pre-dispatch safety and grounding guardrail for @AmazonHelp.
    """
    def __init__(self):
        # Whitelisted authoritative Amazon service endpoints
        self.allowed_domains = [
            r"https?://(www\.)?amazon\.com/orders\b",
            r"https?://(www\.)?amazon\.com/returns\b",
            r"https?://(www\.)?amazon\.com/help\b",
            r"https?://(www\.)?amazon\.com/gp/help\b",
            r"https?://(www\.)?amazon\.com/your-account\b",
            r"https?://(www\.)?amazon\.com/contact-us\b",
            r"amazon\.com/orders\b",
            r"amazon\.com/returns\b",
            r"amazon\.com/help\b",
            r"amzn\.to/[a-zA-Z0-9]+\b"
        ]

        # Dangerous hallucinated financial or operational commitments
        self.hallucinated_commitments = [
            (r"(i\s+have\s+(processed|issued|credited)\s+(your|a)\s+refund)", "UNAUTHORIZED_REFUND_PROMISE"),
            (r"(we\s+have\s+refunded\s+\$?\d+)", "UNAUTHORIZED_REFUND_AMOUNT_PROMISE"),
            (r"(i\s+have\s+cancelled\s+your\s+order)", "UNAUTHORIZED_ORDER_CANCELLATION"),
            (r"(your\s+driver\s+will\s+arrive\s+in\s+\d+)", "UNAUTHORIZED_REALTIME_DRIVER_DISPATCH"),
            (r"(here\s+is\s+a\s+\$?\d+\s+gift\s+card)", "UNAUTHORIZED_COMPENSATION_ISSUANCE")
        ]

        # Prohibited request patterns in public tweets
        self.prohibited_requests = [
            (r"(tweet\s+(us\s+)?your\s+password)", "PROHIBITED_CREDENTIAL_REQUEST"),
            (r"(reply\s+with\s+your\s+card\s+number)", "PROHIBITED_FINANCIAL_CARD_REQUEST"),
            (r"(send\s+your\s+cvv)", "PROHIBITED_CVV_REQUEST")
        ]

    def verify_and_sanitize(
        self,
        draft_reply: str,
        customer_text: str = "",
        should_escalate: bool = False
    ) -> PolicyVerificationResult:
        """
        Runs exhaustive verification on a candidate draft reply before dispatch.
        Returns a sanitized, compliant response.
        """
        violations = []
        adjustments = []
        sanitized = draft_reply.strip()

        # 1. Zero-PII Public Echo Check
        # If customer provided email or phone, verify draft does not publicly repeat it
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        customer_emails = re.findall(email_pattern, customer_text)
        for email in customer_emails:
            if email in sanitized:
                sanitized = sanitized.replace(email, "[email protected]")
                violations.append("PII_ECHO_DETECTED")
                adjustments.append(f"Redacted mirrored email {email}")

        phone_pattern = r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b"
        customer_phones = re.findall(phone_pattern, customer_text)
        for phone in customer_phones:
            if phone in sanitized:
                sanitized = sanitized.replace(phone, "[phone redacted]")
                violations.append("PII_ECHO_DETECTED")
                adjustments.append(f"Redacted mirrored phone {phone}")

        # 2. Prohibited Agent Requests Check
        for pat, violation_code in self.prohibited_requests:
            if re.search(pat, sanitized, re.IGNORECASE):
                violations.append(violation_code)
                sanitized = "I apologize for any confusion. Please never share sensitive passwords or card numbers in public. Reach us safely at amazon.com/help."
                adjustments.append(f"Overrode reply due to prohibited phrase ({violation_code})")
                break

        # 3. False Financial / Operational Commitment Check
        for pat, violation_code in self.hallucinated_commitments:
            if re.search(pat, sanitized, re.IGNORECASE):
                violations.append(violation_code)
                sanitized = re.sub(pat, "you can review refund and order eligibility directly at amazon.com/orders", sanitized, flags=re.IGNORECASE)
                adjustments.append(f"Neutralized hallucinated commitment ({violation_code})")

        # 4. Strict URL Whitelist Check
        # Find all URLs in candidate reply
        urls = re.findall(r"(https?://\S+|www\.\S+|amazon\.com/\S+)", sanitized)
        for url in urls:
            is_whitelisted = any(re.search(allowed, url) for allowed in self.allowed_domains)
            if not is_whitelisted:
                violations.append("UNVERIFIED_EXTERNAL_URL")
                sanitized = sanitized.replace(url, "amazon.com/help")
                adjustments.append(f"Replaced unverified link {url} with verified amazon.com/help")

        # 5. Twitter 280-Character Budget Enforcement
        if len(sanitized) > 280:
            violations.append("TWITTER_CHARACTER_LIMIT_EXCEEDED")
            # Truncate gracefully to the nearest sentence or word before character 276
            trimmed = sanitized[:276]
            last_space = trimmed.rfind(" ")
            if last_space > 200:
                sanitized = trimmed[:last_space] + "..."
            else:
                sanitized = trimmed + "..."
            adjustments.append(f"Trimmed reply length from {len(draft_reply)} to {len(sanitized)} chars")

        is_valid = len(violations) == 0

        return PolicyVerificationResult(
            is_valid=is_valid,
            sanitized_reply=sanitized,
            violations=violations,
            adjustments_made=adjustments
        )
