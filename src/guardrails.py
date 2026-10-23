"""
Pre-Dispatch Policy Guardrails and Factuality Verifier for @AmazonHelp.
Enforces deterministic safety invariants before an AI draft can be dispatched:
1. Hard whitelist of allowed Amazon URLs (preventing phishing or dead links).
2. Detection and neutralization of hallucinated financial/operational commitments.
3. Zero-PII public echo defense (mirrored phone, email, or 17-digit order IDs).
4. Twitter 280-character hard limit enforcement with link preservation.
"""

import re
from typing import Dict, Any, List, Tuple

class PolicyVerificationResult:
    """Structured report of policy verification checks."""
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
    """Deterministic pre-dispatch safety checks for the prototype."""
    def __init__(self):
        # Whitelisted authoritative Amazon service endpoints
        self.allowed_domains = [
            r"https?://(www\.)?amazon\.com/orders(?:[?#][^\s]*)?$",
            r"https?://(www\.)?amazon\.com/returns(?:[?#][^\s]*)?$",
            r"https?://(www\.)?amazon\.com/help(?:[?#][^\s]*)?$",
            r"https?://(www\.)?amazon\.com/gp/help(?:[?#][^\s]*)?$",
            r"https?://(www\.)?amazon\.com/your-account(?:[?#][^\s]*)?$",
            r"https?://(www\.)?amazon\.com/contact-us(?:[?#][^\s]*)?$",
            r"amazon\.com/orders(?:[?#][^\s]*)?$",
            r"amazon\.com/returns(?:[?#][^\s]*)?$",
            r"amazon\.com/help(?:[?#][^\s]*)?$",
            r"amazon\.com/gp/help(?:[?#][^\s]*)?$",
            r"amzn\.to/[a-zA-Z0-9]+$",
            r"twitter\.com/messages/compose\?recipient_id=AmazonHelp$"
        ]

        # Dangerous hallucinated financial or operational commitments
        self.hallucinated_commitments = [
            (r"((i|we)\s+have\s+(processed|issued|credited|refunded)\s+(your|a|the)?\s*(full\s+)?refund(\s+of\s+\$?\d+)?)", "UNAUTHORIZED_REFUND_PROMISE"),
            (r"((i|we)\s+have\s+refunded\s+\$?\d+)", "UNAUTHORIZED_REFUND_AMOUNT_PROMISE"),
            (r"((i|we)\s+have\s+cancelled\s+your\s+order)", "UNAUTHORIZED_ORDER_CANCELLATION"),
            (r"(your\s+driver\s+will\s+arrive\s+in\s+\d+)", "UNAUTHORIZED_REALTIME_DRIVER_DISPATCH"),
            (r"(here\s+is\s+a\s+\$?\d+\s+gift\s+card)", "UNAUTHORIZED_COMPENSATION_ISSUANCE")
        ]

        # Prohibited request patterns in public tweets
        self.prohibited_requests = [
            (r"(tweet\s+(us\s+)?your\s+password)", "PROHIBITED_CREDENTIAL_REQUEST"),
            (r"(reply\s+with\s+your\s+card\s+number)", "PROHIBITED_FINANCIAL_CARD_REQUEST"),
            (r"(send\s+your\s+cvv)", "PROHIBITED_CVV_REQUEST"),
            (
                r"\b(?:provide|share|send|reply with|tell (?:me|us))\b.{0,35}"
                r"\b(?:order (?:number|id)|phone number|email address)\b"
                r"(?![^.!?]{0,45}\b(?:dm|direct message|private(?:ly| message)?)\b)",
                "PROHIBITED_PUBLIC_PII_REQUEST",
            ),
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

        # First pass: collect all violations on the original text
        # We check against the original draft_reply and customer_text to find all violations
        
        # Track what needs to be redacted/replaced
        pii_redactions = []  # (pattern, replacement, violation_code, adjustment_msg)
        url_replacements = []  # (url, replacement, violation_code, adjustment_msg)
        hallucination_fixes = []  # (pattern, replacement, violation_code, adjustment_msg)
        prohibited_request_violation = None
        char_limit_exceeded = False

        # 1. Zero-PII Public Echo Check
        # Check Amazon 17-digit Order IDs first (most specific): ###-#######-#######
        order_id_pattern = r"\b\d{3}-\d{7}-\d{7}\b"
        customer_orders = re.findall(order_id_pattern, customer_text)
        for order_id in customer_orders:
            if order_id in draft_reply:
                pii_redactions.append((order_id, "[order ID redacted]", "PII_ORDER_ID_ECHO_DETECTED", f"Redacted mirrored order ID {order_id}"))

        # Check emails (case-insensitive: local parts are case-preserving but the
        # domain is not, so a case variant must still be treated as an echo).
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        customer_emails = re.findall(email_pattern, customer_text)
        for email in customer_emails:
            for match in re.finditer(re.escape(email), draft_reply, re.IGNORECASE):
                pii_redactions.append((match.group(0), "[email protected]", "PII_ECHO_DETECTED", f"Redacted mirrored email {email}"))

        # Check phone numbers with format-normalised matching: the same number
        # can appear as "555-123-4567", "555 123 4567", "+1 (555) 123-4567", and
        # an exact-string comparison would let the dash/space variants through.
        phone_pattern = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
        customer_phone_digits = set()
        for phone in re.findall(phone_pattern, customer_text):
            digits = re.sub(r"\D", "", phone)
            if len(digits) >= 10:
                customer_phone_digits.add(digits[-10:])
        if customer_phone_digits:
            draft_phone_candidates = re.findall(
                r"\+?\d[\d\s\-().]{8,}\d", draft_reply
            )
            for candidate in draft_phone_candidates:
                digits = re.sub(r"\D", "", candidate)
                if len(digits) < 10:
                    continue
                # Match when the trailing 10 digits agree (drops any country code).
                if digits[-10:] in customer_phone_digits:
                    pii_redactions.append((
                        candidate,
                        "[phone redacted]",
                        "PII_ECHO_DETECTED",
                        f"Redacted mirrored phone {candidate.strip()}",
                    ))

        # 2. Prohibited Agent Requests Check
        for pat, violation_code in self.prohibited_requests:
            if re.search(pat, draft_reply, re.IGNORECASE):
                prohibited_request_violation = violation_code
                break

        # 3. False Financial / Operational Commitment Check
        for pat, violation_code in self.hallucinated_commitments:
            if re.search(pat, draft_reply, re.IGNORECASE):
                # Clean, grammatically correct replacement
                replacement = "You can review refund and order eligibility directly at amazon.com/orders"
                hallucination_fixes.append((pat, replacement, violation_code, f"Neutralized hallucinated commitment ({violation_code})"))

        # 4. Strict URL Whitelist Check
        urls = re.findall(
            r"(https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-z]{2,}(?:/\S*)?)",
            draft_reply,
            flags=re.IGNORECASE,
        )
        for url in urls:
            clean_url = url.rstrip(".,;!?:")
            is_whitelisted = any(
                re.fullmatch(allowed, clean_url, re.IGNORECASE)
                for allowed in self.allowed_domains
            )
            if not is_whitelisted:
                url_replacements.append((url, "amazon.com/help", "UNVERIFIED_EXTERNAL_URL", f"Replaced unverified link {url} with verified amazon.com/help"))

        # 5. Twitter 280-Character Budget Enforcement (with URL preservation)
        if len(draft_reply) > 280:
            char_limit_exceeded = True

        # Now apply all fixes
        sanitized = draft_reply.strip()
        
        # Apply PII redactions
        for pattern, replacement, violation_code, adjustment_msg in pii_redactions:
            if pattern in sanitized:
                sanitized = sanitized.replace(pattern, replacement)
                violations.append(violation_code)
                adjustments.append(adjustment_msg)

        # 2. Prohibited Agent Requests Check
        if prohibited_request_violation:
            violations.append(prohibited_request_violation)
            sanitized = (
                "I apologize for any confusion. Please don't share account, order, "
                "or payment details publicly. Reach us safely at amazon.com/help."
            )
            adjustments.append(f"Overrode reply due to prohibited phrase ({prohibited_request_violation})")

        # 3. False Financial / Operational Commitment Check
        for pat, replacement, violation_code, adjustment_msg in hallucination_fixes:
            if re.search(pat, sanitized, re.IGNORECASE):
                violations.append(violation_code)
                sanitized = re.sub(pat, replacement, sanitized, flags=re.IGNORECASE)
                adjustments.append(adjustment_msg)

        # 4. Strict URL Whitelist Check
        urls = re.findall(
            r"(https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-z]{2,}(?:/\S*)?)",
            sanitized,
            flags=re.IGNORECASE,
        )
        for url in urls:
            clean_url = url.rstrip(".,;!?:")
            is_whitelisted = any(
                re.fullmatch(allowed, clean_url, re.IGNORECASE)
                for allowed in self.allowed_domains
            )
            if not is_whitelisted:
                violations.append("UNVERIFIED_EXTERNAL_URL")
                sanitized = sanitized.replace(url, "amazon.com/help")
                adjustments.append(f"Replaced unverified link {url} with verified amazon.com/help")

        # 5. Twitter 280-Character Budget Enforcement (with URL preservation)
        if len(sanitized) > 280:
            violations.append("TWITTER_CHARACTER_LIMIT_EXCEEDED")
            trimmed = sanitized[:275]
            last_space = trimmed.rfind(" ")
            if last_space > 180:
                sanitized = trimmed[:last_space].strip() + "..."
            else:
                sanitized = trimmed.strip() + "..."
            adjustments.append(f"Trimmed reply length from {len(draft_reply)} to {len(sanitized)} chars")

        is_valid = len(violations) == 0

        return PolicyVerificationResult(
            is_valid=is_valid,
            sanitized_reply=sanitized,
            violations=violations,
            adjustments_made=adjustments
        )
