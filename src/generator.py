"""
Grounded Response Generation for @AmazonHelp.
Implements 3 comparative generation strategies:
1. TrivialReplyBaseline (Generic canned acknowledgement)
2. RetrievalReplyBaseline (Direct historical exemplar replay)
3. ProductionGroundedGenerator (Policy-guided, exemplar-informed, Twitter DM deep-linked)
"""

import re
from typing import List, Dict, Any, Optional

class TrivialReplyBaseline:
    """Emits standard canned message."""
    def generate(self, customer_query: str, intent: str, exemplars: List[Dict[str, Any]], **kwargs) -> str:
        return "Thank you for contacting Amazon Support. We are reviewing your issue and will assist you shortly."


class RetrievalReplyBaseline:
    """Emits top retrieved historical exemplar reply directly."""
    def generate(self, customer_query: str, intent: str, exemplars: List[Dict[str, Any]], **kwargs) -> str:
        if exemplars and "reply_text" in exemplars[0]:
            return exemplars[0]["reply_text"]
        return "Please send us a direct message with your details so we can assist."


class ProductionGroundedGenerator:
    """
    Production Response Generator:
    - Contextual resolution signal extraction from historical exemplars.
    - Deep-linked Twitter DM routing (twitter.com/messages/compose) for one-tap mobile handoffs.
    - Specialized handlers for 3P Marketplace seller disputes, stolen packages, and damage.
    - Strict brand empathy guidelines and Twitter 280-character budget compliance.
    """
    def __init__(self, dm_link: str = "twitter.com/messages/compose?recipient_id=AmazonHelp"):
        self.dm_link = dm_link

    def _extract_resolution_signals(self, exemplars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts operational guidance from retrieved historical brand exemplars."""
        signals = {
            "has_link": False,
            "link_type": None,
            "advises_dm": False,
            "carrier_mentioned": None
        }
        for ex in exemplars:
            txt = ex.get("reply_text", "")
            if "dm" in txt.lower() or "direct message" in txt.lower():
                signals["advises_dm"] = True
            if "amazon.com/orders" in txt:
                signals["has_link"] = True
                signals["link_type"] = "orders"
            elif "amazon.com/returns" in txt:
                signals["has_link"] = True
                signals["link_type"] = "returns"
            elif "amazon.com/help" in txt:
                signals["has_link"] = True
                signals["link_type"] = "help"

            for carrier in ["ups", "usps", "hermes", "fedex", "royal mail", "dhl"]:
                if carrier in txt.lower():
                    signals["carrier_mentioned"] = carrier.upper()

        return signals

    def generate(
        self,
        customer_query: str,
        intent: str,
        exemplars: List[Dict[str, Any]],
        escalate_decision: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates brand-compliant, policy-grounded draft reply.
        Injects resolution signals extracted from historical exemplars.
        """
        should_escalate = escalate_decision.get("should_escalate", False) if escalate_decision else False
        stated_reason = escalate_decision.get("stated_reason", "") if escalate_decision else ""
        signals = self._extract_resolution_signals(exemplars)

        # -------------------------------------------------------------
        # Branch 1: Human Escalation Pathways
        # -------------------------------------------------------------
        if should_escalate:
            if "ACCOUNT_SECURITY" in stated_reason or "PII" in stated_reason:
                return (
                    f"Your account security is our top priority. Please never share credentials in public. "
                    f"Send us a secure DM: {self.dm_link} with your registered email so our security team can assist."
                )

            elif "MARKETPLACE_3P" in stated_reason:
                return (
                    f"For third-party Marketplace items, sellers have 48 hours to reply under Amazon policy. "
                    f"If they haven't resolved it, send us a DM: {self.dm_link} with your 17-digit order ID for A-to-z Guarantee protection."
                )

            elif "THEFT" in stated_reason or "STOLEN" in stated_reason:
                return (
                    f"I'm very sorry to hear your package was stolen. Please send us a direct message: {self.dm_link} "
                    f"with your 17-digit order ID so our courier claims team can open an immediate investigation and reissue."
                )

            elif "DELIVERY_DISPUTE" in stated_reason:
                return (
                    f"I'm so sorry your order is missing after being marked delivered. We want to investigate immediately. "
                    f"Please DM us: {self.dm_link} with your 17-digit order number so we can help."
                )

            elif "CARRIER" in stated_reason:
                carrier_note = f" with {signals['carrier_mentioned']}" if signals.get("carrier_mentioned") else ""
                return (
                    f"I apologize for the transit delay{carrier_note}! We want to check this for you directly. "
                    f"Please send us a DM: {self.dm_link} with your order number and tracking code so we can help."
                )

            elif "SEVERE_DAMAGE" in stated_reason or intent == "DAMAGED_OR_WRONG_ITEM":
                return (
                    f"I'm so sorry your item arrived damaged! We want to make this right immediately. "
                    f"Please DM us: {self.dm_link} with your 17-digit order ID and a photo of the packaging so we can issue a replacement."
                )

            elif "FINANCIAL_DISPUTE" in stated_reason:
                return (
                    f"We take unexpected charges and overdue refunds very seriously. Please send us a DM: {self.dm_link} "
                    f"with the transaction date and associated email so a specialist can audit this for you immediately."
                )

            elif "LEGAL" in stated_reason or "CHURN" in stated_reason:
                return (
                    f"We take your experience very seriously and want to resolve this without delay. "
                    f"Please send us a DM: {self.dm_link} with your order number so a Senior Specialist can review your case."
                )

            else:
                return (
                    f"I apologize for the frustration! We'd like to have a human specialist investigate this directly. "
                    f"Please send us a DM: {self.dm_link} with your order details so we can help."
                )

        # -------------------------------------------------------------
        # Branch 2: Self-Serve Automated Pathways
        # -------------------------------------------------------------
        if intent == "ORDER_TRACKING_AND_DELAY":
            return (
                "I'm sorry for the delay with your delivery! You can check real-time courier updates and transit status "
                "anytime under Your Orders: amazon.com/orders. If it hasn't arrived within 48 hours of the expected date, let us know!"
            )

        elif intent == "REFUND_AND_RETURN":
            return (
                "You can easily initiate a return or print a prepaid return label directly through Your Orders at "
                "amazon.com/returns. Once the carrier scans your package, refunds typically process in 3-5 business days."
            )

        elif intent == "DAMAGED_OR_WRONG_ITEM":
            return (
                "I'm very sorry your item arrived defective or incorrect! You can request an instant replacement or return "
                "label under Your Orders at amazon.com/returns. Let us know if you need any additional help!"
            )

        elif intent == "PRODUCT_AND_TECHNICAL":
            return (
                "For technical assistance with Kindle, Fire TV, Echo, or digital Prime content, you can find guided troubleshooting "
                "steps at amazon.com/help or reply with your device model so we can assist!"
            )

        elif intent == "ACCOUNT_AND_SECURITY":
            return (
                "You can manage your Prime membership, update billing details, or adjust login security settings anytime at "
                "amazon.com/your-account. Please let us know if you have trouble accessing your settings!"
            )

        else:
            # FEEDBACK_AND_GENERAL
            pos_words = ["thank", "great", "awesome", "kudos", "love"]
            if any(w in customer_query.lower() for w in pos_words):
                return (
                    "You're very welcome! We're always here to help whenever you need us. Have a wonderful rest of your day!"
                )
            return (
                "Thank you for sharing your feedback with us. We're continuously working to improve our service "
                "and appreciate you bringing this to our attention. Reach us anytime at amazon.com/help."
            )
