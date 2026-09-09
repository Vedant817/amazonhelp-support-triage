"""Retrieval baselines and resolution-act-conditioned reply generation."""

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

from src.local_llm import LocalInstructionModel
from src.supervision import infer_resolution_act, strip_agent_pii

class TrivialReplyBaseline:
    """Emits standard canned message."""
    def generate(self, customer_query: str = "", intent: str = "", exemplars: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        return "Thank you for contacting Amazon Support. We are reviewing your issue and will assist you shortly."


class RetrievalReplayBaseline:
    """Emits top retrieved historical exemplar reply directly."""
    def generate(self, customer_query: str = "", intent: str = "", exemplars: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        if exemplars and len(exemplars) > 0 and "reply_text" in exemplars[0]:
            return exemplars[0]["reply_text"]
        return "Please send us a direct message with your details so we can assist."

# Alias
RetrievalReplyBaseline = RetrievalReplayBaseline


class BaselineRetrievalOnlyGenerator:
    """Baseline 1: Returns the top retrieved agent historical reply verbatim."""
    def generate(self, exemplars: List[Dict[str, Any]], customer_handle: str = "@customer") -> str:
        if not exemplars:
            return f"{customer_handle} Please reach out to Amazon Customer Support for assistance: amazon.com/help"
        top_reply = exemplars[0]["reply_text"]
        cleaned = re.sub(r"\^[A-Z]{2,3}", "", top_reply).strip()
        return cleaned


class BaselineGenerativeGenerator:
    """Baseline 2: Unconstrained template completion without brand grounding."""
    def generate(self, customer_query: str, intent: str, customer_handle: str = "@customer") -> str:
        return (
            f"{customer_handle} We understand your inquiry regarding {intent.lower().replace('_', ' ')}. "
            f"We are looking into '{customer_query[:50]}' and will resolve your problem as soon as possible. "
            f"Thank you for contacting customer service!"
        )


class ProductionGroundedGenerator:
    """
    Resolution-act-conditioned support generator for @AmazonHelp.

    Local model output is always passed through the separate policy guard. Offline
    mode uses act-specific templates for reproducible runs, not as LLM evidence.
    """
    def __init__(self, mode: str = "offline", local_model: Optional[LocalInstructionModel] = None):
        if mode not in {"offline", "local"}:
            raise ValueError("generation mode must be 'offline' or 'local'")
        self.mode = mode
        self.local_model = local_model
        self.dm_link = "twitter.com/messages/compose?recipient_id=AmazonHelp"

    def select_act(self, exemplars: List[Dict[str, Any]], intent: str, should_escalate: bool) -> str:
        if should_escalate:
            return "DM_WITH_ORDER_ID"
        defaults = {
            "ORDER_TRACKING_AND_DELAY": "LINK_ORDERS",
            "REFUND_AND_RETURN": "LINK_RETURNS",
            "DAMAGED_OR_WRONG_ITEM": "LINK_RETURNS",
            "PRODUCT_AND_TECHNICAL": "TROUBLESHOOT_DEVICE",
            "ACCOUNT_AND_SECURITY": "DM_WITH_ORDER_ID",
        }
        allowed = {
            "ORDER_TRACKING_AND_DELAY": {"ASK_TRACKING_STATUS", "ASK_CARRIER", "LINK_ORDERS", "APOLOGIZE_WAIT"},
            "REFUND_AND_RETURN": {"LINK_RETURNS"},
            "DAMAGED_OR_WRONG_ITEM": {"LINK_RETURNS"},
            "PRODUCT_AND_TECHNICAL": {"TROUBLESHOOT_DEVICE"},
            "ACCOUNT_AND_SECURITY": {"DM_WITH_ORDER_ID"},
            "FEEDBACK_AND_GENERAL": {"GENERAL_SUPPORT"},
        }
        eligible = allowed.get(intent, {"GENERAL_SUPPORT"})
        scores: Dict[str, float] = {}
        first_rank: Dict[str, int] = {}
        for rank, exemplar in enumerate(exemplars):
            act = exemplar.get("resolution_act") or infer_resolution_act(exemplar.get("reply_text", ""))
            if act not in eligible:
                continue
            score = float(exemplar.get("rrf_score", 0.0)) or (1.0 / (rank + 1))
            scores[act] = scores.get(act, 0.0) + score
            first_rank.setdefault(act, rank)
        if scores:
            return max(scores, key=lambda act: (scores[act], -first_rank[act], act))
        return defaults.get(intent, "GENERAL_SUPPORT")

    def _offline_rewrite(self, customer_query: str, intent: str, act: str, reason: str = "") -> str:
        templates = {
            "ASK_TRACKING_STATUS": "I'm sorry about the delay. What does the latest tracking status show, and what was the original delivery date? Please don't post order details publicly.",
            "ASK_CARRIER": "I'm sorry this delivery has been frustrating. Which carrier handled it, and what does the latest tracking update show? Please don't share order details publicly.",
            "DM_WITH_ORDER_ID": f"I'm sorry you're dealing with this. Please send us a private DM so we can review the account-specific details safely: {self.dm_link}",
            "LINK_ORDERS": "I'm sorry for the wait. Check the latest carrier update under Your Orders: amazon.com/orders. If the status shows delivered but it is missing, send us a private DM.",
            "LINK_RETURNS": "I'm sorry the item wasn't right. Review the available return or replacement options under Your Orders: amazon.com/returns. Please don't post order details publicly.",
            "TROUBLESHOOT_DEVICE": "I'm sorry the device isn't working as expected. Tell us the model and the troubleshooting step you've already tried, without sharing account details publicly.",
            "APOLOGIZE_WAIT": "I'm sorry for the wait. Please check the latest status under Your Orders at amazon.com/orders, and let us know if the promised delivery window has passed.",
            "GENERAL_SUPPORT": "Thanks for reaching out. Please tell us a little more about what happened, without posting account or order details publicly, so we can point you to the right next step.",
        }
        escalation_templates = {
            "ACCOUNT_SECURITY_OR_LOCKOUT": f"I'm sorry this happened. For account security, don't post any details here. Send us a private DM so a specialist can review it safely: {self.dm_link}",
            "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING": f"I'm sorry the delivery can't be located. Please send the order details privately so the delivery can be investigated: {self.dm_link}",
            "FINANCIAL_DISPUTE_OR_OVERDUE_REFUND": f"I'm sorry about the billing issue. Please send the order details privately so the charge or refund can be reviewed safely: {self.dm_link}",
            "SEVERE_DAMAGE_OR_MISSING_CONTENTS": f"I'm sorry the order arrived in that condition. Please send the order details and photos privately so the issue can be reviewed: {self.dm_link}",
            "HIGH_HOSTILITY_OR_LEGAL_RISK": f"I'm sorry this remains unresolved. Please send the relevant order details privately so a specialist can review the case: {self.dm_link}",
        }
        if act == "DM_WITH_ORDER_ID" and reason in escalation_templates:
            return escalation_templates[reason]
        return templates[act]

    def _local_rewrite(
        self,
        customer_query: str,
        intent: str,
        act: str,
        exemplars: List[Dict[str, Any]],
        reason: str = "",
    ) -> str:
        if self.local_model is None:
            self.local_model = LocalInstructionModel()
        examples = "\n".join(
            f"Historical reply {idx + 1}: {strip_agent_pii(ex.get('reply_text', ''))}"
            for idx, ex in enumerate(exemplars[:2])
        ) or "No safe historical wording available."
        prompt = f"""Write one public @AmazonHelp reply.
Customer: {customer_query}
Intent: {intent}
Required resolution act: {act}
Escalation reason: {reason or 'none'}
Historical voice examples (style only):
{examples}

Rules: perform the required act; be empathetic and specific; maximum 240 characters before guardrails;
use only amazon.com/orders, amazon.com/returns, amazon.com/help, or {self.dm_link};
never invent refunds, replacements, delivery times, order IDs, or facts; never echo PII;
ask for private details only through DM. Output only the reply."""
        generated = self.local_model.generate(
            [
                {"role": "system", "content": "You draft safe, concise public customer-support replies."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=110,
        )
        normalized = re.sub(r"\s+", " ", generated.lower()).strip()
        for exemplar in exemplars[:2]:
            historical = re.sub(r"\s+", " ", strip_agent_pii(exemplar.get("reply_text", "")).lower()).strip()
            if historical and SequenceMatcher(None, normalized, historical).ratio() >= 0.90:
                return self._offline_rewrite(customer_query, intent, act, reason)
        return generated

    def generate(
        self,
        customer_query: str,
        intent: str,
        exemplars: List[Dict[str, Any]],
        escalate_decision: Optional[Dict[str, Any]] = None,
        policy_guidance: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a grounded, empathetic, policy-compliant reply under 280 characters.
        """
        should_escalate = escalate_decision.get("should_escalate", False) if escalate_decision else False
        reason = escalate_decision.get("stated_reason", "") if escalate_decision else ""
        act = self.select_act(exemplars, intent, should_escalate)

        if self.mode == "local":
            return self._local_rewrite(customer_query, intent, act, exemplars, reason)
        return self._offline_rewrite(customer_query, intent, act, reason)
