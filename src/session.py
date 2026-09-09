"""
Multi-Turn Conversational State & Friction Tracker for @AmazonHelp.
Tracks multi-turn customer dialogues, calculates escalation velocity,
and enforces SLA bounds (preventing public endless loops on Twitter).
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class DialogueTurn:
    turn_index: int
    customer_text: str
    predicted_intent: str
    escalate_decision: bool
    stated_reason: str
    dispatched_reply: str

class CustomerSessionState:
    """
    Stateful conversational session for an individual customer thread.
    """
    def __init__(self, thread_id: str, customer_handle: str = "@customer"):
        self.thread_id = thread_id
        self.customer_handle = customer_handle
        self.turns: List[DialogueTurn] = []
        self.is_resolved: bool = False
        self.escalated_to_human: bool = False
        self.escalation_reason: Optional[str] = None

        # Repetition & escalation triggers
        self.prior_attempt_patterns = [
            r"(already\s+(tried|done|contacted|chatted|emailed|called))",
            r"(told\s+you\s+(this|that|already))",
            r"(still\s+no\s+(help|answer|reply|solution))",
            r"(\b3rd\b|\bthird\b|\bsecond\b|\b2nd\b|\bmultiple\b)\s+time",
            r"(no\s+one\s+(helped|replied|resolved))",
            r"(loop|going\s+in\s+circles)"
        ]

    def add_turn(
        self,
        customer_text: str,
        predicted_intent: str,
        escalate: bool,
        stated_reason: str,
        reply: str
    ) -> DialogueTurn:
        turn = DialogueTurn(
            turn_index=len(self.turns) + 1,
            customer_text=customer_text,
            predicted_intent=predicted_intent,
            escalate_decision=escalate,
            stated_reason=stated_reason,
            dispatched_reply=reply
        )
        self.turns.append(turn)
        if escalate:
            self.escalated_to_human = True
            self.escalation_reason = stated_reason
        return turn

    def evaluate_multi_turn_friction(self, current_text: str) -> Dict[str, Any]:
        """
        Evaluates whether cumulative multi-turn friction requires overriding to human escalation.
        Triggers:
        1. Turn count limit (max 2 public automated replies on Twitter)
        2. Explicit customer repetition / loop frustration
        3. Intent thrashing (flapping between disparate intents)
        """
        lower = current_text.lower()
        turn_count = len(self.turns)

        # 1. Prior attempt / recurring loop friction
        for pat in self.prior_attempt_patterns:
            if re.search(pat, lower):
                return {
                    "override_escalate": True,
                    "reason": "MULTI_TURN_CUSTOMER_REPETITION_OR_CIRCULAR_LOOP",
                    "friction_score": 0.95
                }

        # 2. Maximum Public Turn SLA (Max 2 bot replies on public Twitter timeline)
        if turn_count >= 2:
            return {
                "override_escalate": True,
                "reason": "MULTI_TURN_TWITTER_SLA_EXCEEDED_MAX_TURNS",
                "friction_score": 0.90
            }

        # 3. Repeat identical intent without resolution
        if turn_count >= 1:
            prev_intent = self.turns[-1].predicted_intent
            # If customer is sending another message on the same tracking/refund problem
            if prev_intent in ["ORDER_TRACKING_AND_DELAY", "REFUND_AND_RETURN", "DAMAGED_OR_WRONG_ITEM"]:
                return {
                    "override_escalate": True,
                    "reason": "MULTI_TURN_UNRESOLVED_PRIMARY_INQUIRY",
                    "friction_score": 0.85
                }

        return {
            "override_escalate": False,
            "reason": "MULTI_TURN_FLOW_WITHIN_NORMAL_BOUNDS",
            "friction_score": 0.10
        }
