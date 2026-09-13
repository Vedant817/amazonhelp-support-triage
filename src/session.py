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

        # Polite closure signals (e.g. thank you) to prevent Turn 2 trap
        self.pleasantries = [
            r"\b(thank\s+you|thanks|thx|cheers|appreciated|grateful)\b",
            r"\b(got\s+it|all\s+good|sorted|ok\s+cool)\b"
        ]

    def compose_model_input(self, current_text: str, max_prior_turns: int = 2) -> str:
        """Build the causal conversation prefix used by the learned pipeline."""
        prior = self.turns[-max_prior_turns:] if max_prior_turns > 0 else []
        lines: List[str] = []
        for turn in prior:
            lines.append(f"CUSTOMER: {turn.customer_text}")
            lines.append(f"AGENT: {turn.dispatched_reply}")
        if not lines:
            return current_text
        lines.append(f"CURRENT CUSTOMER: {current_text}")
        return "\n".join(lines)

    def add_turn(
        self,
        customer_text: str,
        intent: str,
        should_escalate: bool,
        stated_reason: str,
        reply: str
    ) -> DialogueTurn:
        """Appends a new turn to session history."""
        turn = DialogueTurn(
            turn_index=len(self.turns) + 1,
            customer_text=customer_text,
            predicted_intent=intent,
            escalate_decision=should_escalate,
            stated_reason=stated_reason,
            dispatched_reply=reply
        )
        self.turns.append(turn)
        if should_escalate:
            self.escalated_to_human = True
            self.escalation_reason = stated_reason
        return turn

    def evaluate_multi_turn_friction(self, current_text: str) -> Dict[str, Any]:
        """
        Evaluates conversational friction and repetition across turns.
        Rules:
        1. Repeated complaints / "already told you" -> instant escalation
        2. Customer pleasantry / thank-you -> no escalation (avoids Turn 2 Trap)
        3. Maximum Public Turn SLA (Max 2 public bot replies) -> deflect to DM
        """
        lower = current_text.lower()
        turn_count = len(self.turns)

        # 1. Pleasantry / closure suppression (prevents Turn 2 Trap)
        for pat in self.pleasantries:
            if re.search(pat, lower) and len(lower.split()) <= 6:
                return {
                    "override_escalate": False,
                    "reason": "CONVERSATIONAL_CLOSURE_OR_GRATITUDE",
                    "friction_score": 0.05
                }

        # 2. Prior attempt / recurring loop friction
        for pat in self.prior_attempt_patterns:
            if re.search(pat, lower):
                return {
                    "override_escalate": True,
                    "reason": "MULTI_TURN_CUSTOMER_REPETITION_OR_CIRCULAR_LOOP",
                    "friction_score": 0.95
                }

        # 3. Maximum Public Turn SLA (Max 2 bot replies on public Twitter timeline)
        if turn_count >= 2:
            return {
                "override_escalate": True,
                "reason": "MULTI_TURN_TWITTER_SLA_EXCEEDED_MAX_TURNS",
                "friction_score": 0.90
            }

        # 4. Repeat identical intent without resolution
        if turn_count >= 1:
            prev_intent = self.turns[-1].predicted_intent
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


class CustomerSessionTracker:
    """
    Session manager mapping thread / user IDs to CustomerSessionState instances.
    """
    def __init__(self):
        self.sessions: Dict[str, CustomerSessionState] = {}

    def get_or_create(self, session_id: str, customer_handle: str = "@customer") -> CustomerSessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = CustomerSessionState(thread_id=session_id, customer_handle=customer_handle)
        return self.sessions[session_id]
