"""Framework-neutral orchestration for the learned support pipeline."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.dual_head import DualHeadSupportModel
from src.escalator import SafetyGatedEscalator
from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard
from src.retriever import HistoricalResolutionRetriever
from src.session import CustomerSessionState


class AmazonSupportAgent:
    def __init__(
        self,
        model: DualHeadSupportModel,
        retriever: HistoricalResolutionRetriever,
        generator: Optional[ProductionGroundedGenerator] = None,
        guard: Optional[ProductionPolicyGuard] = None,
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.generator = generator or ProductionGroundedGenerator(mode="offline")
        self.guard = guard or ProductionPolicyGuard()
        self.escalator = SafetyGatedEscalator(model.escalation_threshold_)

    def process_message(
        self, customer_tweet: str, session: Optional[CustomerSessionState] = None
    ) -> Dict[str, Any]:
        started = time.time()
        classification = self.model.predict_with_confidence(customer_tweet)
        learned = self.model.predict_escalation(customer_tweet, classification["embedding"])
        exemplars = self.retriever.retrieve_similar(
            customer_tweet, intent_filter=classification["intent"], top_k=3
        )
        evidence_act = self.generator.select_act(exemplars, classification["intent"], False)
        decision = self.escalator.evaluate(
            customer_tweet,
            classification["intent"],
            learned["risk_score"],
            is_uncertain=classification["is_uncertain"],
            learned_threshold=learned["threshold"],
            high_precision_threshold=self.model.high_precision_escalation_threshold_,
            resolution_act=evidence_act,
        )
        act = self.generator.select_act(exemplars, classification["intent"], decision.should_escalate)
        raw_reply = self.generator.generate(
            customer_tweet,
            classification["intent"],
            exemplars,
            escalate_decision=decision.to_dict(),
        )
        verified = self.guard.verify_and_sanitize(raw_reply, customer_tweet, decision.should_escalate)
        if session is not None:
            session.add_turn(
                customer_tweet,
                classification["intent"],
                decision.should_escalate,
                decision.stated_reason,
                verified.sanitized_reply,
            )
        return {
            "incoming_tweet": customer_tweet,
            "intent": classification["intent"],
            "confidence": classification["confidence"],
            "should_abstain": classification["is_uncertain"],
            "escalation": decision.to_dict(),
            "resolution_act": act,
            "exemplars": exemplars,
            "generation_mode": self.generator.mode,
            "draft_reply": verified.sanitized_reply,
            "policy_verification": verified.to_dict(),
            "latency_ms": round((time.time() - started) * 1000.0, 2),
        }
