"""
AmazonHelp Support Agent Orchestrator.
Integrates:
1. Calibrated Intent Classifier
2. Multi-factor Risk-Scored Escalation Engine
3. Historical Resolution Retriever
4. Grounded Response Generator
5. Pre-Dispatch Production Policy Guard
6. Multi-Turn Conversational Session Tracker
"""

import time
from typing import Dict, Any, Optional, List
from src.classifier import ProductionIntentClassifier
from src.escalator import ProductionEscalator, EscalationDecision
from src.retriever import HistoricalResolutionRetriever
from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard
from src.session import CustomerSessionState

class AmazonSupportAgent:
    """
    Production Support Agent for @AmazonHelp.
    """
    def __init__(
        self,
        classifier: Optional[ProductionIntentClassifier] = None,
        escalator: Optional[ProductionEscalator] = None,
        retriever: Optional[HistoricalResolutionRetriever] = None,
        generator: Optional[ProductionGroundedGenerator] = None,
        guard: Optional[ProductionPolicyGuard] = None
    ):
        self.classifier = classifier or ProductionIntentClassifier()
        self.escalator = escalator or ProductionEscalator()
        self.retriever = retriever or HistoricalResolutionRetriever()
        self.generator = generator or ProductionGroundedGenerator()
        self.guard = guard or ProductionPolicyGuard()

    def process_message(
        self,
        customer_tweet: str,
        session: Optional[CustomerSessionState] = None
    ) -> Dict[str, Any]:
        """
        End-to-end processing of an incoming customer tweet:
        1. Multi-turn friction check (if ongoing session provided).
        2. Intent classification & posterior confidence.
        3. Risk-scored escalation triage with stated reason.
        4. Historical resolution exemplar retrieval.
        5. Grounded brand response generation.
        6. Pre-dispatch policy guardrail verification & sanitization.
        7. State tracking update.
        """
        start_time = time.time()

        # Step 1: Check multi-turn session friction
        multi_turn_override = None
        if session is not None:
            multi_turn_override = session.evaluate_multi_turn_friction(customer_tweet)

        # Step 2: Intent classification
        cls_result = self.classifier.predict_with_confidence(customer_tweet)
        intent = cls_result["intent"]
        confidence = cls_result["confidence"]
        is_uncertain = cls_result["is_uncertain"]

        # Step 3: Escalation triage
        escalate_decision = self.escalator.evaluate(
            text=customer_tweet,
            intent=intent,
            confidence=confidence,
            is_uncertain=is_uncertain
        )

        # If multi-turn friction warrants override
        if multi_turn_override and multi_turn_override.get("override_escalate"):
            escalate_decision = EscalationDecision(
                should_escalate=True,
                stated_reason=multi_turn_override["reason"],
                risk_score=multi_turn_override["friction_score"],
                triggers=["MULTI_TURN_SESSION_FRICTION"]
            )

        escalate_dict = escalate_decision.to_dict()

        # Step 4: Retrieve historical resolution exemplars
        exemplars = self.retriever.retrieve_similar(query=customer_tweet, intent_filter=intent, top_k=3)

        # Step 5: Draft grounded reply
        raw_reply = self.generator.generate(
            customer_query=customer_tweet,
            intent=intent,
            exemplars=exemplars,
            escalate_decision=escalate_dict
        )

        # Step 6: Pre-dispatch policy guard verification & sanitization
        guard_res = self.guard.verify_and_sanitize(
            draft_reply=raw_reply,
            customer_text=customer_tweet,
            should_escalate=escalate_decision.should_escalate
        )
        final_reply = guard_res.sanitized_reply

        # Step 7: Update session if active
        if session is not None:
            session.add_turn(
                customer_text=customer_tweet,
                predicted_intent=intent,
                escalate=escalate_decision.should_escalate,
                stated_reason=escalate_decision.stated_reason,
                reply=final_reply
            )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "incoming_tweet": customer_tweet,
            "intent": intent,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "escalation": escalate_dict,
            "exemplars": exemplars,
            "raw_reply": raw_reply,
            "draft_reply": final_reply,
            "policy_verification": guard_res.to_dict(),
            "latency_ms": latency_ms
        }
