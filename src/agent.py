"""
AmazonHelp Support Agent Orchestrator.
Integrates:
1. Calibrated Intent Classifier
2. Conformal Prediction Set & Abstention Engine
3. Multi-factor Risk-Scored Escalation Engine
4. Hybrid Historical Resolution Retriever (BM25 + TF-IDF RRF)
5. Grounded Response Generator
6. Pre-Dispatch Production Policy Guard
7. Multi-Turn Conversational Session Tracker
"""

import time
from typing import Dict, Any, Optional, List
from src.classifier import ProductionIntentClassifier
from src.conformal import ConformalPredictor
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
        conformal_predictor: Optional[ConformalPredictor] = None,
        escalator: Optional[ProductionEscalator] = None,
        retriever: Optional[HistoricalResolutionRetriever] = None,
        generator: Optional[ProductionGroundedGenerator] = None,
        guard: Optional[ProductionPolicyGuard] = None
    ):
        self.classifier = classifier or ProductionIntentClassifier()
        self.conformal_predictor = conformal_predictor
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
        3. Conformal uncertainty evaluation & coverage guarantee.
        4. Risk-scored escalation triage with stated reason.
        5. Hybrid resolution exemplar retrieval (BM25 + Dense RRF).
        6. Grounded brand response generation.
        7. Pre-dispatch policy guardrail verification & sanitization.
        8. State tracking update.
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
        prob_vector = cls_result.get("prob_vector", None)

        # Step 3: Conformal Prediction check (if calibrated)
        conformal_res = None
        if self.conformal_predictor is not None and prob_vector is not None:
            conformal_res = self.conformal_predictor.predict_set(prob_vector)
            if conformal_res["should_abstain"]:
                is_uncertain = True

        # Step 4: Escalation triage
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
        elif conformal_res and conformal_res["should_abstain"] and not escalate_decision.should_escalate:
            # Conformal statistical coverage bound triggered abstention
            escalate_decision = EscalationDecision(
                should_escalate=True,
                stated_reason="CONFORMAL_STATISTICAL_UNCERTAINTY",
                risk_score=0.85,
                triggers=["CONFORMAL_PREDICTION_SET_AMBIGUITY"]
            )

        escalate_dict = escalate_decision.to_dict()

        # Step 5: Retrieve historical resolution exemplars (Hybrid BM25 + TF-IDF RRF)
        exemplars = self.retriever.retrieve_similar(query=customer_tweet, intent_filter=intent, top_k=3)

        # Step 6: Draft grounded reply
        raw_reply = self.generator.generate(
            customer_query=customer_tweet,
            intent=intent,
            exemplars=exemplars,
            escalate_decision=escalate_dict
        )

        # Step 7: Pre-dispatch policy guard verification & sanitization
        guard_res = self.guard.verify_and_sanitize(
            draft_reply=raw_reply,
            customer_text=customer_tweet,
            should_escalate=escalate_decision.should_escalate
        )
        final_reply = guard_res.sanitized_reply

        # Step 8: Update session if active
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
            "conformal_prediction": conformal_res,
            "escalation": escalate_dict,
            "exemplars": exemplars,
            "raw_reply": raw_reply,
            "draft_reply": final_reply,
            "policy_verification": guard_res.to_dict(),
            "latency_ms": latency_ms
        }
