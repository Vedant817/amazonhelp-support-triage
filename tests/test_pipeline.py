"""
Unit tests for @AmazonHelp Support AI Pipeline Components.
Tests:
- Data preprocessor
- Intent Classifier inference & confidence
- Multi-factor Escalation Engine rules
- Conformal Prediction set generation & coverage bounds
- Hybrid BM25 + TF-IDF RRF Retriever
- Production Policy Guardrails & Sanitization
- Multi-turn Conversational Session Tracking
- Grounded Reply Generator Twitter character constraints
- LLM Judge scoring rubrics
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifier import ProductionIntentClassifier, preprocess_tweet
from src.escalator import ProductionEscalator
from src.generator import ProductionGroundedGenerator
from src.llm_judge import ReplyQualityJudge
from src.conformal import ConformalPredictor
from src.retriever import HistoricalResolutionRetriever
from src.guardrails import ProductionPolicyGuard
from src.session import CustomerSessionState

class TestPipelineComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Quick fit with minimal representative sample
        cls.clf = ProductionIntentClassifier(confidence_threshold=0.32)
        sample_texts = [
            "Where is my package delivery? Tracking is stuck.",
            "I need a refund for my damaged returned item.",
            "Item arrived shattered and broken into pieces.",
            "My account was hacked and password changed.",
            "My kindle paperwhite won't turn on or charge.",
            "Thank you Amazon for the great customer service!"
        ]
        sample_intents = [
            "ORDER_TRACKING_AND_DELAY",
            "REFUND_AND_RETURN",
            "DAMAGED_OR_WRONG_ITEM",
            "ACCOUNT_AND_SECURITY",
            "PRODUCT_AND_TECHNICAL",
            "FEEDBACK_AND_GENERAL"
        ]
        cls.clf.fit(sample_texts, sample_intents)
        cls.esc = ProductionEscalator(confidence_threshold=0.32)
        cls.gen = ProductionGroundedGenerator()
        cls.judge = ReplyQualityJudge()
        cls.guard = ProductionPolicyGuard()

    def test_preprocess_tweet(self):
        raw = "@AmazonHelp http://t.co/xyz Can you help with order &amp; tracking?"
        cleaned = preprocess_tweet(raw)
        self.assertNotIn("@amazonhelp", cleaned)
        self.assertNotIn("http://", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertIn("tracking", cleaned)

    def test_classifier_output(self):
        res = self.clf.predict_with_confidence("Where is my package delivery?")
        self.assertIn(res["intent"], [
            "ORDER_TRACKING_AND_DELAY", "FEEDBACK_AND_GENERAL", "REFUND_AND_RETURN",
            "DAMAGED_OR_WRONG_ITEM", "ACCOUNT_AND_SECURITY", "PRODUCT_AND_TECHNICAL"
        ])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertIn("prob_vector", res)

    def test_escalation_security_threat(self):
        # Must escalate compromised accounts
        dec = self.esc.evaluate("My account was hacked and someone ordered an iPhone!")
        self.assertTrue(dec.should_escalate)
        self.assertEqual(dec.stated_reason, "ACCOUNT_SECURITY_OR_LOCKOUT")
        self.assertGreater(dec.risk_score, 0.9)

    def test_escalation_delivery_dispute(self):
        # Must escalate false delivery scan
        dec = self.esc.evaluate("Tracking says delivered but package is missing and nowhere outside.")
        self.assertTrue(dec.should_escalate)
        self.assertEqual(dec.stated_reason, "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING")

    def test_escalation_auto_handle(self):
        # Routine tracking should be auto-handled
        dec = self.esc.evaluate("When will my order ship out?", intent="ORDER_TRACKING_AND_DELAY", confidence=0.85)
        self.assertFalse(dec.should_escalate)
        self.assertEqual(dec.stated_reason, "ROUTINE_TRACKING_IN_TRANSIT")

    def test_generator_character_limit(self):
        # All replies must be under 280 characters for Twitter
        reply = self.gen.generate(
            "How do I return this item?",
            "REFUND_AND_RETURN",
            [],
            {"escalate": False, "stated_reason": "ROUTINE_RETURN_POLICY_OR_PORTAL"}
        )
        self.assertLessEqual(len(reply), 280)
        self.assertIn("amazon.com/returns", reply)

    def test_conformal_predictor(self):
        cp = ConformalPredictor(alpha=0.10)
        classes = ["A", "B"]
        dummy_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.85, 0.15]])
        y_true = ["A", "A", "B", "A"]
        cp.calibrate(dummy_probs, y_true, classes)
        self.assertTrue(cp.is_calibrated)
        
        # Test confident prediction
        res_conf = cp.predict_set(np.array([0.95, 0.05]))
        self.assertFalse(res_conf["should_abstain"])
        self.assertEqual(res_conf["set_size"], 1)

    def test_hybrid_retriever(self):
        df = pd.DataFrame([
            {"customer_text": "Where is my package tracking?", "reply_text": "Check amazon.com/orders", "intent": "ORDER_TRACKING_AND_DELAY"},
            {"customer_text": "Item arrived broken and shattered", "reply_text": "Please return at amazon.com/returns", "intent": "DAMAGED_OR_WRONG_ITEM"}
        ])
        retriever = HistoricalResolutionRetriever(top_k=1).build_index(df)
        results = retriever.retrieve_similar("Where is my package?")
        self.assertGreater(len(results), 0)
        self.assertIn("rrf_score", results[0])

    def test_policy_guardrail(self):
        # Must neutralize unapproved promises and strip external links
        unsafe_reply = "We have issued a full refund of $50! Visit http://malicious-site.com/claim"
        sanitized = self.guard.verify_and_sanitize(unsafe_reply, customer_text="refund please", should_escalate=True)
        self.assertNotIn("http://malicious-site.com", sanitized.sanitized_reply)
        self.assertNotIn("issued a full refund", sanitized.sanitized_reply)

    def test_multi_turn_session(self):
        session = CustomerSessionState(thread_id="cust_123")
        session.add_turn("Where is my item?", "ORDER_TRACKING_AND_DELAY", False, "ROUTINE_TRACKING_IN_TRANSIT", "Check amazon.com/orders")
        
        # Subsequent angry turn with friction words
        friction = session.evaluate_multi_turn_friction("I already told you that, still no help for 4 days!")
        self.assertTrue(friction["override_escalate"])
        self.assertGreaterEqual(friction["friction_score"], 0.60)

    def test_llm_judge_scoring(self):
        score = self.judge.score_reply(
            "Package is late",
            "I'm sorry for the delay! Track anytime at amazon.com/orders.",
            "ORDER_TRACKING_AND_DELAY",
            False
        )
        self.assertGreaterEqual(score["overall_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
