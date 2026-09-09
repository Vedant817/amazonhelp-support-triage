"""
Unit tests for @AmazonHelp Support AI Pipeline Components.
Tests:
- Data preprocessor
- Intent Classifier inference & confidence
- Multi-factor Escalation Engine rules
- Grounded Reply Generator Twitter character constraints
- LLM Judge scoring rubrics
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifier import ProductionIntentClassifier, preprocess_tweet
from src.escalator import ProductionEscalator
from src.generator import ProductionGroundedGenerator
from src.llm_judge import ReplyQualityJudge

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

    def test_preprocess_tweet(self):
        raw = "@AmazonHelp http://t.co/xyz Can you help with order &amp; tracking?"
        cleaned = preprocess_tweet(raw)
        self.assertNotIn("@amazonhelp", cleaned)
        self.assertNotIn("http://", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertIn("tracking", cleaned)

    def test_classifier_output(self):
        res = self.clf.predict_with_confidence("Where is my package delivery?")
        self.assertIn(res["intent"], ["ORDER_TRACKING_AND_DELAY", "FEEDBACK_AND_GENERAL", "REFUND_AND_RETURN", "DAMAGED_OR_WRONG_ITEM", "ACCOUNT_AND_SECURITY", "PRODUCT_AND_TECHNICAL"])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

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

    def test_llm_judge_scoring(self):
        score = self.judge.score_reply(
            "Package is late",
            "I'm sorry for the delay! Track anytime at amazon.com/orders.",
            "ORDER_TRACKING_AND_DELAY",
            False
        )
        self.assertGreaterEqual(score["overall_score"], 1.0)
        self.assertLessEqual(score["overall_score"], 5.0)
        self.assertIn("overall_score", score)
        self.assertIn("groundedness", score)
        self.assertIn("safety", score)

if __name__ == "__main__":
    unittest.main()
