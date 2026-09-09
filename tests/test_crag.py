"""
Unit tests for Corrective Retrieval-Augmented Generation (src/crag_retriever.py).
"""

import unittest
from src.crag_retriever import CorrectiveRetriever

class TestCorrectiveRetriever(unittest.TestCase):
    def setUp(self):
        self.crag = CorrectiveRetriever(upper_threshold=0.35, lower_threshold=0.15)

    def test_correct_action_on_high_relevance(self):
        query = "Where is my package? The courier tracking says delayed."
        exemplars = [{
            "customer_text": "My package courier tracking says it is delayed in transit",
            "reply_text": "Please check amazon.com/orders for carrier tracking. ^AB",
            "rrf_score": 0.08
        }]
        res = self.crag.evaluate_and_correct(query, "ORDER_TRACKING_AND_DELAY", exemplars)
        self.assertEqual(res["action"], "CORRECT")
        self.assertEqual(len(res["sanitized_exemplars"]), 1)
        self.assertNotIn("^AB", res["sanitized_exemplars"][0]["reply_text"])

    def test_fallback_action_on_irrelevant_exemplars(self):
        query = "Where is my package? Delayed delivery."
        exemplars = [{
            "customer_text": "Fire TV stick won't connect to remote bluetooth",
            "reply_text": "Try resetting the device battery. ^CD",
            "rrf_score": 0.005
        }]
        res = self.crag.evaluate_and_correct(query, "ORDER_TRACKING_AND_DELAY", exemplars)
        self.assertEqual(res["action"], "INCORRECT_FALLBACK")
        self.assertEqual(len(res["sanitized_exemplars"]), 0)
        self.assertIn("amazon.com/orders", res["policy_guidance"]["url"])

    def test_empty_exemplars(self):
        res = self.crag.evaluate_and_correct("Any query", "REFUND_AND_RETURN", [])
        self.assertEqual(res["action"], "INCORRECT_FALLBACK")
        self.assertEqual(res["relevance_score"], 0.0)
        self.assertIn("amazon.com/returns", res["policy_guidance"]["url"])

if __name__ == "__main__":
    unittest.main()
