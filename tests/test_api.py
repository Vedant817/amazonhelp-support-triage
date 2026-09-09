"""
Integration Tests for FastAPI Support API Service.
Tests all endpoints using FastAPI TestClient.
"""

import unittest
from fastapi.testclient import TestClient
from src.api import app

class TestSupportAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_dashboard_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("@AmazonHelp AI Support Agent", resp.text)
        self.assertIn("Live Triage & Inspection Dashboard", resp.text)

    def test_health_endpoint(self):
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["brand"], "@AmazonHelp")
        self.assertIn("Conformal", data["conformal_coverage_guarantee"])

    def test_conformal_lookup_endpoint(self):
        payload = {"text": "I want to track my package delivery status"}
        resp = self.client.post("/api/v1/conformal", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("prediction_set", data)
        self.assertIn("ORDER_TRACKING_AND_DELAY", data["prediction_set"])
        self.assertGreaterEqual(data["set_size"], 1)
        self.assertIn("confidence", data)

    def test_triage_routine_tracking(self):
        payload = {
            "text": "@AmazonHelp where is my package? The tracking says out for delivery today."
        }
        resp = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "ORDER_TRACKING_AND_DELAY")
        self.assertFalse(data["escalation"]["should_escalate"])
        self.assertIn("amazon.com/orders", data["draft_reply"])
        self.assertLessEqual(data["character_count"], 280)
        self.assertLess(data["latency_ms"], 250.0)
        # Verify typed OOD fields
        self.assertIn("status", data["ood_detection"])
        self.assertIn("is_ood", data["ood_detection"])

    def test_triage_security_escalation_with_pii(self):
        payload = {
            "text": "Help! Someone hacked my prime account, my email is victim@example.com and order is 112-8291029-3829102!"
        }
        resp = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "ACCOUNT_AND_SECURITY")
        self.assertTrue(data["escalation"]["should_escalate"])
        self.assertNotIn("victim@example.com", data["draft_reply"])
        self.assertNotIn("112-8291029-3829102", data["draft_reply"])
        self.assertIn("twitter.com/messages/compose", data["draft_reply"])

    def test_triage_stolen_package_escalation(self):
        payload = {
            "text": "@AmazonHelp my package was marked delivered but a porch pirate stole it from my front door!"
        }
        resp = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["escalation"]["should_escalate"])
        self.assertIn("THEFT", data["escalation"]["stated_reason"])

    def test_multi_turn_session_friction_tracking(self):
        session_id = "test_thread_9999"
        # Turn 1: normal tracking
        payload_1 = {
            "text": "Where is my package? Out for delivery today.",
            "session_id": session_id
        }
        resp1 = self.client.post("/api/v1/triage", json=payload_1)
        self.assertEqual(resp1.status_code, 200)

        # Turn 2: customer reports circular friction / repetition
        payload_2 = {
            "text": "I already told you this, still no answer and no one helped me!",
            "session_id": session_id
        }
        resp2 = self.client.post("/api/v1/triage", json=payload_2)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertTrue(data2["escalation"]["should_escalate"])
        self.assertIn("MULTI_TURN", data2["escalation"]["stated_reason"])

    def test_metrics_endpoint(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        text = resp.text
        self.assertIn("amazonhelp_requests_total", text)
        self.assertIn("amazonhelp_escalations_total", text)
        self.assertIn("amazonhelp_ood_total", text)
        self.assertIn("amazonhelp_latency_ms_avg", text)

if __name__ == "__main__":
    unittest.main()
