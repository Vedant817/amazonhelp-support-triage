import os
import unittest

os.environ.setdefault("SUPPORT_INTENT_CAP_PER_CLASS", "30")
os.environ.setdefault("SUPPORT_RETRIEVAL_SIZE", "100")
os.environ.setdefault("SUPPORT_GENERATION_MODE", "offline")

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

    def test_health_reports_actual_methods(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("minilm-dual-head", data["model_version"])
        self.assertEqual(data["generation_mode"], "offline")
        self.assertGreater(data["escalation_threshold"], 0.0)

    def test_confidence_endpoint(self):
        response = self.client.post("/api/v1/confidence", json={"text": "Where is my package?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("all_probabilities", data)
        self.assertIn("should_abstain", data)

    def test_triage_returns_retrieval_act_and_guarded_reply(self):
        response = self.client.post("/api/v1/triage", json={"text": "Where is my package?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("resolution_act", data)
        self.assertIn("retrieval_evidence", data)
        self.assertLessEqual(data["character_count"], 280)

    def test_hacked_account_is_hard_gated(self):
        response = self.client.post(
            "/api/v1/triage",
            json={"text": "My account was hacked and my email is victim@example.com"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["escalation"]["should_escalate"])
        self.assertIn("HARD_GATE", data["escalation"]["triggers"][0])
        self.assertNotIn("victim@example.com", data["draft_reply"])

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("amazonhelp_requests_total", response.text)


if __name__ == "__main__":
    unittest.main()
