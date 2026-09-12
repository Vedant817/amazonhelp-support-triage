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
        self.assertIn("noise-aware-hybrid-dual-head", data["model_version"])
        self.assertEqual(data["generation_mode"], "offline")
        self.assertGreater(data["escalation_threshold"], 0.0)
        self.assertGreater(data["high_precision_escalation_threshold"], 0.0)
        self.assertEqual(data["intent_encoder"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertIn("character TF-IDF", data["classification_features"])
        self.assertGreater(data["weak_intent_labels_pruned"], 0)
        self.assertEqual(data["generation_backend"], "act-conditioned deterministic rewrite")
        self.assertFalse(data["hosted_api_required"])

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
        self.assertIn("all_probabilities", data)
        self.assertEqual(len(data["pipeline_trace"]), 6)
        self.assertEqual(data["pipeline_trace"][0]["stage"], "Intent classification")
        self.assertTrue(all(stage["duration_ms"] >= 0 for stage in data["pipeline_trace"]))
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

    def test_dashboard_uses_live_api_data_without_external_ui_dependencies(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('fetch("/api/v1/health")', response.text)
        self.assertIn("pipeline_trace", response.text)
        self.assertIn("retrieval_evidence", response.text)
        self.assertNotIn("cdn.tailwindcss.com", response.text)
        self.assertNotIn("bg-gradient", response.text)


if __name__ == "__main__":
    unittest.main()
