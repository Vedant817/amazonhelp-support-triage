import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.dual_head import TemperatureScaler, select_recall_threshold
from src.escalator import SafetyGatedEscalator
from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard
from src.llm_judge import ReplyQualityJudge
from src.retriever import HistoricalResolutionRetriever
from src.supervision import build_reply_supervision, infer_resolution_act, strip_agent_pii


class FakeDenseEncoder:
    vocabulary = ("package", "tracking", "broken", "return", "kindle", "account")

    def encode(self, texts, batch_size=64):
        rows = []
        for text in texts:
            lower = text.lower()
            vector = np.array([lower.count(word) for word in self.vocabulary], dtype=np.float32)
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows)


class FakeLocalModel:
    model_name = "fake-local"
    revision = "test"

    def __init__(self, output):
        self.output = output

    def generate(self, messages, max_new_tokens=128):
        return self.output


class TestPipelineComponents(unittest.TestCase):
    def test_reply_supervision_does_not_use_customer_keyword(self):
        frame = pd.DataFrame([{
            "inbound_id": "1",
            "customer_text": "My package is broken and hacked",
            "reply_text": "What does the latest tracking status show?",
        }])
        labeled = build_reply_supervision(frame)
        self.assertEqual(labeled.iloc[0]["intent_label"], "ORDER_TRACKING_AND_DELAY")
        self.assertFalse(bool(labeled.iloc[0]["escalate_label"]))

    def test_reviewed_labels_reject_invalid_values(self):
        frame = pd.DataFrame([{
            "inbound_id": "1",
            "customer_text": "Help",
            "reply_text": "What does the latest tracking status show?",
        }])
        with self.assertRaises(ValueError):
            build_reply_supervision(frame, pd.DataFrame([{
                "inbound_id": "1",
                "reviewed_intent": "NOT_A_REAL_INTENT",
                "reviewed_escalate": "yes",
            }]))
        with self.assertRaises(ValueError):
            build_reply_supervision(frame, pd.DataFrame([{
                "inbound_id": "1",
                "reviewed_intent": "ORDER_TRACKING_AND_DELAY",
                "reviewed_escalate": "treu",
            }]))

    def test_reviewed_labels_reject_duplicate_ids(self):
        frame = pd.DataFrame([{
            "inbound_id": "1", "customer_text": "Help", "reply_text": "Contact us"
        }])
        reviewed = pd.DataFrame([
            {"inbound_id": "1", "reviewed_intent": "FEEDBACK_AND_GENERAL", "reviewed_escalate": True},
            {"inbound_id": "1", "reviewed_intent": "FEEDBACK_AND_GENERAL", "reviewed_escalate": False},
        ])
        with self.assertRaises(ValueError):
            build_reply_supervision(frame, reviewed)

    def test_resolution_act_and_pii_stripping(self):
        self.assertEqual(infer_resolution_act("Please DM us so we can help. ^AB"), "DM_WITH_ORDER_ID")
        cleaned = strip_agent_pii("@person Email test@example.com and order 123-1234567-1234567. ^AB")
        self.assertNotIn("test@example.com", cleaned)
        self.assertNotIn("123-1234567-1234567", cleaned)
        self.assertNotIn("@person", cleaned)

    def test_temperature_scaling(self):
        logits = np.array([[3.0, 0.2], [2.0, 0.1], [0.2, 2.5], [0.1, 3.0]])
        scaler = TemperatureScaler().fit_multiclass(logits, [0, 0, 1, 1])
        self.assertTrue(scaler.is_fitted_)
        self.assertGreater(scaler.temperature_, 0.0)

    def test_recall_threshold_hits_target(self):
        threshold = select_recall_threshold([True, True, False, False], [0.9, 0.7, 0.6, 0.1], 1.0)
        predicted = np.array([0.9, 0.7, 0.6, 0.1]) >= threshold
        self.assertEqual(int(predicted[:2].sum()), 2)

    def test_hard_safety_gate_cannot_be_vetoed(self):
        escalator = SafetyGatedEscalator(learned_threshold=0.8)
        decision = escalator.evaluate(
            "My account was hacked", "ACCOUNT_AND_SECURITY", learned_risk_score=0.01
        )
        self.assertTrue(decision.should_escalate)
        self.assertIn("HARD_GATE", decision.triggers[0])

    def test_closed_reason_codes(self):
        escalator = SafetyGatedEscalator(learned_threshold=0.5)
        decision = escalator.evaluate(
            "The delivery failed", "ORDER_TRACKING_AND_DELAY", learned_risk_score=0.9
        )
        self.assertEqual(decision.stated_reason, "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING")

    def test_minilm_rrf_path_uses_dense_encoder(self):
        frame = pd.DataFrame([
            {"customer_text": "package tracking", "reply_text": "What does tracking show?", "intent": "ORDER_TRACKING_AND_DELAY"},
            {"customer_text": "kindle broken", "reply_text": "Restart the device", "intent": "PRODUCT_AND_TECHNICAL"},
        ])
        retriever = HistoricalResolutionRetriever(dense_encoder=FakeDenseEncoder()).build_index(frame)
        result = retriever.retrieve_similar("track package", top_k=1)[0]
        self.assertEqual(result["dense_model"], "MiniLM")
        self.assertEqual(result["intent"], "ORDER_TRACKING_AND_DELAY")

    def test_dense_index_reload_rehydrates_query_encoder(self):
        frame = pd.DataFrame([
            {"customer_text": "package tracking", "reply_text": "What does tracking show?", "intent": "ORDER_TRACKING_AND_DELAY"},
            {"customer_text": "kindle broken", "reply_text": "Restart the device", "intent": "PRODUCT_AND_TECHNICAL"},
        ])
        original = HistoricalResolutionRetriever(dense_encoder=FakeDenseEncoder()).build_index(frame)
        with TemporaryDirectory() as directory:
            index_path = Path(directory) / "retriever.pkl"
            original.save_index(str(index_path))
            with patch("src.dual_head.MiniLMEncoder", return_value=FakeDenseEncoder()):
                restored = HistoricalResolutionRetriever().load_index(str(index_path))
            result = restored.retrieve_similar("track package", top_k=1)[0]
        self.assertEqual(result["dense_model"], "MiniLM")
        self.assertEqual(result["intent"], "ORDER_TRACKING_AND_DELAY")

    def test_offline_generation_is_act_conditioned(self):
        generator = ProductionGroundedGenerator(mode="offline")
        exemplars = [{"resolution_act": "ASK_CARRIER", "reply_text": "Which carrier?"}]
        reply = generator.generate("Delivery issue", "ORDER_TRACKING_AND_DELAY", exemplars, {"should_escalate": False})
        self.assertIn("carrier", reply.lower())
        self.assertLessEqual(len(reply), 280)

    def test_local_rewrite_rejects_verbatim_replay(self):
        historical = "What does the current tracking status show?"
        generator = ProductionGroundedGenerator(mode="local", local_model=FakeLocalModel(historical))
        reply = generator.generate(
            "Delivery issue", "ORDER_TRACKING_AND_DELAY",
            [{"resolution_act": "ASK_TRACKING_STATUS", "reply_text": historical}],
            {"should_escalate": False},
        )
        self.assertNotEqual(reply, historical)
        self.assertIn("original delivery date", reply)

    def test_guardrail_blocks_pii_echo_and_unapproved_url(self):
        guard = ProductionPolicyGuard()
        result = guard.verify_and_sanitize(
            "Email test@example.com at https://evil.example/a",
            "My email is test@example.com",
            True,
        )
        self.assertNotIn("test@example.com", result.sanitized_reply)
        self.assertNotIn("evil.example", result.sanitized_reply)

    def test_same_reply_agreement_statistics(self):
        result = ReplyQualityJudge.compute_human_agreement([1, 2, 4, 5], [1, 3, 4, 5])
        self.assertIn("spearman_rho", result)
        self.assertIn("quadratic_weighted_kappa", result)
        self.assertTrue(result["same_reply_required"])

    def test_local_judge_requires_structured_model_output(self):
        fake = FakeLocalModel('{"groundedness":4,"empathy":3,"actionability":4,"safety":5,"rationale":"ok"}')
        score = ReplyQualityJudge(fake).score_reply("late", "check orders", "ORDER_TRACKING_AND_DELAY", False)
        self.assertEqual(score["overall_score"], 4.0)
        self.assertEqual(score["judge_backend"], "fake-local")


if __name__ == "__main__":
    unittest.main()
