import unittest
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.review_store import ReplyRatingStore, TrainingReviewStore
from scripts.adjudicate_reviews import main as adjudicate_main
from scripts.aggregate_reply_ratings import main as aggregate_ratings_main


class TestTrainingReviewStore(unittest.TestCase):
    def _store(self, directory: str) -> TrainingReviewStore:
        root = Path(directory)
        pd.DataFrame([
            {
                "inbound_id": "train-1",
                "conversation_context": "CUSTOMER: Earlier message",
                "customer_text": "Where is my package?",
                "reply_text": "What does tracking show?",
                "provisional_intent": "ACCOUNT_AND_SECURITY",
                "intent_label": "REFUND_AND_RETURN",
            },
            {
                "inbound_id": "train-2",
                "conversation_context": "",
                "customer_text": "My Kindle will not start",
                "reply_text": "Please restart it",
                "provisional_intent": "PRODUCT_AND_TECHNICAL",
                "intent_label": "PRODUCT_AND_TECHNICAL",
            },
        ]).to_csv(root / "queue.csv", index=False)
        pd.DataFrame([{"tweet_id": "gold-1"}]).to_csv(root / "gold.csv", index=False)
        return TrainingReviewStore(
            str(root / "queue.csv"), str(root / "reviews"), str(root / "gold.csv")
        )

    def test_review_item_hides_model_and_weak_labels(self):
        with TemporaryDirectory() as directory:
            item = self._store(directory).next_item("reviewer-a")
        self.assertEqual(item["inbound_id"], "train-1")
        self.assertNotIn("provisional_intent", item)
        self.assertNotIn("intent_label", item)
        self.assertNotIn("gold_intent", item)

    def test_review_save_is_separate_and_source_checked(self):
        with TemporaryDirectory() as directory:
            store = self._store(directory)
            item = store.next_item("reviewer-a")
            payload = {
                "inbound_id": item["inbound_id"],
                "source_sha256": item["source_sha256"],
                "reviewed_intent": "ORDER_TRACKING_AND_DELAY",
                "reviewed_escalate": False,
                "reviewed_reason_code": "ROUTINE_TRACKING_IN_TRANSIT",
                "reviewed_resolution_act": "ASK_TRACKING_STATUS",
                "ambiguous": False,
                "review_notes": "",
            }
            progress = store.save("reviewer-a", payload)
            self.assertEqual(progress["completed"], 1)
            self.assertTrue((Path(directory) / "reviews" / "training-review-reviewer-a.csv").exists())
            self.assertFalse((Path(directory) / "reviews" / "training-review-reviewer-b.csv").exists())
            payload["source_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                store.save("reviewer-b", payload)

    def test_review_rejects_reason_route_mismatch(self):
        with TemporaryDirectory() as directory:
            store = self._store(directory)
            item = store.next_item("reviewer-a")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                store.save("reviewer-a", {
                    "inbound_id": item["inbound_id"],
                    "source_sha256": item["source_sha256"],
                    "reviewed_intent": "ORDER_TRACKING_AND_DELAY",
                    "reviewed_escalate": False,
                    "reviewed_reason_code": "HIGH_HOSTILITY_OR_LEGAL_RISK",
                    "reviewed_resolution_act": "LINK_ORDERS",
                    "ambiguous": False,
                    "review_notes": "",
                })

    def test_review_queue_rejects_gold_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{
                "inbound_id": "same", "conversation_context": "",
                "customer_text": "help", "reply_text": "reply",
            }]).to_csv(root / "queue.csv", index=False)
            pd.DataFrame([{"tweet_id": "same"}]).to_csv(root / "gold.csv", index=False)
            with self.assertRaises(ValueError):
                TrainingReviewStore(
                    str(root / "queue.csv"), str(root / "reviews"), str(root / "gold.csv")
                )

    def test_reply_ratings_hide_judge_scores_and_persist_exact_reply_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{
                "reply_id": "reply-1",
                "customer_text": "Where is it?",
                "draft_reply": "Check Your Orders.",
                "resolution_act": "LINK_ORDERS",
                "judge_groundedness": 5,
            }]).to_csv(root / "ratings.csv", index=False)
            store = ReplyRatingStore(str(root / "ratings.csv"), str(root / "reviews"))
            item = store.next_item("reviewer-a")
            self.assertNotIn("judge_groundedness", item)
            progress = store.save("reviewer-a", {
                "reply_id": item["reply_id"],
                "source_sha256": item["source_sha256"],
                "human_groundedness": 4,
                "human_empathy": 3,
                "human_actionability": 4,
                "human_safety": 5,
                "human_notes": "",
            })
            self.assertEqual(progress["completed"], 1)

    def test_independent_consensus_and_rating_aggregation_scripts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            review_rows = []
            for reviewer in ("reviewer-a", "reviewer-b"):
                review_rows.append({
                    "inbound_id": "train-1", "source_sha256": "a" * 64,
                    "reviewer_id": reviewer,
                    "reviewed_intent": "ORDER_TRACKING_AND_DELAY",
                    "reviewed_escalate": False,
                    "reviewed_reason_code": "ROUTINE_TRACKING_IN_TRANSIT",
                    "reviewed_resolution_act": "ASK_TRACKING_STATUS",
                    "ambiguous": False, "review_notes": "",
                    "reviewed_at_utc": "2026-01-01T00:00:00+00:00",
                })
            review_files = []
            for row in review_rows:
                path = root / f"{row['reviewer_id']}.csv"
                pd.DataFrame([row]).to_csv(path, index=False)
                review_files.append(path)
            consensus = root / "consensus.csv"
            disagreements = root / "disagreements.csv"
            agreement = root / "agreement.json"
            with patch.object(sys, "argv", [
                "adjudicate_reviews.py", *(str(path) for path in review_files),
                "--output", str(consensus), "--disagreements", str(disagreements),
                "--report", str(agreement),
            ]):
                self.assertEqual(adjudicate_main(), 0)
            self.assertEqual(len(pd.read_csv(consensus)), 1)

            rating_files = []
            for reviewer, score in (("reviewer-a", 4), ("reviewer-b", 5)):
                path = root / f"rating-{reviewer}.csv"
                pd.DataFrame([{
                    "reply_id": "reply-1", "source_sha256": "b" * 64,
                    "reviewer_id": reviewer,
                    "human_groundedness": score, "human_empathy": score,
                    "human_actionability": score, "human_safety": score,
                    "human_notes": "", "reviewed_at_utc": "2026-01-01T00:00:00+00:00",
                }]).to_csv(path, index=False)
                rating_files.append(path)
            ratings = root / "ratings.csv"
            rating_report = root / "rating-report.json"
            with patch.object(sys, "argv", [
                "aggregate_reply_ratings.py", *(str(path) for path in rating_files),
                "--output", str(ratings), "--report", str(rating_report),
            ]):
                self.assertEqual(aggregate_ratings_main(), 0)
            aggregated = pd.read_csv(ratings)
            self.assertEqual(aggregated.iloc[0]["source_sha256"], "b" * 64)
            self.assertFalse(json.loads(rating_report.read_text())["recommended_sample_met"])


if __name__ == "__main__":
    unittest.main()
