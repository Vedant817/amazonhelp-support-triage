import json
import unittest
from pathlib import Path

import pandas as pd

from src.evidence_integrity import verify_evaluation_manifest


ROOT = Path(__file__).resolve().parents[1]


class TestEvidenceArtifacts(unittest.TestCase):
    def test_frozen_gold_manifest_matches_files(self):
        verified = verify_evaluation_manifest(ROOT / "data/golden/MANIFEST.json")
        self.assertEqual(set(verified), {"golden_eval_set.csv", "golden_eval_set.jsonl"})
    def test_gold_csv_and_jsonl_are_aligned(self):
        csv = pd.read_csv(ROOT / "data/golden/golden_eval_set.csv")
        with (ROOT / "data/golden/golden_eval_set.jsonl").open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        self.assertEqual(len(csv), 200)
        self.assertEqual(len(records), 200)
        json_by_id = {row["id"]: row for row in records}
        for _, row in csv.iterrows():
            other = json_by_id[row["id"]]
            self.assertEqual(bool(row["gold_escalate"]), bool(other["gold_escalate"]))
            self.assertEqual(row["gold_escalate_reason"], other["gold_escalate_reason"])

    def test_review_sheet_is_400_rows_and_leak_free(self):
        review = pd.read_csv(ROOT / "data/training/labels_for_human_review.csv")
        gold = pd.read_csv(ROOT / "data/golden/golden_eval_set.csv")
        self.assertEqual(len(review), 400)
        self.assertTrue(set(review["inbound_id"].astype(str)).isdisjoint(set(gold["tweet_id"].astype(str))))
        self.assertEqual(int(review["reviewed_intent"].notna().sum()), 0)
        self.assertIn("conversation_context", review.columns)
        self.assertIn("prior_turn_count", review.columns)
        self.assertGreater(float((review["prior_turn_count"] > 0).mean()), 0.0)

    def test_saved_results_have_row_level_evidence(self):
        with (ROOT / "results/headline_results.json").open(encoding="utf-8") as handle:
            results = json.load(handle)
        predictions = pd.read_csv(ROOT / "results/golden_predictions.csv")
        self.assertEqual(results["run_metadata"]["gold_size"], 200)
        self.assertTrue(results["run_metadata"]["gold_ids_excluded_from_training_and_retrieval"])
        self.assertEqual(len(predictions), 200)
        self.assertIn("intent_calibration", results)
        self.assertIn("escalation_risk_score", predictions.columns)
        self.assertGreater(results["run_metadata"]["weak_intent_labels_pruned"], 0)
        self.assertTrue(results["run_metadata"]["context_is_causally_prior_only"])
        self.assertGreater(results["run_metadata"]["historical_context_coverage"], 0.0)
        self.assertEqual(results["run_metadata"]["escalation_selected_c"], 0.25)
        self.assertIn("model_input", predictions.columns)
        self.assertIn("context_used", predictions.columns)
        zero_shot_status = results["run_metadata"]["zero_shot_local_baseline"]
        if zero_shot_status == "run":
            self.assertIn("Baseline 3: zero-shot local Qwen", results["systems"])
            self.assertIn(
                "invalid_structured_outputs",
                results["systems"]["Baseline 3: zero-shot local Qwen"],
            )
        else:
            self.assertEqual(zero_shot_status, "not_run")
            self.assertNotIn("Baseline 3: zero-shot local Qwen", results["systems"])
        self.assertEqual(results["local_llm_judge"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
