"""Tests for the evaluation protocols that produce reported numbers.

These live in ``scripts_score.py`` rather than ``src/``, but they feed the score
matrix, so an error here produces plausible-looking and wrong claims. The two
that matter most:

* the temporal holdout must genuinely be the *newest* rows, otherwise it silently
  degenerates into a random split and stops testing drift at all;
* the escalation frontier must report the most precise threshold that still meets
  each recall floor, because the whole point is to show what recall costs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

from src.evaluator import SupportPipelineEvaluator  # noqa: E402


def _load_score_module():
    """Import scripts_score.py, whose name is not a valid identifier."""
    spec = importlib.util.spec_from_file_location("scripts_score", REPO_ROOT / "scripts_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def score_module():
    return _load_score_module()


class TestTemporalHoldout:
    def test_holdout_is_the_newest_rows(self, score_module):
        holdout = score_module._build_temporal_holdout(fraction=0.15)
        corpus = score_module._labelled_corpus()
        holdout_ids = set(holdout["inbound_id"].astype(str))
        training = corpus[~corpus["inbound_id"].astype(str).isin(holdout_ids)]

        holdout_time = score_module._parse_timestamps(holdout["customer_created_at"])
        training_time = score_module._parse_timestamps(training["customer_created_at"])
        # Every holdout row must be at least as recent as every training row.
        assert holdout_time.min() >= training_time.max()

    def test_fraction_is_respected(self, score_module):
        holdout = score_module._build_temporal_holdout(fraction=0.10)
        corpus = score_module._labelled_corpus()
        assert len(holdout) == pytest.approx(len(corpus) * 0.10, rel=0.02)

    def test_holdout_rows_are_unique(self, score_module):
        holdout = score_module._build_temporal_holdout(fraction=0.15)
        assert holdout["inbound_id"].astype(str).is_unique

    def test_summary_names_a_window(self, score_module):
        holdout = score_module._build_temporal_holdout(fraction=0.15)
        summary = score_module._temporal_summary(holdout)
        assert "temporal holdout" in summary
        assert any(character.isdigit() for character in summary)

    def test_explicit_timestamp_format_parses_cleanly(self, score_module):
        """The corpus format is Twitter-style; parsing must not fall back per row."""
        series = pd.Series(["Tue Oct 31 22:16:32 +0000 2017", "Wed Nov 01 12:50:18 +0000 2017"])
        parsed = score_module._parse_timestamps(series)
        assert parsed.notna().all()
        assert parsed.iloc[0] < parsed.iloc[1]
        assert str(parsed.dt.tz) == "UTC"


class TestRandomHoldout:
    def test_every_class_is_represented(self, score_module):
        holdout = score_module._build_random_holdout(seed=42, fraction=0.15)
        assert holdout["intent_label"].nunique() == 6

    def test_is_deterministic_for_a_seed(self, score_module):
        first = score_module._build_random_holdout(seed=7, fraction=0.15)
        second = score_module._build_random_holdout(seed=7, fraction=0.15)
        assert set(first["inbound_id"].astype(str)) == set(second["inbound_id"].astype(str))

    def test_fraction_controls_the_size(self, score_module):
        small = score_module._build_random_holdout(seed=42, fraction=0.05)
        large = score_module._build_random_holdout(seed=42, fraction=0.15)
        assert len(small) < len(large)

    def test_reason_codes_are_absent_so_the_report_says_na(self, score_module):
        holdout = score_module._build_random_holdout(seed=42, fraction=0.15)
        assert holdout["reviewed_reason_code"].isna().all()


class TestRowPredictionDump:
    def test_writes_a_row_per_evaluation_row(self, score_module, tmp_path):
        eval_frame = pd.DataFrame({"inbound_id": ["a", "b"]})
        rows = {
            "intent_true": ["X", "Y"], "intent_pred": ["X", "X"], "confidence": [0.9, 0.6],
            "esc_true": [True, None], "esc_pred": [True, False], "esc_risk": [0.8, 0.2],
        }
        path = score_module._write_row_predictions(str(tmp_path / "r.json"), eval_frame, rows)
        assert path is not None
        written = pd.read_csv(path)
        assert len(written) == 2
        assert written["inbound_id"].tolist() == ["a", "b"]

    def test_refuses_to_write_misaligned_rows(self, score_module, tmp_path):
        """A shorter collector output means positional pairing is invalid."""
        eval_frame = pd.DataFrame({"inbound_id": ["a", "b", "c"]})
        rows = {
            "intent_true": ["X", "Y"], "intent_pred": ["X", "X"], "confidence": [0.9, 0.6],
            "esc_true": [True, None], "esc_pred": [True, False], "esc_risk": [0.8, 0.2],
        }
        path = score_module._write_row_predictions(str(tmp_path / "r.json"), eval_frame, rows)
        assert path is None

    def test_missing_columns_are_not_fatal(self, score_module, tmp_path):
        path = score_module._write_row_predictions(
            str(tmp_path / "r.json"), pd.DataFrame({"inbound_id": ["a"]}), {}
        )
        assert path is None


class TestLeakGuard:
    """The exclusion list must be per-protocol and verified, not merely written.

    A single shared path is a correctness bug: two runs under different protocols
    overwrite each other's exclusion list, and whichever file is on disk when the
    pipeline is fitted decides what gets excluded. That leaked 1097 of 1268
    evaluation rows (86.5%) into their own training set and reported a 0.9259
    accuracy which was pure memorisation.
    """

    def test_paths_differ_between_protocols(self, score_module):
        random_path = score_module._holdout_ids_path("random", 42, 0.15)
        temporal_path = score_module._holdout_ids_path("temporal", 42, 0.15)
        review_path = score_module._holdout_ids_path("review", 42, 0.15)
        assert len({random_path, temporal_path, review_path}) == 3

    def test_random_path_encodes_seed_and_fraction(self, score_module):
        first = score_module._holdout_ids_path("random", 1, 0.10)
        second = score_module._holdout_ids_path("random", 2, 0.10)
        third = score_module._holdout_ids_path("random", 1, 0.20)
        assert len({first, second, third}) == 3

    def test_accepts_a_matching_guard(self, score_module, tmp_path):
        guard = str(tmp_path / "guard.csv")
        eval_frame = pd.DataFrame({"inbound_id": ["a", "b", "c"]})
        score_module._write_holdout_file(guard, eval_frame["inbound_id"])
        score_module._verify_leak_guard(guard, eval_frame, when="in test")

    def test_rejects_a_guard_missing_evaluation_rows(self, score_module, tmp_path):
        guard = str(tmp_path / "guard.csv")
        pd.DataFrame({"tweet_id": ["a"]}).to_csv(guard, index=False)
        eval_frame = pd.DataFrame({"inbound_id": ["a", "b", "c"]})
        with pytest.raises(RuntimeError, match="in the training set"):
            score_module._verify_leak_guard(guard, eval_frame, when="in test")

    def test_rejects_a_guard_from_another_protocol(self, score_module, tmp_path):
        """The exact failure that produced a contaminated 0.9259 score."""
        guard = str(tmp_path / "guard.csv")
        temporal_rows = pd.DataFrame({"inbound_id": ["x", "y", "z"]})
        score_module._write_holdout_file(guard, temporal_rows["inbound_id"])
        random_rows = pd.DataFrame({"inbound_id": ["a", "b", "c"]})
        with pytest.raises(RuntimeError):
            score_module._verify_leak_guard(guard, random_rows, when="in test")


class TestEscalationFrontier:
    def _scores(self, n: int = 400, seed: int = 0):
        rng = np.random.default_rng(seed)
        truth = rng.random(n) < 0.2
        # Scores separate the classes but overlap substantially.
        scores = np.clip(rng.normal(loc=np.where(truth, 0.7, 0.35), scale=0.2), 0.0, 1.0)
        return truth.tolist(), scores.tolist()

    def test_every_reached_row_meets_its_floor(self):
        truth, scores = self._scores()
        frontier = SupportPipelineEvaluator.precision_at_recall_frontier(truth, scores)
        for row in frontier:
            if row["reached"]:
                assert row["recall"] + 1e-9 >= row["recall_floor"]

    def test_precision_does_not_increase_with_recall(self):
        """Asking for more recall can only cost precision."""
        truth, scores = self._scores()
        frontier = SupportPipelineEvaluator.precision_at_recall_frontier(truth, scores)
        reached = [row for row in frontier if row["reached"]]
        precisions = [row["precision"] for row in reached]
        assert precisions == sorted(precisions, reverse=True)

    def test_higher_floors_need_lower_thresholds(self):
        """Recall floors are listed ascending, so thresholds must descend."""
        truth, scores = self._scores()
        frontier = SupportPipelineEvaluator.precision_at_recall_frontier(truth, scores)
        reached = [row for row in frontier if row["reached"]]
        thresholds = [row["threshold"] for row in reached]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_perfect_separation_reaches_every_floor(self):
        truth = [True] * 20 + [False] * 80
        scores = [0.9] * 20 + [0.1] * 80
        frontier = SupportPipelineEvaluator.precision_at_recall_frontier(truth, scores)
        assert all(row["reached"] for row in frontier)
        assert all(row["precision"] == 1.0 for row in frontier)

    def test_no_positives_returns_empty(self):
        assert SupportPipelineEvaluator.precision_at_recall_frontier([False] * 10, [0.1] * 10) == []
