"""Aggregate independent ratings for the exact same generated replies."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review_store import RATING_COLUMNS, RATING_FIELDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rating_files", nargs="+", help="At least two independent rating CSV files")
    parser.add_argument("--output", default="data/evaluation/human_same_reply_ratings.csv")
    parser.add_argument("--report", default="results/human_reply_interrater_agreement.json")
    args = parser.parse_args()
    if len(args.rating_files) < 2:
        parser.error("at least two rating files are required")

    frames = []
    reviewer_ids = set()
    for file_name in args.rating_files:
        frame = pd.read_csv(ROOT / file_name, dtype={"reply_id": str}).fillna("")
        missing = set(RATING_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{file_name} is missing columns: {sorted(missing)}")
        ids = set(frame["reviewer_id"].astype(str))
        if len(ids) != 1:
            raise ValueError(f"{file_name} must contain exactly one reviewer ID")
        reviewer_id = next(iter(ids))
        if reviewer_id in reviewer_ids:
            raise ValueError("reviewer IDs must be independent and unique")
        reviewer_ids.add(reviewer_id)
        frames.append(frame[list(RATING_COLUMNS)].copy())

    common_ids = set.intersection(*(set(frame["reply_id"]) for frame in frames))
    aggregated = []
    for reply_id in sorted(common_ids):
        rows = [frame[frame["reply_id"] == reply_id].iloc[0] for frame in frames]
        if len({row["source_sha256"] for row in rows}) != 1:
            raise ValueError(f"source hash mismatch for reply_id {reply_id}")
        aggregated.append({
            "reply_id": reply_id,
            "source_sha256": rows[0]["source_sha256"],
            **{
                f"human_{field}": round(float(np.mean([
                    float(row[f"human_{field}"]) for row in rows
                ])), 3)
                for field in RATING_FIELDS
            },
            "reviewer_count": len(rows),
        })

    pairwise = []
    for left, right in combinations(frames, 2):
        merged = left.merge(right, on="reply_id", suffixes=("_left", "_right"))
        left_overall = merged[[f"human_{field}_left" for field in RATING_FIELDS]].astype(float).mean(axis=1)
        right_overall = merged[[f"human_{field}_right" for field in RATING_FIELDS]].astype(float).mean(axis=1)
        rho = float(spearmanr(left_overall, right_overall).statistic) if len(merged) > 1 else None
        pairwise.append({
            "reviewers": [str(left.iloc[0]["reviewer_id"]), str(right.iloc[0]["reviewer_id"])],
            "shared_replies": len(merged),
            "overall_spearman_rho": (
                round(rho, 4) if rho is not None and np.isfinite(rho) else None
            ),
            "quadratic_kappa_by_dimension": {
                field: (None if len(merged) < 2 else (
                    round(value, 4) if np.isfinite(value := float(cohen_kappa_score(
                        merged[f"human_{field}_left"].astype(int),
                        merged[f"human_{field}_right"].astype(int),
                        weights="quadratic",
                    ))) else None
                ))
                for field in RATING_FIELDS
            },
        })

    output = ROOT / args.output
    report_path = ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aggregated).to_csv(output, index=False)
    report = {
        "reviewers": sorted(reviewer_ids),
        "same_reply_count": len(common_ids),
        "minimum_recommended_same_reply_count": 40,
        "recommended_sample_met": len(common_ids) >= 40,
        "pairwise_agreement": pairwise,
        "judge_scores_visible_to_reviewers": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
