"""Create a consensus training file from two or more independent review exports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review_store import REVIEW_COLUMNS


LABEL_COLUMNS = (
    "reviewed_intent", "reviewed_escalate", "reviewed_reason_code",
    "reviewed_resolution_act", "ambiguous",
)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _safe_kappa(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2:
        return None
    value = float(cohen_kappa_score(left, right))
    return round(value, 4) if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_files", nargs="+", help="At least two independent reviewer CSV files")
    parser.add_argument("--output", default="data/training/human_consensus_labels.csv")
    parser.add_argument("--disagreements", default="data/reviews/review_disagreements.csv")
    parser.add_argument("--report", default="results/human_label_agreement.json")
    args = parser.parse_args()
    if len(args.review_files) < 2:
        parser.error("at least two reviewer files are required")

    frames = []
    reviewer_ids = set()
    for file_name in args.review_files:
        frame = pd.read_csv(ROOT / file_name, dtype={"inbound_id": str}).fillna("")
        missing = set(REVIEW_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{file_name} is missing columns: {sorted(missing)}")
        ids = set(frame["reviewer_id"].astype(str))
        if len(ids) != 1:
            raise ValueError(f"{file_name} must contain exactly one reviewer ID")
        reviewer_id = next(iter(ids))
        if reviewer_id in reviewer_ids:
            raise ValueError("reviewer IDs must be independent and unique")
        reviewer_ids.add(reviewer_id)
        frames.append(frame[list(REVIEW_COLUMNS)].copy())

    common_ids = set.intersection(*(set(frame["inbound_id"]) for frame in frames))
    records = []
    disagreements = []
    for inbound_id in sorted(common_ids):
        rows = [frame[frame["inbound_id"] == inbound_id].iloc[0] for frame in frames]
        if len({row["source_sha256"] for row in rows}) != 1:
            raise ValueError(f"source hash mismatch for inbound_id {inbound_id}")
        unanimous = all(len({str(row[column]) for row in rows}) == 1 for column in LABEL_COLUMNS)
        base = {
            "inbound_id": inbound_id,
            "source_sha256": rows[0]["source_sha256"],
            "reviewer_count": len(rows),
        }
        if unanimous and not _as_bool(rows[0]["ambiguous"]):
            records.append({
                **base,
                "reviewed_intent": rows[0]["reviewed_intent"],
                "reviewed_escalate": rows[0]["reviewed_escalate"],
                "reviewed_reason_code": rows[0]["reviewed_reason_code"],
                "reviewed_resolution_act": rows[0]["reviewed_resolution_act"],
                "ambiguous": False,
                "review_notes": "independent exact consensus",
            })
        else:
            disagreements.append({
                **base,
                **{
                    f"{column}_votes": " | ".join(str(row[column]) for row in rows)
                    for column in LABEL_COLUMNS
                },
                "adjudicated_intent": "",
                "adjudicated_escalate": "",
                "adjudicated_reason_code": "",
                "adjudicated_resolution_act": "",
                "adjudication_notes": "",
            })

    pairwise = []
    for left, right in combinations(frames, 2):
        merged = left.merge(right, on="inbound_id", suffixes=("_left", "_right"))
        pairwise.append({
            "reviewers": [str(left.iloc[0]["reviewer_id"]), str(right.iloc[0]["reviewer_id"])],
            "shared_cases": len(merged),
            "intent_kappa": _safe_kappa(
                merged["reviewed_intent_left"], merged["reviewed_intent_right"]
            ),
            "escalation_kappa": _safe_kappa(
                merged["reviewed_escalate_left"].astype(str),
                merged["reviewed_escalate_right"].astype(str),
            ),
            "exact_all_label_agreement": round(float(
                pd.concat([
                    merged[f"{column}_left"].astype(str).eq(
                        merged[f"{column}_right"].astype(str)
                    ) for column in LABEL_COLUMNS
                ], axis=1).all(axis=1).mean()
            ), 4),
        })

    output = ROOT / args.output
    disagreement_path = ROOT / args.disagreements
    report_path = ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    disagreement_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    pd.DataFrame(disagreements).to_csv(disagreement_path, index=False)
    report = {
        "reviewers": sorted(reviewer_ids),
        "shared_case_count": len(common_ids),
        "consensus_training_count": len(records),
        "disagreement_count": len(disagreements),
        "pairwise_agreement": pairwise,
        "gold_labels_visible_to_reviewers": False,
        "model_predictions_visible_to_reviewers": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
