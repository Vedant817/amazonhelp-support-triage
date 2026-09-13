"""Evaluate the local rubric judge on exact human-rated replies and system drafts.

The run is resumable: every valid score is appended to a checkpoint CSV immediately.
No hosted model or API key is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence_integrity import verify_evaluation_manifest
from src.llm_judge import JudgeOutputError, ReplyQualityJudge, RUBRIC_FIELDS
from src.local_llm import LocalInstructionModel, QWEN_MODEL_ID, QWEN_REVISION
from src.review_store import reply_source_hash
from src.supervision import infer_resolution_act, strip_agent_pii


def _stratified_sample(gold: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit < gold["gold_intent"].nunique():
        raise ValueError("limit must cover every intent")
    per_class = limit // gold["gold_intent"].nunique()
    selected = gold.groupby("gold_intent", group_keys=False).sample(
        n=per_class, random_state=seed
    )
    remaining = gold.loc[~gold.index.isin(selected.index)]
    extra = remaining.sample(n=limit - len(selected), random_state=seed) if len(selected) < limit else remaining.head(0)
    return pd.concat([selected, extra]).sort_values("id").reset_index(drop=True)


def _read_checkpoint(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame()


def _save_checkpoint(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def _agreement(historical: pd.DataFrame) -> dict[str, Any]:
    by_dimension: dict[str, Any] = {}
    for field in RUBRIC_FIELDS:
        by_dimension[field] = ReplyQualityJudge.compute_human_agreement(
            historical[f"human_{field}"].astype(float).tolist(),
            historical[f"judge_{field}"].astype(float).tolist(),
        )
    human_overall = historical[[f"human_{field}" for field in RUBRIC_FIELDS]].astype(float).mean(axis=1)
    judge_overall = historical[[f"judge_{field}" for field in RUBRIC_FIELDS]].astype(float).mean(axis=1)
    return {
        "overall": ReplyQualityJudge.compute_human_agreement(
            human_overall.tolist(), judge_overall.tolist()
        ),
        "by_dimension": by_dimension,
    }


def _largest_disagreements(historical: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    scored = historical.copy()
    scored["human_overall"] = scored[
        [f"human_{field}" for field in RUBRIC_FIELDS]
    ].astype(float).mean(axis=1)
    scored["judge_overall"] = scored[
        [f"judge_{field}" for field in RUBRIC_FIELDS]
    ].astype(float).mean(axis=1)
    scored["absolute_gap"] = (scored["human_overall"] - scored["judge_overall"]).abs()
    columns = ["gold_id", "human_overall", "judge_overall", "absolute_gap"]
    return scored.sort_values(
        ["absolute_gap", "gold_id"], ascending=[False, True]
    ).head(limit)[columns].round(3).to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--historical-output", default="results/local_judge_historical_alignment.csv")
    parser.add_argument("--system-output", default="results/local_judge_system_scores.csv")
    parser.add_argument("--report", default="results/local_judge_evaluation.json")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    verify_evaluation_manifest(ROOT / "data/golden/MANIFEST.json")
    gold = pd.read_csv(ROOT / "data/golden/golden_eval_set.csv")
    predictions = pd.read_csv(ROOT / "results/golden_predictions.csv")
    sample = _stratified_sample(gold, min(args.limit, len(gold)), args.seed)
    prediction_map = predictions.set_index("gold_id")
    historical_path = ROOT / args.historical_output
    system_path = ROOT / args.system_output
    historical_records = _read_checkpoint(historical_path).to_dict("records")
    system_records = _read_checkpoint(system_path).to_dict("records")
    historical_done = {str(row["gold_id"]) for row in historical_records}
    system_done = {str(row["gold_id"]) for row in system_records}
    judge = ReplyQualityJudge(LocalInstructionModel())
    invalid_historical: list[str] = []
    invalid_system: list[str] = []

    pending: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        gold_id = str(row["id"])
        if gold_id not in historical_done:
            pending.append({
                "kind": "historical", "gold_id": gold_id, "row": row,
                "customer_tweet": row["customer_text"],
                "generated_reply": strip_agent_pii(row["reference_reply"]),
                "intent": row["gold_intent"],
                "should_escalate": bool(row["gold_escalate"]),
                "resolution_act": infer_resolution_act(row["reference_reply"]),
                "exemplars": [],
            })
        if gold_id not in system_done:
            predicted = prediction_map.loc[gold_id]
            pending.append({
                "kind": "system", "gold_id": gold_id, "row": row,
                "predicted": predicted,
                "customer_tweet": row["customer_text"],
                "generated_reply": predicted["draft_reply"],
                "intent": predicted["predicted_intent"],
                "should_escalate": bool(predicted["predicted_escalate"]),
                "resolution_act": predicted["resolution_act"],
                "exemplars": [],
            })

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        try:
            scores = judge.score_replies(chunk, batch_size=args.batch_size)
        except JudgeOutputError:
            scores = []
            for case in chunk:
                try:
                    scores.append(judge.score_reply(
                        case["customer_tweet"], case["generated_reply"],
                        case["intent"], case["should_escalate"],
                        case["resolution_act"], [],
                    ))
                except JudgeOutputError:
                    scores.append(None)
        for case, score in zip(chunk, scores):
            gold_id = case["gold_id"]
            row = case["row"]
            if case["kind"] == "historical":
                if score is None:
                    invalid_historical.append(gold_id)
                    continue
                historical_records.append({
                    "gold_id": gold_id,
                    "reply_id": f"{gold_id}:historical-v1",
                    "source_sha256": reply_source_hash(
                        f"{gold_id}:historical-v1", case["customer_tweet"],
                        case["generated_reply"], case["resolution_act"],
                    ),
                    "intent": row["gold_intent"],
                    "resolution_act": case["resolution_act"],
                    **{f"human_{field}": row[f"human_{field}_score"] for field in RUBRIC_FIELDS},
                    **{f"judge_{field}": score[field] for field in RUBRIC_FIELDS},
                    "schema_repaired": score["schema_repaired"],
                    "judge_rationale": score["rationale"],
                })
                _save_checkpoint(historical_path, historical_records)
            else:
                if score is None:
                    invalid_system.append(gold_id)
                    continue
                predicted = case["predicted"]
                system_records.append({
                    "gold_id": gold_id,
                    "reply_id": str(predicted["reply_id"]),
                    "source_sha256": reply_source_hash(
                        str(predicted["reply_id"]), case["customer_tweet"],
                        case["generated_reply"], case["resolution_act"],
                    ),
                    "intent": predicted["predicted_intent"],
                    "resolution_act": predicted["resolution_act"],
                    **{f"judge_{field}": score[field] for field in RUBRIC_FIELDS},
                    "schema_repaired": score["schema_repaired"],
                    "judge_rationale": score["rationale"],
                })
                _save_checkpoint(system_path, system_records)
        print(
            f"batch {start // args.batch_size + 1}: historical={len(historical_records)}/{len(sample)} "
            f"system={len(system_records)}/{len(sample)}",
            flush=True,
        )

    historical = pd.DataFrame(historical_records)
    system = pd.DataFrame(system_records)
    sample_ids = set(sample["id"].astype(str))
    historical = historical[historical["gold_id"].astype(str).isin(sample_ids)]
    system = system[system["gold_id"].astype(str).isin(sample_ids)]
    valid_for_agreement = len(historical)
    report = {
        "judge_model": QWEN_MODEL_ID,
        "judge_revision": QWEN_REVISION,
        "hosted_api_required": False,
        "sampling": "stratified across all six intents before deterministic remainder sampling",
        "requested_sample_size": len(sample),
        "human_annotation_source": "existing single-annotator scores on historical reference replies",
        "human_annotation_limitations": (
            "annotator identity and repeat-rating protocol are not recorded; safety scores have zero variance"
        ),
        "same_reply_alignment": True,
        "content_hashes_recorded": True,
        "historical_valid_count": valid_for_agreement,
        "historical_invalid_ids": invalid_historical,
        "agreement_status": "measured" if valid_for_agreement >= 40 else "insufficient_valid_outputs",
        "human_judge_agreement": _agreement(historical) if valid_for_agreement else None,
        "largest_same_reply_disagreements": (
            _largest_disagreements(historical) if valid_for_agreement else []
        ),
        "system_valid_count": len(system),
        "system_invalid_ids": invalid_system,
        "system_judge_means": {
            field: round(float(system[f"judge_{field}"].astype(float).mean()), 3)
            for field in RUBRIC_FIELDS
        } if len(system) else None,
        "system_judge_overall_mean": round(float(
            system[[f"judge_{field}" for field in RUBRIC_FIELDS]]
            .astype(float).mean(axis=1).mean()
        ), 3) if len(system) else None,
        "judge_score_usable_as_headline": False,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if valid_for_agreement >= 40 and len(system) >= 40 else 1


if __name__ == "__main__":
    raise SystemExit(main())
