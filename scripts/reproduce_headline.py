"""Reproduce leak-free metrics for the dual-head support system.

Offline is the default. It may attach an existing local-judge result only when every
content hash matches the current replies; otherwise the judge remains unavailable.
Pass --generation-mode local and --judge-local to run the pinned local Qwen model.
Human agreement is calculated only from ratings keyed to identical reply content.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier import INTENT_TAXONOMY, MajorityBaselineClassifier, TfidfBaselineClassifier
from src.escalator import SafetyGatedEscalator, SimpleKeywordEscalationBaseline, TrivialEscalationBaseline
from src.evaluator import SupportPipelineEvaluator
from src.evidence_integrity import verify_evaluation_manifest
from src.generator import ProductionGroundedGenerator, RetrievalReplayBaseline, TrivialReplyBaseline
from src.guardrails import ProductionPolicyGuard
from src.llm_judge import JudgeOutputError, ReplyQualityJudge, RUBRIC_FIELDS
from src.review_store import reply_source_hash
from src.local_llm import LocalInstructionModel, QWEN_MODEL_ID, QWEN_REVISION
from src.model_factory import build_learned_pipeline
from src.supervision import infer_resolution_act, strip_agent_pii
from src.zero_shot import ZeroShotLocalBaseline


PROPOSED = "Proposed: MiniLM dual-head + safety gates + retrieve/rewrite"


def _pii_echo(customer: str, reply: str) -> bool:
    patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b\d{3}-\d{7}-\d{7}\b",
        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    ]
    return any(any(value in reply for value in re.findall(pattern, customer)) for pattern in patterns)


def _new_system() -> Dict[str, Any]:
    return {"intent_preds": [], "esc_preds": [], "esc_reasons": [], "replies": [], "acts": [], "exemplars": [], "context_used": [], "seconds": 0.0, "invalid_outputs": 0}


def _evaluate_system(
    evaluator: SupportPipelineEvaluator,
    gold: pd.DataFrame,
    data: Dict[str, Any],
    bootstrap_iterations: int,
) -> Dict[str, Any]:
    intents = evaluator.evaluate_intents(gold["gold_intent"].tolist(), data["intent_preds"], labels=INTENT_TAXONOMY)
    escalation = evaluator.evaluate_escalation(
        gold["gold_escalate"].astype(bool).tolist(),
        data["esc_preds"],
        gold["gold_escalate_reason"].tolist(),
        data["esc_reasons"],
    )
    return {
        "intent_metrics": intents,
        "escalation_metrics": escalation,
        "bootstrap_95_ci": {
            metric: evaluator.bootstrap_classification_intervals(
                gold["gold_intent"].tolist() if metric == "macro_f1" else gold["gold_escalate"].astype(bool).tolist(),
                data["intent_preds"] if metric == "macro_f1" else data["esc_preds"],
                metric,
                iterations=bootstrap_iterations,
            )
            for metric in ("macro_f1", "escalate_recall", "escalate_precision")
        },
        "reply_safety": {
            "twitter_length_compliance": round(float(np.mean([len(reply) <= 280 for reply in data["replies"]])), 4),
            "no_pii_echo_rate": round(float(np.mean([
                not _pii_echo(customer, reply)
                for customer, reply in zip(gold["customer_text"], data["replies"])
            ])), 4),
        },
        "mean_latency_ms": round(data["seconds"] / len(gold) * 1000.0, 2),
        "invalid_structured_outputs": int(data.get("invalid_outputs", 0)),
    }


def _same_reply_agreement(judge_rows: pd.DataFrame, ratings_path: Path) -> Dict[str, Any]:
    if not ratings_path.exists():
        return {"status": "not_available", "reason": "same-reply human ratings file not found"}
    ratings = pd.read_csv(ratings_path)
    fields = ["groundedness", "empathy", "actionability", "safety"]
    human_fields = [f"human_{field}" for field in fields]
    required = {"reply_id", "source_sha256", *human_fields}
    if required - set(ratings.columns):
        return {"status": "not_available", "reason": "ratings file is missing required columns"}
    merged = judge_rows.merge(
        ratings[["reply_id", "source_sha256", *human_fields]],
        on="reply_id", how="inner", suffixes=("_judge", "_human"),
    )
    merged = merged[
        merged["source_sha256_judge"] == merged["source_sha256_human"]
    ]
    merged = merged.dropna(subset=human_fields)
    if merged.empty:
        return {"status": "not_available", "reason": "no completed ratings match judged reply IDs"}
    if len(merged) < 40:
        return {
            "status": "not_available",
            "reason": "fewer than 40 exact same-reply human ratings match judge outputs",
            "matching_reply_count": len(merged),
        }
    human = merged[human_fields].mean(axis=1).tolist()
    judged = merged[[f"judge_{field}" for field in fields]].mean(axis=1).tolist()
    result = ReplyQualityJudge.compute_human_agreement(human, judged)
    result["status"] = "measured"
    return result


def _cached_judge_evidence(
    gold: pd.DataFrame,
    predictions: pd.DataFrame,
    report_path: Path = Path("results/local_judge_evaluation.json"),
    historical_path: Path = Path("results/local_judge_historical_alignment.csv"),
    system_path: Path = Path("results/local_judge_system_scores.csv"),
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load only content-addressed, same-reply judge evidence for current drafts."""
    unavailable = (
        {"status": "not_run"},
        {"status": "not_available", "reason": "validated local judge artifact not found"},
    )
    if not all(path.exists() for path in (report_path, historical_path, system_path)):
        return unavailable
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        historical = pd.read_csv(historical_path)
        system = pd.read_csv(system_path)
        if (
            report.get("judge_model") != QWEN_MODEL_ID
            or report.get("judge_revision") != QWEN_REVISION
            or not report.get("same_reply_alignment")
            or len(historical) < 40
            or len(system) < 40
        ):
            raise ValueError("judge metadata or sample size is invalid")
        required = {"gold_id", "reply_id", "source_sha256", *[f"judge_{x}" for x in RUBRIC_FIELDS]}
        if required - set(historical.columns) or required - set(system.columns):
            raise ValueError("judge rows lack required provenance or score columns")
        if historical["gold_id"].astype(str).duplicated().any() or system["gold_id"].astype(str).duplicated().any():
            raise ValueError("judge rows contain duplicate gold IDs")

        gold_by_id = gold.assign(id=gold["id"].astype(str)).set_index("id")
        predictions_by_id = predictions.assign(gold_id=predictions["gold_id"].astype(str)).set_index("gold_id")
        for _, row in historical.iterrows():
            item = gold_by_id.loc[str(row["gold_id"])]
            reply_id = f"{row['gold_id']}:historical-v1"
            act = infer_resolution_act(item["reference_reply"])
            expected = reply_source_hash(
                reply_id, item["customer_text"], strip_agent_pii(item["reference_reply"]), act
            )
            if row["reply_id"] != reply_id or row["source_sha256"] != expected:
                raise ValueError("historical judge evidence hash mismatch")
        for _, row in system.iterrows():
            item = predictions_by_id.loc[str(row["gold_id"])]
            expected = reply_source_hash(
                item["reply_id"], item["customer_text"], item["draft_reply"], item["resolution_act"]
            )
            if row["reply_id"] != item["reply_id"] or row["source_sha256"] != expected:
                raise ValueError("system judge evidence is stale for current drafts")

        by_dimension = {}
        aligned_human_overall = []
        aligned_judge_overall = []
        for field in RUBRIC_FIELDS:
            human = [float(gold_by_id.loc[str(row.gold_id), f"human_{field}_score"]) for row in historical.itertuples()]
            judged = historical[f"judge_{field}"].astype(float).tolist()
            by_dimension[field] = ReplyQualityJudge.compute_human_agreement(human, judged)
        for row in historical.itertuples():
            item = gold_by_id.loc[str(row.gold_id)]
            aligned_human_overall.append(float(np.mean([item[f"human_{field}_score"] for field in RUBRIC_FIELDS])))
            aligned_judge_overall.append(float(np.mean([getattr(row, f"judge_{field}") for field in RUBRIC_FIELDS])))
        agreement = ReplyQualityJudge.compute_human_agreement(
            aligned_human_overall, aligned_judge_overall
        )
        agreement.update({
            "status": "measured_same_reply",
            "by_dimension": by_dimension,
            "annotation_limitations": report.get("human_annotation_limitations"),
        })
        overall = system[[f"judge_{field}" for field in RUBRIC_FIELDS]].astype(float).mean(axis=1)
        judge = {
            "status": "measured_cached_content_verified",
            "valid_sample_size": int(len(system)),
            "mean_overall": round(float(overall.mean()), 3),
            "model": QWEN_MODEL_ID,
            "revision": QWEN_REVISION,
            "hosted_api_required": False,
            "usable_as_headline": False,
        }
        return judge, agreement
    except (KeyError, ValueError, TypeError, json.JSONDecodeError, pd.errors.ParserError) as error:
        return (
            {"status": "stale_or_invalid", "reason": str(error)},
            {"status": "not_available", "reason": "cached judge evidence failed validation"},
        )


def run(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    gold_hashes = verify_evaluation_manifest("data/golden/MANIFEST.json")
    gold = pd.read_csv("data/golden/golden_eval_set.csv")
    artifacts = build_learned_pipeline(
        "data/processed/amazonhelp_context_pairs.csv",
        "data/golden/golden_eval_set.csv",
        reviewed_csv=args.reviewed_labels,
        intent_cap_per_class=args.intent_cap_per_class,
        retrieval_size=args.retrieval_size,
        contrastive_pairs=args.contrastive_pairs,
        escalation_knn=args.escalation_knn,
    )
    model = artifacts.model
    supervision = artifacts.supervision
    intent_rows = supervision[supervision["intent_label"].notna()]

    majority = MajorityBaselineClassifier().fit(
        intent_rows["customer_text"].tolist(), intent_rows["intent_label"].tolist()
    )
    tfidf = TfidfBaselineClassifier().fit(
        intent_rows["customer_text"].tolist(), intent_rows["intent_label"].tolist()
    )
    always_escalate = TrivialEscalationBaseline()
    keyword_escalate = SimpleKeywordEscalationBaseline()
    learned_escalator = SafetyGatedEscalator(model.escalation_threshold_)
    trivial_reply = TrivialReplyBaseline()
    retrieved_reply = RetrievalReplayBaseline()
    local_runtime = LocalInstructionModel() if args.generation_mode == "local" or args.judge_local or args.include_zero_shot_local else None
    generator = ProductionGroundedGenerator(mode=args.generation_mode, local_model=local_runtime)
    zero_shot = ZeroShotLocalBaseline(local_runtime) if args.include_zero_shot_local else None
    guard = ProductionPolicyGuard()
    evaluator = SupportPipelineEvaluator()

    systems = {
        "Baseline 1: majority + always escalate + canned": _new_system(),
        "Baseline 2: TF-IDF + keyword escalation + retrieve/replay": _new_system(),
        PROPOSED: _new_system(),
    }
    if zero_shot is not None:
        systems["Baseline 3: zero-shot local Qwen"] = _new_system()
    confidences: List[float] = []
    risk_scores: List[float] = []

    for _, row in gold.iterrows():
        text = row["customer_text"]
        model_text = artifacts.context_by_id.get(str(row["tweet_id"]), text)

        started = time.time()
        intent = majority.predict([text])[0]
        decision = always_escalate.evaluate(text)
        systems["Baseline 1: majority + always escalate + canned"]["intent_preds"].append(intent)
        systems["Baseline 1: majority + always escalate + canned"]["esc_preds"].append(decision.should_escalate)
        systems["Baseline 1: majority + always escalate + canned"]["esc_reasons"].append(decision.stated_reason)
        systems["Baseline 1: majority + always escalate + canned"]["replies"].append(trivial_reply.generate(text, intent, []))
        systems["Baseline 1: majority + always escalate + canned"]["seconds"] += time.time() - started

        started = time.time()
        intent = tfidf.predict([text])[0]
        decision = keyword_escalate.evaluate(text)
        exemplars = artifacts.retriever.retrieve_similar(text, intent_filter=intent, top_k=1)
        reply = retrieved_reply.generate(text, intent, exemplars)
        systems["Baseline 2: TF-IDF + keyword escalation + retrieve/replay"]["intent_preds"].append(intent)
        systems["Baseline 2: TF-IDF + keyword escalation + retrieve/replay"]["esc_preds"].append(decision.should_escalate)
        systems["Baseline 2: TF-IDF + keyword escalation + retrieve/replay"]["esc_reasons"].append(decision.stated_reason)
        systems["Baseline 2: TF-IDF + keyword escalation + retrieve/replay"]["replies"].append(reply)
        systems["Baseline 2: TF-IDF + keyword escalation + retrieve/replay"]["seconds"] += time.time() - started

        if zero_shot is not None:
            started = time.time()
            zero = zero_shot.predict(text)
            reply = guard.verify_and_sanitize(zero["reply"], text, zero["should_escalate"]).sanitized_reply
            target = systems["Baseline 3: zero-shot local Qwen"]
            target["intent_preds"].append(zero["intent"])
            target["esc_preds"].append(zero["should_escalate"])
            target["esc_reasons"].append(zero["stated_reason"])
            target["replies"].append(reply)
            target["invalid_outputs"] += int(not zero["parse_valid"])
            target["seconds"] += time.time() - started

        started = time.time()
        classification = model.predict_with_confidence(text, model_text)
        current_embedding = classification["current_message_embedding"]
        learned = model.predict_escalation(text, current_embedding)
        reason_prediction = model.predict_reason(text, current_embedding)
        retrieval_query = model_text if classification["context_used"] else text
        exemplars = artifacts.retriever.retrieve_similar(
            retrieval_query, intent_filter=classification["intent"], top_k=12
        )
        evidence_act = generator.select_act(exemplars, classification["intent"], False)
        decision = learned_escalator.evaluate(
            text,
            classification["intent"],
            learned["risk_score"],
            classification["is_uncertain"],
            learned["threshold"],
            model.high_precision_escalation_threshold_,
            evidence_act,
            (reason_prediction or {}).get("reason"),
        )
        act = generator.select_act(exemplars, classification["intent"], decision.should_escalate)
        raw = generator.generate(text, classification["intent"], exemplars, decision.to_dict())
        reply = guard.verify_and_sanitize(raw, text, decision.should_escalate).sanitized_reply
        target = systems[PROPOSED]
        target["intent_preds"].append(classification["intent"])
        target["context_used"].append(classification["context_used"])
        target["esc_preds"].append(decision.should_escalate)
        target["esc_reasons"].append(decision.stated_reason)
        target["replies"].append(reply)
        target["acts"].append(act)
        target["exemplars"].append(exemplars[:3])
        target["seconds"] += time.time() - started
        confidences.append(classification["confidence"])
        risk_scores.append(learned["risk_score"])

    full_results = {
        "run_metadata": {
            "gold_size": len(gold),
            "gold_label_audit": "data/golden/LABEL_AUDIT.md",
            "gold_manifest": "data/golden/MANIFEST.json",
            "gold_sha256": gold_hashes,
            "gold_ids_excluded_from_training_and_retrieval": True,
            "historical_context_coverage": round(artifacts.context_coverage, 6),
            "context_is_causally_prior_only": True,
            "trained_pipeline_cache_hit": artifacts.cache_hit,
            "context_gate_confidence": round(model.context_gate_confidence_, 6),
            "context_gate_margin": round(model.context_gate_margin_, 6),
            "context_gate_calibration_macro_f1_gain": round(
                model.context_gate_calibration_gain_, 6
            ),
            "prototype_blend_weight": round(model.prototype_blend_weight_, 6),
            "prototype_temperature": round(model.prototype_temperature_, 6),
            "prototype_calibration_macro_f1_gain": round(
                model.prototype_calibration_gain_, 6
            ),
            "semantic_blend_weight": round(model.semantic_blend_weight_, 6),
            "semantic_blend_calibration_nll_gain": round(
                model.semantic_blend_nll_gain_, 6
            ),
            "intent_selected_c": model.intent_selected_c_,
            "intent_training_cv_macro_f1": (
                round(model.intent_cv_macro_f1_, 6)
                if model.tune_intent_regularization else None
            ),
            "escalation_selected_c": model.escalation_selected_c_,
            "escalation_selected_class_weight": model.escalation_selected_class_weight_,
            "escalation_training_cv_average_precision": round(
                model.escalation_cv_average_precision_, 6
            ),
            "escalation_knn_blend_weight": round(model.escalation_knn_weight_, 6),
            "escalation_knn_neighbor_count": model.escalation_knn_k_,
            "escalation_knn_calibration_ap_gain": round(
                model.escalation_knn_calibration_ap_gain_, 6
            ),
            "supervision": (
                "reply-side weak labels plus conservative probabilistic abstention filling "
                "and optional human-reviewed overrides"
            ),
            "human_reviewed_label_count": int((supervision["intent_label_source"] == "human_reviewed").sum()),
            "intent_train_size": artifacts.intent_train_size,
            "intent_calibration_size": artifacts.intent_calibration_size,
            "weak_intent_labels_pruned": model.intent_pruned_count_,
            "probabilistic_intent_label_count": artifacts.probabilistic_intent_label_count,
            "classification_features": "MiniLM + word TF-IDF + character TF-IDF",
            "contrastive_training_pairs": model.encoder.contrastive_pairs_seen_,
            "human_reason_training_count": model.reason_training_count_,
            "escalation_train_size": artifacts.escalation_train_size,
            "escalation_calibration_size": artifacts.escalation_calibration_size,
            "generation_mode": args.generation_mode,
            "judge_mode": "local_model" if args.judge_local else "not_run",
            "zero_shot_local_baseline": "run" if args.include_zero_shot_local else "not_run",
        },
        "systems": {
            name: _evaluate_system(evaluator, gold, data, args.bootstrap_iterations)
            for name, data in systems.items()
        },
        "paired_significance": {
            "intent_proposed_vs_tfidf": evaluator.paired_mcnemar(
                gold["gold_intent"].tolist(),
                systems["Baseline 2: TF-IDF + keyword escalation + retrieve/replay"]["intent_preds"],
                systems[PROPOSED]["intent_preds"],
            ),
            "escalation_proposed_vs_keyword": evaluator.paired_mcnemar(
                gold["gold_escalate"].astype(bool).tolist(),
                systems["Baseline 2: TF-IDF + keyword escalation + retrieve/replay"]["esc_preds"],
                systems[PROPOSED]["esc_preds"],
            ),
        },
        "temperature_scaling": {
            "intent_temperature": round(model.intent_temperature.temperature_, 6),
            "escalation_temperature": round(model.escalation_temperature.temperature_, 6),
            "calibration_source": "reply-labeled holdout; no gold labels",
        },
        "intent_calibration": evaluator.compute_ece(
            gold["gold_intent"].tolist(),
            systems[PROPOSED]["intent_preds"],
            confidences,
        ),
        "abstention": {
            "threshold": model.intent_confidence_threshold,
            "coverage_accuracy_curve": evaluator.coverage_accuracy_curve(
                gold["gold_intent"].tolist(), systems[PROPOSED]["intent_preds"], confidences
            ),
        },
        "escalation_threshold": {
            "threshold": round(model.escalation_threshold_, 6),
            "high_precision_threshold": round(model.high_precision_escalation_threshold_, 6),
            "selection_target_recall": model.target_escalation_recall,
            "high_risk_selection_target_precision": model.target_high_risk_precision,
            "selection_source": "reply-labeled holdout; no gold labels",
            "calibration_observed_recall": round(
                model.escalation_threshold_selection_.observed_recall, 6
            ),
            "calibration_recall_lower_bound": round(
                model.escalation_threshold_selection_.recall_lower_bound, 6
            ),
            "calibration_confidence_level": round(
                model.escalation_threshold_selection_.confidence_level, 6
            ),
            "calibration_positive_count": model.escalation_threshold_selection_.positive_count,
            "recall_bound_target_met": model.escalation_threshold_selection_.target_met,
            "gold_cost_curve_diagnostic": evaluator.optimize_cost_threshold(
                gold["gold_escalate"].astype(bool).tolist(), risk_scores, 1.0, 10.0
            ),
        },
    }

    predictions = pd.DataFrame({
        "reply_id": [f"{item_id}:proposed-v2" for item_id in gold["id"]],
        "gold_id": gold["id"],
        "customer_text": gold["customer_text"],
        "model_input": [
            artifacts.context_by_id.get(str(tweet_id), text)
            for tweet_id, text in zip(gold["tweet_id"], gold["customer_text"])
        ],
        "gold_intent": gold["gold_intent"],
        "predicted_intent": systems[PROPOSED]["intent_preds"],
        "confidence": confidences,
        "context_used": systems[PROPOSED]["context_used"],
        "escalation_risk_score": risk_scores,
        "gold_escalate": gold["gold_escalate"],
        "predicted_escalate": systems[PROPOSED]["esc_preds"],
        "gold_reason": gold["gold_escalate_reason"],
        "predicted_reason": systems[PROPOSED]["esc_reasons"],
        "resolution_act": systems[PROPOSED]["acts"],
        "draft_reply": systems[PROPOSED]["replies"],
    })

    judge_rows = []
    judge_invalid_outputs = 0
    if args.judge_local:
        judge = ReplyQualityJudge(local_model=local_runtime)
        for index, row in predictions.head(args.judge_limit).iterrows():
            exemplars = systems[PROPOSED]["exemplars"][index]
            try:
                score = judge.score_reply(
                    row["customer_text"], row["draft_reply"], row["predicted_intent"],
                    bool(row["predicted_escalate"]), row["resolution_act"], exemplars,
                )
                judge_rows.append({
                    "reply_id": row["reply_id"],
                    "source_sha256": reply_source_hash(
                        row["reply_id"], row["customer_text"], row["draft_reply"],
                        row["resolution_act"],
                    ),
                    **{f"judge_{key}": score[key] for key in ("groundedness", "empathy", "actionability", "safety")},
                    "schema_repaired": score["schema_repaired"],
                    "rationale": score["rationale"],
                })
            except JudgeOutputError:
                judge_invalid_outputs += 1
        judge_frame = pd.DataFrame(judge_rows)
        judge_frame.to_csv("results/local_judge_scores.csv", index=False)
        full_results["local_llm_judge"] = {
            "status": "measured" if len(judge_frame) else "invalid",
            "requested_sample_size": min(args.judge_limit, len(predictions)),
            "valid_sample_size": len(judge_frame),
            "invalid_output_count": judge_invalid_outputs,
            "schema_repair_count": int(
                judge_frame.get("schema_repaired", pd.Series(dtype=bool)).sum()
            ),
            "mean_overall": (
                round(float(judge_frame[[f"judge_{x}" for x in ("groundedness", "empathy", "actionability", "safety")]].mean(axis=1).mean()), 3)
                if len(judge_frame) else None
            ),
            "model": local_runtime.model_name,
            "revision": local_runtime.revision,
        }
        full_results["human_judge_agreement"] = _same_reply_agreement(
            judge_frame, Path(args.human_ratings)
        )
    else:
        (
            full_results["local_llm_judge"],
            full_results["human_judge_agreement"],
        ) = _cached_judge_evidence(gold, predictions)
        if full_results["local_llm_judge"]["status"].startswith("measured_cached"):
            full_results["run_metadata"]["judge_mode"] = "cached_content_verified_local_model"

    rating_template = predictions.head(args.judge_limit)[
        ["reply_id", "customer_text", "draft_reply", "resolution_act"]
    ].copy()
    rating_template["source_sha256"] = rating_template.apply(
        lambda row: reply_source_hash(
            row["reply_id"], row["customer_text"], row["draft_reply"],
            row["resolution_act"],
        ), axis=1,
    )
    for field in ("groundedness", "empathy", "actionability", "safety"):
        rating_template[f"human_{field}"] = ""
    rating_template["human_notes"] = ""

    failures = predictions[
        (predictions["gold_intent"] != predictions["predicted_intent"])
        | (predictions["gold_escalate"].astype(bool) != predictions["predicted_escalate"].astype(bool))
    ].head(10)
    Path("results").mkdir(exist_ok=True)
    predictions.to_csv("results/golden_predictions.csv", index=False)
    rating_template.to_csv("results/human_rating_template.csv", index=False)
    failures.to_csv("results/failure_examples.csv", index=False)
    with open("results/headline_results.json", "w", encoding="utf-8") as handle:
        json.dump(full_results, handle, indent=2)

    rows = []
    for name, result in full_results["systems"].items():
        rows.append({
            "System": name,
            "Intent Macro-F1": result["intent_metrics"]["macro_f1"],
            "Escalate Recall": result["escalation_metrics"]["escalate_recall"],
            "Escalate Precision": result["escalation_metrics"]["escalate_precision"],
            "Reason Exact": result["escalation_metrics"]["reason_exact_match"],
            "Length Pass": result["reply_safety"]["twitter_length_compliance"],
            "No PII Echo": result["reply_safety"]["no_pii_echo_rate"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv("results/baseline_comparison.csv", index=False)
    print(summary.to_string(index=False))
    print("Wrote results/headline_results.json and row-level evidence.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-mode", choices=["offline", "local"], default="offline")
    parser.add_argument("--judge-local", action="store_true")
    parser.add_argument("--include-zero-shot-local", action="store_true")
    parser.add_argument("--judge-limit", type=int, default=50)
    parser.add_argument("--human-ratings", default="data/evaluation/human_same_reply_ratings.csv")
    parser.add_argument(
        "--reviewed-labels", default="data/training/human_consensus_labels.csv"
    )
    parser.add_argument("--intent-cap-per-class", type=int, default=1200)
    parser.add_argument("--retrieval-size", type=int, default=8000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--contrastive-pairs", type=int, default=0,
        help="Experimental local contrastive ablation; disabled by default until reviewed labels exist",
    )
    parser.add_argument(
        "--escalation-knn", action="store_true",
        help="Experimental neighbor-risk blend; disabled after policy-gold degradation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
