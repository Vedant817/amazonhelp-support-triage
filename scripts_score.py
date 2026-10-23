"""Score matrix: full evaluation of the current implementation.

Protocol
--------
The rows in ``data/training/labels_for_human_review.csv`` are *excluded* from
training through the pipeline's leak-guard hook and then used as the evaluation
set. That makes the numbers held-out rather than memorised.

Caveat
------
The reference labels in that file are **weak supervision** (derived from
historical agent replies and customer-side heuristics), not independent human
gold. Every value below therefore measures agreement with the supervision
signal, not ground truth. Human review is required before these can be quoted
as quality evidence.

Usage
-----
    python scripts_score.py                 # held-out (retrains once, cached)
    python scripts_score.py --in-sample     # reuse the served pipeline (fast)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.escalator import SafetyGatedEscalator, SimpleKeywordEscalationBaseline
from src.evaluator import SupportPipelineEvaluator
from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard
from src.model_factory import build_learned_pipeline
from src.supervision import infer_reply_escalation, infer_reply_intent

PAIRS_CSV = "data/processed/amazonhelp_context_pairs.csv"
EVAL_CSV = "data/training/labels_for_human_review.csv"
HOLDOUT_IDS = "models/_eval_holdout_ids.csv"
REPORT_JSON = "models/_score_matrix.json"


def _write_holdout_file() -> int:
    """Materialise the evaluation inbound_ids as a leak-guard file."""
    frame = pd.read_csv(EVAL_CSV)
    ids = frame["inbound_id"].astype(str)
    Path(HOLDOUT_IDS).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tweet_id": ids}).to_csv(HOLDOUT_IDS, index=False)
    return len(ids)


def _confusion_table(cm: list[list[int]], labels: list[str]) -> str:
    width = max(len(label) for label in labels) + 1
    header = " " * width + "".join(f"{label[:9]:>11s}" for label in labels)
    rows = [header]
    for label, row in zip(labels, cm):
        rows.append(f"{label:<{width}}" + "".join(f"{value:>11d}" for value in row))
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sample", action="store_true",
                        help="Reuse the served pipeline instead of retraining without the eval rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_frame = pd.read_csv(EVAL_CSV)
    print(f"Evaluation set: {EVAL_CSV} ({len(eval_frame)} rows, weak labels)")

    started = time.time()
    if args.in_sample:
        print("Mode: IN-SAMPLE (evaluation rows were part of training) — optimistic")
        artifacts = build_learned_pipeline(pairs_csv=PAIRS_CSV)
    else:
        count = _write_holdout_file()
        print(f"Mode: HELD-OUT ({count} rows excluded from training)")
        artifacts = build_learned_pipeline(pairs_csv=PAIRS_CSV, gold_csv=HOLDOUT_IDS)
    print(f"Pipeline ready in {time.time() - started:.1f}s (cache_hit={artifacts.cache_hit})")

    model = artifacts.model
    retriever = artifacts.retriever
    escalator = SafetyGatedEscalator(model.escalation_threshold_)
    generator = ProductionGroundedGenerator(mode="offline")
    guard = ProductionPolicyGuard()
    evaluator = SupportPipelineEvaluator()

    # ------------------------------------------------------------------ collect
    rows = {"intent_true": [], "intent_pred": [], "confidence": [], "abstain": [],
            "esc_true": [], "esc_pred": [], "esc_risk": [], "reason_true": [], "reason_pred": [],
            "reply": [], "reference": [], "guard_ok": [], "pii_leak": [], "act_ok": [],
            "baseline_tfidf_pred": []}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    labelled = eval_frame[eval_frame["intent_label"].notna()].reset_index(drop=True)

    # A leak-free baseline: fit TF-IDF on the training corpus only.
    train_pairs = pd.read_csv(PAIRS_CSV)
    holdout_ids = set(eval_frame["inbound_id"].astype(str))
    train_only = train_pairs[~train_pairs["inbound_id"].astype(str).isin(holdout_ids)]
    train_replies = train_only["reply_text"].fillna("")
    train_labels = train_replies.map(infer_reply_intent)
    baseline_ok = train_labels.notna()
    baseline_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = baseline_vectorizer.fit_transform(train_replies[baseline_ok])
    baseline_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    baseline_model.fit(X_train, train_labels[baseline_ok])

    keyword_escalator = SimpleKeywordEscalationBaseline()

    for _, row in eval_frame.iterrows():
        text = str(row["customer_text"])
        reference = str(row["reply_text"]) if pd.notna(row.get("reply_text")) else ""

        classification = model.predict_with_confidence(text)
        learned = model.predict_escalation(text, classification["current_message_embedding"])
        reason_prediction = model.predict_reason(text, classification["current_message_embedding"])
        exemplars = retriever.retrieve_similar(text, intent_filter=classification["intent"], top_k=12)
        evidence_act = generator.select_act(exemplars, classification["intent"], False)
        decision = escalator.evaluate(
            text, classification["intent"], learned["risk_score"],
            is_uncertain=classification["is_uncertain"],
            learned_threshold=learned["threshold"],
            high_precision_threshold=model.high_precision_escalation_threshold_,
            resolution_act=evidence_act,
            predicted_reason=(reason_prediction or {}).get("reason"),
        )
        act = generator.select_act(exemplars, classification["intent"], decision.should_escalate)
        raw_reply, _ = generator.generate_with_metadata(
            text, classification["intent"], exemplars, escalate_decision=decision.to_dict()
        )
        guarded = guard.verify_and_sanitize(raw_reply, customer_text=text, should_escalate=decision.should_escalate)

        true_intent = row["intent_label"]
        true_esc = row["escalate_label"]
        true_reason = row.get("reviewed_reason_code") if pd.notna(row.get("reviewed_reason_code")) else None

        rows["intent_true"].append(str(true_intent) if pd.notna(true_intent) else None)
        rows["intent_pred"].append(classification["intent"])
        rows["confidence"].append(classification["confidence"])
        rows["abstain"].append(classification["is_uncertain"])
        rows["esc_true"].append(bool(true_esc) if pd.notna(true_esc) else None)
        rows["esc_pred"].append(decision.should_escalate)
        rows["esc_risk"].append(decision.risk_score)
        rows["reason_true"].append(true_reason)
        rows["reason_pred"].append(decision.stated_reason)
        rows["reply"].append(guarded.sanitized_reply)
        rows["reference"].append(reference)
        rows["guard_ok"].append(guarded.is_valid)
        rows["pii_leak"].append(any("PII" in v for v in guarded.violations))
        rows["act_ok"].append(generator._realizes_act(guarded.sanitized_reply, act))

        baseline_text = text
        try:
            rows["baseline_tfidf_pred"].append(
                baseline_model.predict(baseline_vectorizer.transform([baseline_text]))[0]
            )
        except Exception:
            rows["baseline_tfidf_pred"].append(None)

    # ------------------------------------------------------------------ intent
    intent_pairs = [(t, p) for t, p in zip(rows["intent_true"], rows["intent_pred"]) if t]
    y_true = [t for t, _ in intent_pairs]
    y_pred = [p for _, p in intent_pairs]
    labels = sorted(set(y_true) | set(y_pred))

    intent_metrics = evaluator.evaluate_intents(y_true, y_pred, labels=labels)
    intent_ci = {
        "accuracy": evaluator.bootstrap_classification_intervals(y_true, y_pred, "accuracy"),
        "macro_f1": evaluator.bootstrap_classification_intervals(y_true, y_pred, "macro_f1"),
    }

    # ------------------------------------------------------------------ escalation
    esc_pairs = [(t, p) for t, p in zip(rows["esc_true"], rows["esc_pred"]) if t is not None]
    esc_true = [t for t, _ in esc_pairs]
    esc_pred = [p for _, p in esc_pairs]
    reason_pairs = [(t, p) for t, p in zip(rows["reason_true"], rows["reason_pred"]) if t]
    esc_metrics = evaluator.evaluate_escalation(
        esc_true, esc_pred,
        [t for t, _ in reason_pairs] or [""],
        [p for _, p in reason_pairs] or [""],
    )
    # With no independently reviewed reason codes there is nothing to match
    # against; report None rather than a vacuous 1.0.
    if not reason_pairs:
        esc_metrics["reason_exact_match"] = None
    esc_ci = {
        "escalate_recall": evaluator.bootstrap_classification_intervals(esc_true, esc_pred, "escalate_recall"),
        "escalate_precision": evaluator.bootstrap_classification_intervals(esc_true, esc_pred, "escalate_precision"),
    }

    # ------------------------------------------------------------------ calibration
    coverage = evaluator.coverage_accuracy_curve(y_true, y_pred, rows["confidence"][:len(y_true)])
    ece = evaluator.compute_ece(y_true, y_pred, [c for c, t in zip(rows["confidence"], rows["intent_true"]) if t])

    # ------------------------------------------------------------------ replies
    reply_metrics = evaluator.evaluate_replies(rows["reference"], rows["reply"])
    reply_metrics["guardrail_pass_rate"] = round(float(np.mean(rows["guard_ok"])), 4)
    reply_metrics["pii_violation_count"] = int(sum(rows["pii_leak"]))
    reply_metrics["act_realization_rate"] = round(float(np.mean(rows["act_ok"])), 4)

    # ------------------------------------------------------------------ cost
    cost = evaluator.optimize_cost_threshold(
        [t for t in rows["esc_true"] if t is not None],
        [r for r, t in zip(rows["esc_risk"], rows["esc_true"]) if t is not None],
    )

    # ------------------------------------------------------------------ baselines
    baseline_true = [t for t, p in zip(rows["intent_true"], rows["baseline_tfidf_pred"]) if t and p]
    baseline_pred = [p for t, p in zip(rows["intent_true"], rows["baseline_tfidf_pred"]) if t and p]
    baseline_intent = evaluator.evaluate_intents(baseline_true, baseline_pred) if baseline_true else {}

    keyword_pred = [keyword_escalator.evaluate(t).should_escalate for t in eval_frame["customer_text"]]
    keyword_pairs = [(t, p) for t, p in zip(rows["esc_true"], keyword_pred) if t is not None]
    keyword_esc = evaluator.evaluate_escalation(
        [t for t, _ in keyword_pairs], [p for _, p in keyword_pairs], [""], [""]
    ) if keyword_pairs else {}

    majority = max(set(y_true), key=y_true.count) if y_true else "?"
    majority_intent = evaluator.evaluate_intents(y_true, [majority] * len(y_true)) if y_true else {}
    always_esc = evaluator.evaluate_escalation(
        esc_true, [True] * len(esc_true), [""], [""]
    ) if esc_true else {}

    report = {
        "protocol": "in-sample" if args.in_sample else "held-out",
        "eval_rows": int(len(eval_frame)),
        "intent_labelled": len(y_true),
        "escalation_labelled": len(esc_true),
        "model_state": {
            "prior_correction_strength": float(getattr(model, "prior_correction_strength_", 0.0)),
            "prior_correction_calibration_gain": float(getattr(model, "prior_correction_calibration_gain_", 0.0)),
            "intent_temperature": float(model.intent_temperature.temperature_),
            "escalation_temperature": float(model.escalation_temperature.temperature_),
            "escalation_threshold": float(model.escalation_threshold_),
            "intent_confidence_threshold": float(model.intent_confidence_threshold),
            "intent_selected_c": float(getattr(model, "intent_selected_c_", 0.0)),
            "intent_cv_macro_f1": float(getattr(model, "intent_cv_macro_f1_", 0.0)),
            "semantic_blend_weight": float(getattr(model, "semantic_blend_weight_", 0.0)),
            "prototype_blend_weight": float(getattr(model, "prototype_blend_weight_", 0.0)),
            "ensemble_weights": [
                float(getattr(model, "ensemble_hybrid_weight_", 1.0)),
                float(getattr(model, "ensemble_semantic_weight_", 0.0)),
                float(getattr(model, "ensemble_prototype_weight_", 0.0)),
            ],
            "context_gate_confidence": float(getattr(model, "context_gate_confidence_", 0.0)),
            "intent_labels_pruned": int(getattr(model, "intent_pruned_count_", 0)),
        },
        "intent": intent_metrics,
        "intent_ci": intent_ci,
        "escalation": esc_metrics,
        "escalation_ci": esc_ci,
        "calibration": {"ece": ece["ece"], "mce": ece["mce"]},
        "coverage_curve": coverage,
        "reply": reply_metrics,
        "cost_optimal": {k: cost[k] for k in ("optimal_threshold", "minimum_loss")},
        "baselines": {
            "majority_intent": majority_intent,
            "always_escalate": always_esc,
            "tfidf_intent": baseline_intent,
            "keyword_escalation": keyword_esc,
        },
    }
    Path(REPORT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ print
    print("\n" + "=" * 76)
    print("  SCORE MATRIX")
    print("=" * 76)

    print("\n[intent] classification")
    for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
        print(f"  {key:<18s} {intent_metrics[key]:.4f}")
    print(f"  abstention rate    {np.mean(rows['abstain']):.4f}")
    print(f"  95% CI accuracy    [{intent_ci['accuracy']['lower']:.4f}, {intent_ci['accuracy']['upper']:.4f}]")
    print(f"  95% CI macro-F1    [{intent_ci['macro_f1']['lower']:.4f}, {intent_ci['macro_f1']['upper']:.4f}]")

    print("\n[escalation] routing")
    for key in ("accuracy", "macro_f1", "escalate_precision", "escalate_recall",
                "escalate_f1", "escalate_recall_one_sided_95_lower",
                "auto_handle_precision", "auto_handle_recall", "reason_exact_match"):
        value = esc_metrics[key]
        rendered = f"{value:.4f}" if isinstance(value, (int, float)) else "n/a (no reviewed reason codes)"
        print(f"  {key:<34s} {rendered}")
    print(f"  95% CI escalate recall             [{esc_ci['escalate_recall']['lower']:.4f}, {esc_ci['escalate_recall']['upper']:.4f}]")
    print(f"  95% CI escalate precision          [{esc_ci['escalate_precision']['lower']:.4f}, {esc_ci['escalate_precision']['upper']:.4f}]")

    print("\n[model state]")
    state = report["model_state"]
    print(f"  prior correction alpha   {state['prior_correction_strength']:.2f}")
    print(f"  prior cal accuracy gain  {state['prior_correction_calibration_gain']:.4f}")
    print(f"  intent C (selected)      {state['intent_selected_c']:.2f}   cv macro-F1 {state['intent_cv_macro_f1']:.4f}")
    print(f"  intent temperature       {state['intent_temperature']:.4f}")
    print(f"  semantic blend weight    {state['semantic_blend_weight']:.4f}")
    print(f"  prototype blend weight   {state['prototype_blend_weight']:.4f}")
    print(f"  ensemble weights         hybrid={state['ensemble_weights'][0]:.2f} "
          f"semantic={state['ensemble_weights'][1]:.2f} prototype={state['ensemble_weights'][2]:.2f}")
    print(f"  context gate confidence  {state['context_gate_confidence']:.4f}")
    print(f"  intent labels pruned     {state['intent_labels_pruned']}")
    print(f"  escalation temperature   {state['escalation_temperature']:.4f}")
    print(f"  escalation threshold     {state['escalation_threshold']:.4f}")

    print("\n[calibration]")
    print(f"  ECE                {ece['ece']:.4f}")
    print(f"  MCE                {ece['mce']:.4f}")

    print("\n[reply] safety and compliance")
    print(f"  guardrail pass     {reply_metrics['guardrail_pass_rate']:.4f}")
    print(f"  PII violations     {reply_metrics['pii_violation_count']}")
    print(f"  act realization    {reply_metrics['act_realization_rate']:.4f}")
    print(f"  280-char comply    {reply_metrics['twitter_length_compliance_rate']:.4f}")
    print(f"  ROUGE-L (diag)     {reply_metrics['rougeL_f1']:.4f}")
    print(f"  BLEU-4  (diag)     {reply_metrics['bleu4']:.4f}")

    print("\n[cost] loss = 1.0*FP + 10.0*FN")
    print(f"  optimal threshold  {cost['optimal_threshold']:.2f}")
    print(f"  minimum loss       {cost['minimum_loss']:.1f}")

    print("\n[coverage] accuracy retained by confidence threshold")
    for row in coverage[::2]:
        acc = f"{row['retained_accuracy']:.4f}" if row["retained_accuracy"] is not None else "  n/a "
        print(f"  t={row['confidence_threshold']:.2f}  coverage={row['coverage']:.4f}  acc={acc}  abstain={row['abstain_count']}")

    print("\n[baselines]")
    if majority_intent:
        print(f"  majority intent macro-F1        {majority_intent['macro_f1']:.4f}")
    if always_esc:
        print(f"  always-escalate recall          {always_esc['escalate_recall']:.4f}")
        print(f"  always-escalate precision       {always_esc['escalate_precision']:.4f}")
    if baseline_intent:
        print(f"  TF-IDF intent macro-F1          {baseline_intent['macro_f1']:.4f}")
    if keyword_esc:
        print(f"  keyword escalation recall       {keyword_esc['escalate_recall']:.4f}")
        print(f"  keyword escalation precision    {keyword_esc['escalate_precision']:.4f}")

    print("\n[confusion] intent (rows = reference, cols = predicted)")
    print(_confusion_table(intent_metrics["confusion_matrix"], labels))

    print(f"\nFull JSON report: {REPORT_JSON}")
    print("NOTE: reference labels are WEAK SUPERVISION, not human gold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())