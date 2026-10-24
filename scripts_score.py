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


def _write_holdout_file(eval_ids) -> int:
    """Materialise the evaluation inbound_ids as a leak-guard file."""
    ids = [str(value) for value in eval_ids]
    Path(HOLDOUT_IDS).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tweet_id": ids}).to_csv(HOLDOUT_IDS, index=False)
    return len(ids)


def _labelled_corpus() -> pd.DataFrame:
    """The reply-labelled corpus with intent and escalation targets attached."""
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["intent_label"] = pairs["reply_text"].fillna("").map(infer_reply_intent)
    pairs["escalate_label"] = pairs["reply_text"].fillna("").map(infer_reply_escalation)
    labelled = pairs[pairs["intent_label"].notna()].copy()
    labelled["reviewed_reason_code"] = None
    return labelled


def _build_random_holdout(seed: int, fraction: float) -> pd.DataFrame:
    """Draw a deployment-representative random stratified holdout.

    The checked-in review set is an *active-learning* sample: it deliberately
    contains the highest-uncertainty rows in the corpus, so scoring on it
    measures worst-case behaviour rather than expected traffic performance.
    This draws a plain stratified random sample instead, labelled with the same
    reply-side supervision.
    """
    labelled = _labelled_corpus()
    parts = [
        group.sample(frac=fraction, random_state=seed)
        for _, group in labelled.groupby("intent_label", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


TWITTER_TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse the corpus's Twitter-style timestamps to UTC.

    The corpus stores ``Tue Oct 31 22:16:32 +0000 2017``-style strings; parsing
    them with an explicit format avoids pandas' per-element dateutil fallback and
    its accompanying warning.
    """
    parsed = pd.to_datetime(series, format=TWITTER_TIMESTAMP_FORMAT, errors="coerce", utc=True)
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return parsed


def _build_temporal_holdout(fraction: float) -> pd.DataFrame:
    """Hold out the most recent ``fraction`` of conversations.

    A random split shares its period with training, so it cannot show whether
    the model survives drift — new products, new phrasings, changed agent
    scripts. Training on the past and scoring the future is the closest offline
    analogue of deploying today and serving next quarter, and it is the only
    split here that can fail for a reason other than sampling noise.
    """
    labelled = _labelled_corpus()
    labelled = labelled.assign(_timestamp=_parse_timestamps(labelled["customer_created_at"]))
    dated = labelled[labelled["_timestamp"].notna()].sort_values("_timestamp")
    cutoff = int(len(dated) * (1.0 - fraction))
    holdout = dated.iloc[cutoff:].drop(columns=["_timestamp"])
    return holdout.reset_index(drop=True)


def _temporal_summary(frame: pd.DataFrame) -> str:
    """Describe the evaluation window so a temporal score is interpretable."""
    timestamps = _parse_timestamps(frame["customer_created_at"]).dropna()
    if timestamps.empty:
        return "temporal holdout (newest rows)"
    return (
        f"temporal holdout (newest rows): "
        f"{timestamps.min():%Y-%m-%d} to {timestamps.max():%Y-%m-%d}"
    )


def _confusion_table(cm: list[list[int]], labels: list[str]) -> str:
    width = max(len(label) for label in labels) + 1
    header = " " * width + "".join(f"{label[:9]:>11s}" for label in labels)
    rows = [header]
    for label, row in zip(labels, cm):
        rows.append(f"{label:<{width}}" + "".join(f"{value:>11d}" for value in row))
    return "\n".join(rows)


def _deployment_accuracy(intent_metrics, train_pairs, labels, y_true):
    """Reweight per-class recall by the traffic prior.

    The evaluation sample is stratified while real traffic is not, so raw
    accuracy depends on the protocol. Reweighting per-class recall by the
    corpus prior gives a figure that is comparable across protocols. The prior
    comes from the training corpus, never from the sample being scored.
    """
    cm = np.asarray(intent_metrics["confusion_matrix"], dtype=float)
    row_totals = cm.sum(axis=1, keepdims=True)
    per_class_recall = np.divide(cm.diagonal()[:, None], np.clip(row_totals, 1, None)).ravel()
    traffic_labels = train_pairs["reply_text"].fillna("").map(infer_reply_intent)
    traffic_counts = traffic_labels.dropna().value_counts(normalize=True)
    traffic_prior = np.asarray([float(traffic_counts.get(label, 0.0)) for label in labels])
    if traffic_prior.sum() > 0:
        traffic_prior = traffic_prior / traffic_prior.sum()
    return float(np.sum(traffic_prior * per_class_recall)), traffic_prior, per_class_recall, traffic_counts


def _write_report(path: str, report: dict) -> None:
    """Persist the report. Called once, after every metric has been attached."""
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _write_row_predictions(path: str, eval_frame, rows: dict) -> Optional[str]:
    """Persist per-row predictions next to the report.

    Comparing two configurations needs a paired test, not two independent
    confidence intervals: the same rows are scored by both, so the disagreement
    pattern carries far more information than the marginal accuracies. Without
    per-row output there is nothing to pair, which is why an earlier comparison
    of two label configurations could only report that the intervals overlapped.
    """
    try:
        frame = pd.DataFrame({
            "inbound_id": eval_frame["inbound_id"].astype(str).tolist()[:len(rows["intent_true"])],
            "intent_true": rows["intent_true"],
            "intent_pred": rows["intent_pred"],
            "confidence": rows["confidence"],
            "escalate_true": rows["esc_true"],
            "escalate_pred": rows["esc_pred"],
            "escalation_risk": rows["esc_risk"],
        })
    except (KeyError, ValueError):
        return None
    destination = Path(f"{path}.rows.csv")
    frame.to_csv(destination, index=False)
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sample", action="store_true",
                        help="Reuse the served pipeline instead of retraining without the eval rows")
    parser.add_argument("--holdout", choices=["review", "random", "temporal"], default="review",
                        help="review = active-learning hard cases (worst case); "
                             "random = stratified traffic sample (expected case); "
                             "temporal = newest rows, i.e. train on the past and score the future")
    parser.add_argument("--random-fraction", type=float, default=0.15,
                        help="Fraction of labelled rows to hold out in random mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=REPORT_JSON,
                        help="Where to write the JSON report (parallel runs need distinct paths)")
    args = parser.parse_args()

    if args.holdout == "random":
        eval_frame = _build_random_holdout(args.seed, args.random_fraction)
        protocol_label = "random traffic holdout (deployment-representative)"
        print(f"Evaluation set: stratified random sample of {PAIRS_CSV} ({len(eval_frame)} rows)")
    elif args.holdout == "temporal":
        eval_frame = _build_temporal_holdout(args.random_fraction)
        protocol_label = _temporal_summary(eval_frame)
        print(f"Evaluation set: newest {args.random_fraction:.0%} of {PAIRS_CSV} ({len(eval_frame)} rows)")
    else:
        eval_frame = pd.read_csv(EVAL_CSV)
        protocol_label = "active-learning review sample (hardest cases)"
        print(f"Evaluation set: {EVAL_CSV} ({len(eval_frame)} rows, weak labels)")

    started = time.time()
    if args.in_sample:
        print("Mode: IN-SAMPLE (evaluation rows were part of training) — optimistic")
        artifacts = build_learned_pipeline(pairs_csv=PAIRS_CSV)
    else:
        count = _write_holdout_file(eval_frame["inbound_id"])
        print(f"Mode: HELD-OUT ({count} rows excluded from training)")
        print(f"Holdout: {protocol_label}")
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

    # Label provenance, so a report records whether supervision included
    # model-generated labels. Those score ~0.90 out-of-fold against ~0.64 for
    # reply-derived labels (scripts_cv_audit.py), so a score computed over a
    # label set containing them is not comparable to one computed without.
    _provenance = labelled.get("intent_label_source")
    if _provenance is not None:
        probabilistic_rows = int((_provenance == "probabilistic_weak_label_model").sum())
        reply_derived_rows = int(len(labelled) - probabilistic_rows)
    else:
        probabilistic_rows, reply_derived_rows = 0, int(len(labelled))

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
    escalation_frontier = evaluator.precision_at_recall_frontier(
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
        "holdout": {
            "mode": args.holdout,
            "description": protocol_label,
            "random_fraction": args.random_fraction if args.holdout == "random" else None,
            "seed": args.seed,
        },
        "label_provenance": {
            "reply_derived_rows": int(reply_derived_rows),
            "probabilistic_rows": int(probabilistic_rows),
        },
        "generated_from": "scripts_score.py",
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
            "blend_admission": {
                name: {
                    "accepted": bool(getattr(model, f"{name}_accepted_", False)),
                    **{
                        key: round(float(value), 4)
                        for key, value in (getattr(model, f"{name}_verification_", None) or {}).items()
                        if isinstance(value, (int, float))
                    },
                }
                for name in ("semantic_blend", "prototype_blend", "ensemble_blend")
            },
            "intent_bagging": {
                "accepted": bool(getattr(model, "intent_bagging_accepted_", False)),
                "calibration_gain": float(getattr(model, "intent_bagging_gain_", 0.0)),
                "heads": len(getattr(model, "intent_bagging_heads_", []) or []),
            },
        },
        "intent": intent_metrics,
        "intent_ci": intent_ci,
        "escalation": esc_metrics,
        "escalation_ci": esc_ci,
        "calibration": {"ece": ece["ece"], "mce": ece["mce"]},
        "coverage_curve": coverage,
        "reply": reply_metrics,
        "cost_optimal": {k: cost[k] for k in ("optimal_threshold", "minimum_loss")},
        "escalation_recall_frontier": escalation_frontier,
        "baselines": {
            "majority_intent": majority_intent,
            "always_escalate": always_esc,
            "tfidf_intent": baseline_intent,
            "keyword_escalation": keyword_esc,
        },
    }
    # NOTE: the report is written from the distribution-matched section below,
    # once every metric has been attached. Writing here previously dropped
    # deployment_accuracy and per_class_recall from the JSON, so the console
    # printed figures that the report never contained.

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
    bagging = state.get("intent_bagging", {})
    print(f"  intent bagging           accepted={bagging.get('accepted')} "
          f"heads={bagging.get('heads')} cal_gain={bagging.get('calibration_gain', 0.0):.4f}")
    for name, verification in (state.get("blend_admission") or {}).items():
        print(f"  {name:<24s} accepted={verification.get('accepted')} "
              f"gain={verification.get('mean_macro_f1_gain', 0.0):+.4f} "
              f"win_rate={verification.get('bootstrap_win_rate', 0.0):.2f}")

    print("\n[calibration]")
    print(f"  ECE                {ece['ece']:.4f}")
    print(f"  MCE                {ece['mce']:.4f}")

    print("\n[distribution-matched accuracy]")
    eval_share = float((np.asarray(y_true) == "ORDER_TRACKING_AND_DELAY").mean()) if len(y_true) else 0.0
    deployment_accuracy, traffic_prior, per_class_recall, traffic_counts = _deployment_accuracy(
        intent_metrics, train_pairs, labels, y_true
    )
    print(f"  Evaluation set share of ORDER_TRACKING_AND_DELAY: {eval_share:.2%}")
    print("  Reweighting per-class recall by the traffic prior gives a")
    print("  deployment-representative estimate that is comparable across protocols.")
    print(f"  traffic prior share of ORDER_TRACKING_AND_DELAY: {traffic_counts.get('ORDER_TRACKING_AND_DELAY', 0.0):.2%}")
    print(f"  empirical accuracy (eval prior)     {intent_metrics['accuracy']:.4f}")
    print(f"  deployment accuracy (traffic prior) {deployment_accuracy:.4f}")
    report["deployment_accuracy"] = round(deployment_accuracy, 4)
    report["evaluation_prior_share_order_tracking"] = round(eval_share, 4)
    report["per_class_recall"] = {l: round(float(r), 4) for l, r in zip(labels, per_class_recall)}
    _write_report(args.report, report)
    rows_path = _write_row_predictions(args.report, eval_frame, rows)
    if rows_path:
        print(f"  per-row predictions:  {rows_path}")


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

    print("\n[escalation frontier] highest precision at each recall floor")
    print(f"  {'recall floor':>12s} {'precision':>10s} {'recall':>8s} {'threshold':>10s}")
    for row in escalation_frontier:
        if not row["reached"]:
            print(f"  {row['recall_floor']:>12.2f} {'unreachable':>10s}")
            continue
        print(f"  {row['recall_floor']:>12.2f} {row['precision']:>10.4f} "
              f"{row['recall']:>8.4f} {row['threshold']:>10.2f}")
    print("  The gap between rows is what extra recall costs in human work:")
    print("  precision near the base rate means the learned score adds little and")
    print("  the deterministic hard gates carry the safety property.")

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

    print(f"\nFull JSON report: {args.report}")
    print("NOTE: reference labels are WEAK SUPERVISION, not human gold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())