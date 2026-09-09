"""
Reproduction Script: Benchmark Proposed Production Agent against Trivial and Simple Baselines.
Evaluates on the 200 Hand-Labelled Golden Evaluation Set.
Generates:
- Baseline comparison table (Accuracy, Macro-F1, Escalation F1, Reason Alignment, ROUGE-L, BLEU-4, Length, Judge)
- Expected Calibration Error (ECE) and Cost-Utility Optimization analysis
- Human-Judge Agreement validation statistics
- Serialized results in results/baseline_comparison.csv and results/headline_results.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.classifier import (
    MajorityBaselineClassifier,
    TfidfBaselineClassifier,
    ProductionIntentClassifier,
    INTENT_TAXONOMY
)
from src.escalator import (
    TrivialEscalationBaseline,
    SimpleKeywordEscalationBaseline,
    ProductionEscalator
)
from src.retriever import HistoricalResolutionRetriever
from src.generator import (
    TrivialReplyBaseline,
    RetrievalReplyBaseline,
    ProductionGroundedGenerator
)
from src.evaluator import SupportPipelineEvaluator
from src.llm_judge import ReplyQualityJudge
from src.guardrails import ProductionPolicyGuard

def run_evaluation():
    print("=" * 90)
    print("          @AmazonHelp AI Agent: Headline Reproduction Benchmark (200 Golden Cases)")
    print("=" * 90)

    # 1. Load Golden Evaluation Set
    golden_csv = "data/golden/golden_eval_set.csv"
    if not os.path.exists(golden_csv):
        print(f"Error: {golden_csv} not found! Please ensure data is present.")
        return

    gold_df = pd.read_csv(golden_csv)
    with open("data/golden/golden_eval_set.jsonl", "r", encoding="utf-8") as f:
        gold_records = [json.loads(line) for line in f]

    print(f"\n[1/5] Loaded {len(gold_df)} hand-curated Golden Evaluation examples.")

    # 2. Load Processed Clean Training Corpus for Model Fitting
    train_csv = "data/processed/amazonhelp_english_pairs.csv"
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found!")
        return

    print(f"\n[2/5] Loading cleaned historical interactions for index & model fitting...")
    train_df = pd.read_csv(train_csv)
    train_sample = train_df.sample(n=min(15000, len(train_df)), random_state=42)

    # 3. Fit Classifiers with Weak Supervision on Cleaned Data
    print(f"\n[3/5] Fitting Classifiers and Vector Indices on {len(train_sample)} interactions...")
    X_train = train_sample["customer_text"].tolist()
    silver_labels = []
    for text in X_train:
        t_low = str(text).lower()
        if any(w in t_low for w in ["damaged", "broken", "wrong item", "defective", "missing item", "empty box", "shattered", "cracked", "scratched", "missing part"]):
            silver_labels.append("DAMAGED_OR_WRONG_ITEM")
        elif any(w in t_low for w in ["login", "password", "otp", "2fa", "hacked", "prime membership", "locked out", "sign in", "cant access account", "verify account"]):
            silver_labels.append("ACCOUNT_AND_SECURITY")
        elif any(w in t_low for w in ["kindle", "fire tv", "firestick", "alexa", "echo", "prime video", "streaming", "bluetooth", "paperwhite", "app crash"]):
            silver_labels.append("PRODUCT_AND_TECHNICAL")
        elif any(w in t_low for w in ["refund", "return", "returned", "cancel order", "charged twice", "money back", "billing", "card charged", "double charge"]):
            silver_labels.append("REFUND_AND_RETURN")
        elif any(w in t_low for w in ["track", "delivery", "carrier", "transit", "where is my", "ups", "usps", "late", "package", "parcel", "arrived", "shipping", "shipped", "shipment", "not arrived", "deliver", "two day shipping"]):
            silver_labels.append("ORDER_TRACKING_AND_DELAY")
        else:
            silver_labels.append("FEEDBACK_AND_GENERAL")
    train_sample["intent"] = silver_labels

    # Initialize Classifiers
    trivial_clf = MajorityBaselineClassifier().fit(X_train, silver_labels)
    simple_clf = TfidfBaselineClassifier().fit(X_train, silver_labels)
    prod_clf = ProductionIntentClassifier(confidence_threshold=0.32).fit(X_train, silver_labels)

    # Initialize Escalators
    trivial_esc = TrivialEscalationBaseline()
    simple_esc = SimpleKeywordEscalationBaseline()
    prod_esc = ProductionEscalator(confidence_threshold=0.32)

    # Initialize Retriever
    retriever = HistoricalResolutionRetriever(top_k=3).build_index(train_sample)

    # Initialize Generators
    trivial_gen = TrivialReplyBaseline()
    simple_gen = RetrievalReplyBaseline()
    prod_gen = ProductionGroundedGenerator()
    guard = ProductionPolicyGuard()

    # Evaluator & Judge
    evaluator = SupportPipelineEvaluator()
    judge = ReplyQualityJudge()

    # 4. Benchmark Inference
    print(f"\n[4/5] Running inference across all 3 systems on 200 Golden test cases...")
    test_texts = gold_df["customer_text"].tolist()
    y_true_intent = gold_df["gold_intent"].tolist()
    y_true_esc = gold_df["gold_escalate"].tolist()
    y_true_reasons = gold_df["gold_escalate_reason"].tolist()
    references = gold_df["reference_reply"].tolist()

    systems = {
        "Baseline 1: Trivial (Majority+AlwaysEsc+Canned)": {
            "intent_preds": [], "esc_preds": [], "esc_reasons": [], "replies": [], "time": 0
        },
        "Baseline 2: Simple (TFIDF+KeywordEsc+Retrieved)": {
            "intent_preds": [], "esc_preds": [], "esc_reasons": [], "replies": [], "time": 0
        },
        "Proposed System: Production Support Agent": {
            "intent_preds": [], "esc_preds": [], "esc_reasons": [], "replies": [], "time": 0
        }
    }

    # System 1: Trivial
    t0 = time.time()
    for text in test_texts:
        pred_i = trivial_clf.predict([text])[0]
        esc_dec = trivial_esc.evaluate(text)
        reply = trivial_gen.generate(text, pred_i, [])
        systems["Baseline 1: Trivial (Majority+AlwaysEsc+Canned)"]["intent_preds"].append(pred_i)
        systems["Baseline 1: Trivial (Majority+AlwaysEsc+Canned)"]["esc_preds"].append(esc_dec.should_escalate)
        systems["Baseline 1: Trivial (Majority+AlwaysEsc+Canned)"]["esc_reasons"].append(esc_dec.stated_reason)
        systems["Baseline 1: Trivial (Majority+AlwaysEsc+Canned)"]["replies"].append(reply)
    systems["Baseline 1: Trivial (Majority+AlwaysEsc+Canned)"]["time"] = time.time() - t0

    # System 2: Simple
    t0 = time.time()
    for text in test_texts:
        pred_i = simple_clf.predict([text])[0]
        esc_dec = simple_esc.evaluate(text)
        exemplars = retriever.retrieve_similar(text, intent_filter=pred_i, top_k=1)
        reply = simple_gen.generate(text, pred_i, exemplars)
        systems["Baseline 2: Simple (TFIDF+KeywordEsc+Retrieved)"]["intent_preds"].append(pred_i)
        systems["Baseline 2: Simple (TFIDF+KeywordEsc+Retrieved)"]["esc_preds"].append(esc_dec.should_escalate)
        systems["Baseline 2: Simple (TFIDF+KeywordEsc+Retrieved)"]["esc_reasons"].append(esc_dec.stated_reason)
        systems["Baseline 2: Simple (TFIDF+KeywordEsc+Retrieved)"]["replies"].append(reply)
    systems["Baseline 2: Simple (TFIDF+KeywordEsc+Retrieved)"]["time"] = time.time() - t0

    # System 3: Production
    prod_confidences = []
    prod_risk_scores = []
    t0 = time.time()
    for text in test_texts:
        res = prod_clf.predict_with_confidence(text)
        pred_i = res["intent"]
        conf = res["confidence"]
        is_unc = res["is_uncertain"]
        prod_confidences.append(conf)

        esc_dec = prod_esc.evaluate(text, intent=pred_i, confidence=conf, is_uncertain=is_unc)
        prod_risk_scores.append(esc_dec.risk_score)

        exemplars = retriever.retrieve_similar(text, intent_filter=pred_i, top_k=3)
        raw_reply = prod_gen.generate(text, pred_i, exemplars, esc_dec.to_dict())
        guard_res = guard.verify_and_sanitize(raw_reply, customer_text=text, should_escalate=esc_dec.should_escalate)
        final_reply = guard_res.sanitized_reply

        systems["Proposed System: Production Support Agent"]["intent_preds"].append(pred_i)
        systems["Proposed System: Production Support Agent"]["esc_preds"].append(esc_dec.should_escalate)
        systems["Proposed System: Production Support Agent"]["esc_reasons"].append(esc_dec.stated_reason)
        systems["Proposed System: Production Support Agent"]["replies"].append(final_reply)
    systems["Proposed System: Production Support Agent"]["time"] = time.time() - t0

    # 5. Evaluate and Compare Metrics
    print(f"\n[5/5] Calculating automated metrics & LLM-as-judge scores...")
    summary_rows = []
    full_results = {}

    for name, s_data in systems.items():
        intent_metrics = evaluator.evaluate_intents(y_true_intent, s_data["intent_preds"])
        esc_metrics = evaluator.evaluate_escalation(
            y_true_esc, s_data["esc_preds"], y_true_reasons, s_data["esc_reasons"]
        )
        reply_metrics = evaluator.evaluate_replies(references, s_data["replies"])

        judge_scores = []
        for text, reply, intent, esc in zip(test_texts, s_data["replies"], s_data["intent_preds"], s_data["esc_preds"]):
            scores = judge.score_reply(text, reply, intent, esc)
            judge_scores.append(scores["overall_score"])
        
        avg_judge_score = round(float(np.mean(judge_scores)), 2)

        summary_rows.append({
            "System": name,
            "Intent Acc": f"{intent_metrics['accuracy']*100:.1f}%",
            "Intent Macro-F1": f"{intent_metrics['macro_f1']:.3f}",
            "Escalation Acc": f"{esc_metrics['accuracy']*100:.1f}%",
            "Escalation F1": f"{esc_metrics['escalate_f1']:.3f}",
            "Reason Alignment": f"{esc_metrics['reason_alignment_rate']*100:.1f}%",
            "ROUGE-L": f"{reply_metrics['rougeL_f1']:.3f}",
            "BLEU-4": f"{reply_metrics['bleu4']:.3f}",
            "Twitter Length %": f"{reply_metrics['twitter_length_compliance_rate']*100:.1f}%",
            "LLM Judge (1-5)": f"{avg_judge_score:.2f}",
            "Latency/Tweet": f"{s_data['time']/len(test_texts)*1000:.1f}ms"
        })

        full_results[name] = {
            "intent_metrics": intent_metrics,
            "escalation_metrics": esc_metrics,
            "reply_metrics": reply_metrics,
            "avg_judge_score": avg_judge_score,
            "total_latency_sec": round(s_data["time"], 3)
        }

    # Calibration & Cost-Utility Decision Theory Optimization
    ece_prod = evaluator.compute_ece(
        y_true=y_true_intent,
        y_pred=systems["Proposed System: Production Support Agent"]["intent_preds"],
        confidences=prod_confidences,
        n_bins=10
    )
    cost_opt = evaluator.optimize_cost_threshold(
        y_true_esc=y_true_esc,
        risk_scores=prod_risk_scores,
        cost_escalate=1.0,
        cost_error_leak=5.0
    )
    full_results["calibration_metrics"] = ece_prod
    full_results["decision_theory_cost_optimization"] = cost_opt

    # Human-Judge Agreement Evaluation on Golden Set
    human_overall = []
    for r in gold_records:
        h_score = (r["human_groundedness_score"] + r["human_empathy_score"] + r["human_actionability_score"] + r["human_safety_score"]) / 4.0
        human_overall.append(h_score)

    judge_scores_prod = [
        judge.score_reply(t, rep, i, e)["overall_score"]
        for t, rep, i, e in zip(
            test_texts,
            systems["Proposed System: Production Support Agent"]["replies"],
            systems["Proposed System: Production Support Agent"]["intent_preds"],
            systems["Proposed System: Production Support Agent"]["esc_preds"]
        )
    ]
    agreement_metrics = judge.compute_human_agreement(human_overall, judge_scores_prod)
    full_results["human_judge_agreement"] = agreement_metrics

    # Save to disk
    os.makedirs("results", exist_ok=True)
    with open("results/headline_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/baseline_comparison.csv", index=False)

    print("\n" + "=" * 115)
    print("                      OFFICIAL HEADLINE BENCHMARK COMPARISON TABLE")
    print("=" * 115)
    print(summary_df.to_string(index=False))
    print("=" * 115)

    print(f"\n[Uncertainty Calibration & Decision Theory Optimization]")
    print(f"  Expected Calibration Error (ECE):   {ece_prod['ece']:.4f} (Max Error: {ece_prod['mce']:.4f})")
    print(f"  Optimal Escalation Risk Threshold:  T* = {cost_opt['optimal_threshold']} (under 5x false-negative penalty ratio)")
    print(f"  Minimum Business Loss:              {cost_opt['minimum_loss']} units")

    print(f"\n[Human-Judge Agreement Validation]")
    print(f"  Pearson Correlation (r):       {agreement_metrics.get('pearson_r', 0.0):.3f}")
    print(f"  Cohen's Weighted Kappa:        {agreement_metrics.get('cohen_kappa_weighted', 0.0):.3f}")
    print(f"  Mean Absolute Error (MAE):     {agreement_metrics.get('mae', 0.0):.3f}")
    print(f"  Adjacent Agreement (±1 point): {agreement_metrics.get('adjacent_agreement_pct', 0.0):.1f}%")

    print(f"\nBenchmark artifacts successfully saved to:")
    print(f"  - results/baseline_comparison.csv")
    print(f"  - results/headline_results.json")

if __name__ == "__main__":
    run_evaluation()
