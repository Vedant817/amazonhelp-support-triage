"""
Automated Evaluation Suite for Intent Classification, Escalation Decisions, and Reply Quality.
Computes:
- Multi-class classification metrics (Accuracy, Macro-F1, Weighted-F1, Confusion Matrix)
- Binary escalation triage metrics (Precision, Recall, F1, Stated Reason Accuracy)
- Text generation metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Length Compliance, Safety Checks)
- Calibration metrics (Expected Calibration Error - ECE, Maximum Calibration Error - MCE)
- Utility-based decision theory cost optimization (cost curve over FP vs FN trade-offs)
"""

import math
import re
from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from rouge_score import rouge_scorer

class SupportPipelineEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_intents(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Calculates comprehensive intent classification metrics."""
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        labels = sorted(list(set(y_true + y_pred)))
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        return {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1_macro), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "labels": labels,
            "confusion_matrix": cm,
            "per_class": {lbl: report.get(lbl, {}) for lbl in labels}
        }

    def evaluate_escalation(
        self,
        y_true_esc: List[bool],
        y_pred_esc: List[bool],
        true_reasons: List[str],
        pred_reasons: List[str]
    ) -> Dict[str, Any]:
        """Calculates escalation triage metrics and stated reason alignment."""
        acc = accuracy_score(y_true_esc, y_pred_esc)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true_esc, y_pred_esc, average='macro', zero_division=0)
        p_esc, r_esc, f1_esc, _ = precision_recall_fscore_support(y_true_esc, y_pred_esc, pos_label=True, average='binary', zero_division=0)
        p_auto, r_auto, f1_auto, _ = precision_recall_fscore_support(y_true_esc, y_pred_esc, pos_label=False, average='binary', zero_division=0)

        # Stated Reason alignment: split on underscores and non-alphanumeric characters
        reason_matches = 0
        for tr, pr in zip(true_reasons, pred_reasons):
            tr_tokens = set(re.findall(r"[a-zA-Z0-9]+", tr.lower()))
            pr_tokens = set(re.findall(r"[a-zA-Z0-9]+", pr.lower()))
            # Remove common filler tokens
            stopwords = {"or", "and", "the", "for", "to", "in", "by", "of", "a", "an"}
            tr_meaningful = tr_tokens - stopwords
            pr_meaningful = pr_tokens - stopwords
            overlap = tr_meaningful.intersection(pr_meaningful)
            if len(overlap) >= 1:
                reason_matches += 1
                
        reason_alignment_rate = round(reason_matches / max(len(true_reasons), 1), 4)

        return {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1_macro), 4),
            "escalate_f1": round(float(f1_esc), 4),
            "escalate_precision": round(float(p_esc), 4),
            "escalate_recall": round(float(r_esc), 4),
            "auto_handle_f1": round(float(f1_auto), 4),
            "auto_handle_precision": round(float(p_auto), 4),
            "auto_handle_recall": round(float(r_auto), 4),
            "reason_alignment_rate": reason_alignment_rate
        }

    def _compute_bleu(self, reference: str, candidate: str, max_n: int = 4) -> float:
        """Computes sentence-level BLEU score with smoothing."""
        ref_tokens = re.findall(r"\w+", reference.lower())
        cand_tokens = re.findall(r"\w+", candidate.lower())
        
        if not cand_tokens or not ref_tokens:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
            
            if not cand_ngrams:
                precisions.append(0.0)
                continue
                
            cand_counts = Counter(cand_ngrams)
            ref_counts = Counter(ref_ngrams)
            clipped_matches = sum(min(cand_counts[ng], ref_counts[ng]) for ng in cand_counts)
            precisions.append((clipped_matches + 1e-5) / (len(cand_ngrams) + 1e-5))

        # Brevity penalty
        bp = math.exp(1 - (len(ref_tokens) / len(cand_tokens))) if len(cand_tokens) < len(ref_tokens) else 1.0
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
        return round(bp * geo_mean, 4)

    def evaluate_replies(
        self,
        references: List[str],
        candidates: List[str]
    ) -> Dict[str, Any]:
        """Calculates text quality metrics: ROUGE, BLEU, length compliance, and safety."""
        r1_list = []
        r2_list = []
        rl_list = []
        bleu_list = []
        length_compliant_count = 0
        safety_violations = 0

        for ref, cand in zip(references, candidates):
            # ROUGE
            scores = self.rouge_scorer.score(ref, cand)
            r1_list.append(scores['rouge1'].fmeasure)
            r2_list.append(scores['rouge2'].fmeasure)
            rl_list.append(scores['rougeL'].fmeasure)

            # BLEU
            bleu_list.append(self._compute_bleu(ref, cand))

            # Twitter character constraint (< 280 chars)
            if len(cand) <= 280:
                length_compliant_count += 1

            # Safety check: ensures public reply never asks for passwords or sensitive card numbers
            lower_cand = cand.lower()
            if any(s in lower_cand for s in ["send password", "tweet your password", "give me your card number", "cvv"]):
                safety_violations += 1

        n = max(len(references), 1)
        return {
            "rouge1_f1": round(float(sum(r1_list) / n), 4),
            "rouge2_f1": round(float(sum(r2_list) / n), 4),
            "rougeL_f1": round(float(sum(rl_list) / n), 4),
            "bleu4": round(float(sum(bleu_list) / n), 4),
            "twitter_length_compliance_rate": round(float(length_compliant_count / n), 4),
            "safety_violations_count": safety_violations
        }

    @staticmethod
    def compute_ece(
        y_true: List[Any],
        y_pred: List[Any],
        confidences: List[float],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        ECE measures the discrepancy between predicted confidence and empirical accuracy.
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        y_t = np.array(y_true)
        y_p = np.array(y_pred)
        confs = np.array(confidences, dtype=float)
        n = len(confs)
        if n == 0:
            return {"ece": 0.0, "mce": 0.0, "bin_details": []}

        correct = (y_t == y_p).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_details = []

        for i in range(n_bins):
            b_low, b_high = bin_boundaries[i], bin_boundaries[i+1]
            in_bin = (confs > b_low) & (confs <= b_high) if i > 0 else (confs >= b_low) & (confs <= b_high)
            bin_size = int(np.sum(in_bin))
            if bin_size > 0:
                bin_acc = float(np.mean(correct[in_bin]))
                bin_conf = float(np.mean(confs[in_bin]))
                diff = abs(bin_acc - bin_conf)
                ece += (bin_size / n) * diff
                mce = max(mce, diff)
                bin_details.append({
                    "bin": f"[{b_low:.1f}, {b_high:.1f}]",
                    "count": bin_size,
                    "acc": round(bin_acc, 3),
                    "conf": round(bin_conf, 3),
                    "gap": round(diff, 3)
                })

        return {
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4),
            "bin_details": bin_details
        }

    @staticmethod
    def optimize_cost_threshold(
        y_true_esc: List[bool],
        risk_scores: List[float],
        cost_escalate: float = 1.0,
        cost_error_leak: float = 5.0
    ) -> Dict[str, Any]:
        """
        Utility-based Decision Theory Optimization:
        Finds the optimal risk threshold T* that minimizes business loss:
          Loss(T) = cost_escalate * FP(T) + cost_error_leak * FN(T)
        where:
          FP: False positive (unnecessary escalation to human agent; cost = labor)
          FN: False negative (missed critical risk / frustrated customer; cost = churn/liability)
        """
        y_true = np.array(y_true_esc, dtype=bool)
        scores = np.array(risk_scores, dtype=float)
        
        candidate_thresholds = np.linspace(0.1, 0.9, 81)
        best_t = 0.5
        min_loss = float("inf")
        loss_curve = []

        for t in candidate_thresholds:
            preds = scores >= t
            fp = int(np.sum((~y_true) & preds))
            fn = int(np.sum(y_true & (~preds)))
            total_loss = (cost_escalate * fp) + (cost_error_leak * fn)
            loss_curve.append({
                "threshold": round(float(t), 2),
                "fp": fp,
                "fn": fn,
                "total_loss": round(float(total_loss), 2)
            })
            if total_loss < min_loss:
                min_loss = total_loss
                best_t = float(t)

        return {
            "optimal_threshold": round(best_t, 3),
            "minimum_loss": round(min_loss, 2),
            "cost_ratio": round(cost_error_leak / cost_escalate, 2),
            "loss_curve_sample": loss_curve[::10]
        }
