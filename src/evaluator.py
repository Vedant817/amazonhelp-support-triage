"""Classification, escalation, calibration, bootstrap, and safety evaluation helpers.

Legacy overlap metrics remain available for diagnostics but are deliberately excluded
from the reply-quality headline evidence.
"""

import re
import math
import numpy as np
from scipy.stats import binomtest
from typing import List, Dict, Any, Tuple, Optional, Sequence
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from scipy.stats import beta

class SupportPipelineEvaluator:
    """
    Comprehensive multi-dimensional evaluation suite.
    """
    def __init__(self):
        pass

    def evaluate_intents(
        self,
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculates classification metrics across intent taxonomy."""
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if labels else []

        return {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1_macro), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "classification_report": report,
            "confusion_matrix": cm
        }

    def evaluate_escalation(
        self,
        y_true_esc: List[bool],
        y_pred_esc: List[bool],
        true_reasons: List[str],
        pred_reasons: List[str]
    ) -> Dict[str, Any]:
        """Calculates escalation triage metrics and strict stated reason alignment."""
        acc = accuracy_score(y_true_esc, y_pred_esc)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true_esc, y_pred_esc, average='macro', zero_division=0)
        p_esc, r_esc, f1_esc, _ = precision_recall_fscore_support(y_true_esc, y_pred_esc, pos_label=True, average='binary', zero_division=0)
        p_auto, r_auto, f1_auto, _ = precision_recall_fscore_support(y_true_esc, y_pred_esc, pos_label=False, average='binary', zero_division=0)

        reason_matches = sum(str(tr) == str(pr) for tr, pr in zip(true_reasons, pred_reasons))
        reason_exact_match = round(reason_matches / max(len(true_reasons), 1), 4)
        positives = int(np.sum(np.asarray(y_true_esc, dtype=bool)))
        true_positives = int(np.sum(
            np.asarray(y_true_esc, dtype=bool) & np.asarray(y_pred_esc, dtype=bool)
        ))
        recall_lower = (
            float(beta.ppf(0.05, true_positives, positives - true_positives + 1))
            if true_positives > 0 else 0.0
        )

        return {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1_macro), 4),
            "escalate_f1": round(float(f1_esc), 4),
            "escalate_precision": round(float(p_esc), 4),
            "escalate_recall": round(float(r_esc), 4),
            "escalate_recall_one_sided_95_lower": round(recall_lower, 4),
            "escalate_positive_count": positives,
            "auto_handle_rate": round(float(np.mean(~np.asarray(y_pred_esc, dtype=bool))), 4),
            "auto_handle_f1": round(float(f1_auto), 4),
            "auto_handle_precision": round(float(p_auto), 4),
            "auto_handle_recall": round(float(r_auto), 4),
            "reason_exact_match": reason_exact_match
        }

    def bootstrap_classification_intervals(
        self,
        y_true: List[Any],
        y_pred: List[Any],
        metric: str,
        iterations: int = 2000,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Non-parametric paired bootstrap CI for headline classification metrics."""
        if len(y_true) != len(y_pred) or not y_true:
            raise ValueError("Bootstrap inputs must be non-empty and aligned")
        rng = np.random.default_rng(seed)
        truth = np.asarray(y_true)
        pred = np.asarray(y_pred)
        values = []
        for _ in range(iterations):
            idx = rng.integers(0, len(truth), size=len(truth))
            if metric == "accuracy":
                value = accuracy_score(truth[idx], pred[idx])
            elif metric == "macro_f1":
                value = precision_recall_fscore_support(
                    truth[idx], pred[idx], average="macro", zero_division=0
                )[2]
            elif metric == "escalate_recall":
                value = precision_recall_fscore_support(
                    truth[idx].astype(bool), pred[idx].astype(bool),
                    pos_label=True, average="binary", zero_division=0
                )[1]
            elif metric == "escalate_precision":
                value = precision_recall_fscore_support(
                    truth[idx].astype(bool), pred[idx].astype(bool),
                    pos_label=True, average="binary", zero_division=0
                )[0]
            else:
                raise ValueError(f"Unsupported bootstrap metric: {metric}")
            values.append(float(value))
        low, high = np.quantile(values, [0.025, 0.975])
        return {"lower": round(float(low), 4), "upper": round(float(high), 4), "iterations": iterations}

    def coverage_accuracy_curve(
        self,
        y_true: List[str],
        y_pred: List[str],
        confidences: List[float],
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Report the accuracy/coverage tradeoff without claiming conformal guarantees."""
        thresholds = thresholds or [round(v, 2) for v in np.arange(0.0, 0.96, 0.05)]
        truth = np.asarray(y_true)
        pred = np.asarray(y_pred)
        conf = np.asarray(confidences, dtype=float)
        rows = []
        for threshold in thresholds:
            retained = conf >= threshold
            coverage = float(np.mean(retained))
            accuracy = float(np.mean(truth[retained] == pred[retained])) if retained.any() else None
            rows.append({
                "confidence_threshold": threshold,
                "coverage": round(coverage, 4),
                "retained_accuracy": round(accuracy, 4) if accuracy is not None else None,
                "abstain_count": int(np.sum(~retained)),
            })
        return rows

    def _compute_bleu(self, reference: str, candidate: str, max_n: int = 4) -> float:
        """Computes sentence-level BLEU score with Chen-Cherry Laplace smoothing."""
        ref_tokens = re.findall(r"\w+", reference.lower())
        cand_tokens = re.findall(r"\w+", candidate.lower())
        
        if not cand_tokens or not ref_tokens:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
            
            if not cand_ngrams:
                # Chen-Cherry smoothing for empty higher-order n-grams
                precisions.append(1.0 / (len(cand_tokens) + 1.0))
                continue
                
            cand_counts = Counter(cand_ngrams)
            ref_counts = Counter(ref_ngrams)
            clipped_matches = sum(min(cand_counts[ng], ref_counts[ng]) for ng in cand_counts)
            
            # Laplace / Chen-Cherry smoothing: strictly positive
            smoothed_p = (clipped_matches + 0.1) / (len(cand_ngrams) + 0.1)
            precisions.append(smoothed_p)

        # Brevity penalty
        bp = math.exp(1 - (len(ref_tokens) / len(cand_tokens))) if len(cand_tokens) < len(ref_tokens) else 1.0
        log_avg = sum(math.log(p) for p in precisions) / max_n
        return bp * math.exp(log_avg)

    def _compute_rouge_l(self, reference: str, candidate: str) -> float:
        """Computes LCS-based ROUGE-L F1 score."""
        ref_tokens = re.findall(r"\w+", reference.lower())
        cand_tokens = re.findall(r"\w+", candidate.lower())

        m, n = len(ref_tokens), len(cand_tokens)
        if m == 0 or n == 0:
            return 0.0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs = dp[m][n]
        prec = lcs / n
        rec = lcs / m
        if prec + rec == 0:
            return 0.0
        return (2 * prec * rec) / (prec + rec)

    def evaluate_replies(
        self,
        references: List[str],
        candidates: List[str],
        char_limit: int = 280
    ) -> Dict[str, Any]:
        """Calculates linguistic overlap and Twitter compliance."""
        rouge_scores = []
        bleu_scores = []
        length_compliant = 0

        for ref, cand in zip(references, candidates):
            rouge_scores.append(self._compute_rouge_l(ref, cand))
            bleu_scores.append(self._compute_bleu(ref, cand))
            if len(cand) <= char_limit:
                length_compliant += 1

        n = max(len(references), 1)
        return {
            "rougeL_f1": round(float(np.mean(rouge_scores)), 4),
            "bleu4": round(float(np.mean(bleu_scores)), 4),
            "twitter_length_compliance_rate": round(length_compliant / n, 4),
            "avg_character_length": round(float(np.mean([len(c) for c in candidates])), 1)
        }

    def compute_ece(
        self,
        y_true: List[str],
        y_pred: List[str],
        confidences: List[float],
        n_bins: int = 10,
        strategy: str = "quantile"
    ) -> Dict[str, Any]:
        """
        Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        Supports equal-width or quantile binning.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        conf_arr = np.array(confidences)
        n = len(conf_arr)

        correctness = (y_true_arr == y_pred_arr).astype(float)
        
        if strategy == "quantile" and n >= n_bins:
            # Equal frequency binning
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.quantile(conf_arr, quantiles)
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0001
        else:
            bin_edges = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        mce = 0.0
        bin_records = []

        for i in range(n_bins):
            b_min, b_max = bin_edges[i], bin_edges[i+1]
            mask = (conf_arr >= b_min) & (conf_arr < b_max)
            bin_size = int(np.sum(mask))

            if bin_size > 0:
                bin_acc = float(np.mean(correctness[mask]))
                bin_conf = float(np.mean(conf_arr[mask]))
                abs_diff = abs(bin_acc - bin_conf)
                ece += (bin_size / n) * abs_diff
                mce = max(mce, abs_diff)

                bin_records.append({
                    "bin": i,
                    "range": [round(float(b_min), 3), round(float(b_max), 3)],
                    "count": bin_size,
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                    "gap": round(abs_diff, 4)
                })

        return {
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4),
            "strategy": strategy,
            "bins": bin_records
        }

    def optimize_cost_threshold(
        self,
        y_true_esc: List[bool],
        risk_scores: List[float],
        cost_escalate: float = 1.0,
        cost_error_leak: float = 10.0
    ) -> Dict[str, Any]:
        """
        Decision-theoretic risk-threshold optimizer with realistic cost penalties.
        Loss(T) = c_esc * FP(T) + c_leak * FN(T)
        """
        y_true_arr = np.array(y_true_esc, dtype=bool)
        scores_arr = np.array(risk_scores)
        
        candidate_thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
        best_t = 0.50
        min_loss = float("inf")
        curve = []

        for t in candidate_thresholds:
            y_pred_t = (scores_arr >= t)
            fp = int(np.sum((y_pred_t == True) & (y_true_arr == False)))
            fn = int(np.sum((y_pred_t == False) & (y_true_arr == True)))
            tp = int(np.sum((y_pred_t == True) & (y_true_arr == True)))
            tn = int(np.sum((y_pred_t == False) & (y_true_arr == False)))

            loss = (cost_escalate * fp) + (cost_error_leak * fn)
            curve.append({
                "threshold": t,
                "loss": round(float(loss), 1),
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "tn": tn
            })

            if loss < min_loss:
                min_loss = loss
                best_t = t

        return {
            "optimal_threshold": best_t,
            "minimum_loss": round(float(min_loss), 1),
            "cost_escalate": cost_escalate,
            "cost_error_leak": cost_error_leak,
            "cost_curve": curve
        }

    @staticmethod
    def precision_at_recall_frontier(
        y_true_esc: Sequence[bool],
        risk_scores: Sequence[float],
        recall_floors: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95, 1.00),
    ) -> List[Dict[str, float]]:
        """Highest precision achievable at each recall floor.

        A single operating point hides the trade-off, and for a safety-critical
        route the useful question is not "what is precision" but "what precision
        can this model hold while still catching X% of escalations". Each row
        reports the most precise threshold whose recall still meets the floor, so
        the gap between rows is exactly what the model buys per point of recall.
        """
        truth = np.asarray(y_true_esc, dtype=bool)
        scores = np.asarray(risk_scores, dtype=float)
        positives = int(truth.sum())
        if positives == 0:
            return []
        candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
        frontier: List[Dict[str, float]] = []
        for floor in recall_floors:
            best_precision = 0.0
            best_threshold = 1.0
            best_recall = 1.0
            reached = False
            for threshold in candidates:
                predicted = scores >= threshold
                tp = int(np.sum(predicted & truth))
                fp = int(np.sum(predicted & ~truth))
                recall = tp / positives
                if recall + 1e-12 < floor:
                    continue
                reached = True
                precision = tp / max(tp + fp, 1)
                # Among thresholds meeting the floor, prefer precision, then the
                # highest threshold (fewest escalations) to break ties upward.
                if (precision, threshold) > (best_precision, best_threshold):
                    best_precision = precision
                    best_threshold = float(threshold)
                    best_recall = recall
            frontier.append({
                "recall_floor": float(floor),
                "reached": bool(reached),
                "threshold": round(best_threshold, 4),
                "precision": round(best_precision, 4),
                "recall": round(best_recall, 4),
                "escalations": None,
            })
        return frontier

    @staticmethod
    def paired_mcnemar(
        truth: Sequence[Any], predictions_a: Sequence[Any], predictions_b: Sequence[Any]
    ) -> Dict[str, Any]:
        """Exact paired test using only cases on which system correctness differs."""
        expected = np.asarray(truth, dtype=object)
        first = np.asarray(predictions_a, dtype=object)
        second = np.asarray(predictions_b, dtype=object)
        if expected.shape != first.shape or expected.shape != second.shape:
            raise ValueError("Truth and paired predictions must have identical shapes")
        first_only = int(np.sum((first == expected) & (second != expected)))
        second_only = int(np.sum((first != expected) & (second == expected)))
        discordant = first_only + second_only
        p_value = (
            float(binomtest(min(first_only, second_only), discordant, 0.5).pvalue)
            if discordant else 1.0
        )
        return {
            "a_correct_b_wrong": first_only,
            "a_wrong_b_correct": second_only,
            "discordant_count": discordant,
            "exact_two_sided_p_value": round(p_value, 6),
        }
