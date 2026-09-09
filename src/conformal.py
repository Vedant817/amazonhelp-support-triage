"""
Split Conformal Prediction for Guaranteed Classification Coverage.
Implements Vovk et al. and Angelopoulos & Bates (2023).
Calibrates non-conformity scores (1 - P(Y = y | X)) on a held-out calibration set
and constructs prediction sets C(X) satisfying finite-sample coverage guarantees:
    P( Y in C(X) ) >= 1 - alpha
"""

import numpy as np
from typing import List, Dict, Any, Optional

class ConformalPredictor:
    """
    Split Conformal Prediction Engine for @AmazonHelp intent classification.
    Provides mathematical marginal coverage guarantees under exchangeability.
    """
    def __init__(self, alpha: float = 0.05):
        """
        alpha: Error rate tolerance (e.g. alpha=0.05 guarantees >= 95% coverage)
        """
        self.alpha = alpha
        self.q_hat = 1.0
        self.classes = None
        self.is_calibrated = False

    def calibrate(
        self,
        probs: np.ndarray,
        y_true: List[str],
        classes: List[str]
    ) -> "ConformalPredictor":
        """
        Calibrate quantile on holdout calibration split using exact finite-sample order statistics.
        
        Args:
            probs: Predicted probability matrix of shape (n_cal, n_classes)
            y_true: True class labels of length n_cal
            classes: Ordered list of class names corresponding to columns of probs
        """
        self.classes = list(classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")
            
        non_conformity_scores = []
        for i in range(n):
            true_label = y_true[i]
            if true_label in class_to_idx:
                col_idx = class_to_idx[true_label]
                prob_true = probs[i, col_idx]
                s = 1.0 - prob_true
                non_conformity_scores.append(s)
                
        scores = np.array(non_conformity_scores)
        
        # Exact finite-sample conformal quantile: k = ceil((n + 1) * (1 - alpha))
        # Index in 0-based sorted array: k - 1
        k = int(np.ceil((n + 1.0) * (1.0 - self.alpha)))
        k = min(max(1, k), n)
        sorted_scores = np.sort(scores)
        self.q_hat = float(sorted_scores[k - 1])
        self.is_calibrated = True
        
        return self

    def predict_set(self, prob_vector: np.ndarray) -> Dict[str, Any]:
        """
        Generate conformal prediction set for a single query probability vector.
        Does NOT mutate empty sets; empty set means non-coverage (abstain).
        
        Returns:
            Dict containing:
                - prediction_set: list of classes in C(x)
                - set_size: |C(x)|
                - is_singleton: True if exactly one class
                - should_abstain: True if |C(x)| != 1 (ambiguous or empty)
                - is_uncertain: Alias for should_abstain
                - confidence: 1 - q_hat
                - non_conformity_scores: dict of {class: 1 - prob}
        """
        if not self.is_calibrated:
            best_idx = int(np.argmax(prob_vector))
            best_cls = self.classes[best_idx] if self.classes else "FEEDBACK_AND_GENERAL"
            return {
                "prediction_set": [best_cls],
                "set_size": 1,
                "is_singleton": True,
                "should_abstain": False,
                "is_uncertain": False,
                "confidence": 1.0,
                "conformal_quantile": 1.0,
                "non_conformity_scores": {}
            }
            
        pred_set = []
        scores = {}
        for idx, cls_name in enumerate(self.classes):
            prob = prob_vector[idx]
            s = 1.0 - prob
            scores[cls_name] = round(float(s), 4)
            if s <= self.q_hat:
                pred_set.append(cls_name)
                
        # Pure conformal prediction: empty set means out-of-distribution / high uncertainty
        set_size = len(pred_set)
        should_abstain = (set_size != 1)
        is_singleton = (set_size == 1)
            
        return {
            "prediction_set": pred_set,
            "set_size": set_size,
            "is_singleton": is_singleton,
            "should_abstain": should_abstain,
            "is_uncertain": should_abstain,
            "confidence": round(float(1.0 - self.q_hat), 4),
            "conformal_quantile": round(self.q_hat, 4),
            "non_conformity_scores": scores
        }

    def evaluate_coverage(self, prob_matrix: np.ndarray, y_true: List[str]) -> Dict[str, Any]:
        """
        Evaluates empirical coverage and set size metrics on an out-of-sample test set.
        """
        covered = 0
        set_sizes = []
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for i, true_label in enumerate(y_true):
            res = self.predict_set(prob_matrix[i])
            pset = res["prediction_set"]
            set_sizes.append(len(pset))
            if true_label in pset:
                covered += 1
                
        n = len(y_true)
        empirical_coverage = covered / max(n, 1)
        avg_set_size = float(np.mean(set_sizes))
        singleton_rate = float(np.mean([1 if s == 1 else 0 for s in set_sizes]))
        empty_rate = float(np.mean([1 if s == 0 else 0 for s in set_sizes]))
        
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": round(empirical_coverage, 4),
            "average_set_size": round(avg_set_size, 3),
            "singleton_rate": round(singleton_rate, 4),
            "empty_rate": round(empty_rate, 4),
            "conformal_quantile": round(self.q_hat, 4)
        }
