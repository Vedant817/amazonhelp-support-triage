"""
Conformal Prediction Engine for Support Automation.
Implements Split Conformal Prediction with exact finite-sample coverage guarantees.
Provides:
1. Conformal Intent Prediction Sets: C(x) = {y: s(x, y) <= q_alpha} guaranteeing P(Y in C(x)) >= 1 - alpha.
2. Conformal Escalation Abstention: Automatically abstains (escalates to human) when prediction set size != 1
   or when non-conformity exceeds calibrated safety threshold.
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple

class ConformalPredictor:
    """
    Split Conformal Predictor for multi-class classification and risk-controlled selective prediction.
    Uses exact finite-sample order statistic indexing:
    k = ceil((n + 1) * (1 - alpha))
    q_hat = sorted_scores[k - 1]
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (error rate tolerance). 1 - alpha is target coverage.
                   e.g. alpha = 0.05 guarantees 95% marginal coverage.
        """
        self.alpha = alpha
        self.q_hat: float = 1.0
        self.classes: List[str] = []
        self.is_calibrated: bool = False

    def calibrate(self, probs: np.ndarray, y_true: List[str], classes: List[str]) -> "ConformalPredictor":
        """
        Calibrate conformal quantile on a holdout calibration set.
        
        Args:
            probs: Predicted class probabilities (N, num_classes)
            y_true: True class labels of length N
            classes: List of class names matching columns of probs
        """
        self.classes = list(classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")
            
        # Non-conformity score: s_i = 1 - P(Y = y_i | X = x_i)
        scores = []
        for i in range(n):
            true_cls = y_true[i]
            if true_cls in class_to_idx:
                idx = class_to_idx[true_cls]
                prob = probs[i, idx]
            else:
                prob = 0.0
            scores.append(1.0 - prob)
            
        scores = np.array(scores)
        
        # Exact finite-sample order statistic calculation:
        # k = ceil((n + 1) * (1 - alpha))
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(n, max(1, k))
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
            "conformal_quantile": round(self.q_hat, 4),
            "non_conformity_scores": scores
        }
