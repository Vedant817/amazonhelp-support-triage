"""
Out-of-Distribution (OOD) & Cross-Domain Intent Detector for @AmazonHelp.
Implements Energy-based OOD detection (Liu et al., NeurIPS 2020) and Max-Logit Scoring.
Detects non-retail queries (e.g. banking transactions, airline bookings, off-brand inquiries, spam)
to prevent silent misclassification and hallucinated automated handling.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional

class EnergyOODDetector:
    """
    Energy-based Out-of-Distribution Detector.
    Free Energy: E(x; T) = -T * log( sum( exp(z_i / T) ) )
    In-distribution samples have significantly lower free energy (higher negative energy).
    OOD samples (cross-domain banking, off-topic spam) have higher free energy.
    """
    def __init__(self, temperature: float = 1.0, quantile_threshold: float = 0.05):
        self.temperature = temperature
        self.quantile_threshold = quantile_threshold
        self.energy_threshold_ = None
        self.calibrated_ = False

    def compute_energy(self, logits: np.ndarray) -> float:
        """Computes free energy score for a given logit vector with log-sum-exp stabilization."""
        logits = np.atleast_1d(np.asarray(logits, dtype=float))
        if len(logits) == 1:
            logits = np.array([-logits[0], logits[0]])
        scaled = logits / max(0.01, self.temperature)
        max_s = float(np.max(scaled))
        lse = max_s + math.log(max(1e-12, float(np.sum(np.exp(scaled - max_s)))))
        energy = -self.temperature * lse
        return float(energy)

    def calibrate(self, in_distribution_logits: List[np.ndarray]) -> "EnergyOODDetector":
        """
        Calibrate energy threshold on in-distribution holdout data.
        Sets energy threshold at (1 - quantile_threshold) percentile of in-distribution energy.
        """
        energies = [self.compute_energy(logits) for logits in in_distribution_logits]
        self.energy_threshold_ = float(np.quantile(energies, 1.0 - self.quantile_threshold))
        self.calibrated_ = True
        return self

    def score(self, logits: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates logits for out-of-distribution detection.
        Returns:
            is_ood: bool (True if sample is out-of-distribution)
            energy: float (calculated energy score)
            threshold: float (calibrated threshold)
        """
        energy = self.compute_energy(logits)
        if not self.calibrated_:
            k_classes = max(2, len(np.atleast_1d(logits)))
            uniform_energy = -self.temperature * math.log(k_classes)
            is_ood = energy >= uniform_energy * 0.85
            threshold = uniform_energy * 0.85
        else:
            is_ood = energy > self.energy_threshold_
            threshold = self.energy_threshold_

        return {
            "is_ood": bool(is_ood),
            "energy_score": round(float(energy), 4),
            "energy_threshold": round(float(threshold), 4) if threshold is not None else None,
            "status": "OUT_OF_DISTRIBUTION" if is_ood else "IN_DISTRIBUTION"
        }
