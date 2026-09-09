"""
Unit tests for Conformal Risk Control Framework (src/conformal_risk.py).
"""

import unittest
import numpy as np
from src.conformal_risk import ConformalRiskController

class TestConformalRiskController(unittest.TestCase):
    def setUp(self):
        self.crc = ConformalRiskController(target_risk_alpha=0.05)

    def test_calibration_bounds_risk(self):
        # 100 positive cases (high risk) with scores mostly high, a few low
        np.random.seed(42)
        scores_pos = list(np.random.uniform(0.6, 0.95, size=80)) + list(np.random.uniform(0.2, 0.5, size=20))
        scores_neg = list(np.random.uniform(0.05, 0.4, size=100))

        risk_scores = scores_pos + scores_neg
        y_true = [True] * len(scores_pos) + [False] * len(scores_neg)

        self.crc.calibrate(risk_scores, y_true)
        self.assertTrue(self.crc.calibrated_)
        self.assertGreater(self.crc.calibrated_threshold_, 0.0)

        # High risk score should trigger escalation
        dec_high = self.crc.decide(0.85)
        self.assertTrue(dec_high["should_escalate"])

        # Very low risk score should not trigger escalation
        dec_low = self.crc.decide(0.05)
        self.assertFalse(dec_low["should_escalate"])

    def test_empty_positives_fallback(self):
        # Edge case: no positive cases
        self.crc.calibrate([0.1, 0.2, 0.3], [False, False, False])
        self.assertTrue(self.crc.calibrated_)
        dec = self.crc.decide(0.5)
        self.assertIn("should_escalate", dec)

if __name__ == "__main__":
    unittest.main()
