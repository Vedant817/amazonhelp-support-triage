"""
Unit tests for Energy-Based Out-of-Distribution Detector (src/ood_detector.py).
"""

import unittest
import numpy as np
from src.ood_detector import EnergyOODDetector

class TestEnergyOODDetector(unittest.TestCase):
    def setUp(self):
        self.detector = EnergyOODDetector(temperature=1.0, quantile_threshold=0.05)

    def test_compute_energy_log_sum_exp_stability(self):
        # Large logits that would overflow naive exp
        large_logits = np.array([1000.0, 999.0, 950.0])
        energy = self.detector.compute_energy(large_logits)
        self.assertTrue(np.isfinite(energy))
        self.assertLess(energy, 0.0)

    def test_calibration_and_detection(self):
        # Synthetic in-distribution logits: sharp confidence on true class
        id_logits = [np.array([4.0, 0.1, -1.0, -2.0]) for _ in range(100)]
        self.detector.calibrate(id_logits)
        self.assertTrue(self.detector.calibrated_)
        self.assertIsNotNone(self.detector.energy_threshold_)

        # In-distribution sample
        res_id = self.detector.score(np.array([4.5, 0.0, -1.0, -2.0]))
        self.assertFalse(res_id["is_ood"])
        self.assertEqual(res_id["status"], "IN_DISTRIBUTION")

        # OOD sample: uniform, low confidence / high entropy
        res_ood = self.detector.score(np.array([0.01, 0.02, 0.01, -0.01]))
        self.assertTrue(res_ood["is_ood"])
        self.assertEqual(res_ood["status"], "OUT_OF_DISTRIBUTION")

    def test_single_scalar_handling(self):
        res = self.detector.score(np.array([2.5]))
        self.assertIn("energy_score", res)
        self.assertIn("status", res)

if __name__ == "__main__":
    unittest.main()
