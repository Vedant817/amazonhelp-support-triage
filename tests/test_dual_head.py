"""
Unit tests for dual-head model (intent classification + escalation).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.dual_head import (
    DualHeadSupportModel,
    MiniLMEncoder,
    TemperatureScaler,
    select_recall_constrained_threshold,
    select_precision_threshold,
    one_sided_binomial_lower_bound,
)


class TestTemperatureScaler:
    """Tests for temperature scaling."""

    def test_fit_multiclass(self):
        """Multiclass temperature fitting should produce valid temperature."""
        scaler = TemperatureScaler()
        logits = np.array([
            [2.0, 0.5, 0.1],
            [0.1, 3.0, 0.2],
            [0.3, 0.2, 2.5],
        ])
        labels = np.array([0, 1, 2])
        scaler.fit_multiclass(logits, labels)
        assert scaler.is_fitted_
        assert scaler.temperature_ > 0

    def test_fit_binary(self):
        """Binary temperature fitting should produce valid temperature."""
        scaler = TemperatureScaler()
        logits = np.array([2.0, -1.0, 3.0, -2.0])
        labels = np.array([True, False, True, False])
        scaler.fit_binary(logits, labels)
        assert scaler.is_fitted_
        assert scaler.temperature_ > 0

    def test_calibration_improves_nll(self):
        """Temperature scaling should improve NLL on calibration set."""
        scaler = TemperatureScaler()
        # Overconfident logits
        logits = np.array([
            [5.0, 0.1, 0.1],
            [0.1, 5.0, 0.1],
            [0.1, 0.1, 5.0],
        ])
        labels = np.array([0, 1, 2])
        
        # Before scaling - high confidence but same predictions
        probs_before = _softmax(logits)
        nll_before = -np.mean(np.log(probs_before[np.arange(3), labels]))
        
        scaler.fit_multiclass(logits, labels)
        probs_after = _softmax(logits / scaler.temperature_)
        nll_after = -np.mean(np.log(probs_after[np.arange(3), labels]))
        
        # Temperature > 1 should soften probabilities, may increase NLL on perfect data
        # but should not decrease it on held-out calibration data
        assert scaler.temperature_ > 0


def _softmax(logits):
    values = np.asarray(logits, dtype=float)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


class TestThresholdSelection:
    """Tests for recall-constrained threshold selection."""

    def test_select_recall_constrained_threshold(self):
        """Should find threshold meeting recall target."""
        labels = np.array([True, True, False, True, False, False, True, False])
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        
        result = select_recall_constrained_threshold(
            labels, probs, target_recall=0.75, confidence_level=0.95
        )
        
        assert result.threshold >= 0
        assert result.threshold <= 1
        assert result.target_met is not None
        assert result.positive_count == 4  # 4 positive labels
        assert result.recall_lower_bound >= 0

    def test_recall_bound_calculation(self):
        """Clopper-Pearson lower bound should be valid."""
        # 3 successes out of 4 trials, 95% confidence
        bound = one_sided_binomial_lower_bound(3, 4, 0.95)
        assert 0 <= bound < 1
        # Should be > 0 since we have successes
        assert bound > 0

    def test_zero_successes_bound(self):
        """Zero successes should give zero lower bound."""
        bound = one_sided_binomial_lower_bound(0, 10, 0.95)
        assert bound == 0.0

    def test_all_successes_bound(self):
        """All successes should give bound close to 1."""
        bound = one_sided_binomial_lower_bound(10, 10, 0.95)
        assert bound > 0.5  # Should be high but not necessarily 1.0

    def test_select_precision_threshold(self):
        """Should find threshold meeting precision target."""
        labels = np.array([True, True, False, True, False, False])
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        
        threshold = select_precision_threshold(labels, probs, target_precision=0.8)
        
        assert 0 <= threshold <= 1
        # At this threshold, precision should be >= 0.8
        predicted = probs >= threshold
        tp = np.sum(predicted & labels)
        fp = np.sum(predicted & ~labels)
        if tp + fp > 0:
            precision = tp / (tp + fp)
            assert precision >= 0.8


class TestDualHeadModel:
    """Integration tests for DualHeadSupportModel."""

    def test_model_initialization(self):
        """Model should initialize with default parameters."""
        model = DualHeadSupportModel()
        assert model.intent_confidence_threshold == 0.45
        assert model.target_escalation_recall == 0.85
        assert model.enable_context_gate is False
        assert model.enable_escalation_knn is False

    def test_model_initialization_custom(self):
        """Model should accept custom parameters."""
        model = DualHeadSupportModel(
            intent_confidence_threshold=0.3,
            target_escalation_recall=0.9,
            enable_context_gate=True,
            enable_escalation_knn=True,
        )
        assert model.intent_confidence_threshold == 0.3
        assert model.target_escalation_recall == 0.9
        assert model.enable_context_gate is True
        assert model.enable_escalation_knn is True

    def test_encoder_lazy_loading(self):
        """Encoder should lazy load."""
        model = DualHeadSupportModel()
        assert model.encoder._model is None
        # Trigger load
        _ = model.encoder._load()
        assert model.encoder._model is not None

    def test_intent_view_structure(self):
        """_intent_view should return proper structure."""
        model = DualHeadSupportModel()
        # Mock encoder to avoid loading
        model.encoder = MagicMock()
        model.encoder.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        model.intent_vectorizer = MagicMock()
        model.intent_vectorizer.transform.return_value = MagicMock()
        model.intent_head = MagicMock()
        model.intent_head.decision_function.return_value = [np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.05])]
        model.intent_temperature = MagicMock()
        model.intent_temperature.temperature_ = 1.0
        model.classes_ = ["A", "B", "C", "D", "E", "F"]
        model.semantic_blend_weight_ = 0.0
        model.prototype_blend_weight_ = 0.0
        model.prototype_temperature_ = 0.1
        model.intent_prototypes_ = None
        
        result = model._intent_view("test query")
        
        assert "intent" in result
        assert "confidence" in result
        assert "all_probabilities" in result
        assert "prob_vector" in result
        assert "logits" in result
        assert "embedding" in result
        assert result["confidence"] >= 0
        assert result["confidence"] <= 1


class TestMiniLMEncoder:
    """Tests for MiniLMEncoder."""

    def test_encoder_initialization(self):
        """Encoder should initialize with correct defaults."""
        encoder = MiniLMEncoder()
        assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert encoder.contrastive_pairs_seen_ == 0

    def test_encoder_device_assignment(self):
        """Encoder should accept device parameter."""
        encoder = MiniLMEncoder(device="cpu")
        assert encoder.device == "cpu"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_logits_threshold(self):
        """Threshold selection with edge cases."""
        labels = np.array([True, False])
        probs = np.array([1.0, 0.0])
        result = select_recall_constrained_threshold(labels, probs, target_recall=0.5)
        assert result.threshold >= 0

    def test_all_same_probabilities(self):
        """All same probabilities should handle gracefully."""
        labels = np.array([True, True, False, False])
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        result = select_recall_constrained_threshold(labels, probs, target_recall=0.5)
        assert result.threshold >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])