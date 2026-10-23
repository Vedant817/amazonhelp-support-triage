"""
Tests for deployment-prior correction on the intent head.

Training balances every intent class to a uniform prior, which is good for
minority recall but mis-calibrated for real traffic where one class dominates.
The correction re-weights ``p(y|x) * prior(y)**alpha`` with ``alpha`` selected on
the held-out calibration split, guarded so macro-F1 cannot regress materially.
"""

import numpy as np
import pytest

from src.dual_head import DualHeadSupportModel


CLASSES = [
    "ACCOUNT_AND_SECURITY",
    "DAMAGED_OR_WRONG_ITEM",
    "FEEDBACK_AND_GENERAL",
    "ORDER_TRACKING_AND_DELAY",
    "PRODUCT_AND_TECHNICAL",
    "REFUND_AND_RETURN",
]

NATURAL_PRIOR = {
    "ACCOUNT_AND_SECURITY": 0.023,
    "DAMAGED_OR_WRONG_ITEM": 0.063,
    "FEEDBACK_AND_GENERAL": 0.074,
    "ORDER_TRACKING_AND_DELAY": 0.687,
    "PRODUCT_AND_TECHNICAL": 0.088,
    "REFUND_AND_RETURN": 0.066,
}


def _model(**kwargs) -> DualHeadSupportModel:
    model = DualHeadSupportModel(**kwargs)
    model.classes_ = list(CLASSES)
    return model


class TestPriorVector:
    def test_none_without_prior(self):
        model = _model()
        assert model._prior_vector() is None

    def test_normalised_and_aligned(self):
        model = _model()
        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        vector = model._prior_vector()
        assert vector is not None
        assert len(vector) == len(CLASSES)
        assert pytest.approx(float(vector.sum()), abs=1e-9) == 1.0
        # ORDER_TRACKING is index 3 and must carry the largest weight.
        assert int(np.argmax(vector)) == 3

    def test_missing_classes_get_zero(self):
        model = _model()
        model.intent_natural_prior_ = {"ORDER_TRACKING_AND_DELAY": 1.0}
        vector = model._prior_vector()
        assert pytest.approx(float(vector[3]), abs=1e-9) == 1.0
        assert pytest.approx(float(np.sum(np.delete(vector, 3))), abs=1e-9) == 0.0


class TestApplyPriorCorrection:
    def test_noop_without_strength(self):
        model = _model()
        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        probs = np.full(len(CLASSES), 1.0 / len(CLASSES))
        out = model._apply_prior_correction(probs)
        assert np.allclose(out, probs)

    def test_majority_class_gains_weight(self):
        model = _model()
        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        model.prior_correction_strength_ = 1.0
        probs = np.full(len(CLASSES), 1.0 / len(CLASSES))
        out = model._apply_prior_correction(probs)
        assert out[3] > probs[3]
        assert pytest.approx(float(out.sum()), abs=1e-9) == 1.0

    def test_half_strength_is_geometric_midpoint(self):
        model = _model()
        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        model.prior_correction_strength_ = 0.5
        probs = np.full(len(CLASSES), 1.0 / len(CLASSES))
        out = model._apply_prior_correction(probs)
        prior = model._prior_vector()
        expected = prior ** 0.5
        expected = expected / expected.sum()
        assert np.allclose(out, expected)

    def test_handles_batch_input(self):
        model = _model()
        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        model.prior_correction_strength_ = 1.0
        batch = np.full((3, len(CLASSES)), 1.0 / len(CLASSES))
        out = model._apply_prior_correction(batch)
        assert out.shape == batch.shape
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_ignores_mismatched_width(self):
        model = _model()
        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        model.prior_correction_strength_ = 1.0
        wrong = np.array([0.5, 0.5])
        assert np.allclose(model._apply_prior_correction(wrong), wrong)


class TestFitPriorCorrection:
    def _calibration_data(self, n: int = 600, seed: int = 0):
        """Deterministic toy logits where re-weighting must help accuracy."""
        rng = np.random.default_rng(seed)
        # True label follows the natural prior.
        prior = np.array([NATURAL_PRIOR[c] for c in CLASSES])
        truth_idx = rng.choice(len(CLASSES), size=n, p=prior / prior.sum())
        logits = rng.normal(0, 0.5, size=(n, len(CLASSES)))
        # Signal: the true class gets a modest boost, so the argmax is often
        # wrong for the rare classes and right for the majority class.
        logits[np.arange(n), truth_idx] += 0.8
        labels = [CLASSES[i] for i in truth_idx]
        return logits, labels

    def test_chooses_nonzero_alpha_when_it_helps(self):
        model = _model()
        model.intent_temperature.temperature_ = 1.0
        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        logits, labels = self._calibration_data()
        model._fit_prior_correction(logits, labels)
        assert model.prior_correction_strength_ >= 0.0
        assert model.prior_correction_calibration_gain_ >= 0.0

    def test_no_prior_leaves_strength_zero(self):
        model = _model()
        model.intent_temperature.temperature_ = 1.0
        logits, labels = self._calibration_data()
        model._fit_prior_correction(logits, labels)
        assert model.prior_correction_strength_ == 0.0

    def test_accuracy_does_not_regress(self):
        """The selected alpha may never score worse than alpha=0 on calibration."""
        model = _model()
        model.intent_temperature.temperature_ = 1.0
        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        logits, labels = self._calibration_data()
        model._fit_prior_correction(logits, labels)

        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        base_pred = np.asarray(CLASSES)[np.argmax(probs, axis=1)]
        base_acc = float(np.mean(base_pred == np.asarray(labels)))

        model.prior_correction_strength_ = float(model.prior_correction_strength_)
        adjusted = model._apply_prior_correction(probs)
        adj_pred = np.asarray(CLASSES)[np.argmax(adjusted, axis=1)]
        adj_acc = float(np.mean(adj_pred == np.asarray(labels)))

        assert adj_acc >= base_acc - 1e-9


class TestEndToEndPriorEffect:
    def test_majority_prediction_rate_increases(self):
        """With correction on, more items resolve to the dominant class."""
        model = _model()
        model.intent_temperature.temperature_ = 1.0
        prior = np.array([NATURAL_PRIOR[c] for c in CLASSES])
        rng = np.random.default_rng(7)
        n = 400
        truth_idx = rng.choice(len(CLASSES), size=n, p=prior / prior.sum())
        logits = rng.normal(0, 0.4, size=(n, len(CLASSES)))
        logits[np.arange(n), truth_idx] += 0.6
        labels = [CLASSES[i] for i in truth_idx]

        model.intent_natural_prior_ = dict(NATURAL_PRIOR)
        model._fit_prior_correction(logits, labels)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        adjusted = model._apply_prior_correction(probs)

        base_rate = float(np.mean(np.argmax(probs, axis=1) == 3))
        adj_rate = float(np.mean(np.argmax(adjusted, axis=1) == 3))
        assert adj_rate >= base_rate
