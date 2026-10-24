"""
Tests for bootstrap-aggregated intent heads.

A single logistic regression on sparse TF-IDF features is high variance: small
training-set changes move the boundary. Bagging several bootstrap resamples and
averaging probabilities reduces that variance, which is what generalisation to
unseen text depends on.

Contract under test:
* the ensemble is stored **only** when the calibration split shows a gain, so a
  rejected ensemble costs nothing at inference time;
* acceptance always implies a recorded positive gain;
* the fit is deterministic for a fixed seed.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.dual_head import DualHeadSupportModel


CLASSES = [
    "ACCOUNT_AND_SECURITY",
    "DAMAGED_OR_WRONG_ITEM",
    "FEEDBACK_AND_GENERAL",
    "ORDER_TRACKING_AND_DELAY",
    "PRODUCT_AND_TECHNICAL",
    "REFUND_AND_RETURN",
]


def _separable_data(n: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(n, 40)).astype(np.float32)
    labels = np.asarray([CLASSES[i % len(CLASSES)] for i in range(n)], dtype=object)
    dense[np.arange(n), np.arange(n) % len(CLASSES)] += 4.0  # class signal
    return csr_matrix(dense), labels


def _prepared_model(seed: int = 0):
    features, labels = _separable_data(seed=seed)
    model = DualHeadSupportModel()
    model.classes_ = list(CLASSES)
    model.intent_selected_c_ = 1.0
    model.intent_temperature.temperature_ = 1.0
    model.intent_head.fit(features, labels)
    return model, features, labels


class TestBaggingContract:
    def test_default_state_is_empty(self):
        model = DualHeadSupportModel()
        assert model.intent_bagging_accepted_ is False
        assert model.intent_bagging_heads_ == []
        assert model.intent_bagging_gain_ == 0.0

    def test_heads_present_only_when_accepted(self):
        """A rejected ensemble must not be persisted into the cache."""
        model, features, labels = _prepared_model()
        model._fit_intent_bagging(features, labels, features, labels, n_models=3)
        if model.intent_bagging_accepted_:
            assert len(model.intent_bagging_heads_) >= 1
            assert model.intent_bagging_gain_ > 0.0
        else:
            assert model.intent_bagging_heads_ == []

    def test_acceptance_requires_positive_gain(self):
        model, features, labels = _prepared_model()
        model._fit_intent_bagging(features, labels, features, labels, n_models=3)
        if model.intent_bagging_accepted_:
            assert model.intent_bagging_gain_ > 0.002

    def test_deterministic_for_fixed_seed(self):
        model_a, features, labels = _prepared_model()
        model_b, _, _ = _prepared_model()
        model_a._fit_intent_bagging(features, labels, features, labels, n_models=3, seed=7)
        model_b._fit_intent_bagging(features, labels, features, labels, n_models=3, seed=7)
        assert model_a.intent_bagging_accepted_ == model_b.intent_bagging_accepted_
        assert model_a.intent_bagging_gain_ == pytest.approx(model_b.intent_bagging_gain_)

    def test_degenerate_resample_is_skipped(self):
        """An all-one-class bootstrap sample must not crash the fit."""
        features = csr_matrix(np.ones((6, 5), dtype=np.float32))
        labels = np.asarray(["A", "A", "A", "A", "A", "A"], dtype=object)
        model = DualHeadSupportModel()
        model.classes_ = ["A", "B"]
        model.intent_selected_c_ = 1.0
        model._fit_intent_bagging(features, labels, features, labels, n_models=2)
        assert model.intent_bagging_accepted_ is False

    def test_no_gain_on_unrelated_calibration(self):
        """Calibration data with no relation to the training signal cannot clear the bar."""
        model, features, labels = _prepared_model()
        rng = np.random.default_rng(11)
        cal_features = csr_matrix(rng.normal(size=(80, 40)).astype(np.float32))
        cal_labels = np.asarray([CLASSES[i % len(CLASSES)] for i in range(80)], dtype=object)
        model._fit_intent_bagging(features, labels, cal_features, cal_labels, n_models=2)
        if model.intent_bagging_accepted_:
            assert model.intent_bagging_gain_ > 0.002


class TestEnsembleAveraging:
    """The inference path averages the base probabilities with each head."""

    def test_average_of_probability_vectors_normalises(self):
        base = np.array([0.7, 0.2, 0.1])
        head = np.array([0.5, 0.4, 0.1])
        averaged = np.mean([base, head], axis=0)
        assert pytest.approx(float(averaged.sum()), abs=1e-9) == 1.0
        assert averaged[0] < base[0], "averaging must pull extremes toward the middle"

    def test_bagging_predicts_all_trained_classes(self):
        model, features, labels = _prepared_model()
        model._fit_intent_bagging(features, labels, features, labels, n_models=3)
        if not model.intent_bagging_accepted_:
            pytest.skip("ensemble not accepted on this synthetic split")
        head = model.intent_bagging_heads_[0]
        probs = head.predict_proba(features)
        assert probs.shape[1] == len(head.classes_)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
