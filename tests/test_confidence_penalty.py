"""Tests for the intent confidence penalty.

The penalty shrinks confidence on predictions that are both confident and wrong.
Getting it right requires three things to agree, and none of them did:

* **Same probabilities at fit and serve.** It was fitted on raw softmax outputs
  while the serving path applies it to temperature-scaled probabilities. A factor
  fitted in one space means something else in the other.
* **A selector observable in production.** It selected rows using the true
  class's probability, which is unknowable at request time, then applied the
  fitted factor to every request regardless.
* **Disabled means disabled.** ``intent_confidence_penalty=0`` set the factor to
  1.0, which replaces the distribution with the uniform one — the strongest
  possible penalty, not the absence of one.
"""

import numpy as np
import pytest

from src.dual_head import DualHeadSupportModel


CLASSES = ["A", "B", "C", "D"]


def _model() -> DualHeadSupportModel:
    model = DualHeadSupportModel()
    model.classes_ = list(CLASSES)
    model.intent_temperature.temperature_ = 1.0
    return model


def _logits_for(labels, correct_share: float, seed: int, sharpness: float = 3.0):
    """Logits that are right on ``correct_share`` of rows and confident throughout."""
    rng = np.random.default_rng(seed)
    logits = np.zeros((len(labels), len(CLASSES)))
    for row, label in enumerate(labels):
        target = CLASSES.index(label)
        if rng.random() > correct_share:
            target = (target + 1) % len(CLASSES)
        logits[row, target] = sharpness
    return logits


class TestDisabledMeansDisabled:
    def test_zero_penalty_knob_yields_no_penalty(self):
        model = DualHeadSupportModel(intent_confidence_penalty=0.0)
        labels = np.asarray([0, 1, 2, 3], dtype=int)
        model._fit_confidence_penalty(np.zeros((4, 4)), labels)
        assert model.confidence_penalty_factor_ == 0.0

    def test_zero_factor_is_identity(self):
        model = _model()
        model.confidence_penalty_factor_ = 0.0
        probs = np.asarray([0.7, 0.2, 0.05, 0.05])
        assert np.allclose(model._apply_confidence_penalty(probs), probs)

    def test_default_state_applies_nothing(self):
        """A freshly built model must not flatten its own probabilities."""
        model = DualHeadSupportModel()
        probs = np.asarray([0.9, 0.05, 0.03, 0.02])
        assert np.allclose(model._apply_confidence_penalty(probs), probs)


class TestApplyOnlyWhereFitted:
    def test_low_confidence_predictions_are_untouched(self):
        """The penalty was never fitted for unconfident rows, so it skips them."""
        model = _model()
        model.confidence_penalty_factor_ = 0.5
        probs = np.asarray([0.30, 0.25, 0.25, 0.20])  # below the 0.7 threshold
        assert np.allclose(model._apply_confidence_penalty(probs), probs)

    def test_confident_predictions_are_flattened(self):
        model = _model()
        model.confidence_penalty_factor_ = 0.5
        probs = np.asarray([0.90, 0.04, 0.03, 0.03])
        penalized = model._apply_confidence_penalty(probs)
        assert penalized[0] < probs[0]
        assert penalized.sum() == pytest.approx(1.0)

    def test_argmax_is_preserved(self):
        """Flattening toward uniform is monotone per row, so labels cannot flip."""
        model = _model()
        model.confidence_penalty_factor_ = 0.9
        probs = np.asarray([0.85, 0.10, 0.03, 0.02])
        assert np.argmax(model._apply_confidence_penalty(probs)) == int(np.argmax(probs))

    def test_two_dimensional_input_masks_per_row(self):
        model = _model()
        model.confidence_penalty_factor_ = 0.5
        probs = np.asarray([
            [0.90, 0.05, 0.03, 0.02],  # confident -> penalised
            [0.30, 0.28, 0.22, 0.20],  # unconfident -> untouched
        ])
        result = model._apply_confidence_penalty(probs)
        assert result[0, 0] < probs[0, 0]
        assert np.allclose(result[1], probs[1])
        assert np.allclose(result.sum(axis=1), 1.0)


class TestFitUsesTemperedProbabilities:
    def test_temperature_changes_the_fitted_factor(self):
        """Fitting must happen in the space the penalty is applied in."""
        labels = np.asarray([i % len(CLASSES) for i in range(200)], dtype=int)
        logits = _logits_for([CLASSES[i] for i in labels], correct_share=0.6, seed=1)

        cold = _model()
        cold.intent_temperature.temperature_ = 1.0
        cold._fit_confidence_penalty(logits, labels)

        warm = _model()
        warm.intent_temperature.temperature_ = 4.0
        warm._fit_confidence_penalty(logits, labels)

        # A flatter distribution has fewer rows above any confidence threshold, so
        # the two spaces cannot agree by accident.
        assert cold.confidence_penalty_factor_ != pytest.approx(warm.confidence_penalty_factor_)

    def test_no_confident_wrong_rows_disables_the_penalty(self):
        labels = np.asarray([i % len(CLASSES) for i in range(120)], dtype=int)
        logits = _logits_for([CLASSES[i] for i in labels], correct_share=1.0, seed=2)
        model = _model()
        model._fit_confidence_penalty(logits, labels)
        assert model.confidence_penalty_factor_ == 0.0

    def test_fitted_factor_respects_the_knob_bound(self):
        labels = np.asarray([i % len(CLASSES) for i in range(200)], dtype=int)
        logits = _logits_for([CLASSES[i] for i in labels], correct_share=0.5, seed=3)
        model = DualHeadSupportModel(intent_confidence_penalty=0.05)
        model.intent_temperature.temperature_ = 1.0
        model._fit_confidence_penalty(logits, labels)
        assert 0.0 <= model.confidence_penalty_factor_ <= 0.05
