"""Tests for recency-weighted escalation training.

The temporal protocol showed the escalation head's ranking decays on recent
traffic: precision at a 0.85 recall floor falls from 1.48x the base rate on
same-period traffic to 1.23x on the newest 15% of conversations, and sweeping
every threshold over that run's own risk scores confirmed no operating point
repairs it. Weighting older rows down is the cheap remedy, so the mechanism has
to be exactly right: silent weight bugs would show up as a quiet change in
regularisation strength rather than as a visible failure.
"""

import numpy as np
import pandas as pd
import pytest

from src.model_factory import _recency_weights


def _frame(days_ago):
    """A frame whose rows are ``days_ago`` days before the newest one."""
    newest = pd.Timestamp("2017-12-03T12:00:00Z")
    stamps = [newest - pd.Timedelta(days=float(offset)) for offset in days_ago]
    return pd.DataFrame({"customer_created_at": [s.strftime("%a %b %d %H:%M:%S %z %Y") for s in stamps]})


class TestDisabledMeansUniform:
    def test_none_half_life_returns_none(self):
        assert _recency_weights(_frame([0, 10, 30]), half_life_days=None) is None

    def test_zero_half_life_returns_none(self):
        assert _recency_weights(_frame([0, 10, 30]), half_life_days=0) is None

    def test_negative_half_life_returns_none(self):
        assert _recency_weights(_frame([0, 10, 30]), half_life_days=-5) is None

    def test_missing_timestamp_column_returns_none(self):
        frame = pd.DataFrame({"other": [1, 2, 3]})
        assert _recency_weights(frame, half_life_days=7) is None

    def test_empty_frame_returns_none(self):
        assert _recency_weights(_frame([]), half_life_days=7) is None


class TestWeights:
    def test_mean_is_one(self):
        """Normalising to mean 1 keeps the effective sample size readable."""
        weights = _recency_weights(_frame([0, 5, 10, 20, 40]), half_life_days=10)
        assert weights is not None
        assert float(weights.mean()) == pytest.approx(1.0)

    def test_newer_rows_weigh_more(self):
        # _frame lists the newest row first, so weights must descend down it.
        weights = _recency_weights(_frame([0, 5, 10, 20, 40]), half_life_days=10)
        assert weights is not None
        assert list(weights) == sorted(weights, reverse=True)
        assert weights[0] > weights[-1]

    def test_half_life_halves_the_weight(self):
        """A row one half-life older must carry half the weight."""
        weights = _recency_weights(_frame([0, 14]), half_life_days=14)
        assert weights is not None
        assert weights[1] / weights[0] == pytest.approx(0.5, abs=1e-9)

    def test_longer_half_life_flattens_the_weights(self):
        steep = _recency_weights(_frame([0, 30]), half_life_days=7)
        flat = _recency_weights(_frame([0, 30]), half_life_days=90)
        assert steep is not None and flat is not None
        assert (steep[1] / steep[0]) < (flat[1] / flat[0])

    def test_all_weights_are_finite_and_positive(self):
        weights = _recency_weights(_frame([0, 1, 500]), half_life_days=3)
        assert weights is not None
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)

    def test_unparseable_timestamps_get_the_median_age(self):
        """A bad timestamp must not be silently treated as brand new."""
        frame = _frame([0, 10, 20])
        frame.loc[len(frame)] = ["not a timestamp"]
        weights = _recency_weights(frame, half_life_days=10)
        assert weights is not None
        assert len(weights) == 4
        assert np.all(np.isfinite(weights))
        # Median age is 10 days, so the bad row lands between the extremes.
        assert weights.min() <= weights[3] <= weights.max()

    def test_rows_sharing_a_timestamp_get_equal_weight(self):
        weights = _recency_weights(_frame([5, 5]), half_life_days=10)
        assert weights is not None
        assert weights[0] == pytest.approx(weights[1])


class TestAlignedSampleWeight:
    """Weights must be validated loudly, not silently ignored.

    A silently-dropped weight vector would look like a successful run that had
    quietly trained with different regularisation than intended — the kind of
    difference that shows up as an unexplained score change much later.
    """

    def test_none_passes_through_as_none(self):
        from src.dual_head import _aligned_sample_weight

        assert _aligned_sample_weight(None, 5) is None

    def test_valid_weights_are_returned_as_float_array(self):
        from src.dual_head import _aligned_sample_weight

        result = _aligned_sample_weight([1, 2, 3], 3)
        assert result is not None
        assert result.dtype == float
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_misaligned_weights_are_rejected(self):
        from src.dual_head import _aligned_sample_weight

        with pytest.raises(ValueError, match="must align"):
            _aligned_sample_weight([1.0, 2.0, 3.0], 5)

    def test_negative_weights_are_rejected(self):
        from src.dual_head import _aligned_sample_weight

        with pytest.raises(ValueError, match="finite and non-negative"):
            _aligned_sample_weight([1.0, -1.0], 2)

    def test_non_finite_weights_are_rejected(self):
        from src.dual_head import _aligned_sample_weight

        with pytest.raises(ValueError, match="finite and non-negative"):
            _aligned_sample_weight([1.0, float("nan")], 2)

    def test_zero_sum_weights_are_rejected(self):
        from src.dual_head import _aligned_sample_weight

        with pytest.raises(ValueError, match="sum to zero"):
            _aligned_sample_weight([0.0, 0.0], 2)

    def test_two_dimensional_weights_are_rejected(self):
        from src.dual_head import _aligned_sample_weight

        with pytest.raises(ValueError, match="must align"):
            _aligned_sample_weight([[1.0, 2.0], [3.0, 4.0]], 2)

_FIT_TEXTS = [
    "my package was marked delivered but it never arrived at my address",
    "someone changed my email and password and i am locked out of my account",
    "the laptop arrived with a cracked screen and it will not power on",
    "i was charged twice for the same order and need a refund for the duplicate",
    "where can i see the latest tracking update for my shipment please",
    "your delivery team was fantastic please pass on my thanks to them",
    "the box was crushed and the item inside is broken beyond repair",
    "i cannot sign in because my verification code never arrives by text",
    "please tell me which carrier has my parcel and when it will arrive",
    "the replacement you sent is also the wrong colour entirely",
    "my account shows an order i never placed and i want it cancelled",
    "delivery was quicker than promised so thank you for the good service",
    "the tablet screen flickers whenever i open the camera application",
    "my refund was approved three weeks ago but never reached my bank",
    "tracking has said out for delivery for four days without any update",
]
# Three intent classes, not two: OneVsRestClassifier returns 1-D logits for a
# binary problem, and the temperature scaler indexes columns. Production always
# has six intents, so this mirrors the real shape rather than working around it.
_FIT_INTENT_LABELS = (["FEEDBACK_AND_GENERAL"] * 5
                      + ["ORDER_TRACKING_AND_DELAY"] * 5
                      + ["PRODUCT_AND_TECHNICAL"] * 5)
_FIT_ESCALATION_LABELS = [index % 3 == 0 for index in range(len(_FIT_TEXTS))]


def _fit_with_captured_weights(monkeypatch, **extra):
    """Run a real ``fit`` and report the ``sample_weight`` the head received.

    Regularisation tuning is off because ``_select_escalation_regularization``
    replaces ``self.escalation_head`` with a freshly constructed estimator, which
    would discard the instrumented instance below.
    """
    from src.dual_head import DualHeadSupportModel

    model = DualHeadSupportModel(tune_escalation_regularization=False)
    model.escalation_vectorizer.fit(_FIT_TEXTS)
    monkeypatch.setattr(
        model.encoder, "encode", lambda batch, batch_size=64: np.zeros((len(batch), 4))
    )
    captured = {}
    real_fit = model.escalation_head.fit

    def capturing_fit(features, labels, sample_weight=None):
        captured["weights"] = sample_weight
        return real_fit(features, labels, sample_weight=sample_weight)

    monkeypatch.setattr(model.escalation_head, "fit", capturing_fit)
    model.fit(
        intent_texts=_FIT_TEXTS, intent_labels=_FIT_INTENT_LABELS,
        escalation_texts=_FIT_TEXTS, escalation_labels=_FIT_ESCALATION_LABELS,
        cal_intent_texts=_FIT_TEXTS, cal_intent_labels=_FIT_INTENT_LABELS,
        cal_escalation_texts=_FIT_TEXTS, cal_escalation_labels=_FIT_ESCALATION_LABELS,
        **extra,
    )
    return captured, model


class TestWeightsReachTheEstimator:
    """A silent failure here would mean recency does nothing while appearing on."""

    def test_supplied_weights_are_passed_through(self, monkeypatch):
        weights = np.linspace(0.5, 2.0, len(_FIT_TEXTS))
        captured, _ = _fit_with_captured_weights(
            monkeypatch, escalation_sample_weight=weights
        )
        assert captured["weights"] is not None
        assert np.allclose(captured["weights"], weights)

    def test_omitted_weights_mean_unweighted(self, monkeypatch):
        captured, _ = _fit_with_captured_weights(monkeypatch)
        assert captured["weights"] is None

    def test_head_is_actually_fitted(self, monkeypatch):
        _, model = _fit_with_captured_weights(
            monkeypatch, escalation_sample_weight=np.ones(len(_FIT_TEXTS))
        )
        assert hasattr(model.escalation_head, "coef_")
