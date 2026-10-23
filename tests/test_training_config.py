"""
Tests for the intent-training configuration knobs.

These pin the behaviour of the two switches that the score-matrix investigation
introduced, so a future refactor cannot silently reintroduce the accuracy loss
they fixed (a capped, force-balanced training set).
"""

import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.model_factory import _balance_intent_classes, _cap_per_class
from src.dual_head import DualHeadSupportModel


@pytest.fixture()
def skewed_frame() -> pd.DataFrame:
    """Mirror the real corpus shape: one dominant class, one tiny class."""
    return pd.DataFrame({
        "intent_label": (
            ["ORDER_TRACKING_AND_DELAY"] * 700
            + ["PRODUCT_AND_TECHNICAL"] * 120
            + ["ACCOUNT_AND_SECURITY"] * 30
        )
    })


class TestBalanceRatio:
    def test_zero_keeps_natural_distribution(self, skewed_frame):
        out = _balance_intent_classes(skewed_frame, "intent_label", seed=42, max_ratio=0.0)
        counts = out["intent_label"].value_counts()
        assert counts["ORDER_TRACKING_AND_DELAY"] == 700
        assert counts["ACCOUNT_AND_SECURITY"] == 30
        assert len(out) == len(skewed_frame)

    def test_positive_ratio_flattens_classes(self, skewed_frame):
        out = _balance_intent_classes(skewed_frame, "intent_label", seed=42, max_ratio=3.0)
        counts = out["intent_label"].value_counts()
        assert counts.nunique() == 1, "every class should end at the same size"
        assert counts.iloc[0] == 90  # smallest class (30) * 3

    def test_env_override_wins(self, skewed_frame):
        with patch.dict(os.environ, {"SUPPORT_INTENT_BALANCE_RATIO": "0"}):
            out = _balance_intent_classes(skewed_frame, "intent_label", seed=42, max_ratio=10.0)
        assert len(out) == len(skewed_frame)


class TestPerClassCap:
    def test_cap_truncates_dominant_class(self, skewed_frame):
        out = _cap_per_class(skewed_frame, "intent_label", cap=100, seed=42)
        counts = out["intent_label"].value_counts()
        assert counts["ORDER_TRACKING_AND_DELAY"] == 100
        assert counts["ACCOUNT_AND_SECURITY"] == 30  # below the cap, untouched

    def test_large_cap_is_a_noop(self, skewed_frame):
        out = _cap_per_class(skewed_frame, "intent_label", cap=20000, seed=42)
        assert len(out) == len(skewed_frame)

    def test_env_override_lifts_cap(self, skewed_frame):
        with patch.dict(os.environ, {"SUPPORT_INTENT_CAP_PER_CLASS": "100000"}):
            out = _cap_per_class(skewed_frame, "intent_label", cap=10, seed=42)
        assert len(out) == len(skewed_frame)


class TestIntentClassWeight:
    def test_default_is_unweighted(self):
        """Natural training + prior correction makes reweighting redundant."""
        estimator = DualHeadSupportModel._intent_estimator(1.0)
        assert estimator.estimator.class_weight is None

    @pytest.mark.parametrize("value", ["balanced", "BALANCED", "1", "true", "yes"])
    def test_env_enables_reweighting(self, value):
        with patch.dict(os.environ, {"SUPPORT_INTENT_CLASS_WEIGHT": value}):
            estimator = DualHeadSupportModel._intent_estimator(1.0)
        assert estimator.estimator.class_weight == "balanced"

    def test_unknown_value_falls_back_to_unweighted(self):
        with patch.dict(os.environ, {"SUPPORT_INTENT_CLASS_WEIGHT": "whatever"}):
            estimator = DualHeadSupportModel._intent_estimator(1.0)
        assert estimator.estimator.class_weight is None

    def test_c_value_is_respected(self):
        estimator = DualHeadSupportModel._intent_estimator(32.0)
        assert estimator.estimator.C == 32.0
