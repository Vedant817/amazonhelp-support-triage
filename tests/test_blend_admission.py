"""Tests for the auxiliary-head blend admission protocol.

Every auxiliary intent head (semantic, prototype, three-way ensemble) has to
earn the right to influence serving. The protocol is:

    select blend weights on a `select` half of the calibration slice by
    minimising log loss, then re-score them on a `verify` half and admit them
    only if they beat the hybrid head beyond sampling noise in a paired
    bootstrap.

The regression this file protects against is specific and was real: blend
weights were previously chosen as the argmax of up to 112 candidates on a single
~1700-row split, ranked by macro-F1 (standard error ~0.01), while admission
required a gain of only 0.001. Acceptance was effectively automatic, and the
served model ended up giving a nearest-centroid head over MiniLM embeddings 0.6
of the final probability, ahead of the far stronger lexical head.
"""

import numpy as np
import pytest

from src.dual_head import DualHeadSupportModel


CLASSES = ["A", "B", "C"]


def _model() -> DualHeadSupportModel:
    model = DualHeadSupportModel()
    model.classes_ = list(CLASSES)
    return model


def _one_hot(labels, smoothing: float = 0.0) -> np.ndarray:
    index = {label: position for position, label in enumerate(CLASSES)}
    probabilities = np.full((len(labels), len(CLASSES)), smoothing)
    for row, label in enumerate(labels):
        probabilities[row, index[label]] += 1.0
    return probabilities


class TestCalibrationSplit:
    def test_stratified_so_every_class_is_on_both_sides(self):
        labels = np.asarray(["A"] * 40 + ["B"] * 20 + ["C"] * 10, dtype=object)
        select, verify = _model()._calibration_split(labels)
        for index in (select, verify):
            assert set(labels[index]) == {"A", "B", "C"}

    def test_covers_every_row_exactly_once(self):
        labels = np.asarray(list("ABCABCABC") * 7, dtype=object)
        select, verify = _model()._calibration_split(labels)
        combined = np.concatenate([select, verify])
        assert sorted(combined.tolist()) == list(range(len(labels)))
        assert len(set(select.tolist()) & set(verify.tolist())) == 0

    def test_verify_fraction_is_respected(self):
        labels = np.asarray(["A"] * 100, dtype=object)
        _, verify = _model()._calibration_split(labels, verify_fraction=0.4)
        assert len(verify) == 40

    def test_deterministic_for_a_fixed_seed(self):
        labels = np.asarray(list("ABC") * 40, dtype=object)
        select_a, verify_a = _model()._calibration_split(labels, seed=3)
        select_b, verify_b = _model()._calibration_split(labels, seed=3)
        assert np.array_equal(select_a, select_b)
        assert np.array_equal(verify_a, verify_b)

    def test_singleton_class_stays_on_one_side(self):
        """A class with one row cannot be split; it must not lose the row."""
        labels = np.asarray(["A"] * 20 + ["B"], dtype=object)
        select, verify = _model()._calibration_split(labels)
        assert len(np.concatenate([select, verify])) == len(labels)


class TestVerifyBlend:
    def _probabilities(self, labels, correct_fraction: float, seed: int) -> np.ndarray:
        """Probabilities that are right on ``correct_fraction`` of the rows."""
        rng = np.random.default_rng(seed)
        probabilities = np.full((len(labels), len(CLASSES)), 0.1)
        index = {label: position for position, label in enumerate(CLASSES)}
        for row, label in enumerate(labels):
            if rng.random() < correct_fraction:
                probabilities[row, index[label]] = 0.9
            else:
                wrong = (index[label] + 1) % len(CLASSES)
                probabilities[row, wrong] = 0.9
        return probabilities

    def test_identical_candidate_is_rejected(self):
        """Tying the baseline is not evidence, so it must not be admitted."""
        labels = np.asarray(list("ABC") * 60, dtype=object)
        baseline = self._probabilities(labels, 0.6, seed=1)
        result = _model()._verify_blend(baseline.copy(), baseline, labels)
        assert result["accepted"] is False
        assert result["mean_macro_f1_gain"] == pytest.approx(0.0, abs=1e-12)

    def test_worse_candidate_is_rejected(self):
        labels = np.asarray(list("ABC") * 60, dtype=object)
        baseline = self._probabilities(labels, 0.95, seed=2)
        worse = self._probabilities(labels, 0.30, seed=3)
        assert _model()._verify_blend(worse, baseline, labels)["accepted"] is False

    def test_clearly_better_candidate_is_accepted(self):
        labels = np.asarray(list("ABC") * 60, dtype=object)
        baseline = self._probabilities(labels, 0.30, seed=4)
        better = self._probabilities(labels, 0.97, seed=5)
        result = _model()._verify_blend(better, baseline, labels)
        assert result["accepted"] is True
        assert result["mean_macro_f1_gain"] > 0
        assert result["bootstrap_win_rate"] >= 0.90

    def test_reports_both_accuracies(self):
        labels = np.asarray(list("ABC") * 40, dtype=object)
        result = _model()._verify_blend(
            self._probabilities(labels, 0.9, seed=6),
            self._probabilities(labels, 0.5, seed=7),
            labels,
        )
        assert 0.0 <= result["baseline_accuracy"] <= 1.0
        assert 0.0 <= result["candidate_accuracy"] <= 1.0


class TestMinimumNllWeights:
    def test_prefers_the_informative_component(self):
        labels = np.asarray(list("ABC") * 60, dtype=object)
        truth_index = np.asarray([CLASSES.index(label) for label in labels])
        model = _model()
        uniform = np.full((len(labels), len(CLASSES)), 1.0 / len(CLASSES))
        informative = _one_hot(labels, smoothing=0.01)

        weights = model._minimum_nll_weights([uniform, informative], truth_index)
        assert weights is not None
        assert weights.sum() == pytest.approx(1.0)
        assert weights[1] > 0.9, "the informative component must take the weight"

    def test_weight_order_follows_component_order(self):
        labels = np.asarray(list("ABC") * 60, dtype=object)
        truth_index = np.asarray([CLASSES.index(label) for label in labels])
        model = _model()
        uniform = np.full((len(labels), len(CLASSES)), 1.0 / len(CLASSES))
        informative = _one_hot(labels, smoothing=0.01)

        weights = model._minimum_nll_weights([informative, uniform], truth_index)
        assert weights[0] > 0.9

    def test_three_components_favour_the_better_two(self):
        labels = np.asarray(list("ABC") * 60, dtype=object)
        truth_index = np.asarray([CLASSES.index(label) for label in labels])
        model = _model()
        uniform = np.full((len(labels), len(CLASSES)), 1.0 / len(CLASSES))
        informative = _one_hot(labels, smoothing=0.01)
        weights = model._minimum_nll_weights([uniform, uniform.copy(), informative], truth_index)
        assert weights.sum() == pytest.approx(1.0)
        assert weights[2] > weights[0]
        assert weights[2] > weights[1]

    def test_returns_none_for_a_single_component(self):
        labels = np.asarray(list("ABC"), dtype=object)
        assert _model()._minimum_nll_weights([_one_hot(labels)], np.asarray([0, 1, 2])) is None

    def test_returns_none_for_no_rows(self):
        empty = np.zeros((0, 3))
        assert _model()._minimum_nll_weights([empty, empty], np.asarray([], dtype=int)) is None


class TestAdmissionIsOffByDefault:
    def test_no_blend_is_admitted_before_fitting(self):
        model = DualHeadSupportModel()
        assert model.semantic_blend_accepted_ is False
        assert model.prototype_blend_accepted_ is False
        assert model.ensemble_blend_accepted_ is False

    def test_rejected_blends_keep_weight_zero(self):
        model = DualHeadSupportModel()
        assert model.semantic_blend_weight_ == 0.0
        assert model.prototype_blend_weight_ == 0.0
        assert model.ensemble_semantic_weight_ == 0.0
        assert model.ensemble_prototype_weight_ == 0.0
        assert model.ensemble_hybrid_weight_ == 1.0

    def test_ensemble_fit_without_prototypes_is_a_no_op(self):
        """The 3-way blend must not run before the prototypes exist."""
        model = _model()
        model.intent_prototypes_ = None
        embeddings = np.zeros((10, 4))
        labels = np.asarray(list("ABCABCABCA"), dtype=object)
        model._fit_ensemble_blend(embeddings, labels, np.zeros((10, 3)))
        assert model.ensemble_blend_accepted_ is False
        assert model.ensemble_hybrid_weight_ == 1.0
