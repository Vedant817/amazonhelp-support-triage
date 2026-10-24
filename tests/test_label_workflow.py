"""Tests for the human labelling workflow.

The intent ceiling is supervision: targets come from historical agent replies by
rule, so they encode what the agent did while the model only sees what the
customer wrote. Human labels are the only fix, which makes this path load-bearing
— and the failure modes are quiet. A wrong kappa accepts labels that only look
like ground truth; a wrong majority silently drops or invents rows.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_workflow():
    spec = importlib.util.spec_from_file_location(
        "scripts_label_workflow", REPO_ROOT / "scripts_label_workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def workflow():
    return _load_workflow()


class TestCohenKappa:
    def test_perfect_agreement_is_one(self, workflow):
        kappa, observed = workflow.cohen_kappa(["A", "B", "C", "A"], ["A", "B", "C", "A"])
        assert kappa == pytest.approx(1.0)
        assert observed == pytest.approx(1.0)

    def test_chance_level_agreement_is_near_zero(self, workflow):
        """Two raters on a balanced binary task who half-agree carry no signal."""
        first = ["A"] * 10 + ["B"] * 10
        second = ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5
        kappa, observed = workflow.cohen_kappa(first, second)
        assert observed == pytest.approx(0.5)
        assert kappa == pytest.approx(0.0, abs=1e-9)

    def test_systematic_disagreement_is_negative(self, workflow):
        first = ["A"] * 10 + ["B"] * 10
        second = ["B"] * 10 + ["A"] * 10
        kappa, observed = workflow.cohen_kappa(first, second)
        assert observed == pytest.approx(0.0)
        assert kappa < 0.0

    def test_degenerate_marginals_with_agreement_report_one(self, workflow):
        """One label everywhere: agreement is guaranteed, so kappa is undefined."""
        kappa, observed = workflow.cohen_kappa(["A"] * 5, ["A"] * 5)
        assert observed == pytest.approx(1.0)
        assert kappa == pytest.approx(1.0)

    def test_length_mismatch_is_rejected(self, workflow):
        with pytest.raises(ValueError, match="same rows"):
            workflow.cohen_kappa(["A", "B"], ["A"])

    def test_empty_input_is_rejected(self, workflow):
        with pytest.raises(ValueError, match="At least one"):
            workflow.cohen_kappa([], [])

    def test_kappa_is_below_raw_agreement_when_one_label_dominates(self, workflow):
        """This is the whole reason for using kappa over raw agreement."""
        first = (["ORDER_TRACKING_AND_DELAY"] * 90) + (["REFUND_AND_RETURN"] * 10)
        second = (["ORDER_TRACKING_AND_DELAY"] * 90) + (["REFUND_AND_RETURN"] * 10)
        kappa, observed = workflow.cohen_kappa(first, second)
        assert observed == pytest.approx(1.0)
        assert kappa == pytest.approx(1.0)
        # Now disagree only on the rare class: raw agreement stays high (0.9),
        # so raw agreement alone would hide that the rare class is unusable.
        third = (["ORDER_TRACKING_AND_DELAY"] * 90) + (["FEEDBACK_AND_GENERAL"] * 10)
        kappa_mixed, observed_mixed = workflow.cohen_kappa(first, third)
        assert observed_mixed == pytest.approx(0.9)
        assert kappa_mixed < 0.5


class TestMajority:
    def test_clear_majority(self, workflow):
        assert workflow._majority(["A", "A", "B"]) == "A"

    def test_tie_returns_none(self, workflow):
        assert workflow._majority(["A", "B"]) is None

    def test_three_way_tie_returns_none(self, workflow):
        assert workflow._majority(["A", "B", "C"]) is None

    def test_unanimous(self, workflow):
        assert workflow._majority(["B", "B", "B"]) == "B"


def _write_sheet(path: Path, rows) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False, quoting=1)
    return path


class TestAdjudicate:
    """The gate is the point: low-agreement labels must not reach training."""

    def _rows(self, intents, escalates, hashes=None):
        hashes = hashes or [f"hash{index}" for index in range(len(intents))]
        return [
            {"inbound_id": f"id{index}", "source_sha256": hashes[index],
             "reviewed_intent": intents[index], "reviewed_escalate": escalates[index]}
            for index in range(len(intents))
        ]

    def test_agreeing_annotators_produce_a_file(self, workflow, tmp_path):
        intents = ["REFUND_AND_RETURN", "ORDER_TRACKING_AND_DELAY", "ACCOUNT_AND_SECURITY"] * 6
        escalates = ["true", "false", "true"] * 6
        alice = _write_sheet(tmp_path / "alice.csv", self._rows(intents, escalates))
        bob = _write_sheet(tmp_path / "bob.csv", self._rows(intents, escalates))
        output = tmp_path / "merged.csv"

        report = workflow.adjudicate([str(alice), str(bob)], output=str(output))
        assert Path(report["written"]) == output
        written = pd.read_csv(output, dtype={"inbound_id": str})
        assert len(written) == len(intents)
        assert set(written["reviewed_intent"]) == set(intents)
        assert report["dimensions"]["reviewed_intent"]["cohen_kappa"] == pytest.approx(1.0)

    def test_disagreeing_annotators_write_nothing(self, workflow, tmp_path):
        """DimDisagreement on every dimension must be refused, not averaged away."""
        intents_a = ["REFUND_AND_RETURN", "ACCOUNT_AND_SECURITY"] * 10
        intents_b = ["ORDER_TRACKING_AND_DELAY", "PRODUCT_AND_TECHNICAL"] * 10
        alice = _write_sheet(
            tmp_path / "alice.csv",
            self._rows(intents_a, ["true", "false"] * 10),
        )
        bob = _write_sheet(
            tmp_path / "bob.csv",
            self._rows(intents_b, ["false", "true"] * 10),
        )
        output = tmp_path / "merged.csv"

        report = workflow.adjudicate([str(alice), str(bob)], output=str(output))
        assert report["written"] is None
        assert not output.exists()
        assert "note" in report
        assert report["dimensions"]["reviewed_intent"]["status"] == "rejected"
        assert report["dimensions"]["reviewed_escalate"]["status"] == "rejected"

    def test_low_agreement_on_one_dimension_does_not_block_the_other(self, workflow, tmp_path):
        """Dimensions are gated independently."""
        intents = ["REFUND_AND_RETURN", "ORDER_TRACKING_AND_DELAY"] * 10
        escalates_a = ["true", "false"] * 10
        escalates_b = ["false", "true"] * 10
        alice = _write_sheet(tmp_path / "alice.csv", self._rows(intents, escalates_a))
        bob = _write_sheet(tmp_path / "bob.csv", self._rows(intents, escalates_b))
        output = tmp_path / "merged.csv"

        report = workflow.adjudicate([str(alice), str(bob)], output=str(output))
        assert report["dimensions"]["reviewed_intent"]["status"] == "accepted"
        assert report["dimensions"]["reviewed_escalate"]["status"] == "rejected"
        written = pd.read_csv(output, dtype={"inbound_id": str})
        assert set(written["reviewed_intent"]) == set(intents)
        # The rejected dimension must be blank rather than filled with a guess.
        assert written["reviewed_escalate"].isna().all() or (
            written["reviewed_escalate"].astype(str).str.strip() == ""
        ).all()

    def test_abstentions_are_not_treated_as_labels(self, workflow, tmp_path):
        intents = ["REFUND_AND_RETURN", "ORDER_TRACKING_AND_DELAY"] * 10
        alice_rows = self._rows(intents, ["true"] * 20)
        bob_rows = self._rows(intents, ["true"] * 20)
        for row in bob_rows[:5]:
            row["reviewed_intent"] = ""       # abstain on the first five
            row["reviewed_escalate"] = ""
        alice = _write_sheet(tmp_path / "alice.csv", alice_rows)
        bob = _write_sheet(tmp_path / "bob.csv", bob_rows)

        report = workflow.adjudicate([str(alice), str(bob)], output=str(tmp_path / "m.csv"))
        # Agreement is measured only on rows both actually labelled.
        assert report["dimensions"]["reviewed_intent"]["paired_rows"] == 15

    def test_three_annotators_resolve_by_majority(self, workflow, tmp_path):
        intents = ["REFUND_AND_RETURN"] * 20
        alice = _write_sheet(tmp_path / "alice.csv", self._rows(intents, ["true"] * 20))
        bob = _write_sheet(tmp_path / "bob.csv", self._rows(intents, ["true"] * 20))
        # Carol dissents on one row; the majority still decides it.
        carol_intents = list(intents)
        carol_intents[0] = "ORDER_TRACKING_AND_DELAY"
        carol = _write_sheet(tmp_path / "carol.csv", self._rows(carol_intents, ["true"] * 20))
        output = tmp_path / "merged.csv"

        report = workflow.adjudicate([str(alice), str(bob), str(carol)], output=str(output))
        written = pd.read_csv(output, dtype={"inbound_id": str})
        assert report["written"] is not None
        assert written.loc[written["inbound_id"] == "id0", "reviewed_intent"].iloc[0] == "REFUND_AND_RETURN"

    def test_boolean_escalation_column_is_normalised(self, workflow, tmp_path):
        """pandas reads a true/false column back as bool dtype, not as strings.

        Validating the raw vote against {"true","false"} therefore rejected every
        escalation row while the intent dimension (whose values are never coerced)
        kept working — a silent half-failure of the workflow.
        """
        intents = ["REFUND_AND_RETURN", "ORDER_TRACKING_AND_DELAY"] * 10
        rows = self._rows(intents, [True, False] * 10)
        alice = _write_sheet(tmp_path / "alice.csv", rows)
        bob = _write_sheet(tmp_path / "bob.csv", rows)
        output = tmp_path / "merged.csv"

        report = workflow.adjudicate([str(alice), str(bob)], output=str(output))
        assert report["dimensions"]["reviewed_escalate"]["status"] == "accepted"
        assert report["dimensions"]["reviewed_escalate"]["labelled_rows"] == 20
        written = pd.read_csv(output, dtype={"inbound_id": str})
        values = {str(v).strip().lower() for v in written["reviewed_escalate"].dropna()}
        assert values == {"true", "false"}

    def test_string_escalation_column_is_normalised(self, workflow, tmp_path):
        """The same column written as text must produce the same result."""
        intents = ["REFUND_AND_RETURN", "ORDER_TRACKING_AND_DELAY"] * 10
        rows = self._rows(intents, ["true", "false"] * 10)
        alice = _write_sheet(tmp_path / "alice.csv", rows)
        bob = _write_sheet(tmp_path / "bob.csv", rows)
        output = tmp_path / "merged.csv"

        report = workflow.adjudicate([str(alice), str(bob)], output=str(output))
        assert report["dimensions"]["reviewed_escalate"]["labelled_rows"] == 20

    def test_mixed_case_values_are_normalised(self, workflow, tmp_path):
        intents = ["REFUND_AND_RETURN"] * 10
        alice = _write_sheet(tmp_path / "alice.csv", self._rows(intents, ["TRUE"] * 10))
        bob = _write_sheet(tmp_path / "bob.csv", self._rows(intents, ["True"] * 10))
        report = workflow.adjudicate([str(alice), str(bob)], output=str(tmp_path / "m.csv"))
        assert report["dimensions"]["reviewed_escalate"]["labelled_rows"] == 10

    def test_invalid_intent_is_not_written(self, workflow, tmp_path):
        """A label outside the taxonomy must never reach training.

        Escalation still clears its own gate and is written, because the two
        dimensions are judged independently — refusing one must not discard
        evidence that is perfectly usable.
        """
        intents = ["NOT_A_REAL_INTENT"] * 10
        alice = _write_sheet(tmp_path / "alice.csv", self._rows(intents, ["true"] * 10))
        bob = _write_sheet(tmp_path / "bob.csv", self._rows(intents, ["true"] * 10))
        output = tmp_path / "merged.csv"

        report = workflow.adjudicate([str(alice), str(bob)], output=str(output))
        written = pd.read_csv(output, dtype={"inbound_id": str})
        # Empty cells round-trip through CSV as NaN, so normalise before checking.
        intents_written = written["reviewed_intent"].fillna("").astype(str).str.strip()
        assert set(intents_written) == {""}, "invalid intent must be omitted"
        assert report["dimensions"]["reviewed_intent"]["labelled_rows"] == 0
        assert report["dimensions"]["reviewed_escalate"]["labelled_rows"] == 10

    def test_duplicate_ids_are_rejected(self, workflow, tmp_path):
        rows = self._rows(["REFUND_AND_RETURN"] * 4, ["true"] * 4)
        rows[1]["inbound_id"] = rows[0]["inbound_id"]
        alice = _write_sheet(tmp_path / "alice.csv", rows)
        bob = _write_sheet(tmp_path / "bob.csv", rows)
        with pytest.raises(ValueError, match="duplicate inbound_id"):
            workflow.adjudicate([str(alice), str(bob)], output=str(tmp_path / "m.csv"))

    def test_missing_columns_are_rejected(self, workflow, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"inbound_id": ["a"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            workflow.adjudicate([str(path), str(path)], output=str(tmp_path / "m.csv"))

    def test_single_sheet_cannot_be_adjudicated(self, workflow, tmp_path):
        alice = _write_sheet(tmp_path / "alice.csv", self._rows(["REFUND_AND_RETURN"] * 6, ["true"] * 6))
        with pytest.raises(ValueError, match="at least two"):
            workflow.adjudicate([str(alice)])

    def test_source_hash_is_carried_through(self, workflow, tmp_path):
        """Labels stay bound to the exact row the annotator saw."""
        hashes = [f"abc{index}" for index in range(6)]
        intents = ["REFUND_AND_RETURN"] * 6
        alice = _write_sheet(tmp_path / "alice.csv", self._rows(intents, ["true"] * 6, hashes))
        bob = _write_sheet(tmp_path / "bob.csv", self._rows(intents, ["true"] * 6, hashes))
        output = tmp_path / "merged.csv"
        workflow.adjudicate([str(alice), str(bob)], output=str(output))
        written = pd.read_csv(output, dtype={"inbound_id": str})
        assert written["source_sha256"].tolist() == hashes


class TestPipelinePicksUpLabels:
    """Producing labels must actually change training, or the path is decorative."""

    def test_missing_file_resolves_to_none(self, monkeypatch):
        from src import model_factory

        monkeypatch.delenv("SUPPORT_ADJUDICATED_LABELS", raising=False)
        monkeypatch.setattr(model_factory, "DEFAULT_ADJUDICATED_LABELS", "does/not/exist.csv")
        assert model_factory._resolve_reviewed_csv(None) is None

    def test_default_path_is_used_when_present(self, monkeypatch, tmp_path):
        from src import model_factory

        labels = tmp_path / "labels.csv"
        labels.write_text("inbound_id\n", encoding="utf-8")
        monkeypatch.delenv("SUPPORT_ADJUDICATED_LABELS", raising=False)
        monkeypatch.setattr(model_factory, "DEFAULT_ADJUDICATED_LABELS", str(labels))
        assert model_factory._resolve_reviewed_csv(None) == str(labels)

    def test_explicit_path_wins_over_the_convention(self, monkeypatch, tmp_path):
        from src import model_factory

        labels = tmp_path / "explicit.csv"
        labels.write_text("inbound_id\n", encoding="utf-8")
        monkeypatch.delenv("SUPPORT_ADJUDICATED_LABELS", raising=False)
        assert model_factory._resolve_reviewed_csv(str(labels)) == str(labels)

    def test_environment_can_opt_out(self, monkeypatch, tmp_path):
        from src import model_factory

        labels = tmp_path / "labels.csv"
        labels.write_text("inbound_id\n", encoding="utf-8")
        monkeypatch.setattr(model_factory, "DEFAULT_ADJUDICATED_LABELS", str(labels))
        monkeypatch.setenv("SUPPORT_ADJUDICATED_LABELS", "")
        assert model_factory._resolve_reviewed_csv(None) is None

    def test_environment_can_point_elsewhere(self, monkeypatch, tmp_path):
        from src import model_factory

        labels = tmp_path / "custom.csv"
        labels.write_text("inbound_id\n", encoding="utf-8")
        monkeypatch.setenv("SUPPORT_ADJUDICATED_LABELS", str(labels))
        assert model_factory._resolve_reviewed_csv(None) == str(labels)
