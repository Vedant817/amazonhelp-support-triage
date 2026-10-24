"""Compare two score-matrix reports side by side.

Configuration changes are only worth keeping if they survive the same evaluation
protocol, so this reads the JSON reports written by ``scripts_score.py`` and
prints the metrics that decide the question, with the delta and a flag for
whether the difference is inside the metric's own confidence interval.

Usage:
    python scripts_compare_reports.py models/_score_A_prob.json models/_score_B_replyonly.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _get(report: Dict[str, Any], path: str) -> Optional[float]:
    """Look up a dotted path, accepting numeric segments as list indices."""
    node: Any = report
    for key in path.split("."):
        if isinstance(node, dict):
            if key not in node:
                return None
            node = node[key]
        elif isinstance(node, list) and key.isdigit():
            index = int(key)
            if index >= len(node):
                return None
            node = node[index]
        else:
            return None
    return node if isinstance(node, (int, float)) and not isinstance(node, bool) else None


METRICS: List[Tuple[str, str]] = [
    ("intent accuracy", "intent.accuracy"),
    ("intent macro-F1", "intent.macro_f1"),
    ("intent macro precision", "intent.macro_precision"),
    ("intent macro recall", "intent.macro_recall"),
    ("deployment accuracy", "deployment_accuracy"),
    ("ECE", "calibration.ece"),
    ("MCE", "calibration.mce"),
    ("escalate recall", "escalation.escalate_recall"),
    ("escalate precision", "escalation.escalate_precision"),
    ("auto-handle precision", "escalation.auto_handle_precision"),
    ("auto-handle recall", "escalation.auto_handle_recall"),
    ("guardrail pass", "reply.guardrail_pass_rate"),
    ("PII violations", "reply.pii_violation_count"),
]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    first_path, second_path = Path(sys.argv[1]), Path(sys.argv[2])
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))

    label_a = first_path.stem
    label_b = second_path.stem
    # `protocol` is the string "held-out"/"in-sample"; `holdout` carries the mode.
    mode_a = (first.get("holdout") or {}).get("mode", "?")
    mode_b = (second.get("holdout") or {}).get("mode", "?")
    protocol_a = f"{first.get('protocol', '?')}/{mode_a}"
    protocol_b = f"{second.get('protocol', '?')}/{mode_b}"

    print("=" * 78)
    print("  SCORE MATRIX COMPARISON")
    print("=" * 78)
    print(f"  A = {label_a}   holdout={protocol_a}   rows={first.get('intent_labelled', '?')}")
    print(f"  B = {label_b}   holdout={protocol_b}   rows={second.get('intent_labelled', '?')}")
    if protocol_a != protocol_b:
        print()
        print("  WARNING: the two reports were produced under different holdout")
        print("  protocols, so the comparison is not apples to apples.")
    print()

    print(f"  {'metric':<26s} {'A':>9s} {'B':>9s} {'B - A':>9s}  better")
    print(f"  {'-' * 26} {'-' * 9} {'-' * 9} {'-' * 9}  {'-' * 8}")
    for name, path in METRICS:
        value_a = _get(first, path)
        value_b = _get(second, path)
        if value_a is None or value_b is None:
            print(f"  {name:<26s} {'n/a':>9s} {'n/a':>9s}")
            continue
        delta = value_b - value_a
        # Lower is better only for calibration error and PII count.
        lower_is_better = name in {"ECE", "MCE", "PII violations"}
        if abs(delta) < 1e-9:
            verdict = "tie"
        elif (delta < 0) == lower_is_better:
            verdict = "B"
        else:
            verdict = "A"
        print(f"  {name:<26s} {value_a:>9.4f} {value_b:>9.4f} {delta:>+9.4f}  {verdict}")

    print()
    print("  Model state")
    print(f"  {'-' * 26} {'-' * 9} {'-' * 9}")
    for name, path in [
        ("intent C", "model_state.intent_selected_c"),
        ("cv macro-F1 (in-sample)", "model_state.intent_cv_macro_f1"),
        ("intent temperature", "model_state.intent_temperature"),
        ("ensemble hybrid weight", "model_state.ensemble_weights.0"),
        ("ensemble semantic weight", "model_state.ensemble_weights.1"),
        ("ensemble prototype weight", "model_state.ensemble_weights.2"),
        ("prior correction alpha", "model_state.prior_correction_strength"),
        ("escalation threshold", "model_state.escalation_threshold"),
    ]:
        value_a = _get(first, path)
        value_b = _get(second, path)
        rendered_a = f"{value_a:>9.4f}" if value_a is not None else f"{'n/a':>9s}"
        rendered_b = f"{value_b:>9.4f}" if value_b is not None else f"{'n/a':>9s}"
        print(f"  {name:<26s} {rendered_a} {rendered_b}")

    print()
    print("  Blend admission")
    for side, report in ((label_a, first), (label_b, second)):
        admission = (report.get("model_state") or {}).get("blend_admission") or {}
        for name, verification in admission.items():
            print(f"  {side:<22s} {name:<18s} accepted={verification.get('accepted')} "
                  f"gain={verification.get('mean_macro_f1_gain', 0.0):+.4f}")
        bagging = (report.get("model_state") or {}).get("intent_bagging") or {}
        print(f"  {side:<22s} {'intent_bagging':<18s} accepted={bagging.get('accepted')} "
              f"heads={bagging.get('heads')}")

    _paired_test(first_path, second_path, label_a, label_b)
    return 0


def _paired_test(
    first_path: Path, second_path: Path, label_a: str, label_b: str
) -> None:
    """Run McNemar on the two runs' per-row predictions when both exist.

    Two configurations scored on the same rows must be compared as paired data.
    Marginal confidence intervals can overlap heavily while the paired test is
    decisive, because most rows are easy and both get them right — those rows
    carry no information about which system is better.
    """
    rows_a = first_path.with_name(first_path.name + ".rows.csv")
    rows_b = second_path.with_name(second_path.name + ".rows.csv")
    print()
    print("  Paired significance (McNemar, exact)")
    if not rows_a.exists() or not rows_b.exists():
        missing = [str(path) for path in (rows_a, rows_b) if not path.exists()]
        print(f"  skipped: no per-row predictions for {', '.join(missing)}")
        return
    frame_a = pd.read_csv(rows_a)
    frame_b = pd.read_csv(rows_b)
    if len(frame_a) != len(frame_b):
        print(f"  skipped: different row counts ({len(frame_a)} vs {len(frame_b)})")
        return
    shared = frame_a.merge(
        frame_b, on="inbound_id", suffixes=("_a", "_b"), how="inner"
    )
    if shared.empty:
        print("  skipped: no shared inbound_id between the two runs")
        return
    from src.evaluator import SupportPipelineEvaluator

    result = SupportPipelineEvaluator.paired_mcnemar(
        shared["intent_true_a"].tolist(),
        shared["intent_pred_a"].tolist(),
        shared["intent_pred_b"].tolist(),
    )
    print(f"  rows compared                  {len(shared)}")
    print(f"  {label_a} right, {label_b} wrong   {result['a_correct_b_wrong']}")
    print(f"  {label_a} wrong, {label_b} right   {result['a_wrong_b_correct']}")
    print(f"  discordant pairs               {result['discordant_count']}")
    print(f"  exact two-sided p-value        {result['exact_two_sided_p_value']}")
    if result["exact_two_sided_p_value"] < 0.05:
        winner = label_b if result["a_wrong_b_correct"] > result["a_correct_b_wrong"] else label_a
        print(f"  verdict: {winner} is better, and the difference is not attributable to chance")
    else:
        print("  verdict: no significant difference; the two agree aside from noise")


if __name__ == "__main__":
    raise SystemExit(main())
