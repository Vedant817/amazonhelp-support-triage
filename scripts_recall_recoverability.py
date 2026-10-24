"""Can the escalation recall shortfall under drift be fixed by recalibration?

The temporal protocol reports 0.768 escalation recall against a 0.85 target: on
the newest 15% of conversations the learned score ranks true escalations lower
than it does on random traffic. There are two possible causes and they call for
opposite responses.

* **Calibration drift.** The score still ranks well, but its scale shifted, so a
  threshold derived on older traffic sits in the wrong place. Re-deriving the
  threshold on recent data fixes this, and the fix is operational.
* **Ranking decay.** The head genuinely orders recent traffic worse. No
  threshold can repair ordering, and the fix has to be retraining or features.

The distinction is measurable without retraining anything: take per-row risk
scores from a saved run and sweep every threshold, asking whether a recall floor
is reachable at a precision materially better than the base rate. If it is, the
operating point was simply stale. If every threshold that reaches the floor sits
at the base rate, the ordering is gone.

Usage:
    python scripts_recall_recoverability.py models/_final_temporal.json.rows.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RECALL_FLOORS = (0.80, 0.85, 0.90, 0.95)
SERVED_THRESHOLD_CANDIDATES = (0.14, 0.16, 0.18, 0.20, 0.24, 0.26, 0.30, 0.35, 0.40)


def _truth(series: pd.Series) -> np.ndarray:
    """Parse the escalation target, which is written as a nullable bool."""
    return series.astype(str).str.strip().str.lower().eq("true").to_numpy()


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "models/_final_temporal.json.rows.csv")
    frame = pd.read_csv(path)
    if "escalate_true" not in frame.columns or "escalation_risk" not in frame.columns:
        print(f"{path} has no escalation columns; rerun scripts_score.py to regenerate it")
        return 2

    labelled = frame[frame["escalate_true"].notna()].copy()
    truth = _truth(labelled["escalate_true"])
    risk = labelled["escalation_risk"].astype(float).to_numpy()
    positives = int(truth.sum())
    base_rate = positives / len(truth)

    print("=" * 74)
    print(f"  RECALL RECOVERABILITY  ({path.name})")
    print("=" * 74)
    print(f"  labelled rows     {len(truth)}")
    print(f"  positives         {positives}")
    print(f"  base rate         {base_rate:.4f}   (precision of escalating everything)")

    print()
    print("  Every threshold, so the whole operating range is visible")
    print(f"  {'threshold':>10s} {'recall':>8s} {'precision':>10s} {'lift':>7s} {'TP':>5s} {'FP':>6s}")
    for threshold in SERVED_THRESHOLD_CANDIDATES:
        predicted = risk >= threshold
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        recall = tp / positives if positives else 0.0
        precision = tp / max(tp + fp, 1)
        lift = precision / base_rate if base_rate else 0.0
        print(f"  {threshold:>10.4f} {recall:>8.4f} {precision:>10.4f} "
              f"{lift:>6.2f}x {tp:>5d} {fp:>6d}")

    print()
    print("  Most precise achievable threshold at each recall floor")
    print(f"  {'floor':>6s} {'precision':>10s} {'lift':>7s} {'threshold':>10s} {'recall':>8s}")
    verdicts = []
    for floor in RECALL_FLOORS:
        best = None
        for threshold in np.unique(risk):
            predicted = risk >= threshold
            tp = int((predicted & truth).sum())
            fp = int((predicted & ~truth).sum())
            recall = tp / positives if positives else 0.0
            if recall + 1e-12 < floor:
                continue
            precision = tp / max(tp + fp, 1)
            if best is None or (precision, threshold) > (best[0], best[1]):
                best = (precision, float(threshold), recall)
        if best is None:
            print(f"  {floor:>6.2f} {'unreachable at any threshold':>44s}")
            continue
        precision, threshold, recall = best
        lift = precision / base_rate if base_rate else 0.0
        print(f"  {floor:>6.2f} {precision:>10.4f} {lift:>6.2f}x {threshold:>10.4f} {recall:>8.4f}")
        verdicts.append((floor, lift))

    print()
    print("  Reading")
    if not verdicts:
        print("  No recall floor is reachable, so the score carries no usable signal")
        print("  on this slice at all.")
        return 0
    worst_lift = min(lift for _, lift in verdicts)
    if worst_lift < 1.25:
        print(f"  Recalibration cannot help: at the tightest reachable floor, precision is")
        print(f"  only {worst_lift:.2f}x the base rate. The score still reaches the floor, but")
        print("  only by escalating almost everything, so the ordering has decayed and the")
        print("  fix is retraining or new features, not a new threshold.")
    else:
        print(f"  Recalibration is worth doing: precision holds at {worst_lift:.2f}x the base")
        print("  rate even at the tightest reachable floor, so the operating point is simply")
        print("  stale and should be re-derived on recent traffic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
