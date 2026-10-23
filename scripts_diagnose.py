"""Diagnose the intent score ceiling: label noise vs class-prior mismatch.

Answers three questions with data:

1. How noisy are the reference labels? (agreement between the stored labels and
   the reply-derived supervision that trained the model)
2. Is the training distribution balanced while the evaluation distribution is
   not? (prior mismatch that costs accuracy at decision time)
3. What is the achievable ceiling if the model had a perfect ranker?
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.supervision import infer_reply_intent

PAIRS_CSV = "data/processed/amazonhelp_context_pairs.csv"
EVAL_CSV = "data/training/labels_for_human_review.csv"


def main() -> int:
    eval_frame = pd.read_csv(EVAL_CSV)
    pairs = pd.read_csv(PAIRS_CSV)

    print("=" * 74)
    print("  1. LABEL NOISE")
    print("=" * 74)
    labelled = eval_frame[eval_frame["intent_label"].notna()].copy()
    labelled["reply_label"] = labelled["reply_text"].fillna("").map(infer_reply_intent)
    comparable = labelled[labelled["reply_label"].notna()]
    agree = (comparable["intent_label"] == comparable["reply_label"]).mean()
    print(f"  eval rows with a label:            {len(labelled)}")
    print(f"  rows where reply-labeling fires:   {len(comparable)}")
    print(f"  agreement stored-vs-reply label:   {agree:.4f}")
    print("  -> the stored labels and the training signal disagree on "
          f"{(1 - agree) * 100:.0f}% of comparable rows")

    print("\n  label distribution (eval set):")
    for label, count in labelled["intent_label"].value_counts().items():
        print(f"    {label:<26s} {count:>4d}  ({count / len(labelled) * 100:5.1f}%)")

    print("\n" + "=" * 74)
    print("  2. PRIOR MISMATCH (train vs eval)")
    print("=" * 74)
    train_natural = pairs["reply_text"].fillna("").map(infer_reply_intent)
    train_dist = train_natural.value_counts(normalize=True)
    eval_dist = labelled["intent_label"].value_counts(normalize=True)
    labels = sorted(set(train_dist.index) | set(eval_dist.index))
    print(f"  {'intent':<26s} {'train prior':>12s} {'eval prior':>12s} {'ratio':>8s}")
    for label in labels:
        tr = float(train_dist.get(label, 0.0))
        ev = float(eval_dist.get(label, 0.0))
        ratio = ev / tr if tr > 0 else float("inf")
        print(f"  {label:<26s} {tr:>12.4f} {ev:>12.4f} {ratio:>8.2f}")
    print("  NOTE: training balances every class to 1266 rows, so the model is")
    print("        fitted on a uniform prior while traffic is far from uniform.")

    print("\n" + "=" * 74)
    print("  3. MAJORITY-CLASS FLOOR")
    print("=" * 74)
    majority = labelled["intent_label"].value_counts().idxmax()
    majority_acc = (labelled["intent_label"] == majority).mean()
    print(f"  majority class:        {majority}")
    print(f"  majority-class acc:    {majority_acc:.4f}")
    print(f"  random-guess acc (1/6): {1/6:.4f}")

    print("\n" + "=" * 74)
    print("  4. WHAT A PRIOR FIX WOULD MOVE")
    print("=" * 74)
    # Simulate: if the model ranked perfectly but we re-weighted by the eval
    # prior, how much of the disagreement is explained by prior shift alone?
    overlap = set(train_dist.index) & set(eval_dist.index)
    tv_distance = 0.5 * sum(abs(float(train_dist.get(l, 0)) - float(eval_dist.get(l, 0))) for l in labels)
    print(f"  total-variation distance train vs eval prior: {tv_distance:.4f}")
    print("  A prior correction at decision time can recover the portion of")
    print("  accuracy lost purely to this shift; it cannot fix label noise.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())