"""Apples-to-apples: simple recipe vs current pipeline on the SAME held-out rows.

The ceiling diagnostic trained on a random split of the whole corpus, which is
an easier (natural-distribution) test than the 382-row review sample. This
script instead:

1. trains the simple TF-IDF + logistic-regression recipe with the 382 review
   rows excluded, and
2. scores it on exactly those 382 rows,

so the numbers are directly comparable with the pipeline's held-out score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.supervision import infer_reply_intent

PAIRS_CSV = "data/processed/amazonhelp_context_pairs.csv"
EVAL_CSV = "data/training/labels_for_human_review.csv"


def main() -> int:
    pairs = pd.read_csv(PAIRS_CSV)
    eval_frame = pd.read_csv(EVAL_CSV)
    holdout_ids = set(eval_frame["inbound_id"].astype(str))

    pairs["reply_label"] = pairs["reply_text"].fillna("").map(infer_reply_intent)
    train = pairs[pairs["reply_label"].notna() & ~pairs["inbound_id"].astype(str).isin(holdout_ids)]
    test = eval_frame[eval_frame["intent_label"].notna()]

    print(f"train rows: {len(train)}   test rows: {len(test)}")
    print(f"train prior: {train['reply_label'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"test  prior: {test['intent_label'].value_counts(normalize=True).round(3).to_dict()}")

    results = {}
    for C in (1.0, 3.0, 10.0, 30.0):
        vectorizer = TfidfVectorizer(
            max_features=60000, ngram_range=(1, 3), sublinear_tf=True, min_df=2
        )
        X_train = vectorizer.fit_transform(train["customer_text"].fillna(""))
        X_test = vectorizer.transform(test["customer_text"].fillna(""))
        model = LogisticRegression(max_iter=3000, class_weight="balanced", C=C)
        model.fit(X_train, train["reply_label"])
        pred = model.predict(X_test)
        acc = accuracy_score(test["intent_label"], pred)
        macro = f1_score(test["intent_label"], pred, average="macro", zero_division=0)
        results[C] = (acc, macro)
        print(f"  C={C:<5.0f} accuracy={acc:.4f}  macro_f1={macro:.4f}")

    best_C = max(results, key=lambda c: results[c][0])
    print(f"\nbest simple recipe: C={best_C} accuracy={results[best_C][0]:.4f} macro_f1={results[best_C][1]:.4f}")

    # The current pipeline's held-out numbers, for direct comparison.
    print("\ncurrent pipeline (natural distribution, from the last run):")
    print("  accuracy=0.4517  macro_f1=0.2686")

    majority = test["intent_label"].value_counts().idxmax()
    majority_acc = accuracy_score(test["intent_label"], [majority] * len(test))
    print(f"\nmajority-class baseline on this test set: accuracy={majority_acc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())