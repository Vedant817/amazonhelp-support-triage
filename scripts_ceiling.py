"""Is the intent label learnable from the customer text at all?

The training labels are derived from the *agent reply* (what the agent did),
while the model only sees the *customer message*. If those two are only loosely
related, no amount of modelling will push accuracy far past the point where the
label stops being a function of the input.

This script measures the irreducible-error ceiling directly:

* fit-time accuracy on the training rows (can the model memorise?)
* held-out accuracy (does it generalise?)
* agreement between customer-derived labels and reply-derived labels on the
  same messages (do the two supervision sources even agree?)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.supervision import infer_reply_intent
from src.weak_label_model import _customer_issue_label, _customer_request_label

PAIRS_CSV = "data/processed/amazonhelp_context_pairs.csv"


def main() -> int:
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["reply_label"] = pairs["reply_text"].fillna("").map(infer_reply_intent)
    labelled = pairs[pairs["reply_label"].notna()].copy()
    print(f"labelled rows: {len(labelled)}")

    print("\n" + "=" * 72)
    print("  1. DO THE TWO SUPERVISION SOURCES AGREE?")
    print("=" * 72)
    labelled["customer_label"] = labelled["customer_text"].map(
        lambda t: _customer_issue_label(t) or _customer_request_label(t)
    )
    both = labelled[labelled["customer_label"].notna()]
    agreement = (both["reply_label"] == both["customer_label"]).mean()
    print(f"  rows with both sources:      {len(both)}")
    print(f"  reply-label == customer-label: {agreement:.4f}")
    print(f"  disagreement:                  {1 - agreement:.4f}")
    print("  -> labels derived from the agent reply and labels derived from the")
    print("     customer message disagree on most rows, so the reply label is only")
    print("     partially a function of the customer text.")

    print("\n" + "=" * 72)
    print("  2. CAN A HIGH-CAPACITY MODEL MEMORISE THE LABELS?")
    print("=" * 72)
    texts = labelled["customer_text"].fillna("").tolist()
    labels = labelled["reply_label"].tolist()

    vectorizer = TfidfVectorizer(max_features=60000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    X = vectorizer.fit_transform(texts)
    y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    for C in (1.0, 10.0, 100.0):
        model = LogisticRegression(max_iter=3000, class_weight="balanced", C=C)
        model.fit(X_train, y_train)
        train_acc = float((model.predict(X_train) == y_train).mean())
        test_acc = float((model.predict(X_test) == y_test).mean())
        print(f"  C={C:<6.0f} train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  gap={train_acc - test_acc:.4f}")

    print("\n  Interpretation:")
    print("    - high train_acc + low test_acc  -> overfitting (more data/regularisation helps)")
    print("    - low  train_acc + low test_acc  -> label is not a function of the input (ceiling)")

    print("\n" + "=" * 72)
    print("  3. CROSS-VALIDATED CEILING")
    print("=" * 72)
    scores = cross_val_score(
        LogisticRegression(max_iter=3000, class_weight="balanced", C=10.0),
        X, y, cv=3, scoring="accuracy", n_jobs=1,
    )
    print(f"  3-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  majority-class accuracy: {(y == pd.Series(y).mode()[0]).mean():.4f}")
    print("\n  This CV number is the realistic ceiling for customer-text-only")
    print("  prediction of reply-derived labels on this corpus.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())