"""A/B the tweet preprocessing on a fixed holdout.

Identifier-like digit runs (order numbers, tracking IDs, phone numbers) are
effectively unique, so a model that learns them cannot generalise: the value it
memorised will never appear again. Normalising them to shared placeholder
tokens should reduce vocabulary churn on unseen data.

This measures that directly with the fast recipe rather than a full pipeline
fit, so the comparison is cheap and the only variable is the preprocessor.
"""

from __future__ import annotations

import html
import re
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


def current_preprocess(text: str) -> str:
    """The preprocessor currently used by the pipeline."""
    if not isinstance(text, str):
        return ""
    value = html.unescape(text)
    value = re.sub(r"https?://\S+|www\.\S+", "[URL]", value)
    value = re.sub(r"@\w+", "[MENTION]", value)
    return re.sub(r"\s+", " ", value).strip()


def normalized_preprocess(text: str) -> str:
    """Adds identifier normalisation on top of the current behaviour."""
    value = current_preprocess(text)
    value = re.sub(r"\b\d{3}-\d{7}-\d{7}\b", " [ORDERID] ", value)
    value = re.sub(r"\b\+?\d[\d\s\-]{8,}\d\b", " [LONGNUM] ", value)
    value = re.sub(r"\b\d{5,}\b", " [LONGNUM] ", value)
    return re.sub(r"\s+", " ", value).strip()


def evaluate(preprocess, train, test) -> tuple[float, float, int]:
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess, max_features=60000, ngram_range=(1, 3),
        sublinear_tf=True, min_df=2,
    )
    X_train = vectorizer.fit_transform(train["customer_text"].fillna(""))
    X_test = vectorizer.transform(test["customer_text"].fillna(""))
    model = LogisticRegression(max_iter=3000, C=30)
    model.fit(X_train, train["intent_label"])
    pred = model.predict(X_test)
    return (
        accuracy_score(test["intent_label"], pred),
        f1_score(test["intent_label"], pred, average="macro", zero_division=0),
        len(vectorizer.vocabulary_),
    )


def main() -> int:
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["intent_label"] = pairs["reply_text"].fillna("").map(infer_reply_intent)
    labelled = pairs[pairs["intent_label"].notna()].copy()

    rng = np.random.default_rng(42)
    mask = rng.random(len(labelled)) < 0.25
    test = labelled[mask]
    train = labelled[~mask]
    print(f"train={len(train)}  test={len(test)}")

    print(f"\n{'preprocessor':<24s} {'accuracy':>9s} {'macro_f1':>9s} {'vocab':>8s}")
    results = {}
    for name, fn in (("current", current_preprocess), ("normalized", normalized_preprocess)):
        acc, macro, vocab = evaluate(fn, train, test)
        results[name] = (acc, macro)
        print(f"{name:<24s} {acc:>9.4f} {macro:>9.4f} {vocab:>8d}")

    delta_acc = results["normalized"][0] - results["current"][0]
    delta_f1 = results["normalized"][1] - results["current"][1]
    print(f"\ndelta accuracy {delta_acc:+.4f}   delta macro-F1 {delta_f1:+.4f}")

    # Show what normalisation actually collapses.
    sample = "Order 408-0107705-3081963 phone 555-123-4567 code 998877"
    print(f"\ncurrent   : {current_preprocess(sample)}")
    print(f"normalized: {normalized_preprocess(sample)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())