"""A/B the intent feature stack on a deployment-representative holdout.

The pipeline fuses word TF-IDF + character TF-IDF + MiniLM embeddings and fits
one-vs-rest liblinear logistic regression. A plain word-TF-IDF + multinomial
recipe scores higher on both the hard-case review sample and a random holdout,
so this isolates which component is responsible.

Variants (all on the same 75/25 random split, no tuning on the test side):
  word              : word TF-IDF only, multinomial
  word+char         : adds character n-grams
  word+minilm       : adds frozen MiniLM embeddings
  word+char+minilm  : the current pipeline feature stack
  word (ovr)        : word TF-IDF with the pipeline's one-vs-rest liblinear head
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.supervision import infer_reply_intent

PAIRS_CSV = "data/processed/amazonhelp_context_pairs.csv"


def _word(texts):
    vectorizer = TfidfVectorizer(max_features=60000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    return vectorizer.fit_transform(texts), vectorizer


def _char(texts):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 6), max_features=60000, sublinear_tf=True, min_df=3
    )
    return vectorizer.fit_transform(texts), vectorizer


def main() -> int:
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["intent_label"] = pairs["reply_text"].fillna("").map(infer_reply_intent)
    labelled = pairs[pairs["intent_label"].notna()].copy()

    rng = np.random.default_rng(42)
    mask = rng.random(len(labelled)) < 0.25
    train = labelled[~mask]
    test = labelled[mask]
    train_text = train["customer_text"].fillna("").tolist()
    test_text = test["customer_text"].fillna("").tolist()
    y_train, y_test = train["intent_label"], test["intent_label"]
    print(f"train={len(train)}  test={len(test)}\n")

    # Fit the shared components once.
    word_train, word_vec = _word(train_text)
    word_test = word_vec.transform(test_text)
    char_train, char_vec = _char(train_text)
    char_test = char_vec.transform(test_text)

    from src.dual_head import MiniLMEncoder

    encoder = MiniLMEncoder()
    emb_train = csr_matrix(encoder.encode(train_text))
    emb_test = csr_matrix(encoder.encode(test_text))

    variants = {
        "word": (word_train, word_test, "multinomial"),
        "word+char": (hstack([word_train, char_train]), hstack([word_test, char_test]), "multinomial"),
        "word+minilm": (hstack([word_train, emb_train]), hstack([word_test, emb_test]), "multinomial"),
        "word+char+minilm": (
            hstack([word_train, char_train, emb_train]),
            hstack([word_test, char_test, emb_test]),
            "multinomial",
        ),
        "word (ovr)": (word_train, word_test, "ovr"),
    }

    print(f"{'variant':<20s} {'accuracy':>9s} {'macro_f1':>9s}")
    results = {}
    for name, (X_train, X_test, head) in variants.items():
        if head == "ovr":
            model = OneVsRestClassifier(
                LogisticRegression(max_iter=2000, C=30, solver="liblinear", random_state=42)
            )
        else:
            model = LogisticRegression(max_iter=3000, C=30)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        macro = f1_score(y_test, pred, average="macro", zero_division=0)
        results[name] = (acc, macro)
        print(f"{name:<20s} {acc:>9.4f} {macro:>9.4f}")

    best = max(results, key=lambda k: results[k][0])
    print(f"\nbest variant: {best}  accuracy={results[best][0]:.4f}  macro_f1={results[best][1]:.4f}")
    print("\nInterpretation: whichever variant wins identifies the component the")
    print("pipeline should keep; the rest is dilution on this corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())