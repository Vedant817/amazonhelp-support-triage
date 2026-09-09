"""Transparent intent baselines retained for benchmark comparison."""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INTENT_TAXONOMY = [
    "ORDER_TRACKING_AND_DELAY",
    "REFUND_AND_RETURN",
    "DAMAGED_OR_WRONG_ITEM",
    "ACCOUNT_AND_SECURITY",
    "PRODUCT_AND_TECHNICAL",
    "FEEDBACK_AND_GENERAL",
]


def preprocess_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    value = html.unescape(text)
    value = re.sub(r"https?://\S+|www\.\S+", "[URL]", value)
    value = re.sub(r"@\w+", "[MENTION]", value)
    return re.sub(r"\s+", " ", value).strip()


class MajorityBaselineClassifier:
    def __init__(self):
        self.majority_class_ = "FEEDBACK_AND_GENERAL"

    def fit(self, texts: List[str], labels: List[str]):
        if labels:
            self.majority_class_ = max(set(labels), key=labels.count)
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_class_] * len(texts)


class BaselineHeuristicClassifier:
    """Customer-keyword baseline; never used as production supervision."""

    def __init__(self):
        self.rules = {
            "DAMAGED_OR_WRONG_ITEM": ["damaged", "broken", "wrong item", "defective", "missing item"],
            "ACCOUNT_AND_SECURITY": ["login", "password", "otp", "2fa", "hacked", "account locked"],
            "PRODUCT_AND_TECHNICAL": ["kindle", "fire tv", "alexa", "echo", "prime video", "device"],
            "REFUND_AND_RETURN": ["refund", "return", "cancel", "money back", "charged twice"],
            "ORDER_TRACKING_AND_DELAY": ["track", "delivery", "carrier", "transit", "package", "late", "shipping"],
        }

    def predict(self, texts: List[str]) -> List[str]:
        results = []
        for text in texts:
            lower = text.lower()
            results.append(next((intent for intent, words in self.rules.items() if any(word in lower for word in words)), "FEEDBACK_AND_GENERAL"))
        return results


class BaselineLogisticClassifier:
    """TF-IDF logistic baseline trained on the same reply-side labels as the encoder."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, preprocessor=preprocess_tweet)
        self.clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        matrix = self.vectorizer.fit_transform(texts)
        self.clf.fit(matrix, labels)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return list(self.clf.predict(self.vectorizer.transform(texts)))

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        probs = self.clf.predict_proba(self.vectorizer.transform([text]))[0]
        index = int(np.argmax(probs))
        return {
            "intent": self.classes_[index],
            "confidence": float(probs[index]),
            "all_probabilities": dict(zip(self.classes_, map(float, probs))),
        }


TfidfBaselineClassifier = BaselineLogisticClassifier
