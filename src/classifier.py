"""
Intent Classification Engine for @AmazonHelp.
Implements:
1. Majority Baseline Classifier
2. Simple TF-IDF Classifier Baseline
3. Production Calibrated Intent Classifier with sublinear TF-IDF, n-gram lexical priors, and confidence scoring.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# 6 Core Target Intents derived from Amazon Customer Service Domain
INTENT_TAXONOMY = [
    "ORDER_TRACKING_AND_DELAY",   # Where is package, carrier transit delays, missing delivery
    "REFUND_AND_RETURN",          # Returns, refund status, return labels, cancellations, billing
    "DAMAGED_OR_WRONG_ITEM",      # Broken items, wrong product shipped, missing parts/items
    "ACCOUNT_AND_SECURITY",       # Login issues, password reset, 2FA/OTP, unauthorized charges, hacked
    "PRODUCT_AND_TECHNICAL",      # Kindle, Echo/Alexa, Prime Video app errors, device troubleshooting
    "FEEDBACK_AND_GENERAL"        # Feedback, store hours, policies, brand praise, general inquiries
]

def preprocess_tweet(text: str) -> str:
    """Clean Twitter handles, normalize whitespace, lowercase, and preserve key punctuation."""
    if not isinstance(text, str):
        return ""
    # Strip user mentions (@AmazonHelp, @115821, etc.)
    text = re.sub(r"@\w+", "", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Strip HTML entities (&amp;, etc.)
    text = re.sub(r"&\w+;", " ", text)
    # Normalize excessive spaces
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


class MajorityBaselineClassifier:
    """Trivial Baseline: Always predicts the most frequent class."""
    def __init__(self):
        self.dummy = DummyClassifier(strategy="most_frequent")

    def fit(self, X: List[str], y: List[str]):
        self.dummy.fit(X, y)
        return self

    def predict(self, X: List[str]) -> List[str]:
        return list(self.dummy.predict(X))

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        pred = self.dummy.predict([text])[0]
        return {
            "intent": pred,
            "confidence": 0.55,
            "is_uncertain": False,
            "all_probabilities": {pred: 0.55}
        }


class TfidfBaselineClassifier:
    """Simple Baseline: Standard unigram TF-IDF with vanilla Logistic Regression."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", preprocessor=preprocess_tweet)
        self.clf = LogisticRegression(max_iter=500, random_state=42)
        self.classes_ = None

    def fit(self, X: List[str], y: List[str]):
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict(self, X: List[str]) -> List[str]:
        X_vec = self.vectorizer.transform(X)
        return list(self.clf.predict(X_vec))

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        X_vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X_vec)[0]
        best_idx = int(np.argmax(probs))
        return {
            "intent": self.classes_[best_idx],
            "confidence": float(probs[best_idx]),
            "is_uncertain": float(probs[best_idx]) < 0.40,
            "all_probabilities": {cls: float(probs[i]) for i, cls in enumerate(self.classes_)}
        }


class ProductionIntentClassifier:
    """
    Production Support Intent Classifier:
    - Sublinear TF-IDF scaling + character and word n-grams.
    - Regularized calibrated Logistic Regression.
    - Intent domain lexicons for high-precision prior boosting.
    - Contextual precedence (device troubleshooting prioritized for Amazon hardware).
    - Calibrated confidence estimation with out-of-domain / ambiguity flagging.
    """
    def __init__(self, confidence_threshold: float = 0.32):
        self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            preprocessor=preprocess_tweet
        )
        self.clf = LogisticRegression(
            C=2.5,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
        self.classes_ = None

        # Domain lexicon priors for ambiguity tie-breaking
        self.lexicons = {
            "ORDER_TRACKING_AND_DELAY": [
                "tracking", "shipped", "carrier", "late", "delay", "delivery", "delivered",
                "arrive", "where is my", "package", "parcel", "transit", "ups", "usps", "hermes",
                "shipping", "ship date", "not delivered", "two day shipping", "still waiting"
            ],
            "REFUND_AND_RETURN": [
                "refund", "return", "returned", "returning", "drop off", "label", "money back",
                "charged", "charge", "cancel", "cancellation", "credit card", "bank", "billing",
                "overcharge", "overcharged", "billed"
            ],
            "DAMAGED_OR_WRONG_ITEM": [
                "damaged", "shattered", "scratched", "wrong item", "different item",
                "missing parts", "empty box", "fake", "counterfeit", "expired", "torn", "missing item"
            ],
            "ACCOUNT_AND_SECURITY": [
                "login", "password", "otp", "2fa", "verification", "locked", "hacked",
                "unauthorized", "sign in", "prime membership", "compromised", "identity", "account"
            ],
            "PRODUCT_AND_TECHNICAL": [
                "kindle", "fire tv", "firestick", "alexa", "echo", "prime video", "app",
                "streaming", "screen", "wifi", "bluetooth", "update", "audio", "device", "paperwhite",
                "firmware", "sync"
            ],
            "FEEDBACK_AND_GENERAL": [
                "worst service", "awful", "terrible", "disappointed", "thank you", "great service",
                "useless", "scam", "shame on you", "complaint", "feedback", "customer care", "kudos"
            ]
        }

    def fit(self, X: List[str], y: List[str]):
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        """Predicts intent along with calibrated confidence and fallback signals."""
        cleaned = preprocess_tweet(text)
        X_vec = self.vectorizer.transform([text])
        raw_probs = self.clf.predict_proba(X_vec)[0]

        # Prior boost based on domain keywords
        boosted_probs = raw_probs.copy()
        
        # Check if text contains Amazon device names -> strong prior for PRODUCT_AND_TECHNICAL
        has_device = any(d in cleaned for d in ["kindle", "alexa", "echo", "paperwhite", "fire tv", "firestick"])

        for idx, cls_name in enumerate(self.classes_):
            keywords = self.lexicons.get(cls_name, [])
            match_count = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", cleaned))
            if match_count > 0:
                boosted_probs[idx] += match_count * 0.10

            if cls_name == "PRODUCT_AND_TECHNICAL" and has_device:
                boosted_probs[idx] += 0.70
            elif cls_name == "DAMAGED_OR_WRONG_ITEM" and has_device:
                boosted_probs[idx] *= 0.20

        # Re-normalize
        boosted_probs = boosted_probs / np.sum(boosted_probs)

        best_idx = int(np.argmax(boosted_probs))
        predicted_intent = self.classes_[best_idx]
        confidence = float(boosted_probs[best_idx])
        
        is_uncertain = confidence < self.confidence_threshold

        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "all_probabilities": {cls: float(boosted_probs[i]) for i, cls in enumerate(self.classes_)}
        }
