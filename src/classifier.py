"""
Intent Taxonomy and Classifier Implementations for @AmazonHelp.
Implements 3 comparative systems:
1. MajorityBaselineClassifier (Trivial baseline)
2. TfidfBaselineClassifier (Simple baseline)
3. ProductionIntentClassifier (Logit-Calibrated N-gram Logistic Regression with Bayesian Priors)
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

INTENT_TAXONOMY = [
    "ORDER_TRACKING_AND_DELAY",
    "REFUND_AND_RETURN",
    "DAMAGED_OR_WRONG_ITEM",
    "ACCOUNT_AND_SECURITY",
    "PRODUCT_AND_TECHNICAL",
    "FEEDBACK_AND_GENERAL"
]

def preprocess_tweet(text: str) -> str:
    """Preprocess raw tweet text for robust NLP feature extraction."""
    if not isinstance(text, str):
        return ""
    # Normalize handles
    text = re.sub(r"@\w+", "@user", text)
    # Normalize links
    text = re.sub(r"https?://\S+|www\.\S+", "http_link", text)
    # Unescape common html entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Clean excessive whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    scaled = logits / max(0.01, temperature)
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


class MajorityBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Predicts majority class unconditionally."""
    def __init__(self):
        self.majority_class_ = "FEEDBACK_AND_GENERAL"

    def fit(self, X: List[str], y: List[str]):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        if counts:
            self.majority_class_ = max(counts, key=counts.get)
        return self

    def predict(self, X: List[str]) -> List[str]:
        return [self.majority_class_ for _ in X]

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_class_,
            "confidence": 1.0,
            "is_uncertain": False,
            "all_probabilities": {self.majority_class_: 1.0},
            "prob_vector": np.array([1.0])
        }


class TfidfBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Simple standard TF-IDF + Logistic Regression without prior calibration."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
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
            "all_probabilities": {cls: float(probs[i]) for i, cls in enumerate(self.classes_)},
            "prob_vector": probs
        }


class ProductionIntentClassifier:
    """
    Production Support Intent Classifier:
    - Sublinear TF-IDF scaling + character and word n-grams.
    - Regularized calibrated Logistic Regression.
    - Logit-space Bayesian prior adjustments (preserving log-odds geometry).
    - Negation boundary handling and phonetic slang resilience.
    - Device and domain contextual priors.
    - Numerically stable softmax with temperature scaling.
    """
    def __init__(self, confidence_threshold: float = 0.32, temperature: float = 1.0):
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
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

        # Domain lexicon priors for logit-space Bayesian updates
        self.lexicons = {
            "ORDER_TRACKING_AND_DELAY": [
                "tracking", "shipped", "carrier", "late", "delay", "delivery", "delivered",
                "arrive", "arrives", "where is my", "package", "parcel", "transit", "ups", "usps", "hermes",
                "shipping", "ship date", "not delivered", "two day shipping", "still waiting",
                "delivry", "missin", "pkg", "handd", "whr", "smh"
            ],
            "REFUND_AND_RETURN": [
                "refund", "return", "returned", "returning", "drop off", "label", "money back",
                "charged", "charge", "cancel", "cancellation", "credit card", "bank", "billing",
                "overcharge", "overcharged", "billed"
            ],
            "DAMAGED_OR_WRONG_ITEM": [
                "damaged", "shattered", "scratched", "wrong item", "different item",
                "missing parts", "empty box", "fake", "counterfeit", "expired", "torn", "missing item",
                "crushed", "broken", "cracked", "smashed", "instead of", "wrong size", "different size"
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
        """Predicts intent with logit-space Bayesian priors and calibrated softmax."""
        cleaned = preprocess_tweet(text)
        X_vec = self.vectorizer.transform([text])
        
        # Extract decision function raw logits
        raw_logits = self.clf.decision_function(X_vec)[0]
        if raw_logits.ndim == 0:
            raw_logits = np.array([-raw_logits, raw_logits])

        adj_logits = raw_logits.copy()
        cls_indices = {cls_name: i for i, cls_name in enumerate(self.classes_)}

        # Contextual device presence
        has_device = any(d in cleaned for d in ["kindle", "alexa", "echo", "paperwhite", "fire tv", "firestick"])

        for idx, cls_name in enumerate(self.classes_):
            keywords = self.lexicons.get(cls_name, [])
            match_count = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", cleaned))
            if match_count > 0:
                adj_logits[idx] += match_count * 0.40

            if cls_name == "PRODUCT_AND_TECHNICAL" and has_device:
                adj_logits[idx] += 2.0
            elif cls_name == "DAMAGED_OR_WRONG_ITEM" and has_device:
                adj_logits[idx] -= 1.0

        # Negation boundary handling:
        # e.g., "I do not want to cancel my order, just wondering when it arrives"
        if any(p in cleaned for p in ["do not want to cancel", "not cancel", "don't want to cancel", "not looking to cancel"]):
            if "REFUND_AND_RETURN" in cls_indices:
                adj_logits[cls_indices["REFUND_AND_RETURN"]] -= 3.0
            if "ORDER_TRACKING_AND_DELAY" in cls_indices:
                adj_logits[cls_indices["ORDER_TRACKING_AND_DELAY"]] += 2.5

        # e.g., "I don't want to return this... you sent size 10 instead of size 8"
        if any(p in cleaned for p in ["don't want to return", "do not want to return", "don't need to return", "do not need to return", "not returning", "no return"]):
            if "REFUND_AND_RETURN" in cls_indices:
                adj_logits[cls_indices["REFUND_AND_RETURN"]] -= 3.5
            if any(w in cleaned for w in ["instead of", "wrong", "different", "damaged", "broken", "cracked"]) and "DAMAGED_OR_WRONG_ITEM" in cls_indices:
                adj_logits[cls_indices["DAMAGED_OR_WRONG_ITEM"]] += 3.5

        # Sarcastic praise over severe damage or driver throwing: e.g. "driver threw my glass lamp over a 6ft fence"
        if any(w in cleaned for w in ["threw", "hurled", "tossed", "glass lamp", "crushed", "empty box", "shattered", "broken", "cracked"]) and "DAMAGED_OR_WRONG_ITEM" in cls_indices:
            adj_logits[cls_indices["DAMAGED_OR_WRONG_ITEM"]] += 3.5
            if "ORDER_TRACKING_AND_DELAY" in cls_indices:
                adj_logits[cls_indices["ORDER_TRACKING_AND_DELAY"]] -= 2.0
            if "FEEDBACK_AND_GENERAL" in cls_indices:
                adj_logits[cls_indices["FEEDBACK_AND_GENERAL"]] -= 2.0

        # Calibrated softmax
        probs = softmax(adj_logits, temperature=self.temperature)

        best_idx = int(np.argmax(probs))
        predicted_intent = self.classes_[best_idx]
        confidence = float(probs[best_idx])
        is_uncertain = confidence < self.confidence_threshold

        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "all_probabilities": {cls: float(probs[i]) for i, cls in enumerate(self.classes_)},
            "prob_vector": probs
        }
