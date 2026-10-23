"""Probabilistic aggregation of abstaining intent labeling functions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp


ABSTAIN = -1


@dataclass
class WeakLabelResult:
    probabilities: np.ndarray
    labels: list[Optional[str]]
    confidences: np.ndarray
    vote_counts: np.ndarray


class DawidSkeneLabelModel:
    """Small multiclass EM label model with abstaining labeling functions."""

    def __init__(self, class_names: Sequence[str], smoothing: float = 1.0, max_iter: int = 100):
        self.class_names = list(class_names)
        self.smoothing = float(smoothing)
        self.max_iter = int(max_iter)
        self.class_prior_: Optional[np.ndarray] = None
        self.confusions_: Optional[np.ndarray] = None

    def fit_predict_proba(self, label_matrix: np.ndarray) -> np.ndarray:
        labels = np.asarray(label_matrix, dtype=int)
        if labels.ndim != 2 or labels.shape[1] < 2:
            raise ValueError("label_matrix must contain at least two labeling functions")
        n, m = labels.shape
        k = len(self.class_names)
        if np.any((labels < ABSTAIN) | (labels >= k)):
            raise ValueError("label_matrix contains an invalid class index")

        # Majority-vote initialization anchors latent components to class names.
        posterior = np.ones((n, k), dtype=float)
        for class_index in range(k):
            posterior[:, class_index] += np.sum(labels == class_index, axis=1)
        posterior /= posterior.sum(axis=1, keepdims=True)

        previous = posterior
        for _ in range(self.max_iter):
            prior = (previous.sum(axis=0) + self.smoothing) / (n + self.smoothing * k)
            confusion = np.empty((m, k, k), dtype=float)
            for source in range(m):
                observed = labels[:, source]
                for true_class in range(k):
                    weighted = np.array([
                        previous[observed == emitted, true_class].sum()
                        for emitted in range(k)
                    ]) + self.smoothing
                    confusion[source, true_class] = weighted / weighted.sum()

            log_posterior = np.tile(np.log(prior), (n, 1))
            for source in range(m):
                observed = labels[:, source]
                present = observed != ABSTAIN
                if not present.any():
                    continue
                for true_class in range(k):
                    log_posterior[present, true_class] += np.log(
                        confusion[source, true_class, observed[present]]
                    )
            log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
            posterior = np.exp(log_posterior)
            if np.max(np.abs(posterior - previous)) < 1e-6:
                break
            previous = posterior

        self.class_prior_ = prior
        self.confusions_ = confusion
        return posterior


def _customer_issue_label(text: str) -> Optional[str]:
    value = str(text or "").lower()
    patterns = (
        ("ACCOUNT_AND_SECURITY", r"\b(?:hacked|phishing|password|login|log in|sign in|otp|2fa|verification code|account locked|suspended|blocked|unauthorized access|credential|2 factor|two factor|reset.*password)\b"),
        ("DAMAGED_OR_WRONG_ITEM", r"\b(?:damaged|broken|defective|wrong item|empty box|missing (?:piece|part|contents)|not what i ordered|incorrect item|different item|cracked|shattered|wrong.*product|wrong.*item)\b"),
        ("REFUND_AND_RETURN", r"\b(?:refund|return|money back|charged twice|double charged|overcharged|cancel.*order|chargeback|dispute.*charge|replacement|exchange)\b"),
        ("ORDER_TRACKING_AND_DELAY", r"\b(?:tracking|delivery|delivered|shipment|shipping|carrier|courier|package|late|not arrived|where.*order|when.*arrive|ship.*date|out for delivery|estimated delivery|prime.*delivery|next day|two day|delayed|stuck|lost|missing.*package|failed delivery)\b"),
        ("PRODUCT_AND_TECHNICAL", r"\b(?:kindle|fire tv|firestick|alexa|echo|prime video|device|bluetooth|wi-?fi|app|website|screen|frozen|not working|setup|connect|wifi|update|firmware|reset|reboot|battery|charge|error|crash|bug|glitch|compatible)\b"),
        ("FEEDBACK_AND_GENERAL", r"\b(?:complaint|feedback|suggestion|praise|thank you|terrible|worst|awful|horrible|disappointed|frustrated|angry|furious|unacceptable|ridiculous|pathetic|useless|worst.*service|never.*again)\b"),
    )
    matches = [label for label, pattern in patterns if re.search(pattern, value)]
    return matches[0] if len(matches) == 1 else None


def _customer_request_label(text: str) -> Optional[str]:
    value = str(text or "").lower()
    if re.search(r"\b(?:how|where)\b.{0,30}\b(?:return|refund|money back)\b", value):
        return "REFUND_AND_RETURN"
    if re.search(r"\b(?:where|track|when)\b.{0,35}\b(?:order|package|delivery|arrive)\b", value):
        return "ORDER_TRACKING_AND_DELAY"
    if re.search(r"\b(?:thank|love|great|excellent|wonderful|amazing)\b", value) and not re.search(
        r"\b(?:late|broken|wrong|refund|hacked|terrible|worst)\b", value
    ):
        return "FEEDBACK_AND_GENERAL"
    if re.search(r"\b(?:help|assist|support)\b.{0,20}\b(?:account|login|password|security)\b", value):
        return "ACCOUNT_AND_SECURITY"
    if re.search(r"\b(?:help|fix|troubleshoot|setup|configure)\b.{0,20}\b(?:kindle|fire|alexa|echo|device|app|prime video|firestick)\b", value):
        return "PRODUCT_AND_TECHNICAL"
    if re.search(r"\b(?:replace|exchange|wrong.*item|wrong.*product)\b", value):
        return "DAMAGED_OR_WRONG_ITEM"
    return None


def _customer_sentiment_label(text: str) -> Optional[str]:
    """Use sentiment + issue keywords to infer intent for ambiguous cases."""
    value = str(text or "").lower()
    negative = re.search(r"\b(?:angry|furious|frustrat|disappoint|upset|unhappy|unacceptable|ridiculous|pathetic|useless|worst|terrible|awful|horrible|never.*again|fed up|done with)\b", value)
    positive = re.search(r"\b(?:thank|great|excellent|amazing|wonderful|love|happy|satisfied|pleased|impressed)\b", value)
    
    if positive and not negative:
        return "FEEDBACK_AND_GENERAL"
    if negative:
        if re.search(r"\b(?:delivery|package|ship|arrive|carrier|courier|tracking)\b", value):
            return "ORDER_TRACKING_AND_DELAY"
        if re.search(r"\b(?:refund|return|money|charge|charged|billing)\b", value):
            return "REFUND_AND_RETURN"
        if re.search(r"\b(?:broken|damaged|wrong|defect|cracked|shattered)\b", value):
            return "DAMAGED_OR_WRONG_ITEM"
        if re.search(r"\b(?:account|login|password|access|credential|locked|suspended)\b", value):
            return "ACCOUNT_AND_SECURITY"
        if re.search(r"\b(?:device|kindle|fire|alexa|echo|app|screen|battery|firmware)\b", value):
            return "PRODUCT_AND_TECHNICAL"
    return None


def aggregate_intent_weak_labels(
    frame: pd.DataFrame,
    class_names: Sequence[str],
    minimum_confidence: float = 0.58,
) -> WeakLabelResult:
    """Aggregate reply, resolution-act, and customer-side weak sources."""
    classes = list(class_names)
    class_index = {label: index for index, label in enumerate(classes)}
    act_map = {
        "ASK_TRACKING_STATUS": "ORDER_TRACKING_AND_DELAY",
        "ASK_CARRIER": "ORDER_TRACKING_AND_DELAY",
        "LINK_ORDERS": "ORDER_TRACKING_AND_DELAY",
        "APOLOGIZE_WAIT": "ORDER_TRACKING_AND_DELAY",
        "LINK_RETURNS": "REFUND_AND_RETURN",
        "TROUBLESHOOT_DEVICE": "PRODUCT_AND_TECHNICAL",
        "DM_WITH_ORDER_ID": "ACCOUNT_AND_SECURITY",
        "GENERAL_SUPPORT": "FEEDBACK_AND_GENERAL",
    }
    sources = []
    for _, row in frame.iterrows():
        customer_input = row.get("customer_text", "")
        # Use reply_weak as primary (covers all 6 classes), then customer-side LFs
        emitted = [
            row.get("intent_label"),  # reply_weak - high precision, covers all classes
            act_map.get(str(row.get("resolution_act", ""))),
            _customer_issue_label(customer_input),
            _customer_request_label(customer_input),
            _customer_sentiment_label(customer_input),
            _customer_urgency_label(customer_input),
        ]
        sources.append([class_index.get(label, ABSTAIN) for label in emitted])
    matrix = np.asarray(sources, dtype=int)
    probabilities = DawidSkeneLabelModel(classes).fit_predict_proba(matrix)
    best = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(frame)), best]
    votes = np.sum(matrix != ABSTAIN, axis=1)
    labels = [
        classes[index] if votes[row] > 0 and confidence[row] >= minimum_confidence else None
        for row, index in enumerate(best)
    ]
    return WeakLabelResult(probabilities, labels, confidence, votes)


def _customer_urgency_label(text: str) -> Optional[str]:
    """Detect urgency signals that correlate with specific intents."""
    value = str(text or "").lower()
    # High urgency + delivery = tracking dispute
    if re.search(r"\b(?:urgent|asap|immediately|right now|emergency|critical)\b", value):
        if re.search(r"\b(?:delivery|package|ship|arrive|carrier)\b", value):
            return "ORDER_TRACKING_AND_DELAY"
        if re.search(r"\b(?:refund|money|charge)\b", value):
            return "REFUND_AND_RETURN"
        if re.search(r"\b(?:hacked|security|locked)\b", value):
            return "ACCOUNT_AND_SECURITY"
    # Polite/patient = likely general feedback or routine tracking
    if re.search(r"\b(?:please|kindly|would you|could you|when you have time)\b", value):
        if re.search(r"\b(?:thank|appreciate)\b", value):
            return "FEEDBACK_AND_GENERAL"
    return None