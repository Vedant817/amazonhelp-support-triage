"""MiniLM encoder with calibrated intent and escalation heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion

from src.classifier import INTENT_TAXONOMY, preprocess_tweet


MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


class TemperatureScaler:
    """A single positive temperature fitted on a label holdout by NLL."""

    def __init__(self) -> None:
        self.temperature_ = 1.0
        self.is_fitted_ = False

    def fit_multiclass(self, logits: np.ndarray, labels: Sequence[int]) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)

        def nll(log_temperature: float) -> float:
            probs = _softmax(logits / math.exp(log_temperature))
            return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))))

        result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        self.temperature_ = float(math.exp(result.x))
        self.is_fitted_ = True
        return self

    def fit_binary(self, logits: np.ndarray, labels: Sequence[bool]) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=float)

        def nll(log_temperature: float) -> float:
            probs = np.clip(_sigmoid(logits / math.exp(log_temperature)), 1e-12, 1.0 - 1e-12)
            return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))

        result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        self.temperature_ = float(math.exp(result.x))
        self.is_fitted_ = True
        return self


class MiniLMEncoder:
    """Lazily loaded, revision-pinned local sentence-transformer encoder."""

    def __init__(
        self,
        model_name: str = MINILM_MODEL_ID,
        revision: str = MINILM_REVISION,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                revision=self.revision,
                device=self.device,
                trust_remote_code=False,
            )
        return self._model

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(
                [str(text) for text in texts],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


def select_recall_threshold(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    target_recall: float = 0.85,
) -> float:
    """Choose the highest threshold that achieves the target positive-class recall."""
    labels_arr = np.asarray(labels, dtype=bool)
    probs_arr = np.asarray(probabilities, dtype=float)
    positives = int(labels_arr.sum())
    if positives == 0:
        raise ValueError("Escalation calibration requires at least one positive example")
    candidates = np.unique(np.concatenate(([0.0], probs_arr, [1.0])))
    valid = []
    for threshold in candidates:
        predicted = probs_arr >= threshold
        tp = int(np.sum(predicted & labels_arr))
        fp = int(np.sum(predicted & ~labels_arr))
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        if recall >= target_recall:
            valid.append((float(threshold), precision, recall))
    if not valid:
        return 0.0
    return max(valid, key=lambda item: (item[1], item[0]))[0]


def select_precision_threshold(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    target_precision: float = 0.80,
) -> float:
    """Choose the lowest threshold meeting a precision floor, then maximize recall."""
    labels_arr = np.asarray(labels, dtype=bool)
    probs_arr = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate((probs_arr, [1.0])))
    valid = []
    for threshold in candidates:
        predicted = probs_arr >= threshold
        tp = int(np.sum(predicted & labels_arr))
        fp = int(np.sum(predicted & ~labels_arr))
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / max(int(labels_arr.sum()), 1)
        if precision >= target_precision:
            valid.append((float(threshold), recall, precision))
    if not valid:
        return 1.0
    return max(valid, key=lambda item: (item[1], -item[0]))[0]


class DualHeadSupportModel:
    """Semantic-plus-lexical features with calibrated intent and escalation heads."""

    def __init__(
        self,
        encoder: Optional[MiniLMEncoder] = None,
        intent_confidence_threshold: float = 0.45,
        target_escalation_recall: float = 0.85,
        target_high_risk_precision: float = 0.80,
        intent_noise_prune_fraction: float = 0.05,
    ) -> None:
        self.encoder = encoder or MiniLMEncoder()
        self.intent_confidence_threshold = intent_confidence_threshold
        self.target_escalation_recall = target_escalation_recall
        self.target_high_risk_precision = target_high_risk_precision
        self.intent_noise_prune_fraction = intent_noise_prune_fraction
        self.intent_vectorizer = self._make_vectorizer()
        self.escalation_vectorizer = self._make_vectorizer()
        self.intent_head = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
                C=1.0,
                solver="liblinear",
            )
        )
        self.escalation_head = LogisticRegression(
            max_iter=2000,
            random_state=42,
            C=0.5,
        )
        self.intent_temperature = TemperatureScaler()
        self.escalation_temperature = TemperatureScaler()
        self.escalation_threshold_ = 0.5
        self.high_precision_escalation_threshold_ = 1.0
        self.intent_pruned_count_ = 0
        self.classes_: List[str] = []

    @staticmethod
    def _make_vectorizer() -> FeatureUnion:
        return FeatureUnion([
            ("word", TfidfVectorizer(
                preprocessor=preprocess_tweet,
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
            )),
            ("character", TfidfVectorizer(
                preprocessor=preprocess_tweet,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_features=40000,
                sublinear_tf=True,
            )),
        ])

    @staticmethod
    def _hybrid_features(vectorizer: FeatureUnion, texts: Sequence[str], embeddings: np.ndarray):
        lexical = vectorizer.transform([str(text) for text in texts])
        return hstack([lexical, csr_matrix(embeddings)], format="csr")

    def _intent_keep_mask(self, embeddings: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        """Cross-fitted self-confidence filter for the noisiest weak labels."""
        labels_arr = np.asarray(labels)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        audit_head = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42, C=2.0
        )
        oof = cross_val_predict(
            audit_head, embeddings, labels_arr, cv=splitter, method="predict_proba"
        )
        classes = np.unique(labels_arr)
        class_index = {label: idx for idx, label in enumerate(classes)}
        label_indices = np.asarray([class_index[label] for label in labels_arr])
        self_confidence = oof[np.arange(len(labels_arr)), label_indices]
        keep = np.ones(len(labels_arr), dtype=bool)
        for label in classes:
            indices = np.flatnonzero(labels_arr == label)
            cutoff = np.quantile(self_confidence[indices], self.intent_noise_prune_fraction)
            keep[indices] = self_confidence[indices] >= cutoff
        return keep

    def fit(
        self,
        intent_texts: Sequence[str],
        intent_labels: Sequence[str],
        escalation_texts: Sequence[str],
        escalation_labels: Sequence[bool],
        cal_intent_texts: Sequence[str],
        cal_intent_labels: Sequence[str],
        cal_escalation_texts: Sequence[str],
        cal_escalation_labels: Sequence[bool],
    ) -> "DualHeadSupportModel":
        intent_embeddings = self.encoder.encode(intent_texts)
        escalation_embeddings = self.encoder.encode(escalation_texts)
        intent_keep = self._intent_keep_mask(intent_embeddings, intent_labels)
        self.intent_pruned_count_ = int((~intent_keep).sum())
        self.intent_vectorizer.fit([str(text) for text in intent_texts])
        intent_features = self._hybrid_features(
            self.intent_vectorizer, intent_texts, intent_embeddings
        )
        self.intent_head.fit(intent_features[intent_keep], np.asarray(intent_labels)[intent_keep])

        self.escalation_vectorizer.fit([str(text) for text in escalation_texts])
        escalation_features = self._hybrid_features(
            self.escalation_vectorizer, escalation_texts, escalation_embeddings
        )
        self.escalation_head.fit(escalation_features, list(escalation_labels))
        self.classes_ = list(self.intent_head.classes_)

        cal_intent_embeddings = self.encoder.encode(cal_intent_texts)
        cal_intent_features = self._hybrid_features(
            self.intent_vectorizer, cal_intent_texts, cal_intent_embeddings
        )
        cal_intent_logits = self.intent_head.decision_function(cal_intent_features)
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        numeric_labels = [class_to_index[label] for label in cal_intent_labels]
        self.intent_temperature.fit_multiclass(cal_intent_logits, numeric_labels)

        cal_escalation_embeddings = self.encoder.encode(cal_escalation_texts)
        cal_escalation_features = self._hybrid_features(
            self.escalation_vectorizer, cal_escalation_texts, cal_escalation_embeddings
        )
        cal_escalation_logits = self.escalation_head.decision_function(cal_escalation_features)
        self.escalation_temperature.fit_binary(cal_escalation_logits, cal_escalation_labels)
        cal_escalation_probs = _sigmoid(
            cal_escalation_logits / self.escalation_temperature.temperature_
        )
        self.escalation_threshold_ = select_recall_threshold(
            cal_escalation_labels,
            cal_escalation_probs,
            self.target_escalation_recall,
        )
        self.high_precision_escalation_threshold_ = select_precision_threshold(
            cal_escalation_labels,
            cal_escalation_probs,
            self.target_high_risk_precision,
        )
        return self

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        embedding = self.encoder.encode([text])
        features = self._hybrid_features(self.intent_vectorizer, [text], embedding)
        logits = np.asarray(self.intent_head.decision_function(features)[0], dtype=float)
        probs = _softmax(logits / self.intent_temperature.temperature_)
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        return {
            "intent": self.classes_[best_idx],
            "confidence": confidence,
            "is_uncertain": confidence < self.intent_confidence_threshold,
            "intent_resolution": "full_message",
            "all_probabilities": {label: float(probs[idx]) for idx, label in enumerate(self.classes_)},
            "prob_vector": probs,
            "logits": logits,
            "embedding": embedding[0],
        }

    def predict_escalation(self, text: str, embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        dense = (
            np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            if embedding is not None
            else self.encoder.encode([text])
        )
        features = self._hybrid_features(self.escalation_vectorizer, [text], dense)
        logit = float(self.escalation_head.decision_function(features)[0])
        probability = float(_sigmoid(np.array([logit / self.escalation_temperature.temperature_]))[0])
        return {
            "should_escalate": probability >= self.escalation_threshold_,
            "risk_score": probability,
            "threshold": self.escalation_threshold_,
            "logit": logit,
        }
