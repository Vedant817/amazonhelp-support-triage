"""MiniLM encoder with calibrated intent and escalation heads."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import f1_score
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
        self.contrastive_pairs_seen_ = 0

    def __getstate__(self):
        """Keep generated pipeline caches small; reload pinned weights on demand."""
        state = self.__dict__.copy()
        state["_model"] = None
        return state

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

    def fine_tune_contrastive(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        pair_count: int = 1200,
        batch_size: int = 32,
        epochs: int = 1,
        seed: int = 42,
        output_path: Optional[str] = None,
    ) -> "MiniLMEncoder":
        """Fine-tune locally on balanced positive/negative intent pairs."""
        if pair_count <= 0:
            return self
        values = np.asarray([str(text) for text in texts], dtype=object)
        targets = np.asarray([str(label) for label in labels], dtype=object)
        fingerprint = hashlib.sha256()
        fingerprint.update(f"{self.model_name}:{self.revision}:{pair_count}:{seed}".encode())
        for text, label in zip(values, targets):
            fingerprint.update(str(label).encode("utf-8"))
            fingerprint.update(b"\0")
            fingerprint.update(str(text).encode("utf-8"))
            fingerprint.update(b"\n")
        expected_fingerprint = fingerprint.hexdigest()
        cache_path = Path(output_path) if output_path else None
        metadata_path = cache_path / "support_training_metadata.json" if cache_path else None
        if metadata_path and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("training_fingerprint") == expected_fingerprint:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    str(cache_path), device=self.device, trust_remote_code=False
                )
                self.contrastive_pairs_seen_ = int(metadata["contrastive_pairs"])
                return self

        import torch
        from torch.utils.data import DataLoader
        from sentence_transformers import InputExample, losses

        groups = {label: np.flatnonzero(targets == label) for label in np.unique(targets)}
        groups = {label: indices for label, indices in groups.items() if len(indices) >= 2}
        if len(groups) < 2:
            raise ValueError("Contrastive fine-tuning needs at least two classes with two examples each")
        rng = np.random.default_rng(seed)
        class_names = sorted(groups)
        examples = []
        for index in range(pair_count):
            anchor_class = class_names[index % len(class_names)]
            anchor, positive = rng.choice(groups[anchor_class], size=2, replace=False)
            if index % 2 == 0:
                examples.append(InputExample(texts=[values[anchor], values[positive]], label=1.0))
            else:
                negative_class = rng.choice([label for label in class_names if label != anchor_class])
                negative = int(rng.choice(groups[negative_class]))
                examples.append(InputExample(texts=[values[anchor], values[negative]], label=0.0))

        model = self._load()
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            examples, shuffle=True, batch_size=batch_size, generator=generator
        )
        objective = losses.ContrastiveLoss(model=model, margin=0.5)
        steps = max(1, math.ceil(len(examples) / batch_size) * epochs)
        model.fit(
            train_objectives=[(loader, objective)],
            epochs=epochs,
            warmup_steps=max(1, steps // 10),
            optimizer_params={"lr": 2e-5},
            show_progress_bar=False,
        )
        self.contrastive_pairs_seen_ = len(examples)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(cache_path))
            metadata_path.write_text(
                json.dumps(
                    {
                        "base_model": self.model_name,
                        "base_revision": self.revision,
                        "contrastive_pairs": len(examples),
                        "training_fingerprint": expected_fingerprint,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return self


@dataclass(frozen=True)
class RecallThresholdSelection:
    threshold: float
    observed_recall: float
    recall_lower_bound: float
    positive_count: int
    true_positive_count: int
    target_recall: float
    confidence_level: float
    target_met: bool


def one_sided_binomial_lower_bound(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> float:
    """Exact Clopper-Pearson lower confidence bound for a binomial rate."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence_level, successes, trials - successes + 1))


def select_recall_constrained_threshold(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    target_recall: float = 0.85,
    confidence_level: float = 0.95,
) -> RecallThresholdSelection:
    """Maximize precision while the exact recall lower bound meets its target."""
    labels_arr = np.asarray(labels, dtype=bool)
    probs_arr = np.asarray(probabilities, dtype=float)
    if labels_arr.shape != probs_arr.shape or labels_arr.ndim != 1:
        raise ValueError("Escalation labels and probabilities must be aligned vectors")
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
        lower = one_sided_binomial_lower_bound(tp, positives, confidence_level)
        if lower >= target_recall:
            valid.append((float(threshold), precision, recall, lower, tp))
    if valid:
        threshold, _, recall, lower, tp = max(valid, key=lambda item: (item[1], item[0]))
        return RecallThresholdSelection(
            threshold=threshold,
            observed_recall=recall,
            recall_lower_bound=lower,
            positive_count=positives,
            true_positive_count=tp,
            target_recall=target_recall,
            confidence_level=confidence_level,
            target_met=True,
        )

    # A zero threshold predicts every finite score positive. If even its exact
    # bound misses the target, the calibration sample is too small to certify it.
    tp = positives
    lower = one_sided_binomial_lower_bound(tp, positives, confidence_level)
    return RecallThresholdSelection(
        threshold=0.0,
        observed_recall=1.0,
        recall_lower_bound=lower,
        positive_count=positives,
        true_positive_count=tp,
        target_recall=target_recall,
        confidence_level=confidence_level,
        target_met=lower >= target_recall,
    )


def select_recall_threshold(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    target_recall: float = 0.85,
) -> float:
    """Compatibility wrapper returning the statistically constrained threshold."""
    return select_recall_constrained_threshold(labels, probabilities, target_recall).threshold


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
        contrastive_pair_count: int = 0,
        contrastive_output_path: Optional[str] = None,
        enable_context_gate: bool = False,
        minimum_auxiliary_macro_f1_gain: float = 0.005,
        tune_intent_regularization: bool = False,
        tune_escalation_regularization: bool = True,
    ) -> None:
        self.encoder = encoder or MiniLMEncoder()
        self.intent_confidence_threshold = intent_confidence_threshold
        self.target_escalation_recall = target_escalation_recall
        self.target_high_risk_precision = target_high_risk_precision
        self.intent_noise_prune_fraction = intent_noise_prune_fraction
        self.contrastive_pair_count = int(contrastive_pair_count)
        self.contrastive_output_path = contrastive_output_path
        self.enable_context_gate = bool(enable_context_gate)
        self.minimum_auxiliary_macro_f1_gain = float(minimum_auxiliary_macro_f1_gain)
        self.tune_intent_regularization = bool(tune_intent_regularization)
        self.tune_escalation_regularization = bool(tune_escalation_regularization)
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
        self.escalation_threshold_selection_: Optional[RecallThresholdSelection] = None
        self.high_precision_escalation_threshold_ = 1.0
        self.intent_pruned_count_ = 0
        self.classes_: List[str] = []
        self.reason_head: Optional[LogisticRegression] = None
        self.reason_classes_: List[str] = []
        self.reason_training_count_ = 0
        self.context_gate_confidence_ = 0.0
        self.context_gate_margin_ = 1.0
        self.context_gate_calibration_gain_ = 0.0
        self.intent_prototypes_: Optional[np.ndarray] = None
        self.prototype_temperature_ = 0.1
        self.prototype_blend_weight_ = 0.0
        self.prototype_calibration_gain_ = 0.0
        self.semantic_head = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42, C=2.0
        )
        self.semantic_temperature = TemperatureScaler()
        self.semantic_blend_weight_ = 0.0
        self.semantic_blend_nll_gain_ = 0.0
        self.intent_selected_c_ = 1.0
        self.intent_cv_macro_f1_ = 0.0
        self.escalation_selected_c_ = 0.5
        self.escalation_selected_class_weight_: Optional[str] = None
        self.escalation_cv_average_precision_ = 0.0

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
        reason_texts: Optional[Sequence[str]] = None,
        reason_labels: Optional[Sequence[str]] = None,
        cal_intent_context_texts: Optional[Sequence[str]] = None,
    ) -> "DualHeadSupportModel":
        if self.contrastive_pair_count > 0:
            self.encoder.fine_tune_contrastive(
                intent_texts,
                intent_labels,
                pair_count=self.contrastive_pair_count,
                output_path=self.contrastive_output_path,
            )
        intent_embeddings = self.encoder.encode(intent_texts)
        escalation_embeddings = self.encoder.encode(escalation_texts)
        intent_keep = self._intent_keep_mask(intent_embeddings, intent_labels)
        self.intent_pruned_count_ = int((~intent_keep).sum())
        self.intent_vectorizer.fit([str(text) for text in intent_texts])
        intent_features = self._hybrid_features(
            self.intent_vectorizer, intent_texts, intent_embeddings
        )
        if self.tune_intent_regularization:
            self._select_intent_regularization(
                intent_features[intent_keep], np.asarray(intent_labels)[intent_keep]
            )
        self.intent_head.fit(intent_features[intent_keep], np.asarray(intent_labels)[intent_keep])
        self.semantic_head.fit(
            intent_embeddings[intent_keep], np.asarray(intent_labels)[intent_keep]
        )

        self.escalation_vectorizer.fit([str(text) for text in escalation_texts])
        escalation_features = self._hybrid_features(
            self.escalation_vectorizer, escalation_texts, escalation_embeddings
        )
        if self.tune_escalation_regularization:
            self._select_escalation_regularization(
                escalation_features, np.asarray(escalation_labels, dtype=bool)
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
        cal_semantic_logits = self.semantic_head.decision_function(cal_intent_embeddings)
        self.semantic_temperature.fit_multiclass(cal_semantic_logits, numeric_labels)
        self._fit_semantic_blend(cal_intent_logits, cal_semantic_logits, numeric_labels)
        self._fit_prototype_ensemble(
            intent_embeddings[intent_keep],
            np.asarray(intent_labels)[intent_keep],
            cal_intent_embeddings,
            cal_intent_labels,
            cal_intent_logits,
        )
        if self.enable_context_gate and cal_intent_context_texts is not None:
            self._fit_context_gate(
                cal_intent_texts,
                cal_intent_context_texts,
                cal_intent_labels,
                cal_intent_logits,
            )

        cal_escalation_embeddings = self.encoder.encode(cal_escalation_texts)
        cal_escalation_features = self._hybrid_features(
            self.escalation_vectorizer, cal_escalation_texts, cal_escalation_embeddings
        )
        cal_escalation_logits = self.escalation_head.decision_function(cal_escalation_features)
        self.escalation_temperature.fit_binary(cal_escalation_logits, cal_escalation_labels)
        cal_escalation_probs = _sigmoid(
            cal_escalation_logits / self.escalation_temperature.temperature_
        )
        self.escalation_threshold_selection_ = select_recall_constrained_threshold(
            cal_escalation_labels,
            cal_escalation_probs,
            self.target_escalation_recall,
        )
        self.escalation_threshold_ = self.escalation_threshold_selection_.threshold
        self.high_precision_escalation_threshold_ = select_precision_threshold(
            cal_escalation_labels,
            cal_escalation_probs,
            self.target_high_risk_precision,
        )
        if reason_texts and reason_labels and len(set(reason_labels)) >= 2:
            reason_embeddings = self.encoder.encode(reason_texts)
            reason_features = self._hybrid_features(
                self.intent_vectorizer, reason_texts, reason_embeddings
            )
            self.reason_head = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            ).fit(reason_features, reason_labels)
            self.reason_classes_ = list(self.reason_head.classes_)
            self.reason_training_count_ = len(reason_labels)
        return self

    @staticmethod
    def _intent_estimator(c_value: float) -> OneVsRestClassifier:
        return OneVsRestClassifier(LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            C=c_value,
            solver="liblinear",
        ))

    def _select_intent_regularization(self, features, labels: np.ndarray) -> None:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
        best = (-1.0, -1.0)
        for c_value in (0.25, 0.5, 1.0, 2.0, 4.0):
            score = float(np.mean(cross_val_score(
                self._intent_estimator(c_value),
                features,
                labels,
                cv=splitter,
                scoring="f1_macro",
                n_jobs=1,
            )))
            candidate = (score, -c_value)
            if candidate > best:
                best = candidate
                self.intent_selected_c_ = c_value
                self.intent_cv_macro_f1_ = score
        self.intent_head = self._intent_estimator(self.intent_selected_c_)

    def _select_escalation_regularization(self, features, labels: np.ndarray) -> None:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
        best = (-1.0, -1.0, "")
        for class_weight in (None, "balanced"):
            for c_value in (0.1, 0.25, 0.5, 1.0, 2.0):
                estimator = LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=42,
                    C=c_value,
                )
                score = float(np.mean(cross_val_score(
                    estimator,
                    features,
                    labels,
                    cv=splitter,
                    scoring="average_precision",
                    n_jobs=1,
                )))
                # Prefer less regularization complexity and no reweighting on ties.
                candidate = (score, -c_value, "1" if class_weight is None else "0")
                if candidate > best:
                    best = candidate
                    self.escalation_selected_c_ = c_value
                    self.escalation_selected_class_weight_ = class_weight
                    self.escalation_cv_average_precision_ = score
        self.escalation_head = LogisticRegression(
            max_iter=2000,
            class_weight=self.escalation_selected_class_weight_,
            random_state=42,
            C=self.escalation_selected_c_,
        )

    def _fit_semantic_blend(
        self,
        hybrid_logits: np.ndarray,
        semantic_logits: np.ndarray,
        numeric_labels: Sequence[int],
    ) -> None:
        hybrid = _softmax(np.asarray(hybrid_logits) / self.intent_temperature.temperature_)
        semantic = _softmax(
            np.asarray(semantic_logits) / self.semantic_temperature.temperature_
        )
        labels = np.asarray(numeric_labels, dtype=int)

        def nll(weight: float) -> float:
            probabilities = (1.0 - weight) * hybrid + weight * semantic
            return float(-np.mean(np.log(np.clip(
                probabilities[np.arange(len(labels)), labels], 1e-12, 1.0
            ))))

        baseline = nll(0.0)
        result = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
        gain = baseline - float(result.fun)
        if gain > 1e-4:
            self.semantic_blend_weight_ = float(result.x)
            self.semantic_blend_nll_gain_ = gain

    def _prototype_probabilities(
        self, embeddings: np.ndarray, temperature: Optional[float] = None
    ) -> np.ndarray:
        if self.intent_prototypes_ is None:
            raise RuntimeError("Intent prototypes are not fitted")
        scale = temperature or self.prototype_temperature_
        return _softmax(np.asarray(embeddings) @ self.intent_prototypes_.T / scale)

    def _fit_prototype_ensemble(
        self,
        train_embeddings: np.ndarray,
        train_labels: Sequence[str],
        calibration_embeddings: np.ndarray,
        calibration_labels: Sequence[str],
        calibration_logits: np.ndarray,
    ) -> None:
        """Learn semantic class prototypes and a conservative blend on holdout data."""
        labels = np.asarray(train_labels, dtype=object)
        prototypes = []
        for class_name in self.classes_:
            centroid = np.mean(train_embeddings[labels == class_name], axis=0)
            norm = np.linalg.norm(centroid)
            prototypes.append(centroid / norm if norm else centroid)
        self.intent_prototypes_ = np.asarray(prototypes, dtype=np.float32)
        logistic = _softmax(
            np.asarray(calibration_logits) / self.intent_temperature.temperature_
        )
        truth = np.asarray(calibration_labels, dtype=object)
        class_values = np.asarray(self.classes_, dtype=object)
        baseline_pred = class_values[np.argmax(logistic, axis=1)]
        baseline = float(f1_score(truth, baseline_pred, average="macro", zero_division=0))
        best = (baseline, 0.0, self.prototype_temperature_)
        for temperature in (0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3):
            prototype = self._prototype_probabilities(calibration_embeddings, temperature)
            for weight in np.arange(0.05, 0.51, 0.05):
                blended = (1.0 - weight) * logistic + weight * prototype
                predicted = class_values[np.argmax(blended, axis=1)]
                score = float(f1_score(truth, predicted, average="macro", zero_division=0))
                candidate = (score, -float(weight), float(temperature))
                if candidate > (best[0], -best[1], best[2]):
                    best = (score, float(weight), float(temperature))
        self.prototype_calibration_gain_ = max(0.0, best[0] - baseline)
        if self.prototype_calibration_gain_ >= self.minimum_auxiliary_macro_f1_gain:
            self.prototype_blend_weight_ = best[1]
            self.prototype_temperature_ = best[2]

    def _intent_view(self, text: str) -> Dict[str, Any]:
        embedding = self.encoder.encode([text])
        features = self._hybrid_features(self.intent_vectorizer, [text], embedding)
        logits = np.asarray(self.intent_head.decision_function(features)[0], dtype=float)
        probs = _softmax(logits / self.intent_temperature.temperature_)
        if self.semantic_blend_weight_ > 0.0:
            semantic_logits = self.semantic_head.decision_function(embedding)
            semantic_probs = _softmax(
                semantic_logits / self.semantic_temperature.temperature_
            )[0]
            probs = (
                (1.0 - self.semantic_blend_weight_) * probs
                + self.semantic_blend_weight_ * semantic_probs
            )
        if self.intent_prototypes_ is not None and self.prototype_blend_weight_ > 0.0:
            prototype_probs = self._prototype_probabilities(embedding)[0]
            probs = (
                (1.0 - self.prototype_blend_weight_) * probs
                + self.prototype_blend_weight_ * prototype_probs
            )
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        return {
            "intent": self.classes_[best_idx],
            "confidence": confidence,
            "all_probabilities": {label: float(probs[idx]) for idx, label in enumerate(self.classes_)},
            "prob_vector": probs,
            "logits": logits,
            "embedding": embedding[0],
        }

    def _fit_context_gate(
        self,
        current_texts: Sequence[str],
        context_texts: Sequence[str],
        labels: Sequence[str],
        current_logits: np.ndarray,
    ) -> None:
        """Select a conservative dual-view gate on the intent calibration split."""
        if len(current_texts) != len(context_texts):
            raise ValueError("Current and contextual calibration inputs must align")
        current_probs = _softmax(
            np.asarray(current_logits) / self.intent_temperature.temperature_
        )
        context_embeddings = self.encoder.encode(context_texts)
        context_features = self._hybrid_features(
            self.intent_vectorizer, context_texts, context_embeddings
        )
        context_logits = self.intent_head.decision_function(context_features)
        context_probs = _softmax(context_logits / self.intent_temperature.temperature_)
        current_pred = np.asarray(self.classes_, dtype=object)[np.argmax(current_probs, axis=1)]
        context_pred = np.asarray(self.classes_, dtype=object)[np.argmax(context_probs, axis=1)]
        current_conf = np.max(current_probs, axis=1)
        context_conf = np.max(context_probs, axis=1)
        truth = np.asarray(labels, dtype=object)
        baseline = float(f1_score(truth, current_pred, average="macro", zero_division=0))
        best = (baseline, 0, 0.0, 1.0)
        for confidence_cutoff in np.arange(0.35, 0.86, 0.05):
            for margin in np.arange(0.0, 0.26, 0.05):
                use_context = (
                    (current_conf <= confidence_cutoff)
                    & ((context_conf - current_conf) >= margin)
                    & (np.asarray(current_texts, dtype=object) != np.asarray(context_texts, dtype=object))
                )
                predictions = np.where(use_context, context_pred, current_pred)
                score = float(f1_score(truth, predictions, average="macro", zero_division=0))
                candidate = (score, -int(use_context.sum()), confidence_cutoff, margin)
                if candidate > best:
                    best = candidate
        self.context_gate_calibration_gain_ = max(0.0, best[0] - baseline)
        self.context_gate_confidence_ = float(best[2])
        self.context_gate_margin_ = float(best[3])

    def predict_with_confidence(
        self, text: str, context_text: Optional[str] = None
    ) -> Dict[str, Any]:
        current = self._intent_view(text)
        chosen = current
        context = None
        context_used = False
        if context_text and context_text != text and self.context_gate_confidence_ > 0.0:
            context = self._intent_view(context_text)
            context_used = (
                current["confidence"] <= self.context_gate_confidence_
                and context["confidence"] - current["confidence"] >= self.context_gate_margin_
            )
            if context_used:
                chosen = context
        return {
            **chosen,
            "is_uncertain": chosen["confidence"] < self.intent_confidence_threshold,
            "intent_resolution": "causal_context" if context_used else "current_message",
            "context_used": context_used,
            "current_message_confidence": current["confidence"],
            "context_confidence": context["confidence"] if context is not None else None,
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

    def predict_reason(
        self, text: str, embedding: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """Predict a reviewed closed-set escalation reason when a head is available."""
        if self.reason_head is None:
            return None
        dense = (
            np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            if embedding is not None else self.encoder.encode([text])
        )
        features = self._hybrid_features(self.intent_vectorizer, [text], dense)
        probabilities = self.reason_head.predict_proba(features)[0]
        index = int(np.argmax(probabilities))
        return {
            "reason": self.reason_classes_[index],
            "confidence": float(probabilities[index]),
        }
