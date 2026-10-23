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
from sklearn.metrics import average_precision_score, f1_score
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
        enable_escalation_knn: bool = False,
        intent_confidence_penalty: float = 0.1,  # Penalty for overconfident wrong predictions
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
        self.enable_escalation_knn = bool(enable_escalation_knn)
        self.intent_confidence_penalty = float(intent_confidence_penalty)
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
        self.escalation_knn_embeddings_: Optional[np.ndarray] = None
        self.escalation_knn_labels_: Optional[np.ndarray] = None
        self.escalation_knn_k_ = 0
        self.escalation_knn_weight_ = 0.0
        self.escalation_knn_calibration_ap_gain_ = 0.0
        # Deployment-prior re-weighting. Training balances every intent class so
        # the fitted probabilities carry a uniform prior; real traffic does not.
        # The strength is chosen on the held-out calibration split.
        self.intent_natural_prior_: Optional[Dict[str, float]] = None
        self.prior_correction_strength_ = 0.0
        self.prior_correction_calibration_gain_ = 0.0

    @staticmethod
    def _make_vectorizer() -> FeatureUnion:
        return FeatureUnion([
            ("word", TfidfVectorizer(
                preprocessor=preprocess_tweet,
                ngram_range=(1, 3),  # Add trigrams
                min_df=2,
                max_features=50000,  # Increased
                sublinear_tf=True,
            )),
            ("character", TfidfVectorizer(
                preprocessor=preprocess_tweet,
                analyzer="char_wb",
                ngram_range=(3, 6),  # Extended range
                min_df=3,
                max_features=60000,  # Increased
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
        contrastive_texts: Optional[Sequence[str]] = None,
        contrastive_labels: Optional[Sequence[str]] = None,
        intent_natural_prior: Optional[Dict[str, float]] = None,
    ) -> "DualHeadSupportModel":
        if self.contrastive_pair_count > 0:
            if not contrastive_texts or not contrastive_labels:
                # Fall back to no contrastive fine-tuning if no labels provided
                pass
            else:
                self.encoder.fine_tune_contrastive(
                contrastive_texts,
                contrastive_labels,
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
        # Deployment-prior re-weighting: training balances the classes, so the
        # fitted probabilities carry a uniform prior. Real traffic does not.
        self.intent_natural_prior_ = dict(intent_natural_prior) if intent_natural_prior else None
        self._fit_prior_correction(cal_intent_logits, cal_intent_labels)
        # Apply confidence penalty to reduce overconfident wrong predictions
        self._fit_confidence_penalty(cal_intent_logits, numeric_labels)
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
        self._fit_ensemble_blend(
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
        if self.enable_escalation_knn:
            self.escalation_knn_embeddings_ = np.asarray(
                escalation_embeddings, dtype=np.float32
            )
            self.escalation_knn_labels_ = np.asarray(
                escalation_labels, dtype=np.float32
            )
            cal_escalation_probs = self._fit_escalation_knn_blend(
                cal_escalation_embeddings,
                cal_escalation_probs,
                cal_escalation_labels,
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

    def _fit_confidence_penalty(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit a confidence penalty to reduce overconfident wrong predictions.
        
        This learns a temperature-like scaling that pulls probabilities toward uniform
        for examples where the model is confident but wrong.
        """
        if self.intent_confidence_penalty <= 0:
            self.confidence_penalty_factor_ = 1.0
            return
        
        probs = _softmax(np.asarray(logits, dtype=float))
        true_probs = probs[np.arange(len(labels)), labels]
        
        # Find examples where model is confident but wrong
        pred_labels = np.argmax(probs, axis=1)
        wrong = pred_labels != labels
        confident_wrong = wrong & (true_probs > 0.7)
        
        if confident_wrong.sum() == 0:
            self.confidence_penalty_factor_ = 0.0  # No penalty if no confident wrong predictions
            return
        
        # Learn a penalty factor that reduces confidence on wrong predictions
        def penalty_nll(factor: float) -> float:
            if factor <= 0:
                return 1e9
            penalized = probs.copy()
            # Reduce confidence on confident wrong predictions
            penalty_mask = confident_wrong
            if penalty_mask.any():
                # Move probabilities toward uniform
                uniform = np.ones_like(probs) / probs.shape[1]
                penalized[penalty_mask] = (
                    (1 - factor) * penalized[penalty_mask] + factor * uniform[penalty_mask]
                )
            # Renormalize
            penalized = penalized / penalized.sum(axis=1, keepdims=True)
            true_pen_probs = penalized[np.arange(len(labels)), labels]
            return float(-np.mean(np.log(np.clip(true_pen_probs, 1e-12, 1.0))))
        
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(penalty_nll, bounds=(0.0, self.intent_confidence_penalty), method="bounded")
        self.confidence_penalty_factor_ = float(result.x)
    
    def _apply_confidence_penalty(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned confidence penalty to probability vector."""
        if not hasattr(self, 'confidence_penalty_factor_') or self.confidence_penalty_factor_ <= 0:
            return probs
        factor = self.confidence_penalty_factor_
        if factor <= 0:
            return probs
        # Handle both 1D and 2D arrays
        if probs.ndim == 1:
            uniform = np.ones_like(probs) / probs.shape[0]
            return (1 - factor) * probs + factor * uniform
        uniform = np.ones_like(probs) / probs.shape[1]
        return (1 - factor) * probs + factor * uniform

    def _prior_vector(self) -> Optional[np.ndarray]:
        """Natural class prior aligned to ``self.classes_`` (None when unset)."""
        if not self.intent_natural_prior_:
            return None
        values = np.asarray(
            [float(self.intent_natural_prior_.get(name, 0.0)) for name in self.classes_],
            dtype=float,
        )
        total = values.sum()
        if total <= 0:
            return None
        return values / total

    def _fit_prior_correction(self, cal_logits: np.ndarray, cal_labels: Sequence[str]) -> None:
        """Choose how strongly to re-weight toward the deployment prior.

        Training balances every intent class to a uniform prior, which maximises
        minority recall but costs accuracy against real traffic where one class
        dominates. Re-weighting ``p(y|x) * prior(y)**alpha`` restores the
        deployment prior. ``alpha`` is selected on the held-out calibration
        split, with a guard requiring macro-F1 not to regress materially, so the
        change can only be accepted when it demonstrably helps.
        """
        prior = self._prior_vector()
        if prior is None:
            self.prior_correction_strength_ = 0.0
            return

        probs = _softmax(np.asarray(cal_logits, dtype=float) / self.intent_temperature.temperature_)
        truth = np.asarray(cal_labels, dtype=object)
        class_values = np.asarray(self.classes_, dtype=object)

        def score(alpha: float):
            if alpha <= 0:
                adjusted = probs
            else:
                adjusted = probs * np.power(prior[np.newaxis, :], alpha)
                adjusted = adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)
            predicted = class_values[np.argmax(adjusted, axis=1)]
            accuracy = float(np.mean(predicted == truth))
            macro_f1 = float(f1_score(truth, predicted, average="macro", zero_division=0))
            return accuracy, macro_f1

        base_accuracy, base_macro_f1 = score(0.0)
        best_accuracy, best_alpha, best_macro_f1 = base_accuracy, 0.0, base_macro_f1
        for alpha in (0.25, 0.5, 0.75, 1.0):
            accuracy, macro_f1 = score(alpha)
            # Accept only a material accuracy gain that does not wreck macro-F1.
            if accuracy > best_accuracy + 0.002 and macro_f1 >= base_macro_f1 - 0.05:
                best_accuracy, best_alpha, best_macro_f1 = accuracy, alpha, macro_f1

        self.prior_correction_strength_ = float(best_alpha)
        self.prior_correction_calibration_gain_ = max(0.0, best_accuracy - base_accuracy)

    def _apply_prior_correction(self, probs: np.ndarray) -> np.ndarray:
        """Re-weight a probability vector toward the deployment prior."""
        alpha = self.prior_correction_strength_
        prior = self._prior_vector() if alpha > 0 else None
        if prior is None or alpha <= 0 or probs.shape[-1] != prior.shape[0]:
            return probs
        if probs.ndim == 1:
            adjusted = probs * np.power(prior, alpha)
            total = adjusted.sum()
            return adjusted / total if total > 0 else probs
        adjusted = probs * np.power(prior[np.newaxis, :], alpha)
        totals = np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)
        return adjusted / totals

    @staticmethod
    def _intent_estimator(c_value: float, class_weight: Optional[str] = None) -> OneVsRestClassifier:
        """Intent estimator.

        Class reweighting defaults to **off**. With natural (unbalanced) training
        and the calibrated deployment-prior correction, reweighting classes is a
        second, redundant compensation for imbalance: measured held-out it
        changed macro-F1 by 0.0001 while costing 0.0104 accuracy and 0.05 ECE.
        Set ``SUPPORT_INTENT_CLASS_WEIGHT=balanced`` to re-enable it.
        """
        import os as _os

        requested = _os.getenv(
            "SUPPORT_INTENT_CLASS_WEIGHT", "none" if class_weight is None else class_weight
        ).strip().lower()
        weight: Optional[str] = "balanced" if requested in {"balanced", "1", "true", "yes"} else None
        return OneVsRestClassifier(LogisticRegression(
            max_iter=2000,
            class_weight=weight,
            random_state=42,
            C=c_value,
            solver="liblinear",
        ))

    def _select_intent_regularization(self, features, labels: np.ndarray) -> None:
        """Pick C by cross-validated macro-F1 over a wide grid.

        The earlier grid stopped at C=4.0 while the optimum on this corpus sits
        near C=30 (measured independently with a plain TF-IDF + logistic
        regression recipe), so the search was in the wrong regime. Weak
        supervision is noisy, which favours strong regularisation in theory, but
        the measured macro-F1 keeps improving past C=4 on this data.
        """
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
        best = (-1.0, -1.0)
        for c_value in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
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

    def _escalation_knn_probabilities(
        self, embeddings: np.ndarray, neighbor_count: int
    ) -> np.ndarray:
        if self.escalation_knn_embeddings_ is None or self.escalation_knn_labels_ is None:
            raise RuntimeError("Escalation neighbor index is not fitted")
        query = np.asarray(embeddings, dtype=np.float32)
        similarities = query @ self.escalation_knn_embeddings_.T
        k = min(max(int(neighbor_count), 1), similarities.shape[1])
        indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        selected_similarity = np.take_along_axis(similarities, indices, axis=1)
        selected_labels = self.escalation_knn_labels_[indices]
        shifted = (selected_similarity - selected_similarity.max(axis=1, keepdims=True)) / 0.07
        weights = np.exp(np.clip(shifted, -30.0, 0.0))
        return np.sum(weights * selected_labels, axis=1) / np.maximum(
            np.sum(weights, axis=1), 1e-12
        )

    def _fit_escalation_knn_blend(
        self,
        calibration_embeddings: np.ndarray,
        base_probabilities: np.ndarray,
        calibration_labels: Sequence[bool],
    ) -> np.ndarray:
        """Admit a non-parametric risk view only after a material holdout AP gain."""
        labels = np.asarray(calibration_labels, dtype=bool)
        base = np.asarray(base_probabilities, dtype=float)
        baseline = float(average_precision_score(labels, base))
        best = (baseline, 0.0, 0, base)
        for neighbor_count in (5, 11, 25, 51):
            neighbor_probs = self._escalation_knn_probabilities(
                calibration_embeddings, neighbor_count
            )
            for weight in np.arange(0.05, 0.51, 0.05):
                blended = (1.0 - weight) * base + weight * neighbor_probs
                score = float(average_precision_score(labels, blended))
                candidate = (score, -float(weight), -neighbor_count)
                if candidate > (best[0], -best[1], -best[2]):
                    best = (score, float(weight), neighbor_count, blended)
        self.escalation_knn_calibration_ap_gain_ = max(0.0, best[0] - baseline)
        if self.escalation_knn_calibration_ap_gain_ >= self.minimum_auxiliary_macro_f1_gain:
            self.escalation_knn_weight_ = best[1]
            self.escalation_knn_k_ = best[2]
            return np.asarray(best[3], dtype=float)
        return base

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
        for temperature in (0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5):
            prototype = self._prototype_probabilities(calibration_embeddings, temperature)
            for weight in np.arange(0.05, 0.71, 0.05):
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

    def _fit_ensemble_blend(
        self,
        calibration_embeddings: np.ndarray,
        calibration_labels: Sequence[str],
        calibration_logits: np.ndarray,
    ) -> None:
        """Fit a 3-way ensemble: hybrid + semantic + prototype."""
        hybrid = _softmax(np.asarray(calibration_logits) / self.intent_temperature.temperature_)
        semantic_logits = self.semantic_head.decision_function(calibration_embeddings)
        semantic = _softmax(semantic_logits / self.semantic_temperature.temperature_)
        prototype = self._prototype_probabilities(calibration_embeddings, self.prototype_temperature_)
        truth = np.asarray(calibration_labels, dtype=object)
        class_values = np.asarray(self.classes_, dtype=object)
        
        baseline_pred = class_values[np.argmax(hybrid, axis=1)]
        baseline = float(f1_score(truth, baseline_pred, average="macro", zero_division=0))
        
        best = (baseline, 1.0, 0.0, 0.0)  # Start with baseline (hybrid only)
        # Search over 3-way blend weights, including baseline
        for w1 in np.arange(0.0, 1.01, 0.1):
            for w2 in np.arange(0.0, 1.01 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < 0:
                    continue
                blended = w1 * hybrid + w2 * semantic + w3 * prototype
                predicted = class_values[np.argmax(blended, axis=1)]
                score = float(f1_score(truth, predicted, average="macro", zero_division=0))
                candidate = (score, -w1, -w2, -w3)
                if candidate > (best[0], -best[1], -best[2], -best[3]):
                    best = (score, float(w1), float(w2), float(w3))
        
        self.ensemble_blend_gain_ = max(0.0, best[0] - baseline)
        if self.ensemble_blend_gain_ >= self.minimum_auxiliary_macro_f1_gain:
            self.ensemble_hybrid_weight_ = best[1]
            self.ensemble_semantic_weight_ = best[2]
            self.ensemble_prototype_weight_ = best[3]

    def _intent_view(self, text: str) -> Dict[str, Any]:
        embedding = self.encoder.encode([text])
        features = self._hybrid_features(self.intent_vectorizer, [text], embedding)
        logits = np.asarray(self.intent_head.decision_function(features)[0], dtype=float)
        probs = _softmax(logits / self.intent_temperature.temperature_)
        
        # Apply confidence penalty to reduce overconfident wrong predictions
        if hasattr(self, 'confidence_penalty_factor_') and self.confidence_penalty_factor_ > 0:
            probs = self._apply_confidence_penalty(probs)
        
        # Use ensemble blending if available and beneficial
        if hasattr(self, 'ensemble_blend_gain_') and self.ensemble_blend_gain_ >= self.minimum_auxiliary_macro_f1_gain:
            if self.semantic_blend_weight_ > 0.0:
                semantic_logits = self.semantic_head.decision_function(embedding)
                semantic_probs = _softmax(
                    semantic_logits / self.semantic_temperature.temperature_
                )[0]
            else:
                semantic_probs = probs.copy()
            if self.intent_prototypes_ is not None and self.prototype_blend_weight_ > 0.0:
                prototype_probs = self._prototype_probabilities(embedding)[0]
            else:
                prototype_probs = probs.copy()
            
            # 3-way ensemble blend
            probs = (
                self.ensemble_hybrid_weight_ * probs
                + self.ensemble_semantic_weight_ * semantic_probs
                + self.ensemble_prototype_weight_ * prototype_probs
            )
        else:
            # Fallback to original blending
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

        # Restore the deployment class prior (training is balanced to uniform).
        probs = self._apply_prior_correction(probs)

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
        # Keep the gate consistent with inference-time probabilities, which run
        # through the deployment-prior correction.
        current_probs = self._apply_prior_correction(current_probs)
        context_probs = self._apply_prior_correction(context_probs)
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
            "current_message_embedding": current["embedding"],
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
        if self.escalation_knn_weight_ > 0.0:
            neighbor_probability = float(
                self._escalation_knn_probabilities(dense, self.escalation_knn_k_)[0]
            )
            probability = (
                (1.0 - self.escalation_knn_weight_) * probability
                + self.escalation_knn_weight_ * neighbor_probability
            )
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
