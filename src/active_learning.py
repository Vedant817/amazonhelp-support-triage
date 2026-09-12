"""Uncertainty-and-diversity sampling for human policy annotation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Entropy in [0, 1] for a matrix of categorical probabilities."""
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape (examples, classes >= 2)")
    return -np.sum(probs * np.log(probs), axis=1) / np.log(probs.shape[1])


def select_active_review_batch(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    intent_probabilities: np.ndarray,
    intent_classes: Sequence[str],
    escalation_probabilities: Sequence[float],
    batch_size: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Select uncertain, suspicious, routing-sensitive, and diverse examples.

    Inputs are predictions made without the frozen evaluation set. Existing weak
    labels are used only to prioritize review; they are copied into the output as
    context and never treated as reviewed labels.
    """
    if len(frame) == 0:
        raise ValueError("active review selection requires candidate examples")
    dense = np.asarray(embeddings, dtype=np.float32)
    intent_probs = np.asarray(intent_probabilities, dtype=float)
    escalation_probs = np.asarray(escalation_probabilities, dtype=float)
    if dense.shape[0] != len(frame) or intent_probs.shape[0] != len(frame):
        raise ValueError("candidate frame, embeddings, and predictions must align")
    if escalation_probs.shape != (len(frame),):
        raise ValueError("escalation probabilities must be one aligned vector")

    classes = np.asarray(list(intent_classes), dtype=object)
    predicted_indices = np.argmax(intent_probs, axis=1)
    predicted_intents = classes[predicted_indices]
    uncertainty = normalized_entropy(intent_probs)

    weak_intents = frame["intent_label"].astype(object).to_numpy()
    class_index = {label: idx for idx, label in enumerate(classes)}
    suspicion = np.full(len(frame), 0.5, dtype=float)
    intent_disagreement = np.zeros(len(frame), dtype=float)
    for index, weak_label in enumerate(weak_intents):
        if pd.isna(weak_label) or weak_label not in class_index:
            continue
        suspicion[index] = 1.0 - intent_probs[index, class_index[weak_label]]
        intent_disagreement[index] = float(weak_label != predicted_intents[index])

    route_margin = 1.0 - 2.0 * np.abs(escalation_probs - 0.5)
    weak_routes = frame["escalate_label"].astype(object).to_numpy()
    route_disagreement = np.zeros(len(frame), dtype=float)
    for index, weak_route in enumerate(weak_routes):
        if pd.isna(weak_route):
            continue
        route_disagreement[index] = float(bool(weak_route) != (escalation_probs[index] >= 0.5))
    disagreement = np.maximum(intent_disagreement, route_disagreement)
    priority = 0.35 * uncertainty + 0.25 * suspicion + 0.20 * route_margin + 0.20 * disagreement

    cluster_count = min(max(batch_size // 3, 2), len(frame))
    clusters = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        batch_size=min(1024, len(frame)),
        n_init=3,
    ).fit_predict(dense)

    candidates = frame.copy().reset_index(drop=True)
    candidates["provisional_intent"] = predicted_intents
    candidates["intent_uncertainty"] = uncertainty
    candidates["weak_label_suspicion"] = suspicion
    candidates["route_margin"] = route_margin
    candidates["model_weak_disagreement"] = disagreement
    candidates["active_learning_score"] = priority
    candidates["diversity_cluster"] = clusters

    # Equal predicted-class quotas prevent the majority tracking class from
    # consuming the annotation budget. A second pass fills unused quota.
    quota = int(np.ceil(batch_size / len(classes)))
    selected_indices: list[int] = []
    used_clusters: set[int] = set()
    for label in classes:
        group = candidates[candidates["provisional_intent"] == label].sort_values(
            "active_learning_score", ascending=False
        )
        diverse = group[~group["diversity_cluster"].isin(used_clusters)]
        chosen = pd.concat([diverse, group]).loc[lambda x: ~x.index.duplicated()].head(quota)
        selected_indices.extend(chosen.index.tolist())
        used_clusters.update(chosen["diversity_cluster"].astype(int).tolist())

    if len(selected_indices) < batch_size:
        remaining = candidates.drop(index=set(selected_indices)).sort_values(
            "active_learning_score", ascending=False
        )
        selected_indices.extend(remaining.head(batch_size - len(selected_indices)).index.tolist())

    selected = candidates.loc[selected_indices[:batch_size]].copy()
    return selected.sort_values(
        ["provisional_intent", "active_learning_score"], ascending=[True, False]
    ).reset_index(drop=True)

