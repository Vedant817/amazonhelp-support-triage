"""Leak-free construction of the learned pipeline from reply-side supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.dual_head import DualHeadSupportModel, MiniLMEncoder
from src.retriever import HistoricalResolutionRetriever
from src.supervision import build_reply_supervision


@dataclass
class PipelineArtifacts:
    model: DualHeadSupportModel
    retriever: HistoricalResolutionRetriever
    supervision: pd.DataFrame
    intent_train_size: int
    intent_calibration_size: int
    escalation_train_size: int
    escalation_calibration_size: int
    probabilistic_intent_label_count: int
    reason_train_size: int


def _cap_per_class(frame: pd.DataFrame, label_column: str, cap: int, seed: int) -> pd.DataFrame:
    return pd.concat(
        [group.sample(n=min(cap, len(group)), random_state=seed) for _, group in frame.groupby(label_column)],
        ignore_index=True,
    )


def build_learned_pipeline(
    pairs_csv: str,
    gold_csv: str,
    reviewed_csv: Optional[str] = None,
    intent_cap_per_class: int = 1200,
    retrieval_size: int = 8000,
    contrastive_pairs: int = 0,
    contrastive_cache_dir: str = "models",
    seed: int = 42,
) -> PipelineArtifacts:
    pairs = pd.read_csv(pairs_csv)
    gold = pd.read_csv(gold_csv)
    gold_ids = set(gold["tweet_id"].astype(str))
    pairs = pairs[~pairs["inbound_id"].astype(str).isin(gold_ids)].reset_index(drop=True)

    reviewed = None
    if reviewed_csv and Path(reviewed_csv).exists():
        reviewed = pd.read_csv(reviewed_csv)
    supervision = build_reply_supervision(
        pairs, reviewed, use_probabilistic_intent_labels=True
    )

    intent_rows = supervision[supervision["intent_label"].notna()].copy()
    intent_rows = _cap_per_class(intent_rows, "intent_label", intent_cap_per_class, seed)
    escalation_rows = supervision[supervision["escalate_label"].notna()].copy()
    escalation_rows["escalate_label"] = escalation_rows["escalate_label"].astype(bool)
    reason_rows = supervision[
        supervision["reason_label"].notna()
        & supervision["escalate_label"].eq(True)
    ].copy()

    intent_train, intent_cal = train_test_split(
        intent_rows, test_size=0.20, random_state=seed, stratify=intent_rows["intent_label"]
    )
    escalation_train, escalation_cal = train_test_split(
        escalation_rows,
        test_size=0.20,
        random_state=seed,
        stratify=escalation_rows["escalate_label"],
    )

    encoder = MiniLMEncoder()
    model = DualHeadSupportModel(
        encoder=encoder,
        contrastive_pair_count=contrastive_pairs,
        contrastive_output_path=(
            str(Path(contrastive_cache_dir) / f"minilm-contrastive-{contrastive_pairs}")
            if contrastive_pairs > 0 else None
        ),
    ).fit(
        intent_train["customer_text"].tolist(),
        intent_train["intent_label"].tolist(),
        escalation_train["customer_text"].tolist(),
        escalation_train["escalate_label"].tolist(),
        intent_cal["customer_text"].tolist(),
        intent_cal["intent_label"].tolist(),
        escalation_cal["customer_text"].tolist(),
        escalation_cal["escalate_label"].tolist(),
        reason_rows["customer_text"].tolist(),
        reason_rows["reason_label"].tolist(),
    )

    retrieval_frame = supervision.copy()
    retrieval_frame["intent"] = retrieval_frame["intent_label"].fillna("UNKNOWN")
    if len(retrieval_frame) > retrieval_size:
        retrieval_frame = retrieval_frame.sample(n=retrieval_size, random_state=seed)
    retriever = HistoricalResolutionRetriever(
        top_k=3, rrf_k=20, dense_encoder=encoder
    ).build_index(retrieval_frame)

    return PipelineArtifacts(
        model=model,
        retriever=retriever,
        supervision=supervision,
        intent_train_size=len(intent_train),
        intent_calibration_size=len(intent_cal),
        escalation_train_size=len(escalation_train),
        escalation_calibration_size=len(escalation_cal),
        probabilistic_intent_label_count=int(
            (supervision["intent_label_source"] == "probabilistic_weak_label_model").sum()
        ),
        reason_train_size=len(reason_rows),
    )
