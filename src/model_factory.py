"""Leak-free construction of the learned pipeline from reply-side supervision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib
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
    context_by_id: dict[str, str]
    context_coverage: float
    cache_hit: bool = False


PIPELINE_CACHE_VERSION = "context-probabilistic-v1"


def _pipeline_fingerprint(paths: list[Path], parameters: dict) -> str:
    digest = hashlib.sha256()
    digest.update(PIPELINE_CACHE_VERSION.encode("utf-8"))
    digest.update(json.dumps(parameters, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


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
    pipeline_cache_dir: Optional[str] = "models/pipeline_cache",
    seed: int = 42,
) -> PipelineArtifacts:
    source_root = Path(__file__).resolve().parent
    fingerprint_paths = [Path(pairs_csv), Path(gold_csv)]
    if reviewed_csv and Path(reviewed_csv).exists():
        fingerprint_paths.append(Path(reviewed_csv))
    fingerprint_paths.extend([
        source_root / name for name in (
            "model_factory.py", "dual_head.py", "supervision.py", "weak_label_model.py", "retriever.py"
        )
    ])
    fingerprint = _pipeline_fingerprint(
        fingerprint_paths,
        {
            "intent_cap_per_class": intent_cap_per_class,
            "retrieval_size": retrieval_size,
            "contrastive_pairs": contrastive_pairs,
            "seed": seed,
        },
    )
    cache_file = (
        Path(pipeline_cache_dir) / f"pipeline-{fingerprint[:16]}.joblib"
        if pipeline_cache_dir else None
    )
    if cache_file and cache_file.exists():
        artifacts = joblib.load(cache_file)
        artifacts.cache_hit = True
        return artifacts

    pairs = pd.read_csv(pairs_csv)
    gold = pd.read_csv(gold_csv)
    if "conversation_context" not in pairs.columns or "prior_turn_count" not in pairs.columns:
        raise ValueError(
            "Context-enriched pairs are required. Run scripts/build_conversation_context.py first."
        )
    pairs["conversation_context"] = pairs["conversation_context"].fillna("").astype(str)
    pairs["model_text"] = pairs.apply(
        lambda row: (
            f"{row['conversation_context']}\nCURRENT CUSTOMER: {row['customer_text']}"
            if row["conversation_context"] else str(row["customer_text"])
        ),
        axis=1,
    )
    context_by_id = dict(zip(pairs["inbound_id"].astype(str), pairs["model_text"]))
    context_coverage = float((pairs["prior_turn_count"] > 0).mean())
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
    human_intent_count = int(
        (supervision["intent_label_source"] == "human_reviewed").sum()
    )

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
        enable_context_gate=human_intent_count >= 60,
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
        intent_cal["model_text"].tolist() if human_intent_count >= 60 else None,
    )

    retrieval_frame = supervision.copy()
    retrieval_frame["intent"] = retrieval_frame["intent_label"].fillna("UNKNOWN")
    if len(retrieval_frame) > retrieval_size:
        retrieval_frame = retrieval_frame.sample(n=retrieval_size, random_state=seed)
    retriever = HistoricalResolutionRetriever(
        top_k=3, rrf_k=20, dense_encoder=encoder
    ).build_index(retrieval_frame)

    artifacts = PipelineArtifacts(
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
        context_by_id=context_by_id,
        context_coverage=context_coverage,
    )
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, cache_file, compress=3)
    return artifacts
