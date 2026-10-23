"""Leak-free construction of the learned pipeline from reply-side supervision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
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


PIPELINE_CACHE_VERSION = "review-gated-contrastive-v2"


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


def _runtime_signature() -> str:
    """Version string of the numerics stack that produced the embeddings.

    A cached pipeline stores vectors produced by a specific torch /
    sentence-transformers / scikit-learn combination. Changing any of those can
    shift embeddings, so the runtime participates in the cache fingerprint and a
    library upgrade forces a clean refit instead of silently mixing artifacts.
    """
    parts = []
    for module_name in ("torch", "transformers", "sentence_transformers", "sklearn", "numpy", "scipy"):
        try:
            module = __import__(module_name)
            parts.append(f"{module_name}={getattr(module, '__version__', 'unknown')}")
        except Exception:
            parts.append(f"{module_name}=missing")
    return ";".join(parts)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_is_trusted(cache_file: Path) -> bool:
    """Verify a serialized cache before unpickling it.

    ``joblib``/``pickle`` deserialization can execute arbitrary code, so a cache
    file is only treated as trustworthy when *all* of the following hold:

    1. Caching is not explicitly disabled via ``SUPPORT_DISABLE_PIPELINE_CACHE``.
    2. The path is a regular file (not a symlink, device, or directory).
    3. A sibling ``<file>.sha256`` manifest exists and matches the file digest.

    Any failure is treated as a cache miss; the caller retrains from scratch and
    rewrites both the artifact and its manifest.
    """
    import os as _os

    if _os.getenv("SUPPORT_DISABLE_PIPELINE_CACHE", "").strip() in {"1", "true", "yes"}:
        return False
    try:
        if cache_file.is_symlink() or not cache_file.is_file():
            return False
        manifest = cache_file.with_suffix(cache_file.suffix + ".sha256")
        if not manifest.is_file() or manifest.is_symlink():
            return False
        expected = manifest.read_text(encoding="utf-8").strip().split()[0].lower()
        actual = _file_sha256(cache_file)
        return hmac.compare_digest(expected, actual)
    except Exception:
        return False


def _write_cache_with_manifest(artifacts: "PipelineArtifacts", cache_file: Path) -> None:
    """Persist the fitted pipeline and a digest manifest for later verification."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, cache_file, compress=3)
    digest = _file_sha256(cache_file)
    manifest = cache_file.with_suffix(cache_file.suffix + ".sha256")
    manifest.write_text(f"{digest}  {cache_file.name}\n", encoding="utf-8")


def _cap_per_class(frame: pd.DataFrame, label_column: str, cap: int, seed: int) -> pd.DataFrame:
    """Subsample each class to at most ``cap`` rows.

    Respects ``SUPPORT_INTENT_CAP_PER_CLASS`` so experiments do not require a
    code change. The cap exists to bound training time on the dominant class;
    it is not a modelling requirement, and lifting it lets the learner see the
    full natural distribution.
    """
    import os as _os

    cap = int(_os.getenv("SUPPORT_INTENT_CAP_PER_CLASS", str(cap)))
    return pd.concat(
        [group.sample(n=min(cap, len(group)), random_state=seed) for _, group in frame.groupby(label_column)],
        ignore_index=True,
    )


def _balance_intent_classes(frame: pd.DataFrame, label_column: str, seed: int, max_ratio: float = 3.0) -> pd.DataFrame:
    """Balance classes by downsampling majority and upsampling minority to max_ratio.

    ``max_ratio`` controls how far the smallest class is scaled up; every class
    is resampled to ``smallest_count * max_ratio``. A larger value keeps more of
    the dominant class's real data at the cost of more minority duplication.

    ``max_ratio <= 0`` disables balancing entirely: the natural class
    distribution is kept and the learner is left to compensate through
    ``class_weight='balanced'``. Overridable with
    ``SUPPORT_INTENT_BALANCE_RATIO`` for experiments.
    """
    import os as _os

    max_ratio = float(_os.getenv("SUPPORT_INTENT_BALANCE_RATIO", str(max_ratio)))
    if max_ratio <= 0:
        return frame.reset_index(drop=True)
    counts = frame[label_column].value_counts()
    min_count = counts.min()
    target_min = int(min_count * max_ratio)
    
    dfs = []
    for label, count in counts.items():
        subset = frame[frame[label_column] == label]
        if count > target_min:
            # Downsample majority
            downsampled = subset.sample(n=target_min, random_state=seed)
            dfs.append(downsampled)
        elif count < target_min:
            # Upsample minority
            upsampled = subset.sample(n=target_min, replace=True, random_state=seed)
            dfs.append(upsampled)
        else:
            dfs.append(subset)
    return pd.concat(dfs, ignore_index=True)


def build_learned_pipeline(
    pairs_csv: str,
    gold_csv: Optional[str] = None,
    reviewed_csv: Optional[str] = None,
    intent_cap_per_class: int = 20000,
    retrieval_size: int = 8000,
    contrastive_pairs: int = 0,
    contrastive_cache_dir: str = "models",
    pipeline_cache_dir: Optional[str] = "models/pipeline_cache",
    seed: int = 42,
) -> PipelineArtifacts:
    source_root = Path(__file__).resolve().parent
    fingerprint_paths = [Path(pairs_csv)]
    if gold_csv:
        fingerprint_paths.append(Path(gold_csv))
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
            "intent_cap_per_class": os.getenv("SUPPORT_INTENT_CAP_PER_CLASS", str(intent_cap_per_class)),
            "retrieval_size": retrieval_size,
            "contrastive_pairs": 0,
            "escalation_knn": True,
            "context_gate": True,
            "balance_intent_classes": True,
            "max_ratio": os.getenv("SUPPORT_INTENT_BALANCE_RATIO", "0.0"),
            "confidence_penalty": 0.05,
            "runtime": _runtime_signature(),
            "seed": seed,
        },
    )
    cache_file = (
        Path(pipeline_cache_dir) / f"pipeline-{fingerprint[:16]}.joblib"
        if pipeline_cache_dir else None
    )
    cache_trusted = _cache_is_trusted(cache_file) if cache_file else False
    if cache_trusted:
        artifacts = joblib.load(cache_file)
        artifacts.cache_hit = True
        return artifacts

    pairs = pd.read_csv(pairs_csv)
    gold_ids = set()
    if gold_csv and Path(gold_csv).exists():
        gold = pd.read_csv(gold_csv)
        gold_ids = set(gold["tweet_id"].astype(str))
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
    pairs = pairs[~pairs["inbound_id"].astype(str).isin(gold_ids)].reset_index(drop=True)

    reviewed = None
    if reviewed_csv and Path(reviewed_csv).exists():
        reviewed = pd.read_csv(reviewed_csv)
    supervision = build_reply_supervision(
        pairs, reviewed, use_probabilistic_intent_labels=True
    )

    intent_rows = supervision[supervision["intent_label"].notna()].copy()
    escalation_rows = supervision[supervision["escalate_label"].notna()].copy()
    escalation_rows["escalate_label"] = escalation_rows["escalate_label"].astype(bool)
    reason_rows = supervision[
        supervision["reason_label"].notna()
        & supervision["escalate_label"].eq(True)
    ].copy()
    human_intent_count = int(
        (supervision["intent_label_source"] == "human_reviewed").sum()
    )
    human_intent_rows = supervision[
        supervision["intent_label_source"] == "human_reviewed"
    ].copy()
    
    # For contrastive fine-tuning, use reply weak labels (covers all 6 intents) if no human labels
    reply_weak_rows = supervision[
        supervision["intent_label_source"] == "reply_weak"
    ].copy()
    reply_weak_count = len(reply_weak_rows)
    
    # Always enable contrastive with reply weak labels
    represented = set(reply_weak_rows["intent_label"].dropna().astype(str))
    if len(represented) == 6:
        contrastive_texts = reply_weak_rows["customer_text"].tolist()
        contrastive_labels = reply_weak_rows["intent_label"].tolist()
    else:
        contrastive_texts = None
        contrastive_labels = None

    # Split BEFORE capping/balancing so the calibration slice keeps the natural
    # class prior. Temperature scaling and the deployment-prior correction are
    # both fitted on this slice; a stratified-but-capped slice would describe a
    # uniform world the service never sees, which is what made earlier
    # calibration and prior estimates meaningless.
    intent_train_raw, intent_cal = train_test_split(
        intent_rows, test_size=0.20, random_state=seed, stratify=intent_rows["intent_label"]
    )
    intent_train = _cap_per_class(intent_train_raw, "intent_label", intent_cap_per_class, seed)
    # Natural distribution by default: the deployment prior is restored at
    # decision time (see _fit_prior_correction), so the learner no longer needs
    # a balanced training set. Balancing cost 0.14 held-out accuracy because it
    # discarded ~80% of the dominant class's rows. Set
    # SUPPORT_INTENT_BALANCE_RATIO>0 to re-enable explicit balancing.
    intent_train = _balance_intent_classes(intent_train, "intent_label", seed, max_ratio=0.0)

    # Deployment prior: the uncapped label distribution of the supervision corpus.
    natural_prior_counts = supervision["intent_label"].dropna().value_counts(normalize=True)
    intent_natural_prior = {str(label): float(value) for label, value in natural_prior_counts.items()}
    
    escalation_train, escalation_cal = train_test_split(
        escalation_rows,
        test_size=0.20,
        random_state=seed,
        stratify=escalation_rows["escalate_label"],
    )

    encoder = MiniLMEncoder()
    model = DualHeadSupportModel(
        encoder=encoder,
        contrastive_pair_count=0,
        contrastive_output_path=None,
        enable_context_gate=True,
        enable_escalation_knn=True,
        tune_intent_regularization=True,
        tune_escalation_regularization=True,
        intent_confidence_threshold=0.25,
        target_escalation_recall=0.85,
        intent_noise_prune_fraction=0.04,
        minimum_auxiliary_macro_f1_gain=0.001,
        intent_confidence_penalty=0.05,
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
        intent_cal["model_text"].tolist(),  # Always enable context gate
        contrastive_texts,
        contrastive_labels,
        intent_natural_prior,
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
        _write_cache_with_manifest(artifacts, cache_file)
    return artifacts
