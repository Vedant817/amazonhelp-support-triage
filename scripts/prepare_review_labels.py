"""Create a leak-free uncertainty-and-diversity human review batch."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.active_learning import select_active_review_batch
from src.dual_head import MiniLMEncoder
from src.supervision import build_reply_supervision


def _out_of_fold_probabilities(
    embeddings: np.ndarray,
    labels: pd.Series,
    all_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    known = labels.notna().to_numpy()
    y = labels.loc[known].to_numpy()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, embeddings[known], y, cv=folds, method="predict_proba")
    model.fit(embeddings[known], y)
    probabilities = model.predict_proba(all_embeddings)
    probabilities[known] = oof
    return probabilities, model.classes_


def _escalation_probabilities(
    embeddings: np.ndarray,
    labels: pd.Series,
) -> np.ndarray:
    known = labels.notna().to_numpy()
    y = labels.loc[known].astype(bool).to_numpy()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, embeddings[known], y, cv=folds, method="decision_function")
    model.fit(embeddings[known], y)
    logits = model.decision_function(embeddings)
    logits[known] = oof
    return expit(logits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--force", action="store_true", help="Replace an existing unreviewed sheet")
    args = parser.parse_args()

    pairs = pd.read_csv(ROOT / "data" / "processed" / "amazonhelp_english_pairs.csv")
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_eval_set.csv")
    pairs = pairs[~pairs["inbound_id"].astype(str).isin(set(gold["tweet_id"].astype(str)))]
    labeled = build_reply_supervision(pairs)
    output = ROOT / "data" / "training" / "labels_for_human_review.csv"
    if output.exists():
        existing = pd.read_csv(output)
        reviewed_columns = [column for column in existing if column.startswith("reviewed_")]
        completed = int(existing[reviewed_columns].notna().any(axis=1).sum()) if reviewed_columns else 0
        if completed:
            raise RuntimeError(f"Refusing to replace {completed} completed reviews in {output}")
        if not args.force:
            raise RuntimeError(f"{output} already exists; pass --force to replace the unreviewed sheet")

    encoder = MiniLMEncoder()
    embeddings = encoder.encode(labeled["customer_text"].astype(str).tolist())
    intent_probs, intent_classes = _out_of_fold_probabilities(
        embeddings, labeled["intent_label"], embeddings
    )
    escalation_probs = _escalation_probabilities(embeddings, labeled["escalate_label"])
    sample = select_active_review_batch(
        labeled,
        embeddings,
        intent_probs,
        intent_classes,
        escalation_probs,
        batch_size=args.batch_size,
    )
    sample["reviewed_intent"] = ""
    sample["reviewed_escalate"] = ""
    sample["reviewed_reason_code"] = ""
    sample["reviewed_resolution_act"] = ""
    sample["ambiguous"] = ""
    sample["review_notes"] = ""
    columns = [
        "inbound_id", "customer_text", "reply_text", "resolution_act",
        "intent_label", "escalate_label", "provisional_intent", "intent_uncertainty",
        "weak_label_suspicion", "route_margin", "model_weak_disagreement",
        "active_learning_score", "diversity_cluster", "reviewed_intent",
        "reviewed_escalate", "reviewed_reason_code", "reviewed_resolution_act",
        "ambiguous", "review_notes",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(output, index=False)
    print(f"Wrote {len(sample)} leak-free active-learning rows to {output}")


if __name__ == "__main__":
    main()
