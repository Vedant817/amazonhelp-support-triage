"""Evaluate retrieval on replies held out before index construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dual_head import MiniLMEncoder
from src.retrieval_evaluation import ranked_act_metrics
from src.retriever import HistoricalResolutionRetriever
from src.supervision import build_reply_supervision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--index-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", default="results/retrieval_evaluation.json")
    args = parser.parse_args()

    pairs = pd.read_csv(ROOT / "data/processed/amazonhelp_context_pairs.csv")
    gold = pd.read_csv(ROOT / "data/golden/golden_eval_set.csv")
    pairs = pairs[~pairs["inbound_id"].astype(str).isin(set(gold["tweet_id"].astype(str)))]
    labeled = build_reply_supervision(pairs)
    labeled = labeled[
        labeled["resolution_act"].notna() & labeled["customer_text"].notna()
    ].copy()
    train, held_out = train_test_split(
        labeled,
        test_size=min(max(args.queries, 1), len(labeled) - 1),
        random_state=args.seed,
        stratify=labeled["resolution_act"],
    )
    if len(train) > args.index_size:
        train = train.sample(args.index_size, random_state=args.seed)
    held_out = held_out.head(args.queries)
    encoder = MiniLMEncoder()
    retriever = HistoricalResolutionRetriever(
        top_k=3, rrf_k=20, dense_encoder=encoder
    ).build_index(train.assign(intent=train["intent_label"].fillna("UNKNOWN")))
    rankings = [
        retriever.retrieve_similar(
            str(row["customer_text"]),
            intent_filter=(str(row["intent_label"]) if pd.notna(row["intent_label"]) else None),
            top_k=12,
        )
        for _, row in held_out.iterrows()
    ]
    metrics = {
        "top_3": ranked_act_metrics(
            held_out["resolution_act"].tolist(), rankings, k=3
        ),
        "candidate_pool_12": ranked_act_metrics(
            held_out["resolution_act"].tolist(), rankings, k=12
        ),
    }
    majority_act = train["resolution_act"].mode().iloc[0]
    metrics.update({
        "target_source": "resolution act inferred from the held-out historical AmazonHelp reply",
        "held_out_before_index_build": True,
        "gold_ids_excluded": True,
        "intent_filter": "held-out reply-side weak intent when available; diagnostic upper bound",
        "index_size": len(train),
        "majority_act": majority_act,
        "majority_recall_at_3": round(float((held_out["resolution_act"] == majority_act).mean()), 4),
        "model": encoder.model_name,
        "revision": encoder.revision,
    })
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
