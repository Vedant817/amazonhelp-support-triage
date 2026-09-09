"""Create a stratified 400-row sheet for independent human training-label review."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supervision import build_reply_supervision


def main() -> None:
    pairs = pd.read_csv(ROOT / "data" / "processed" / "amazonhelp_english_pairs.csv")
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_eval_set.csv")
    pairs = pairs[~pairs["inbound_id"].astype(str).isin(set(gold["tweet_id"].astype(str)))]
    labeled = build_reply_supervision(pairs)
    eligible = labeled[labeled["intent_label"].notna()].copy()
    sample = pd.concat(
        [group.sample(n=min(67, len(group)), random_state=42) for _, group in eligible.groupby("intent_label")],
        ignore_index=True,
    ).head(400)
    sample["reviewed_intent"] = ""
    sample["reviewed_escalate"] = ""
    sample["review_notes"] = ""
    columns = [
        "inbound_id", "customer_text", "reply_text", "resolution_act",
        "intent_label", "escalate_label", "reviewed_intent", "reviewed_escalate", "review_notes",
    ]
    output = ROOT / "data" / "training" / "labels_for_human_review.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(output, index=False)
    print(f"Wrote {len(sample)} leak-free rows to {output}")


if __name__ == "__main__":
    main()
