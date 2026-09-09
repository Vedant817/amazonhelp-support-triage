"""Apply the documented policy audit to the frozen golden set.

Only clear policy contradictions are changed. The script asserts prior values so it
cannot silently relabel a subsequently edited gold file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "golden" / "golden_eval_set.csv"
JSONL_PATH = ROOT / "data" / "golden" / "golden_eval_set.jsonl"

CHANGES = {
    "GOLD-005": "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING",
    "GOLD-008": "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING",
    "GOLD-009": "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING",
    "GOLD-011": "DELIVERY_DISPUTE_MARKED_DELIVERED_BUT_MISSING",
}


def main() -> None:
    frame = pd.read_csv(CSV_PATH)
    for item_id, reason in CHANGES.items():
        mask = frame["id"] == item_id
        if int(mask.sum()) != 1:
            raise RuntimeError(f"Expected exactly one row for {item_id}")
        current_reason = str(frame.loc[mask, "gold_escalate_reason"].iloc[0])
        if current_reason not in {"ROUTINE_TRACKING_IN_TRANSIT", reason}:
            raise RuntimeError(f"Unexpected existing reason for {item_id}: {current_reason}")
        frame.loc[mask, "gold_escalate"] = True
        frame.loc[mask, "gold_escalate_reason"] = reason
        frame.loc[mask, "notes"] = "Policy audit: delivery failure/misdelivery requires human review"
    frame.to_csv(CSV_PATH, index=False)

    records = []
    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["id"] in CHANGES:
                record["gold_escalate"] = True
                record["gold_escalate_reason"] = CHANGES[record["id"]]
                record["notes"] = "Policy audit: delivery failure/misdelivery requires human review"
            records.append(record)
    with JSONL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Audited {len(frame)} cases; changed {len(CHANGES)} documented contradictions.")


if __name__ == "__main__":
    main()
