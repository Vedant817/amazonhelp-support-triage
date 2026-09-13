"""Run a small, real local-model qualification without any hosted API key."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard
from src.local_llm import LocalInstructionModel, QWEN_MODEL_ID, QWEN_REVISION


CASES = (
    ("Tracking stopped updating three days ago. Where is it?", "ORDER_TRACKING_AND_DELAY", "ASK_TRACKING_STATUS", False),
    ("The courier says they never received my parcel.", "ORDER_TRACKING_AND_DELAY", "ASK_CARRIER", False),
    ("It says delivered, but the box is not at my home.", "ORDER_TRACKING_AND_DELAY", "DM_WITH_ORDER_ID", True),
    ("The expected date moved again.", "ORDER_TRACKING_AND_DELAY", "LINK_ORDERS", False),
    ("These shoes are the wrong size and I need to send them back.", "REFUND_AND_RETURN", "LINK_RETURNS", False),
    ("My Kindle freezes every time it starts.", "PRODUCT_AND_TECHNICAL", "TROUBLESHOOT_DEVICE", False),
    ("The promised delivery window has passed.", "ORDER_TRACKING_AND_DELAY", "APOLOGIZE_WAIT", False),
    ("I have feedback about the packaging.", "FEEDBACK_AND_GENERAL", "GENERAL_SUPPORT", False),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/local_generator_qualification.json")
    args = parser.parse_args()
    generator = ProductionGroundedGenerator(
        mode="local", local_model=LocalInstructionModel()
    )
    guard = ProductionPolicyGuard()
    records = []
    started = time.perf_counter()
    for index, (customer, intent, act, escalate) in enumerate(CASES, start=1):
        raw, generation = generator.generate_with_metadata(
            customer,
            intent,
            [{
                "resolution_act": act,
                "reply_text": "Please use the appropriate safe next step for this request.",
                "rrf_score": 1.0,
            }],
            {"should_escalate": escalate, "stated_reason": ""},
        )
        checked = guard.verify_and_sanitize(raw, customer, escalate)
        records.append({
            "case": index,
            "required_act": act,
            "actual_backend": generation["backend"],
            "fallback_reason": generation["fallback_reason"],
            "act_realized_after_fallback": generator._realizes_act(
                checked.sanitized_reply, act
            ),
            "guardrail_valid": checked.is_valid,
            "character_count": len(checked.sanitized_reply),
            "reply": checked.sanitized_reply,
        })
    report = {
        "model": QWEN_MODEL_ID,
        "revision": QWEN_REVISION,
        "hosted_api_required": False,
        "case_count": len(records),
        "local_generation_count": sum(
            row["actual_backend"] == "local_qwen" for row in records
        ),
        "safe_fallback_count": sum(
            row["actual_backend"] != "local_qwen" for row in records
        ),
        "all_required_acts_realized": all(
            row["act_realized_after_fallback"] for row in records
        ),
        "all_guardrails_valid": all(row["guardrail_valid"] for row in records),
        "all_within_280_characters": all(
            row["character_count"] <= 280 for row in records
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "cases": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all((
        report["all_required_acts_realized"],
        report["all_guardrails_valid"],
        report["all_within_280_characters"],
    )) else 1


if __name__ == "__main__":
    raise SystemExit(main())
