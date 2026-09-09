"""Load the pinned local generator and exercise one no-key rewrite."""

from __future__ import annotations

import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard
from src.supervision import strip_agent_pii


def main() -> int:
    exemplar = {
        "reply_text": (
            "I'm sorry your order hasn't arrived! What is the most recent tracking "
            "scan from the carrier showing? You can check here: "
            "https://t.co/PyACxvY8Qo ^BN"
        ),
        "resolution_act": "ASK_TRACKING_STATUS",
    }
    generated = ProductionGroundedGenerator(mode="local").generate(
        customer_query="My package is late. What should I do?",
        intent="ORDER_TRACKING_AND_DELAY",
        exemplars=[exemplar],
        escalate_decision={"should_escalate": False},
    )
    guarded_result = ProductionPolicyGuard().verify_and_sanitize(
        generated,
        customer_text="My package is late. What should I do?",
    )
    guarded = guarded_result.sanitized_reply
    historical = strip_agent_pii(exemplar["reply_text"])
    similarity = SequenceMatcher(None, generated.lower(), historical.lower()).ratio()

    assert generated.strip(), "local model returned an empty reply"
    assert len(guarded) <= 280, "guarded reply exceeds the public limit"
    assert "t.co" not in guarded, "unapproved historical URL survived guardrails"
    print("local_qwen_rewrite=PASS")
    print(f"similarity_to_exemplar={similarity:.3f}")
    print(f"guarded_output={guarded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
