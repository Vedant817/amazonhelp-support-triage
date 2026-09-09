"""Zero-shot local-LLM baseline used only in explicit local benchmark runs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from src.classifier import INTENT_TAXONOMY
from src.escalator import AUTO_REASON_BY_INTENT, ESCALATE_REASON_BY_INTENT
from src.local_llm import LocalInstructionModel


class ZeroShotLocalBaseline:
    def __init__(self, local_model: Optional[LocalInstructionModel] = None):
        self.local_model = local_model or LocalInstructionModel()

    def predict(self, customer_text: str) -> Dict[str, Any]:
        prompt = f"""Classify and answer this public Amazon support message with no retrieved examples.
Message: {customer_text}
Allowed intents: {', '.join(INTENT_TAXONOMY)}
Escalate true for public PII, account takeover, legal threats, delivery disputes, unresolved refunds,
severe damage, or hardware failure. Otherwise false. Reply must be safe and under 240 characters.
Return only JSON: {{"intent":"...","escalate":true,"reply":"..."}}"""
        raw = self.local_model.generate(
            [
                {"role": "system", "content": "You are a zero-shot customer support baseline."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=150,
        )
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {
                "intent": "INVALID_OUTPUT",
                "should_escalate": False,
                "stated_reason": "INVALID_OUTPUT",
                "reply": raw.strip(),
                "parse_valid": False,
            }
        try:
            payload = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return {
                "intent": "INVALID_OUTPUT",
                "should_escalate": False,
                "stated_reason": "INVALID_OUTPUT",
                "reply": raw.strip(),
                "parse_valid": False,
            }
        intent = str(payload["intent"])
        if intent not in INTENT_TAXONOMY:
            intent = "INVALID_OUTPUT"
        escalate = bool(payload["escalate"])
        return {
            "intent": intent,
            "should_escalate": escalate,
            "stated_reason": (
                (ESCALATE_REASON_BY_INTENT if escalate else AUTO_REASON_BY_INTENT)[intent]
                if intent in INTENT_TAXONOMY else "INVALID_OUTPUT"
            ),
            "reply": str(payload["reply"]).strip(),
            "parse_valid": intent in INTENT_TAXONOMY,
        }
