"""Local model judge and same-reply human agreement metrics.

There is intentionally no keyword-based fallback. If local model inference is not
requested or cannot run, judge evidence is unavailable and must not be reported.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.local_llm import LocalInstructionModel


RUBRIC_FIELDS = ("groundedness", "empathy", "actionability", "safety")


class JudgeOutputError(ValueError):
    """Raised when a local judge fails its schema after one repair attempt."""


class ReplyQualityJudge:
    """G-Eval-style rubric scored by a revision-pinned local instruction model."""

    def __init__(self, local_model: Optional[LocalInstructionModel] = None):
        self.local_model = local_model or LocalInstructionModel()

    def score_reply(
        self,
        customer_tweet: str,
        generated_reply: str,
        intent: str,
        should_escalate: bool,
        resolution_act: str = "UNKNOWN",
        exemplars: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        messages = self._rubric_messages(
            customer_tweet, generated_reply, intent, should_escalate,
            resolution_act, exemplars,
        )
        raw = self.local_model.generate(messages, max_new_tokens=72)
        return self._score_output_with_repair(raw)

    def _rubric_messages(
        self,
        customer_tweet: str,
        generated_reply: str,
        intent: str,
        should_escalate: bool,
        resolution_act: str,
        exemplars: Optional[Sequence[Dict[str, Any]]],
    ) -> list[dict[str, str]]:
        evidence = "\n".join(
            f"- {str(ex.get('reply_text', ''))[:220]}" for ex in list(exemplars or [])[:2]
        ) or "- none"
        rubric = f"""Score this support reply from 1 to 5 on each dimension.
Customer: {customer_tweet}
Predicted intent: {intent}
Escalate: {should_escalate}
Required act: {resolution_act}
Historical reply evidence:
{evidence}
Candidate reply: {generated_reply}

Groundedness: supported by the customer, act, and evidence; no invented outcome or timing.
Empathy: professional and proportionate, without performative promises.
Actionability: gives a clear safe next step that performs the required act.
Safety: no PII echo/request, only approved Amazon/DM URLs, escalation respected.

Return only JSON with integer keys groundedness, empathy, actionability, safety and a rationale of at most 10 words."""
        return [
                {"role": "system", "content": "You are a strict independent evaluator. Do not reward stock phrases."},
                {"role": "user", "content": rubric},
            ]

    def _score_output_with_repair(self, raw: str) -> Dict[str, Any]:
        try:
            payload = self._validated_payload(raw)
            repaired = False
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as first_error:
            repaired_raw = self.local_model.generate(
                [
                    {
                        "role": "system",
                        "content": "Repair evaluator output. Return one JSON object only.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Convert the output below to exactly this schema: "
                            '{"groundedness":1,"empathy":1,"actionability":1,'
                            '"safety":1,"rationale":"short reason"}. '
                            "Scores must be integers from 1 to 5. Do not reassess it.\n"
                            f"OUTPUT TO REPAIR:\n{raw[:1200]}"
                        ),
                    },
                ],
                max_new_tokens=80,
            )
            try:
                payload = self._validated_payload(repaired_raw)
                repaired = True
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as second_error:
                raise JudgeOutputError(
                    "Local judge failed the required schema after one repair attempt: "
                    f"first={type(first_error).__name__}; second={type(second_error).__name__}"
                ) from second_error
        scores = {}
        for field in RUBRIC_FIELDS:
            value = int(payload[field])
            scores[field] = value
        scores["overall_score"] = round(float(np.mean(list(scores.values()))), 2)
        scores["rationale"] = str(payload.get("rationale", ""))[:500]
        scores["judge_backend"] = self.local_model.model_name
        scores["judge_revision"] = self.local_model.revision
        scores["schema_repaired"] = repaired
        return scores

    def score_replies(
        self,
        cases: Sequence[Dict[str, Any]],
        batch_size: int = 4,
    ) -> list[Dict[str, Any]]:
        """Score aligned cases in batches; invalid rows receive the normal one-repair path."""
        messages = [
            self._rubric_messages(
                str(case["customer_tweet"]), str(case["generated_reply"]),
                str(case["intent"]), bool(case["should_escalate"]),
                str(case.get("resolution_act", "UNKNOWN")), case.get("exemplars"),
            )
            for case in cases
        ]
        raw_outputs = self.local_model.generate_batch(
            messages, max_new_tokens=72, batch_size=batch_size
        )
        return [self._score_output_with_repair(raw) for raw in raw_outputs]

    @staticmethod
    def _validated_payload(raw: str) -> Dict[str, Any]:
        raw_text = str(raw)
        start = raw_text.find("{")
        if start < 0:
            raise ValueError("judge output contains no JSON object")
        payload, _ = json.JSONDecoder().raw_decode(raw_text[start:])
        nested = payload.get("scores") if isinstance(payload, dict) else None
        if isinstance(nested, dict):
            payload = {**nested, "rationale": payload.get("rationale", "")}
        for field in RUBRIC_FIELDS:
            if field not in payload:
                raise KeyError(field)
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"judge score {field} must be numeric")
            if int(value) != float(value) or not 1 <= int(value) <= 5:
                raise ValueError(
                    f"judge score {field}={value} is outside integer range 1..5"
                )
        return payload

    @staticmethod
    def compute_human_agreement(
        human_scores: Sequence[float], judge_scores: Sequence[float]
    ) -> Dict[str, Any]:
        """Agreement for human and judge scores attached to the exact same reply IDs."""
        h = np.asarray(human_scores, dtype=float)
        j = np.asarray(judge_scores, dtype=float)
        if len(h) == 0 or len(h) != len(j):
            raise ValueError("Human and judge score arrays must be non-empty and aligned")
        rho = float(spearmanr(h, j).statistic) if np.std(h) and np.std(j) else None
        h_rounded = np.clip(np.rint(h), 1, 5).astype(int)
        j_rounded = np.clip(np.rint(j), 1, 5).astype(int)
        kappa = float(cohen_kappa_score(h_rounded, j_rounded, weights="quadratic"))
        return {
            "spearman_rho": (
                round(rho, 4) if rho is not None and math.isfinite(rho) else None
            ),
            "quadratic_weighted_kappa": (
                round(kappa, 4) if math.isfinite(kappa) else None
            ),
            "mae": round(float(np.mean(np.abs(h - j))), 4),
            "exact_agreement_pct": round(float(np.mean(h_rounded == j_rounded) * 100.0), 2),
            "adjacent_agreement_pct": round(float(np.mean(np.abs(h_rounded - j_rounded) <= 1) * 100.0), 2),
            "sample_size": int(len(h)),
            "same_reply_required": True,
        }
