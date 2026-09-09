"""Local model judge and same-reply human agreement metrics.

There is intentionally no keyword-based fallback. If local model inference is not
requested or cannot run, judge evidence is unavailable and must not be reported.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.local_llm import LocalInstructionModel


RUBRIC_FIELDS = ("groundedness", "empathy", "actionability", "safety")


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

Return only JSON with integer keys groundedness, empathy, actionability, safety and a short rationale."""
        raw = self.local_model.generate(
            [
                {"role": "system", "content": "You are a strict independent evaluator. Do not reward stock phrases."},
                {"role": "user", "content": rubric},
            ],
            max_new_tokens=150,
        )
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Local judge did not return JSON: {raw[:160]}")
        payload = json.loads(match.group(0))
        scores = {}
        for field in RUBRIC_FIELDS:
            value = int(payload[field])
            if value < 1 or value > 5:
                raise ValueError(f"Judge score {field}={value} is outside 1..5")
            scores[field] = value
        scores["overall_score"] = round(float(np.mean(list(scores.values()))), 2)
        scores["rationale"] = str(payload.get("rationale", ""))[:500]
        scores["judge_backend"] = self.local_model.model_name
        scores["judge_revision"] = self.local_model.revision
        return scores

    @staticmethod
    def compute_human_agreement(
        human_scores: Sequence[float], judge_scores: Sequence[float]
    ) -> Dict[str, Any]:
        """Agreement for human and judge scores attached to the exact same reply IDs."""
        h = np.asarray(human_scores, dtype=float)
        j = np.asarray(judge_scores, dtype=float)
        if len(h) == 0 or len(h) != len(j):
            raise ValueError("Human and judge score arrays must be non-empty and aligned")
        rho = float(spearmanr(h, j).statistic) if np.std(h) and np.std(j) else 0.0
        h_rounded = np.clip(np.rint(h), 1, 5).astype(int)
        j_rounded = np.clip(np.rint(j), 1, 5).astype(int)
        kappa = float(cohen_kappa_score(h_rounded, j_rounded, weights="quadratic"))
        return {
            "spearman_rho": round(rho, 4),
            "quadratic_weighted_kappa": round(kappa, 4),
            "mae": round(float(np.mean(np.abs(h - j))), 4),
            "exact_agreement_pct": round(float(np.mean(h_rounded == j_rounded) * 100.0), 2),
            "adjacent_agreement_pct": round(float(np.mean(np.abs(h_rounded - j_rounded) <= 1) * 100.0), 2),
            "sample_size": int(len(h)),
            "same_reply_required": True,
        }
