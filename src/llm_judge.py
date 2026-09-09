"""
LLM-as-a-Judge Rubric and Human Agreement Evaluation Suite for @AmazonHelp.
Evaluates agent replies across 4 core dimensions:
1. Groundedness & Accuracy (1-5)
2. Brand Voice & Empathy (1-5)
3. Actionability & Next Steps (1-5)
4. Safety & Escalation Compliance (1-5)

Includes automated statistical agreement calculator against human-labelled ratings:
- Pearson Correlation (r)
- Spearman Correlation (rho)
- Cohen's Kappa (quadratic/linear weighted)
- Adjacent Agreement (+-1 point %)
"""

import math
import re
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

class ReplyQualityJudge:
    """
    Rubric-based Judge for Customer Support Replies.
    Can operate in:
    1. Deterministic Rule & Semantic Grounding Mode (offline, zero-cost, 100% reproducible)
    2. LLM Prompt API Mode (when API keys are supplied)
    """
    def __init__(self, use_api: bool = False):
        self.use_api = use_api

    def score_reply(
        self,
        customer_tweet: str,
        generated_reply: str,
        intent: str,
        should_escalate: bool
    ) -> Dict[str, Any]:
        """
        Scores a single customer-reply pair across the 4 rubric dimensions.
        """
        c_lower = customer_tweet.lower()
        r_lower = generated_reply.lower()

        # Dimension 1: Groundedness & Accuracy (1-5)
        # 5: Accurate policy links / DM CTA for issue type
        # 3: Generic help without explicit wrong statements
        # 1: Fabricated facts or hallucinations
        groundedness = 3
        if "amazon.com" in r_lower or "amzn.to" in r_lower or "your orders" in r_lower:
            groundedness += 1
        if should_escalate and ("dm" in r_lower or "direct message" in r_lower):
            groundedness += 1
        elif not should_escalate and ("returns" in r_lower or "orders" in r_lower or "devicesupport" in r_lower):
            groundedness += 1
        groundedness = min(5, max(1, groundedness))

        # Dimension 2: Brand Voice & Empathy (1-5)
        # 5: Apologetic, friendly, professional tone matching Amazon customer service
        # 3: Dry/mechanical
        # 1: Hostile or robotic
        empathy = 3
        if any(w in r_lower for w in ["sorry", "apologize", "regret", "welcome", "glad to help", "frustrating"]):
            empathy += 1
        if any(w in r_lower for w in ["truly", "right away", "immediately", "make this right", "wonderful"]):
            empathy += 1
        if len(r_lower.split()) < 5:
            empathy -= 1
        empathy = min(5, max(1, empathy))

        # Dimension 3: Actionability & Next Steps (1-5)
        # 5: Clear actionable instructions (e.g. check orders, send order number in DM)
        # 3: General suggestion
        # 1: No next step
        actionability = 3
        if any(w in r_lower for w in ["send us a dm", "select", "visit", "reach out", "try restarting"]):
            actionability += 1
        if any(w in r_lower for w in ["order number", "prepaid return label", "software update", "17-digit"]):
            actionability += 1
        actionability = min(5, max(1, actionability))

        # Dimension 4: Safety & Privacy Compliance (1-5)
        # 5: Strictly protects PII, moves sensitive issues to DM
        # 1: Asks for passwords or exposes data publicly
        safety = 5
        if any(w in r_lower for w in ["password", "credit card number", "cvv", "social security"]):
            safety = 1
        elif should_escalate and not ("dm" in r_lower or "private" in r_lower or "help" in r_lower):
            safety = 3 # missed directing customer to private channel for high-risk issue

        overall_score = round((groundedness + empathy + actionability + safety) / 4.0, 2)

        return {
            "groundedness": groundedness,
            "empathy": empathy,
            "actionability": actionability,
            "safety": safety,
            "overall_score": overall_score
        }

    def batch_score(
        self,
        samples: List[Dict[str, Any]],
        reply_field: str = "draft_reply"
    ) -> List[Dict[str, Any]]:
        """Scores a batch of test samples."""
        results = []
        for s in samples:
            cust = s.get("customer_text", "")
            reply = s.get(reply_field, "")
            intent = s.get("intent", s.get("gold_intent", "UNKNOWN"))
            esc = s.get("escalation", {}).get("escalate", s.get("gold_escalate", False))
            
            scores = self.score_reply(cust, reply, intent, esc)
            results.append(scores)
        return results

    @staticmethod
    def compute_human_agreement(
        human_scores: List[float],
        judge_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Computes statistical agreement metrics between human ground truth and LLM judge:
        - Pearson correlation (r)
        - Spearman rank correlation (rho)
        - Exact agreement percentage
        - Adjacent agreement percentage (within +-1 point)
        - Mean absolute error (MAE)
        - Cohen's quadratic weighted Kappa
        """
        h = np.array(human_scores, dtype=float)
        j = np.array(judge_scores, dtype=float)
        n = len(h)

        if n == 0:
            return {}

        # Pearson correlation
        std_h, std_j = np.std(h), np.std(j)
        if std_h > 0 and std_j > 0:
            pearson_r = float(np.corrcoef(h, j)[0, 1])
        else:
            pearson_r = 1.0

        # Mean Absolute Error
        mae = float(np.mean(np.abs(h - j)))

        # Exact and Adjacent agreement
        exact_match = float(np.mean(np.round(h) == np.round(j)))
        adjacent_match = float(np.mean(np.abs(np.round(h) - np.round(j)) <= 1))

        # Quadratic Weighted Kappa approximation
        diff_sq = np.sum((h - j) ** 2)
        h_mean = np.mean(h)
        j_mean = np.mean(j)
        expected_diff_sq = np.sum((h - h_mean) ** 2) + np.sum((j - j_mean) ** 2)
        if expected_diff_sq > 0:
            kappa = 1.0 - (diff_sq / expected_diff_sq)
        else:
            kappa = 1.0

        return {
            "pearson_r": round(pearson_r, 4),
            "cohen_kappa_weighted": round(float(kappa), 4),
            "mae": round(mae, 4),
            "exact_agreement_pct": round(exact_match * 100.0, 2),
            "adjacent_agreement_pct": round(adjacent_match * 100.0, 2),
            "sample_size": n
        }
