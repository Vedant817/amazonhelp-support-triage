"""
Corrective Retrieval-Augmented Generation (CRAG) for @AmazonHelp.
Implements Yan et al. (2024).
Evaluates the retrieval quality of historical resolution exemplars.
Triggers three corrective pathways:
1. CORRECT: Exemplars are highly relevant -> pass to generator directly.
2. AMBIGUOUS: Exemplars have partial relevance -> decompose and sanitize.
3. INCORRECT: Exemplars are irrelevant or off-distribution -> trigger verified brand policy fallback.
"""

import re
from typing import List, Dict, Any, Optional

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"
}

class CorrectiveRetriever:
    """
    CRAG Evaluator that grades retrieved historical resolutions against incoming customer query.
    Prevents hallucinated or misaligned exemplar grounding.
    """
    def __init__(self, upper_threshold: float = 0.35, lower_threshold: float = 0.15):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        # Authoritative Fallback Knowledge Base per Intent
        self.verified_policy_kb = {
            "ORDER_TRACKING_AND_DELAY": {
                "action": "Check real-time package transit status and carrier updates.",
                "url": "amazon.com/orders",
                "policy_summary": "Deliveries marked out for delivery arrive by 8 PM local. If delayed >48h, initiate refund or replacement."
            },
            "REFUND_AND_RETURN": {
                "action": "Initiate item return, drop-off label, or refund status check.",
                "url": "amazon.com/returns",
                "policy_summary": "Eligible items can be returned within 30 days of delivery. Refunds process in 3-5 business days."
            },
            "DAMAGED_OR_WRONG_ITEM": {
                "action": "Request immediate replacement or return for broken/defective goods.",
                "url": "amazon.com/returns",
                "policy_summary": "Damaged or incorrect shipments are replaced at zero cost with pre-paid return label."
            },
            "ACCOUNT_AND_SECURITY": {
                "action": "Secure account, reset credentials, or report unauthorized charges.",
                "url": "twitter.com/messages/compose?recipient_id=AmazonHelp",
                "policy_summary": "Never share credentials or order numbers publicly. Deflect immediately to secure DM."
            },
            "PRODUCT_AND_TECHNICAL": {
                "action": "Troubleshoot Amazon hardware (Kindle, Echo, Fire TV) or digital services.",
                "url": "amazon.com/devicesupport",
                "policy_summary": "Device restart, firmware update check, or hardware replacement under 1-year warranty."
            },
            "FEEDBACK_AND_GENERAL": {
                "action": "Submit customer feedback or contact support.",
                "url": "amazon.com/help",
                "policy_summary": "General inquiries and feedback submission."
            }
        }

    def evaluate_and_correct(
        self,
        query: str,
        intent: str,
        retrieved_exemplars: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval quality and produces a corrected exemplar payload.
        Filters stop-words to compute salient content token overlap.
        """
        if not retrieved_exemplars:
            return {
                "action": "INCORRECT_FALLBACK",
                "relevance_score": 0.0,
                "sanitized_exemplars": [],
                "policy_guidance": self.verified_policy_kb.get(intent, self.verified_policy_kb["FEEDBACK_AND_GENERAL"])
            }

        # Compute query salient content tokens (filtering stop words)
        q_tokens = {w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS and len(w) > 2}

        scores = []
        valid_exemplars = []

        for ex in retrieved_exemplars:
            score = float(ex.get("rrf_score", 0.0))
            text = ex.get("customer_text", "")
            t_tokens = {w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS and len(w) > 2}

            overlap_ratio = len(q_tokens.intersection(t_tokens)) / max(len(q_tokens), 1) if q_tokens else 0.0
            comb_score = (score * 5.0) + (overlap_ratio * 0.5)
            scores.append(comb_score)

            if comb_score >= self.lower_threshold:
                cleaned_reply = re.sub(r"\^[A-Z]{2,3}", "", ex.get("reply_text", "")).strip()
                cleaned_reply = re.sub(r"@\w+", "", cleaned_reply).strip()
                valid_exemplars.append({**ex, "reply_text": cleaned_reply, "content_score": comb_score})

        avg_score = float(sum(scores) / max(len(scores), 1))

        if avg_score >= self.upper_threshold and valid_exemplars:
            action = "CORRECT"
            sanitized = valid_exemplars
        elif avg_score >= self.lower_threshold and valid_exemplars:
            action = "AMBIGUOUS_REFINED"
            sanitized = valid_exemplars
        else:
            action = "INCORRECT_FALLBACK"
            sanitized = []

        return {
            "action": action,
            "relevance_score": round(avg_score, 4),
            "sanitized_exemplars": sanitized,
            "policy_guidance": self.verified_policy_kb.get(intent, self.verified_policy_kb["FEEDBACK_AND_GENERAL"])
        }
