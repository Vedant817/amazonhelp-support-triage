"""Independent, provenance-preserving storage for local human review."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.classifier import INTENT_TAXONOMY
from src.escalator import AUTO_REASON_BY_INTENT, ESCALATE_REASON_BY_INTENT
from src.supervision import RESOLUTION_ACTS, training_review_source_hash


REASON_CODES = tuple(sorted({
    *AUTO_REASON_BY_INTENT.values(),
    *ESCALATE_REASON_BY_INTENT.values(),
    "PII_PUBLIC_DEFLECTION_TO_PRIVATE_CHANNEL",
}))
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,32}$")
REVIEW_COLUMNS = (
    "inbound_id",
    "source_sha256",
    "reviewer_id",
    "reviewed_intent",
    "reviewed_escalate",
    "reviewed_reason_code",
    "reviewed_resolution_act",
    "ambiguous",
    "review_notes",
    "reviewed_at_utc",
)
RATING_FIELDS = ("groundedness", "empathy", "actionability", "safety")
RATING_COLUMNS = (
    "reply_id", "source_sha256", "reviewer_id",
    *(f"human_{field}" for field in RATING_FIELDS),
    "human_notes", "reviewed_at_utc",
)


def reply_source_hash(
    reply_id: object, customer_text: object, draft_reply: object, resolution_act: object
) -> str:
    return hashlib.sha256("\0".join(str(value) for value in (
        reply_id, customer_text, draft_reply, resolution_act
    )).encode("utf-8")).hexdigest()


class TrainingReviewStore:
    """Stores each reviewer's labels separately; model suggestions are never returned."""

    def __init__(
        self,
        queue_path: str = "data/training/labels_for_human_review.csv",
        output_dir: str = "data/reviews",
        gold_path: str = "data/golden/golden_eval_set.csv",
    ) -> None:
        self.queue_path = Path(queue_path)
        self.output_dir = Path(output_dir)
        self.gold_path = Path(gold_path)
        self._lock = threading.Lock()
        self.queue = pd.read_csv(self.queue_path).fillna("")
        required = {
            "inbound_id", "conversation_context", "customer_text", "reply_text",
        }
        missing = required - set(self.queue.columns)
        if missing:
            raise ValueError(f"Review queue is missing columns: {sorted(missing)}")
        if self.queue["inbound_id"].astype(str).duplicated().any():
            raise ValueError("Review queue contains duplicate inbound IDs")
        if self.gold_path.exists():
            gold_ids = set(pd.read_csv(self.gold_path)["tweet_id"].astype(str))
            overlap = set(self.queue["inbound_id"].astype(str)) & gold_ids
            if overlap:
                raise ValueError("Review queue overlaps the frozen gold set")
        self.queue["source_sha256"] = self.queue.apply(
            training_review_source_hash, axis=1
        )

    @staticmethod
    def validate_reviewer_id(reviewer_id: str) -> str:
        value = str(reviewer_id).strip()
        if not REVIEWER_PATTERN.fullmatch(value):
            raise ValueError("Reviewer ID must be 2-32 letters, numbers, hyphens, or underscores")
        return value

    def _review_path(self, reviewer_id: str) -> Path:
        safe_id = self.validate_reviewer_id(reviewer_id)
        return self.output_dir / f"training-review-{safe_id}.csv"

    def _read_reviews(self, reviewer_id: str) -> pd.DataFrame:
        path = self._review_path(reviewer_id)
        if not path.exists():
            return pd.DataFrame(columns=REVIEW_COLUMNS)
        frame = pd.read_csv(path, dtype={"inbound_id": str}).fillna("")
        missing = set(REVIEW_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Reviewer file is missing columns: {sorted(missing)}")
        return frame[list(REVIEW_COLUMNS)]

    def progress(self, reviewer_id: str) -> Dict[str, int]:
        reviews = self._read_reviews(reviewer_id)
        completed = int(reviews["inbound_id"].astype(str).nunique())
        return {"completed": completed, "total": len(self.queue), "remaining": len(self.queue) - completed}

    def next_item(self, reviewer_id: str) -> Optional[Dict[str, Any]]:
        reviews = self._read_reviews(reviewer_id)
        completed = set(reviews["inbound_id"].astype(str))
        remaining = self.queue[~self.queue["inbound_id"].astype(str).isin(completed)]
        if remaining.empty:
            return None
        row = remaining.iloc[0]
        progress = self.progress(reviewer_id)
        return {
            "inbound_id": str(row["inbound_id"]),
            "source_sha256": row["source_sha256"],
            "conversation_context": str(row["conversation_context"]),
            "customer_text": str(row["customer_text"]),
            "historical_reply": str(row["reply_text"]),
            "position": progress["completed"] + 1,
            **progress,
            "intent_options": list(INTENT_TAXONOMY),
            "reason_options": list(REASON_CODES),
            "act_options": list(RESOLUTION_ACTS),
        }

    def save(self, reviewer_id: str, payload: Dict[str, Any]) -> Dict[str, int]:
        safe_id = self.validate_reviewer_id(reviewer_id)
        inbound_id = str(payload.get("inbound_id", "")).strip()
        matches = self.queue[self.queue["inbound_id"].astype(str) == inbound_id]
        if len(matches) != 1:
            raise ValueError("Inbound ID is not present exactly once in the review queue")
        source_sha = str(payload.get("source_sha256", ""))
        if source_sha != str(matches.iloc[0]["source_sha256"]):
            raise ValueError("Review source changed; reload this case before saving")
        intent = str(payload.get("reviewed_intent", "")).strip()
        if intent not in INTENT_TAXONOMY:
            raise ValueError("Select one valid intent")
        escalate = payload.get("reviewed_escalate")
        if not isinstance(escalate, bool):
            raise ValueError("Select whether a specialist is required")
        reason = str(payload.get("reviewed_reason_code", "")).strip()
        if reason not in REASON_CODES:
            raise ValueError("Select one valid routing reason")
        allowed_reasons = (
            {*ESCALATE_REASON_BY_INTENT.values(), "PII_PUBLIC_DEFLECTION_TO_PRIVATE_CHANNEL"}
            if escalate else set(AUTO_REASON_BY_INTENT.values())
        )
        if reason not in allowed_reasons:
            route = "specialist" if escalate else "auto-handle"
            raise ValueError(f"Routing reason is inconsistent with the {route} decision")
        act = str(payload.get("reviewed_resolution_act", "")).strip()
        if act not in RESOLUTION_ACTS:
            raise ValueError("Select one valid resolution act")
        ambiguous = payload.get("ambiguous", False)
        if not isinstance(ambiguous, bool):
            raise ValueError("Ambiguous must be a boolean")
        notes = str(payload.get("review_notes", "")).strip()[:1000]
        if ambiguous and not notes:
            raise ValueError("Explain why an ambiguous case cannot be labeled confidently")
        record = {
            "inbound_id": inbound_id,
            "source_sha256": source_sha,
            "reviewer_id": safe_id,
            "reviewed_intent": intent,
            "reviewed_escalate": escalate,
            "reviewed_reason_code": reason,
            "reviewed_resolution_act": act,
            "ambiguous": ambiguous,
            "review_notes": notes,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            current = self._read_reviews(safe_id)
            current = current[current["inbound_id"].astype(str) != inbound_id]
            updated = pd.concat([current, pd.DataFrame([record])], ignore_index=True)
            updated = updated[list(REVIEW_COLUMNS)]
            path = self._review_path(safe_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            updated.to_csv(temporary, index=False)
            os.replace(temporary, path)
        return self.progress(safe_id)


class ReplyRatingStore:
    """Blind same-reply quality ratings stored independently by reviewer."""

    def __init__(
        self,
        queue_path: str = "results/human_rating_template.csv",
        output_dir: str = "data/reviews",
    ) -> None:
        self.queue_path = Path(queue_path)
        self.output_dir = Path(output_dir)
        self._lock = threading.Lock()
        self.queue = pd.read_csv(self.queue_path).fillna("")
        required = {"reply_id", "customer_text", "draft_reply", "resolution_act"}
        missing = required - set(self.queue.columns)
        if missing:
            raise ValueError(f"Reply-rating queue is missing columns: {sorted(missing)}")
        if self.queue["reply_id"].astype(str).duplicated().any():
            raise ValueError("Reply-rating queue contains duplicate reply IDs")
        self.queue["source_sha256"] = self.queue.apply(
            lambda row: reply_source_hash(
                row["reply_id"], row["customer_text"], row["draft_reply"],
                row["resolution_act"],
            ),
            axis=1,
        )

    def _path(self, reviewer_id: str) -> Path:
        safe_id = TrainingReviewStore.validate_reviewer_id(reviewer_id)
        return self.output_dir / f"reply-rating-{safe_id}.csv"

    def _read(self, reviewer_id: str) -> pd.DataFrame:
        path = self._path(reviewer_id)
        if not path.exists():
            return pd.DataFrame(columns=RATING_COLUMNS)
        frame = pd.read_csv(path, dtype={"reply_id": str}).fillna("")
        missing = set(RATING_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Reply-rating file is missing columns: {sorted(missing)}")
        return frame[list(RATING_COLUMNS)]

    def progress(self, reviewer_id: str) -> Dict[str, int]:
        completed = int(self._read(reviewer_id)["reply_id"].astype(str).nunique())
        return {"completed": completed, "total": len(self.queue), "remaining": len(self.queue) - completed}

    def next_item(self, reviewer_id: str) -> Optional[Dict[str, Any]]:
        completed = set(self._read(reviewer_id)["reply_id"].astype(str))
        remaining = self.queue[~self.queue["reply_id"].astype(str).isin(completed)]
        if remaining.empty:
            return None
        row = remaining.iloc[0]
        progress = self.progress(reviewer_id)
        return {
            "reply_id": str(row["reply_id"]),
            "source_sha256": row["source_sha256"],
            "customer_text": str(row["customer_text"]),
            "draft_reply": str(row["draft_reply"]),
            "resolution_act": str(row["resolution_act"]),
            "position": progress["completed"] + 1,
            **progress,
        }

    def save(self, reviewer_id: str, payload: Dict[str, Any]) -> Dict[str, int]:
        safe_id = TrainingReviewStore.validate_reviewer_id(reviewer_id)
        reply_id = str(payload.get("reply_id", "")).strip()
        matches = self.queue[self.queue["reply_id"].astype(str) == reply_id]
        if len(matches) != 1:
            raise ValueError("Reply ID is not present exactly once in the rating queue")
        if str(payload.get("source_sha256", "")) != str(matches.iloc[0]["source_sha256"]):
            raise ValueError("Reply changed; reload this item before saving")
        record: Dict[str, Any] = {
            "reply_id": reply_id,
            "source_sha256": str(payload["source_sha256"]),
            "reviewer_id": safe_id,
        }
        for field in RATING_FIELDS:
            value = payload.get(f"human_{field}")
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"{field} must be an integer from 1 to 5")
            record[f"human_{field}"] = value
        record["human_notes"] = str(payload.get("human_notes", "")).strip()[:1000]
        record["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            current = self._read(safe_id)
            current = current[current["reply_id"].astype(str) != reply_id]
            updated = pd.concat([current, pd.DataFrame([record])], ignore_index=True)
            path = self._path(safe_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            updated[list(RATING_COLUMNS)].to_csv(temporary, index=False)
            os.replace(temporary, path)
        return self.progress(safe_id)
