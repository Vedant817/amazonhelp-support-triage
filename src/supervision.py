"""Reply-derived supervision and resolution-act extraction.

Labels in this module are deliberately inferred from historical AmazonHelp replies,
not from the customer-keyword baseline. Ambiguous examples are left unlabeled rather
than forced into a class. Optional reviewed labels can override weak labels without
ever touching the frozen test split.
"""

from __future__ import annotations

import hashlib
import re
from numbers import Real
from typing import Any, Optional

import numpy as np
import pandas as pd


RESOLUTION_ACTS = (
    "ASK_TRACKING_STATUS",
    "ASK_CARRIER",
    "DM_WITH_ORDER_ID",
    "LINK_ORDERS",
    "LINK_RETURNS",
    "TROUBLESHOOT_DEVICE",
    "APOLOGIZE_WAIT",
    "GENERAL_SUPPORT",
)

VALID_INTENTS = {
    "ORDER_TRACKING_AND_DELAY",
    "REFUND_AND_RETURN",
    "DAMAGED_OR_WRONG_ITEM",
    "ACCOUNT_AND_SECURITY",
    "PRODUCT_AND_TECHNICAL",
    "FEEDBACK_AND_GENERAL",
}


def training_review_source_hash(row: Any) -> str:
    """Bind a human label to the exact historical pair shown to the reviewer."""
    digest = hashlib.sha256()
    for column in ("inbound_id", "conversation_context", "customer_text", "reply_text"):
        value = row.get(column, "") if hasattr(row, "get") else ""
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_reviewed_escalation(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    raise ValueError(f"Invalid reviewed_escalate value: {value!r}")


def strip_agent_pii(text: str) -> str:
    """Remove handles, agent signatures, emails, phones, and order IDs from exemplars."""
    value = str(text or "")
    value = re.sub(r"\^[A-Z]{2,3}\s*$", "", value).strip()
    value = re.sub(r"@\w+", "", value)
    value = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[email redacted]", value)
    value = re.sub(r"\b\d{3}-\d{7}-\d{7}\b", "[order ID redacted]", value)
    value = re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[phone redacted]", value)
    return re.sub(r"\s+", " ", value).strip()


def infer_resolution_act(reply_text: str) -> str:
    text = str(reply_text or "").lower()
    if re.search(r"\b(dm|direct message|message us privately|private message)\b", text):
        return "DM_WITH_ORDER_ID"
    if re.search(r"amazon\.com/(orders|gp/css/order-history)|\byour orders\b", text):
        return "LINK_ORDERS"
    if re.search(r"amazon\.com/(returns|gp/orc)|\breturn (center|label|portal)\b", text):
        return "LINK_RETURNS"
    if re.search(r"\b(who|which|what) (was |is )?(the )?(carrier|courier)\b", text):
        return "ASK_CARRIER"
    if re.search(r"\bwhat does (the |your )?(tracking|shipping status)|\btracking (show|say|status)", text):
        return "ASK_TRACKING_STATUS"
    if re.search(r"\b(restart|reset|reboot|update|unplug|settings|troubleshoot)\b", text):
        return "TROUBLESHOOT_DEVICE"
    if re.search(r"\b(sorry for the (wait|delay)|thanks for your patience|allow .* hours|wait .* hours)\b", text):
        return "APOLOGIZE_WAIT"
    return "GENERAL_SUPPORT"


def infer_reply_intent(reply_text: str) -> Optional[str]:
    """Return a high-precision intent weak label based only on the agent reply."""
    text = str(reply_text or "").lower()
    act = infer_resolution_act(text)
    if re.search(r"\b(kindle|fire tv|firestick|alexa|echo|prime video|device|bluetooth|wi-?fi|reboot|restart)\b", text):
        return "PRODUCT_AND_TECHNICAL"
    if re.search(r"\b(password|sign[ -]?in|login|account security|verification code|two-step|2fa|account locked)\b", text):
        return "ACCOUNT_AND_SECURITY"
    if re.search(r"\b(damaged|broken|defective|wrong item|replacement|missing (piece|part|contents))\b", text):
        return "DAMAGED_OR_WRONG_ITEM"
    if act == "LINK_RETURNS" or re.search(r"\b(refund|return policy|return label|money back|charged)\b", text):
        return "REFUND_AND_RETURN"
    if act in {"ASK_TRACKING_STATUS", "ASK_CARRIER", "LINK_ORDERS", "APOLOGIZE_WAIT"} or re.search(
        r"\b(delivery|shipment|tracking|carrier|courier|shipping|arrive|dispatch)\b", text
    ):
        return "ORDER_TRACKING_AND_DELAY"
    if re.search(r"\b(thanks for (the|your) (feedback|kind words)|glad|appreciate your feedback|wonderful day)\b", text):
        return "FEEDBACK_AND_GENERAL"
    return None


def infer_reply_escalation(reply_text: str) -> Optional[bool]:
    """Infer whether the historical agent transferred the case to a private/human channel."""
    text = str(reply_text or "").lower()
    if re.search(
        r"\b(dm|direct message|private message|message us privately|phone or chat|call or chat|call us|give us a (call|ring)|contact (our )?customer service( team)?|specialist|security team)\b",
        text,
    ):
        return True
    if infer_resolution_act(text) in {
        "ASK_TRACKING_STATUS",
        "ASK_CARRIER",
        "LINK_ORDERS",
        "LINK_RETURNS",
        "TROUBLESHOOT_DEVICE",
        "APOLOGIZE_WAIT",
    }:
        return False
    return None


def build_reply_supervision(
    pairs: pd.DataFrame,
    reviewed: Optional[pd.DataFrame] = None,
    use_probabilistic_intent_labels: bool = False,
) -> pd.DataFrame:
    """Add weak labels, acts, and provenance; merge optional human-reviewed labels by inbound_id."""
    frame = pairs.copy()
    frame["resolution_act"] = frame["reply_text"].map(infer_resolution_act)
    frame["intent_label"] = frame["reply_text"].map(infer_reply_intent)
    frame["escalate_label"] = frame["reply_text"].map(infer_reply_escalation)
    frame["intent_label_source"] = frame["intent_label"].map(lambda x: "reply_weak" if pd.notna(x) else "unlabeled")
    frame["escalate_label_source"] = frame["escalate_label"].map(lambda x: "reply_weak" if pd.notna(x) else "unlabeled")
    frame["intent_label_confidence"] = frame["intent_label"].map(lambda x: 1.0 if pd.notna(x) else 0.0)
    frame["reason_label"] = None
    frame["reason_label_source"] = "unlabeled"
    frame["resolution_act_source"] = "reply_weak"

    if use_probabilistic_intent_labels:
        from src.classifier import INTENT_TAXONOMY
        from src.weak_label_model import aggregate_intent_weak_labels

        aggregated = aggregate_intent_weak_labels(
            frame, INTENT_TAXONOMY, minimum_confidence=0.75
        )
        for index, (label, confidence) in enumerate(zip(aggregated.labels, aggregated.confidences)):
            frame.at[frame.index[index], "intent_label_confidence"] = float(confidence)
            # Preserve the deliberately high-precision reply label. The label
            # model expands coverage only where reply supervision abstained;
            # conflicts are routed to active review instead of auto-correction.
            if label is not None and pd.isna(frame.at[frame.index[index], "intent_label"]):
                frame.at[frame.index[index], "intent_label"] = label
                frame.at[frame.index[index], "intent_label_source"] = "probabilistic_weak_label_model"

    if reviewed is not None and not reviewed.empty:
        required = {
            "inbound_id", "source_sha256", "reviewed_intent", "reviewed_escalate"
        }
        missing = required - set(reviewed.columns)
        if missing:
            raise ValueError(f"Reviewed label file is missing columns: {sorted(missing)}")
        review_map = reviewed.copy()
        review_map["inbound_id"] = review_map["inbound_id"].astype(str)
        if review_map["inbound_id"].duplicated().any():
            duplicates = sorted(review_map.loc[review_map["inbound_id"].duplicated(), "inbound_id"].unique())
            raise ValueError(f"Reviewed label file contains duplicate inbound_id values: {duplicates[:5]}")
        review_map = review_map.set_index("inbound_id")
        ids = frame["inbound_id"].astype(str)
        for idx, inbound_id in zip(frame.index, ids):
            if inbound_id not in review_map.index:
                continue
            row = review_map.loc[inbound_id]
            expected_hash = training_review_source_hash(frame.loc[idx])
            if str(row["source_sha256"]) != expected_hash:
                raise ValueError(
                    f"Reviewed source hash does not match current training row {inbound_id}"
                )
            if pd.notna(row["reviewed_intent"]):
                reviewed_intent = str(row["reviewed_intent"]).strip()
                if reviewed_intent not in VALID_INTENTS:
                    raise ValueError(f"Invalid reviewed_intent for inbound_id {inbound_id}: {reviewed_intent!r}")
                frame.at[idx, "intent_label"] = reviewed_intent
                frame.at[idx, "intent_label_source"] = "human_reviewed"
            if pd.notna(row["reviewed_escalate"]):
                frame.at[idx, "escalate_label"] = _parse_reviewed_escalation(row["reviewed_escalate"])
                frame.at[idx, "escalate_label_source"] = "human_reviewed"
            if "reviewed_reason_code" in review_map.columns and pd.notna(row["reviewed_reason_code"]):
                reason = str(row["reviewed_reason_code"]).strip()
                if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,79}", reason):
                    raise ValueError(f"Invalid reviewed_reason_code for inbound_id {inbound_id}: {reason!r}")
                frame.at[idx, "reason_label"] = reason
                frame.at[idx, "reason_label_source"] = "human_reviewed"
            if "reviewed_resolution_act" in review_map.columns and pd.notna(row["reviewed_resolution_act"]):
                act = str(row["reviewed_resolution_act"]).strip()
                if act not in RESOLUTION_ACTS:
                    raise ValueError(f"Invalid reviewed_resolution_act for inbound_id {inbound_id}: {act!r}")
                frame.at[idx, "resolution_act"] = act
                frame.at[idx, "resolution_act_source"] = "human_reviewed"
    return frame
