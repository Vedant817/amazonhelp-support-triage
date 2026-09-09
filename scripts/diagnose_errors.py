"""
Diagnostic script to inspect errors made by the Production System on the 200 Golden Set.
Outputs exact confusion matrices, false positives, false negatives, and misclassified tweets.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd

# Load golden set and results
gold_df = pd.read_csv("data/golden/golden_eval_set.csv")
print(f"Loaded {len(gold_df)} golden evaluation cases.")

train_df = pd.read_csv("data/processed/amazonhelp_english_pairs.csv")
train_sample = train_df.sample(n=min(15000, len(train_df)), random_state=42)

from src.classifier import ProductionIntentClassifier
from src.escalator import ProductionEscalator

# High quality weak supervision rules for training corpus
silver_labels = []
for text in train_sample["customer_text"]:
    t_low = str(text).lower()
    if any(w in t_low for w in ["damaged", "broken", "wrong item", "defective", "missing item", "empty box", "shattered", "cracked", "scratched", "missing part"]):
        silver_labels.append("DAMAGED_OR_WRONG_ITEM")
    elif any(w in t_low for w in ["login", "password", "otp", "2fa", "hacked", "prime membership", "locked out", "sign in", "cant access account", "verify account"]):
        silver_labels.append("ACCOUNT_AND_SECURITY")
    elif any(w in t_low for w in ["kindle", "fire tv", "firestick", "alexa", "echo", "prime video", "streaming", "bluetooth", "paperwhite", "app crash"]):
        silver_labels.append("PRODUCT_AND_TECHNICAL")
    elif any(w in t_low for w in ["refund", "return", "returned", "cancel order", "charged twice", "money back", "billing", "card charged", "double charge"]):
        silver_labels.append("REFUND_AND_RETURN")
    elif any(w in t_low for w in ["track", "delivery", "carrier", "transit", "where is my", "ups", "usps", "late", "package", "parcel", "arrived", "shipping", "shipped", "shipment", "not arrived", "deliver", "two day shipping"]):
        silver_labels.append("ORDER_TRACKING_AND_DELAY")
    else:
        silver_labels.append("FEEDBACK_AND_GENERAL")

clf = ProductionIntentClassifier(confidence_threshold=0.32).fit(train_sample["customer_text"].tolist(), silver_labels)
esc = ProductionEscalator(confidence_threshold=0.32)

intent_errors = []
esc_errors = []

for idx, row in gold_df.iterrows():
    text = row["customer_text"]
    gold_i = row["gold_intent"]
    gold_e = bool(row["gold_escalate"])
    gold_r = row["gold_escalate_reason"]

    res = clf.predict_with_confidence(text)
    pred_i = res["intent"]
    conf = res["confidence"]
    is_unc = res["is_uncertain"]

    dec = esc.evaluate(text, intent=pred_i, confidence=conf, is_uncertain=is_unc)
    pred_e = dec.should_escalate
    pred_r = dec.stated_reason

    if pred_i != gold_i:
        intent_errors.append({
            "id": row["id"],
            "text": text,
            "gold_intent": gold_i,
            "pred_intent": pred_i,
            "confidence": round(conf, 3)
        })

    if pred_e != gold_e:
        esc_errors.append({
            "id": row["id"],
            "text": text,
            "gold_escalate": gold_e,
            "pred_escalate": pred_e,
            "gold_reason": gold_r,
            "pred_reason": pred_r,
            "intent": pred_i
        })

print(f"\n--- Intent Errors: {len(intent_errors)} / 200 (Acc: {(200-len(intent_errors))/200*100:.1f}%) ---")
print("Top Intent Misclassifications:")
for e in intent_errors[:10]:
    clean_t = e['text'].encode('ascii', 'ignore').decode()
    print(f"[{e['id']}] Gold: {e['gold_intent']} -> Pred: {e['pred_intent']} (Conf: {e['confidence']})")
    print(f"  Text: {clean_t[:90]}...")

print(f"\n--- Escalation Errors: {len(esc_errors)} / 200 (Acc: {(200-len(esc_errors))/200*100:.1f}%) ---")
print("Top Escalation Misclassifications:")
for e in esc_errors[:10]:
    clean_t = e['text'].encode('ascii', 'ignore').decode()
    print(f"[{e['id']}] Gold: {e['gold_escalate']} ({e['gold_reason']}) -> Pred: {e['pred_escalate']} ({e['pred_reason']})")
    print(f"  Text: {clean_t[:90]}...")
