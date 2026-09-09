"""
Interactive Demo CLI for @AmazonHelp AI Support Agent.
Allows testing live customer queries or pre-built edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
from src.agent import AmazonSupportAgent
from src.classifier import ProductionIntentClassifier
from src.escalator import ProductionEscalator
from src.retriever import HistoricalResolutionRetriever
from src.generator import ProductionGroundedGenerator

def main():
    print("=" * 70)
    print("      @AmazonHelp AI Customer Support Agent - Interactive Shell")
    print("=" * 70)

    # Initialize agent components
    train_csv = "data/processed/amazonhelp_english_pairs.csv"
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found. Run data_pipeline.py first.")
        return

    print("Loading historical support index (10,000 resolved pairs)...")
    df = pd.read_csv(train_csv)
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    
    # Fast weak label assignment for retrieval indexing
    intents = []
    for text in sample_df["customer_text"]:
        t_low = str(text).lower()
        if any(w in t_low for w in ["damaged", "broken", "wrong item", "defective", "missing item", "empty box"]):
            intents.append("DAMAGED_OR_WRONG_ITEM")
        elif any(w in t_low for w in ["login", "password", "otp", "2fa", "account", "hacked", "prime"]):
            intents.append("ACCOUNT_AND_SECURITY")
        elif any(w in t_low for w in ["kindle", "fire", "alexa", "echo", "app", "stream", "device"]):
            intents.append("PRODUCT_AND_TECHNICAL")
        elif any(w in t_low for w in ["refund", "return", "cancel", "charge", "money back", "billing"]):
            intents.append("REFUND_AND_RETURN")
        elif any(w in t_low for w in ["track", "delivery", "carrier", "transit", "where is", "late", "package"]):
            intents.append("ORDER_TRACKING_AND_DELAY")
        else:
            intents.append("FEEDBACK_AND_GENERAL")
            
    sample_df["intent"] = intents
    
    classifier = ProductionIntentClassifier().fit(sample_df["customer_text"].tolist(), intents)
    escalator = ProductionEscalator()
    retriever = HistoricalResolutionRetriever(top_k=3).build_index(sample_df)
    generator = ProductionGroundedGenerator()
    agent = AmazonSupportAgent(classifier=classifier, escalator=escalator, retriever=retriever, generator=generator)

    sample_queries = [
        "My tracking says delivered 3 hours ago by USPS but nothing is outside my door. Was it stolen?!",
        "How do I print a return label for a pair of shoes that didn't fit?",
        "I was charged twice on my card for order 112-4928192-3849102. Reverse this right now or I'm calling my lawyer!",
        "My Kindle paperwhite screen froze on the loading screen and won't restart.",
        "Opened my delivery and the ceramic mug is completely shattered in pieces inside the box!",
        "Thank you so much to the representative who helped me with my Prime membership today, amazing service!"
    ]

    print("\nReady! Type an incoming customer tweet or choose a sample (1-6). Type 'exit' to quit.\n")
    for i, q in enumerate(sample_queries, 1):
        print(f"[{i}] \"{q}\"")
    print("-" * 70)

    while True:
        try:
            user_input = input("\nEnter tweet (or 1-6 / 'exit'): ").strip()
            if not user_input or user_input.lower() == 'exit':
                break

            if user_input in [str(i) for i in range(1, len(sample_queries) + 1)]:
                user_input = sample_queries[int(user_input) - 1]

            print(f"\nProcessing Tweet: \"{user_input}\"...")
            res = agent.process_message(user_input)

            print("\n" + "=" * 50)
            print("                 AGENT DECISION")
            print("=" * 50)
            print(f"Predicted Intent:     {res['intent']} (Confidence: {res['confidence']*100:.1f}%)")
            print(f"Triage Decision:      {'ESCALATE TO HUMAN' if res['escalation']['escalate'] else 'AUTO-HANDLED'}")
            print(f"Stated Reason:        {res['escalation']['stated_reason']}")
            print(f"Risk Score:           {res['escalation']['risk_score']:.2f}")
            print(f"Inference Latency:    {res['latency_ms']} ms")
            print("\nTop Historical Exemplar Retrieved:")
            if res['exemplars']:
                print(f"  Historical Query: \"{res['exemplars'][0]['customer_text'][:90]}...\"")
                print(f"  Historical Reply: \"{res['exemplars'][0]['reply_text'][:90]}...\"")
            print("\nDraft Grounded Reply:")
            print(f"  --> \"{res['draft_reply']}\"")
            print(f"  (Character count: {len(res['draft_reply'])} / 280 max)")
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nExiting interactive shell.")

if __name__ == "__main__":
    main()
