"""
Filters Amazon conversation pairs to high-quality English conversations.
"""

import pandas as pd
import json
import re

ENG_STOPWORDS = {
    'the', 'my', 'is', 'to', 'and', 'have', 'your', 'for', 'you', 'with', 
    'in', 'it', 'on', 'this', 'was', 'not', 'we', 'are', 'can', 'do', 'order',
    'package', 'delivery', 'delayed', 'refund', 'help', 'account', 'item'
}

def is_pure_english(text: str) -> bool:
    if not isinstance(text, str):
        return False
    words = set(re.findall(r"\b[a-z]{2,}\b", text.lower()))
    if not words:
        return False
    # Must contain at least 2 common English stopwords/markers
    overlap = words.intersection(ENG_STOPWORDS)
    if len(overlap) < 2:
        return False
    # Check ASCII ratio
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / max(len(text), 1)) >= 0.90

def clean_subset():
    df = pd.read_csv("data/processed/amazonhelp_pairs.csv")
    print(f"Total raw pairs: {len(df)}")
    
    mask = df["customer_text"].apply(is_pure_english) & df["reply_text"].apply(is_pure_english)
    eng_df = df[mask].copy()
    print(f"Pure English pairs: {len(eng_df)}")
    
    # Clean text: remove excessive whitespace
    eng_df["customer_text"] = eng_df["customer_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    eng_df["reply_text"] = eng_df["reply_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    
    # Filter out extremely short or trivial one-word tweets
    eng_df = eng_df[(eng_df["customer_text"].str.len() >= 25) & (eng_df["reply_text"].str.len() >= 25)]
    print(f"Final clean high-quality English pairs: {len(eng_df)}")
    
    eng_df.to_csv("data/processed/amazonhelp_english_pairs.csv", index=False)
    
    with open("data/processed/amazonhelp_english_pairs.jsonl", "w", encoding="utf-8") as f:
        for _, row in eng_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            
    print("Saved clean English subset!")

if __name__ == "__main__":
    clean_subset()
