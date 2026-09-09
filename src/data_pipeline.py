"""
Data processing pipeline for Twitter Customer Support dataset.
Filters and formats multi-turn Twitter customer support threads specifically for @AmazonHelp.
"""

import os
import csv
import json
import logging
import pandas as pd
from typing import Optional, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_amazon_conversations(
    raw_csv_path: str = "data/raw/twcs.csv",
    output_pairs_csv: str = "data/processed/amazonhelp_pairs.csv",
    output_pairs_jsonl: str = "data/processed/amazonhelp_pairs.jsonl",
    max_pairs: Optional[int] = 50000,
    min_customer_len: int = 15,
    min_reply_len: int = 15
) -> pd.DataFrame:
    """
    Reads twcs.csv, finds AmazonHelp responses and links them to customer inbound tweets.
    
    twcs.csv schema:
    - tweet_id
    - author_id
    - inbound (True/False)
    - created_at
    - text
    - response_tweet_id
    - in_response_to_tweet_id
    """
    logger.info(f"Loading raw tweets from {raw_csv_path}...")
    
    # Read relevant columns in chunks to be memory-efficient
    # We first collect all AmazonHelp response tweets and inbound customer tweets
    chunk_size = 200000
    amazon_replies = [] # list of dicts: {tweet_id, author_id, in_response_to_tweet_id, reply_text, created_at}
    candidate_inbound_ids = set()
    
    logger.info("Pass 1: Identifying AmazonHelp replies...")
    for chunk in pd.read_csv(
        raw_csv_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text", "in_response_to_tweet_id"],
        chunksize=chunk_size,
        dtype={"tweet_id": str, "author_id": str, "inbound": bool, "in_response_to_tweet_id": str, "text": str}
    ):
        # Filter for AmazonHelp outbound tweets that respond to an inbound tweet
        mask = (chunk["author_id"] == "AmazonHelp") & (~chunk["inbound"]) & (chunk["in_response_to_tweet_id"].notna())
        matches = chunk[mask]
        for _, row in matches.iterrows():
            in_resp_id = str(row["in_response_to_tweet_id"]).strip()
            # In twcs.csv, in_response_to_tweet_id might be float string like '115712.0'
            if in_resp_id.endswith(".0"):
                in_resp_id = in_resp_id[:-2]
            if in_resp_id:
                candidate_inbound_ids.add(in_resp_id)
                amazon_replies.append({
                    "reply_tweet_id": str(row["tweet_id"]),
                    "in_response_to_tweet_id": in_resp_id,
                    "reply_text": str(row["text"]).strip(),
                    "reply_created_at": row["created_at"]
                })
        
        if max_pairs and len(amazon_replies) >= max_pairs * 2:
            logger.info(f"Collected sufficient reply candidates ({len(amazon_replies)}). Proceeding...")
            break
            
    logger.info(f"Found {len(amazon_replies)} AmazonHelp replies referencing {len(candidate_inbound_ids)} customer tweets.")
    
    # Pass 2: Find the matching inbound customer tweets
    logger.info("Pass 2: Extracting customer inbound tweets...")
    inbound_map = {} # tweet_id -> {customer_text, customer_created_at, author_id}
    
    for chunk in pd.read_csv(
        raw_csv_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text"],
        chunksize=chunk_size,
        dtype={"tweet_id": str, "author_id": str, "inbound": bool, "text": str}
    ):
        # Filter for candidate IDs
        chunk["clean_tweet_id"] = chunk["tweet_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        matched_inbound = chunk[chunk["clean_tweet_id"].isin(candidate_inbound_ids) & chunk["inbound"]]
        
        for _, row in matched_inbound.iterrows():
            inbound_map[row["clean_tweet_id"]] = {
                "customer_text": str(row["text"]).strip(),
                "customer_created_at": row["created_at"],
                "customer_author_id": row["author_id"]
            }
            
        if len(inbound_map) >= len(candidate_inbound_ids):
            break

    logger.info(f"Matched {len(inbound_map)} customer inbound tweets.")
    
    # Pair them together
    pairs = []
    seen_inbound = set()
    
    for reply in amazon_replies:
        in_id = reply["in_response_to_tweet_id"]
        if in_id in inbound_map and in_id not in seen_inbound:
            cust_info = inbound_map[in_id]
            cust_text = cust_info["customer_text"]
            reply_text = reply["reply_text"]
            
            # Basic sanity filtering
            if len(cust_text) >= min_customer_len and len(reply_text) >= min_reply_len:
                # Exclude purely non-English characters or junk
                pairs.append({
                    "inbound_id": in_id,
                    "customer_author_id": cust_info["customer_author_id"],
                    "customer_text": cust_text,
                    "customer_created_at": cust_info["customer_created_at"],
                    "reply_id": reply["reply_tweet_id"],
                    "reply_text": reply_text,
                    "reply_created_at": reply["reply_created_at"]
                })
                seen_inbound.add(in_id)
                
            if max_pairs and len(pairs) >= max_pairs:
                break
                
    pairs_df = pd.DataFrame(pairs)
    logger.info(f"Total valid conversation pairs generated: {len(pairs_df)}")
    
    os.makedirs(os.path.dirname(output_pairs_csv), exist_ok=True)
    pairs_df.to_csv(output_pairs_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Saved CSV to {output_pairs_csv}")
    
    with open(output_pairs_jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info(f"Saved JSONL to {output_pairs_jsonl}")
    
    return pairs_df

if __name__ == "__main__":
    process_amazon_conversations()
