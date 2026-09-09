"""
Historical Resolution Retriever for @AmazonHelp.
Indexes resolved customer support interactions and retrieves top-k similar historical issues
and successful brand resolutions to ground agent replies.
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HistoricalResolutionRetriever:
    """
    Retrieves top historical conversation pairs from @AmazonHelp to ground draft replies.
    Supports intent-filtered search or global search.
    """
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            stop_words="english"
        )
        self.records: List[Dict[str, Any]] = []
        self.tfidf_matrix = None

    def build_index(self, pairs_df: pd.DataFrame):
        """Builds search index from DataFrame of conversation pairs."""
        self.records = []
        texts = []

        for idx, row in pairs_df.iterrows():
            cust_text = str(row.get("customer_text", "")).strip()
            reply_text = str(row.get("reply_text", "")).strip()
            intent = row.get("intent", "UNKNOWN")

            if len(cust_text) > 10 and len(reply_text) > 10:
                self.records.append({
                    "id": idx,
                    "customer_text": cust_text,
                    "reply_text": reply_text,
                    "intent": intent,
                    "reply_id": row.get("reply_id", "")
                })
                texts.append(cust_text)

        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self

    def retrieve_similar(
        self,
        query: str,
        intent_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar historical customer queries and their brand responses.
        """
        k = top_k or self.top_k
        if not self.records or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Optional intent filtering
        candidate_indices = range(len(self.records))
        if intent_filter and intent_filter != "UNKNOWN":
            filtered = [i for i in candidate_indices if self.records[i].get("intent") == intent_filter]
            if len(filtered) >= k:
                candidate_indices = filtered

        # Sort candidate indices by similarity
        ranked_indices = sorted(candidate_indices, key=lambda i: sim_scores[i], reverse=True)[:k]

        results = []
        for idx in ranked_indices:
            rec = self.records[idx].copy()
            rec["similarity_score"] = round(float(sim_scores[idx]), 4)
            results.append(rec)

        return results

    def save_index(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "records": self.records, "tfidf_matrix": self.tfidf_matrix}, f)

    def load_index(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.records = data["records"]
            self.tfidf_matrix = data["tfidf_matrix"]
        return self
