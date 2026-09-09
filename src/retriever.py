"""
Hybrid Resolution Retriever for @AmazonHelp.
Combines:
1. Okapi BM25 sparse keyword ranking (exact tokens, order numbers, carriers, error terms)
2. Sublinear N-gram TF-IDF vector similarity (semantic phrase matching)
3. Reciprocal Rank Fusion (RRF with k=20) over positive candidates
Includes zero-score protection against degenerate queries.
"""

import os
import re
import math
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BM25Index:
    """Lightweight pure-python Okapi BM25 index."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_freqs = Counter()
        self.inverted_index = {}
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        # Preserve alphanumeric codes, numbers, and short model identifiers
        return re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())

    def fit(self, texts: List[str]):
        self.corpus_size = len(texts)
        self.tokenized_corpus = [self._tokenize(t) for t in texts]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_len = float(np.mean(self.doc_lens)) if self.doc_lens else 1.0

        for doc_id, doc in enumerate(self.tokenized_corpus):
            seen = set()
            for token in doc:
                if token not in seen:
                    self.doc_freqs[token] += 1
                    seen.add(token)
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][doc_id] = self.inverted_index[token].get(doc_id, 0) + 1
        return self

    def score(self, query_tokens: List[str]) -> np.ndarray:
        if not query_tokens or self.corpus_size == 0:
            return np.zeros(self.corpus_size, dtype=np.float32)

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        unique_tokens = Counter(query_tokens)

        for token, q_tf in unique_tokens.items():
            if token not in self.inverted_index:
                continue
            df = self.doc_freqs[token]
            # Standard Robertson-Spärck Jones IDF
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            for doc_id, tf in self.inverted_index[token].items():
                doc_len = self.doc_lens[doc_id]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_id] += idf * (numerator / denominator) * min(q_tf, 2)
        return scores


class HistoricalResolutionRetriever:
    """
    Hybrid Retriever combining Okapi BM25 and Dense TF-IDF via Reciprocal Rank Fusion (RRF).
    Tuned with k_rrf=20 for short customer support tweets.
    """
    def __init__(self, top_k: int = 3, rrf_k: int = 20):
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=25000
        )
        self.bm25 = BM25Index()
        self.records: List[Dict[str, Any]] = []
        self.tfidf_matrix = None

    def build_index(self, pairs_df: pd.DataFrame):
        """Builds both sparse BM25 and dense TF-IDF search indices."""
        self.records = []
        texts = []

        for idx, row in pairs_df.iterrows():
            cust_text = str(row.get("customer_text", "")).strip()
            reply_text = str(row.get("reply_text", "")).strip()
            intent = row.get("intent", "UNKNOWN")

            if len(cust_text) > 5 and len(reply_text) > 5:
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
            self.bm25.fit(texts)
        return self

    def retrieve_similar(
        self,
        query: str,
        intent_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k historical exemplars using Reciprocal Rank Fusion of BM25 and TF-IDF.
        Safely returns empty list on zero-score/empty queries.
        """
        k = top_k or self.top_k
        if not self.records or self.tfidf_matrix is None or not query.strip():
            return []

        # 1. Dense TF-IDF Cosine Similarity Ranking
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # 2. Sparse BM25 Ranking
        q_tokens = self.bm25._tokenize(query)
        bm25_scores = self.bm25.score(q_tokens)

        # Zero-score check: if both models have zero match, return empty
        max_tfidf = float(np.max(tfidf_scores)) if len(tfidf_scores) > 0 else 0.0
        max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) > 0 else 0.0
        if max_tfidf <= 0.0 and max_bm25 <= 0.0:
            return []

        # Candidate pool filtering
        candidate_ids = set(range(len(self.records)))
        if intent_filter and intent_filter != "UNKNOWN":
            filtered = {i for i in candidate_ids if self.records[i].get("intent") == intent_filter}
            if len(filtered) >= k:
                candidate_ids = filtered

        # 3. Restrict ranking to positive score documents (Top-50 pool per modality)
        pos_tfidf = [i for i in candidate_ids if tfidf_scores[i] > 0.0]
        pos_bm25 = [i for i in candidate_ids if bm25_scores[i] > 0.0]

        top_tfidf = sorted(pos_tfidf, key=lambda i: tfidf_scores[i], reverse=True)[:50]
        top_bm25 = sorted(pos_bm25, key=lambda i: bm25_scores[i], reverse=True)[:50]

        rank_tfidf = {doc_id: rank + 1 for rank, doc_id in enumerate(top_tfidf)}
        rank_bm25 = {doc_id: rank + 1 for rank, doc_id in enumerate(top_bm25)}

        pooled_candidates = set(top_tfidf).union(set(top_bm25))
        if not pooled_candidates:
            return []

        # 4. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for doc_id in pooled_candidates:
            score = 0.0
            if doc_id in rank_tfidf:
                score += 1.0 / (self.rrf_k + rank_tfidf[doc_id])
            if doc_id in rank_bm25:
                score += 1.0 / (self.rrf_k + rank_bm25[doc_id])
            rrf_scores[doc_id] = score

        sorted_candidates = sorted(pooled_candidates, key=lambda i: rrf_scores[i], reverse=True)[:k]

        results = []
        for idx in sorted_candidates:
            rec = self.records[idx].copy()
            rec["rrf_score"] = round(float(rrf_scores[idx]), 5)
            rec["tfidf_score"] = round(float(tfidf_scores[idx]), 4)
            rec["bm25_score"] = round(float(bm25_scores[idx]), 4)
            results.append(rec)

        return results

    def save_index(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "records": self.records,
                "tfidf_matrix": self.tfidf_matrix,
                "bm25": self.bm25
            }, f)

    def load_index(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.records = data["records"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.bm25 = data.get("bm25", BM25Index())
        return self
