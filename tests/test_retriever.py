"""
Unit tests for historical retrieval system.
"""

import pytest
import numpy as np
import pandas as pd
from src.retriever import HistoricalResolutionRetriever


class TestHistoricalResolutionRetriever:
    """Tests for the HistoricalResolutionRetriever class."""

    def setup_method(self):
        # Create a small test dataset
        self.test_records = pd.DataFrame([
            {
                "reply_id": "1",
                "customer_text": "My package is late",
                "reply_text": "Sorry for the delay. Check amazon.com/orders",
                "intent": "ORDER_TRACKING_AND_DELAY",
                "resolution_act": "LINK_ORDERS",
                "rrf_score": 0.8,
                "dense_score": 0.7,
                "bm25_score": 0.6,
            },
            {
                "reply_id": "2",
                "customer_text": "Item arrived damaged",
                "reply_text": "Sorry for the damage. Check amazon.com/returns",
                "intent": "DAMAGED_OR_WRONG_ITEM",
                "resolution_act": "LINK_RETURNS",
                "rrf_score": 0.7,
                "dense_score": 0.8,
                "bm25_score": 0.5,
            },
            {
                "reply_id": "3",
                "customer_text": "Kindle not working",
                "reply_text": "Try restarting your device",
                "intent": "PRODUCT_AND_TECHNICAL",
                "resolution_act": "TROUBLESHOOT_DEVICE",
                "rrf_score": 0.6,
                "dense_score": 0.6,
                "bm25_score": 0.7,
            },
        ])
        
        # Mock the encoder
        from unittest.mock import MagicMock
        self.mock_encoder = MagicMock()
        self.mock_encoder.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ], dtype=np.float32)
        
        self.retriever = HistoricalResolutionRetriever(top_k=2, rrf_k=20, dense_encoder=self.mock_encoder)
        self.retriever.build_index(self.test_records)

    def test_build_index(self):
        """Index should be built successfully."""
        assert self.retriever.records is not None
        assert len(self.retriever.records) == 3
        assert self.retriever.bm25 is not None

    def test_retrieve_similar_no_filter(self):
        """Retrieval without intent filter should return top-k."""
        query = "package late"
        results = self.retriever.retrieve_similar(query, top_k=2)
        assert len(results) == 2
        assert all("reply_id" in r for r in results)
        assert all("rrf_score" in r for r in results)

    def test_retrieve_similar_with_intent_filter(self):
        """Retrieval with intent filter should only return matching intent."""
        query = "package late"
        results = self.retriever.retrieve_similar(query, intent_filter="ORDER_TRACKING_AND_DELAY", top_k=1)
        assert len(results) <= 1
        for r in results:
            assert r["intent"] == "ORDER_TRACKING_AND_DELAY"

    def test_retrieve_similar_empty_filter(self):
        """Filter with no matches should return empty list."""
        results = self.retriever.retrieve_similar("query", intent_filter="NONEXISTENT_INTENT", top_k=1)
        assert results == []

    def test_rrf_score_calculation(self):
        """RRF scores should be calculated and included."""
        results = self.retriever.retrieve_similar("package late", top_k=3)
        for r in results:
            assert "rrf_score" in r
            assert isinstance(r["rrf_score"], float)
            assert 0 <= r["rrf_score"] <= 1

    def test_dense_score_calculation(self):
        """Dense scores should be calculated."""
        results = self.retriever.retrieve_similar("package late", top_k=3)
        for r in results:
            assert "dense_score" in r
            assert isinstance(r["dense_score"], float)

    def test_bm25_score_calculation(self):
        """BM25 scores should be calculated."""
        results = self.retriever.retrieve_similar("package late", top_k=3)
        for r in results:
            assert "bm25_score" in r
            assert isinstance(r["bm25_score"], float)

    def test_top_k_respected(self):
        """Should not return more than top_k results."""
        results = self.retriever.retrieve_similar("package late", top_k=1)
        assert len(results) == 1
        
        results = self.retriever.retrieve_similar("package late", top_k=5)
        assert len(results) <= 3  # Limited by dataset size

    def test_empty_query(self):
        """Empty query should return empty list."""
        results = self.retriever.retrieve_similar("", top_k=3)
        assert results == []


class TestRetrieverEdgeCases:
    """Edge case tests for retriever."""

    def test_empty_index(self):
        """Empty index should handle gracefully."""
        from unittest.mock import MagicMock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[]], dtype=np.float32)
        
        retriever = HistoricalResolutionRetriever(top_k=3, rrf_k=20, dense_encoder=mock_encoder)
        retriever.build_index(pd.DataFrame())
        results = retriever.retrieve_similar("query", top_k=3)
        assert results == []

    def test_single_record(self):
        """Single record should work."""
        from unittest.mock import MagicMock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
        
        records = pd.DataFrame([{
            "reply_id": "1",
            "customer_text": "Test query here",
            "reply_text": "Test reply here",
            "intent": "ORDER_TRACKING_AND_DELAY",
            "resolution_act": "LINK_ORDERS",
            "rrf_score": 0.5,
            "dense_score": 0.5,
            "bm25_score": 0.5,
        }])
        
        retriever = HistoricalResolutionRetriever(top_k=3, rrf_k=20, dense_encoder=mock_encoder)
        retriever.build_index(records)
        results = retriever.retrieve_similar("test", top_k=3)
        assert len(results) == 1
        assert results[0]["reply_id"] == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])