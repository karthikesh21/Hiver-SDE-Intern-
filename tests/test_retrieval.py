"""
Unit tests for Historical Conversation Resolution Retriever.
"""

import pytest
from src.retriever import HistoricalRetriever

@pytest.fixture(scope="module")
def retriever():
    r = HistoricalRetriever(top_k=3)
    r.build_or_load_index()
    return r

def test_retriever_load(retriever):
    assert len(retriever.corpus_records) > 0
    assert retriever.corpus_embeddings is not None
    assert len(retriever.corpus_records) == retriever.corpus_embeddings.shape[0]

def test_retriever_top_k(retriever):
    query = "Where is my delivered package?"
    hits = retriever.retrieve(query, top_k=3)
    assert len(hits) == 3
    # Verify similarity ordering (descending)
    assert hits[0]["similarity"] >= hits[1]["similarity"] >= hits[2]["similarity"]
    for hit in hits:
        assert "conversation_id" in hit
        assert "similarity" in hit
        assert "resolution" in hit
        assert hit["similarity"] <= 1.0

def test_retriever_zero_leakage_exclusion(retriever):
    query = "Where is my package?"
    initial_hits = retriever.retrieve(query, top_k=3)
    assert len(initial_hits) > 0
    
    # Exclude the top retrieved conversation ID
    prohibited_id = initial_hits[0]["conversation_id"]
    new_hits = retriever.retrieve(query, exclude_conversation_ids={prohibited_id}, top_k=3)
    
    retrieved_cids = [h["conversation_id"] for h in new_hits]
    assert prohibited_id not in retrieved_cids
