"""
Historical Conversation Resolution Retriever:
Embeds customer problem descriptions and retrieves grounded historical resolutions
using semantic vector search with cosine similarity and strict leakage isolation.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config import (
    HISTORICAL_RESOLUTION_PATH,
    PROCESSED_DATA_DIR,
    RETRIEVAL_TOP_K,
    RANDOM_SEED
)

EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_DIR / "historical_embeddings.npy"
METADATA_CACHE_PATH = PROCESSED_DATA_DIR / "historical_metadata.json"

class HistoricalRetriever:
    """
    Semantic vector retriever with SentenceTransformer ('all-MiniLM-L6-v2')
    and TF-IDF fallback, enforcing strict evaluation set leakage isolation.
    """
    def __init__(self, top_k: int = RETRIEVAL_TOP_K, model_name: str = "all-MiniLM-L6-v2"):
        self.top_k = top_k
        self.model_name = model_name
        self.encoder = None
        self.corpus_records: List[Dict[str, Any]] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def _init_encoder(self):
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[Retriever] SentenceTransformer initialization warning: {e}. Falling back to TF-IDF.")
                self.encoder = None
                
    def build_or_load_index(self, force_rebuild: bool = False):
        """Build or load pre-computed embeddings and metadata."""
        if not force_rebuild and EMBEDDINGS_CACHE_PATH.exists() and METADATA_CACHE_PATH.exists():
            try:
                self.corpus_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
                with open(METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
                    self.corpus_records = json.load(f)
                self._init_encoder()
                return self
            except Exception as e:
                print(f"[Retriever] Cache load failed ({e}), rebuilding index...")
                
        self.corpus_records = []
        with open(HISTORICAL_RESOLUTION_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.corpus_records.append(json.loads(line))
                    
        texts = [r["customer_message"] for r in self.corpus_records]
        
        self._init_encoder()
        if self.encoder is not None:
            print(f"[Retriever] Encoding {len(texts)} historical resolutions with {self.model_name}...")
            embeddings = self.encoder.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self.corpus_embeddings = embeddings
            np.save(EMBEDDINGS_CACHE_PATH, embeddings)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.corpus_embeddings = self.tfidf_matrix.toarray()
            np.save(EMBEDDINGS_CACHE_PATH, self.corpus_embeddings)
            
        with open(METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.corpus_records, f)
            
        print(f"[Retriever] Index ready with {len(self.corpus_records)} historical cases.")
        return self

    def retrieve(
        self,
        query: str,
        intent_filter: Optional[str] = None,
        exclude_conversation_ids: Optional[Set[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k historical resolution records matching the query.
        Ensures strict zero-leakage by filtering out excluded IDs.
        """
        if self.corpus_embeddings is None:
            self.build_or_load_index()
            
        k = top_k or self.top_k
        exclude_ids = exclude_conversation_ids or set()
        
        # Encode query
        if self.encoder is not None:
            query_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            similarities = np.dot(self.corpus_embeddings, query_emb.T).flatten()
        elif self.tfidf_vectorizer is not None:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        else:
            self._init_encoder()
            query_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            similarities = np.dot(self.corpus_embeddings, query_emb.T).flatten()
            
        # Rank by descending similarity
        ranked_indices = np.argsort(-similarities)
        
        results = []
        for idx in ranked_indices:
            rec = self.corpus_records[idx]
            cid = rec.get("conversation_id", "")
            
            # Strict Leakage Guard: skip any prohibited conversation ID
            if cid in exclude_ids:
                continue
                
            # Optional intent filter
            if intent_filter and rec.get("intent") != intent_filter:
                continue
                
            score = float(similarities[idx])
            results.append({
                "conversation_id": cid,
                "similarity": round(score, 4),
                "intent": rec.get("intent", ""),
                "customer_message": rec.get("customer_message", ""),
                "brand_response": rec.get("brand_response", ""),
                "resolution": rec.get("resolution", "")
            })
            
            if len(results) >= k:
                break
                
        return results

if __name__ == "__main__":
    retriever = HistoricalRetriever()
    retriever.build_or_load_index()
    q = "My package says delivered but I can't find it on my porch"
    hits = retriever.retrieve(q, top_k=3)
    print(f"Top 3 retrieved hits for '{q}':")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [Sim: {h['similarity']:.3f}] Intent: {h['intent']} | Resolution: {h['resolution'][:80]}...")
