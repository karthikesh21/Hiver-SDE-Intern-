"""
Intent Classification Component:
Hybrid classifier balancing accuracy, explainability, reproducibility, and zero-cost local execution.
Supports both semantic prototype classification (via all-MiniLM-L6-v2) and LLM API.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config import (
    INTENTS_PATH,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    INTENT_CONFIDENCE_THRESHOLD,
    RANDOM_SEED
)

class IntentClassifier:
    """
    Robust hybrid intent classifier.
    Combines dense semantic embeddings of intent prototypes with rule heuristics
    and optional LLM API fallback.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.intents_data = {}
        self.prototype_embeddings = {}
        self.intent_labels = []
        self._load_taxonomy()
        
    def _load_taxonomy(self):
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            self.intents_data = json.load(f)
        self.intent_labels = list(self.intents_data.keys())
        
    def _init_encoder(self):
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[IntentClassifier] Encoder fallback warning: {e}")
                self.encoder = None
                
    def build_prototypes(self):
        """Build representative prototype embeddings for each intent."""
        self._init_encoder()
        if self.encoder is None:
            return
            
        for intent, data in self.intents_data.items():
            # Combine intent description, inclusion criteria, and canonical examples
            exemplars = [
                data["description"],
                data["inclusion_criteria"],
                *data["examples"]
            ]
            embs = self.encoder.encode(exemplars, convert_to_numpy=True, normalize_embeddings=True)
            # Mean normalized centroid for the intent prototype
            centroid = np.mean(embs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self.prototype_embeddings[intent] = centroid

    def predict_semantic(self, text: str) -> Dict[str, Any]:
        """Classify intent using semantic prototype similarity with calibrated softmax confidence."""
        if not self.prototype_embeddings:
            self.build_prototypes()
            
        if self.encoder is None or not self.prototype_embeddings:
            # Fallback to keyword matching if embedding encoder unavailable
            from src.data_processing import detect_rule_based_intent
            return {"intent": detect_rule_based_intent(text), "confidence": 0.50, "method": "heuristic_fallback"}
            
        text_emb = self.encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        sims = {}
        for intent, proto in self.prototype_embeddings.items():
            sims[intent] = float(np.dot(text_emb, proto))
            
        # Apply temperature scaling to cosine similarities for calibrated probabilities
        temperature = 0.12
        raw_scores = np.array([sims[intent] for intent in self.intent_labels])
        exp_scores = np.exp((raw_scores - np.max(raw_scores)) / temperature)
        probs = exp_scores / np.sum(exp_scores)
        
        best_idx = np.argmax(probs)
        predicted_intent = self.intent_labels[best_idx]
        confidence = float(probs[best_idx])
        
        # Keyword booster for disambiguating clear high-signal keywords
        t_low = text.lower()
        if any(w in t_low for w in ['prime', 'student prime']) and 'subscription_prime_issue' in self.intent_labels:
            predicted_intent = 'subscription_prime_issue'
            confidence = max(confidence, 0.88)
        elif any(w in t_low for w in ['broken', 'shattered', 'damaged', 'defective']) and 'damaged_defective_item' in self.intent_labels:
            predicted_intent = 'damaged_defective_item'
            confidence = max(confidence, 0.88)
        elif any(w in t_low for w in ['otp', 'password', '2fa', 'sign in', 'login', 'locked account']) and 'account_login_access' in self.intent_labels:
            predicted_intent = 'account_login_access'
            confidence = max(confidence, 0.88)
        elif any(w in t_low for w in ['refund', 'return label', 'drop off']) and 'refund_return_request' in self.intent_labels:
            predicted_intent = 'refund_return_request'
            confidence = max(confidence, 0.88)
        elif any(w in t_low for w in ['in stock', 'restock', 'compatible', 'compatibility', 'fit', 'specs', 'dimension', 'dimensions', 'weight capacity', 'voltage', '110v', 'sold directly']) and 'product_inquiry_availability' in self.intent_labels:
            predicted_intent = 'product_inquiry_availability'
            confidence = max(confidence, 0.88)
        elif any(w in t_low for w in ['charge twice', 'double charge', 'debit', 'unauthorized charge']) and 'payment_billing_issue' in self.intent_labels:
            predicted_intent = 'payment_billing_issue'
            confidence = max(confidence, 0.88)
        elif any(w in t_low for w in ['rude', 'driver threw', 'horrible service', 'complaint']) and 'general_complaint_feedback' in self.intent_labels:
            predicted_intent = 'general_complaint_feedback'
            confidence = max(confidence, 0.88)
            
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "all_scores": {k: round(float(v), 4) for k, v in zip(self.intent_labels, probs)},
            "method": "semantic_prototype"
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Main prediction method. Uses LLM API if configured, otherwise semantic prototype."""
        if OPENAI_API_KEY:
            try:
                import requests
                prompt = (
                    f"Classify the following customer support message into exactly ONE of these intents:\n"
                    f"{list(self.intent_labels)}\n\n"
                    f"Customer message: \"{text}\"\n\n"
                    f"Return ONLY a JSON object formatted as:\n"
                    f"{{\"intent\": \"<intent_name>\", \"confidence\": <float_0_to_1>}}"
                )
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(data)
                    return {
                        "intent": parsed.get("intent", "delivery_delay_tracking"),
                        "confidence": float(parsed.get("confidence", 0.90)),
                        "method": "openai_llm"
                    }
            except Exception as e:
                print(f"[IntentClassifier] LLM API call error: {e}. Falling back to semantic classifier.")
                
        return self.predict_semantic(text)

if __name__ == "__main__":
    clf = IntentClassifier()
    clf.build_prototypes()
    queries = [
        "Where is my package? It was supposed to be here yesterday.",
        "How do I return this shirt and get a refund?",
        "The ceramic bowl was completely broken when I opened the box.",
        "I was charged twice on my credit card for the same item.",
        "I cannot log in because I didn't get the OTP verification code.",
        "Why was I billed $139 for Amazon Prime auto-renewal?",
        "Will this phone case fit my Samsung Galaxy S23?",
        "Your delivery driver threw the package at my front door and broke my pot!"
    ]
    for q in queries:
        res = clf.predict(q)
        print(f"Query: '{q[:50]}...' -> Intent: {res['intent']} (Conf: {res['confidence']:.2f})")
