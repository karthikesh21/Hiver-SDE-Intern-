"""
Baseline models for intent classification and customer support response:
1. Trivial Baseline (Majority class intent + static fallback reply)
2. Simple ML Baseline (TF-IDF + Logistic Regression)
"""

import json
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.config import HISTORICAL_RESOLUTION_PATH, RANDOM_SEED

class TrivialBaseline:
    """
    Baseline 1: Trivial heuristic that always predicts the majority class intent
    ('delivery_delay_tracking') and emits a static template response.
    """
    def __init__(self, majority_intent: str = "delivery_delay_tracking"):
        self.majority_intent = majority_intent
        
    def predict(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": 0.25, # Random chance among classes or static nominal score
            "reply": "Thank you for contacting Amazon Help. Please check your tracking status in Your Orders or send us a DM with your order number so we can assist.",
            "decision": "AUTO_HANDLE",
            "decision_reason": "Trivial default auto-handle policy."
        }
        
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]


class SimpleMLBaseline:
    """
    Baseline 2: Traditional transparent ML classifier using TF-IDF n-grams
    paired with Logistic Regression.
    """
    def __init__(self, random_seed: int = RANDOM_SEED):
        self.vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            C=1.5,
            max_iter=500,
            random_state=random_seed,
            class_weight='balanced'
        )
        self.is_trained = False
        
    def fit(self, training_records: List[Dict[str, Any]]):
        texts = [r["customer_message"] for r in training_records]
        labels = [r["intent"] for r in training_records]
        
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        return self
        
    def fit_from_historical_data(self):
        records = []
        with open(HISTORICAL_RESOLUTION_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return self.fit(records)
        
    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_trained:
            self.fit_from_historical_data()
            
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        max_idx = probs.argmax()
        predicted_intent = str(self.classifier.classes_[max_idx])
        confidence = float(probs[max_idx])
        
        # Simple ML escalation policy: escalate if model is uncertain (< 0.40)
        decision = "AUTO_HANDLE" if confidence >= 0.40 else "ESCALATE"
        reason = f"TF-IDF Logistic Regression confidence {confidence:.2f} {'meets' if decision == 'AUTO_HANDLE' else 'below'} 0.40 threshold."
        
        reply = (
            f"Thank you for reaching out regarding your {predicted_intent.replace('_', ' ')}. "
            "We are checking our records and will assist you shortly. Please review your account or reply with details."
        )
        
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "reply": reply,
            "decision": decision,
            "decision_reason": reason
        }
        
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]

if __name__ == "__main__":
    ml_base = SimpleMLBaseline()
    ml_base.fit_from_historical_data()
    sample = "Where is my order? It was supposed to be here yesterday."
    res = ml_base.predict(sample)
    print("Simple ML sample prediction:", res)
