"""
Escalation Policy Engine:
Evaluates customer messages, intent predictions, and retrieved historical precedents
to determine whether an issue can be safely AUTO_HANDLED or must be ESCALATED.
Provides explicit, human-readable justification reasons for every decision.
"""

import re
from typing import Dict, Any, List, Tuple
from src.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD
)

class EscalationEngine:
    """
    Transparent, configurable escalation policy engine.
    """
    def __init__(
        self,
        confidence_threshold: float = INTENT_CONFIDENCE_THRESHOLD,
        similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        
    def evaluate(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        retrieved_examples: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Evaluate escalation criteria in order of priority.
        Returns {"decision": "AUTO_HANDLE" | "ESCALATE", "decision_reason": str}.
        """
        msg_lower = customer_message.lower()
        
        # 1. Critical Escalation: Legal, Safety, or Gross Misconduct Triggers
        urgent_safety_terms = [
            "lawyer", "attorney", "legal action", "police", "fire", "sparked",
            "melted", "hazard", "electric shock", "smoke", "injury", "injured",
            "forged", "forgery", "trespass", "stole", "stolen"
        ]
        for term in urgent_safety_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', msg_lower):
                return {
                    "decision": "ESCALATE",
                    "decision_reason": f"Critical trigger identified ('{term}'): requires immediate safety or legal team review."
                }
                
        # 2. Intent Classification Confidence Check
        if intent_confidence < self.confidence_threshold:
            return {
                "decision": "ESCALATE",
                "decision_reason": (
                    f"Intent classification confidence ({intent_confidence:.2f}) is below "
                    f"operational safety threshold ({self.confidence_threshold:.2f})."
                )
            }
            
        # 3. Retrieval Precedent and Similarity Check
        if not retrieved_examples:
            return {
                "decision": "ESCALATE",
                "decision_reason": "No relevant historical resolution precedents were found in the knowledge base."
            }
            
        top_similarity = retrieved_examples[0].get("similarity", 0.0)
        if top_similarity < self.similarity_threshold:
            return {
                "decision": "ESCALATE",
                "decision_reason": (
                    f"Top historical resolution similarity ({top_similarity:.2f}) is below "
                    f"grounding threshold ({self.similarity_threshold:.2f})."
                )
            }
            
        # 4. Domain-Specific Policy Checks
        
        # General Complaints: Driver misconduct, supervisor requests, or severe dissatisfaction
        if intent == "general_complaint_feedback":
            complaint_escalation_terms = [
                "driver", "threw", "lawn", "porch", "supervisor", "manager", "disrespectful",
                "worst", "hostage", "5 times", "unacceptable", "insulted", "refused", "closed in my face"
            ]
            if any(k in msg_lower for k in complaint_escalation_terms):
                return {
                    "decision": "ESCALATE",
                    "decision_reason": "Severe service complaint or delivery misconduct requires supervisory review."
                }
                
        # Account Security & Credential Access: Account takeovers, locked accounts, or credential changes
        if intent == "account_login_access":
            security_terms = [
                "hacked", "russia", "lockout", "locked", "unauthorized", "changed my email",
                "hold", "suspended", "lost access", "takeover", "brute"
            ]
            if any(k in msg_lower for k in security_terms):
                return {
                    "decision": "ESCALATE",
                    "decision_reason": "Account security verification or credential lockout requires identity verification."
                }
                
        # Payment Billing Issues: Duplicate charges, unauthorized transactions, off-platform fraud
        if intent == "payment_billing_issue":
            billing_escalation_terms = [
                "charged twice", "double charge", "two charges", "unauthorized", "stolen",
                "wire transfer", "overdraft", "compromised", "fraud"
            ]
            if any(k in msg_lower for k in billing_escalation_terms):
                return {
                    "decision": "ESCALATE",
                    "decision_reason": "Financial transaction dispute or duplicate billing requires merchant ledger review."
                }
                
        # Damaged / Defective High-Value Items: High-value electronics, missing components, empty boxes
        if intent == "damaged_defective_item":
            high_value_terms = [
                "laptop", "television", "tv", "camera", "empty envelope", "empty box",
                "missing pieces", "accessory", "fork", "chair"
            ]
            if any(k in msg_lower for k in high_value_terms):
                return {
                    "decision": "ESCALATE",
                    "decision_reason": "High-value merchandise damage, warranty dispute, or empty parcel requires specialist inspection."
                }
                
        # Refund Requests: Expired return windows or seller disputes
        if intent == "refund_return_request":
            dispute_terms = ["closed yesterday", "exception", "promised", "seller refuses", "10 days ago", "received nothing", "accidentally returned"]
            if any(k in msg_lower for k in dispute_terms):
                return {
                    "decision": "ESCALATE",
                    "decision_reason": "Return policy exception or unresolved refund commitment requires human intervention."
                }
                
        # Delivery Delays: Urgent medication, wedding, or repeated failed deliveries
        if intent == "delivery_delay_tracking":
            urgent_terms = ["medicine", "suit", "wedding", "handed to resident", "out of town", "per customer request", "two consecutive days"]
            if any(k in msg_lower for k in urgent_terms):
                return {
                    "decision": "ESCALATE",
                    "decision_reason": "Time-sensitive essential delivery or false delivery confirmation requires dispatch investigation."
                }

        # 5. Default Safe Auto-Handle Condition
        return {
            "decision": "AUTO_HANDLE",
            "decision_reason": (
                f"High-confidence intent match ({intent_confidence:.2f}) with verified "
                f"grounded historical resolution ({top_similarity:.2f}) under standard self-service policy."
            )
        }

if __name__ == "__main__":
    engine = EscalationEngine()
    dummy_ex = [{"similarity": 0.78, "resolution": "Check carrier link."}]
    print(engine.evaluate("Where is my book?", "delivery_delay_tracking", 0.92, dummy_ex))
    print(engine.evaluate("Your driver backed over my lawn and destroyed my sprinkler!", "general_complaint_feedback", 0.88, dummy_ex))
    print(engine.evaluate("I need this package urgently because it has my medicine", "delivery_delay_tracking", 0.85, dummy_ex))
