"""
Grounded Reply Generator:
Generates customer-facing replies strictly grounded in retrieved historical resolutions,
preventing hallucinations of policies, amounts, or links.
Supports LLM API (OpenAI) with a deterministic, locally reproducible grounded template synthesizer.
"""

import re
from typing import List, Dict, Any, Optional
from src.config import OPENAI_API_KEY, OPENAI_MODEL

REPLY_PROMPT_TEMPLATE = """You are an AI customer support specialist for Amazon.
Draft a concise, empathetic, and professional reply to the customer message below.

CRITICAL INSTRUCTIONS:
1. GROUNDING: Use ONLY information supported by the historical examples provided below.
2. NO HALLUCINATION: Do NOT invent company policies, refund amounts, order details, or claim actions were completed when they were not.
3. NO FAKE LINKS: Do NOT fabricate URLs or hyperlinks. If a link is referenced in evidence, refer to it generically as "the support link in Your Orders" or "Your Orders".
4. ESCALATION SAFETY: If the historical evidence does not provide a clear, safe resolution for the specific problem, advise the customer that their issue requires specialist review and guide them accordingly.
5. FINANCIAL DISPUTES: For billing and charge disputes (duplicate charges, unauthorized charges, unexpected debits, payment discrepancies):
   - Directly acknowledge the customer's specific stated problem with empathy (e.g. "We understand your concern about being charged twice for the same order.").
   - Do NOT question or contradict the customer's claim (never suggest the charge is merely an authorization hold).
   - Do NOT guess causes or diagnose charges as pending authorizations.
   - Do NOT promise refunds or claim an account has already been reviewed or an action already taken.
   - Direct the customer to check Your Orders or contact an Amazon customer specialist with order details for account review.

Retrieved Historical Examples:
{evidence_block}

Current Customer Message:
"{customer_message}"

Generate a professional customer support response:"""

def sanitize_historical_text(text: str) -> str:
    """Clean Twitter noise, customer names, agent signatures, and multipart markers."""
    # Strip agent initials at end e.g. ^CC, ^HD, ^RA, ^LB
    cleaned = re.sub(r'\^[A-Z]{2,3}\b', '', text)
    # Strip multi-part tweet markers e.g. (1/2), (2/2), 1/2, 2/2
    cleaned = re.sub(r'\(?\b[1-3]/[2-3]\)?', '', cleaned)
    # Strip Twitter-specific phrases
    cleaned = re.sub(r'our (twitter|page) is (visible to )?public\b.*', '', cleaned, flags=re.IGNORECASE)
    # Strip leading greetings with names: "Hi David, ", "Hey Shiva! ", "Sorry to hear that, Shireen! "
    cleaned = re.sub(r'^(hi|hello|hey|dear)\s+[a-z]+[,\.!:]*\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^(sorry to hear that|apologies for the delay|i\'m sorry),\s+[a-z]+[,\.!:]*\s*', r'\1, ', cleaned, flags=re.IGNORECASE)
    # Strip trailing or comma names e.g. ", Will!" or ", Julie!" or ", Shiva."
    cleaned = re.sub(r',\s+[A-Z][a-z]+([!\.]|\s*$)', r'\1', cleaned)
    # Safe link replacement
    cleaned = cleaned.replace('[link]', 'the support link in Your Orders')
    # Collapse multiple spaces
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned

class GroundedReplyGenerator:
    """
    Grounded response generator that enforces historical adherence and anti-hallucination rules,
    with dedicated safe handling for financial and billing disputes.
    """
    def __init__(self, model_name: str = OPENAI_MODEL):
        self.model_name = model_name
        self.api_key = OPENAI_API_KEY
        
    def _format_evidence(self, retrieved_examples: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, ex in enumerate(retrieved_examples, 1):
            cust = ex.get("customer_message", "").strip()
            res = ex.get("resolution", "").strip()
            blocks.append(f"Example {i}:\nCustomer Issue: {cust}\nHistorical Resolution: {res}")
        return "\n\n".join(blocks) if blocks else "No historical evidence found."

    def is_financial_dispute(
        self,
        customer_message: str,
        intent: str = "",
        decision: Optional[str] = None
    ) -> bool:
        """Identify if inquiry represents a financial, billing, or charge dispute."""
        msg_lower = customer_message.lower()
        dispute_patterns = [
            "charged twice", "duplicate charge", "double charge", "two charges",
            "charged 2 times", "billed twice", "double billed", "charged multiple times",
            "payment taken multiple", "unexpected charge", "unrecognized charge",
            "unknown charge", "unauthorized charge", "fraudulent charge", "mystery charge",
            "extra charge", "incorrect billing", "billed wrong", "overcharged",
            "wrong amount", "payment discrepancy", "billing discrepancy",
            "charged but", "charged and", "paid but", "took my money", "money deducted",
            "charged without", "less than what i paid", "short by", "unauthorized transaction",
            "charged for an order", "charged for the same", "billed for the same"
        ]
        if any(p in msg_lower for p in dispute_patterns):
            return True
        if intent == "payment_billing_issue" and any(k in msg_lower for k in [
            "charge", "billed", "debit", "fee", "deducted", "paid", "money", "twice", "double", "unauthorized"
        ]):
            return True
        return False

    def generate_financial_dispute_reply(
        self,
        customer_message: str,
        retrieved_examples: List[Dict[str, Any]],
        decision: Optional[str] = "ESCALATE"
    ) -> str:
        """
        Synthesize an empathetic, safe, non-hallucinatory reply for financial disputes.
        Directly acknowledges the customer's specific problem, avoids speculative diagnoses
        (e.g., authorization holds), makes no unsupported refund promises, and guides customer
        to account review.
        """
        msg_lower = customer_message.lower()
        
        # 1. Directly and specifically acknowledge the customer's stated issue
        if re.search(r'\b(charged twice for the same order|charged twice for the same item)\b', msg_lower):
            acknowledgement = "We understand your concern about being charged twice for the same order."
        elif "subscription" in msg_lower and any(w in msg_lower for w in ["twice", "double", "two charges"]):
            acknowledgement = "We understand your concern about being charged twice for your subscription."
        elif re.search(r'\b(charged twice|billed twice|charged 2 times|billed 2 times)\b', msg_lower):
            acknowledgement = "We understand your concern about being charged twice."
        elif re.search(r'\b(duplicate charge|double charge|two charges|double billed)\b', msg_lower):
            acknowledgement = "We understand your concern regarding this duplicate charge."
        elif re.search(r'\b(charged multiple times|payment taken multiple times|taken multiple times)\b', msg_lower):
            acknowledgement = "We understand your concern about being charged multiple times."
        elif re.search(r'\b(unauthorized charge|unauthorized transaction|unauthorized payment|fraudulent charge)\b', msg_lower):
            acknowledgement = "We understand your concern regarding this unauthorized charge."
        elif re.search(r'\b(unexpected charge|unrecognized charge|unknown charge|mystery charge|extra charge)\b', msg_lower):
            acknowledgement = "We understand your concern regarding this unexpected charge."
        elif re.search(r'\bcharged\b', msg_lower) and re.search(r'\b(not received|haven\'t received|never received|missing|cancelled|canceled)\b', msg_lower):
            acknowledgement = "We understand your concern about being charged for an order you have not received."
        elif re.search(r'\b(payment discrepancy|billing discrepancy|incorrect billing|overcharged|billed wrong|wrong amount)\b', msg_lower):
            acknowledgement = "We understand your concern regarding this payment discrepancy."
        else:
            acknowledgement = "We understand your concern regarding this charge."

        # 2. Guidance next step (safe, transparent, customer-facing)
        if "unauthorized" in msg_lower or "fraud" in msg_lower:
            guidance = (
                "Please check Your Orders or contact an Amazon customer specialist immediately "
                "with your account details so they can investigate the unauthorized activity and assist you."
            )
        elif "missing" in msg_lower or "not received" in msg_lower or "cancel" in msg_lower:
            guidance = (
                "Please check Your Orders or contact an Amazon customer specialist with your "
                "order details so they can locate the transaction and assist you."
            )
        else:
            guidance = (
                "Please check Your Orders or contact an Amazon customer specialist with your "
                "order details so they can review the charges and help resolve the issue."
            )

        return f"{acknowledgement} {guidance}"

    def generate_local_grounded_reply(
        self,
        customer_message: str,
        retrieved_examples: List[Dict[str, Any]],
        intent: str,
        decision: Optional[str] = None
    ) -> str:
        """
        Locally synthesizes a grounded support response derived directly from the
        top retrieved historical resolutions without requiring external API tokens.
        """
        # Targeted rule: financial disputes require safe acknowledgment without speculative diagnosis
        if self.is_financial_dispute(customer_message, intent, decision):
            return self.generate_financial_dispute_reply(customer_message, retrieved_examples, decision)

        if not retrieved_examples:
            return (
                "Thank you for contacting Amazon Help. We want to make sure your issue is handled accurately. "
                "Because your request requires specific account verification, our customer service team will look into this."
            )
            
        top_ex = retrieved_examples[0]
        top_resolution = top_ex.get("resolution", "").strip()
        top_sim = top_ex.get("similarity", 0.0)
        
        # Clean specific customer names or twitter noise from historical resolution
        cleaned_res = sanitize_historical_text(top_resolution)
        
        # Synthesize professional grounded reply
        openings = {
            "delivery_delay_tracking": "We apologize for the delay with your delivery.",
            "refund_return_request": "We are glad to assist you with your return or refund.",
            "damaged_defective_item": "We are very sorry to hear that your item arrived in that condition.",
            "payment_billing_issue": "We understand your concern regarding this charge.",
            "account_login_access": "We apologize for the trouble accessing your account.",
            "subscription_prime_issue": "We are happy to clarify your Amazon Prime membership details.",
            "product_inquiry_availability": "Thank you for inquiring about this product.",
            "general_complaint_feedback": "We take your feedback very seriously and apologize for this experience."
        }
        
        opening = openings.get(intent, "Thank you for contacting Amazon Help.")
        
        if top_sim < 0.50:
            # Low similarity indicates unusual or ambiguous case
            return (
                f"{opening} To make sure this is resolved properly, please visit Your Orders or contact "
                "an Amazon customer specialist directly with your order details so we can investigate."
            )
            
        return f"{opening} {cleaned_res}"

    def generate(
        self,
        customer_message: str,
        retrieved_examples: List[Dict[str, Any]],
        intent: str,
        decision: Optional[str] = None,
        decision_reason: Optional[str] = None
    ) -> str:
        """
        Generate a customer-facing support reply using LLM API if available,
        falling back seamlessly to local grounded synthesis.
        """
        if self.api_key:
            try:
                import requests
                evidence_block = self._format_evidence(retrieved_examples)
                prompt = REPLY_PROMPT_TEMPLATE.format(
                    evidence_block=evidence_block,
                    customer_message=customer_message
                )
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 150
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    reply_text = resp.json()["choices"][0]["message"]["content"].strip()
                    return reply_text
            except Exception as e:
                print(f"[ReplyGenerator] LLM API call error: {e}. Falling back to local grounded synthesis.")
                
        return self.generate_local_grounded_reply(customer_message, retrieved_examples, intent, decision=decision)

if __name__ == "__main__":
    gen = GroundedReplyGenerator()
    dummy_evidence = [{
        "customer_message": "Where is my book?",
        "resolution": "Please check your tracking link in Your Orders. If it does not arrive by 9 PM tonight, let us know and we will issue a replacement.",
        "similarity": 0.82
    }]
    reply = gen.generate("My package is delayed, when will it get here?", dummy_evidence, "delivery_delay_tracking")
    print("Generated Grounded Reply:\n", reply)
