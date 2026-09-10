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
3. NO FAKE LINKS: Do NOT fabricate URLs or hyperlinks. If a link is referenced in evidence, refer to it generically as "Your Orders" or "the Amazon Help link".
4. ESCALATION SAFETY: If the historical evidence does not provide a clear, safe resolution for the specific problem, advise the customer that their issue requires specialist review and guide them accordingly.

Retrieved Historical Examples:
{evidence_block}

Current Customer Message:
"{customer_message}"

Generate a professional customer support response:"""

class GroundedReplyGenerator:
    """
    Grounded response generator that enforces historical adherence and anti-hallucination rules.
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

    def generate_local_grounded_reply(
        self,
        customer_message: str,
        retrieved_examples: List[Dict[str, Any]],
        intent: str
    ) -> str:
        """
        Locally synthesizes a grounded support response derived directly from the
        top retrieved historical resolutions without requiring external API tokens.
        """
        if not retrieved_examples:
            return (
                "Thank you for contacting Amazon Help. We want to make sure your issue is handled accurately. "
                "Because your request requires specific account verification, our customer service team will look into this."
            )
            
        top_ex = retrieved_examples[0]
        top_resolution = top_ex.get("resolution", "").strip()
        top_sim = top_ex.get("similarity", 0.0)
        
        # Clean specific customer names or twitter noise from historical resolution
        cleaned_res = re.sub(r'^[A-Z][a-z]+,\s*', '', top_resolution) # Strip leading names like "Hi David, "
        cleaned_res = re.sub(r'\^[A-Z]{2,3}$', '', cleaned_res).strip() # Strip agent initials like "^CC"
        cleaned_res = cleaned_res.replace('[link]', 'the link in Your Orders')
        
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
        intent: str
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
                
        return self.generate_local_grounded_reply(customer_message, retrieved_examples, intent)

if __name__ == "__main__":
    gen = GroundedReplyGenerator()
    dummy_evidence = [{
        "customer_message": "Where is my book?",
        "resolution": "Please check your tracking link in Your Orders. If it does not arrive by 9 PM tonight, let us know and we will issue a replacement.",
        "similarity": 0.82
    }]
    reply = gen.generate("My package is delayed, when will it get here?", dummy_evidence, "delivery_delay_tracking")
    print("Generated Grounded Reply:\n", reply)
