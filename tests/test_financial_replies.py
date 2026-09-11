"""
Unit tests for targeted financial and billing dispute reply generation and escalation safety.
Covers:
1. Duplicate charge
2. Unexpected charge
3. Unauthorized charge
4. Payment discrepancy
5. Charged but order missing
6. Normal non-financial customer query
"""

import pytest
from src.agent import CustomerSupportAgent
from src.reply_generator import GroundedReplyGenerator

@pytest.fixture(scope="module")
def agent():
    return CustomerSupportAgent().initialize()

@pytest.fixture(scope="module")
def generator():
    return GroundedReplyGenerator()

class TestFinancialDisputeReplies:
    
    def test_duplicate_charge(self, agent):
        """Case 1: Duplicate charge must escalate, acknowledge problem, and not diagnose authorization hold."""
        query = "I was charged twice for the same order."
        res = agent.process_message(query)
        
        assert res["decision"] == "ESCALATE", "Duplicate charge must trigger ESCALATE"
        assert "charged twice" in res["decision_reason"].lower()
        
        reply = res["reply"].lower()
        assert "charged twice" in reply, "Reply must directly acknowledge being charged twice"
        assert "authorization" not in reply, "Reply must not diagnose or question customer about authorization holds"
        assert "refund" not in reply or "promise" not in reply, "Reply must not promise an unverified refund"
        assert "http" not in reply and ".com" not in reply, "Reply must not fabricate external URLs"
        assert "your orders" in reply or "specialist" in reply, "Reply must guide customer to safe account review"

    def test_unexpected_charge(self, agent):
        """Case 2: Unexpected charge must escalate and give clear guidance."""
        query = "There is an unexpected charge of $45 on my bank statement from Amazon."
        res = agent.process_message(query)
        
        assert res["decision"] == "ESCALATE", "Unexpected charge must trigger ESCALATE"
        reply = res["reply"].lower()
        assert "unexpected charge" in reply or "charge" in reply
        assert "authorization" not in reply
        assert "http" not in reply and ".com" not in reply

    def test_unauthorized_charge(self, agent):
        """Case 3: Unauthorized charge must escalate and guide to secure account review."""
        query = "Someone made an unauthorized charge on my account with my credit card."
        res = agent.process_message(query)
        
        assert res["decision"] == "ESCALATE", "Unauthorized charge must trigger ESCALATE"
        reply = res["reply"].lower()
        assert "unauthorized charge" in reply or "charge" in reply
        assert "specialist" in reply or "your orders" in reply
        assert "authorization" not in reply

    def test_payment_discrepancy(self, agent):
        """Case 4: Payment discrepancy must escalate for human review."""
        query = "I noticed a payment discrepancy on my credit card statement for my order."
        res = agent.process_message(query)
        
        assert res["decision"] == "ESCALATE", "Payment discrepancy must trigger ESCALATE"
        reply = res["reply"].lower()
        assert "payment discrepancy" in reply or "charge" in reply
        assert "specialist" in reply or "your orders" in reply
        assert "authorization" not in reply

    def test_charged_but_order_missing(self, agent):
        """Case 5: Charged but order missing must escalate and acknowledge the situation."""
        query = "My credit card was charged but the order is missing from my account."
        res = agent.process_message(query)
        
        assert res["decision"] == "ESCALATE", "Charged without order must trigger ESCALATE"
        reply = res["reply"].lower()
        assert "charged" in reply
        assert "missing" in reply or "not received" in reply or "order" in reply
        assert "specialist" in reply or "your orders" in reply

    def test_normal_non_financial_query_unchanged(self, agent):
        """Case 6: Standard delivery delay query behavior remains unchanged."""
        query = "Where is my package? Tracking says in transit for 5 days"
        res = agent.process_message(query)
        
        assert res["intent"] == "delivery_delay_tracking"
        assert res["decision"] == "AUTO_HANDLE"
        assert res["intent_confidence"] >= 0.50
        reply = res["reply"].lower()
        assert "delivery" in reply or "order" in reply
        assert "charged twice" not in reply
