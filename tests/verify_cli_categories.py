"""Test representative query categories through the AI Customer Support Agent."""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.agent import CustomerSupportAgent

def main():
    agent = CustomerSupportAgent()
    queries = [
        ("Delivery Issue", "Where is my order? Tracking says in transit for 5 days"),
        ("Refund Issue", "I returned the shoes a week ago, when will I get my refund credited to my bank?"),
        ("Payment Issue", "Why was I charged twice for my subscription this month?"),
        ("Account Issue", "My account got locked and I cannot sign in with my two factor code"),
        ("Product Question", "Do you have this laptop case available in blue color?"),
        ("Ambiguous Complaint", "This is the worst service ever, nothing works right anymore!"),
        ("Security-Sensitive Issue", "Someone hacked into my account and placed an order with my credit card")
    ]
    for cat, q in queries:
        res = agent.process_message(q)
        print(f"=== {cat} ===")
        print(f"Message:         {q}")
        print(f"Intent:          {res['intent']} (Confidence: {res['intent_confidence']:.2f})")
        print(f"Decision:        {res['decision']}")
        print(f"Decision Reason: {res['decision_reason']}")
        print(f"Reply:           {res['reply']}")
        print("-" * 75)

if __name__ == "__main__":
    main()
