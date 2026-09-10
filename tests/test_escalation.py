"""
Unit tests for Escalation Policy Engine.
"""

import pytest
from src.escalation import EscalationEngine

@pytest.fixture
def engine():
    return EscalationEngine(confidence_threshold=0.65, similarity_threshold=0.60)

def test_safety_critical_triggers(engine):
    dummy_ex = [{"similarity": 0.85, "resolution": "Standard resolution."}]
    res = engine.evaluate("Your phone charger sparked and caught fire on my rug!", "damaged_defective_item", 0.90, dummy_ex)
    assert res["decision"] == "ESCALATE"
    assert "safety" in res["decision_reason"].lower() or "trigger" in res["decision_reason"].lower()

def test_low_confidence_escalation(engine):
    dummy_ex = [{"similarity": 0.80, "resolution": "Standard resolution."}]
    res = engine.evaluate("Some ambiguous text", "delivery_delay_tracking", 0.45, dummy_ex)
    assert res["decision"] == "ESCALATE"
    assert "below" in res["decision_reason"].lower()

def test_low_similarity_escalation(engine):
    dummy_low_sim = [{"similarity": 0.42, "resolution": "Standard resolution."}]
    res = engine.evaluate("Where is my shipment?", "delivery_delay_tracking", 0.90, dummy_low_sim)
    assert res["decision"] == "ESCALATE"
    assert "similarity" in res["decision_reason"].lower()

def test_safe_auto_handle(engine):
    dummy_good = [{"similarity": 0.78, "resolution": "Check tracking link in Your Orders."}]
    res = engine.evaluate("Can you tell me the tracking number for my order?", "delivery_delay_tracking", 0.92, dummy_good)
    assert res["decision"] == "AUTO_HANDLE"
    assert "decision_reason" in res and len(res["decision_reason"]) > 10
