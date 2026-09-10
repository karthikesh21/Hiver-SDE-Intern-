"""
Unit tests for Escalation Policy Engine, threshold boundaries,
financial discrepancy safety rules, and validation of the frozen policy.
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from src.escalation import EscalationEngine
from src.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    RESULTS_DIR,
    DATA_DIR
)

@pytest.fixture
def calibrated_engine():
    return EscalationEngine()

def test_calibrated_defaults(calibrated_engine):
    assert calibrated_engine.confidence_threshold == 0.50
    assert calibrated_engine.similarity_threshold == 0.45

def test_safety_critical_triggers(calibrated_engine):
    dummy_ex = [{"similarity": 0.85, "resolution": "Standard resolution."}]
    res = calibrated_engine.evaluate(
        "Your phone charger sparked and caught fire on my rug!",
        "damaged_defective_item",
        0.90,
        dummy_ex
    )
    assert res["decision"] == "ESCALATE"
    assert "safety" in res["decision_reason"].lower() or "trigger" in res["decision_reason"].lower()

def test_financial_ledger_discrepancy_triggers(calibrated_engine):
    dummy_ex = [{"similarity": 0.85, "resolution": "Standard return."}]
    
    # 1. Partial refund dispute
    res1 = calibrated_engine.evaluate(
        "My refund was $20 less than what I paid on my order.",
        "refund_return_request",
        0.88,
        dummy_ex
    )
    assert res1["decision"] == "ESCALATE"
    assert "financial ledger dispute" in res1["decision_reason"].lower()

    # 2. Free trial dispute
    res2 = calibrated_engine.evaluate(
        "I was charged for Prime after signing up for what was advertised as a free trial.",
        "subscription_prime_issue",
        0.88,
        dummy_ex
    )
    assert res2["decision"] == "ESCALATE"

    # 3. Post-cancellation billing
    res3 = calibrated_engine.evaluate(
        "I cancelled Prime 2 weeks ago but was billed again today.",
        "subscription_prime_issue",
        0.88,
        dummy_ex
    )
    assert res3["decision"] == "ESCALATE"

def test_threshold_boundary_behavior(calibrated_engine):
    dummy_at_bound = [{"similarity": 0.45, "resolution": "Standard resolution."}]
    dummy_below_bound = [{"similarity": 0.449, "resolution": "Standard resolution."}]
    
    # Exactly at boundary (conf=0.50, sim=0.45) -> AUTO_HANDLE
    res_at = calibrated_engine.evaluate("Where is my order?", "delivery_delay_tracking", 0.50, dummy_at_bound)
    assert res_at["decision"] == "AUTO_HANDLE"
    
    # Slightly below similarity boundary -> ESCALATE
    res_below_sim = calibrated_engine.evaluate("Where is my order?", "delivery_delay_tracking", 0.50, dummy_below_bound)
    assert res_below_sim["decision"] == "ESCALATE"
    assert "similarity" in res_below_sim["decision_reason"].lower()
    
    # Slightly below confidence boundary -> ESCALATE
    res_below_conf = calibrated_engine.evaluate("Where is my order?", "delivery_delay_tracking", 0.499, dummy_at_bound)
    assert res_below_conf["decision"] == "ESCALATE"
    assert "confidence" in res_below_conf["decision_reason"].lower()

def test_safe_auto_handle(calibrated_engine):
    dummy_good = [{"similarity": 0.78, "resolution": "Check tracking link in Your Orders."}]
    res = calibrated_engine.evaluate(
        "Can you tell me the tracking number for my order?",
        "delivery_delay_tracking",
        0.88,
        dummy_good
    )
    assert res["decision"] == "AUTO_HANDLE"
    assert len(res["decision_reason"]) > 10

def test_final_policy_json_validity():
    policy_path = RESULTS_DIR / "final_policy.json"
    assert policy_path.exists()
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)
        
    assert policy["intent_confidence_threshold"] == 0.50
    assert policy["retrieval_similarity_threshold"] == 0.45
    assert policy["selection_method"] == "development_set_only"
    assert "safety_constraints" in policy

def test_no_accidental_test_set_tuning_leakage():
    dev_path = DATA_DIR / "golden_dev_set.csv"
    test_path = DATA_DIR / "golden_test_set.csv"
    assert dev_path.exists() and test_path.exists()
    
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    
    dev_ids = set(dev_df["id"].tolist())
    test_ids = set(test_df["id"].tolist())
    
    # Strictly zero overlap between dev set and held-out test set
    overlap = dev_ids.intersection(test_ids)
    assert len(overlap) == 0
    assert len(dev_df) == 140
    assert len(test_df) == 60
