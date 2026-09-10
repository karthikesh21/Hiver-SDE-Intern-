"""
Unit and integration tests for Main CustomerSupportAgent and output schema validation.
"""

import pytest
from src.agent import CustomerSupportAgent

@pytest.fixture(scope="module")
def agent():
    return CustomerSupportAgent().initialize()

def test_agent_output_schema(agent):
    msg = "How do I return my shoes for a refund?"
    output = agent.process_message(msg)
    
    # Required keys according to assignment spec
    assert "intent" in output
    assert "intent_confidence" in output
    assert "retrieved_examples" in output
    assert "reply" in output
    assert "decision" in output
    assert "decision_reason" in output
    
    # Value types and constraints
    assert isinstance(output["intent"], str)
    assert 0.0 <= output["intent_confidence"] <= 1.0
    assert output["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert isinstance(output["decision_reason"], str) and len(output["decision_reason"]) > 5
    assert isinstance(output["reply"], str) and len(output["reply"]) > 10
    assert isinstance(output["retrieved_examples"], list)
    
    # Retrieved example schema
    if output["retrieved_examples"]:
        first = output["retrieved_examples"][0]
        assert "similarity" in first
        assert "resolution" in first
        assert "conversation_id" in first

def test_agent_prohibited_leakage_filter(agent):
    msg = "Where is my delivered package?"
    out_initial = agent.process_message(msg)
    if out_initial["retrieved_examples"]:
        top_cid = out_initial["retrieved_examples"][0]["conversation_id"]
        out_filtered = agent.process_message(msg, exclude_conversation_ids={top_cid})
        retrieved_ids = [ex["conversation_id"] for ex in out_filtered["retrieved_examples"]]
        assert top_cid not in retrieved_ids
