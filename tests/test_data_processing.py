"""
Unit tests for data processing and tweet cleaning functions.
"""

import pytest
from src.data_processing import clean_tweet_text, parse_turns, detect_rule_based_intent

def test_clean_tweet_text_entities():
    raw = "@AmazonHelp &amp; @user123 My package didn't arrive... Check https://t.co/abc1234"
    cleaned = clean_tweet_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "@user123" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "[link]" in cleaned
    assert "..." in cleaned

def test_clean_tweet_text_empty():
    assert clean_tweet_text("") == ""
    assert clean_tweet_text(None) == ""

def test_parse_turns_basic():
    convo = """Customer: Where is my order #123?
Support: We can check that for you! Please check your tracking link: [link] ^CC"""
    cust, supp, res, turns = parse_turns(convo)
    assert "Where is my order #123?" in cust
    assert "We can check that for you" in supp
    assert "tracking link" in res
    assert turns == 2

def test_detect_rule_based_intent():
    assert detect_rule_based_intent("My glass teapot arrived shattered into pieces") == "damaged_defective_item"
    assert detect_rule_based_intent("Where is my package? The courier is delayed") == "delivery_delay_tracking"
    assert detect_rule_based_intent("How do I return this item for a refund?") == "refund_return_request"
    assert detect_rule_based_intent("I was charged twice on my credit card") == "payment_billing_issue"
    assert detect_rule_based_intent("I am locked out because of 2FA verification code") == "account_login_access"
