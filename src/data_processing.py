"""
Data preprocessing, cleaning, and conversation reconstruction pipeline
for AmazonHelp Twitter customer support interactions.
"""

import re
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from src.config import (
    PROCESSED_DATA_DIR,
    HISTORICAL_RESOLUTION_PATH,
    RANDOM_SEED,
    INTENTS_PATH
)

def clean_tweet_text(text: str) -> str:
    """Clean Twitter handles, URLs, and excess whitespace while preserving context."""
    if not text:
        return ""
    # Normalize HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Remove @mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Normalize URLs
    text = re.sub(r'https?://\S+', '[link]', text)
    # Remove excessive punctuation repetitions
    text = re.sub(r'\.{3,}', '...', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_turns(raw_conversation: str) -> Tuple[str, str, str, int]:
    """
    Parse reconstructed conversation text into customer initial message,
    brand response, and key resolution action.
    """
    lines = [l.strip() for l in raw_conversation.split('\n') if l.strip()]
    cust_turns = []
    supp_turns = []
    
    for l in lines:
        if l.startswith('Customer:'):
            cleaned = clean_tweet_text(l.replace('Customer:', '').strip())
            if cleaned:
                cust_turns.append(cleaned)
        elif l.startswith('Support:'):
            cleaned = clean_tweet_text(l.replace('Support:', '').strip())
            if cleaned:
                supp_turns.append(cleaned)
                
    if not cust_turns or not supp_turns:
        return "", "", "", 0
        
    primary_customer_msg = cust_turns[0]
    # If customer had a follow-up clarification, incorporate concise context
    if len(cust_turns) > 1 and len(cust_turns[1].split()) > 3:
        primary_customer_msg = f"{cust_turns[0]} (Details: {cust_turns[1]})"
        
    brand_response = " ".join(supp_turns)
    
    # Extract resolution summary: prioritize the most substantive support reply
    substantive_supp = max(supp_turns, key=lambda s: len(s.split()))
    resolution = substantive_supp
    
    return primary_customer_msg, brand_response, resolution, len(lines)

def detect_rule_based_intent(text: str) -> str:
    """Rule-based intent heuristic classifier for historical data tagging."""
    t = text.lower()
    
    if any(k in t for k in ['broken', 'damaged', 'defective', 'wrong item', 'cracked', 'shattered', 'spilled', 'tear', 'hole']):
        return 'damaged_defective_item'
    if any(k in t for k in ['track', 'where is', 'delivery', 'deliver', 'courier', 'late', 'delayed', 'lost package', 'usps', 'ups', 'carrier', 'porch', 'transit', 'eta']):
        return 'delivery_delay_tracking'
    if any(k in t for k in ['refund', 'return', 'returns', 'return label', 'drop off', 'cancel order', 'cancelled order', 'send back', 'exchange']):
        return 'refund_return_request'
    if any(k in t for k in ['charge', 'charged', 'double charge', 'credit card', 'debit', 'declined', 'billing', 'invoice', 'gift card', 'balance']):
        return 'payment_billing_issue'
    if any(k in t for k in ['login', 'log in', 'password', 'sign in', 'locked', 'otp', 'verification code', '2fa', 'two-step', 'suspended', 'account access']):
        return 'account_login_access'
    if any(k in t for k in ['prime', 'membership', 'annual fee', 'subscription', 'prime video', 'student discount']):
        return 'subscription_prime_issue'
    if any(k in t for k in ['driver', 'rude', 'terrible', 'worst', 'horrible', 'complaint', 'manager', 'supervisor', 'unacceptable', 'disgusting', 'lawsuit']):
        return 'general_complaint_feedback'
    if any(k in t for k in ['in stock', 'compatible', 'specs', 'dimension', 'size', 'restock', 'when will you have', 'available', 'seller']):
        return 'product_inquiry_availability'
        
    return 'delivery_delay_tracking' # Default fallback for general retail support

def process_brand_dataset(max_samples: int = 5000) -> List[Dict[str, Any]]:
    """
    Process AmazonHelp conversations into structured historical resolutions.
    """
    print(f"Loading AmazonHelp conversations from dataset...")
    ds = load_dataset('TNE-AI/customer-support-on-twitter-conversation', split='train')
    
    random.seed(RANDOM_SEED)
    raw_amazon = []
    
    for row in ds:
        if row.get('company') == 'AmazonHelp':
            raw_amazon.append(row)
            if len(raw_amazon) >= max_samples * 2:
                break
                
    random.shuffle(raw_amazon)
    
    processed_records = []
    seen_messages = set()
    
    for r in raw_amazon:
        cid = r.get('conversation_id', '')
        raw_text = r.get('conversation', '')
        cust_msg, brand_resp, resolution, turns = parse_turns(raw_text)
        
        # Validation filter: meaningful customer message and brand resolution
        if len(cust_msg.split()) < 4 or len(resolution.split()) < 5:
            continue
            
        # Avoid duplicate customer messages
        msg_sig = cust_msg.lower()[:80]
        if msg_sig in seen_messages:
            continue
        seen_messages.add(msg_sig)
        
        intent = detect_rule_based_intent(cust_msg)
        
        rec = {
            "conversation_id": cid,
            "customer_message": cust_msg,
            "brand_response": brand_resp,
            "resolution": resolution,
            "intent": intent,
            "turn_count": turns
        }
        processed_records.append(rec)
        if len(processed_records) >= max_samples:
            break
            
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(HISTORICAL_RESOLUTION_PATH, "w", encoding="utf-8") as f:
        for rec in processed_records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Successfully processed and saved {len(processed_records)} historical interactions to {HISTORICAL_RESOLUTION_PATH}")
    return processed_records

if __name__ == "__main__":
    process_brand_dataset(max_samples=4000)
