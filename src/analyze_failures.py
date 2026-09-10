"""
Script to extract and categorize real failure modes from evaluation results.
"""

import json
from collections import Counter

def analyze():
    with open("results/agent_detailed_predictions.json", "r", encoding="utf-8") as f:
        preds = json.load(f)
        
    intent_errors = [p for p in preds if p["gold_intent"] != p["pred_intent"]]
    esc_errors = [p for p in preds if p["gold_escalate"] != p["pred_escalate"]]
    false_auto_handles = [p for p in preds if p["gold_escalate"] == "ESCALATE" and p["pred_escalate"] == "AUTO_HANDLE"]
    unnecessary_esc = [p for p in preds if p["gold_escalate"] == "AUTO_HANDLE" and p["pred_escalate"] == "ESCALATE"]
    
    print(f"Total intent errors: {len(intent_errors)} / {len(preds)}")
    print(f"Total escalation errors: {len(esc_errors)} / {len(preds)}")
    print(f"False Auto-Handles: {len(false_auto_handles)}")
    print(f"Unnecessary Escalations: {len(unnecessary_esc)}")
    
    # Analyze confusion pairs
    pair_counts = Counter([(p["gold_intent"], p["pred_intent"]) for p in intent_errors])
    print("\nTop Intent Confusion Pairs:")
    for pair, cnt in pair_counts.most_common(6):
        print(f"  {pair[0]} -> {pair[1]}: {cnt} times")
        
    print("\n--- False Auto Handles Details ---")
    for f in false_auto_handles:
        print(f"ID {f['id']} [{f['gold_intent']}]: {f['customer_message']}")
        print(f"  Pred Intent: {f['pred_intent']}")
        print(f"  Reason: {f['decision_reason']}")
        
    print("\n--- Intent Misclassifications Sample ---")
    for ie in intent_errors[:6]:
        print(f"ID {ie['id']}: '{ie['customer_message']}'")
        print(f"  Gold: {ie['gold_intent']} | Pred: {ie['pred_intent']}")

if __name__ == "__main__":
    analyze()
