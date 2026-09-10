"""
Diagnostic script to inspect escalation decisions across the 200 evaluation examples.
"""
import json
from collections import Counter

def diagnose():
    with open("results/agent_detailed_predictions.json", "r", encoding="utf-8") as f:
        preds = json.load(f)

    total = len(preds)
    auto_handled = sum(1 for p in preds if p["pred_escalate"] == "AUTO_HANDLE")
    escalated = sum(1 for p in preds if p["pred_escalate"] == "ESCALATE")
    gold_auto = sum(1 for p in preds if p["gold_escalate"] == "AUTO_HANDLE")
    gold_esc = sum(1 for p in preds if p["gold_escalate"] == "ESCALATE")

    false_auto = [p for p in preds if p["gold_escalate"] == "ESCALATE" and p["pred_escalate"] == "AUTO_HANDLE"]
    false_esc = [p for p in preds if p["gold_escalate"] == "AUTO_HANDLE" and p["pred_escalate"] == "ESCALATE"]

    print(f"Total: {total}")
    print(f"Gold: {gold_auto} AUTO_HANDLE, {gold_esc} ESCALATE")
    print(f"Pred: {auto_handled} AUTO_HANDLE, {escalated} ESCALATE")
    print(f"False Auto-Handles: {len(false_auto)}")
    print(f"False Escalations: {len(false_esc)}")

    reasons = Counter()
    intents = Counter()

    for p in false_esc:
        intents[p["gold_intent"]] += 1
        r = p["decision_reason"]
        if "below grounding threshold" in r:
            reasons["similarity_below_threshold"] += 1
        elif "below operational safety threshold" in r:
            reasons["intent_confidence_below_threshold"] += 1
        elif "Critical trigger identified" in r:
            reasons["critical_safety_trigger"] += 1
        elif "Severe service complaint" in r:
            reasons["complaint_rule"] += 1
        elif "Account security verification" in r:
            reasons["account_security_rule"] += 1
        elif "Financial transaction dispute" in r:
            reasons["billing_rule"] += 1
        elif "High-value merchandise" in r:
            reasons["damaged_high_value_rule"] += 1
        elif "Return policy exception" in r:
            reasons["refund_dispute_rule"] += 1
        elif "Time-sensitive essential delivery" in r:
            reasons["delivery_urgency_rule"] += 1
        else:
            reasons[r] += 1

    print("\nFalse Escalations by Reason:")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")

    print("\nFalse Escalations by Gold Intent:")
    for k, v in intents.most_common():
        print(f"  {k}: {v}")

    print("\nSample similarity_below_threshold queries (showing max similarity):")
    sim_cases = [p for p in false_esc if "below grounding threshold" in p["decision_reason"]]
    sim_values = [p["retrieved_examples"][0]["similarity"] for p in sim_cases if p["retrieved_examples"]]
    print(f"  Mean similarity of similarity-escalated cases: {sum(sim_values)/len(sim_values):.3f}")
    print(f"  Min similarity: {min(sim_values):.3f}, Max similarity: {max(sim_values):.3f}")
    
    print("\nTop 10 cases blocked by similarity < 0.60:")
    for p in sim_cases[:10]:
        sim = p["retrieved_examples"][0]["similarity"] if p["retrieved_examples"] else 0.0
        print(f"  [Sim: {sim:.3f}] Intent: {p['gold_intent']} | Msg: {p['customer_message'][:80]}")

    print("\nConfidence of cases escalated due to low confidence:")
    conf_cases = [p for p in false_esc if "operational safety threshold" in p["decision_reason"]]
    import re
    confs = []
    for p in conf_cases:
        m = re.search(r"confidence \((\d+\.\d+)\)", p["decision_reason"])
        if m:
            confs.append(float(m.group(1)))
    if confs:
        print(f"  Count: {len(confs)}")
        print(f"  Mean: {sum(confs)/len(confs):.3f}")
        print(f"  Min: {min(confs):.3f}, Max: {max(confs):.3f}")
        print(f"  Cases with conf >= 0.40: {sum(1 for c in confs if c >= 0.40)}")
        print(f"  Cases with conf >= 0.50: {sum(1 for c in confs if c >= 0.50)}")
        print(f"  Cases with conf >= 0.55: {sum(1 for c in confs if c >= 0.55)}")

if __name__ == "__main__":
    diagnose()
