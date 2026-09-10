"""
Script to extract and format False Auto-Handles and False Escalations
from the final evaluation run.
"""
import json

def extract():
    with open("results/agent_detailed_predictions.json", "r", encoding="utf-8") as f:
        preds = json.load(f)

    with open("data/golden_test_set.csv", "r", encoding="utf-8") as f:
        import pandas as pd
        test_df = pd.read_csv("data/golden_test_set.csv")
        test_ids = set(test_df["id"].tolist())

    false_autos_all = [p for p in preds if p["gold_escalate"] == "ESCALATE" and p["pred_escalate"] == "AUTO_HANDLE"]
    false_autos_test = [p for p in false_autos_all if p["id"] in test_ids]

    false_escs_all = [p for p in preds if p["gold_escalate"] == "AUTO_HANDLE" and p["pred_escalate"] == "ESCALATE"]
    false_escs_test = [p for p in false_escs_all if p["id"] in test_ids]

    print("=================================================================")
    print(f"FALSE AUTO-HANDLES (Total: {len(false_autos_all)}, In Test Set: {len(false_autos_test)})")
    print("=================================================================")
    for fa in false_autos_all:
        in_test = "[HELD-OUT TEST SET]" if fa["id"] in test_ids else "[DEV SET]"
        sim = fa["retrieved_examples"][0]["similarity"] if fa["retrieved_examples"] else 0.0
        print(f"Case ID #{fa['id']} {in_test}")
        print(f"  Customer Message:   \"{fa['customer_message']}\"")
        print(f"  Gold Intent:        {fa['gold_intent']}")
        print(f"  Predicted Intent:   {fa['pred_intent']}")
        print(f"  Top Similarity:     {sim:.3f}")
        print(f"  Decision Reason:    {fa['decision_reason']}")
        print(f"  Generated Reply:    \"{fa['reply']}\"")
        print(f"  Expected Resolution:{fa['expected_resolution']}")
        print("-" * 65)

    print("\n=================================================================")
    print(f"FALSE ESCALATIONS SAMPLE (Total: {len(false_escs_all)}, In Test Set: {len(false_escs_test)})")
    print("=================================================================")
    for fe in false_escs_test[:8]:
        sim = fe["retrieved_examples"][0]["similarity"] if fe["retrieved_examples"] else 0.0
        print(f"Case ID #{fe['id']} [HELD-OUT TEST SET]")
        print(f"  Customer Message:   \"{fe['customer_message']}\"")
        print(f"  Gold Intent:        {fe['gold_intent']}")
        print(f"  Predicted Intent:   {fe['pred_intent']}")
        print(f"  Top Similarity:     {sim:.3f}")
        print(f"  Decision Reason:    {fe['decision_reason']}")
        print("-" * 65)

if __name__ == "__main__":
    extract()
