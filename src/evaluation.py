"""
Automated Evaluation Harness:
Comprehensively evaluates Intent Classification, Historical Retrieval, and Escalation Decisions
across Trivial Baseline, Simple ML Baseline, and Main AI Agent against the Golden Evaluation Set.
Generates metrics.json, metrics.csv, and confusion_matrix.png.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from src.config import (
    GOLDEN_SET_PATH,
    RESULTS_DIR,
    EVALUATION_DIR,
    INTENTS_PATH,
    RANDOM_SEED
)
from src.baselines import TrivialBaseline, SimpleMLBaseline
from src.agent import CustomerSupportAgent

def plot_and_save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    title: str,
    output_path: Path
):
    """Plot and save high-resolution confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='medium',
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
            
    fig.colorbar(cax)
    clean_labels = [l.replace('_', '\n') for l in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(clean_labels, rotation=45, ha='left', fontsize=9)
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

class EvaluationHarness:
    """
    Evaluation runner testing all systems on golden_set.csv.
    """
    def __init__(self, golden_path: Path = GOLDEN_SET_PATH):
        self.golden_path = golden_path
        self.golden_df = pd.read_csv(golden_path)
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            self.intents_data = json.load(f)
        self.intent_labels = list(self.intents_data.keys())
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
    def evaluate_intent_classification(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        macro_prec = precision_score(y_true, y_pred, labels=self.intent_labels, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, labels=self.intent_labels, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, labels=self.intent_labels, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=self.intent_labels, average='weighted', zero_division=0)
        
        per_class = classification_report(
            y_true, y_pred,
            labels=self.intent_labels,
            output_dict=True,
            zero_division=0
        )
        
        per_intent_metrics = {}
        for intent in self.intent_labels:
            intent_stats = per_class.get(intent, {})
            per_intent_metrics[intent] = {
                "precision": round(intent_stats.get("precision", 0.0), 4),
                "recall": round(intent_stats.get("recall", 0.0), 4),
                "f1_score": round(intent_stats.get("f1-score", 0.0), 4),
                "support": intent_stats.get("support", 0)
            }
            
        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_intent": per_intent_metrics
        }
        
    def evaluate_escalation(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        Escalation metrics. Pay special attention to False Auto-Handles
        (where human escalation was required but system falsely claimed auto-handle).
        """
        acc = accuracy_score(y_true, y_pred)
        labels = ["AUTO_HANDLE", "ESCALATE"]
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # Confusion matrix structure:
        # Rows: True [AUTO_HANDLE, ESCALATE]
        # Cols: Pred [AUTO_HANDLE, ESCALATE]
        tn, fp, fn, tp = cm.ravel()
        # tp: True ESCALATE predicted ESCALATE
        # fn: True ESCALATE predicted AUTO_HANDLE (DANGEROUS FALSE AUTO-HANDLE)
        # fp: True AUTO_HANDLE predicted ESCALATE (UNNECESSARY ESCALATION)
        # tn: True AUTO_HANDLE predicted AUTO_HANDLE
        
        precision_esc = precision_score(y_true, y_pred, pos_label="ESCALATE", zero_division=0)
        recall_esc = recall_score(y_true, y_pred, pos_label="ESCALATE", zero_division=0)
        f1_esc = f1_score(y_true, y_pred, pos_label="ESCALATE", zero_division=0)
        
        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        false_auto_handle_rate = fn / max((tp + fn), 1)
        unnecessary_escalation_rate = fp / max((tn + fp), 1)
        
        return {
            "accuracy": round(acc, 4),
            "precision": round(precision_esc, 4),
            "recall": round(recall_esc, 4),
            "f1": round(f1_esc, 4),
            "escalate_precision": round(precision_esc, 4),
            "escalate_recall": round(recall_esc, 4),
            "escalate_f1": round(f1_esc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "true_auto_handle": int(tn),
            "false_escalate": int(fp),
            "false_auto_handle_critical": int(fn),
            "true_escalate": int(tp),
            "false_auto_handle_rate": round(false_auto_handle_rate, 4),
            "unnecessary_escalation_rate": round(unnecessary_escalation_rate, 4)
        }

    def run_all_evaluations(self) -> Dict[str, Any]:
        print(f"Starting benchmark evaluation over {len(self.golden_df)} golden set examples...")
        
        y_true_intent = (self.golden_df["gold_intent"] if "gold_intent" in self.golden_df.columns else self.golden_df["intent"]).tolist()
        y_true_escalate = (self.golden_df["gold_decision"] if "gold_decision" in self.golden_df.columns else self.golden_df["should_escalate"]).tolist()
        customer_messages = self.golden_df["customer_message"].tolist()
        gold_convo_ids = set(self.golden_df["conversation_id"].tolist())
        
        # 1. Evaluate Trivial Baseline
        print("\n--- Evaluating Baseline 1: Trivial Baseline ---")
        trivial_model = TrivialBaseline()
        trivial_preds = trivial_model.batch_predict(customer_messages)
        t_intent_pred = [p["intent"] for p in trivial_preds]
        t_esc_pred = [p["decision"] for p in trivial_preds]
        
        trivial_intent_metrics = self.evaluate_intent_classification(y_true_intent, t_intent_pred)
        trivial_esc_metrics = self.evaluate_escalation(y_true_escalate, t_esc_pred)
        
        # 2. Evaluate Simple ML Baseline
        print("\n--- Evaluating Baseline 2: Simple ML Baseline (TF-IDF + LogReg) ---")
        ml_baseline = SimpleMLBaseline()
        ml_baseline.fit_from_historical_data()
        ml_preds = ml_baseline.batch_predict(customer_messages)
        ml_intent_pred = [p["intent"] for p in ml_preds]
        ml_esc_pred = [p["decision"] for p in ml_preds]
        
        ml_intent_metrics = self.evaluate_intent_classification(y_true_intent, ml_intent_pred)
        ml_esc_metrics = self.evaluate_escalation(y_true_escalate, ml_esc_pred)
        
        # 3. Evaluate Main AI Support Agent
        print("\n--- Evaluating Main AI Customer Support Agent ---")
        agent = CustomerSupportAgent().initialize()
        agent_preds = []
        retrieval_similarities = []
        retrieval_recall_at_1 = []
        retrieval_recall_at_3 = []
        
        for idx, row in self.golden_df.iterrows():
            msg = row["customer_message"]
            expected_intent = row["intent"]
            
            # Pass gold_convo_ids to enforce STRICT ZERO LEAKAGE
            out = agent.process_message(msg, exclude_conversation_ids=gold_convo_ids)
            agent_preds.append(out)
            
            # Retrieval analysis
            retrieved = out.get("retrieved_examples", [])
            sims = [r["similarity"] for r in retrieved]
            if sims:
                retrieval_similarities.append(max(sims))
            intents_retrieved = [r["intent"] for r in retrieved]
            retrieval_recall_at_1.append(1 if intents_retrieved and intents_retrieved[0] == expected_intent else 0)
            retrieval_recall_at_3.append(1 if expected_intent in intents_retrieved else 0)
            
        agent_intent_pred = [p["intent"] for p in agent_preds]
        agent_esc_pred = [p["decision"] for p in agent_preds]
        
        agent_intent_metrics = self.evaluate_intent_classification(y_true_intent, agent_intent_pred)
        agent_esc_metrics = self.evaluate_escalation(y_true_escalate, agent_esc_pred)
        
        retrieval_metrics = {
            "mean_top1_similarity": round(float(np.mean(retrieval_similarities)), 4) if retrieval_similarities else 0.0,
            "min_top1_similarity": round(float(np.min(retrieval_similarities)), 4) if retrieval_similarities else 0.0,
            "max_top1_similarity": round(float(np.max(retrieval_similarities)), 4) if retrieval_similarities else 0.0,
            "intent_retrieval_recall@1": round(float(np.mean(retrieval_recall_at_1)), 4),
            "intent_retrieval_recall@3": round(float(np.mean(retrieval_recall_at_3)), 4)
        }
        
        # Confusion matrix for Main Agent
        cm_matrix = confusion_matrix(y_true_intent, agent_intent_pred, labels=self.intent_labels).tolist()
        
        # Save Confusion Matrix Plot for Main Agent
        cm_path_results = RESULTS_DIR / "confusion_matrix.png"
        cm_path_eval = EVALUATION_DIR / "confusion_matrix.png"
        plot_and_save_confusion_matrix(
            y_true_intent,
            agent_intent_pred,
            labels=self.intent_labels,
            title="Main AI Agent Intent Confusion Matrix (AmazonHelp)",
            output_path=cm_path_results
        )
        plot_and_save_confusion_matrix(
            y_true_intent,
            agent_intent_pred,
            labels=self.intent_labels,
            title="Main AI Agent Intent Confusion Matrix (AmazonHelp)",
            output_path=cm_path_eval
        )
        print(f"Saved confusion matrix plot to {cm_path_results} and {cm_path_eval}")
        
        # Generate Comparative comparison_table / csv_rows
        csv_rows = [
            {
                "System": "Majority Baseline",
                "Intent Accuracy": trivial_intent_metrics["accuracy"],
                "Intent Macro Precision": trivial_intent_metrics["macro_precision"],
                "Intent Macro Recall": trivial_intent_metrics["macro_recall"],
                "Intent Macro F1": trivial_intent_metrics["macro_f1"],
                "Decision Accuracy": trivial_esc_metrics["accuracy"],
                "Decision Precision": trivial_esc_metrics["precision"],
                "Decision Recall": trivial_esc_metrics["recall"],
                "Decision F1": trivial_esc_metrics["f1"],
                "False Auto-Handles (Critical Hazard)": trivial_esc_metrics["false_auto_handle_critical"],
                "Retrieval Recall@3": "N/A"
            },
            {
                "System": "TF IDF + Logistic Regression",
                "Intent Accuracy": ml_intent_metrics["accuracy"],
                "Intent Macro Precision": ml_intent_metrics["macro_precision"],
                "Intent Macro Recall": ml_intent_metrics["macro_recall"],
                "Intent Macro F1": ml_intent_metrics["macro_f1"],
                "Decision Accuracy": ml_esc_metrics["accuracy"],
                "Decision Precision": ml_esc_metrics["precision"],
                "Decision Recall": ml_esc_metrics["recall"],
                "Decision F1": ml_esc_metrics["f1"],
                "False Auto-Handles (Critical Hazard)": ml_esc_metrics["false_auto_handle_critical"],
                "Retrieval Recall@3": "N/A"
            },
            {
                "System": "My AI Agent",
                "Intent Accuracy": agent_intent_metrics["accuracy"],
                "Intent Macro Precision": agent_intent_metrics["macro_precision"],
                "Intent Macro Recall": agent_intent_metrics["macro_recall"],
                "Intent Macro F1": agent_intent_metrics["macro_f1"],
                "Decision Accuracy": agent_esc_metrics["accuracy"],
                "Decision Precision": agent_esc_metrics["precision"],
                "Decision Recall": agent_esc_metrics["recall"],
                "Decision F1": agent_esc_metrics["f1"],
                "False Auto-Handles (Critical Hazard)": agent_esc_metrics["false_auto_handle_critical"],
                "Retrieval Recall@3": retrieval_metrics["intent_retrieval_recall@3"]
            }
        ]
        
        # Compile Combined Benchmark Results
        summary_results = {
            "dataset_info": {
                "golden_set_size": len(self.golden_df),
                "num_classes": len(self.intent_labels),
                "auto_handle_count": sum(1 for e in y_true_escalate if e == "AUTO_HANDLE"),
                "escalate_count": sum(1 for e in y_true_escalate if e == "ESCALATE")
            },
            "comparison_table": csv_rows,
            "intent_confusion_matrix": {
                "labels": self.intent_labels,
                "matrix": cm_matrix
            },
            "models": {
                "trivial_baseline": {
                    "intent_classification": trivial_intent_metrics,
                    "escalation": trivial_esc_metrics
                },
                "simple_ml_baseline": {
                    "intent_classification": ml_intent_metrics,
                    "escalation": ml_esc_metrics
                },
                "main_ai_agent": {
                    "intent_classification": agent_intent_metrics,
                    "escalation": agent_esc_metrics,
                    "retrieval": retrieval_metrics
                }
            }
        }
        
        # Save results.json to evaluation/ and metrics.json to results/
        eval_json_path = EVALUATION_DIR / "results.json"
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
        print(f"Saved evaluation results to {eval_json_path}")
        
        results_json_path = RESULTS_DIR / "metrics.json"
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
        print(f"Saved complete metrics to {results_json_path}")
        
        # Save metrics.csv & comparison_metrics.csv
        pd.DataFrame(csv_rows).to_csv(EVALUATION_DIR / "comparison_metrics.csv", index=False)
        pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "comparison_metrics.csv", index=False)
        pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "metrics.csv", index=False)
        print(f"Saved comparative metrics tables to {EVALUATION_DIR} and {RESULTS_DIR}")
        
        # Print Consolidated Report
        print("\n" + "=" * 95)
        print("CONSOLIDATED BENCHMARK SUMMARY (Golden Set N=200)")
        print("=" * 95)
        print(f"{'System / Model':<30} | {'Intent Acc':<10} | {'Macro F1':<9} | {'Esc Acc':<8} | {'False Auto-Handle':<18}")
        print("-" * 95)
        for r in csv_rows:
            print(f"{r['System']:<30} | {r['Intent Accuracy']:<10} | {r['Intent Macro F1']:<9} | {r['Decision Accuracy']:<8} | {r['False Auto-Handles (Critical Hazard)']:<18}")
        print("=" * 95)
        
        # Generate evaluation/predictions.csv with all requested fields
        pred_csv_rows = []
        for i, row in self.golden_df.iterrows():
            retrieved = agent_preds[i].get("retrieved_examples", [])
            top_evidence = retrieved[0]["resolution"] if retrieved else ""
            top_score = retrieved[0]["similarity"] if retrieved else 0.0
            pred_csv_rows.append({
                "customer_message": row["customer_message"],
                "gold_intent": row.get("gold_intent", row["intent"]),
                "predicted_intent": agent_intent_pred[i],
                "intent_confidence": round(float(agent_preds[i].get("intent_confidence", 0.0)), 4),
                "gold_decision": row.get("gold_decision", row.get("should_escalate")),
                "predicted_decision": agent_esc_pred[i],
                "generated_reply": agent_preds[i].get("reply", ""),
                "retrieved_evidence": top_evidence,
                "grounding_score": round(float(top_score), 4),
                "decision_reason": agent_preds[i].get("decision_reason", "")
            })
        pred_csv_path = EVALUATION_DIR / "predictions.csv"
        pd.DataFrame(pred_csv_rows).to_csv(pred_csv_path, index=False)
        print(f"Saved per-example predictions to {pred_csv_path}")
        
        # Also save detailed JSON records for Judge and Failure Analysis
        pred_records = []
        for i, row in self.golden_df.iterrows():
            pred_records.append({
                "id": int(row.get("id", i + 1)),
                "customer_message": row["customer_message"],
                "gold_intent": row.get("gold_intent", row["intent"]),
                "pred_intent": agent_intent_pred[i],
                "gold_escalate": row.get("gold_decision", row.get("should_escalate")),
                "pred_escalate": agent_esc_pred[i],
                "expected_resolution": row.get("expected_resolution", ""),
                "reply": agent_preds[i]["reply"],
                "decision_reason": agent_preds[i]["decision_reason"],
                "retrieved_examples": agent_preds[i]["retrieved_examples"]
            })
            
        with open(RESULTS_DIR / "agent_detailed_predictions.json", "w", encoding="utf-8") as f:
            json.dump(pred_records, f, indent=2)
            
        return summary_results

if __name__ == "__main__":
    harness = EvaluationHarness()
    harness.run_all_evaluations()
