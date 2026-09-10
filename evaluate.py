"""
Top-Level Evaluation Runner:
Executes the comprehensive evaluation pipeline across all baselines and the main agent,
producing all metric files, plots, judge scores, and agreement reports.
"""

import sys
import time
from pathlib import Path
from src.evaluation import EvaluationHarness
from src.judge import run_human_agreement_experiment
from src.config import RESULTS_DIR

def run_evaluation_suite():
    start_time = time.time()
    print("=" * 80)
    print("HIVERA SDE ASSIGNMENT: COMPREHENSIVE BENCHMARK EVALUATION SUITE")
    print("=" * 80)
    
    # 1. Run Core Pipeline Evaluations
    harness = EvaluationHarness()
    results = harness.run_all_evaluations()
    
    # 2. Run LLM Judge & Human Agreement Analysis
    print("\n--- Running LLM-as-Judge and Human Agreement Experiment ---")
    run_human_agreement_experiment(sample_size=40)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"EVALUATION SUITE COMPLETED IN {elapsed:.1f} SECONDS")
    print(f"Artifacts generated in: {RESULTS_DIR}")
    print("  - metrics.json")
    print("  - metrics.csv")
    print("  - confusion_matrix.png")
    print("  - judge_ratings.json")
    print("  - judge_human_agreement.md")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation_suite()
