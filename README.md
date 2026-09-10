# AI Customer Support Agent for AmazonHelp

An end-to-end, production-grade AI Customer Support Agent built for `AmazonHelp` using the Kaggle **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`).

The system classifies incoming customer inquiries into domain-derived intents, retrieves grounded historical support resolutions, synthesizes non-hallucinatory customer replies, executes transparent escalation decisions, and provides an automated evaluation harness with an LLM-as-judge and human agreement study.

---

## Architecture Overview

```
Incoming Customer Tweet
          │
          ▼
┌──────────────────────────────────────┐
│     Stage 1: Intent Classifier       │
│  (Semantic Centroid + Heuristics)    │
└──────────────────┬───────────────────┘
                   │ Intent & Confidence
                   ▼
┌──────────────────────────────────────┐
│    Stage 2: Historical Retriever     │
│ (all-MiniLM-L6-v2 + Cosine Ranking)  │
└──────────────────┬───────────────────┘
                   │ Top-K Precedents & Similarities
                   ▼
┌──────────────────────────────────────┐
│  Stage 3: Grounded Reply Generator   │
│  (Strict Anti-Hallucination Prompt)  │
└──────────────────┬───────────────────┘
                   │ Draft Grounded Response
                   ▼
┌──────────────────────────────────────┐
│   Stage 4: Escalation Policy Engine  │
│(Rule Safety Checks & Multi-Threshold)│
└──────────────────┬───────────────────┘
                   │ Decision & Explicit Reason
                   ▼
┌──────────────────────────────────────┐
│   Stage 5: Structured Agent Output   │
│  {Intent, Confidence, Decision, ...} │
└──────────────────┬───────────────────┘
```

---

## Evaluation Benchmark & Headline Results

### Comparative Performance Table (Golden Set N=200 & Held-Out Test N=60)

| System / Model | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation Precision | Escalation Recall | False Auto-Handles (Critical Hazard) | Automation Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trivial Baseline** | 12.50% | 0.0278 | 66.50% | 0.0000 | 0.0000 | 67 | 100.0% |
| **Simple ML Baseline (TF-IDF+LogReg)** | 44.50% | 0.4344 | 65.50% | 0.6000 | 0.3704 | 49 | 80.5% |
| **Main AI Agent (Previous Policy)** | 76.50% | 0.7571 | 45.50% | 42.13% | 92.59% | **6** | 18.0% |
| **Main AI Agent (Improved Policy - Held-Out Test N=60)** | **73.33%** | **0.7340** | **55.00%** | **41.46%** | **85.00%** | **3 (out of 20)** | **31.7%** |
| **Main AI Agent (Improved Policy - Full Benchmark N=200)**| **77.00%** | **0.7650** | **61.50% (+16.0%)** | **46.21%** | **91.04%** | **6 (Safety Preserved)** | **34.0% (Nearly 2x)** |

- **False Escalation Reduction**: Slashed from **103** down to **71** unnecessary human handoffs (-32 reduction!).
- **Retrieval Recall@3**: `45.5%` on strict zero-leakage cross-intent ranking.
- **LLM-as-Judge Overall Rating**: `4.42 / 5.0`
- **Human vs. Judge Near-Agreement ($\le 0.5$)**: `97.5%` (Spearman $\rho = 0.370$, $p = 0.018$)

---

## Evaluation Split & Calibration Methodology

To guarantee scientific rigor without data leakage or test set snooping:

```
200 Manually Verified Golden Examples
                │
        ┌───────┴───────┐
        ▼               ▼
70% Development Set   30% Held-Out Test Set
  (140 samples)           (60 samples)
        │                       │
        ▼                       │
Threshold Grid Search           │
& Safety Rule Refinement        │
        │                       │
        ▼                       ▼
Frozen Policy (results/final_policy.json) ──► Unbiased Final Evaluation
```

1. **Development Set (140 samples / 70%)**: Used exclusively for error diagnosis, threshold grid search, and policy experimentation.
2. **Held-Out Test Set (60 samples / 30%)**: Strictly quarantined and preserved until the policy was finalized and frozen.
3. **Safety-Constrained Optimization**: Imposed a hard constraint requiring False Auto-Handles $\le 3$ on the development set (Recall $\ge 90\%$) before maximizing accuracy.
4. **Frozen Parameters**:
   - `intent_confidence_threshold`: `0.50` (calibrated to plurality confidence).
   - `retrieval_similarity_threshold`: `0.45` (calibrated to social media tweet embedding dynamics).
   - Refined financial discrepancy rules (`"less than what i paid"`, `"free trial"`, `"double billed"`, `"same box by accident"`).

---

## 15-Minute Fast Reproduction Guide

### 1. Requirements
- Python 3.10+ (tested on Python 3.11)
- ~1.5 GB disk space for environment and embeddings
- Windows / macOS / Linux compatible

### 2. Environment Setup
Clone or enter the project directory and create a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Dataset Setup
The system uses the `Customer Support on Twitter` dataset (`thoughtvector/customer-support-on-twitter`).

The preprocessed historical resolutions (`4,000` multi-turn cases) and golden evaluation benchmark (`200` samples with 70/30 dev/test splits) are already bundled inside `data/`:
- `data/golden_set.csv`: Full 200 manually verified golden evaluation samples.
- `data/golden_dev_set.csv`: 140 development evaluation samples.
- `data/golden_test_set.csv`: 60 held-out final test evaluation samples.
- `data/intents.json`: 8 canonical domain intent definitions.
- `data/processed/AmazonHelp_historical_resolutions.jsonl`: Cleaned historical support conversations.

*(Optional)* To regenerate the stratified splits from scratch:
```bash
python -m src.create_splits
```

### 4. Build / Warm Up Retrieval Embeddings
Pre-compute the dense embedding index (runs in ~20 seconds locally on CPU):
```bash
python -m src.retriever
```

### 5. Reproduce Escalation Threshold Search
Run the threshold search on the Development Set:
```bash
python -m src.tune_escalation
```
This evaluates 56 threshold combinations on the 140 development examples, enforces safety constraints, and saves `results/escalation_threshold_search.csv` and `results/final_policy.json`.

### 6. Run Complete Benchmark & Held-Out Evaluation
Run the evaluation harness on both the held-out test set and full benchmark:
```bash
python -m src.evaluate_splits
```
This produces:
- `results/test_set_metrics.json`
- `results/comparison_metrics.csv`
- `results/agent_detailed_predictions.json`
- `results/confusion_matrix.png`

Run the complete baseline suite and human agreement study:
```bash
python evaluate.py
```

### 7. Run the Agent (CLI)
Test individual customer queries:
```bash
python main.py --message "Where is my package? Tracking says delivered yesterday but I checked everywhere."
```
Output:
```text
===========================================================================
AI CUSTOMER SUPPORT AGENT RESPONSE
===========================================================================
INTENT:             delivery_delay_tracking (Confidence: 0.89)
DECISION:           AUTO_HANDLE
DECISION REASON:    High-confidence match (0.89) with verified grounded resolution (0.78).
---------------------------------------------------------------------------
RETRIEVED HISTORICAL PRECEDENTS:
  [1] [Sim: 0.782 | ID: 09ce3604ef7c] We'd like to look into this with you. Please send us more details about this order here: [link] ^CL
  [2] [Sim: 0.774 | ID: fe4541b9979a] We can take a closer look at the order. When you have time, give us a call. We're avail 24/7 here: [link] ^EP
---------------------------------------------------------------------------
DRAFT CUSTOMER-FACING REPLY:
"We apologize for the delay with your delivery. We'd like to look into this with you. Please send us more details about this order here: the link in Your Orders"
===========================================================================
```

Raw JSON format:
```bash
python main.py --message "Your driver broke my flower pot!" --json
```

Interactive chat mode:
```bash
python main.py --interactive
```

### 8. Run Unit Tests
Execute the pytest suite (16 tests covering schemas, boundary conditions, safety rules, and zero-leakage):
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
hiver-sde-intern-assignment/
├── README.md                      # Reproduction & architecture guide
├── requirements.txt               # Pinned Python dependencies
├── .gitignore                     # Git ignore rules
├── .env.example                   # Environment configuration template
├── pytest.ini                     # Pytest path configuration
├── main.py                        # Interactive CLI for agent inference
├── evaluate.py                    # Top-level benchmark evaluation runner
├── decision_log.md                # 13 non-obvious engineering decisions & tradeoffs
│
├── data/
│   ├── README.md                  # Dataset & golden set sampling protocol
│   ├── intents.json               # 8-intent domain taxonomy schema
│   ├── golden_set.csv             # Full 200 stratified golden evaluation cases
│   ├── golden_dev_set.csv         # 140 development evaluation cases (70%)
│   ├── golden_test_set.csv        # 60 held-out test evaluation cases (30%)
│   └── processed/
│       ├── AmazonHelp_historical_resolutions.jsonl  # 4,000 historical turns
│       ├── historical_embeddings.npy                # Cached 384d vectors
│       └── historical_metadata.json                 # Fast metadata index
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Global paths, seeds, thresholds
│   ├── brand_analysis.py          # Empirical brand comparison script
│   ├── data_processing.py         # Tweet cleaning & turn parser
│   ├── build_golden_set.py        # Curates test benchmark & purges leakage
│   ├── create_splits.py           # Generates 70/30 stratified dev/test splits
│   ├── tune_escalation.py         # Grid search & safety-aware policy optimization
│   ├── evaluate_splits.py         # Evaluates dev, held-out test, and full benchmark
│   ├── extract_errors.py          # Formats false auto-handles and false escalations
│   ├── diagnose_escalation.py     # Diagnostic script for escalation bottleneck
│   ├── baselines.py               # Trivial & TF-IDF+LogReg baselines
│   ├── intent_classifier.py       # Hybrid semantic prototype classifier
│   ├── retriever.py               # MiniLM semantic retriever with leakage guard
│   ├── reply_generator.py         # Grounded reply synthesizer
│   ├── escalation.py              # Rule & confidence escalation engine
│   ├── agent.py                   # Orchestrator combining all stages
│   ├── evaluation.py              # Metrics, confusion matrix, and reporting
│   ├── judge.py                   # LLM-as-judge & human agreement study
│   └── analyze_failures.py        # Empirical error mining script
│
├── tests/
│   ├── test_data_processing.py    # Tweet cleaning & turn parsing unit tests
│   ├── test_retrieval.py          # Retriever & zero-leakage exclusion tests
│   ├── test_escalation.py         # Escalation triggers & safety policy tests
│   └── test_agent.py              # End-to-end schema & pipeline validation tests
│
├── results/
│   ├── final_policy.json          # Frozen escalation thresholds and constraints
│   ├── escalation_analysis.md     # Diagnostic analysis of over-escalation
│   ├── escalation_threshold_search.csv # 56-grid search on development set
│   ├── test_set_metrics.json      # Held-out test set evaluation metrics (N=60)
│   ├── comparison_metrics.csv     # Side-by-side previous vs improved policy table
│   ├── metrics.json               # Full machine-readable evaluation metrics
│   ├── metrics.csv                # Benchmark comparison table
│   ├── confusion_matrix.png       # High-res intent confusion matrix plot
│   ├── failure_analysis.md        # Top 5 empirical failure modes analysis
│   ├── judge_ratings.json         # 40-case judge and human scores
│   └── judge_human_agreement.md   # Statistical agreement report & case studies
│
└── report/
    └── report.md                  # Complete technical report with Escalation Improvement section
```

---

## Reproducibility & Fixed Parameters

- **Random Seed**: Fixed globally to `42` (`RANDOM_SEED = 42`).
- **Target Brand**: `AmazonHelp` (81,092 conversations in dataset pool).
- **Historical Corpus**: 4,000 multi-turn resolution records.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- **Calibrated Intent Confidence Threshold**: `0.50` (frozen on development set).
- **Calibrated Retrieval Grounding Threshold**: `0.45` (frozen on development set).
- **Zero Leakage**: Golden evaluation set examples and conversation IDs are dynamically excluded during retrieval ranking.

---

## Known Limitations

1. **Remaining Conservative Escalations**: 71 false escalations remain across the full benchmark, representing queries with dispersed confidence (<0.50). Refusing to automate uncertain queries is an intentional safety design.
2. **Historical Deflection Bias**: Pre-2018 Twitter support agents frequently used standard link deflections (*"Please contact us at [link]"*), which the retriever inherits.
3. **Held-Out Sample Size (N=60)**: While methodologically sound, a 60-sample test set carries wider confidence intervals than large-scale enterprise suites.
