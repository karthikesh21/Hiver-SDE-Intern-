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

## Key Headline Results (Evaluated on Golden Set N=200)

| System / Model | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation Precision | Escalation Recall | False Auto-Handles (Critical Hazard) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trivial Baseline** | 12.5% | 0.0278 | 66.5% | 0.0000 | 0.0000 | 67 |
| **Simple ML Baseline (TF-IDF+LogReg)** | 44.5% | 0.4344 | 65.5% | 0.6000 | 0.3704 | 49 |
| **Main AI Customer Support Agent** | **76.5%** | **0.7571** | **45.5%** | **0.4213** | **0.9259** | **6 (87.8% Reduction)** |

- **Retrieval Recall@3**: `86.5%`
- **LLM-as-Judge Overall Quality**: `4.42 / 5.0`
- **Human vs. Judge Near-Agreement ($\le 0.5$)**: `97.5%` (Spearman $\rho = 0.370$, $p = 0.018$)

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

The preprocessed historical resolutions (`4,000` multi-turn cases) and golden evaluation benchmark (`200` samples) are already bundled inside `data/`:
- `data/golden_set.csv`: 200 manually verified golden evaluation samples.
- `data/intents.json`: 8 canonical domain intent definitions.
- `data/processed/AmazonHelp_historical_resolutions.jsonl`: Cleaned historical support conversations.

*(Optional)* To re-download or regenerate raw data from scratch:
```bash
python -m src.brand_analysis       # Re-runs quantitative brand comparison
python -m src.data_processing      # Reconstructs AmazonHelp conversations from HF
python -m src.build_golden_set     # Curates and validates golden evaluation set
```

### 4. Build / Warm Up Retrieval Embeddings
Pre-compute the dense embedding index (runs in ~20 seconds locally on CPU):
```bash
python -m src.retriever
```

### 5. Run Complete Benchmark Evaluation
Run the automated evaluation harness across all baselines and the main agent:
```bash
python evaluate.py
```
This automatically produces:
- `results/metrics.json`
- `results/metrics.csv`
- `results/confusion_matrix.png`
- `results/judge_ratings.json`
- `results/judge_human_agreement.md`

### 6. Run the Agent (CLI)
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
DECISION REASON:    High-confidence intent match (0.89) with verified grounded historical resolution (0.78) under standard self-service policy.
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

### 7. Run Unit Tests
Execute the pytest suite:
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
├── decision_log.md                # 12 non-obvious engineering decisions & tradeoffs
│
├── data/
│   ├── README.md                  # Dataset & golden set sampling protocol
│   ├── intents.json               # 8-intent domain taxonomy schema
│   ├── golden_set.csv             # 200 stratified golden evaluation cases
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
│   ├── metrics.json               # Full machine-readable evaluation metrics
│   ├── metrics.csv                # Benchmark comparison table
│   ├── confusion_matrix.png       # High-res intent confusion matrix plot
│   ├── failure_analysis.md        # Top 5 empirical failure modes analysis
│   ├── judge_ratings.json         # 40-case judge and human scores
│   └── judge_human_agreement.md   # Statistical agreement report & case studies
│
└── report/
    └── report.md                  # Complete technical report (6 pages equivalent)
```

---

## Reproducibility & Fixed Parameters

- **Random Seed**: Fixed globally to `42` (`RANDOM_SEED = 42`).
- **Target Brand**: `AmazonHelp` (81,092 conversations in dataset pool).
- **Historical Corpus**: 4,000 multi-turn resolution records.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized Euclidean length).
- **Intent Confidence Threshold**: `0.65`.
- **Retrieval Grounding Similarity Threshold**: `0.60`.
- **Zero Leakage**: Golden evaluation set examples and conversation IDs are dynamically excluded during retrieval ranking.

---

## Known Limitations

1. **Conservative Escalation Bias**: Prioritizing safety and reducing false auto-handles to 6 resulted in 103 unnecessary escalations on borderline similarity cases ($0.55 - 0.59$).
2. **Historical Deflection Bias**: Pre-2018 Twitter support agents frequently used standard link deflections (*"Please contact us at [link]"*), which the retriever inherits.
3. **Lexical Saliency**: High-frequency brand tokens (e.g. `"Prime"`) can overshadow syntactic intent verbs without explicit dependency parsing.
