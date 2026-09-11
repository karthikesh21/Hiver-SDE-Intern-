# AI Customer Support Agent for AmazonHelp

**Project:** AI Customer Support Agent  
**Brand:** AmazonHelp  
**Dataset:** Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)  

### Core Capabilities
- **Intent Classification:** Semantic centroid prototype matching with calibrated temperature scaling across 8 enterprise support intents.
- **Historical Resolution Retrieval:** Dense cross-corpus retrieval via `all-MiniLM-L6-v2` with strict zero-leakage negative filtering.
- **Grounded Reply Generation:** Non-hallucinatory reply synthesis strictly conditioned on verified historical precedents.
- **Auto Handle vs. Escalation:** Safety-constrained, multi-tiered decision engine enforcing rule-based guardrails and multi-threshold gating.
- **Evaluation & Safety Analysis:** End-to-end benchmark against Majority and TF-IDF+LogReg baselines, LLM-as-judge with human agreement, and empirical error diagnostics.

### Headline Results
- **Intent Accuracy:** 77.0%
- **Intent Macro F1:** 0.7650
- **Escalation Recall:** 91.04%
- **Dangerous False Auto Handles:** 6 / 67 (8.96% hazard rate vs. 73.1% for ML baseline, 100% for Majority)
- **Tests:** 22/22 passing
- **Evaluation Runtime:** approximately 16 seconds on CPU

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

### Primary Benchmark Comparison Table (Golden Set N=200)

| System | Intent Accuracy | Intent Macro F1 | Intent Macro Precision | Intent Macro Recall | Escalation Accuracy | Escalation Precision | Escalation Recall | Escalation F1 | Critical False Auto-Handles |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 12.50% | 0.0278 | 0.0156 | 0.1250 | 66.50% | 0.00% | 0.00% | 0.00% | 67 / 67 (100% hazard rate) |
| **Simple ML Baseline (TF-IDF + LogReg)** | 44.50% | 0.4344 | 0.7487 | 0.4450 | 65.50% | 47.37% | 26.87% | 34.29% | 49 / 67 (73.1% hazard rate) |
| **My AI Customer Support Agent** | **77.00%** | **0.7650** | **0.7759** | **0.7700** | **61.50%** | **46.21%** | **91.04%** | **61.31%** | **6 / 67 (8.96% hazard rate)** |

### Comparative Analysis of All Three Systems
1. **Majority Baseline**: Always predicts the most frequent training class (`delivery_delay_tracking`, which constitutes 87.8% of historical Twitter resolutions) and defaults to `AUTO_HANDLE`. On the balanced 8-class golden set (25 samples per class), it yields $25/200 = 12.50\%$ accuracy and Macro F1 of $0.0278$. Because it naively auto-handles every ticket, it misses **100% of required escalations (67 critical false auto-handles)**.
2. **Simple ML Baseline (TF-IDF + Logistic Regression)**: Trained strictly on historical training data without accessing golden labels. Achieves **44.50% accuracy** and **0.4344 Macro F1**. While precision on frequent classes is acceptable, it struggles on lower-frequency intents (such as general complaints and account access). On escalation, it misses **49 out of 67 escalations (73.1% false auto-handle rate)**.
3. **My AI Customer Support Agent**: Delivers **77.00% Intent Accuracy** (+32.5% over ML baseline) and **0.7650 Macro F1** (+0.33 over ML baseline) through semantic centroid prototype matching with calibrated temperature scaling ($T=0.12$). Crucially, on safety-critical escalation, the multi-tiered policy engine achieves **91.04% recall**, suppressing dangerous false auto-handles to just 6 cases (an 8.96% hazard rate compared to 73.1% for ML and 100% for Majority baseline).

- **Retrieval Recall@3**: `45.5%` on strict zero-leakage cross-corpus vector retrieval.
- **LLM-as-Judge Overall Rating**: `4.42 / 5.0`
- **Human vs. Judge Agreement (40 cases)**: Spearman $\rho = 0.3844$, Pearson $r = 0.3134$, Mean Absolute Difference (MAD) = `0.167`, Near-Agreement ($\le 0.5$) = `95.0%`.
- **Machine-Readable Artifacts**: Complete benchmark results stored in `evaluation/results.json` and per-example predictions stored in `evaluation/predictions.csv`.

---

## What is misleading about my headline number?

**77% intent accuracy does NOT mean 77% end-to-end customer resolution quality.** Presenting headline metrics without context obscures several operational realities:

1. **Golden Set Size & Statistical Variance:** The golden evaluation benchmark contains 200 curated examples (140 development, 60 held-out test). While rigorously audited for zero leakage, a sample of 200 (and test partition of 60) carries a binomial 95% confidence interval of approximately $\pm 6\%$ to $\pm 12\%$. Individual customer edge cases have a noticeable mathematical impact on percentages.
2. **Intentional Class Balancing vs. Real-World Power Laws:** The benchmark is intentionally balanced with 25 examples per intent across 8 classes (12.5% each). In real-world customer support on Twitter, `delivery_delay_tracking` alone accounts for ~87% of all inquiries, while account security or payment disputes account for less than 3%. An agent optimized purely on production volume would achieve higher nominal accuracy simply by predicting delivery tracking, yet fail catastrophically on low-frequency high-risk cases.
3. **Intent Accuracy vs. End-to-End Reply Quality:** Intent accuracy measures only whether a query was sorted into the correct topical category. It does not measure whether the generated response was factually accurate, actionable, empathetic, or actually solved the customer's problem.
4. **Retrieval Limitations:** Dense sentence embeddings (`all-MiniLM-L6-v2`) compress sentence semantics into 384-dimensional vectors. While strong at capturing semantic paraphrasing, bi-encoders can miss subtle lexical distinctions such as order numbers, tracking event modifiers, or arithmetic differences.
5. **Imperfect Historical Evidence & Deflection Bias:** The historical Twitter corpus (2017) reflects public agent interactions that heavily relied on generic canned deflections (*"Please contact us at [link]"*). The retriever inherits this bias, favoring generic self-service links over deep in-channel resolution.
6. **Escalation Tradeoffs: Cost Asymmetry:** Escalation decisions are inherently cost-asymmetric. A **false auto-handle** (telling a customer who was double-charged or locked out of their account that everything is fine) causes severe churn, financial liability, and safety risks. In contrast, a **false escalation** (routing a routine shipping query to a human agent) is merely a minor human labor overhead. Our system intentionally accepts a lower decision accuracy (61.50%) to achieve 91.04% escalation recall.
7. **LLM Judge Limitations:** Automated LLM rubrics gave an average score of 4.42 / 5.0. However, our human-judge agreement study (Spearman $\rho = 0.3844$, Pearson $r = 0.3134$) showed that automated judges systematically reward polite tone and valid link templates, whereas human evaluators penalize canned deflections on high-anxiety edge cases.
8. **Confidence Calibration Gaps:** Temperature scaling ($T=0.12$) sharpens probabilities, but novel customer phrasings can still disperse confidence (<0.50), causing conservative safe escalation on otherwise routine questions.
9. **Unseen Production Intents:** The system classifies across 8 discovered intents, but real production e-commerce contains hundreds of unmodeled edge cases (e.g., gift card fraud, AWS billing, B2B tax exemptions) that must be safely trapped by fallback escalation rules.
10. **Safety vs. Automation Tradeoff & Dangerous False Auto-Handles:** The **Dangerous False Auto-Handle Rate** (achieving 6 / 67 or 8.96% on our benchmark, compared to 73.1% for the ML baseline and 100% for the Majority baseline) is a vital operational safety metric because it quantifies the system's ability to prevent high-risk customer failures from being mistakenly automated. However, it is **not the universal or only "north star"** metric: a trivial system that escalates 100% of all customer messages achieves a perfect 0% dangerous false auto-handle rate, but provides 0% automation utility and collapses human support queues. A viable production system must balance automation rate (currently 34.0%) against false auto-handle risk.

---

## Golden Evaluation Set Methodology

The Golden Evaluation Set consists of **200 customer support examples** from `AmazonHelp`.

### 1. Sampling Strategy
- **Source Data**: Canonical Twitter Customer Support corpus (`thoughtvector/customer-support-on-twitter`) filtered for `AmazonHelp`.
- **Filtering**: Multi-turn customer interactions were selected, discarding bot spam, single-word inquiries, and corrupted strings.
- **Linguistic Diversity**: Includes short urgent messages (*"Someone hacked my account"*, *"Package stolen"*), detailed narrative queries with tracking numbers, and conversational nuance.
- **Sample Size**: Exactly 200 samples, fulfilling the 150–250 requirement.

### 2. Labelling Protocol
- **Intent Labeling (`gold_intent`)**: Every example was mapped into one of the 8 domain-discovered intents (`delivery_delay_tracking`, `refund_return_request`, `damaged_defective_item`, `payment_billing_issue`, `account_login_access`, `subscription_prime_issue`, `product_inquiry_availability`, `general_complaint_feedback`).
- **Escalation Labeling (`gold_decision`)**: Labeled as `AUTO_HANDLE` (133 samples) or `ESCALATE` (67 samples) based on enterprise policy:
  - `AUTO_HANDLE`: Standard self-service queries (tracking link lookups, return window checks, Sunday shipping queries).
  - `ESCALATE`: Security/2FA lockouts, severe courier misconduct, urgent delivery of life-critical items (medicine), and financial ledger disputes.
- **Evidence Guidance (`expected_resolution`)**: Specifies the verified resolution procedure expected from human support agents.

### 3. Class Balance Handling
- In production data, delivery tracking queries dominate (~87%). To ensure unbiased evaluation across all customer problem types, the golden set is strictly stratified with **25 examples per intent** across all 8 classes ($25 \times 8 = 200$).
- Escalation distribution contains 133 `AUTO_HANDLE` (66.5%) and 67 `ESCALATE` (33.5%), mirroring realistic operational operational ratios.

### 4. Zero Data Leakage Separation
- **Separation from Retrieval/Training Data**: Evaluating an agent against queries present in its vector index or training set would artificially inflate metrics through memorization.
- **Leakage Elimination**: All 200 golden messages and conversation IDs were actively purged from `data/processed/AmazonHelp_historical_resolutions.jsonl` (verified overlap = 0).
- **Runtime Negative Filter**: During evaluation, all golden conversation IDs are passed to the retriever's negative filter (`exclude_conversation_ids`), preventing any sample from retrieving itself.

---

## 15-Minute Fast Reproduction Guide

### 1. Requirements & Environment
- **Python Version:** Python 3.10+ (tested and verified on Python 3.11)
- **Disk Space:** ~1.5 GB for virtual environment and embedding weights
- **OS Support:** Windows (PowerShell/CMD), macOS, and Linux compatible
- **Model Download Behavior:** On first execution, `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) is downloaded from Hugging Face Hub and cached locally in `~/.cache/huggingface/hub`. Subsequent executions run 100% offline.
- **Dataset Scope:** Preprocessed historical resolutions (`4,000` multi-turn cases) and golden benchmark (`200` stratified samples) are bundled in `data/`. The project intentionally uses this documented, curated subsample rather than indexing the full 3-million-tweet corpus.

```bash
# 1. Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### 2. Run Unit Tests (22 Tests Passing)
Execute the complete pytest suite covering schemas, safety rules, financial dispute policies, retriever mechanics, and zero-leakage guards:
```bash
python -m pytest tests/ -v
```
- **Expected Result:** `22 passed in ~30s`
- **Zero Failures / Regressions**

### 3. Run Benchmark Evaluation Suite (~16–18 Seconds on CPU)
Run the automated evaluation harness across the Majority Baseline, TF-IDF + Logistic Regression baseline, Main AI Agent, and LLM-as-Judge agreement:
```bash
python evaluate.py
```
- **Expected Runtime:** ~16 to 18 seconds on standard CPU
- **Generated Artifact Locations:**
  - `evaluation/results.json` & `results/metrics.json`: Complete benchmark metrics and confusion matrix
  - `evaluation/predictions.csv`: Per-example predictions, ground truth, retrieved evidence, and decision reasons
  - `evaluation/comparison_metrics.csv` & `results/comparison_metrics.csv`: Comparative baseline table
  - `evaluation/confusion_matrix.png` & `results/confusion_matrix.png`: High-resolution confusion matrix heatmap
  - `evaluation/judge_ratings.json` & `results/judge_ratings.json`: 40-case rubric ratings
  - `evaluation/judge_human_agreement.md` & `results/judge_human_agreement.md`: Spearman ($\rho = 0.3844$), Pearson ($r = 0.3134$), and MAD ($0.167$) report

---

## Representative Customer Query Demos (CLI)

Test individual customer inquiries using the interactive command-line interface:
```bash
python main.py --message "Where is my package?"
```

Below are 8 representative scenarios demonstrating the agent across diverse problem categories:

#### 1. Delivery Tracking (Standard Self-Service)
```bash
python main.py --message "Where is my package? Tracking says delivered yesterday but I checked everywhere."
```
- **Intent:** `delivery_delay_tracking` (0.89)
- **Decision:** `AUTO_HANDLE`
- **Reason:** High-confidence intent match (0.89) with verified grounded historical resolution (0.78).
- **Reply:** *"We apologize for the delay with your delivery. We'd like to look into this with you. Please send us more details about this order here: the link in Your Orders"*

#### 2. Delayed Package (In-Transit Investigation)
```bash
python main.py --message "My order was supposed to arrive two days ago and tracking hasn't updated since Monday."
```
- **Intent:** `delivery_delay_tracking` (0.75)
- **Decision:** `AUTO_HANDLE`
- **Reason:** High-confidence intent match (0.75) with verified grounded historical resolution (0.79).
- **Reply:** *"We apologize for the delay with your delivery. Hi, sorry, what's the status of the tracking and delivery date? Have you looked into an A-Z Claim: the link in Your Orders"*

#### 3. Refund / Return Request
```bash
python main.py --message "How do I return this unopened sweater and get a refund?"
```
- **Intent:** `refund_return_request` (0.88)
- **Decision:** `AUTO_HANDLE`
- **Reason:** High-confidence intent match (0.88) with verified grounded historical resolution (0.59).
- **Reply:** *"We are glad to assist you with your return or refund. Have you checked with our support team here: the link in Your Orders for options to fix this for you?"*

#### 4. Damaged / Defective Item
```bash
python main.py --message "The glass teapot arrived shattered inside the package with broken shards everywhere."
```
- **Intent:** `damaged_defective_item` (0.88)
- **Decision:** `AUTO_HANDLE`
- **Reason:** High-confidence intent match (0.88) with verified grounded historical resolution (0.51).
- **Reply:** *"We are very sorry to hear that your item arrived in that condition. I'm so sorry they came damaged! Have you reported this to us or left packaging feedback? Was this through Prime Pantry?"*

#### 5. Duplicate / Financial Ledger Dispute (Critical Safety Escalation)
```bash
python main.py --message "I was charged twice on my credit card for the same order."
```
- **Intent:** `payment_billing_issue` (0.80)
- **Decision:** `ESCALATE`
- **Reason:** Financial ledger dispute or unfulfilled commitment trigger ('charged twice'): requires human account review.
- **Reply:** *"We understand your concern about being charged twice. Please check Your Orders or contact an Amazon customer specialist with your order details so they can review the charges and help resolve the issue."*

#### 6. Account Access / Security Lockout (Security Escalation)
```bash
python main.py --message "Someone hacked into my account and changed my password and 2FA settings."
```
- **Intent:** `account_login_access` (0.88)
- **Decision:** `ESCALATE`
- **Reason:** Account security verification or credential lockout requires identity verification.
- **Reply:** *"We apologize for the trouble accessing your account. - Can you report this to us directly via: the support link in Your Orders (It won't require you to sign in or anything)"*

#### 7. Prime Membership Issue
```bash
python main.py --message "Why was my Prime membership auto-renewed without my permission?"
```
- **Intent:** `subscription_prime_issue` (0.90)
- **Decision:** `AUTO_HANDLE`
- **Reason:** High-confidence intent match (0.90) with verified grounded historical resolution (0.73).
- **Reply:** *"We are happy to clarify your Amazon Prime membership details. did you receive an email confirming the cancellation? Can you see an option to cancel - the support link in Your Orders ?"*

#### 8. Courier Misconduct / Urgent Escalation
```bash
python main.py --message "My package contains urgent prescription insulin and your driver threw it onto the roof!"
```
- **Intent:** `general_complaint_feedback` (0.88)
- **Decision:** `ESCALATE`
- **Reason:** Severe service complaint or delivery misconduct requires supervisory review.
- **Reply:** *"We take your feedback very seriously and apologize for this experience. Please give us a ring here: the support link in Your Orders so we can look into this with you!"*

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
│   ├── test_financial_replies.py  # Financial billing disputes & safe reply tests
│   └── test_agent.py              # End-to-end schema & pipeline validation tests
│
├── evaluation/
│   ├── results.json                   # Machine-readable benchmark evaluation metrics & confusion matrix
│   ├── predictions.csv                # Per-example predictions, ground truth, retrieved evidence & scores
│   ├── comparison_metrics.csv         # Comparative metrics table (Majority, ML Baseline, Main Agent)
│   ├── confusion_matrix.png           # 8-class intent confusion matrix heatmap
│   ├── judge_ratings.json             # 40-case LLM-as-judge rubric ratings
│   └── judge_human_agreement.md       # Statistical agreement study (Spearman, Pearson, MAD)
│
├── results/
│   ├── final_policy.json              # Frozen escalation thresholds and constraints
│   ├── escalation_analysis.md         # Diagnostic analysis of over-escalation
│   ├── escalation_threshold_search.csv # 56-grid search on development set
│   ├── test_set_metrics.json          # Held-out test set evaluation metrics (N=60)
│   ├── comparison_metrics.csv         # Side-by-side previous vs improved policy table
│   ├── metrics.json                   # Full machine-readable evaluation metrics
│   ├── metrics.csv                    # Benchmark comparison table
│   ├── confusion_matrix.png           # High-res intent confusion matrix plot
│   ├── failure_analysis.md            # Top 5 empirical failure modes analysis
│   ├── judge_ratings.json             # 40-case judge and human scores
│   └── judge_human_agreement.md       # Statistical agreement report & case studies
│
└── report/
    └── report.md                      # Complete technical report with Escalation Improvement section
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
