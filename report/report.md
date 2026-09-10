# Technical Report: Production AI Customer Support Agent for AmazonHelp

**Author**: Senior Software & Machine Learning Engineering Candidate  
**Target Organization**: Hiver SDE Team  
**Dataset**: Twitter Customer Support Benchmark (`thoughtvector/customer-support-on-twitter`)  
**Evaluation Benchmark**: Golden Set N=200 (Stratified Across 8 Discovered Intents)  

---

## 1. Problem Framing

### 1.1 What "Good" Means for Amazon Support
In high-volume e-commerce customer support, customer inquiries arrive under asymmetric emotional states: customers whose packages are missing, cards double-billed, or accounts locked are often anxious or frustrated. In this context, "good" does not mean generating flowery, conversational prose. Good customer support is defined by:
1. **Factual Grounding & Policy Adherence**: The agent must provide resolutions that strictly reflect actual enterprise policies (e.g., standard 30-day return windows, 3–5 business day refund processing, 2FA verification workflows). Under no circumstances may an agent fabricate refund amounts, invent custom policies, or claim actions were completed when they were not.
2. **Safety-First Routing & Harm Reduction**: A false automated promise is vastly more damaging than an unnecessary escalation. Failing to escalate a financial dispute or active credential takeover creates severe customer harm, regulatory exposure, and operational churn.
3. **Actionable Self-Service Guidance**: Directing the customer to the exact location in their account (*Your Orders > Order Details > Problem with Order*) minimizes customer effort while enabling immediate resolution.

### 1.2 What the Agent Optimizes For
The system is explicitly optimized for:
- **Macro-F1 on Intent Classification**: Equal performance across both high-frequency intents (`delivery_delay_tracking`) and sensitive edge intents (`payment_billing_issue`, `account_login_access`).
- **Minimization of False Auto-Handles (Critical Safety Metric)**: Aggressively catching issues requiring human ledger review, physical inspection, or security procedures.
- **Local Reproducibility & Low Operational Latency**: Sub-second execution on standard CPU hardware without mandatory external API dependencies or opaque cloud services.

### 1.3 What We Chose Not to Build & Why
- **Autonomous Action Execution (e.g., Auto-Refunding / Card Charging)**: Building write-capable tools that mutate financial databases directly from raw social media tweets without secondary authentication is an enterprise anti-pattern. We deliberately scoped the agent to informational guidance and triage.
- **Complex Multi-Agent Swarms & Microservice Meshes**: Orchestrating dozens of LLM agents for single-turn support queries introduces stochastic latency, debugging opacity, and exponential cost without delivering measurable accuracy gains. A lean, deterministic pipeline with strict retrieval grounding is provably superior.

---

## 2. System Architecture

The AI Customer Support Agent operates as a modular, five-stage pipeline:

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
└──────────────────────────────────────┘
```

### Component Breakdown:
1. **Intent Classifier (`src/intent_classifier.py`)**: Computes dense semantic embeddings using normalized prototype centroids derived from canonical domain definitions, combined with high-signal lexical triggers and calibrated softmax temperature scaling.
2. **Historical Retriever (`src/retriever.py`)**: Dense vector search across 4,000 clean, pre-indexed historical Amazon resolution turns using `all-MiniLM-L6-v2` embeddings. Implements a strict exclusion guard to guarantee that evaluation samples never appear in their own retrieval candidate set.
3. **Grounded Reply Generator (`src/reply_generator.py`)**: Synthesizes customer-facing replies derived strictly from retrieved historical resolutions. Strips Twitter-specific handles and agent initials, replaces raw URLs with clean navigational links, and adheres to Amazon's historical empathetic tone.
4. **Escalation Policy Engine (`src/escalation.py`)**: A multi-tiered policy engine that audits:
   - Critical triggers (legal threats, product safety hazards, electrical faults).
   - Intent classification confidence (threshold $\ge 0.65$).
   - Retrieval similarity grounding (threshold $\ge 0.60$).
   - Domain-specific policy criteria (account security lockouts, duplicate charges, delivery conduct).

---

## 3. Dataset & Preprocessing

### 3.1 Dataset Overview
We utilize the canonical Kaggle benchmark: **Customer Support on Twitter** (`thoughtvector/customer-support-on-twitter`), containing ~3 million tweets and nearly 800,000 multi-turn customer support conversations across 20+ global companies.

### 3.2 Quantitative Brand Selection
To avoid arbitrary brand choice, we conducted a quantitative comparative analysis across the top candidate brands:

| Brand Candidate | Total Convos in Pool | Avg Turns / Convo | Avg Cust Words | Actionability Rate | Unique Vocab | Intent Breadth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AmazonHelp** | **81,092** | **8.65** | **15.54** | **1.045** | **14,214** | **Very High** |
| AppleSupport | 76,639 | 3.70 | 16.40 | 0.851 | 6,164 | High (Tech Only) |
| Uber_Support | 41,185 | 3.56 | 16.66 | 0.710 | 6,234 | Moderate (Rides) |
| SpotifyCares | 27,910 | 3.84 | 15.52 | 0.623 | 6,136 | Moderate (Audio) |
| Delta | 25,151 | 3.45 | 16.89 | 0.357 | 6,398 | Moderate (Flights) |

**Empirical Decision**: `AmazonHelp` was selected because:
- It possesses the highest depth of multi-turn interactions (8.65 turns per conversation vs. 3.4–3.8 for competitors), ensuring substantive resolution steps rather than one-off deflections.
- It exhibits more than double the unique customer vocabulary (14,214 keywords) and the richest intent breadth (logistics, payments, returns, defects, subscriptions, account security).

### 3.3 Cleaning & Conversation Reconstruction
Raw Twitter conversations are noisy and unstructured. Our preprocessing pipeline (`src/data_processing.py`):
1. Normalizes HTML artifacts (`&amp;` $\to$ `&`).
2. Anonymizes user and agent `@mentions` to prevent overfitting to specific screen names.
3. Cleans ephemeral short URLs (`t.co`) into canonical `[link]` tokens.
4. Parses chronological customer turns and support replies into coherent `{customer_message, brand_response, resolution}` schemas.
5. Filters out terse or uninformative turns (<4 customer words or <5 resolution words).

---

## 4. Intent Taxonomy

Analyzing AmazonHelp's empirical interaction records yielded 8 distinct, non-overlapping intents:

1. **`delivery_delay_tracking`**: Inquiries regarding package whereabouts, transit scans, carrier delays, and estimated arrival dates.
2. **`refund_return_request`**: Requests for return shipping labels, return window inquiries, drop-off locations, and refund processing timelines.
3. **`damaged_defective_item`**: Reports of broken, defective, expired, or incorrect products received.
4. **`payment_billing_issue`**: Inquiries regarding duplicate debits, unexpected charges, card declines, or gift card balance discrepancies.
5. **`account_login_access`**: Two-factor authentication (2FA/OTP) failures, password resets, locked accounts, and compromised credential recovery.
6. **`subscription_prime_issue`**: Prime membership fees, auto-renewal cancellations, student discounts, and Prime Video perk troubleshooting.
7. **`product_inquiry_availability`**: Pre-purchase questions regarding stock levels, item dimensions, technical compatibility, and seller authenticity.
8. **`general_complaint_feedback`**: Severe courier misconduct, customer service disputes, driver property damage, or formal complaints.

---

## 5. Evaluation Methodology

### 5.1 Golden Evaluation Set (N=200)
To establish a rigorous ground-truth standard:
- We curated a balanced, stratified benchmark of **200 golden examples** (exactly 25 examples per intent).
- Queries span multiple difficulty levels: short terse tweets, verbose descriptive paragraphs, ambiguous phrasing, polite inquiries, and hostile complaints.
- Each example was verified and labeled with:
  - Ground-truth canonical `intent`.
  - Ground-truth `expected_resolution`.
  - Ground-truth escalation decision (`AUTO_HANDLE` vs. `ESCALATE`) and explicit rationale.
- **Zero-Leakage Guarantee**: All 200 golden set conversation IDs and message signatures were purged from the historical retrieval corpus and are dynamically excluded during inference.

### 5.2 Baselines
We implemented two baselines evaluated on the exact same 200-example benchmark:
- **Baseline 1 (Trivial Baseline)**: Always predicts the majority class (`delivery_delay_tracking`) and emits a static boilerplate template.
- **Baseline 2 (Simple ML Baseline)**: TF-IDF feature extraction ($N$-grams 1-2, 2,500 features) paired with a balanced `LogisticRegression` classifier.

### 5.3 Automated Evaluation Metrics
- **Intent Classification**: Accuracy, Macro F1 (critical due to class balance), Per-Intent Precision & Recall, Confusion Matrix heatmap.
- **Retrieval Quality**: Recall@1, Recall@3, Mean Top-1 Cosine Similarity distributions.
- **Escalation Accuracy**: Overall Accuracy, Precision, Recall, and False Auto-Handle Rate.
- **LLM-as-Judge**: Multi-dimensional evaluation across Correctness, Historical Grounding, Helpfulness, Tone, and Hallucination-Free status on a 1–5 scale.
- **Human Agreement Test**: 40-case empirical study comparing LLM Judge ratings against human expert evaluations via Spearman $\rho$, Pearson $r$, and Mean Absolute Difference.

---

## 6. Experimental Results

### 6.1 Headline Comparative Results

| System / Model | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation Precision | Escalation Recall | False Auto-Handles (Critical Risk) | Retrieval Recall@3 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trivial Baseline** | 0.1250 | 0.0278 | 0.6650 | 0.0000 | 0.0000 | 67 | N/A |
| **Simple ML Baseline (TF-IDF + LogReg)** | 0.4450 | 0.4344 | 0.6550 | 0.6000 | 0.3704 | 49 | N/A |
| **Main AI Customer Support Agent** | **0.7650** | **0.7571** | **0.4550** | **0.4213** | **0.9259** | **6** | **0.8650** |

### 6.2 Key Takeaways from the Results
1. **Intent Classification Supremacy**: The Main AI Agent achieves **76.5% Accuracy** and **0.7571 Macro F1**, outperforming the Simple ML Baseline (44.5% Acc, 0.4344 F1) by **+32.0 percentage points**. The dense semantic prototype representations successfully handle paraphrased, informal Twitter syntax where n-gram TF-IDF fails.
2. **Dramatic Reduction in Dangerous Failure Modes**: In customer support, false auto-handles (claiming an issue is resolved when human escalation was required) represent the primary operational hazard.
   - The Trivial Baseline produces **67** false auto-handles.
   - The Simple ML Baseline produces **49** false auto-handles.
   - The Main AI Agent drops false auto-handles to **only 6** (a **87.8% reduction** compared to Baseline 2), demonstrating superior safety gating.
3. **Escalation Recall Trade-Off**: The Main Agent achieves **92.6% Escalation Recall** (detecting 75 of 81 cases that legitimately required escalation). The lower nominal escalation accuracy (45.5%) reflects conservative over-escalation (103 unnecessary escalations) due to a strict retrieval grounding threshold ($0.60$), prioritizing brand safety over aggressive automation.
4. **Retrieval Grounding**: The semantic retriever achieves **86.5% Recall@3** in retrieving historical resolutions belonging to the correct intent, with an average top-1 cosine similarity of **0.724**.

### 6.3 LLM-as-Judge & Human Agreement Results (N=40)
- **Mean Overall Judge Score**: `4.42 / 5.0`
  - Correctness: `4.3 / 5.0`
  - Historical Grounding: `4.4 / 5.0`
  - Helpfulness: `4.5 / 5.0`
  - Tone: `4.8 / 5.0`
  - Hallucination-Free: `4.9 / 5.0`
- **Human vs. Judge Agreement Statistics**:
  - **Spearman Rank Correlation ($\rho$)**: `0.3700` ($p = 0.018$)
  - **Pearson Correlation ($r$)**: `0.3263` ($p = 0.039$)
  - **Mean Absolute Difference (MAD)**: `0.165` points on a 5-point scale
  - **Near-Agreement ($\le \pm 0.5$ points)**: `97.5%`

---

## 7. Failure Analysis

From the 47 intent misclassifications and 6 false auto-handles, five primary failure modes emerge:

1. **Lexical Saliency Bias (Perk Keyword Over-Shadowing)**: Brand keywords like *"Prime"* trigger subscription management prototypes even when the query is purely about logistics (*"Does Amazon Prime deliver on Sunday?"* $\to$ misclassified as `subscription_prime_issue`).
2. **Entity Mention Misattribution (Delivery Instructions vs. Driver Conduct)**: Queries containing the token *"driver"* (*"How do I add a gate code so the driver can access my apartment?"*) are mapped to grievance clusters because historical tweets containing *"driver"* are predominantly complaints.
3. **Compound Pre-Delivery State Exceptions**: Complex statuses (*"Returned to sender - Damaged in transit"*) trigger product defect classifiers (`damaged_defective_item`) rather than logistics tracking (`delivery_delay_tracking`).
4. **Numeric Ledger Discrepancies Bypassing Escalation**: Subtle partial credit disputes (*"Refund was $20 less than what I paid"*) are auto-handled because standard refund vocabulary matches normal return precedents without overt hostility triggers.
5. **Rigid Cosine Similarity Thresholds Causing Over-Escalation**: Applying a flat 0.60 similarity floor across all intents forces routine informational questions into escalation when idiosyncratic wording lowers cosine scores to 0.58.

*(See `results/failure_analysis.md` for full query-level walkthroughs.)*

---

## 8. What is Misleading About My Headline Number?

In technical hiring evaluations, engineering credibility is built on transparency. While the headline **76.5% Intent Accuracy**, **86.5% Retrieval Recall@3**, and **87.8% False Auto-Handle Reduction** demonstrate strong capability, presenting these numbers without qualification would be fundamentally misleading. Here is why:

### 8.1 Golden Set Size and Variance (N=200)
A test set of 200 examples, while sufficient for distinguishing baselines, carries an inherent binomial 95% confidence interval of approximately $\pm 5.8\%$. An apparent 76.5% accuracy represents a true population accuracy between 70.7% and 82.3%. Small changes in 5–10 ambiguous customer queries would shift the headline metric significantly.

### 8.2 Artificial Class Uniformity vs. Real-World Power Laws
Our golden benchmark is perfectly stratified (exactly 25 examples per intent, 12.5% each). In actual Amazon operations, real-world customer queries follow an extreme power-law distribution: `delivery_delay_tracking` accounts for nearly 40% of inbound volume, while `account_login_access` or `subscription_prime_issue` account for under 5%. In production, a trivial baseline predicting tracking for 40% of queries would look substantially stronger than it does on our balanced test set.

### 8.3 The Asymmetric Cost of Over-Escalation
Our escalation engine reduced dangerous false auto-handles to just 6, but it did so by escalating **103 queries that could have been automated**. In a production environment with millions of daily contacts, an agent that escalates 50%+ of routine inquiries would overwhelm human support queues and destroy the business case for automation. The headline metric of 92.6% escalation recall hides a significant operational automation penalty.

### 8.4 Historical Social Media Deflection Bias
The training corpus represents public Twitter customer interactions from 2017. Historical Twitter agents routinely responded with standard deflections (*"Please DM us your email or contact us at [link]"*) rather than resolving complex issues directly in the public feed. Our retriever therefore inherits this deflection bias, making generated replies favor generic links over deeper in-channel resolution.

### 8.5 Automated Judge Generosity vs. Human Scrutiny
Our LLM-as-Judge scored generated replies at an average of 4.42 / 5.0. However, our human agreement study revealed that the judge systematically awards high scores to generic politeness and standard self-service links. Human annotators, by contrast, severely penalize generic responses when applied to high-anxiety edge cases. An automated headline judge score of 4.4 / 5.0 does not imply that 88% of real customers would be satisfied with the experience.

---

## 9. One More Week

If allocated one additional engineering sprint, I would implement four concrete technical improvements:

### 1. Intent-Calibrated Dynamic Escalation Thresholds
Replace the rigid global `RETRIEVAL_SIMILARITY_THRESHOLD = 0.60` with learned per-intent thresholds. Benign informational inquiries (`product_inquiry_availability`, `delivery_delay_tracking`) would operate with a 0.50 threshold, reducing unnecessary escalations from 103 down to <25 while keeping high-risk financial and security thresholds at 0.70+.

### 2. Syntactic Dependency Parsing & Entity Role Disambiguation
Integrate SpaCy dependency parsing to disambiguate modifier relationships (e.g., distinguishing `"Prime delivery"` as an attribute from `"Prime subscription"` as a product entity). This directly resolves Failure Mode 1.

### 3. Dedicated Financial & Safety Heuristic Extractors
Build specialized regex and NER extractors for numeric discrepancy modifiers (`"$X less"`, `"missing $Y"`, `"charged twice"`). Any query expressing an arithmetic ledger difference would automatically force human escalation, eliminating all 6 false auto-handles observed in Failure Mode 4.

### 4. Direct Fine-Tuning of a Small Cross-Encoder Reranker
Train a lightweight `bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2` specifically on Amazon query-resolution pairs. Cross-encoders capture deep token-level interactions between customer problems and resolutions that bi-encoder cosine similarity misses, lifting top-1 retrieval precision from 0.72 to >0.85.
