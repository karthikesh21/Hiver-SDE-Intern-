# Dataset & Golden Evaluation Set Documentation

## Overview
This directory contains dataset assets used by the AI Customer Support Agent for `AmazonHelp`.
The data is derived from the canonical Twitter Customer Support dataset (`thoughtvector/customer-support-on-twitter`).

## Files
- `golden_set.csv`: Full 200 manually verified, stratified golden test examples.
- `golden_dev_set.csv`: 140 development evaluation samples (70% stratified split).
- `golden_test_set.csv`: 60 held-out test evaluation samples (30% stratified split).
- `intents.json`: 8-intent domain taxonomy schema with definitions, inclusion/exclusion rules, and typical resolutions.
- `processed/AmazonHelp_historical_resolutions.jsonl`: 4,000 cleaned, multi-turn historical support interactions used for retrieval.
- `processed/historical_embeddings.npy`: Precomputed 384-dimensional dense embeddings for historical records.
- `processed/historical_metadata.json`: Precomputed metadata index for fast retrieval.

---

## Golden Evaluation Set Methodology

### 1. Sampling Strategy
- **Source Selection**: Real customer support inquiries from the `AmazonHelp` brand in the Twitter Customer Support dataset.
- **Filtering**: We selected multi-turn customer interactions, discarding one-word tweets, spam, automated bot pings, and non-English messages.
- **Coverage**: To stress-test agent capabilities realistically, sampled messages include short terse complaints (*"Package stolen."*), detailed multi-sentence accounts (*"Where is my package? Tracking number 940011... says delivered yesterday but I checked everywhere."*), ambiguous phrasing, and policy edge cases.
- **Size**: Exactly 200 examples (meeting the 150–250 requirement target).

### 2. Labelling Protocol
- **Intent Labeling (`gold_intent`)**: Every example was mapped to one of the 8 discovered domain intents based on the criteria in `intents.json`:
  1. `delivery_delay_tracking`
  2. `refund_return_request`
  3. `damaged_defective_item`
  4. `payment_billing_issue`
  5. `account_login_access`
  6. `subscription_prime_issue`
  7. `product_inquiry_availability`
  8. `general_complaint_feedback`
- **Escalation Labeling (`gold_decision`)**: Labeled as `AUTO_HANDLE` or `ESCALATE` using enterprise policy principles:
  - `AUTO_HANDLE`: Standard self-service workflows (tracking checks, return policy window inquiries, Sunday delivery questions, general catalog inquiries).
  - `ESCALATE`: Security/credential lockouts (2FA failure), severe courier misconduct, urgent delivery emergencies (medicine/essential goods), unauthorized credit card charges, and account-takeover hazards.
- **Evidence Guidance (`expected_resolution`)**: Contains the verified reference resolution guideline specifying how human support historically resolves the issue.

### 3. Class Balance Handling
- **Intent Stratification**: In raw production traffic, delivery queries comprise ~87% of inquiries. To evaluate classification power equally across all issue types, the golden set enforces an exact uniform balance of **25 examples per intent** ($25 \times 8 = 200$).
- **Decision Balance**: Contains 133 `AUTO_HANDLE` (66.5%) and 67 `ESCALATE` (33.5%) cases across the 200 samples, reflecting a realistic operational ratio of resolvable queries versus high-risk hazards.

### 4. Separation from Training & Retrieval Data (Zero Data Leakage)
- **Why Separated**: Evaluating an agent on messages present in its retrieval index or training corpus artificially inflates metrics to near 100%, masking failure modes.
- **Purge Verification**: The build script (`src/build_golden_set.py`) computes message signatures and conversation IDs for all 200 golden examples, actively stripping any overlapping records from `processed/AmazonHelp_historical_resolutions.jsonl` (verified overlap = 0).
- **Dynamic Retrieval Guard**: During runtime evaluation, all golden conversation IDs are injected into the retriever's negative filter (`exclude_conversation_ids`), guaranteeing that no test query can retrieve itself or its own thread as historical precedent.

---

## Fields in `golden_set.csv`

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Unique identifier (1 to 200) |
| `customer_message` | String | Raw text of customer inquiry |
| `gold_intent` | String | Ground truth intent category |
| `intent` | String | Ground truth intent (alias for compatibility) |
| `gold_decision` | String | Ground truth action: `AUTO_HANDLE` or `ESCALATE` |
| `should_escalate` | String | Ground truth action (alias for compatibility) |
| `expected_resolution` | String | Verified historical resolution guideline / evidence |
| `conversation_id` | String | Unique evaluation thread identifier |
| `source` | String | Dataset source attribution (`curated_evaluation_benchmark`) |
