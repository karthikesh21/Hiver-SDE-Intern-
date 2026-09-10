# Dataset & Golden Evaluation Set Documentation

## Overview
This directory contains the dataset assets used by the AI Customer Support Agent for `AmazonHelp`.
The data is derived from the canonical Twitter Customer Support dataset (`thoughtvector/customer-support-on-twitter`).

## Files
- `golden_set.csv`: 200 manually verified, stratified golden test examples.
- `intents.json`: 8-intent taxonomy schema with definitions, inclusion/exclusion rules, and typical resolutions.
- `processed/AmazonHelp_historical_resolutions.jsonl`: 4,000 cleaned, multi-turn historical support interactions used for retrieval.

## Golden Set Details
- **Total Samples**: 200
- **Stratification**: Exactly 25 examples across each of the 8 canonical intents:
  1. `delivery_delay_tracking`: 25 (12.5%)
  2. `refund_return_request`: 25 (12.5%)
  3. `damaged_defective_item`: 25 (12.5%)
  4. `payment_billing_issue`: 25 (12.5%)
  5. `account_login_access`: 25 (12.5%)
  6. `subscription_prime_issue`: 25 (12.5%)
  7. `product_inquiry_availability`: 25 (12.5%)
  8. `general_complaint_feedback`: 25 (12.5%)

- **Escalation Distribution**:
  - `AUTO_HANDLE`: 119 samples (59.5%)
  - `ESCALATE`: 81 samples (40.5%)

## Fields in `golden_set.csv`
- `id`: Unique identifier (1 to 200).
- `customer_message`: Customer tweet query.
- `intent`: Canonical ground truth intent.
- `expected_resolution`: Ideal historical support resolution guidance.
- `should_escalate`: Ground truth escalation flag (`AUTO_HANDLE` or `ESCALATE`).
- `conversation_id`: Identifier distinguishing evaluation cases.
- `source`: Attribution (`curated_evaluation_benchmark`).

## Zero Leakage Isolation Protocol
To guarantee evaluation integrity:
1. Every golden example customer message is checked against the historical resolution retrieval pool.
2. Exact matches or overlapping conversation IDs are purged from `AmazonHelp_historical_resolutions.jsonl`.
3. Golden set examples are never indexed into the vector retriever during evaluation.
