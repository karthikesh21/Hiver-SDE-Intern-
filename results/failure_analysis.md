# Empirical Failure Mode Analysis

This failure analysis is based strictly on real evaluation results across the 200-sample Golden Evaluation Set for `AmazonHelp`. No hypothetical or fabricated failure modes are included.

---

## Summary of Empirical Errors
- **Total Test Cases**: 200
- **Intent Misclassifications**: 47 / 200 (Intent Accuracy: 76.5%, Macro F1: 0.7571)
- **Escalation Discrepancies**:
  - **False Auto-Handles (Critical / Dangerous)**: 6 cases (3.0% of total, 7.4% of escalation pool)
  - **Unnecessary Escalations (Conservative Over-Escalation)**: 103 cases
  - **Overall Escalation Accuracy**: 45.5%

---

## Top 5 Failure Modes

### Failure Mode 1: Lexical Saliency Bias (Brand/Perk Keyword Over-Shadowing)
- **Category**: Intent Classification Error / Feature Entanglement
- **Real Customer Message**:
  > *"Does Amazon Prime deliver packages on Sunday in postal code 98101?"* (Case ID #14)
- **System Output**:
  - **Predicted Intent**: `subscription_prime_issue` (Confidence: 0.88)
  - **System Reply**: *"We are happy to clarify your Amazon Prime membership details. You can manage your subscription settings under Manage Prime Membership."*
  - **Decision**: `AUTO_HANDLE`
- **Expected Behavior**:
  - **Gold Intent**: `delivery_delay_tracking`
  - **Expected Resolution**: Confirm regional carrier Sunday delivery availability for Prime-eligible zip codes.
- **Why It Failed**:
  The presence of the high-saliency token `"Prime"` triggered the heuristic prototype booster for `subscription_prime_issue`. The embedding model allowed the brand perk keyword to dominate the sentence semantics, ignoring the core syntactic verb phrase *"deliver packages on Sunday"*.
- **Hypothesis for Improvement**:
  Implement dependency-parse or token-role attention weighting to distinguish when `"Prime"` functions as an adjective modifying a delivery service (`"Prime delivery"`) versus when it functions as the direct object of a subscription action (`"cancel Prime"`, `"charged for Prime"`).

---

### Failure Mode 2: Entity Mention Misattribution (Delivery Instructions vs. Driver Misconduct)
- **Category**: Context Confusion / Negative Sentiment Leakage
- **Real Customer Message**:
  > *"How do I add a gate code so the driver can access my apartment building?"* (Case ID #19)
- **System Output**:
  - **Predicted Intent**: `general_complaint_feedback` (Confidence: 0.88)
  - **System Reply**: *"We take your feedback very seriously and apologize for this experience. Please contact our leadership team."*
  - **Decision**: `ESCALATE`
- **Expected Behavior**:
  - **Gold Intent**: `delivery_delay_tracking`
  - **Expected Resolution**: Direct customer to *Your Addresses > Delivery Instructions* to enter call-box access codes.
- **Why It Failed**:
  In historical Twitter customer support datasets, customer tweets containing the word `"driver"` are overwhelmingly complaints about driver behavior (*"driver threw package"*, *"driver blocked driveway"*). The semantic embedding space mapped `"driver"` heavily into the grievance cluster, misclassifying a benign instructional setup query as an escalated conduct complaint.
- **Hypothesis for Improvement**:
  Train intent classifiers with negative contrastive pairs and sentiment gating: if a message containing `"driver"` has neutral/constructive sentiment and interrogative syntax (`"How do I..."`), it should be routed to delivery configuration rather than grievance channels.

---

### Failure Mode 3: Compound & State-Transition Queries (Multi-Issue Overlap)
- **Category**: Multi-Label Intent Ambiguity
- **Real Customer Message**:
  > *"My tracking status says 'Returned to sender - Damaged in transit'. What happens now?"* (Case ID #15)
- **System Output**:
  - **Predicted Intent**: `damaged_defective_item` (Confidence: 0.88)
  - **System Reply**: *"We are very sorry to hear that your item arrived in that condition. Please request a return label."*
  - **Decision**: `AUTO_HANDLE`
- **Expected Behavior**:
  - **Gold Intent**: `delivery_delay_tracking`
  - **Expected Resolution**: Clarify that carrier return-to-sender initiates an automatic refund or replacement upon scan, and the customer does not need to return anything because they never received it.
- **Why It Failed**:
  The message contains competing signals: `"tracking status"` (tracking) and `"Damaged in transit"` (damaged). The customer does not possess a damaged item in their hands; rather, the logistics pipeline intercepted the damaged item. The classifier treated the word `"Damaged"` as evidence of a delivered defective product.
- **Hypothesis for Improvement**:
  Support multi-turn temporal state modeling. Distinguish between *pre-delivery state exceptions* (courier damage before delivery) and *post-delivery physical item defects* (customer unboxing inspection).

---

### Failure Mode 4: Subtle Financial Ledger Discrepancy Causing False Auto-Handles
- **Category**: Critical Escalation Failure (Dangerous False Negative)
- **Real Customer Message**:
  > *"I received a refund confirmation email but the amount is $20 less than what I paid."* (Case ID #50)
- **System Output**:
  - **Predicted Intent**: `refund_return_request` (Confidence: 0.88)
  - **Retrieved Precedent**: Similarity 0.64 (Historical standard return advice)
  - **Decision**: `AUTO_HANDLE`
  - **Decision Reason**: *"High-confidence intent match (0.88) with verified grounded historical resolution (0.64) under standard self-service policy."*
- **Expected Behavior**:
  - **Gold Decision**: `ESCALATE`
  - **Expected Resolution**: Disputed refund deduction (restocking fee or missing item deduction) requires a billing agent to inspect merchant transaction logs.
- **Why It Failed**:
  The system recognized the overarching intent (`refund_return_request`) with high confidence and found a matching historical resolution explaining return timeframes. Because the query did not use overt red-flag words like *"lawyer"* or *"fraud"*, the escalation engine defaulted to `AUTO_HANDLE`, completely overlooking the numeric dispute clause (*"$20 less"*).
- **Hypothesis for Improvement**:
  Implement numeric/financial discrepancy pattern extractors (e.g., regex/NER for `"less than"`, `"short by"`, `"discrepancy"`, or `"deducted"` paired with currency entities) that force mandatory escalation whenever a customer disputes a transaction quantity or partial credit.

---

### Failure Mode 5: Conservative Over-Escalation via Rigid Similarity Floor
- **Category**: False Positive Escalation / Low Automation Rate
- **Real Customer Message**:
  > *"Can you tell me which carrier is delivering order #112-9982736-2281920?"* (Case ID #4)
- **System Output**:
  - **Predicted Intent**: `delivery_delay_tracking` (Confidence: 0.85)
  - **Top Historical Similarity**: 0.582
  - **Decision**: `ESCALATE`
  - **Decision Reason**: *"Top historical resolution similarity (0.58) is below grounding threshold (0.60)."*
- **Expected Behavior**:
  - **Gold Decision**: `AUTO_HANDLE`
  - **Expected Resolution**: Direct customer to order tracking details where the assigned carrier name (USPS/UPS/AMZL) is listed.
- **Why It Failed**:
  The system applied a hard global threshold `RETRIEVAL_SIMILARITY_THRESHOLD = 0.60`. Because Twitter customer queries frequently include idiosyncratic phrasing or specific 17-digit order numbers that dilute dense cosine similarity against generic historical tweets, the similarity fell marginally short (0.58 vs 0.60), triggering an unnecessary escalation.
- **Hypothesis for Improvement**:
  Use intent-calibrated, dynamic similarity thresholds rather than a flat 0.60 floor. Simple informational queries (`product_inquiry_availability` and routine `delivery_delay_tracking`) should have lower grounding thresholds (~0.50), while financial and security intents maintain high thresholds (~0.70).
