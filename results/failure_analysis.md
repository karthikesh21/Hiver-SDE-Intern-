# Empirical Failure Mode Analysis (Post-Escalation Improvement)

This failure analysis is based strictly on real evaluation results across the 200-sample Golden Evaluation Set and the 60-sample Held-Out Test Set for `AmazonHelp`. No hypothetical or fabricated failure modes are included.

---

## Summary of Empirical Benchmark Errors
- **Total Test Cases**: 200 (140 Development, 60 Held-Out Test)
- **Intent Misclassifications**: 46 / 200 (Intent Accuracy: **77.0%**, Macro F1: **0.7650**)
- **Escalation Discrepancies**:
  - **False Auto-Handles (Critical / Dangerous Misses)**: **6 cases** (3 in Dev, 3 in Test; 91.04% Escalation Recall)
  - **False Escalations (Unnecessary Human Handoffs)**: **71 cases** (down from 103 on original baseline, a **31.1% reduction**)
  - **Overall Escalation Accuracy**: **61.50%** (up from 45.50% on original policy, a **+16.0% gain**)
  - **Automation Rate**: **34.00%** (nearly doubled from 18.00%)

---

## Top 5 Empirical Failure Modes

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
  The presence of the high-saliency token `"Prime"` triggered the prototype booster for `subscription_prime_issue`. The embedding model allowed the brand perk keyword to dominate sentence semantics, ignoring the core syntactic verb phrase *"deliver packages on Sunday"*.
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

### Failure Mode 4: Subtle Verbal Grievances Bypassing Escalation (False Auto-Handles)
- **Category**: Critical Escalation Failure (Dangerous False Negative)
- **Real Customer Message**:
  > *"The customer service representative could barely understand basic English and hung up."* (Case ID #189)
- **System Output**:
  - **Predicted Intent**: `general_complaint_feedback` (Confidence: 0.63)
  - **Top Historical Similarity**: 0.535 (Historical agent apologies)
  - **Decision**: `AUTO_HANDLE`
  - **Decision Reason**: *"High-confidence match (0.63) with verified grounded resolution (0.53)."*
- **Expected Behavior**:
  - **Gold Decision**: `ESCALATE`
  - **Expected Resolution**: Support quality grievance requiring supervisory audit of contact transcript.
- **Why It Failed**:
  The query correctly classified as `general_complaint_feedback` and had similarity 0.535 (above the calibrated 0.45 threshold). However, because the customer expressed the misconduct with the words *"could barely understand basic English and hung up"* rather than explicit profanity or standard trigger keywords (*"supervisor"*, *"disrespectful"*, *"lawyer"*), the escalation rules missed it and allowed an automatic canned reply.
- **Hypothesis for Improvement**:
  Expand agent conduct escalation triggers to encompass conversational termination events (`"hung up"`, `"disconnected the chat"`, `"closed window in my face"`) and communication barrier grievances.

---

### Failure Mode 5: Borderline Uncertainty Safety Escalation (Conservative False Escalation)
- **Category**: False Positive Escalation / Remaining Efficiency Bottleneck
- **Real Customer Message**:
  > *"My order has been stuck in 'Departed Facility' in Memphis for 5 days with no new scan."* (Case ID #2)
- **System Output**:
  - **Predicted Intent**: `delivery_delay_tracking` (Confidence: 0.48)
  - **Top Historical Similarity**: 0.573
  - **Decision**: `ESCALATE`
  - **Decision Reason**: *"Intent classification confidence (0.48) below threshold (0.50)."*
- **Expected Behavior**:
  - **Gold Decision**: `AUTO_HANDLE`
  - **Expected Resolution**: Advise customer of carrier transit buffer (up to 48 hours past ETA) and direct them to tracking updates in Your Orders.
- **Why It Failed**:
  The intent prediction was 100% correct (`delivery_delay_tracking`), and the historical resolution was grounded (similarity 0.573). However, because the query contained descriptive geographical and facility phrasing (*"Departed Facility in Memphis for 5 days"*), softmax probability was slightly dispersed across 8 classes, yielding a confidence of 0.48—just 0.02 shy of the 0.50 threshold.
- **Hypothesis for Improvement**:
  Implement confidence margin checking: if the top intent's score is >2x higher than the runner-up intent (even if absolute softmax is 0.48), treat it as a decisive plurality and allow auto-handling for benign self-service tracking inquiries.
