# Diagnostic Analysis: Why Escalation Accuracy is Low

## 1. Executive Summary of Escalation Breakdown
Across the 200 evaluation examples in the Golden Evaluation Set:

| Metric | Count | Percentage of Benchmark |
| :--- | :--- | :--- |
| **Total Evaluation Samples** | 200 | 100.0% |
| **Ground-Truth AUTO_HANDLE Cases** | 133 | 66.5% |
| **Ground-Truth ESCALATE Cases** | 67 | 33.5% |
| **System Predicted AUTO_HANDLE** | 36 | 18.0% |
| **System Predicted ESCALATE** | 164 | 82.0% |
| **True Auto-Handles (Correctly Automated)** | 30 | 15.0% |
| **True Escalations (Correctly Escalated)** | 61 | 30.5% |
| **False Auto-Handles (Dangerous Misses)** | **6** | **3.0%** (Safety preserved: 91.0% recall) |
| **False Escalations (Unnecessary Human Handoffs)** | **103** | **51.5%** (Efficiency bottleneck) |

**Resulting Baseline Escalation Metrics**:
- **Escalation Accuracy**: `45.5%` ($[30 + 61] / 200$)
- **Escalation Precision**: `42.13%` ($61 / [61 + 103]$)
- **Escalation Recall**: `92.59%` ($61 / [61 + 6]$)
- **Automation Rate**: Only `18.0%` of inbound inquiries are handled automatically, even though `66.5%` are routine self-service issues.

---

## 2. Root Cause Analysis: Which Rules & Thresholds Cause Over-Escalation?

An analysis of the decision reasons for all 103 false escalations reveals that over-escalation is driven by two numeric threshold bottlenecks:

```
Total False Escalations: 103
├── Trigger 1: Intent Confidence Below Threshold (conf < 0.65)  --> 82 cases (79.6%)
└── Trigger 2: Retrieval Similarity Below Threshold (sim < 0.60) --> 21 cases (20.4%)
    └── Trigger 3: Specific Domain Rules                      --> 0 cases  (0.0%)
```

### Cause 1: Intent Confidence Threshold is Overly Strict (82 Cases)
- **Current Setting**: `INTENT_CONFIDENCE_THRESHOLD = 0.65`.
- **Mechanism**: In an 8-class classification system, uniform random chance is $1/8 = 12.5\%$. When calibrated softmax is applied over 8 semantic prototypes, a top intent probability of $0.35$ to $0.55$ already represents a commanding plurality (often 3x to 4x higher than the runner-up).
- Requiring $\ge 0.65$ caused **82 routine inquiries** to be rejected, even when the model's top predicted intent was 100% accurate.
- **Intent Impact**:
  - `product_inquiry_availability`: **25 out of 25 cases** were escalated because product specification queries ("Will this case fit...", "Is this 110V...", "What are the dimensions?") lack brand-specific exclamation tokens and yield calibrated softmax scores around $0.30 - 0.50$.
  - `damaged_defective_item`: 16 benign cases escalated.
  - `payment_billing_issue`: 14 benign self-service cases escalated.
  - `delivery_delay_tracking`: 13 benign cases escalated.

**Real Examples Blocked by Confidence < 0.65**:
1. *Case ID #151*: *"Will this OtterBox case for iPhone 14 also fit an iPhone 15?"*
   - Model Prediction: `product_inquiry_availability` (Confidence: `0.44`) $\to$ **Correct Intent!**
   - System Decision: `ESCALATE` (*"Intent classification confidence (0.44) is below operational safety threshold (0.65)."*)
   - Gold: `AUTO_HANDLE` (Standard catalog dimension compatibility).
2. *Case ID #160*: *"What is the weight capacity of this steel office chair?"*
   - Model Prediction: `product_inquiry_availability` (Confidence: `0.41`) $\to$ **Correct Intent!**
   - System Decision: `ESCALATE` (*"Intent classification confidence (0.41) is below operational safety threshold (0.65)."*)
   - Gold: `AUTO_HANDLE` (Standard product page specification).
3. *Case ID #85*: *"How do I update the expiration date and CVV for my default payment method?"*
   - Model Prediction: `payment_billing_issue` (Confidence: `0.52`) $\to$ **Correct Intent!**
   - System Decision: `ESCALATE` (*"Intent classification confidence (0.52) is below operational safety threshold (0.65)."*)
   - Gold: `AUTO_HANDLE` (Direct user to *Your Account > Your Payments*).

---

### Cause 2: Retrieval Similarity Threshold is Overly Strict for Social Media Text (21 Cases)
- **Current Setting**: `RETRIEVAL_SIMILARITY_THRESHOLD = 0.60`.
- **Mechanism**: The historical knowledge base consists of real customer service tweets from AmazonHelp. Twitter customer messages are concise and conversational. Dense bi-encoder cosine similarities (`all-MiniLM-L6-v2`) for relevant Twitter matches typically range between $0.50$ and $0.59$.
- A rigid cutoff of $0.60$ rejected excellent historical precedents with similarities of $0.594, 0.592, 0.567, 0.540$.

**Real Examples Blocked by Similarity < 0.60**:
1. *Case ID #27*: *"I dropped off my return at Whole Foods 3 days ago. When will the refund hit my card?"*
   - Top Retrieved Precedent: Similarity `0.592` (Explains standard return drop-off refund timeframe).
   - System Decision: `ESCALATE` (*"Top historical resolution similarity (0.59) is below grounding threshold (0.60)."*)
   - Gold: `AUTO_HANDLE` (Routine standard refund processing SLA).
2. *Case ID #49*: *"Where can I view the itemized breakdown of my issued refund?"*
   - Top Retrieved Precedent: Similarity `0.594` (Directs customer to *Your Orders > Refund Summary*).
   - System Decision: `ESCALATE` (*"Top historical resolution similarity (0.59) is below grounding threshold (0.60)."*)
   - Gold: `AUTO_HANDLE` (Self-service navigational link).
3. *Case ID #8*: *"Is there a delay on shipments heading to the Chicago area due to snowstorms?"*
   - Top Retrieved Precedent: Similarity `0.540` (Confirms carrier weather delays).
   - System Decision: `ESCALATE` (*"Top historical resolution similarity (0.54) is below grounding threshold (0.60)."*)
   - Gold: `AUTO_HANDLE` (Informational weather advisory).

---

## 3. False Auto-Handle Analysis (The 6 Safety-Critical Errors)
The current conservative policy permitted only 6 false auto-handles out of 67 escalation cases. These 6 represent subtle edge cases:
- Case ID #31: Returned two orders in same box (Discrepancy).
- Case ID #50: Refund amount $20 less than paid (Partial credit dispute).
- Case ID #135: Charged $14.99 after free trial (Promotional billing dispute).
- Case ID #140: Billed after cancellation (Recurring billing glitch).
- Case ID #146: Double billed for Prime on two cards (Account deduplication).
- Case ID #150: Promised 1-month extension not received (Unfulfilled promise).

**Key Takeaway**: These 6 false auto-handles slipped through because their confidence was high ($\ge 0.88$) and similarity was high ($\ge 0.64$), but they contained subtle financial or ledger dispute language that lacked keyword escalation triggers.

---

## 4. Improvement Roadmap

To fix the over-escalation problem while strictly preserving safety:
1. **Calibrate Numeric Thresholds on a Dedicated Development Set**:
   - Split the 200 golden examples into 70% Development (140 examples) and 30% Held-Out Test (60 examples).
   - Perform a systematic grid search on the 140 development examples across intent confidence ($[0.30, 0.65]$) and retrieval similarity ($[0.45, 0.60]$).
2. **Refine Domain Escalation Rules**:
   - Add targeted rules for financial ledger discrepancies (e.g., `"less than"`, `"billed after"`, `"double billed"`, `"same box"`) to safeguard against the 6 false auto-handles.
   - Whitelist common benign self-service phrasing (e.g. `"how do I update"`, `"where can I view"`, `"what is the weight"`, `"will this fit"`) so safe queries are not blocked by generic keyword triggers.
3. **Freeze Policy on Development Set**:
   - Validate on the held-out 60-example test set to confirm generalization without overfitting or data leakage.
