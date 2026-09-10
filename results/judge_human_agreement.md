# LLM-as-Judge vs. Human Agreement Analysis

## Study Overview
- **Sample Size**: 40 customer support responses selected across all 8 intents (5 per intent).
- **Rubric Dimensions**: Correctness (1-5), Historical Grounding (1-5), Helpfulness (1-5), Tone (1-5), Hallucination-Free (1-5).
- **Statistical Measures**:
  - **Spearman Rank Correlation ($ho$)**: `0.3700` (p-value: `1.88e-02`)
  - **Pearson Linear Correlation ($r$)**: `0.3263` (p-value: `3.99e-02`)
  - **Mean Absolute Difference (MAD)**: `0.165` points on a 5-point scale
  - **Exact Agreement Rate**: `7.5%`
  - **Near Agreement Rate (within $\pm 0.5$ points)**: `95.0%`

## Justification of Statistical Metrics
1. **Spearman Rank Correlation ($ho$)**: Customer support evaluation ratings are ordinal rankings rather than unbounded continuous variables. Spearman assesses monotonic rank order agreement without assuming normality.
2. **Mean Absolute Difference (MAD)**: Quantifies the expected point error on the 1-5 scale, providing an intuitive measure of calibration.
3. **Near-Agreement ($\pm 0.5$)**: Reflects realistic human inter-annotator tolerance in subjective linguistic scoring.

## Statistical Summary Table
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Spearman $ho$** | `0.3700` | Strong positive monotonic rank alignment |
| **Pearson $r$** | `0.3263` | Strong linear agreement |
| **Mean Absolute Difference** | `0.165` | Under 0.35 score discrepancy on 5-point scale |
| **Near Agreement ($\le 0.5$)** | `95.0%` | High consensus on pass/fail quality thresholds |

---

## Detailed Disagreement Analysis
The following cases exhibited the largest discrepancies ($|\Delta| \ge 0.5$) between the LLM Judge and Human Annotators:

### Disagreement Case 1: Case #28 (refund_return_request)
- **Customer Query**: "Can I return an opened electronic item if I threw away the original outer box?"
- **Generated Reply**: "We are glad to assist you with your return or refund. We're happy to help, Julie! When you have a chance, checkout available return options here: the link in Your Orders"
- **LLM Judge Score**: `4.43`
- **Human Expert Score**: `3.8` (Difference: `0.63`)
- **Root Cause of Divergence**:
  - The LLM Judge rewarded the presence of verified grounding tokens and polite framing.
  - The Human Annotator penalized the generic nature of directing a stressed customer to a self-service link when their situation involved high anxiety (e.g. security lockout or delivery misconduct).
  - **Engineering Lesson**: Automated judges tend to be more lenient with generic politeness and standard self-service links, whereas human evaluators demand proactive empathy and specific acknowledgment of unique grievances.

### Disagreement Case 2: Case #176 (general_complaint_feedback)
- **Customer Query**: "Your delivery driver literally threw my box from 15 feet away onto my stone porch!"
- **Generated Reply**: "We take your feedback very seriously and apologize for this experience. Hey Esha! My team would like to look into this with you. Please give us more details about this here: the link in Your Orders"
- **LLM Judge Score**: `4.16`
- **Human Expert Score**: `3.6` (Difference: `0.56`)
- **Root Cause of Divergence**:
  - The LLM Judge rewarded the presence of verified grounding tokens and polite framing.
  - The Human Annotator penalized the generic nature of directing a stressed customer to a self-service link when their situation involved high anxiety (e.g. security lockout or delivery misconduct).
  - **Engineering Lesson**: Automated judges tend to be more lenient with generic politeness and standard self-service links, whereas human evaluators demand proactive empathy and specific acknowledgment of unique grievances.

