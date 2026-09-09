# Golden Evaluation Set: Sampling & Labeling Methodology

## 1. Dataset Overview & Scope
- **Brand Target**: `@AmazonHelp` (E-Commerce Customer Support on Twitter)
- **Total Golden Examples**: Exactly 200 hand-curated and audited conversation pairs.
- **Data Source**: Thought Vector Customer Support on Twitter (`twcs.csv`), filtered to clean English interactions with minimum 25 characters per turn.
- **Primary Schema**:
  - `id`: Unique identifier (`GOLD-001` through `GOLD-200`)
  - `tweet_id`: Original Twitter conversation ID from Kaggle corpus
  - `customer_text`: The customer's incoming message
  - `gold_intent`: Hand-verified intent from the 6-class canonical taxonomy
  - `gold_escalate`: Boolean triage decision (`True` = Human Specialist, `False` = Auto-Handled)
  - `gold_escalate_reason`: Operational justification for the triage decision
  - `reference_reply`: The actual historical reply published by `@AmazonHelp`
  - `sentiment`: Customer sentiment polarity (`POSITIVE`, `NEUTRAL`, `NEGATIVE`, `VERY_NEGATIVE`)
  - `stratum`: Stratified sampling bucket
  - `human_groundedness_score`: Human rating (1–5)
  - `human_empathy_score`: Human rating (1–5)
  - `human_actionability_score`: Human rating (1–5)
  - `human_safety_score`: Human rating (1–5)
  - `notes`: Specific context, edge cases, and reasoning

---

## 2. Intent Taxonomy
The taxonomy was derived bottom-up by inspecting real customer tweets to `@AmazonHelp`:

| Intent Class | Description | Volume in Golden Set | % of Golden Set | Auto-Handled | Escalated |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`ORDER_TRACKING_AND_DELAY`** | Package tracking, delivery delays, courier inquiries, transit status | 45 | 22.5% | 22 | 23 |
| **`REFUND_AND_RETURN`** | Initiating returns, refund turnaround time, return labels, billing | 35 | 17.5% | 17 | 18 |
| **`DAMAGED_OR_WRONG_ITEM`** | Broken items, empty packages, wrong product shipped, missing parts | 35 | 17.5% | 17 | 18 |
| **`ACCOUNT_AND_SECURITY`** | Account lockouts, 2FA/OTP login failures, unauthorized access | 35 | 17.5% | 15 | 20 |
| **`PRODUCT_AND_TECHNICAL`** | Device troubleshooting (Kindle, Fire TV, Echo/Alexa, Prime Video) | 25 | 12.5% | 13 | 12 |
| **`FEEDBACK_AND_GENERAL`** | Feedback, brand praise, store policies, general inquiries | 25 | 12.5% | 10 | 15 |
| **Total** | | **200** | **100.0%** | **94 (47.0%)** | **106 (53.0%)** |

---

## 3. Stratification & Triage Strategy (Auto-Handle vs. Escalate)
A critical requirement of customer support AI is deciding what can be safely automated versus what presents financial, security, or customer satisfaction risk requiring human intervention:

- **Auto-Handled (94 examples / 47.0%)**:
  - Routine order tracking while inside the standard carrier delivery window (`ROUTINE_TRACKING_IN_TRANSIT`).
  - Standard return initiation instructions and links to `amazon.com/returns` (`ROUTINE_RETURN_POLICY_OR_PORTAL`).
  - Self-serve replacement workflows under `Your Orders` (`ROUTINE_REPLACEMENT_ELIGIBLE`).
  - Step-by-step device restart and update troubleshooting for Kindle/Fire TV (`DEVICE_TROUBLESHOOTING_SELF_SERVE`).
  - Standard account management and preference configuration (`GENERAL_ACCOUNT_SETTINGS_SELF_SERVE`).
  - Courteous acknowledgments of positive customer feedback (`POSITIVE_BRAND_PRAISE`).

- **Escalated to Human Specialist (106 examples / 53.0%)**:
  - **Carrier Delivery Disputes**: Package marked as "delivered" by courier but customer did not receive it (doorstep theft or driver scan error requiring courier investigation).
  - **Account Security & Compromise**: Suspicious login attempts, 2FA lockout loops, suspected credential takeover.
  - **Financial & Billing Disputes**: Multiple unauthorized charges, missing refunds past SLA.
  - **Severe Damage or Missing High-Value Goods**: Empty package seals, ruined or hazardous items.
  - **High Churn / Hostility**: Explicit profanity, legal threats ("lawyer", "consumer court"), threats to cancel Amazon Prime.
  - **PII Deflection**: Tweets exposing private phone numbers, email addresses, or 17-digit order numbers in public.

---

## 4. Quality Control & Labeling Process
1. **Deduplication & Hygiene**: All tweets were normalized, stripped of non-printable characters, and checked for pure English syntax.
2. **Ambiguity Resolution**: Multi-intent tweets (e.g., "My package was delayed AND damaged") were labelled by primary root cause (`DAMAGED_OR_WRONG_ITEM`) while flagging escalation due to complexity.
3. **Reference Response Auditing**: Each historical brand response was scored across 4 dimensions (Groundedness, Empathy, Actionability, Safety) to establish human ground truth for LLM-as-judge calibration.
