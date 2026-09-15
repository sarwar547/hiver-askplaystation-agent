# AskPlayStation AI Customer Support Agent — Evaluation Report

## 1. Framing

This project builds a narrow AI customer-support agent for AskPlayStation using the Customer Support on Twitter dataset.

The system performs three tasks:

1. Classifies customer messages into a compact 12-intent taxonomy.
2. Retrieves historically similar AskPlayStation support interactions as evidence.
3. Drafts a response and decides whether the case should be auto-handled or escalated.

The objective is not to build a fully autonomous customer-support system. The prototype focuses on measurable classification, evidence retrieval, response drafting, and conservative escalation.

### What I did not build

- Account-level actions
- Refund processing
- Payment processing
- Identity verification
- Real-time PlayStation account access
- Guaranteed issue resolution
- Production deployment

These require systems and permissions that are outside the scope of this prototype.

---

## 2. Data and Golden Set

The original Customer Support on Twitter dataset contains approximately 3M tweets across many brands.

I selected AskPlayStation.

The extracted data contains:

- 19,098 AskPlayStation support tweets
- 18,650 customer tweets
- 18,675 reconstructed customer-support pairs

A 200-example golden set was created for evaluation.

The golden set was sampled from the reconstructed AskPlayStation interactions and manually labelled using the 12-intent taxonomy. The labels were reviewed against the actual customer message and surrounding support context where available.

The 12 intents are:

- account_access
- account_security
- payments_billing
- refunds
- ps_plus_subscription
- game_purchase_store
- download_installation
- console_hardware
- network_psn
- codes_dlc_content
- content_availability
- other

---

## 3. System

### Classification

The agent uses a hybrid approach:

- High-signal domain rules for obvious support cases
- TF-IDF word and bigram features
- Logistic Regression with balanced class weights

The rules handle cases where domain-specific language is particularly informative, while the ML classifier provides broader coverage.

### Retrieval

Historical customer-support interactions are indexed using TF-IDF.

For an incoming customer message, the system retrieves similar historical interactions and exposes the corresponding support response as evidence.

This provides grounding for the drafted answer instead of allowing unrestricted response generation.

### Escalation

The current policy conservatively escalates:

- Account security issues
- Refund requests
- Payment/billing issues

The agent does not claim to perform actions it cannot actually perform.

---

## 4. Results

Evaluation was performed on 200 human-labelled examples.

| System | Intent Accuracy | Intent Macro-F1 |
|---|---:|---:|
| Majority baseline | 33.5% | 4.18% |
| Hybrid agent | 89.5% | 82.76% |
| Keyword baseline | 96.0% | 89.42% |

The hybrid agent achieved:

**Intent accuracy: 89.5%**

**Intent macro-F1: 82.76%**

**Action accuracy: 95.0%**

The action accuracy measures whether the agent's auto-handle versus escalation decision agrees with the golden-set action label.

### Per-intent observations

The strongest categories were:

- console_hardware — 97.14% F1
- other — 94.29% F1
- account_security — 90.91% F1
- game_purchase_store — 90.91% F1
- codes_dlc_content — 88.89% F1

The weakest categories were:

- content_availability — 50.0% F1
- ps_plus_subscription — 57.14% F1
- payments_billing — 66.67% F1

---

## 5. Baseline Comparison

The majority baseline performs poorly because it predicts `other` for every example:

- Accuracy: 33.5%
- Macro-F1: 4.18%

A manually constructed keyword baseline performs surprisingly strongly:

- Accuracy: 96.0%
- Macro-F1: 89.42%

Therefore, the current hybrid classifier does not outperform the keyword baseline on this golden set.

This is an important finding.

The keyword baseline benefits from strong domain-specific vocabulary such as PSN error codes, refund terms, download terms, controller/console terms, and PS Plus terminology.

The hybrid agent provides additional capabilities beyond classification:

- Historical response retrieval
- Evidence for responses
- Auto-handle versus escalation decisions
- Escalation reasoning
- A unified support-agent pipeline

The baseline result therefore identifies classification quality and intent-boundary design as the main areas for improvement.

---

## 6. What Is Misleading About My Headline Number?

The headline intent accuracy of 89.5% does not mean that 89.5% of real customer conversations can be safely automated.

First, the golden set is imbalanced. The `other` intent contains 67 of the 200 examples. A model can therefore achieve relatively high accuracy while performing worse on smaller categories.

For this reason, macro-F1 is reported alongside accuracy.

Second, the keyword baseline achieves 96.0% accuracy and 89.42% macro-F1. Therefore, the current hybrid classifier is not superior to a strong domain heuristic on this evaluation set.

Third, intent accuracy does not measure response quality or whether the escalation decision is safe.

The 95.0% action accuracy should also not be interpreted as proof that the agent can safely resolve 95% of customer cases. The action policy is evaluated only against the 200-example golden set and is deliberately conservative.

---

## 7. Top Five Failure Modes

### Failure 1 — Content availability vs PS Plus

Example:

> "when are the ps plus games going to be revealed for next month?"

The message contains a strong PS Plus signal but the actual question is about availability/release timing.

The classifier can therefore predict `ps_plus_subscription` instead of `content_availability`.

**Hypothesis:** lexical signals such as "PS Plus" dominate the more subtle question intent.

**Improvement:** introduce explicit precedence rules for release/availability questions and collect more labelled examples for this boundary.

---

### Failure 2 — Network error codes

Example:

> "Still giving me error code (NW-31295-0)"

The word "code" can cause confusion with voucher/DLC/code-related messages.

**Hypothesis:** generic "code" terminology overlaps with the `codes_dlc_content` intent.

**Improvement:** recognize PlayStation network error-code patterns such as `NW-` and route them to `network_psn`.

---

### Failure 3 — Account access vs account security

Messages can combine hacking, changed credentials, missing games, and inability to access an account.

This creates ambiguity between `account_access` and `account_security`.

**Hypothesis:** real customer conversations do not always respect clean taxonomy boundaries.

**Improvement:** apply a security-first policy and use recent conversation context before selecting the final intent.

---

### Failure 4 — Short/context-dependent messages

Examples such as:

> "thank you"

or:

> "I'll try that when I'm home"

may be impossible to classify correctly without the previous conversation.

**Hypothesis:** the latest customer message alone does not contain enough semantic information.

**Improvement:** include the previous one or two conversation turns in both classification and retrieval.

---

### Failure 5 — Small and overlapping categories

`payments_billing` and `ps_plus_subscription` contain relatively few golden examples, making their F1 scores unstable.

Some messages also contain overlapping purchase, subscription, and payment terminology.

**Hypothesis:** both limited sample size and overlapping vocabulary contribute to errors.

**Improvement:** create a larger stratified golden set and refine the intent definitions using hard negative examples.

---

## 8. LLM-as-Judge

An LLM-as-judge harness was implemented for a 50-example sample.

The rubric evaluates:

- Groundedness
- Correctness
- Actionability
- Safety
- No hallucination
- Overall quality

The human evaluation sample is stored at:

`reports/judge_human_sample.csv`

The external OpenAI-compatible endpoint returned HTTP 429 rate-limit responses during execution.

Therefore, LLM judge scores and human-vs-LLM agreement are **not reported**.

No agreement number was fabricated.

The harness can be executed when a working API endpoint with available quota is provided.

---

## 9. Decision Log

### 1. Selected AskPlayStation
The brand had enough customer-support interactions to create a meaningful evaluation set.

### 2. Reconstructed customer-support pairs
The raw dataset was converted into customer/support interactions rather than treating tweets independently.

### 3. Used a compact taxonomy
A 12-intent taxonomy keeps the problem measurable while covering the major support topics.

### 4. Kept an `other` category
Short and ambiguous messages should not be forced into an arbitrary intent.

### 5. Used a hybrid classifier
Domain rules provide strong signals while Logistic Regression handles broader language patterns.

### 6. Used TF-IDF retrieval
TF-IDF provides a lightweight and reproducible retrieval system.

### 7. Grounded replies in historical support
Historical support behaviour provides evidence for response drafting.

### 8. Conservative escalation
Security, refunds, and billing cases are escalated because the prototype cannot safely perform account or financial actions.

### 9. Created a human-labelled golden set
A manually labelled set provides a more meaningful evaluation than weak labels alone.

### 10. Added a keyword baseline
The baseline tests whether the ML component actually adds classification value.

### 11. Reported macro-F1
Macro-F1 prevents the large `other` class from dominating interpretation.

### 12. Separated classification from action
Intent prediction and escalation are evaluated as different tasks.

### 13. Did not fabricate LLM agreement
The external endpoint returned HTTP 429, so unsupported agreement statistics were excluded.

### 14. Kept the system narrow
The objective is an evaluation-ready prototype, not a production support platform.

---

## 10. One More Week

If another week were available, I would prioritize:

### 1. Conversation context
Include recent customer/support turns in classification and retrieval.

### 2. Better taxonomy boundaries
Collect additional hard examples for content availability, PS Plus, billing, and account access/security.

### 3. Semantic retrieval
Compare TF-IDF retrieval against embedding-based retrieval.

### 4. Larger evaluation
Expand the golden set and ensure adequate examples for every intent.

### 5. Response-quality evaluation
Run the LLM judge on a larger sample and complete human-vs-LLM agreement analysis.

### 6. Robustness testing
Evaluate multilingual messages, short messages, error codes, ambiguous requests, and adversarial inputs.

---

## 11. Reproducibility

From the project root:

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python baselines.py
python judge_prepare.py