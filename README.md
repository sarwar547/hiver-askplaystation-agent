# AskPlayStation AI Customer Support Agent

An end-to-end AI customer-support agent built for the Hiver SDE Intern assignment using the Customer Support on Twitter dataset.

The system focuses on one brand, **AskPlayStation**, and performs three tasks:

1. Classifies an incoming customer message into a compact intent taxonomy.
2. Retrieves historically similar AskPlayStation support responses as evidence.
3. Decides whether the message should be auto-handled or escalated, with a reason.

---

## 1. Problem Framing

Customer-support conversations on Twitter are noisy, short, multi-turn, and often lack context.

Instead of attempting to build a fully autonomous support system, this project focuses on a narrow and measurable workflow:

**Customer message → Intent classification → Historical evidence retrieval → Draft response → Escalation decision**

### What this project intentionally does NOT build

- A fully autonomous customer-support system
- Account/payment actions
- Refund processing
- Authentication or identity verification
- Real-time access to PlayStation customer accounts
- Guaranteed resolution of customer issues
- A production deployment

The agent only drafts responses and recommends whether the case should be handled automatically or escalated.

---

## 2. Dataset

Source:

`thoughtvector/customer-support-on-twitter`

The dataset contains approximately 3M tweets across multiple customer-support brands.

For this project, the brand **AskPlayStation** was selected.

### Extracted data

- AskPlayStation support tweets: 19,098
- Customer tweets: 18,650
- Reconstructed customer-support pairs: 18,675
- Golden evaluation examples: 200
- Human-labelled evaluation examples: 200

Conversation pairs were reconstructed using the available Twitter conversation metadata and support/customer relationships.

---

## 3. Intent Taxonomy

The final taxonomy contains 12 intents:

| Intent | Description |
|---|---|
| `account_access` | Login, password, account access problems |
| `account_security` | Hacking, unauthorized access, bans/security concerns |
| `payments_billing` | Charges, payment methods, account funding |
| `refunds` | Refund and purchase cancellation requests |
| `ps_plus_subscription` | PlayStation Plus subscriptions and benefits |
| `game_purchase_store` | Game purchases, PlayStation Store and pricing |
| `download_installation` | Downloads, installations and corrupted downloads |
| `console_hardware` | Console, controller, disc and hardware issues |
| `network_psn` | PSN, network and connection problems |
| `codes_dlc_content` | Codes, vouchers, DLC and entitlements |
| `content_availability` | Availability, catalogue and release questions |
| `other` | Messages that do not contain enough information for a specific intent |

The `other` category is intentionally retained because many real support messages are short or ambiguous.

---

## 4. System Architecture

```text
                    Customer Message
                           |
                           v
                  Hybrid Intent Classifier
                    /              \
             High-signal rules    ML fallback
                    \              /
                           |
                           v
                  Predicted Intent
                           |
                           v
                TF-IDF Historical Retrieval
                           |
                           v
                 Similar Support Evidence
                           |
                           v
                 Escalation Policy
                    /              \
              Auto-handle        Escalate
                    \              /
                           v
                    Draft Response

## 6. Baselines

Two simple baselines were implemented.

### Majority-class baseline

The majority-class baseline predicts `other` for every example.

Results:

- Intent accuracy: **33.5%**
- Intent macro-F1: **4.18%**
- Action accuracy: **88.5%**

### Keyword baseline

A manually defined domain-keyword classifier was also evaluated.

Results:

- Intent accuracy: **96.0%**
- Intent macro-F1: **89.42%**

### Interpretation

The keyword baseline currently outperforms the hybrid agent on the 200-example golden set.

This is an important finding rather than a result to hide. It suggests that the current ML classifier does not yet provide enough additional classification value over carefully designed domain heuristics.

The hybrid system nevertheless provides capabilities beyond the keyword baseline:

- Historical support-response retrieval
- Evidence-backed response drafting
- Explicit auto-handle vs escalation decisions
- Escalation reasoning
- A unified support-agent pipeline

The result motivates further work on intent-boundary refinement, conversation context, and semantic retrieval.

