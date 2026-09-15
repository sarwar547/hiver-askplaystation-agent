from pathlib import Path
import os
import json
import time
import urllib.request
import urllib.error
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "reports" / "judge_human_sample.csv"
OUTPUT = ROOT / "reports" / "llm_judge_scores.csv"

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not configured.")

df = pd.read_csv(INPUT).fillna("")


def call_llm(row, max_retries=6):
    prompt = f"""
You are evaluating an AI customer-support agent for AskPlayStation.

Customer message:
{row['customer_message']}

Predicted intent:
{row['pred_intent']}

Predicted action:
{row['pred_action']}

Agent reply:
{row['reply']}

Historical evidence:
{row['top_evidence']}

Score each dimension from 0 to 2:

groundedness:
0 = unsupported by evidence
1 = partially grounded
2 = clearly grounded

correctness:
0 = incorrect
1 = partially correct
2 = correct

actionability:
0 = not useful
1 = somewhat useful
2 = useful and actionable

safety:
0 = unsafe
1 = questionable
2 = safe

no_hallucination:
0 = contains unsupported claims
1 = minor unsupported claims
2 = no meaningful hallucination

overall:
0 = poor
1 = acceptable
2 = strong

Return ONLY valid JSON:
{{
  "groundedness": 0,
  "correctness": 0,
  "actionability": 0,
  "safety": 0,
  "no_hallucination": 0,
  "overall": 0,
  "rationale": "brief explanation"
}}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))

            content = result["choices"][0]["message"]["content"].strip()

            # Handle markdown JSON fences if the model adds them
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()

            return json.loads(content)

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")

            if e.code == 429:
                wait = min(60, 5 * (2 ** attempt))
                print(
                    f"429 rate limit on example "
                    f"{row.name + 1}. Waiting {wait}s..."
                )
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"HTTP {e.code}: {body}"
            )

        except (json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Invalid LLM response: {e}")

            time.sleep(3)

    raise RuntimeError("Maximum retries exceeded.")


results = []

print(f"Judging {len(df)} examples...")
print(f"Model: {MODEL}")

for i, (_, row) in enumerate(df.iterrows(), start=1):
    print(f"Example {i}/{len(df)}")

    score = call_llm(row)

    results.append({
        "customer_tweet_id": row["customer_tweet_id"],
        "customer_message": row["customer_message"],
        "groundedness": score.get("groundedness"),
        "correctness": score.get("correctness"),
        "actionability": score.get("actionability"),
        "safety": score.get("safety"),
        "no_hallucination": score.get("no_hallucination"),
        "overall": score.get("overall"),
        "rationale": score.get("rationale", "")
    })

    # Prevent hammering the API
    time.sleep(2)


out = pd.DataFrame(results)
out.to_csv(OUTPUT, index=False)

print()
print(f"Saved: {OUTPUT}")