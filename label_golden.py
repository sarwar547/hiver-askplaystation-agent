import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/playstation_golden_set.csv")
OUTPUT_FILE = Path("data/playstation_golden_set_labeled.csv")

INTENTS = [
    "account_access",
    "account_security",
    "payments_billing",
    "refunds",
    "ps_plus_subscription",
    "game_purchase_store",
    "download_installation",
    "console_hardware",
    "network_psn",
    "codes_dlc_content",
    "content_availability",
    "other",
]


# --------------------------------------------------
# Intent keyword suggestions
# --------------------------------------------------

KEYWORDS = {
    "account_access": [
        "login", "log in", "sign in", "password",
        "account", "can't access", "locked out",
        "username", "email", "dob", "date of birth"
    ],

    "account_security": [
        "hack", "hacked", "security", "stolen",
        "unauthorized", "ban", "banned",
        "suspended", "suspension", "2fa",
        "two factor"
    ],

    "payments_billing": [
        "charged", "charge", "payment", "billing",
        "bill", "credit card", "debit card",
        "transaction"
    ],

    "refunds": [
        "refund", "refunds", "money back",
        "return", "cancel purchase"
    ],

    "ps_plus_subscription": [
        "ps plus", "playstation plus", "ps+",
        "subscription", "renewal", "renew",
        "monthly games", "plus games"
    ],

    "game_purchase_store": [
        "buy", "bought", "purchase", "purchased",
        "store", "game sharing", "digital game",
        "price", "sale"
    ],

    "download_installation": [
        "download", "downloading", "install",
        "installation", "installing",
        "corrupt", "corrupted", "update"
    ],

    "console_hardware": [
        "ps4", "ps5", "console", "controller",
        "disc", "eject", "power", "turns on",
        "overheat", "button", "buttons"
    ],

    "network_psn": [
        "network", "internet", "wifi", "wi-fi",
        "connection", "connect", "psn",
        "nw-", "dns", "online"
    ],

    "codes_dlc_content": [
        "code", "voucher", "redeem", "dlc",
        "add-on", "addon", "points",
        "entitlement"
    ],

    "content_availability": [
        "available", "availability", "coming",
        "release", "when", "added",
        "catalogue", "catalog", "offerings"
    ],

    "other": []
}


def suggest_intent(text):

    text = str(text).lower()

    scores = {}

    for intent, keywords in KEYWORDS.items():

        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[intent] = score

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "other"

    return best


def suggest_action(intent, text):

    text = str(text).lower()

    escalation_words = [
        "banned",
        "suspended",
        "hacked",
        "stolen",
        "refund",
        "charged",
        "payment",
        "can't access",
        "locked out",
        "not received",
        "missing",
        "unauthorized"
    ]

    if any(word in text for word in escalation_words):
        return "escalate"

    if intent in [
        "account_security",
        "refunds",
        "payments_billing"
    ]:
        return "escalate"

    return "auto_handle"


def suggest_difficulty(text, intent):

    text = str(text).lower()

    if intent == "other":
        return "hard"

    if any(word in text for word in [
        "banned",
        "hacked",
        "refund",
        "charged",
        "not received"
    ]):
        return "hard"

    if len(text) > 150:
        return "hard"

    if len(text) > 70:
        return "medium"

    return "easy"


# --------------------------------------------------
# Load existing labels if available
# --------------------------------------------------

if OUTPUT_FILE.exists():

    print("Resuming existing labels...")

    df = pd.read_csv(
        OUTPUT_FILE,
        dtype=str
    ).fillna("")

else:

    print("Creating new labeling file...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str
    ).fillna("")


# --------------------------------------------------
# Find customer text column automatically
# --------------------------------------------------

possible_text_columns = [
    "customer_text",
    "customer_message",
    "text",
    "message",
    "customer_tweet",
    "customer"
]

TEXT_COLUMN = None

for column in possible_text_columns:

    if column in df.columns:
        TEXT_COLUMN = column
        break


if TEXT_COLUMN is None:

    print("\nERROR: Could not find customer message column.")

    print("\nAvailable columns:")
    print(list(df.columns))

    input(
        "\nPress ENTER to exit..."
    )

    raise SystemExit


print(f"Using customer text column: {TEXT_COLUMN}")


# --------------------------------------------------
# Create annotation columns
# --------------------------------------------------

for column in [
    "intent",
    "expected_action",
    "reason_for_escalation",
    "difficulty"
]:

    if column not in df.columns:
        df[column] = ""

    df[column] = df[column].fillna("").astype(str)


# --------------------------------------------------
# Label examples
# --------------------------------------------------

for i in range(len(df)):

    # Skip already labelled examples
    if str(df.loc[i, "intent"]).strip():
        continue

    text = str(df.loc[i, TEXT_COLUMN])

    suggested_intent = suggest_intent(text)

    suggested_action = suggest_action(
        suggested_intent,
        text
    )

    suggested_difficulty = suggest_difficulty(
        text,
        suggested_intent
    )

    print("\n" + "=" * 70)
    print(f"Example {i + 1}/{len(df)}")
    print("=" * 70)

    print("\nCUSTOMER:")
    print(text)

    print("\nAI SUGGESTION")
    print("-" * 30)

    print(
        f"Intent     : {suggested_intent}"
    )

    print(
        f"Action     : {suggested_action}"
    )

    print(
        f"Difficulty : {suggested_difficulty}"
    )

    print("\nPress ENTER to accept.")
    print("Type a correction if needed.")

    # Intent
    intent_input = input(
        f"\nIntent [{suggested_intent}]: "
    ).strip()

    if intent_input:
        intent = intent_input
    else:
        intent = suggested_intent

    # Action
    action_input = input(
        f"Action [{suggested_action}] "
        "(auto_handle/escalate): "
    ).strip()

    if action_input:
        action = action_input
    else:
        action = suggested_action

    # Reason
    if action == "escalate":

        reason_input = input(
            "Escalation reason "
            "(ENTER for default): "
        ).strip()

        if reason_input:
            reason = reason_input
        else:
            reason = (
                "Requires account-specific verification "
                "or further troubleshooting."
            )

    else:

        reason = ""

    # Difficulty
    difficulty_input = input(
        f"Difficulty [{suggested_difficulty}] "
        "(easy/medium/hard): "
    ).strip()

    if difficulty_input:
        difficulty = difficulty_input
    else:
        difficulty = suggested_difficulty

    # Save immediately
    df.loc[i, "intent"] = intent
    df.loc[i, "expected_action"] = action
    df.loc[i, "reason_for_escalation"] = reason
    df.loc[i, "difficulty"] = difficulty

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n✓ Saved.")


print("\n" + "=" * 70)
print("FINISHED")
print("=" * 70)

print(
    f"\nSaved labelled dataset to:\n{OUTPUT_FILE}"
)