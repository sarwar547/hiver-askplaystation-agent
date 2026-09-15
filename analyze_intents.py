import pandas as pd
from pathlib import Path
from collections import Counter
import re


INPUT_FILE = Path("data/playstation_customer_support_pairs.csv")


def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():

    print("\n" + "=" * 70)
    print("PLAYSTATION INTENT DISCOVERY")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nSupport pairs: {len(df):,}")

    df["clean_message"] = (
        df["customer_message"]
        .fillna("")
        .apply(clean_text)
    )

    # --------------------------------------------------------
    # Keyword/topic discovery
    # --------------------------------------------------------

    keyword_groups = {

        "account_login": [
            "account",
            "login",
            "log in",
            "sign in",
            "password",
            "username",
            "email",
            "locked",
            "reset password"
        ],

        "billing_payment": [
            "payment",
            "pay",
            "billing",
            "charged",
            "charge",
            "credit card",
            "debit card",
            "card",
            "refund",
            "money"
        ],

        "subscription_ps_plus": [
            "ps plus",
            "playstation plus",
            "subscription",
            "renew",
            "renewal",
            "membership"
        ],

        "game_download_install": [
            "download",
            "install",
            "installation",
            "downloading",
            "won't download",
            "cant download"
        ],

        "game_purchase": [
            "bought",
            "buy",
            "purchase",
            "purchased",
            "store",
            "playstation store"
        ],

        "console_hardware": [
            "ps4",
            "ps5",
            "console",
            "controller",
            "dualshock",
            "dualsense",
            "disc",
            "eject",
            "overheat"
        ],

        "error_issue": [
            "error",
            "error code",
            "not working",
            "doesn't work",
            "doesnt work",
            "won't work",
            "wont work",
            "problem",
            "issue"
        ],

        "network_connection": [
            "internet",
            "wifi",
            "wi-fi",
            "network",
            "connection",
            "connect",
            "online"
        ],

        "refund_return": [
            "refund",
            "refunds",
            "return",
            "money back"
        ],

        "gift_card_code": [
            "gift card",
            "voucher",
            "code",
            "redeem",
            "redeemed"
        ],

        "dlc_content": [
            "dlc",
            "add-on",
            "addon",
            "expansion",
            "content"
        ],

        "security_privacy": [
            "hack",
            "hacked",
            "security",
            "stolen",
            "privacy",
            "2fa",
            "two factor",
            "verification"
        ],

        "general_help": [
            "help",
            "please help",
            "question",
            "support"
        ]
    }


    counts = Counter()

    examples = {}

    for _, row in df.iterrows():

        text = row["clean_message"]

        matched = []

        for intent, keywords in keyword_groups.items():

            if any(keyword in text for keyword in keywords):
                matched.append(intent)

        for intent in matched:

            counts[intent] += 1

            if intent not in examples:
                examples[intent] = []

            if len(examples[intent]) < 5:
                examples[intent].append(
                    row["customer_message"]
                )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOPIC FREQUENCY")
    print("=" * 70)

    for intent, count in counts.most_common():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"{intent:25} "
            f"{count:6,} "
            f"({percentage:5.1f}%)"
        )


    print("\n" + "=" * 70)
    print("EXAMPLES BY TOPIC")
    print("=" * 70)

    for intent, count in counts.most_common():

        print("\n" + "-" * 70)

        print(
            f"{intent.upper()} "
            f"({count:,} matches)"
        )

        for example in examples[intent]:

            print(
                f"\n• {example}"
            )


    print("\n" + "=" * 70)
    print("INTENT DISCOVERY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()