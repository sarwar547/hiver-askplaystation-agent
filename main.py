import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("twcs_askplaystation.csv")

OUTPUT_DIR = Path("data")

BRAND = "AskPlayStation"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING ASKPLAYSTATION DATA")
    print("=" * 70)

    if not INPUT_FILE.exists():

        print("\nERROR: Dataset not found:")
        print(INPUT_FILE.resolve())

        return None

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(f"\nRows loaded: {len(df):,}")

    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ============================================================
# STEP 2 — CLEAN DATA
# ============================================================

def clean_data(df):

    print("\n" + "=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Tweet IDs
    # --------------------------------------------------------

    df["tweet_id"] = (
        df["tweet_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    # --------------------------------------------------------
    # Author IDs
    # --------------------------------------------------------

    df["author_id"] = (
        df["author_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Inbound
    # --------------------------------------------------------

    df["inbound"] = (
        df["inbound"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False
        })
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="mixed",
        errors="coerce"
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Relationship columns
    # --------------------------------------------------------

    df["in_response_to_tweet_id"] = (
        df["in_response_to_tweet_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["response_tweet_id"] = (
        df["response_tweet_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    print("\nInbound values:")

    print(
        df["inbound"].value_counts(dropna=False)
    )

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    return df


# ============================================================
# STEP 3 — BASIC ANALYSIS
# ============================================================

def analyze_data(df):

    print("\n" + "=" * 70)
    print("BASIC DATASET ANALYSIS")
    print("=" * 70)

    customer_df = df[
        df["inbound"] == True
    ]

    support_df = df[
        df["author_id"] == BRAND
    ]

    print(
        f"\nTotal tweets: {len(df):,}"
    )

    print(
        f"Customer tweets: {len(customer_df):,}"
    )

    print(
        f"AskPlayStation tweets: {len(support_df):,}"
    )

    print(
        f"Unique authors: "
        f"{df['author_id'].nunique():,}"
    )


# ============================================================
# STEP 4 — BUILD TWEET LOOKUP
# ============================================================

def build_tweet_lookup(df):

    print("\n" + "=" * 70)
    print("BUILDING TWEET LOOKUP")
    print("=" * 70)

    # Remove duplicate IDs if any

    df = df.drop_duplicates(
        subset=["tweet_id"],
        keep="first"
    )

    tweet_lookup = {}

    for _, row in df.iterrows():

        tweet_id = row["tweet_id"]

        tweet_lookup[tweet_id] = row

    print(
        f"Tweet lookup entries: "
        f"{len(tweet_lookup):,}"
    )

    return tweet_lookup


# ============================================================
# STEP 5 — FIND CUSTOMER → SUPPORT PAIRS
# ============================================================

def find_customer_support_pairs(df, tweet_lookup):

    print("\n" + "=" * 70)
    print("FINDING CUSTOMER → ASKPLAYSTATION PAIRS")
    print("=" * 70)

    support_df = df[
        df["author_id"] == BRAND
    ].copy()

    pairs = []

    # --------------------------------------------------------
    # Method 1:
    #
    # AskPlayStation tweet points backwards to customer tweet
    #
    # in_response_to_tweet_id
    # --------------------------------------------------------

    for _, support_row in support_df.iterrows():

        parent_value = (
            support_row["in_response_to_tweet_id"]
        )

        if not parent_value:
            continue

        if parent_value in ["nan", "None"]:
            continue

        # Handle comma-separated IDs

        parent_ids = [
            x.strip()
            for x in parent_value.split(",")
            if x.strip()
        ]

        for parent_id in parent_ids:

            if parent_id not in tweet_lookup:
                continue

            customer_row = tweet_lookup[parent_id]

            # Must be an inbound/customer message

            if customer_row["inbound"] != True:
                continue

            # Must not be AskPlayStation

            if customer_row["author_id"] == BRAND:
                continue

            pairs.append({

                "customer_tweet_id":
                    customer_row["tweet_id"],

                "support_tweet_id":
                    support_row["tweet_id"],

                "customer_message":
                    customer_row["text"],

                "support_response":
                    support_row["text"],

                "customer_created_at":
                    customer_row["created_at"],

                "support_created_at":
                    support_row["created_at"],

                "pairing_method":
                    "support_parent"

            })

    # --------------------------------------------------------
    # Method 2:
    #
    # Customer tweet points forward to support tweet
    #
    # response_tweet_id
    #
    # This catches relationships that Method 1 misses.
    # --------------------------------------------------------

    customer_df = df[
        df["inbound"] == True
    ].copy()

    for _, customer_row in customer_df.iterrows():

        response_value = (
            customer_row["response_tweet_id"]
        )

        if not response_value:
            continue

        if response_value in ["nan", "None"]:
            continue

        response_ids = [
            x.strip()
            for x in response_value.split(",")
            if x.strip()
        ]

        for response_id in response_ids:

            if response_id not in tweet_lookup:
                continue

            support_row = tweet_lookup[response_id]

            # Make sure this is actually PlayStation

            if support_row["author_id"] != BRAND:
                continue

            pairs.append({

                "customer_tweet_id":
                    customer_row["tweet_id"],

                "support_tweet_id":
                    support_row["tweet_id"],

                "customer_message":
                    customer_row["text"],

                "support_response":
                    support_row["text"],

                "customer_created_at":
                    customer_row["created_at"],

                "support_created_at":
                    support_row["created_at"],

                "pairing_method":
                    "customer_response"

            })

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    columns = [

        "customer_tweet_id",
        "support_tweet_id",
        "customer_message",
        "support_response",
        "customer_created_at",
        "support_created_at",
        "pairing_method"

    ]

    pairs_df = pd.DataFrame(
        pairs,
        columns=columns
    )

    # --------------------------------------------------------
    # Remove duplicate relationships
    # --------------------------------------------------------

    pairs_df = pairs_df.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "support_tweet_id"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        "playstation_customer_support_pairs.csv"
    )

    pairs_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nCustomer → Support pairs found: "
        f"{len(pairs_df):,}"
    )

    print(
        f"Saved to: {output_file}"
    )

    if not pairs_df.empty:

        print("\nPairing methods:")

        print(
            pairs_df["pairing_method"]
            .value_counts()
        )

    return pairs_df


# ============================================================
# STEP 6 — SHOW EXAMPLES
# ============================================================

def show_examples(pairs_df):

    print("\n" + "=" * 70)
    print("REAL PLAYSTATION SUPPORT EXAMPLES")
    print("=" * 70)

    if pairs_df.empty:

        print(
            "\nWARNING: No pairs were found."
        )

        print(
            "\nWe will NOT create a golden set yet."
        )

        return

    examples = pairs_df.head(15)

    for i, (_, row) in enumerate(
        examples.iterrows(),
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"EXAMPLE {i}"
        )

        print("\nCUSTOMER:")

        print(
            row["customer_message"]
        )

        print("\nASKPLAYSTATION:")

        print(
            row["support_response"]
        )


# ============================================================
# STEP 7 — CREATE GOLDEN CANDIDATES
# ============================================================

def create_golden_candidates(pairs_df):

    print("\n" + "=" * 70)
    print("CREATING GOLDEN SET CANDIDATES")
    print("=" * 70)

    if pairs_df.empty:

        print(
            "\nSkipped because there are no pairs."
        )

        return

    # Remove duplicate customer messages

    pairs_df = pairs_df.drop_duplicates(
        subset=["customer_message"]
    )

    sample_size = min(
        200,
        len(pairs_df)
    )

    golden = pairs_df.sample(
        n=sample_size,
        random_state=42
    ).copy()

    golden.insert(
        0,
        "example_id",
        range(1, len(golden) + 1)
    )

    # Human labels

    golden["intent"] = ""

    golden["expected_action"] = ""

    golden["reason_for_escalation"] = ""

    golden["difficulty"] = ""

    output_file = (
        OUTPUT_DIR /
        "playstation_golden_set.csv"
    )

    golden.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nGolden candidates created: "
        f"{len(golden):,}"
    )

    print(
        f"Saved to: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("🎮" * 20)

    print(
        "HIVER — PLAYSTATION SUPPORT AGENT"
    )

    print("🎮" * 20)

    # 1. Load

    df = load_data()

    if df is None:
        return

    # 2. Clean

    df = clean_data(df)

    # 3. Analyze

    analyze_data(df)

    # 4. Lookup

    tweet_lookup = build_tweet_lookup(df)

    # 5. Pair conversations

    pairs_df = find_customer_support_pairs(
        df,
        tweet_lookup
    )

    # 6. Examples

    show_examples(
        pairs_df
    )

    # 7. Golden candidates

    create_golden_candidates(
        pairs_df
    )

    print("\n" + "=" * 70)

    print("PHASE 2 COMPLETE")

    print("=" * 70)


if __name__ == "__main__":

    main()