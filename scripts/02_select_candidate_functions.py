from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = PROJECT_ROOT / "datasets" / "metadata" / "metadata.csv"
OUTPUT_FILE = PROJECT_ROOT / "datasets" / "metadata" / "selected_candidate_functions.csv"


def score_function(row):
    score = 0

    if row.get("has_if", False):
        score += 3

    if row.get("has_switch", False):
        score += 3

    if row.get("has_loop", False):
        score += 2

    if row.get("has_comparison", False):
        score += 3

    if row.get("human_test_available", False):
        score += 2

    code_length = row.get("code_length", 0)

    if 100 <= code_length <= 3000:
        score += 2

    return score


def main():
    df = pd.read_csv(METADATA_FILE)

    print("Total functions in metadata:", len(df))

    df["selection_score"] = df.apply(score_function, axis=1)

    selected_df = df[df["selection_score"] >= 7].copy()

    selected_df = selected_df.sort_values(
        by=["selection_score", "code_length"],
        ascending=[False, True]
    )

    selected_df.to_csv(OUTPUT_FILE, index=False)

    print("Selected candidate functions:", len(selected_df))
    print("Saved at:", OUTPUT_FILE)

    print("\nTop 20 candidates:")
    print(selected_df[[
        "id",
        "function_name",
        "selection_score",
        "code_length",
        "has_if",
        "has_switch",
        "has_loop",
        "has_comparison",
        "human_test_available"
    ]].head(20))


if __name__ == "__main__":
    main()