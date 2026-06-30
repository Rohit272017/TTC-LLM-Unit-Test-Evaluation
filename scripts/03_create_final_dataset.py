from pathlib import Path
import pandas as pd
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "datasets" / "metadata" / "selected_candidate_functions.csv"

FINAL_DIR = PROJECT_ROOT / "datasets" / "final"
FINAL_FUNCTIONS_DIR = FINAL_DIR / "functions"
FINAL_TESTS_DIR = FINAL_DIR / "human_tests"

OUTPUT_FILE = FINAL_DIR / "final_selected_functions.csv"

FINAL_FUNCTIONS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_FILE)

    # Take top 60 for thesis experiment
    final_df = df.head(60).copy()

    for _, row in final_df.iterrows():
        src_func = PROJECT_ROOT / row["function_file"]

        if src_func.exists():
            dst_func = FINAL_FUNCTIONS_DIR / src_func.name
            shutil.copy(src_func, dst_func)

        if isinstance(row.get("human_test_file", ""), str) and row["human_test_file"]:
            src_test = PROJECT_ROOT / row["human_test_file"]
            if src_test.exists():
                dst_test = FINAL_TESTS_DIR / src_test.name
                shutil.copy(src_test, dst_test)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("Final selected functions:", len(final_df))
    print("Saved at:", OUTPUT_FILE)
    print("Functions copied to:", FINAL_FUNCTIONS_DIR)
    print("Human tests copied to:", FINAL_TESTS_DIR)


if __name__ == "__main__":
    main()