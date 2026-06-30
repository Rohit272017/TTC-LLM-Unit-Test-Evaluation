from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

GROUND_TRUTH_CSV = PROJECT_ROOT / "datasets" / "final" / "ground_truth_final.csv"
GENERATED_TEST_DIR = PROJECT_ROOT / "generated_tests" / "gpt55"
OUTPUT_CSV = PROJECT_ROOT / "experimental_results" / "gpt55_generated_tests.csv"

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

gt_df = pd.read_csv(GROUND_TRUTH_CSV)

rows = []

for _, row in gt_df.iterrows():
    function_id = row["id"]
    function_name = row["function_name"]

    possible_files = list(GENERATED_TEST_DIR.glob(f"{function_id}*"))

    if not possible_files:
        rows.append({
            "function_id": function_id,
            "function_name": function_name,
            "test_file": "",
            "generated_test_code": "",
            "test_available": False
        })
        continue

    test_file = possible_files[0]
    test_code = test_file.read_text(encoding="utf-8", errors="ignore")

    rows.append({
        "function_id": function_id,
        "function_name": function_name,
        "test_file": str(test_file.relative_to(PROJECT_ROOT)),
        "generated_test_code": test_code,
        "test_available": True
    })

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

print(f"Collected {len(rows)} rows")
print(f"Available tests: {sum(r['test_available'] for r in rows)}")
print(f"Saved to: {OUTPUT_CSV}")