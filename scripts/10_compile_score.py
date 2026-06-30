from pathlib import Path
import subprocess
import pandas as pd

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

GROUND_TRUTH_CSV = PROJECT_ROOT / "datasets" / "final" / "ground_truth_final.csv"
GENERATED_DIR = PROJECT_ROOT / "generated_tests" / "gpt55"
BUILD_DIR = PROJECT_ROOT / "build" / "gpt55_compile"
OUTPUT_CSV = PROJECT_ROOT / "experimental_results" / "compile_results_gpt55.csv"

BUILD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(GROUND_TRUTH_CSV)

rows = []

for _, row in df.iterrows():
    function_id = row["id"]
    function_name = row["function_name"]

    matches = list(GENERATED_DIR.glob(f"{function_id}*"))

    if not matches:
        rows.append({
            "function_id": function_id,
            "function_name": function_name,
            "test_file": "",
            "compile_success": False,
            "error": "Generated test file missing"
        })
        continue

    test_file = matches[0]
    output_obj = BUILD_DIR / f"{function_id}.o"

    cmd = [
        "g++",
        "-std=c++17",
        "-I.",
        "-I/usr/include",
        "-c",
        str(test_file),
        "-o",
        str(output_obj),
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    rows.append({
        "function_id": function_id,
        "function_name": function_name,
        "test_file": str(test_file.relative_to(PROJECT_ROOT)),
        "compile_success": result.returncode == 0,
        "error": result.stderr[:3000]
    })

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_CSV, index=False)

total = len(result_df)
passed = int(result_df["compile_success"].sum())

print(f"Compile results saved to: {OUTPUT_CSV}")
print(f"Total tests: {total}")
print(f"Compiled successfully: {passed}")
print(f"Compile success rate: {(passed / total) * 100:.2f}%")