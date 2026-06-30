import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

INPUT_CSV = PROJECT_ROOT / "datasets/final/ground_truth_final.csv"
OUTPUT_CSV = PROJECT_ROOT / "prompts/evaluation_prompts.csv"

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

prompts = []

for _, row in df.iterrows():

    function_path = PROJECT_ROOT / row["function_file"]

    if not function_path.exists():
        print(f"Missing function file: {function_path}")
        continue

    with open(function_path, "r", encoding="utf-8", errors="ignore") as f:
        function_code = f.read()

    prompt = f"""
You are a software testing expert.

Generate Google Test unit tests for the following C++ function.

Focus on:
- Boundary Value Analysis (BVA)
- Equivalence Class Partitioning (ECP)
- valid inputs
- invalid inputs
- edge cases

C++ Function:

{function_code}

Return only Google Test code.
"""

    prompts.append({
        "function_id": row["id"],
        "function_name": row["function_name"],
        "function_file": row["function_file"],
        "prompt": prompt
    })

pd.DataFrame(prompts).to_csv(OUTPUT_CSV, index=False)

print(f"Generated {len(prompts)} prompts")
print(f"Saved to {OUTPUT_CSV}")