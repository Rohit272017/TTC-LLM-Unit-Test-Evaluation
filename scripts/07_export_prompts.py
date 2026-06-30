import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

INPUT_CSV = PROJECT_ROOT / "prompts" / "evaluation_prompts.csv"
OUTPUT_DIR = PROJECT_ROOT / "generated_tests" / "manual_prompts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

for _, row in df.iterrows():
    function_id = row["function_id"]
    function_name = row["function_name"]

    prompt_file = OUTPUT_DIR / f"{function_id}_{function_name}_prompt.txt"

    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(row["prompt"])

print(f"Generated {len(df)} prompt text files")
print(f"Saved to: {OUTPUT_DIR}")