from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

TTC_CSV = PROJECT_ROOT / "experimental_results" / "ttc_results_gpt55.csv"
COMPILE_CSV = PROJECT_ROOT / "experimental_results" / "compile_results_gpt55.csv"
OUTPUT_CSV = PROJECT_ROOT / "experimental_results" / "final_results_table_gpt55.csv"

ttc_df = pd.read_csv(TTC_CSV)
compile_df = pd.read_csv(COMPILE_CSV)

merged = pd.merge(
    ttc_df,
    compile_df[["function_id", "compile_success"]],
    on="function_id",
    how="left"
)

final_df = merged[[
    "function_id",
    "function_name",
    "bva_score",
    "ecp_score",
    "ttc_score",
    "compile_success"
]]

final_df.to_csv(OUTPUT_CSV, index=False)

print(final_df.head(10))
print(f"\nSaved to: {OUTPUT_CSV}")