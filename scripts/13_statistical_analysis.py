from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

TTC_CSV = PROJECT_ROOT / "experimental_results" / "ttc_results_gpt55.csv"
COMPILE_CSV = PROJECT_ROOT / "experimental_results" / "compile_results_gpt55.csv"
OUTPUT_CSV = PROJECT_ROOT / "experimental_results" / "statistical_summary_gpt55.csv"

ttc_df = pd.read_csv(TTC_CSV)
compile_df = pd.read_csv(COMPILE_CSV)

summary_rows = []

for col in ["bva_score", "ecp_score", "ttc_score"]:
    summary_rows.append({
        "metric": col,
        "count": ttc_df[col].count(),
        "mean": round(ttc_df[col].mean(), 4),
        "median": round(ttc_df[col].median(), 4),
        "std": round(ttc_df[col].std(), 4),
        "min": round(ttc_df[col].min(), 4),
        "max": round(ttc_df[col].max(), 4),
    })

compile_success_rate = compile_df["compile_success"].mean() * 100

summary_rows.append({
    "metric": "compile_success_rate",
    "count": len(compile_df),
    "mean": round(compile_success_rate, 2),
    "median": "",
    "std": "",
    "min": "",
    "max": "",
})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_CSV, index=False)

print(summary_df)
print(f"\nSaved to: {OUTPUT_CSV}")