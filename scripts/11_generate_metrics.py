from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

TTC_CSV = PROJECT_ROOT / "experimental_results" / "ttc_results_gpt55.csv"
COMPILE_CSV = PROJECT_ROOT / "experimental_results" / "compile_results_gpt55.csv"

ttc_df = pd.read_csv(TTC_CSV)
compile_df = pd.read_csv(COMPILE_CSV)

avg_ttc = ttc_df["ttc_score"].mean()

compile_rate = (
    compile_df["compile_success"].sum()
    / len(compile_df)
) * 100

summary = pd.DataFrame([
    {
        "metric": "Average TTC Score",
        "value": round(avg_ttc, 4)
    },
    {
        "metric": "Compile Success Rate (%)",
        "value": round(compile_rate, 2)
    },
    {
        "metric": "Functions Evaluated",
        "value": len(ttc_df)
    }
])

output_file = PROJECT_ROOT / "experimental_results" / "final_metrics_gpt55.csv"

summary.to_csv(output_file, index=False)

print(summary)
print()
print(f"Saved: {output_file}")