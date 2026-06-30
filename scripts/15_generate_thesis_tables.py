from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

GROUND_TRUTH_CSV = PROJECT_ROOT / "datasets" / "final" / "ground_truth_final.csv"
TTC_CSV = PROJECT_ROOT / "experimental_results" / "ttc_results_gpt55.csv"
COMPILE_CSV = PROJECT_ROOT / "experimental_results" / "compile_results_gpt55.csv"

OUTPUT_DIR = PROJECT_ROOT / "experimental_results" / "thesis_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gt_df = pd.read_csv(GROUND_TRUTH_CSV)
ttc_df = pd.read_csv(TTC_CSV)
compile_df = pd.read_csv(COMPILE_CSV)

# Table 4.1 Dataset Summary
dataset_summary = pd.DataFrame([
    {"Metric": "Total selected functions", "Value": len(gt_df)},
    {"Metric": "BVA applicable functions", "Value": (gt_df["bva_applicable"].astype(str).str.lower() == "yes").sum()},
    {"Metric": "ECP applicable functions", "Value": (gt_df["ecp_applicable"].astype(str).str.lower() == "yes").sum()},
    {"Metric": "Generated GPT-5.5 test files evaluated", "Value": ttc_df["test_available"].sum() if "test_available" in ttc_df.columns else len(ttc_df)},
])

dataset_summary.to_csv(OUTPUT_DIR / "table_4_1_dataset_summary.csv", index=False)


# Table 4.2 TTC Summary
ttc_summary = pd.DataFrame([
    {"Metric": "Average BVA Score", "Value": round(ttc_df["bva_score"].mean(), 4)},
    {"Metric": "Average ECP Score", "Value": round(ttc_df["ecp_score"].mean(), 4)},
    {"Metric": "Average TTC Score", "Value": round(ttc_df["ttc_score"].mean(), 4)},
    {"Metric": "Minimum TTC Score", "Value": round(ttc_df["ttc_score"].min(), 4)},
    {"Metric": "Maximum TTC Score", "Value": round(ttc_df["ttc_score"].max(), 4)},
])

ttc_summary.to_csv(OUTPUT_DIR / "table_4_2_ttc_summary.csv", index=False)


# Table 4.3 Compile Summary
total_compile = len(compile_df)
compiled = int(compile_df["compile_success"].sum())
failed = total_compile - compiled

compile_summary = pd.DataFrame([
    {"Metric": "Total generated tests", "Value": total_compile},
    {"Metric": "Compiled successfully", "Value": compiled},
    {"Metric": "Compile failures", "Value": failed},
    {"Metric": "Compile success rate (%)", "Value": round((compiled / total_compile) * 100, 2) if total_compile else 0},
])

compile_summary.to_csv(OUTPUT_DIR / "table_4_3_compile_summary.csv", index=False)


# Table 4.4 Top and Bottom TTC Functions
top_bottom = pd.concat([
    ttc_df.sort_values("ttc_score", ascending=False).head(5),
    ttc_df.sort_values("ttc_score", ascending=True).head(5),
])

cols = ["function_id", "function_name", "bva_score", "ecp_score", "ttc_score"]
top_bottom[cols].to_csv(OUTPUT_DIR / "table_4_4_top_bottom_ttc.csv", index=False)


print("Thesis tables generated:")
for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file)