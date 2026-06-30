from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

TTC_CSV = PROJECT_ROOT / "experimental_results" / "ttc_results_gpt55.csv"
COMPILE_CSV = PROJECT_ROOT / "experimental_results" / "compile_results_gpt55.csv"
FIGURE_DIR = PROJECT_ROOT / "experimental_results" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

ttc_df = pd.read_csv(TTC_CSV)
compile_df = pd.read_csv(COMPILE_CSV)

# Figure 1: TTC Score Distribution
plt.figure()
plt.hist(ttc_df["ttc_score"].dropna(), bins=10)
plt.xlabel("TTC Score")
plt.ylabel("Number of Functions")
plt.title("GPT-5.5 TTC Score Distribution")
plt.savefig(FIGURE_DIR / "figure_1_ttc_score_distribution.png", bbox_inches="tight")
plt.close()

# Figure 2: Average BVA/ECP/TTC Scores
scores = {
    "BVA": ttc_df["bva_score"].mean(),
    "ECP": ttc_df["ecp_score"].mean(),
    "TTC": ttc_df["ttc_score"].mean(),
}

plt.figure()
plt.bar(scores.keys(), scores.values())
plt.ylabel("Average Score")
plt.ylim(0, 1)
plt.title("Average BVA, ECP, and TTC Scores")
plt.savefig(FIGURE_DIR / "figure_2_average_scores.png", bbox_inches="tight")
plt.close()

# Figure 3: Compile Success Rate
compile_counts = compile_df["compile_success"].value_counts()

labels = ["Compiled" if v else "Failed" for v in compile_counts.index]
values = compile_counts.values

plt.figure()
plt.pie(values, labels=labels, autopct="%1.1f%%")
plt.title("GPT-5.5 Generated Test Compile Success Rate")
plt.savefig(FIGURE_DIR / "figure_3_compile_success_rate.png", bbox_inches="tight")
plt.close()

# Figure 4: TTC Score Per Function
plt.figure(figsize=(12, 5))
plt.plot(range(len(ttc_df)), ttc_df["ttc_score"])
plt.xlabel("Function Index")
plt.ylabel("TTC Score")
plt.ylim(0, 1)
plt.title("TTC Score Per Function")
plt.savefig(FIGURE_DIR / "figure_4_ttc_per_function.png", bbox_inches="tight")
plt.close()

print("Figures generated successfully:")
for fig in sorted(FIGURE_DIR.glob("*.png")):
    print(fig)