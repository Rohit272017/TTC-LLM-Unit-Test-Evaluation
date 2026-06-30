from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path("/home/how1kor/sandbox/Thesis")

GROUND_TRUTH_CSV = PROJECT_ROOT / "datasets" / "final" / "ground_truth_final.csv"
GENERATED_DIR = PROJECT_ROOT / "generated_tests" / "gpt55"
OUTPUT_CSV = PROJECT_ROOT / "experimental_results" / "ttc_results_gpt55.csv"

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def split_values(text):
    if pd.isna(text) or str(text).strip().upper() in ["", "N/A", "NA"]:
        return []

    text = str(text)
    parts = re.split(r"[;,]", text)
    return [p.strip() for p in parts if p.strip()]


def normalize(text):
    return str(text).lower().replace(" ", "").replace("_", "").replace("-", "")


def contains_value(test_code, value):
    if not value:
        return False

    value = str(value).strip()
    if value.upper() in ["N/A", "NA"]:
        return False

    return normalize(value) in normalize(test_code)


def score_expected_values(test_code, expected_values):
    values = split_values(expected_values)

    if not values:
        return 0.0, 0, 0

    matched = sum(1 for value in values if contains_value(test_code, value))
    score = matched / len(values)

    return score, matched, len(values)


def find_generated_test(function_id):
    matches = list(GENERATED_DIR.glob(f"{function_id}*"))
    if not matches:
        return None
    return matches[0]


def main():
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)

    rows = []

    for _, row in gt_df.iterrows():
        function_id = row["id"]
        function_name = row["function_name"]

        test_file = find_generated_test(function_id)

        if test_file is None:
            rows.append({
                "function_id": function_id,
                "function_name": function_name,
                "test_file": "",
                "test_available": False,
                "bva_score": 0.0,
                "ecp_score": 0.0,
                "ttc_score": 0.0,
                "notes": "Generated test file missing"
            })
            continue

        test_code = test_file.read_text(encoding="utf-8", errors="ignore")

        if str(row["bva_applicable"]).lower() == "yes":
            bva_score, bva_matched, bva_total = score_expected_values(
                test_code, row["bva_expected_values"]
            )
        else:
            bva_score, bva_matched, bva_total = 1.0, 0, 0

        if str(row["ecp_applicable"]).lower() == "yes":
            ecp_score, ecp_matched, ecp_total = score_expected_values(
                test_code, row["ecp_expected_values"]
            )
        else:
            ecp_score, ecp_matched, ecp_total = 1.0, 0, 0

        applicable_count = 0
        total_score = 0.0

        if str(row["bva_applicable"]).lower() == "yes":
            applicable_count += 1
            total_score += bva_score

        if str(row["ecp_applicable"]).lower() == "yes":
            applicable_count += 1
            total_score += ecp_score

        ttc_score = total_score / applicable_count if applicable_count else 0.0

        rows.append({
            "function_id": function_id,
            "function_name": function_name,
            "test_file": str(test_file.relative_to(PROJECT_ROOT)),
            "test_available": True,
            "bva_applicable": row["bva_applicable"],
            "ecp_applicable": row["ecp_applicable"],
            "bva_score": round(bva_score, 3),
            "bva_matched": bva_matched,
            "bva_total": bva_total,
            "ecp_score": round(ecp_score, 3),
            "ecp_matched": ecp_matched,
            "ecp_total": ecp_total,
            "ttc_score": round(ttc_score, 3),
            "notes": ""
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved TTC results to: {OUTPUT_CSV}")
    print(f"Average TTC: {result_df['ttc_score'].mean():.3f}")
    print(f"Average BVA: {result_df['bva_score'].mean():.3f}")
    print(f"Average ECP: {result_df['ecp_score'].mean():.3f}")


if __name__ == "__main__":
    main()