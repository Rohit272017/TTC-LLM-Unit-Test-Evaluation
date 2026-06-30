from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "datasets" / "final" / "ground_truth_template.csv"
OUTPUT_FILE = PROJECT_ROOT / "datasets" / "final" / "ground_truth_preannotated.csv"


def read_code(function_file):
    path = PROJECT_ROOT / function_file
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_numeric_constants(code):
    numbers = re.findall(r"(?<![A-Za-z_])[-]?\d+(?:\.\d+)?", code)
    return sorted(set(numbers), key=lambda x: float(x))


def has_boundary_logic(code):
    return any(op in code for op in [">=", "<=", ">", "<"])


def has_partition_logic(code):
    return any(keyword in code for keyword in ["if", "else", "switch", "case", "return"])


def main():
    df = pd.read_csv(INPUT_FILE)

    for idx, row in df.iterrows():
        code = read_code(row["function_file"])
        numbers = extract_numeric_constants(code)

        if has_boundary_logic(code):
            df.at[idx, "bva_applicable"] = "review_yes"
        else:
            df.at[idx, "bva_applicable"] = "review_no"

        if len(numbers) >= 2:
            df.at[idx, "min_boundary"] = numbers[0]
            df.at[idx, "max_boundary"] = numbers[-1]
            df.at[idx, "bva_expected_values"] = ",".join(numbers)

            try:
                min_val = float(numbers[0])
                max_val = float(numbers[-1])

                if min_val.is_integer():
                    df.at[idx, "below_min"] = str(int(min_val - 1))
                else:
                    df.at[idx, "below_min"] = str(min_val - 0.1)

                if max_val.is_integer():
                    df.at[idx, "above_max"] = str(int(max_val + 1))
                else:
                    df.at[idx, "above_max"] = str(max_val + 0.1)

            except ValueError:
                pass

        if has_partition_logic(code):
            df.at[idx, "ecp_applicable"] = "review_yes"
            df.at[idx, "valid_classes"] = "review_required"
            df.at[idx, "invalid_classes"] = "review_required"
            df.at[idx, "ecp_expected_values"] = "review_required"
        else:
            df.at[idx, "ecp_applicable"] = "review_no"

        df.at[idx, "notes"] = "Auto-preannotated. Please manually review."

    df.to_csv(OUTPUT_FILE, index=False)

    print("Preannotated ground truth created:")
    print(OUTPUT_FILE)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()