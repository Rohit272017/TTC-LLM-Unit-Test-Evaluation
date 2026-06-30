from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "datasets" / "final" / "final_selected_functions.csv"
OUTPUT_FILE = PROJECT_ROOT / "datasets" / "final" / "ground_truth_template.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    gt = pd.DataFrame({
        "id": df["id"],
        "function_name": df["function_name"],
        "function_file": df["function_file"],

        # Fill manually
        "bva_applicable": "",
        "min_boundary": "",
        "max_boundary": "",
        "below_min": "",
        "above_max": "",
        "bva_expected_values": "",

        "ecp_applicable": "",
        "valid_classes": "",
        "invalid_classes": "",
        "ecp_expected_values": "",

        "notes": ""
    })

    gt.to_csv(OUTPUT_FILE, index=False)

    print("Ground truth template created:")
    print(OUTPUT_FILE)
    print("Rows:", len(gt))


if __name__ == "__main__":
    main()