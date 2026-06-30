from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "datasets" / "extracted"
FUNCTIONS_DIR = EXTRACTED_DIR / "functions"
HUMAN_TESTS_DIR = EXTRACTED_DIR / "human_tests"
METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"

FUNCTIONS_DIR.mkdir(parents=True, exist_ok=True)
HUMAN_TESTS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_METADATA = METADATA_DIR / "metadata.csv"


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    return text[:80]


def detect_code_column(columns):
    for col in columns:
        name = col.lower()
        if name in ["code", "function_code", "source_code"]:
            return col
        if "code" in name and "test" not in name:
            return col
    return None


def detect_test_column(columns):
    for col in columns:
        name = col.lower()
        if "unit test" in name or "test" in name:
            return col
    return None


def detect_repo_column(columns):
    for col in columns:
        name = col.lower()
        if "repository" in name or "repo" in name:
            return col
    return None


def detect_file_column(columns):
    for col in columns:
        name = col.lower()
        if "file" in name or "path" in name:
            return col
    return None


def extract_function_name(code):
    if not isinstance(code, str):
        return "unknown_function"

    pattern = r"([a-zA-Z_][a-zA-Z0-9_:<>~]*)\s*\([^;{}]*\)\s*\{"
    matches = re.findall(pattern, code)

    if matches:
        candidate = matches[0].split("::")[-1]
        return safe_name(candidate)

    return "unknown_function"


def looks_like_cpp_test_code(text):
    if not isinstance(text, str):
        return False

    stripped = text.strip()

    # Reject simple file paths like:
    # googletest/test/gtest_unittest.cc
    # tensorstore/internal/metrics/metadata_test.cc
    if "\n" not in stripped and "/" in stripped:
        return False

    # Reject single filename paths
    if "\n" not in stripped and stripped.endswith((".cc", ".cpp", ".cxx", ".h", ".hpp")):
        return False

    test_indicators = [
        "TEST(",
        "TEST_F(",
        "TEST_P(",
        "TYPED_TEST(",
        "EXPECT_",
        "ASSERT_",
        "#include",
    ]

    return any(indicator in stripped for indicator in test_indicators)


def main():
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {RAW_DIR}")

    all_rows = []
    global_id = 1

    for parquet_file in parquet_files:
        print(f"Reading: {parquet_file.name}")
        df = pd.read_parquet(parquet_file)

        print("Columns:", df.columns.tolist())

        code_col = detect_code_column(df.columns)
        test_col = detect_test_column(df.columns)
        repo_col = detect_repo_column(df.columns)
        file_col = detect_file_column(df.columns)

        if code_col is None:
            raise ValueError(f"Could not detect code column in {parquet_file.name}")

        for _, row in df.iterrows():
            code = row.get(code_col, "")
            test_code = row.get(test_col, "") if test_col else ""

            if not isinstance(code, str) or len(code.strip()) == 0:
                continue

            function_name = extract_function_name(code)
            item_id = f"function_{global_id:04d}"

            function_file = FUNCTIONS_DIR / f"{item_id}_{function_name}.cpp"
            test_file = HUMAN_TESTS_DIR / f"{item_id}_{function_name}_test.cpp"

            function_file.write_text(code, encoding="utf-8")

            if looks_like_cpp_test_code(test_code):
                test_file.write_text(test_code, encoding="utf-8")
                human_test_available = True
                human_test_file_value = str(test_file.relative_to(PROJECT_ROOT))
            else:
                human_test_available = False
                human_test_file_value = ""

            all_rows.append({
                "id": item_id,
                "source_split": parquet_file.stem,
                "function_name": function_name,
                "repository": row.get(repo_col, "") if repo_col else "",
                "original_file_name": row.get(file_col, "") if file_col else "",
                "function_file": str(function_file.relative_to(PROJECT_ROOT)),
                "human_test_file": human_test_file_value,
                "human_test_available": human_test_available,
                "code_length": len(code),
                "has_if": "if" in code,
                "has_switch": "switch" in code,
                "has_loop": any(k in code for k in ["for", "while"]),
                "has_comparison": any(k in code for k in [">=", "<=", "==", "!=", ">", "<"]),
                "raw_code_column": code_col,
                "raw_test_column": test_col if test_col else "",
            })

            global_id += 1

    metadata_df = pd.DataFrame(all_rows)
    metadata_df.to_csv(OUTPUT_METADATA, index=False)

    print("\nDONE")
    print("Total extracted functions:", len(metadata_df))
    print("Human tests extracted:", metadata_df["human_test_available"].sum())
    print("Metadata saved at:", OUTPUT_METADATA)


if __name__ == "__main__":
    main()