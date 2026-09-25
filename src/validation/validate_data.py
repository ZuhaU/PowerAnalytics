import pandas as pd
from pathlib import Path

INPUT_FILE = "Data/Silver/electricity_generation_silver.csv"


def validate_data(df):
    errors = []

    # 1. Check required columns
    required_columns = [
        "date",
        "source",
        "generation_gwh",
        "unit"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing columns: {missing_columns}"
        )
        return errors  # can't run the rest of the checks without these columns

    # 2. Check for missing values
    if df[required_columns].isnull().any().any():
        errors.append("Missing values found in required columns.")

    # 3. Check for duplicate rows
    if df.duplicated().any():
        errors.append("Duplicate rows found.")

    # 4. Check for negative generation
    negative_rows = df[df["generation_gwh"] < 0]

    if not negative_rows.empty:
        errors.append(
        f"Negative generation values found: {len(negative_rows)} row(s)."
    )
    valid_units = {" GWh", "GWh"}

    invalid_units = set(df["unit"].dropna().unique()) - valid_units

    if invalid_units:
        errors.append(
            f"Unexpected units found: {invalid_units}"
        )

    return errors


def main():
    if not Path(INPUT_FILE).exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} rows.")

    errors = validate_data(df)

    print("\n--- DATA QUALITY REPORT ---")

    if not errors:
        print("PASS: No data quality issues detected.")
    else:
        print("FAIL: Data quality issues detected:")

        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()