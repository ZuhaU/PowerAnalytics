import pandas as pd
import re
from pathlib import Path


# -----------------------------
# File paths
# -----------------------------

INPUT_FILE = "Data/Raw/Generation of Electricity by Sector.csv"
OUTPUT_FILE = "Data/Silver/electricity_generation_silver.csv"


# -----------------------------
# Load raw data
# -----------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Raw dataset shape: {df.shape}")


# -----------------------------
# Identify monthly columns
# -----------------------------

month_columns = [
    col for col in df.columns
    if re.match(r"^[A-Z]{3}-\d{4}$", str(col))
]

print(f"Number of monthly columns: {len(month_columns)}")


# -----------------------------
# Convert wide → long
# -----------------------------

df_long = df.melt(
    id_vars=["ITEMS", "UNIT"],
    value_vars=month_columns,
    var_name="date",
    value_name="generation_gwh"
)


# -----------------------------
# Clean source names
# -----------------------------

def clean_source_name(name):
    name = str(name).strip()

    # Remove leading dots
    name = re.sub(r"^\.+\s*", "", name)

    # Remove numbering such as "1 ", "10 ", "12 "
    name = re.sub(r"^\d+\s+", "", name)

    return name.strip()


df_long["source"] = df_long["ITEMS"].apply(clean_source_name)


# -----------------------------
# Clean dates
# -----------------------------

df_long["date"] = pd.to_datetime(
    df_long["date"],
    format="%b-%Y"
)


# -----------------------------
# Clean numeric values
# -----------------------------

df_long["generation_gwh"] = pd.to_numeric(
    df_long["generation_gwh"],
    errors="coerce"
)


# -----------------------------
# Remove rows with no value
# -----------------------------

df_long = df_long.dropna(
    subset=["generation_gwh"]
)


# -----------------------------
# Keep useful columns
# -----------------------------

df_silver = df_long[
    ["date", "source", "generation_gwh", "UNIT"]
].copy()

df_silver = df_silver.rename(
    columns={"UNIT": "unit"}
)

df_silver["unit"] = df_silver["unit"].str.strip()

# -----------------------------
# Save Silver dataset
# -----------------------------

Path("Data/Silver").mkdir(
    parents=True,
    exist_ok=True
)

df_silver.to_csv(
    OUTPUT_FILE,
    index=False
)


# -----------------------------
# Validation
# -----------------------------

print("\n--- SILVER DATASET ---")
print(df_silver.head(20))

print("\n--- SHAPE ---")
print(df_silver.shape)

print("\n--- SOURCES ---")
print(df_silver["source"].unique())

print("\n--- DATE RANGE ---")
print(df_silver["date"].min())
print(df_silver["date"].max())

print(f"\nSaved to: {OUTPUT_FILE}")