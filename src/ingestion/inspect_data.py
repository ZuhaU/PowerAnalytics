import pandas as pd

file_path = "Data/Raw/Generation of Electricity by Sector.csv"

df = pd.read_csv(file_path)

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isna().sum())

print("\n--- ELECTRICITY SOURCES ---")
print(df["ITEMS"].tolist())