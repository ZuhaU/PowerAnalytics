import os
import ast
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


# configuration

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("SBP_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "SBP_API_KEY was not found in the .env file."
    )

BASE_URL = (
    "https://easydata.sbp.org.pk/api/v1/series"
)

DATASET_CODE = "TS_GP_RLS_ELECGEN_M"

START_DATE = "2012-07-01"

OUTPUT_DIR = PROJECT_ROOT / "Data" / "Raw"

OUTPUT_FILE = (
    OUTPUT_DIR /
    "Generation of Electricity by Sector.csv"
)


# exact sbp series

SERIES = {
    "Total Electricity Generation by all sources":
        "TS_GP_RLS_ELECGEN_M.E_001000",

    "Electricity Generation by Hydel":
        "TS_GP_RLS_ELECGEN_M.E_002000",

    "Electricity Generation by Coal":
        "TS_GP_RLS_ELECGEN_M.E_003000",

    "Electricity Generation by High Speed Diesel (HSD)":
        "TS_GP_RLS_ELECGEN_M.E_004000",

    "Electricity Generation by Residual Fuel Oil (RFO)":
        "TS_GP_RLS_ELECGEN_M.E_005000",

    "Electricity Generation by Gas":
        "TS_GP_RLS_ELECGEN_M.E_006000",

    "Electricity Generation by Regasification Liquefied Natural Gas(RLNG)":
        "TS_GP_RLS_ELECGEN_M.E_007000",

    "Electricity Generation by Nuclear":
        "TS_GP_RLS_ELECGEN_M.E_008000",

    "Electricity Imported from Iran":
        "TS_GP_RLS_ELECGEN_M.E_009000",

    "Electricity Generation by Mixed":
        "TS_GP_RLS_ELECGEN_M.E_010000",

    "Electricity Generation by Wind":
        "TS_GP_RLS_ELECGEN_M.E_011000",

    "Electricity Generation by bagasse":
        "TS_GP_RLS_ELECGEN_M.E_012000",

    "Electricity Generation by Solar":
        "TS_GP_RLS_ELECGEN_M.E_013000",
}


# parse sbp response

def parse_sbp_response(text):

    text = text.strip()

    # Normal JSON response
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # SBP may sometimes return Python-style
    # list/dictionary representation
    try:
        return ast.literal_eval(text)

    except (ValueError, SyntaxError) as exc:
        raise RuntimeError(
            "Could not parse SBP API response."
        ) from exc


# download one series

def download_series(page, series_name, series_key):

    print()
    print("-" * 60)
    print(f"Downloading: {series_name}")
    print(f"Series key: {series_key}")

    url = (
        f"{BASE_URL}/"
        f"{series_key}/data"
        f"?api_key={API_KEY}"
        f"&start_date={START_DATE}"
        f"&format=json"
    )

    response = page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=60000,
    )

    if response is None:
        raise RuntimeError(
            f"No response received for {series_key}"
        )

    print(
        f"HTTP status: {response.status}"
    )

    if response.status != 200:
        body = page.locator("body").inner_text()

        raise RuntimeError(
            f"SBP API failed for {series_key}\n"
            f"Status: {response.status}\n"
            f"Response: {body[:1000]}"
        )

    body = page.locator("body").inner_text()

    data = parse_sbp_response(body)

    if "rows" not in data:
        raise RuntimeError(
            f"No rows returned for {series_key}"
        )

    df = pd.DataFrame(
        data["rows"],
        columns=data["columns"],
    )

    print(
        f"Rows received: {len(df)}"
    )

    return df


# convert api data to original wide format

def convert_to_wide(series_data):

    print()
    print("Converting API data to raw wide format...")

    wide_rows = []

    for series_name, df in series_data.items():

        if df.empty:
            print(
                f"WARNING: No data for {series_name}"
            )
            continue

        # Convert observation date
        df["Observation Date"] = pd.to_datetime(
            df["Observation Date"],
            errors="coerce",
        )

        # Convert values to numeric
        df["Observation Value"] = pd.to_numeric(
            df["Observation Value"],
            errors="coerce",
        )

        # Create row
        row = {
           "ITEMS": series_name,
           "UNIT": "GWh",
        }

        for _, record in df.iterrows():

            date = record["Observation Date"]
            value = record["Observation Value"]

            if pd.isna(date):
                continue

            if pd.isna(value):
                continue

            # Original dataset uses labels such as:
            # JUL-2012, AUG-2012, etc.
            month_label = (
                date.strftime("%b-%Y")
                .upper()
            )

            row[month_label] = value

        wide_rows.append(row)

    result = pd.DataFrame(
        wide_rows
    )

    # Put Source and Unit first
    metadata_columns = [
        "ITEMS",
        "UNIT",
    ]

    month_columns = [
        col
        for col in result.columns
        if col not in metadata_columns
    ]

    # Sort chronologically
    month_columns = sorted(
        month_columns,
        key=lambda x: pd.to_datetime(
            x,
            format="%b-%Y"
        )
    )

    result = result[
        metadata_columns + month_columns
    ]

    return result


# main ingestion

def download_sbp_dataset():

    print()
    print("=" * 60)
    print("SBP ELECTRICITY DATA INGESTION")
    print("=" * 60)

    print(
        f"Dataset: {DATASET_CODE}"
    )

    print(
        f"Start date: {START_DATE}"
    )

    print(
        f"Series count: {len(SERIES)}"
    )

    print(
        "API key loaded: YES"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    series_data = {}

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False
        )

        page = browser.new_page()

        try:

            for series_name, series_key in SERIES.items():

                df = download_series(
                    page,
                    series_name,
                    series_key,
                )

                series_data[
                    series_name
                ] = df

        finally:

            browser.close()

    # Convert to same structure
    # expected by existing transformation
    final_data = convert_to_wide(
        series_data
    )

    # Save
    final_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 60)
    print("SBP INGESTION SUCCESSFUL")
    print("=" * 60)

    print(
        f"Rows: {len(final_data)}"
    )

    print(
        f"Columns: {len(final_data.columns)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    month_columns = [
        col
        for col in final_data.columns
        if col not in ["ITEMS", "UNIT"]
    ]

    if month_columns:

        print(
            f"First month: {month_columns[0]}"
        )

        print(
            f"Latest month: {month_columns[-1]}"
        )


# entry point

if __name__ == "__main__":
    download_sbp_dataset()