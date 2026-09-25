"""
Quick diagnostic script to confirm that SBP_API_KEY is set correctly
and that the State Bank of Pakistan Easydata API is reachable.

This is not part of the pipeline (src/pipeline.py) -- run it manually
whenever you need to sanity-check API access before a live ingestion run:

    python src/ingestion/test_sbp_connection.py
"""

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("SBP_API_KEY")

if not API_KEY:
    raise RuntimeError("SBP_API_KEY was not found")
url = "https://easydata.sbp.org.pk/api/v1/series/TS_GP_RLS_ELECGEN_M.E_001000/data"

params = {
    "api_key": API_KEY,
    "start_date": "2012-07-01",
    "format": "json",
}

response = requests.get(url, params=params, timeout=60)

print("Status:", response.status_code)
print("Response:", response.text[:1000])