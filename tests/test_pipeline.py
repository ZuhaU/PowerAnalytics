"""
Basic tests for the pipeline outputs + the two files that actually
have testable functions (validate_data.py and generate_insights.py).

Run the pipeline first, then:
    pip install pytest
    pytest
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "validation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "intelligence"))

import validate_data
import generate_insights


@pytest.fixture(scope="module")
def silver():
    return pd.read_csv(DATA_DIR / "Silver" / "electricity_generation_silver_clean.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def monthly_gold():
    return pd.read_csv(DATA_DIR / "Gold" / "electricity_monthly_gold.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def source_gold():
    return pd.read_csv(DATA_DIR / "Gold" / "electricity_source_gold.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def forecast():
    return pd.read_csv(DATA_DIR / "Gold" / "electricity_generation_forecast.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def evaluation():
    return pd.read_csv(DATA_DIR / "Gold" / "forecast_evaluation.csv")


@pytest.fixture(scope="module")
def insights():
    return pd.read_csv(DATA_DIR / "Intelligence" / "insights.csv")


# ---- silver layer ----

def test_silver_has_required_columns(silver):
    assert {"date", "source", "generation_gwh", "unit"}.issubset(silver.columns)


def test_silver_has_no_nulls(silver):
    assert not silver[["date", "source", "generation_gwh", "unit"]].isnull().any().any()


def test_silver_no_duplicates(silver):
    assert not silver.duplicated().any()


def test_silver_generation_not_negative(silver):
    assert (silver["generation_gwh"] >= 0).all()


def test_silver_units_are_consistent(silver):
    assert set(silver["unit"].str.strip().unique()) == {"GWh"}


# ---- gold: monthly ----

def test_monthly_gold_columns(monthly_gold):
    required = {"date", "total_generation_gwh", "renewable_generation_gwh", "renewable_share_pct"}
    assert required.issubset(monthly_gold.columns)


def test_monthly_gold_dates_sorted_and_unique(monthly_gold):
    assert monthly_gold["date"].is_monotonic_increasing
    assert monthly_gold["date"].is_unique


def test_renewable_share_between_0_and_100(monthly_gold):
    share = monthly_gold["renewable_share_pct"].dropna()
    assert (share >= 0).all() and (share <= 100).all()


# ---- gold: per source ----

def test_source_categories_are_known(source_gold):
    allowed = {"Renewable", "Nuclear", "Fossil / Other", "Other"}
    assert set(source_gold["category"].unique()).issubset(allowed)


# ---- forecast ----

def test_forecast_is_12_months(forecast):
    assert len(forecast) == 12


def test_forecast_bounds_make_sense(forecast):
    assert (forecast["lower_bound_gwh"] <= forecast["forecast_generation_gwh"]).all()
    assert (forecast["forecast_generation_gwh"] <= forecast["upper_bound_gwh"]).all()


def test_evaluation_errors_not_negative(evaluation):
    assert (evaluation["mae_gwh"] >= 0).all()
    assert (evaluation["rmse_gwh"] >= 0).all()


# ---- insights.csv ----

def test_insights_has_no_blank_explanations(insights):
    assert not insights["ai_explanation"].isnull().any()
    assert not (insights["ai_explanation"] == "").any()


def test_insights_severity_values(insights):
    allowed = {"Low", "Medium", "High", "Positive", "Attention"}
    assert set(insights["severity"].dropna().unique()).issubset(allowed)


# ---- validate_data.py ----

def _sample_df():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"],
        "source": ["Hydel", "Solar"],
        "generation_gwh": [100.0, 50.0],
        "unit": ["GWh", "GWh"],
    })


def test_valid_data_passes_validation():
    assert validate_data.validate_data(_sample_df()) == []


def test_negative_generation_is_flagged():
    df = _sample_df()
    df.loc[0, "generation_gwh"] = -10.0
    errors = validate_data.validate_data(df)
    assert any("Negative generation" in e for e in errors)


def test_duplicate_rows_are_flagged():
    df = pd.concat([_sample_df(), _sample_df().iloc[[0]]])
    errors = validate_data.validate_data(df)
    assert any("Duplicate rows" in e for e in errors)


def test_missing_column_is_flagged():
    df = _sample_df().drop(columns=["unit"])
    errors = validate_data.validate_data(df)
    assert any("Missing columns" in e for e in errors)


# ---- generate_insights.py ----

def test_growth_explanation_mentions_percent_and_direction():
    text = generate_insights.deterministic_explanation(
        title="x", category="Growth", metric="YoY", value=100, change_pct=7.07, severity="Positive",
    )
    assert "increased" in text and "7.07%" in text


def test_forecast_bias_underestimate_wording():
    text = generate_insights.deterministic_explanation(
        title="x", category="Forecast Bias", metric="bias", value=2.87, change_pct=2.87, severity="Medium",
    )
    assert "underestimated" in text


def test_create_insight_has_expected_fields():
    row = generate_insights.create_insight(
        title="Test insight", category="Growth", metric="YoY Generation Growth",
        value=1000.0, change_pct=5.0, severity="Positive", insight_type="YoY Growth",
    )
    assert row["ai_explanation"] != ""
    assert row["category"] == "Growth"
