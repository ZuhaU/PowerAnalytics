"""
Power Analytics - Intelligence Engine

Generates business-facing insights from:
    - electricity_monthly_gold.csv
    - forecast_evaluation.csv
    - electricity_generation_forecast.csv

Explanations are generated deterministically (see deterministic_explanation
below) so the numbers in insights.csv always match the calculations exactly.

If ENABLE_AI_SUMMARY=true and Ollama is running locally, generate_ai_summary()
takes the finished insight rows and asks phi3:mini to turn them into a short
paragraph. It only ever sees the rows already written below, not the raw data.

Output:
    Data/Intelligence/insights.csv
    Data/Intelligence/executive_summary.txt   (only if ENABLE_AI_SUMMARY=true)
"""

import json
import os
import subprocess
from pathlib import Path
import re
import pandas as pd
import numpy as np


# paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = PROJECT_ROOT / "Data" / "Gold"
INTELLIGENCE_DIR = PROJECT_ROOT / "Data" / "Intelligence"

MONTHLY_GOLD_FILE = GOLD_DIR / "electricity_monthly_gold.csv"
FORECAST_EVALUATION_FILE = GOLD_DIR / "forecast_evaluation.csv"
FORECAST_FILE = GOLD_DIR / "electricity_generation_forecast.csv"

OUTPUT_FILE = INTELLIGENCE_DIR / "insights.csv"
EXECUTIVE_SUMMARY_FILE = INTELLIGENCE_DIR / "executive_summary.txt"

# off by default -- don't want the pipeline to break on a machine
# without ollama installed (docker, github actions, etc)
ENABLE_AI_SUMMARY = os.getenv("ENABLE_AI_SUMMARY", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT_SECONDS = 120


# helpers

def clean_text(value):
    """Clean text safely."""
    if value is None:
        return ""

    value = str(value)

    # Remove ANSI terminal escape sequences
    value = re.sub(
        r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])",
        "",
        value
    )

    return value.strip()


def safe_float(value):
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def detect_generation_column(df):
    """Detect the electricity generation column."""

    possible_columns = [
        "generation_gwh",
        "total_generation_gwh",
        "electricity_generation_gwh",
        "generation",
        "total_generation",
        "value",
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    # Fallback: identify a single obvious numeric column
    numeric_columns = []

    for column in df.columns:
        if column == "date":
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.notna().sum() > 0:
            numeric_columns.append(column)

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    return None


# load monthly gold data

def load_monthly_gold():
    """Load the monthly gold dataset."""

    if not MONTHLY_GOLD_FILE.exists():
        print(
            f"WARNING: Monthly gold file not found: "
            f"{MONTHLY_GOLD_FILE}"
        )
        return pd.DataFrame()

    df = pd.read_csv(MONTHLY_GOLD_FILE)

    print(f"Monthly gold rows loaded: {len(df)}")
    print(f"Monthly gold columns: {list(df.columns)}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    return df


# load forecast evaluation

def load_forecast_evaluation():
    """Load historical forecast evaluation data."""

    if not FORECAST_EVALUATION_FILE.exists():
        print(
            f"WARNING: Forecast evaluation file not found: "
            f"{FORECAST_EVALUATION_FILE}"
        )
        return pd.DataFrame()

    df = pd.read_csv(FORECAST_EVALUATION_FILE)

    print(f"Forecast evaluation rows loaded: {len(df)}")
    print(
        f"Forecast evaluation columns: "
        f"{list(df.columns)}"
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    return df


# load future forecast

def load_future_forecast():
    """Load the future forecast."""

    if not FORECAST_FILE.exists():
        print(
            f"WARNING: Future forecast file not found: "
            f"{FORECAST_FILE}"
        )
        return pd.DataFrame()

    df = pd.read_csv(FORECAST_FILE)

    print(f"Future forecast rows loaded: {len(df)}")
    print(
        f"Future forecast columns: "
        f"{list(df.columns)}"
    )

    if "date" not in df.columns:
        print(
            "WARNING: Future forecast has no date column."
        )
        return pd.DataFrame()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Detect forecast column
    # --------------------------------------------------------

    possible_columns = [
        "predicted_gwh",
        "forecast_gwh",
        "forecast",
        "predicted",
        "prediction",
        "yhat",
        "y_pred",
        "prediction_gwh",
        "forecasted_gwh",
    ]

    forecast_column = None

    for column in possible_columns:
        if column in df.columns:
            forecast_column = column
            break

    # --------------------------------------------------------
    # Numeric fallback
    # --------------------------------------------------------

    if forecast_column is None:

        numeric_candidates = []

        for column in df.columns:

            if column == "date":
                continue

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if converted.notna().sum() > 0:
                numeric_candidates.append(column)

        if len(numeric_candidates) == 1:
            forecast_column = numeric_candidates[0]

    if forecast_column is None:
        print(
            "WARNING: No forecast value column "
            "could be detected."
        )
        return pd.DataFrame()

    print(
        f"Using forecast column: {forecast_column}"
    )

    df["predicted_gwh"] = pd.to_numeric(
        df[forecast_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "predicted_gwh"
        ]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


# deterministic business explanations

def deterministic_explanation(
    title,
    category,
    metric,
    value,
    change_pct,
    severity,
    absolute_change=None,
):
    """
    Generate clean, factual business explanations.

    No LLM is used here. All numerical values come directly
    from the calculated dataset metrics.
    """

    value = safe_float(value)
    change_pct = safe_float(change_pct)
    absolute_change = safe_float(absolute_change)

    # --------------------------------------------------------
    # Forecast Accuracy
    # --------------------------------------------------------

    if category == "Forecast Accuracy":

        return (
            f"The forecast evaluation recorded a mean "
            f"absolute percentage error of {abs(value):.2f}%, "
            f"representing the average difference between "
            f"predicted and actual generation."
        )

    # --------------------------------------------------------
    # Forecast Bias
    # --------------------------------------------------------

    if category == "Forecast Bias":

        if value >= 0:
            return (
                f"The forecast underestimated actual "
                f"electricity generation by approximately "
                f"{abs(value):.2f}% over the evaluation period."
            )

        return (
            f"The forecast overestimated actual electricity "
            f"generation by approximately "
            f"{abs(value):.2f}% over the evaluation period."
        )

    # --------------------------------------------------------
    # Year-over-Year Growth
    # --------------------------------------------------------

    if category == "Growth":

        if change_pct >= 0:
            return (
                f"Electricity generation increased by "
                f"{abs(change_pct):.2f}% compared with the "
                f"same period in the previous year."
            )

        return (
            f"Electricity generation decreased by "
            f"{abs(change_pct):.2f}% compared with the "
            f"same period in the previous year."
        )

    # --------------------------------------------------------
    # Long-Term Trend
    # --------------------------------------------------------

    if category == "Long-Term Trend":

        if change_pct >= 0:

            if pd.notna(absolute_change):
                return (
                    f"Long-term electricity generation "
                    f"increased by {abs(change_pct):.2f}%, "
                    f"representing a total change of "
                    f"{abs(absolute_change):,.2f} MWh across "
                    f"the available historical period."
                )

            return (
                f"Long-term electricity generation increased "
                f"by {abs(change_pct):.2f}% across the available "
                f"historical period."
            )

        if pd.notna(absolute_change):
            return (
                f"Long-term electricity generation decreased "
                f"by {abs(change_pct):.2f}%, representing a "
                f"total change of {abs(absolute_change):,.2f} "
                f"MWh across the available historical period."
            )

        return (
            f"Long-term electricity generation decreased "
            f"by {abs(change_pct):.2f}% across the available "
            f"historical period."
        )

    # --------------------------------------------------------
    # Future Outlook
    # --------------------------------------------------------

    if category == "Future Outlook":

        if change_pct >= 0:
            return (
                f"The forecast indicates average future "
                f"generation approximately {abs(change_pct):.2f}% "
                f"higher than the recent historical average."
            )

        return (
            f"The forecast indicates average future "
            f"generation approximately {abs(change_pct):.2f}% "
            f"lower than the recent historical average."
        )

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    return (
        f"{title}. The observed value is {value:.2f}, "
        f"with a reported change of "
        f"{change_pct:.2f}%."
    )


# insight creation

def create_insight(
    title,
    category,
    metric,
    value,
    change_pct,
    severity,
    insight_type,
    date=None,
    z_score=None,
    absolute_change=None,
):
    """Create one insight record."""

    value = safe_float(value)
    change_pct = safe_float(change_pct)
    z_score = safe_float(z_score)
    absolute_change = safe_float(absolute_change)

    explanation = deterministic_explanation(
        title=title,
        category=category,
        metric=metric,
        value=value,
        change_pct=change_pct,
        severity=severity,
        absolute_change=absolute_change,
    )

    return {
        "date": date,
        "type": insight_type,
        "category": category,
        "title": clean_text(title),
        "metric": clean_text(metric),
        "value": value,
        "change_pct": change_pct,
        "absolute_change_mwh": absolute_change,
        "z_score": z_score,
        "severity": clean_text(severity),
        "ai_explanation": clean_text(explanation),
    }


# ollama executive summary (optional)

def generate_ai_summary(insights_df):
    """Ask a local Ollama model to turn the insight rows into a short summary.
    Only sees the rounded values already in insights_df, never the raw data."""

    if insights_df.empty:
        return "No significant insights detected."

    facts = []
    for _, row in insights_df.iterrows():
        facts.append({
            "category": row.get("category"),
            "title": row.get("title"),
            "metric": row.get("metric"),
            "value": None if pd.isna(row.get("value")) else round(float(row["value"]), 2),
            "change_pct": None if pd.isna(row.get("change_pct")) else round(float(row["change_pct"]), 2),
            "severity": row.get("severity"),
        })

    prompt = f"""You are an electricity analytics assistant.

Summarize only the findings below. Rules:
- use only the supplied findings, don't invent facts, dates, or new numbers
- keep every percentage exactly as given
- forecast accuracy and future forecasts are two different things, don't mix them up
- pick the 2-3 most important findings and write 2-3 sentences, professional tone

Findings:
{json.dumps(facts, indent=2)}
"""

    print()
    print(f"Running Ollama AI summarization ({OLLAMA_MODEL})...")

    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=OLLAMA_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            print("WARNING: Ollama failed.")
            if result.stderr:
                print(result.stderr)
            return "AI summary unavailable."

        output = result.stdout.strip()
        if not output:
            print("WARNING: Ollama returned an empty response.")
            return "AI summary unavailable."

        # strip ANSI codes / stray control chars that sometimes leak through
        output = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", output)
        output = "".join(c for c in output if c.isprintable() or c in "\n\r\t")
        return output.strip()

    except FileNotFoundError:
        print("WARNING: Ollama executable was not found.")
        print(f"Make sure Ollama is installed and `ollama pull {OLLAMA_MODEL}` has been run.")
        return "AI summary unavailable."

    except subprocess.TimeoutExpired:
        print("WARNING: Ollama timed out.")
        return "AI summary unavailable."

    except Exception as exc:
        print(f"WARNING: Could not run Ollama: {exc}")
        return "AI summary unavailable."


# main intelligence logic

def generate_insights():

    print()
    print("=" * 60)
    print("POWER ANALYTICS - INTELLIGENCE ENGINE")
    print("=" * 60)

    INTELLIGENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    monthly = load_monthly_gold()
    evaluation = load_forecast_evaluation()
    future = load_future_forecast()

    insights = []

    # 1. forecast accuracy

    if not evaluation.empty:

        mape_column = None

        for column in [
            "mape",
            "mape_pct",
            "MAPE",
            "mape_percentage",
        ]:
            if column in evaluation.columns:
                mape_column = column
                break

        if mape_column:

            mape = pd.to_numeric(
                evaluation[mape_column],
                errors="coerce"
            ).dropna()

            if not mape.empty:

                mean_mape = float(
                    mape.mean()
                )

                severity = (
                    "Low"
                    if mean_mape < 5
                    else "Medium"
                    if mean_mape < 10
                    else "High"
                )

                insights.append(
                    create_insight(
                        title="Forecast accuracy assessment",
                        category="Forecast Accuracy",
                        metric="MAPE",
                        value=mean_mape,
                        change_pct=mean_mape,
                        severity=severity,
                        insight_type="Forecast Accuracy",
                    )
                )

        # forecast bias

        actual_column = None
        predicted_column = None

        for column in [
            "actual_gwh",
            "actual",
            "actual_generation_gwh",
        ]:
            if column in evaluation.columns:
                actual_column = column
                break

        for column in [
            "predicted_gwh",
            "forecast_gwh",
            "predicted",
            "forecast",
        ]:
            if column in evaluation.columns:
                predicted_column = column
                break

        if actual_column and predicted_column:

            actual = pd.to_numeric(
                evaluation[actual_column],
                errors="coerce"
            )

            predicted = pd.to_numeric(
                evaluation[predicted_column],
                errors="coerce"
            )

            valid = pd.DataFrame(
                {
                    "actual": actual,
                    "predicted": predicted,
                }
            ).dropna()

            if not valid.empty:

                actual_total = valid["actual"].sum()
                predicted_total = valid["predicted"].sum()

                if actual_total != 0:

                    bias_pct = (
                        (
                            actual_total
                            - predicted_total
                        )
                        / actual_total
                    ) * 100

                    if abs(bias_pct) < 1:
                        severity = "Low"
                    elif abs(bias_pct) < 5:
                        severity = "Medium"
                    else:
                        severity = "High"

                    if bias_pct >= 0:
                        title = (
                            "Forecast underestimated actual generation"
                        )
                    else:
                        title = (
                            "Forecast overestimated actual generation"
                        )

                    insights.append(
                        create_insight(
                            title=title,
                            category="Forecast Bias",
                            metric="Forecast Bias",
                            value=bias_pct,
                            change_pct=bias_pct,
                            severity=severity,
                            insight_type="Forecast Bias",
                        )
                    )

    # 2. generation analysis

    generation_column = None

    if not monthly.empty and "date" in monthly.columns:

        generation_column = detect_generation_column(
            monthly
        )

        if generation_column:

            monthly[generation_column] = pd.to_numeric(
                monthly[generation_column],
                errors="coerce"
            )

            monthly = monthly.dropna(
                subset=[
                    "date",
                    generation_column
                ]
            ).sort_values(
                "date"
            ).reset_index(drop=True)

            # latest yoy growth

            if len(monthly) >= 13:

                latest_date = monthly["date"].max()

                latest = monthly[
                    monthly["date"] == latest_date
                ]

                previous_year_date = (
                    latest_date
                    - pd.DateOffset(years=1)
                )

                previous = monthly[
                    monthly["date"] == previous_year_date
                ]

                if (
                    not latest.empty
                    and not previous.empty
                ):

                    latest_value = float(
                        latest[generation_column].iloc[0]
                    )

                    previous_value = float(
                        previous[generation_column].iloc[0]
                    )

                    if previous_value != 0:

                        yoy_change = (
                            (
                                latest_value
                                - previous_value
                            )
                            / abs(previous_value)
                        ) * 100

                        severity = (
                            "Positive"
                            if yoy_change >= 0
                            else "Attention"
                        )

                        title = (
                            "Latest annual generation increased"
                            if yoy_change >= 0
                            else
                            "Latest annual generation decreased"
                        )

                        insights.append(
                            create_insight(
                                title=title,
                                category="Growth",
                                metric="YoY Generation Growth",
                                value=latest_value,
                                change_pct=yoy_change,
                                severity=severity,
                                insight_type="YoY Growth",
                                date=latest_date,
                            )
                        )

            # long-term trend

            if len(monthly) >= 24:

                first_value = float(
                    monthly[
                        generation_column
                    ].iloc[0]
                )

                last_value = float(
                    monthly[
                        generation_column
                    ].iloc[-1]
                )

                if first_value != 0:

                    long_term_change = (
                        (
                            last_value
                            - first_value
                        )
                        / abs(first_value)
                    ) * 100

                    absolute_change = (
                        last_value
                        - first_value
                    )

                    severity = (
                        "Positive"
                        if long_term_change >= 0
                        else "Attention"
                    )

                    title = (
                        "Long-term generation trend is upward"
                        if long_term_change >= 0
                        else
                        "Long-term generation trend is downward"
                    )

                    insights.append(
                        create_insight(
                            title=title,
                            category="Long-Term Trend",
                            metric="Long-Term Generation Change",
                            value=last_value,
                            change_pct=long_term_change,
                            absolute_change=absolute_change,
                            severity=severity,
                            insight_type="Long-Term Trend",
                            date=monthly["date"].iloc[-1],
                        )
                    )

    # 3. future forecast outlook

    if (
        not future.empty
        and not monthly.empty
        and generation_column
    ):

        recent_values = pd.to_numeric(
            monthly[generation_column],
            errors="coerce"
        ).dropna()

        future_values = pd.to_numeric(
            future["predicted_gwh"],
            errors="coerce"
        ).dropna()

        if (
            not recent_values.empty
            and not future_values.empty
        ):

            # Compare future average against
            # latest 12 historical observations.

            recent_window = recent_values.tail(12)

            historical_average = float(
                recent_window.mean()
            )

            future_average = float(
                future_values.mean()
            )

            if historical_average != 0:

                outlook_change = (
                    (
                        future_average
                        - historical_average
                    )
                    / abs(historical_average)
                ) * 100

                severity = (
                    "Positive"
                    if outlook_change >= 0
                    else "Attention"
                )

                if outlook_change >= 0:
                    title = (
                        "Future forecast points to higher generation"
                    )
                else:
                    title = (
                        "Future forecast points to lower generation"
                    )

                insights.append(
                    create_insight(
                        title=title,
                        category="Future Outlook",
                        metric="Forecast vs Recent Average",
                        value=future_average,
                        change_pct=outlook_change,
                        severity=severity,
                        insight_type="Future Outlook",
                        date=future["date"].min(),
                    )
                )

    # remove duplicates

    if insights:

        df = pd.DataFrame(
            insights
        )

        df = df.drop_duplicates(
            subset=[
                "category",
                "title",
            ],
            keep="first",
        )

        columns = [
            "date",
            "type",
            "category",
            "title",
            "metric",
            "value",
            "change_pct",
            "absolute_change_mwh",
            "z_score",
            "severity",
            "ai_explanation",
        ]

        df = df[columns]

    else:

        print(
            "WARNING: No insights could be generated."
        )

        df = pd.DataFrame(
            columns=[
                "date",
                "type",
                "category",
                "title",
                "metric",
                "value",
                "change_pct",
                "absolute_change_mwh",
                "z_score",
                "severity",
                "ai_explanation",
            ]
        )

    # save

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # optional: ollama executive summary

    if ENABLE_AI_SUMMARY:

        summary = generate_ai_summary(df)

        EXECUTIVE_SUMMARY_FILE.write_text(
            summary,
            encoding="utf-8",
        )

        print()
        print(f"Executive summary saved to: {EXECUTIVE_SUMMARY_FILE}")

    else:

        summary = None

    print()
    print("=" * 60)
    print("INTELLIGENCE ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Insights generated: {len(df)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    if not df.empty:

        print()
        print("Generated insights:")

        for _, row in df.iterrows():

            print(
                f"- {row['title']}"
            )

            print(
                f"  {row['ai_explanation']}"
            )

    if ENABLE_AI_SUMMARY and summary:

        print()
        print(f"EXECUTIVE SUMMARY (Ollama / {OLLAMA_MODEL}):")
        print(summary)

    print("=" * 60)


# entry point

if __name__ == "__main__":
    generate_insights()