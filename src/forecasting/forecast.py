import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "Gold"
    / "electricity_monthly_gold.csv"
)

FORECAST_OUTPUT = (
    PROJECT_ROOT
    / "Data"
    / "Gold"
    / "electricity_generation_forecast.csv"
)

EVALUATION_OUTPUT = (
    PROJECT_ROOT
    / "Data"
    / "Gold"
    / "forecast_evaluation.csv"
)


def load_data():

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df[["date", "total_generation_gwh"]]
        .dropna()
        .sort_values("date")
    )

    df = df.set_index("date")

    # Ensure monthly frequency
    series = df["total_generation_gwh"].asfreq("MS")

    return series


def train_sarima(series):

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return model.fit(disp=False)


def evaluate_model(series, test_periods=12):

    print("\n--- MODEL EVALUATION ---")

    train = series.iloc[:-test_periods]
    test = series.iloc[-test_periods:]

    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")

    model = train_sarima(train)

    predictions = model.get_forecast(
        steps=test_periods
    ).predicted_mean

    predictions.index = test.index

    mae = mean_absolute_error(
        test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            predictions
        )
    )
    mape = (
    np.mean(
        np.abs(
            (test.values - predictions.values)
            / test.values
        )
    ) * 100
)

    evaluation = pd.DataFrame({
        "date": test.index,
        "actual_gwh": test.values,
        "predicted_gwh": predictions.values,
        "absolute_error_gwh": np.abs(
            test.values - predictions.values
        )
    })

    evaluation["mae_gwh"] = mae
    evaluation["rmse_gwh"] = rmse

    EVALUATION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation.to_csv(
        EVALUATION_OUTPUT,
        index=False
    )

    print(f"\nMAE: {mae:,.2f} GWh")
    print(f"RMSE: {rmse:,.2f} GWh")
    print(f"MAPE: {mape:.2f}%")

    print(
        f"\nEvaluation saved to: "
        f"{EVALUATION_OUTPUT}"
    )

    return mae, rmse


def create_future_forecast(series, periods=12):

    print("\n--- FUTURE FORECAST ---")

    print(
        f"Training on all {len(series)} "
        f"historical observations..."
    )

    model = train_sarima(series)

    forecast = model.get_forecast(
        steps=periods
    )

    forecast_values = forecast.predicted_mean
    confidence = forecast.conf_int()

    result = pd.DataFrame({
        "date": forecast_values.index,
        "forecast_generation_gwh": forecast_values.values,
        "lower_bound_gwh": confidence.iloc[:, 0].values,
        "upper_bound_gwh": confidence.iloc[:, 1].values
    })

    result["forecast_generation_gwh"] = (
        result["forecast_generation_gwh"].clip(lower=0)
    )

    result["lower_bound_gwh"] = (
        result["lower_bound_gwh"].clip(lower=0)
    )

    result["upper_bound_gwh"] = (
        result["upper_bound_gwh"].clip(lower=0)
    )

    result.to_csv(
        FORECAST_OUTPUT,
        index=False
    )

    print("\nNext 12 months:")

    print(result.to_string(index=False))

    print(
        f"\nForecast saved to: "
        f"{FORECAST_OUTPUT}"
    )

    return result


def main():

    print("=" * 60)
    print("POWER ANALYTICS - FORECASTING")
    print("=" * 60)

    series = load_data()

    print(
        f"\nHistorical period: "
        f"{series.index.min().date()} → "
        f"{series.index.max().date()}"
    )

    print(
        f"Historical observations: {len(series)}"
    )

    # Evaluate using the final 12 historical months
    evaluate_model(
        series,
        test_periods=12
    )

    # Train on all historical data and forecast future
    create_future_forecast(
        series,
        periods=12
    )

    print("\n" + "=" * 60)
    print("FORECASTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()