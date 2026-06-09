"""
model_inference.py
Loads the saved Walmart model artifact, validates inputs,
generates a 13-week forecast, and prints a business-ready terminal report.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from data_preprocessing import load_walmart_data, add_features, FEATURE_COLS

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model(artifact_path=None):
    """Load the saved model artifact (.pkl file)."""
    if artifact_path is None:
        artifact_path = MODELS_DIR / "best_model_latest.pkl"
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def validate_inputs(year: int, week: int):
    """Validate year/week inputs. Raises ValueError on bad input."""
    if not (2010 <= year <= 2030):
        raise ValueError(f"Year must be between 2010 and 2030 (got {year}).")
    if not (1 <= week <= 52):
        raise ValueError(f"Week must be between 1 and 52 (got {week}).")


def _build_feature_row(history: pd.DataFrame, target_date: pd.Timestamp,
                        econ_defaults: dict) -> pd.Series:
    """
    Build a single feature row for target_date using the history DataFrame
    (which must include all prior weekly sales for lag computation).
    econ_defaults: dict of mean economic feature values to use for future weeks.
    """
    sales = history["total_sales"].values
    n     = len(sales)

    def lag(k):
        return sales[-k] if n >= k else np.nan

    def rolling_mean(window):
        arr = sales[-window:] if n >= window else sales
        return arr.mean() if len(arr) > 0 else np.nan

    def rolling_std(window):
        arr = sales[-window:] if n >= window else sales
        return arr.std() if len(arr) > 1 else 0.0

    woy    = int(target_date.isocalendar()[1])
    month  = target_date.month
    qtr    = (month - 1) // 3 + 1
    year   = target_date.year

    row = {
        "week_of_year":         woy,
        "month":                month,
        "quarter":              qtr,
        "year":                 year,
        "trend":                n,
        "is_holiday":           int(woy in [6, 36, 47, 52]),   # Super Bowl, Labor Day, Thanksgiving, Christmas
        "is_thanksgiving_week": int(woy == 47),
        "is_christmas_week":    int(woy == 52),
        "is_superbowl_week":    int(woy == 6),
        "is_q4":                int(qtr == 4),
        "is_december":          int(month == 12),
        "is_january":           int(month == 1),
        "lag_1":                lag(1),
        "lag_2":                lag(2),
        "lag_4":                lag(4),
        "lag_8":                lag(8),
        "lag_52":               lag(52),
        "rolling_mean_4":       rolling_mean(4),
        "rolling_mean_12":      rolling_mean(12),
        "rolling_std_4":        rolling_std(4),
        "yoy_growth":           ((lag(1) - lag(52)) / lag(52)) if (lag(52) and lag(52) != 0) else 0.0,
        "week_sin":             np.sin(2 * np.pi * woy / 52),
        "week_cos":             np.cos(2 * np.pi * woy / 52),
        # Economic features — use last known values for future predictions
        "temperature":          econ_defaults.get("temperature", 60.0),
        "fuel_price":           econ_defaults.get("fuel_price", 3.5),
        "cpi":                  econ_defaults.get("cpi", 220.0),
        "unemployment":         econ_defaults.get("unemployment", 8.0),
        "total_markdown":       econ_defaults.get("total_markdown", 0.0),
    }
    return pd.Series(row)


def forecast_weeks(n_weeks: int = 13, artifact=None) -> list:
    """
    Iteratively forecast the next n_weeks weeks.
    Each week's prediction is appended to history so subsequent lag features are correct.
    Returns list of (date_str, predicted_sales) tuples.
    """
    if artifact is None:
        artifact = load_model()

    model    = artifact["model"]
    weekly   = load_walmart_data()
    featured = add_features(weekly)

    # Build history from the full training dataset (to compute lags)
    history = weekly[["Date", "total_sales",
                       "temperature", "fuel_price", "cpi",
                       "unemployment", "total_markdown"]].copy()

    # Economic defaults = mean of last 13 weeks of available data
    econ_defaults = {
        col: history[col].iloc[-13:].mean()
        for col in ["temperature", "fuel_price", "cpi", "unemployment", "total_markdown"]
    }

    last_date  = history["Date"].max()
    forecast   = []

    for i in range(n_weeks):
        next_date = last_date + timedelta(weeks=i + 1)
        row = _build_feature_row(history, next_date, econ_defaults)
        X   = pd.DataFrame([row])[FEATURE_COLS]
        pred = float(model.predict(X)[0])
        pred = max(pred, 0)     # sales can't be negative

        forecast.append((next_date.strftime("%Y-%m-%d"), round(pred, 2)))

        # Append prediction to history for next iteration's lags
        new_row = pd.DataFrame([{
            "Date": next_date, "total_sales": pred,
            **econ_defaults
        }])
        history = pd.concat([history, new_row], ignore_index=True)

    return forecast


def predict_single(year: int, week: int, artifact=None) -> dict:
    """
    Predict total sales for a specific year and week number.
    Returns a dict with prediction and confidence interval.
    """
    validate_inputs(year, week)

    if artifact is None:
        artifact = load_model()

    model  = artifact["model"]
    mape_v = artifact["metrics"].get("MAPE", 10.0)

    weekly = load_walmart_data()
    history = weekly[["Date", "total_sales",
                       "temperature", "fuel_price", "cpi",
                       "unemployment", "total_markdown"]].copy()

    # Use Jan 4 of the given year + (week-1)*7 days as an approximate date
    target_date = pd.Timestamp(year=year, month=1, day=4) + timedelta(weeks=week - 1)

    econ_defaults = {
        col: history[col].iloc[-13:].mean()
        for col in ["temperature", "fuel_price", "cpi", "unemployment", "total_markdown"]
    }

    row   = _build_feature_row(history, target_date, econ_defaults)
    X     = pd.DataFrame([row])[FEATURE_COLS]
    pred  = float(model.predict(X)[0])
    pred  = max(pred, 0)

    margin = pred * (mape_v / 100)
    return {
        "year":           year,
        "week":           week,
        "date":           target_date.strftime("%Y-%m-%d"),
        "predicted_sales": round(pred, 2),
        "lower_bound":    round(pred - margin, 2),
        "upper_bound":    round(pred + margin, 2),
        "confidence_pct": round(100 - mape_v, 1),
    }


def print_forecast():
    """Print a full forecast report to the terminal."""
    artifact = load_model()
    metrics  = artifact["metrics"]
    fc       = forecast_weeks(13, artifact)

    print("\n" + "="*60)
    print("  WALMART SALES FORECAST REPORT")
    print("="*60)
    print(f"  Model   : {artifact['name']}  (v{artifact['version']})")
    print(f"  MAE     : ${metrics['MAE']:,.0f}")
    print(f"  RMSE    : ${metrics.get('RMSE', 0):,.0f}")
    print(f"  R2      : {metrics['R2']:.3f}")
    print(f"  MAPE    : {metrics['MAPE']:.1f}%")
    print(f"\n  13-Week Forecast:")
    print(f"  {'Week':>4}  {'Date':<14}  {'Predicted Sales':>18}")
    print(f"  {'-'*42}")

    total = 0
    for i, (date, sales) in enumerate(fc, 1):
        print(f"  {i:>4}  {date:<14}  ${sales:>17,.0f}")
        total += sales

    print(f"  {'-'*42}")
    print(f"  {'TOTAL':>4}  {'(13 weeks)':<14}  ${total:>17,.0f}")
    print("="*60)


if __name__ == "__main__":
    print_forecast()
