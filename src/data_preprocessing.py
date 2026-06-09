"""
data_preprocessing.py
Loads and merges the Walmart Sales Forecast dataset (train.csv, features.csv, stores.csv),
aggregates to total weekly sales across all stores, and engineers features for ML models.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# All features used by the ML models
FEATURE_COLS = [
    "week_of_year", "month", "quarter", "year", "trend",
    "is_holiday",
    "is_thanksgiving_week", "is_christmas_week", "is_superbowl_week",
    "is_q4", "is_december", "is_january",
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_4",
    "yoy_growth",
    "week_sin", "week_cos",
    # External economic features (store-level averages)
    "temperature", "fuel_price", "cpi", "unemployment",
    "total_markdown",
]


def load_walmart_data() -> pd.DataFrame:
    """
    Load and merge train.csv + features.csv + stores.csv.
    Aggregate to one row per week (total sales across all 45 stores).
    Returns a weekly time series DataFrame.
    """
    train    = pd.read_csv(DATA_DIR / "train.csv",    parse_dates=["Date"])
    features = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["Date"])
    stores   = pd.read_csv(DATA_DIR / "stores.csv")

    # Fill missing MarkDown values with 0 (no markdown event that week)
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    features[markdown_cols] = features[markdown_cols].fillna(0)
    features["total_markdown"] = features[markdown_cols].sum(axis=1)

    # Merge store metadata onto features
    features = features.merge(stores, on="Store", how="left")

    # Merge features onto train
    df = train.merge(
        features[["Store", "Date", "Temperature", "Fuel_Price",
                   "CPI", "Unemployment", "total_markdown"]],
        on=["Store", "Date"],
        how="left",
    )

    # Aggregate to total weekly sales across all stores
    weekly = (
        df.groupby("Date", as_index=False)
        .agg(
            total_sales=("Weekly_Sales", "sum"),
            is_holiday=("IsHoliday", "max"),       # True if any store had holiday
            temperature=("Temperature", "mean"),
            fuel_price=("Fuel_Price", "mean"),
            cpi=("CPI", "mean"),
            unemployment=("Unemployment", "mean"),
            total_markdown=("total_markdown", "sum"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Remove any weeks with zero or negative sales (data quality)
    weekly = weekly[weekly["total_sales"] > 0].copy()

    return weekly


def add_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-series features from the weekly sales DataFrame.
    Drops rows at the start that don't have enough history for lag features.
    """
    df = weekly.copy()
    df["is_holiday"] = df["is_holiday"].astype(int)

    # ── Calendar features ────────────────────────────────────────────
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["Date"].dt.month
    df["quarter"]      = df["Date"].dt.quarter
    df["year"]         = df["Date"].dt.year
    df["trend"]        = np.arange(len(df))

    # Cyclical encoding for week-of-year (captures seasonality continuity)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # ── Holiday flags ────────────────────────────────────────────────
    # Thanksgiving ~ week 47, Christmas ~ week 52, Super Bowl ~ week 6
    df["is_thanksgiving_week"] = (df["week_of_year"] == 47).astype(int)
    df["is_christmas_week"]    = (df["week_of_year"] == 52).astype(int)
    df["is_superbowl_week"]    = (df["week_of_year"] == 6).astype(int)

    # Seasonal flags
    df["is_q4"]       = (df["quarter"] == 4).astype(int)
    df["is_december"] = (df["month"] == 12).astype(int)
    df["is_january"]  = (df["month"] == 1).astype(int)

    # ── Lag features ─────────────────────────────────────────────────
    for lag in [1, 2, 4, 8, 52]:
        df[f"lag_{lag}"] = df["total_sales"].shift(lag)

    # ── Rolling statistics ───────────────────────────────────────────
    df["rolling_mean_4"]  = df["total_sales"].shift(1).rolling(4).mean()
    df["rolling_mean_12"] = df["total_sales"].shift(1).rolling(12).mean()
    df["rolling_std_4"]   = df["total_sales"].shift(1).rolling(4).std()

    # ── Year-over-year growth ────────────────────────────────────────
    df["yoy_growth"] = (df["total_sales"] - df["total_sales"].shift(52)) / \
                        df["total_sales"].shift(52).replace(0, np.nan)

    # Drop rows that don't have full lag history (need 52 weeks of history)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    return df


if __name__ == "__main__":
    weekly   = load_walmart_data()
    featured = add_features(weekly)

    print(f"Raw weekly rows   : {len(weekly)}")
    print(f"Usable rows (after feature engineering): {len(featured)}")
    print(f"Date range        : {featured['Date'].min().date()} --> {featured['Date'].max().date()}")
    print(f"Avg weekly sales  : ${featured['total_sales'].mean():,.0f}")
    print(f"Features          : {len(FEATURE_COLS)} features")

    # Save processed data
    out_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_dir / "weekly_sales.csv", index=False)
    featured.to_csv(out_dir / "featured_data.csv", index=False)
    print(f"\nSaved --> data/processed/weekly_sales.csv + featured_data.csv")
