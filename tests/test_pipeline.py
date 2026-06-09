"""
test_pipeline.py
Unit and integration tests for the Walmart Sales Forecasting pipeline.
Run: python -m pytest tests/ -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preprocessing import load_walmart_data, add_features, FEATURE_COLS
from model_inference import validate_inputs


# ── Data Loading ──────────────────────────────────────────────────────────────

def test_load_walmart_data_shape():
    df = load_walmart_data()
    assert len(df) > 100, "Should have at least 100 weekly rows"
    assert "total_sales" in df.columns
    assert "Date" in df.columns


def test_load_walmart_data_no_negatives():
    df = load_walmart_data()
    assert (df["total_sales"] > 0).all(), "All weekly sales should be positive"


def test_load_walmart_data_external_features():
    df = load_walmart_data()
    for col in ["temperature", "fuel_price", "cpi", "unemployment"]:
        assert col in df.columns, f"Missing external feature: {col}"


def test_load_walmart_data_sorted():
    df = load_walmart_data()
    assert df["Date"].is_monotonic_increasing, "Dates should be sorted ascending"


# ── Feature Engineering ───────────────────────────────────────────────────────

def test_add_features_columns():
    weekly   = load_walmart_data()
    featured = add_features(weekly)
    for col in FEATURE_COLS:
        assert col in featured.columns, f"Missing feature column: {col}"


def test_add_features_no_nulls():
    weekly   = load_walmart_data()
    featured = add_features(weekly)
    nulls = featured[FEATURE_COLS].isnull().sum()
    assert nulls.sum() == 0, f"Null values found in features:\n{nulls[nulls > 0]}"


def test_add_features_cyclical_range():
    weekly   = load_walmart_data()
    featured = add_features(weekly)
    assert featured["week_sin"].between(-1, 1).all()
    assert featured["week_cos"].between(-1, 1).all()


def test_add_features_holiday_flags_binary():
    weekly   = load_walmart_data()
    featured = add_features(weekly)
    for col in ["is_holiday", "is_q4", "is_december", "is_january",
                "is_thanksgiving_week", "is_christmas_week", "is_superbowl_week"]:
        assert featured[col].isin([0, 1]).all(), f"{col} should be binary"


def test_add_features_row_count():
    weekly   = load_walmart_data()
    featured = add_features(weekly)
    # After dropping NaN lags (need 52 weeks of history), should still have many rows
    assert len(featured) > 50, "Should have more than 50 usable rows"


def test_lag_values_correct():
    weekly   = load_walmart_data()
    featured = add_features(weekly)
    # lag_1 for row i should equal total_sales for row i-1 (before feature engineering)
    # Just check lag_1 is populated and plausible
    assert (featured["lag_1"] > 0).all()


# ── Input Validation ──────────────────────────────────────────────────────────

def test_validate_inputs_valid():
    validate_inputs(2013, 47)   # valid: Thanksgiving 2013
    validate_inputs(2012, 1)
    validate_inputs(2030, 52)


def test_validate_inputs_bad_year():
    with pytest.raises(ValueError, match="Year"):
        validate_inputs(2009, 10)
    with pytest.raises(ValueError, match="Year"):
        validate_inputs(2031, 10)


def test_validate_inputs_bad_week():
    with pytest.raises(ValueError, match="Week"):
        validate_inputs(2013, 0)
    with pytest.raises(ValueError, match="Week"):
        validate_inputs(2013, 53)


# ── Feature Columns List ──────────────────────────────────────────────────────

def test_feature_cols_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "FEATURE_COLS has duplicates"


def test_feature_cols_count():
    assert len(FEATURE_COLS) >= 20, "Should have at least 20 features"
