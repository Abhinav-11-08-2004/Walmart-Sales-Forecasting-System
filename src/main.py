"""
main.py — Walmart Sales Forecasting Pipeline
Runs the full pipeline: preprocess -> train -> forecast.
Run: python src/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import load_walmart_data, add_features
from model_training import train
from model_inference import print_forecast, load_model


def main():
    print("\n" + "="*60)
    print("  WALMART SALES FORECASTING — FULL PIPELINE")
    print("="*60)

    # Step 1: Load and preprocess data
    print("\n[1/3] Loading and preprocessing Walmart data...")
    weekly   = load_walmart_data()
    featured = add_features(weekly)

    # Save processed files
    out_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_dir / "weekly_sales.csv", index=False)
    featured.to_csv(out_dir / "featured_data.csv", index=False)
    print(f"    Weekly rows    : {len(weekly)}")
    print(f"    Usable rows    : {len(featured)}")
    print(f"    Saved to       : data/processed/")

    # Step 2: Train and evaluate models
    print("\n[2/3] Training models...")
    train()

    # Step 3: Print forecast
    print("\n[3/3] Generating 13-week forecast...")
    print_forecast()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
