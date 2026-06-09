# Walmart Sales Forecasting System
**End-to-End Machine Learning Project**
Predicts weekly total sales across all 45 Walmart stores using historical data,
external economic features (CPI, fuel price, unemployment), and holiday indicators.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (preprocess + train + forecast)
python src/main.py

# 3. Launch web app
python src/web_app.py
# Open: http://127.0.0.1:8501
```

Or run everything at once:
```bash
bash scripts/run_pipeline.sh
```

---

## Project Structure

```
├── data/
│   ├── raw/        train.csv, test.csv, features.csv, stores.csv
│   └── processed/  weekly_sales.csv, featured_data.csv
├── models/         best_model_latest.pkl, model_metadata.json
├── notebooks/      eda.ipynb  (exploratory analysis)
├── src/
│   ├── main.py               Entry point — full terminal pipeline
│   ├── data_preprocessing.py Load, merge, aggregate, feature engineer
│   ├── model_training.py     Train 6 models, tune, stack, save best
│   ├── model_inference.py    Load model, forecast, validate inputs
│   └── web_app.py            Flask web app + JSON API
├── templates/      index.html, predict.html, error.html
├── static/css/     style.css
├── tests/          test_pipeline.py (14 unit tests)
├── docs/           documentation.md
├── config/         config.json
├── scripts/        run_pipeline.sh
└── requirements.txt
```

---

## Dataset

**Walmart Sales Forecast** — Kaggle
- 421,570 weekly sales records across 45 stores and 99 departments
- Date range: Feb 2010 – Oct 2012 (~143 weeks)
- External features: Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5, IsHoliday
- Aggregated to total weekly sales (sum across all stores)

---

## Models Compared

| Model              | Description                         |
|--------------------|-------------------------------------|
| Linear Regression  | Baseline (no regularization)        |
| Ridge              | L2-regularized linear model         |
| Lasso              | L1-regularized (sparse features)    |
| ElasticNet         | L1 + L2 combined regularization     |

Linear models only — the dominant feature (lag_52, ~50% importance) creates a
near-linear year-over-year relationship. Tree models and ensembles are
counterproductive on this dataset.

Best model selected automatically by lowest test-set MAE.
Target: MAPE < 10%, R² > 0.80

---

## Key Features Engineered

- Lag features: lag_1, lag_2, lag_4, lag_8, lag_52 (year-over-year)
- Rolling statistics: rolling_mean_4/12, rolling_std_4
- Holiday flags: is_holiday, is_thanksgiving_week, is_christmas_week, is_superbowl_week
- Cyclical encoding: week_sin, week_cos
- External: temperature, fuel_price, cpi, unemployment, total_markdown
- Calendar: week_of_year, month, quarter, year, trend

---

## API Endpoints

| Endpoint        | Method   | Description               |
|-----------------|----------|---------------------------|
| `/`             | GET      | Dashboard                 |
| `/predict`      | GET/POST | Prediction form           |
| `/api/forecast` | GET      | 13-week forecast (JSON)   |
| `/api/metrics`  | GET      | Model metrics (JSON)      |
| `/api/predict`  | POST     | Single prediction (JSON)  |

```bash
# Predict sales for week 47 of 2013 (Thanksgiving)
curl -X POST http://localhost:8501/api/predict \
     -H "Content-Type: application/json" \
     -d '{"year": 2013, "week": 47}'
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```
