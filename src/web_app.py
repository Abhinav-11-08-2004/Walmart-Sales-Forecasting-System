"""
web_app.py
Flask web application for the Walmart Sales Forecasting system.
Routes:
  GET  /             -> Dashboard (chart + metrics + feature importance)
  GET/POST /predict  -> Single week/year prediction form
  GET  /api/forecast -> JSON: 13-week forecast
  GET  /api/metrics  -> JSON: model metrics
  POST /api/predict  -> JSON: predict any year/week
"""

import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, str(Path(__file__).parent))
from model_inference import load_model, forecast_weeks, predict_single, validate_inputs

BASE_DIR = Path(__file__).parent.parent
app = Flask(__name__,
            template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "static"))

# Load model once at startup
try:
    ARTIFACT = load_model()
    FORECAST = forecast_weeks(13, ARTIFACT)
    print(f"Model loaded: {ARTIFACT['name']} (v{ARTIFACT['version']})")
except Exception as e:
    ARTIFACT = None
    FORECAST = []
    print(f"Model not loaded: {e}. Run model_training.py first.")


@app.route("/")
def index():
    if not ARTIFACT:
        return render_template("error.html", message="Model not found. Run model_training.py first.")

    metrics = ARTIFACT["metrics"]
    fi      = ARTIFACT.get("feature_importance", [])[:8]

    hist_labels, hist_values = [], []
    try:
        from data_preprocessing import load_walmart_data
        weekly = load_walmart_data()
        # Show last 52 weeks of historical data
        recent = weekly.tail(52)
        hist_labels = recent["Date"].dt.strftime("%Y-%m-%d").tolist()
        hist_values = [round(v, 2) for v in recent["total_sales"].tolist()]
    except Exception:
        pass

    fc_labels = [d for d, _ in FORECAST]
    fc_values = [v for _, v in FORECAST]

    return render_template("index.html",
        model_name   = ARTIFACT["name"],
        version      = ARTIFACT["version"],
        metrics      = metrics,
        fi_data      = fi,
        hist_labels  = json.dumps(hist_labels),
        hist_values  = json.dumps(hist_values),
        fc_labels    = json.dumps(fc_labels),
        fc_values    = json.dumps(fc_values),
        forecast_rows= FORECAST,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result   = None
    errors   = []
    year_val = ""
    week_val = ""

    if request.method == "POST":
        year_val = request.form.get("year", "")
        week_val = request.form.get("week", "")
        try:
            year = int(year_val)
            week = int(week_val)
            validate_inputs(year, week)
            result = predict_single(year, week, ARTIFACT)
        except (ValueError, TypeError) as e:
            errors = [str(e)]
        except Exception as e:
            errors = [f"Prediction error: {e}"]

    return render_template("predict.html",
        result     = result,
        errors     = errors,
        year_val   = year_val,
        week_val   = week_val,
        model_name = ARTIFACT["name"] if ARTIFACT else "N/A",
        mape       = ARTIFACT["metrics"]["MAPE"] if ARTIFACT else 0,
    )


@app.route("/api/forecast")
def api_forecast():
    if not ARTIFACT:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify([{"date": d, "predicted_sales": v} for d, v in FORECAST])


@app.route("/api/metrics")
def api_metrics():
    if not ARTIFACT:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "model_name":  ARTIFACT["name"],
        "version":     ARTIFACT["version"],
        "metrics":     ARTIFACT["metrics"],
        "all_results": ARTIFACT.get("all_results", {}),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ARTIFACT:
        return jsonify({"error": "Model not loaded"}), 503
    data = request.get_json(force=True) or {}
    try:
        year = int(data.get("year"))
        week = int(data.get("week"))
        return jsonify(predict_single(year, week, ARTIFACT))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\nStarting Walmart Sales Forecasting Web App...")
    print("   URL: http://127.0.0.1:8501\n")
    app.run(debug=True, host="0.0.0.0", port=8501)
