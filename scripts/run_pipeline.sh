#!/bin/bash
# run_pipeline.sh — Full pipeline end-to-end (Walmart Sales Forecasting)

set -e
echo "========================================"
echo "  WALMART SALES FORECASTING PIPELINE"
echo "========================================"

echo -e "\n[1/4] Installing dependencies..."
pip install -r requirements.txt -q

echo -e "\n[2/4] Preprocessing Walmart data..."
python src/data_preprocessing.py

echo -e "\n[3/4] Training & comparing models..."
python src/model_training.py

echo -e "\n[4/4] Running 13-week forecast..."
python src/model_inference.py

echo -e "\nDone! Launch web app: python src/web_app.py"
echo "   URL: http://127.0.0.1:8501"
