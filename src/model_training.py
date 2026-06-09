"""
model_training.py
Trains 4 linear models on Walmart weekly sales data, tunes each one,
evaluates on a held-out test set, and saves the best model with versioning.

Why linear-only: the dominant feature (lag_52, ~50% importance) creates a
near-linear relationship (this week ≈ same week last year × growth factor).
Tree-based models and stacking ensembles are counterproductive here.
"""

import numpy as np
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.linear_model    import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_walmart_data, add_features, FEATURE_COLS

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

TEST_SIZE = 13   # hold out last 13 weeks (~3 months)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_pipeline(model):
    return Pipeline([("sc", StandardScaler()), ("m", model)])


def get_models():
    return {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Ridge":             make_pipeline(Ridge()),
        "Lasso":             make_pipeline(Lasso(max_iter=5000)),
        "ElasticNet":        make_pipeline(ElasticNet(max_iter=5000)),
    }


PARAM_GRIDS = {
    "Ridge":      {"m__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso":      {"m__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "ElasticNet": {"m__alpha": [0.001, 0.01, 0.1, 1.0], "m__l1_ratio": [0.2, 0.5, 0.8]},
    "Linear Regression": {},
}


def evaluate_cv(pipe, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s, mapes = [], [], [], []
    for tr, val in tscv.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds = pipe.predict(X.iloc[val])
        maes.append(mean_absolute_error(y.iloc[val], preds))
        rmses.append(rmse(y.iloc[val], preds))
        r2s.append(r2_score(y.iloc[val], preds))
        mapes.append(mape(y.iloc[val], preds))
    return np.mean(maes), np.mean(rmses), np.mean(r2s), np.mean(mapes)


def tune_model(name, pipe, X, y):
    grid = PARAM_GRIDS.get(name, {})
    if not grid:
        pipe.fit(X, y)
        return pipe, {}
    gs = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=5),
                      scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def get_feature_importance(model, feature_names):
    """Extract feature coefficients as relative importance."""
    try:
        coefs = np.abs(model.named_steps["m"].coef_)
        total = coefs.sum() or 1
        fi = sorted(zip(feature_names, coefs / total), key=lambda x: -x[1])
        return [(f, round(float(i) * 100, 2)) for f, i in fi]
    except Exception:
        return []


def train():
    # ── Load data ────────────────────────────────────────────────────
    weekly   = load_walmart_data()
    featured = add_features(weekly)

    train_df = featured.iloc[:-TEST_SIZE]
    test_df  = featured.iloc[-TEST_SIZE:]
    X_train, y_train = train_df[FEATURE_COLS], train_df["total_sales"]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df["total_sales"]

    print(f"\n  Dataset   : Walmart Sales Forecast (45 stores, weekly)")
    print(f"  Usable weeks : {len(featured)}  |  Train: {len(train_df)}  |  Test: {TEST_SIZE}")
    print(f"  Date range: {featured['Date'].min().date()} --> {featured['Date'].max().date()}")

    # ── Step 1: Cross-validate all models ───────────────────────────
    print("\n" + "="*60)
    print("  STEP 1 — MODEL COMPARISON (5-fold TimeSeriesSplit CV)")
    print("="*60)
    print(f"  {'Model':<22} {'MAE':>14}  {'RMSE':>14}  {'R2':>7}  {'MAPE':>7}")
    print(f"  {'-'*70}")

    cv_results = {}
    for name, pipe in get_models().items():
        mae, rms, r2, mp = evaluate_cv(pipe, X_train, y_train)
        cv_results[name] = {"MAE": mae, "RMSE": rms, "R2": r2, "MAPE": mp}
        print(f"  {name:<22} ${mae:>13,.0f}  ${rms:>13,.0f}  {r2:>7.3f}  {mp:>6.1f}%")

    # ── Step 2: Tune all models ──────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2 — HYPERPARAMETER TUNING")
    print("="*60)

    tuned = {}
    for name, pipe in get_models().items():
        best_pipe, best_params = tune_model(name, pipe, X_train, y_train)
        tuned[name] = best_pipe
        if best_params:
            print(f"  {name}: {best_params}")
        else:
            print(f"  {name}: no hyperparameters to tune")

    # ── Step 3: Test set evaluation ───────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 — TEST SET EVALUATION (last 13 weeks, unseen data)")
    print("="*60)
    print(f"\n  {'Model':<22} {'Test MAE':>14}  {'Test RMSE':>14}  {'Test R2':>8}  {'MAPE':>7}")
    print(f"  {'-'*72}")

    final_results = {}
    for name, model in tuned.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rms   = rmse(y_test, preds)
        r2    = r2_score(y_test, preds)
        mp    = mape(y_test, preds)
        final_results[name] = {"MAE": mae, "RMSE": rms, "R2": r2, "MAPE": mp, "model": model}
        print(f"  {name:<22} ${mae:>13,.0f}  ${rms:>13,.0f}  {r2:>8.3f}  {mp:>6.1f}%")

    best_name = min(final_results, key=lambda k: final_results[k]["MAE"])
    best_m    = final_results[best_name]

    print(f"\n  Best Model : {best_name}")
    print(f"  MAE        : ${best_m['MAE']:,.0f}")
    print(f"  RMSE       : ${best_m['RMSE']:,.0f}")
    print(f"  R2         : {best_m['R2']:.3f}")
    print(f"  MAPE       : {best_m['MAPE']:.1f}%")
    if best_m["MAPE"] < 10:
        print(f"  Target MAPE < 10% : ACHIEVED")
    if best_m["R2"] > 0.80:
        print(f"  Target R2 > 0.80  : ACHIEVED")

    # Retrain best model on ALL available data
    best_model = best_m["model"]
    best_model.fit(featured[FEATURE_COLS], featured["total_sales"])

    # ── Feature importance ───────────────────────────────────────────
    fi = get_feature_importance(best_model, FEATURE_COLS)
    if fi:
        print("\n  Feature Importance (top 10):")
        for feat, imp in fi[:10]:
            bar = "#" * int(imp / 2)
            print(f"  {feat:<24} {imp:>5.1f}%  {bar}")

    # ── Business insights ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  BUSINESS INSIGHTS — WALMART WEEKLY SALES")
    print("="*60)
    avg_sales   = featured["total_sales"].mean()
    q4_avg      = featured[featured["quarter"] == 4]["total_sales"].mean()
    q1_avg      = featured[featured["quarter"] == 1]["total_sales"].mean()
    holiday_avg = featured[featured["is_holiday"] == 1]["total_sales"].mean()
    nonhol_avg  = featured[featured["is_holiday"] == 0]["total_sales"].mean()
    thanksgiving = featured[featured["is_thanksgiving_week"] == 1]["total_sales"].mean()
    christmas    = featured[featured["is_christmas_week"] == 1]["total_sales"].mean()

    print(f"  Avg weekly sales             : ${avg_sales:,.0f}")
    print(f"  Q4 avg (Oct-Dec)             : ${q4_avg:,.0f}  ({q4_avg/avg_sales*100:.0f}% of annual avg)")
    print(f"  Q1 avg (Jan-Mar)             : ${q1_avg:,.0f}  ({q1_avg/avg_sales*100:.0f}% of annual avg)")
    print(f"  Holiday week avg             : ${holiday_avg:,.0f}  (+{(holiday_avg/nonhol_avg-1)*100:.0f}% vs non-holiday)")
    print(f"  Thanksgiving week avg        : ${thanksgiving:,.0f}")
    print(f"  Christmas week avg           : ${christmas:,.0f}")
    if fi:
        print(f"  Strongest predictor          : {fi[0][0]} ({fi[0][1]:.1f}% importance)")

    # ── Save model ───────────────────────────────────────────────────
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_res_clean = {
        k: {mk: round(mv, 4) for mk, mv in r.items() if mk != "model"}
        for k, r in final_results.items()
    }

    artifact = {
        "model":              best_model,
        "name":               best_name,
        "features":           FEATURE_COLS,
        "metrics":            {k: round(v, 4) for k, v in best_m.items() if k != "model"},
        "all_results":        all_res_clean,
        "feature_importance": fi,
        "version":            version,
        "trained_on":         str(datetime.now()),
        "n_train":            len(train_df),
        "n_test":             TEST_SIZE,
        "dataset":            "Walmart Sales Forecast",
    }

    for fname in [f"model_v{version}.pkl", "best_model_latest.pkl"]:
        with open(MODELS_DIR / fname, "wb") as f:
            pickle.dump(artifact, f)

    meta = {k: v for k, v in artifact.items() if k != "model"}
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved model_v{version}.pkl + best_model_latest.pkl")
    print(f"  Metadata: model_metadata.json")

    return best_model, best_name, final_results, fi


if __name__ == "__main__":
    train()
