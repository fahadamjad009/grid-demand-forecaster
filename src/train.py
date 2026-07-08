"""Train LightGBM demand forecaster, walk-forward validate against ETS and naive baselines."""
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pathlib import Path

FEATURES = [
    "temp_max", "temp_min", "temp_mean", "precip_sum", "wind_max",
    "dow", "month", "is_weekend", "doy", "year", "is_holiday",
    "demand_mean_lag1", "demand_mean_lag7", "temp_mean_lag1", "cdd", "hdd",
]
TARGET = "demand_mean"
N_FOLDS = 5
TEST_SIZE = 60  # days per fold

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def walk_forward(df):
    n = len(df)
    results = []
    for fold in range(N_FOLDS):
        test_end = n - fold * TEST_SIZE
        test_start = test_end - TEST_SIZE
        train_end = test_start
        if train_end < 180:  # need a reasonable training history
            break
        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]

        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]

        # LightGBM
        gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, verbose=-1)
        gbm.fit(X_train, y_train)
        pred_gbm = gbm.predict(X_test)

        # Naive lag-7 baseline
        pred_naive = test["demand_mean_lag7"].values

        # ETS / Holt-Winters baseline (univariate on train target only)
        try:
            ets = ExponentialSmoothing(
                y_train.values, trend="add", seasonal="add", seasonal_periods=7
            ).fit()
            pred_ets = ets.forecast(len(y_test))
        except Exception as e:
            pred_ets = np.full(len(y_test), y_train.mean())

        results.append({
            "fold": fold,
            "test_start": str(test.iloc[0]["date"]),
            "test_end": str(test.iloc[-1]["date"]),
            "lightgbm_mae": float(np.mean(np.abs(y_test - pred_gbm))),
            "lightgbm_mape": mape(y_test.values, pred_gbm),
            "ets_mae": float(np.mean(np.abs(y_test - pred_ets))),
            "ets_mape": mape(y_test.values, pred_ets),
            "naive_mae": float(np.mean(np.abs(y_test - pred_naive))),
            "naive_mape": mape(y_test.values, pred_naive),
        })
        print(f"Fold {fold} [{test.iloc[0]['date']} -> {test.iloc[-1]['date']}]: "
              f"LGBM MAE={results[-1]['lightgbm_mae']:.1f} | "
              f"ETS MAE={results[-1]['ets_mae']:.1f} | "
              f"Naive MAE={results[-1]['naive_mae']:.1f}")
    return results

def main():
    df = pd.read_csv("data/model_ready.csv", parse_dates=["date"])
    results = walk_forward(df)

    Path("models").mkdir(exist_ok=True)
    with open("models/walk_forward_results.json", "w") as f:
        json.dump(results, f, indent=2)

    avg_lgbm = np.mean([r["lightgbm_mape"] for r in results])
    avg_ets = np.mean([r["ets_mape"] for r in results])
    avg_naive = np.mean([r["naive_mape"] for r in results])
    print(f"\nAverage MAPE across {len(results)} folds:")
    print(f"  LightGBM: {avg_lgbm:.2f}%")
    print(f"  ETS:      {avg_ets:.2f}%")
    print(f"  Naive:    {avg_naive:.2f}%")

    # Final model trained on all available data, for the live app
    final_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, verbose=-1)
    final_model.fit(df[FEATURES], df[TARGET])
    joblib.dump(final_model, "models/lgbm_final.pkl")
    print("\nSaved final model -> models/lgbm_final.pkl")

if __name__ == "__main__":
    main()