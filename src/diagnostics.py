"""Train on all-but-last-90-days, predict the held-out window for honest actual-vs-predicted diagnostics."""
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATURES = [
    "temp_max", "temp_min", "temp_mean", "precip_sum", "wind_max",
    "dow", "month", "is_weekend", "doy", "year", "is_holiday",
    "demand_mean_lag1", "demand_mean_lag7", "temp_mean_lag1", "cdd", "hdd",
]
TARGET = "demand_mean"
HOLDOUT_DAYS = 90

def main():
    df = pd.read_csv("data/model_ready.csv", parse_dates=["date"])
    train = df.iloc[:-HOLDOUT_DAYS]
    holdout = df.iloc[-HOLDOUT_DAYS:].copy()

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, verbose=-1)
    model.fit(train[FEATURES], train[TARGET])

    holdout["predicted"] = model.predict(holdout[FEATURES])
    holdout["residual"] = holdout[TARGET] - holdout["predicted"]
    holdout["abs_pct_error"] = (holdout["residual"].abs() / holdout[TARGET]) * 100

    out = holdout[["date", TARGET, "predicted", "residual", "abs_pct_error"]]
    out.to_csv("data/diagnostics.csv", index=False)
    print(f"Held-out window: {holdout['date'].min().date()} to {holdout['date'].max().date()}")
    print(f"Out-of-sample MAPE: {out['abs_pct_error'].mean():.2f}%")
    print(f"Saved -> data/diagnostics.csv")

if __name__ == "__main__":
    main()