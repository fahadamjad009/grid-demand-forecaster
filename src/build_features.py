"""Aggregate half-hourly AEMO demand to daily, merge weather + calendar features."""
import pandas as pd
import numpy as np

def main():
    demand = pd.read_csv("data/nem_price_demand.csv", parse_dates=["SETTLEMENTDATE"])
    demand["date"] = demand["SETTLEMENTDATE"].dt.date

    daily = demand.groupby("date").agg(
        demand_mean=("TOTALDEMAND", "mean"),
        demand_max=("TOTALDEMAND", "max"),
        demand_min=("TOTALDEMAND", "min"),
        price_mean=("RRP", "mean"),
        price_max=("RRP", "max"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    weather = pd.read_csv("data/weather_sydney.csv", parse_dates=["date"])
    df = daily.merge(weather, on="date", how="inner")

    # Calendar features
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    # Simple AU public holiday flag (national fixed-date holidays only, approximation)
    fixed_holidays = df["date"].dt.strftime("%m-%d").isin(
        ["01-01", "01-26", "04-25", "12-25", "12-26"]
    )
    df["is_holiday"] = fixed_holidays.astype(int)

    # Lag features (yesterday, same day last week)
    df = df.sort_values("date").reset_index(drop=True)
    df["demand_mean_lag1"] = df["demand_mean"].shift(1)
    df["demand_mean_lag7"] = df["demand_mean"].shift(7)
    df["temp_mean_lag1"] = df["temp_mean"].shift(1)

    # Heating/cooling degree-day style features (base 18C)
    df["cdd"] = (df["temp_mean"] - 18).clip(lower=0)
    df["hdd"] = (18 - df["temp_mean"]).clip(lower=0)

    df = df.dropna().reset_index(drop=True)
    df.to_csv("data/model_ready.csv", index=False)
    print(f"Saved {len(df)} daily rows, {df.shape[1]} columns -> data/model_ready.csv")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

if __name__ == "__main__":
    main()