"""Pull daily Sydney weather history from Open-Meteo (free, no API key) to match the AEMO date range."""
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import date, timedelta

SYDNEY_LAT, SYDNEY_LON = -33.8688, 151.2093

def main():
    demand = pd.read_csv("data/nem_price_demand.csv", parse_dates=["SETTLEMENTDATE"])
    start_date = demand["SETTLEMENTDATE"].min().date().isoformat()
    end_date = min(demand["SETTLEMENTDATE"].max().date(), date.today() - timedelta(days=1)).isoformat()
    print(f"Fetching weather {start_date} to {end_date}...")

    cache_session = requests_cache.CachedSession(".cache_weather", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    resp = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": SYDNEY_LAT,
            "longitude": SYDNEY_LON,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                      "precipitation_sum", "wind_speed_10m_max"],
            "timezone": "Australia/Sydney",
        },
    )[0]

    daily = resp.Daily()
    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    )
    dates = dates.tz_convert("Australia/Sydney").tz_localize(None).normalize()

    df = pd.DataFrame({
        "date": dates,
        "temp_max": daily.Variables(0).ValuesAsNumpy(),
        "temp_min": daily.Variables(1).ValuesAsNumpy(),
        "temp_mean": daily.Variables(2).ValuesAsNumpy(),
        "precip_sum": daily.Variables(3).ValuesAsNumpy(),
        "wind_max": daily.Variables(4).ValuesAsNumpy(),
    })
    df.to_csv("data/weather_sydney.csv", index=False)
    print(f"Saved {len(df)} daily weather rows -> data/weather_sydney.csv")

if __name__ == "__main__":
    main()