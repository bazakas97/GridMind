"""
Download historical hourly weather near the UCI household location.

The UCI household dataset was collected near Sceaux, France, close to Paris.
These weather features are exogenous inputs for load forecasting; in a real
day-ahead system they would come from a weather forecast provider.
"""
from pathlib import Path

import pandas as pd
import requests


PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "hourly_clean.csv"
WEATHER_PATH = Path(__file__).parent.parent / "data" / "processed" / "weather.csv"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = 48.78
LONGITUDE = 2.29
HOURLY_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m"


def download_weather(start_date, end_date) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": HOURLY_VARS,
        "timezone": "Europe/Paris",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    hourly = payload["hourly"]
    weather = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
        "wind_speed_10m": hourly["wind_speed_10m"],
    })
    weather = weather.set_index("datetime").sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]
    return weather


if __name__ == "__main__":
    load = pd.read_csv(PROCESSED_PATH, index_col="datetime", parse_dates=True)
    start_date = load.index.min().date()
    end_date = load.index.max().date()

    print(f"Downloading weather for {start_date} to {end_date}...")
    weather = download_weather(start_date, end_date)

    WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(WEATHER_PATH)
    print(f"Saved: {WEATHER_PATH}")
    print(f"Rows: {len(weather):,}")
    print(weather.describe().round(2).to_string())
