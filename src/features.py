import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "hourly_clean.csv"
FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
WEATHER_PATH = Path(__file__).parent.parent / "data" / "processed" / "weather.csv"


def make_continuous_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df = df.interpolate(method="time", limit_direction="both")
    df.index.name = "datetime"
    df.index.freq = pd.tseries.frequencies.to_offset("h")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if WEATHER_PATH.exists():
        weather = pd.read_csv(WEATHER_PATH, index_col="datetime", parse_dates=True).sort_index()
        weather = weather[~weather.index.duplicated(keep="first")]
        weather = weather.reindex(df.index).interpolate(method="time", limit_direction="both")
        df = df.join(weather)

    # time features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # sine/cosine encoding for cyclical features (hour and day repeat)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # lag features
    df["lag_1h"] = df["Global_active_power"].shift(1)     # previous hour
    df["lag_2h"] = df["Global_active_power"].shift(2)     # two hours ago
    df["lag_3h"] = df["Global_active_power"].shift(3)     # three hours ago
    df["lag_24h"] = df["Global_active_power"].shift(24)   # same hour yesterday
    df["lag_48h"] = df["Global_active_power"].shift(48)   # same hour 2 days ago
    df["lag_168h"] = df["Global_active_power"].shift(168) # same hour last week

    # rolling statistics
    df["rolling_mean_3h"] = df["Global_active_power"].shift(1).rolling(3).mean()
    df["rolling_mean_6h"] = df["Global_active_power"].shift(1).rolling(6).mean()
    df["rolling_mean_24h"] = df["Global_active_power"].shift(1).rolling(24).mean()
    df["rolling_std_24h"] = df["Global_active_power"].shift(1).rolling(24).std()

    if "temperature_2m" in df.columns:
        df["heating_degree_c"] = np.clip(18.0 - df["temperature_2m"], 0, None)
        df["cooling_degree_c"] = np.clip(df["temperature_2m"] - 22.0, 0, None)
        df["temperature_lag_24h"] = df["temperature_2m"].shift(24)
        df["temperature_roll_24h"] = df["temperature_2m"].shift(1).rolling(24).mean()

    # drop rows where lag features are NaN (first 168 hours)
    df = df.dropna()

    return df


def split_data(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"Train: {train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)")
    print(f"Val:   {val.index.min().date()} to {val.index.max().date()} ({len(val)} rows)")
    print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)")

    return train, val, test


if __name__ == "__main__":
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_PATH, index_col="datetime", parse_dates=True)

    print("Making hourly index continuous...")
    df = make_continuous_hourly(df)

    print("Adding features...")
    df = add_features(df)

    print("Splitting...")
    train, val, test = split_data(df)

    df.to_csv(FEATURES_PATH)
    print(f"\nSaved: {FEATURES_PATH}")
    print(f"Total shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
