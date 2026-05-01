from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "hourly_clean.csv"

NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
MINUTES_PER_DAY = 24 * 60


def fill_missing_previous_day(df: pd.DataFrame) -> pd.DataFrame:
    """Fill minute-level gaps with the same minute from the previous day."""
    df = df.copy()
    values = df[NUMERIC_COLS].copy()
    missing_before = int(values.isna().sum().sum())

    # The UCI/Kaggle examples usually fill a missing minute from the same clock
    # minute on the previous day before any resampling is done.
    for _ in range(14):
        missing_now = int(values.isna().sum().sum())
        if missing_now == 0:
            break

        values = values.fillna(values.shift(MINUTES_PER_DAY))
        missing_after = int(values.isna().sum().sum())
        if missing_after == missing_now:
            break

    missing_after_previous_day = int(values.isna().sum().sum())

    # Fallback for the first day or long gaps where previous-day values are also
    # missing. This keeps the hourly series continuous for lag features.
    values = values.interpolate(method="time", limit_direction="both")
    values = values.fillna(values.mean(numeric_only=True))

    df[NUMERIC_COLS] = values
    missing_after = int(df[NUMERIC_COLS].isna().sum().sum())
    print(
        "Missing numeric values: "
        f"{missing_before} -> {missing_after_previous_day} after previous-day fill "
        f"-> {missing_after} after fallback fill"
    )
    return df


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, sep=";", na_values=["?", ""], low_memory=False)

    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")

    full_minute_idx = pd.date_range(df.index.min(), df.index.max(), freq="min")
    df = df.reindex(full_minute_idx)
    df.index.name = "datetime"

    df = fill_missing_previous_day(df)

    # Active energy not captured by the three household sub-meterings, in Wh.
    df["Sub_metering_4"] = (
        df["Global_active_power"] * 1000 / 60
        - df["Sub_metering_1"]
        - df["Sub_metering_2"]
        - df["Sub_metering_3"]
    )

    # Resample to 1 hour, keeping the same mean-kW target used by the models.
    df_hourly = df.resample("1h").mean()

    # Defensive guard in case the raw file has missing rows at an hourly edge.
    df_hourly = df_hourly.asfreq("h").interpolate(method="time", limit_direction="both")

    return df_hourly


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH)
    print(f"Saved: {PROCESSED_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Range: {df.index.min()} to {df.index.max()}")


if __name__ == "__main__":
    print("Loading and cleaning...")
    df = load_and_clean()
    save_processed(df)
