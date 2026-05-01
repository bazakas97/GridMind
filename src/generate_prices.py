"""
Synthetic day-ahead electricity price generation.

Produces a realistic dynamic price signal correlated with household load,
mimicking European spot market behavior (daily pattern + demand correlation
+ random day-to-day volatility).
"""
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
PRICES_PATH   = Path(__file__).parent.parent / "data" / "processed" / "prices.csv"


def generate_dynamic_prices(df: pd.DataFrame) -> pd.Series:
    """
    price[t] = base_curve[hour] + load_component + daily_noise + hourly_noise

    Base curve: sinusoidal daily pattern typical of European day-ahead market
      - valley around 03:00–05:00
      - morning ramp 06:00–09:00
      - midday plateau
      - evening peak around 18:00–20:00
    """
    np.random.seed(42)
    hours = df.index.hour.values

    # Two harmonics give a realistic double-hump shape
    base = (
        0.13
        + 0.07 * np.sin(2 * np.pi * (hours - 5)  / 24)
        + 0.03 * np.sin(4 * np.pi * (hours - 4)  / 24)
    )

    # Load-correlated component (demand ↑ → price ↑)
    load      = df["Global_active_power"].values
    load_norm = (load - load.mean()) / (load.std() + 1e-8)
    load_comp = 0.025 * load_norm

    # Day-level volatility (same offset for every hour of a given calendar day)
    dates        = df.index.date
    unique_dates = np.unique(dates)
    daily_shock  = {d: np.random.normal(0, 0.014) for d in unique_dates}
    daily_noise  = np.array([daily_shock[d] for d in dates])

    # Hour-level noise
    hourly_noise = np.random.normal(0, 0.007, len(df))

    prices = base + load_comp + daily_noise + hourly_noise
    prices = np.clip(prices, 0.03, 0.45)   # realistic European market bounds

    return pd.Series(prices, index=df.index, name="price_eur_kwh")


if __name__ == "__main__":
    df     = pd.read_csv(FEATURES_PATH, index_col="datetime", parse_dates=True)
    prices = generate_dynamic_prices(df)

    prices.to_csv(PRICES_PATH, header=True)
    print(f"Saved: {PRICES_PATH}")
    print(f"Rows  : {len(prices)}")
    print(f"Min   : {prices.min():.4f} €/kWh")
    print(f"Max   : {prices.max():.4f} €/kWh")
    print(f"Mean  : {prices.mean():.4f} €/kWh")
    print(f"Std   : {prices.std():.4f} €/kWh")
