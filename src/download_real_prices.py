"""
Download real European day-ahead electricity prices from OPSD
(Open Power System Data, sourced from ENTSO-E Transparency Platform)
and build a seasonally-mapped price series aligned to the UCI load dataset.

Source column: DE_LU_price_day_ahead (Germany-Luxembourg, €/MWh)
Note: French and German day-ahead prices are tightly coupled (CWE market
      coupling zone), making DE prices a valid proxy for the 2006-2010 period
      when direct FR data is unavailable.

Seasonal mapping:
  OPSD covers 2015-2020; the UCI load data covers 2006-2010.
  For every hour in the load dataset we sample a real price from the
  matching (month, hour-of-day, day-of-week) bucket.  This preserves
  genuine daily, weekly, and seasonal market patterns.
"""
import csv
import io
import numpy as np
import pandas as pd
import requests
from pathlib import Path

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
PRICES_PATH   = Path(__file__).parent.parent / "data" / "processed" / "prices.csv"
CACHE_PATH    = Path(__file__).parent.parent / "data" / "processed" / "opsd_de_prices.csv"

OPSD_URL       = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
PRICE_COL      = "DE_LU_price_day_ahead"   # €/MWh
TIMESTAMP_COL  = "utc_timestamp"


def fetch_opsd_prices(force_download: bool = False) -> pd.Series:
    """
    Returns an hourly Series of real European day-ahead prices (€/kWh).
    Streams the OPSD CSV line-by-line, extracting only the two columns
    needed; caches the result locally (~3 MB) for reuse.
    """
    if CACHE_PATH.exists() and not force_download:
        print(f"Loading cached OPSD prices from {CACHE_PATH.name}")
        return (
            pd.read_csv(CACHE_PATH, index_col=TIMESTAMP_COL, parse_dates=True)
            [PRICE_COL] / 1000.0
        )

    print("Streaming OPSD time-series (extracting 2 columns from ~124 MB file)…")

    with requests.get(OPSD_URL, stream=True, timeout=300) as r:
        r.raise_for_status()

        # Decode stream line by line
        rows   = []
        header = None
        ts_idx = None
        px_idx = None
        n_downloaded = 0

        for raw_line in r.iter_lines():
            n_downloaded += len(raw_line) + 1

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parsed = next(csv.reader(io.StringIO(line)))

            if header is None:
                header = parsed
                ts_idx = header.index(TIMESTAMP_COL)
                px_idx = header.index(PRICE_COL)
                print(f"  Found columns: [{TIMESTAMP_COL}] [{PRICE_COL}]")
                continue

            val = parsed[px_idx].strip()
            if val:
                rows.append((parsed[ts_idx], float(val)))

            if n_downloaded % (5 << 20) < 300:   # progress every ~5 MB
                print(f"\r  {n_downloaded/1e6:.0f} MB streamed, "
                      f"{len(rows):,} valid rows", end="", flush=True)

    print(f"\n  Done — {len(rows):,} rows")

    df = pd.DataFrame(rows, columns=[TIMESTAMP_COL, PRICE_COL])
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.set_index(TIMESTAMP_COL).sort_index()
    df.index = df.index.tz_localize(None)   # drop UTC offset for easy matching

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH)
    print(f"Cached: {CACHE_PATH.name}  ({CACHE_PATH.stat().st_size/1e6:.1f} MB)")

    return df[PRICE_COL] / 1000.0   # €/MWh → €/kWh


def seasonal_map_prices(load_index: pd.DatetimeIndex,
                        real_prices: pd.Series,
                        seed: int = 42) -> pd.Series:
    """
    For every timestamp in load_index sample a real price from the
    (month, hour, day-of-week) bucket derived from real_prices.
    """
    rng = np.random.default_rng(seed)

    rp = real_prices.copy()
    rp.index = rp.index.tz_localize(None) if rp.index.tz is not None else rp.index

    # Build lookup table
    lookup: dict = {}
    for ts, val in rp.items():
        key = (ts.month, ts.hour, ts.dayofweek)
        lookup.setdefault(key, []).append(val)
    lookup = {k: np.array(v) for k, v in lookup.items()}

    prices_out = np.empty(len(load_index))
    for i, ts in enumerate(load_index):
        key = (ts.month, ts.hour, ts.dayofweek)
        bucket = lookup.get(key)
        if bucket is None or len(bucket) == 0:
            prices_out[i] = rp.median()
        else:
            prices_out[i] = rng.choice(bucket)

    return pd.Series(prices_out, index=load_index, name="price_eur_kwh")


if __name__ == "__main__":
    real_prices = fetch_opsd_prices()
    print(f"\nOPSD prices: {real_prices.index.min()} → {real_prices.index.max()}")
    print(f"  Range : {real_prices.min():.4f} – {real_prices.max():.4f} €/kWh")
    print(f"  Mean  : {real_prices.mean():.4f} €/kWh")

    features      = pd.read_csv(FEATURES_PATH, index_col="datetime", parse_dates=True)
    prices_mapped = seasonal_map_prices(features.index, real_prices)

    prices_mapped.to_csv(PRICES_PATH, header=True)
    print(f"\nSaved: {PRICES_PATH}")
    print(f"  Rows  : {len(prices_mapped):,}")
    print(f"  Range : {prices_mapped.min():.4f} – {prices_mapped.max():.4f} €/kWh")
    print(f"  Mean  : {prices_mapped.mean():.4f} €/kWh")
