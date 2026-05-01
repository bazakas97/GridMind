"""
XGBoost day-ahead electricity price forecaster.

Uses the same temporal features as the load forecaster.
Train/val/test split mirrors the load model (70/15/15 chronological).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
PRICES_PATH   = Path(__file__).parent.parent / "data" / "processed" / "prices.csv"
METRICS_PATH  = Path(__file__).parent.parent / "outputs" / "metrics"
PLOTS_PATH    = Path(__file__).parent.parent / "outputs" / "plots"

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "month",
    "lag_24h", "lag_48h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h",
]
WEATHER_FEATURE_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "heating_degree_c", "cooling_degree_c",
    "temperature_lag_24h", "temperature_roll_24h",
]


def load_splits():
    features = pd.read_csv(FEATURES_PATH, index_col="datetime", parse_dates=True)
    prices   = pd.read_csv(PRICES_PATH, index_col="datetime", parse_dates=True)["price_eur_kwh"]

    feature_cols = FEATURE_COLS + [col for col in WEATHER_FEATURE_COLS if col in features.columns]
    df = features[feature_cols].copy()
    df["price"] = prices

    # Add price lags (past prices help predict future prices)
    df["price_lag_24h"]  = df["price"].shift(24)
    df["price_lag_48h"]  = df["price"].shift(48)
    df["price_lag_168h"] = df["price"].shift(168)
    df["price_roll_24h"] = df["price"].shift(1).rolling(24).mean()

    df = df.dropna()

    n = len(df)
    train = df.iloc[:int(n * 0.70)]
    val   = df.iloc[int(n * 0.70):int(n * 0.85)]
    test  = df.iloc[int(n * 0.85):]

    price_feat_cols = feature_cols + [
        "price_lag_24h", "price_lag_48h", "price_lag_168h", "price_roll_24h"
    ]
    return train, val, test, price_feat_cols


def compute_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    price_range = float(np.max(y_true) - np.min(y_true))
    nmae = mae / price_range if price_range > 0 else np.nan
    print(f"{name:30s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  nMAE={nmae:.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "nMAE": nmae}


def train_price_model():
    train, val, test, feat_cols = load_splits()

    X_train, y_train = train[feat_cols], train["price"]
    X_val,   y_val   = val[feat_cols],   val["price"]
    X_test,  y_test  = test[feat_cols],  test["price"]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_test)
    metrics = compute_metrics(y_test.values, preds, "XGBoost Price Forecaster")

    # Save test predictions
    pred_df = pd.DataFrame({
        "price_actual":   y_test.values,
        "price_forecast": preds,
    }, index=test.index)
    pred_df.index.name = "datetime"

    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(METRICS_PATH / "price_forecasts.csv")
    pd.DataFrame([metrics]).to_csv(METRICS_PATH / "price_model_results.csv", index=False)
    print(f"Saved: price_forecasts.csv, price_model_results.csv")

    # Quick plot of one sample week
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    sample = pred_df.iloc[:168]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sample["price_actual"].values,   label="Actual price",    color="steelblue", linewidth=1.5)
    ax.plot(sample["price_forecast"].values, label="Forecast price",  color="tomato",    linewidth=1.5, linestyle="--")
    ax.fill_between(range(len(sample)),
                    sample["price_actual"].values,
                    sample["price_forecast"].values,
                    alpha=0.15, color="tomato")
    ax.set_xlabel("Hour")
    ax.set_ylabel("€/kWh")
    ax.set_title("Electricity Price: Actual vs Forecast (first week of test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "price_forecast.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: price_forecast.png")

    return model, metrics


if __name__ == "__main__":
    train_price_model()
