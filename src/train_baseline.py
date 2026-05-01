import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
METRICS_PATH = Path(__file__).parent.parent / "outputs" / "metrics" / "baseline_results.csv"
LOAD_FORECASTS_PATH = Path(__file__).parent.parent / "outputs" / "metrics" / "load_forecasts.csv"

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
DIRECT_FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "month", "horizon_hour",
    "lag_24h", "lag_48h", "lag_168h",
    "pre_lag_1h", "pre_lag_2h", "pre_lag_3h",
    "pre_rolling_mean_24h", "pre_rolling_std_24h",
]
TARGET = "Global_active_power"
FORECAST_HOURS = 24


def available_feature_cols(df: pd.DataFrame) -> list[str]:
    return FEATURE_COLS + [col for col in WEATHER_FEATURE_COLS if col in df.columns]


def available_direct_feature_cols(frame: pd.DataFrame) -> list[str]:
    return DIRECT_FEATURE_COLS + [col for col in WEATHER_FEATURE_COLS if col in frame.columns]


def load_features():
    df = pd.read_csv(FEATURES_PATH, index_col="datetime", parse_dates=True).sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    if len(df) != len(full_idx) or not df.index.equals(full_idx):
        df = df.reindex(full_idx).interpolate(method="time", limit_direction="both").dropna()
        df.index.name = "datetime"
    df.index.freq = pd.tseries.frequencies.to_offset("h")
    return df


def compute_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"{name:<22} MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  R²={r2:.4f}")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2), "R2": round(r2, 4)}


def split(df):
    n = len(df)
    train = df.iloc[:int(n * 0.70)]
    val = df.iloc[int(n * 0.70):int(n * 0.85)]
    test = df.iloc[int(n * 0.85):]
    return train, val, test


def first_full_day_start(test: pd.DataFrame) -> pd.Timestamp:
    """Start forecast evaluation on the first complete calendar day."""
    for date in sorted(set(test.index.date)):
        day = test[test.index.date == date]
        if len(day) == FORECAST_HOURS and day.index.min().hour == 0:
            return day.index.min()
    return test.index.min()


def day_ahead_eval_slice(test: pd.DataFrame) -> pd.DataFrame:
    start_ts = first_full_day_start(test)
    eval_test = test.loc[start_ts:]
    covered = (len(eval_test) // FORECAST_HOURS) * FORECAST_HOURS
    return eval_test.iloc[:covered]


def predict_moving_average_day_ahead(df, test):
    eval_test = day_ahead_eval_slice(test)
    actual_history = df.loc[:eval_test.index[0] - pd.Timedelta(hours=1), TARGET].copy()
    actuals, preds = [], []

    for start in range(0, len(eval_test), FORECAST_HOURS):
        window = eval_test.iloc[start:start + FORECAST_HOURS]
        rolling_history = actual_history.copy()

        for ts, row in window.iterrows():
            previous_24h = rolling_history.loc[ts - pd.Timedelta(hours=24):ts - pd.Timedelta(hours=1)]
            pred = float(previous_24h.mean())
            preds.append(pred)
            actuals.append(float(row[TARGET]))
            rolling_history.loc[ts] = pred

        actual_history = pd.concat([actual_history, window[TARGET]])

    return np.array(actuals), np.array(preds)


def predict_model_day_ahead(model, df, test, model_name, feature_cols=None):
    feature_cols = feature_cols or available_feature_cols(df)
    eval_test = day_ahead_eval_slice(test)

    actual_history = df.loc[:eval_test.index[0] - pd.Timedelta(hours=1), TARGET].copy()
    rows = []

    for start in range(0, len(eval_test), FORECAST_HOURS):
        window = eval_test.iloc[start:start + FORECAST_HOURS]
        issue_time = window.index[0]
        rolling_history = actual_history.copy()

        for horizon, (ts, row) in enumerate(window.iterrows(), start=1):
            features = row[["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month"]].to_dict()
            features["lag_1h"] = rolling_history.loc[ts - pd.Timedelta(hours=1)]
            features["lag_2h"] = rolling_history.loc[ts - pd.Timedelta(hours=2)]
            features["lag_3h"] = rolling_history.loc[ts - pd.Timedelta(hours=3)]
            features["lag_24h"] = rolling_history.loc[ts - pd.Timedelta(hours=24)]
            features["lag_48h"] = rolling_history.loc[ts - pd.Timedelta(hours=48)]
            features["lag_168h"] = rolling_history.loc[ts - pd.Timedelta(hours=168)]

            previous_24h = rolling_history.loc[ts - pd.Timedelta(hours=24):ts - pd.Timedelta(hours=1)]
            features["rolling_mean_3h"] = rolling_history.loc[ts - pd.Timedelta(hours=3):ts - pd.Timedelta(hours=1)].mean()
            features["rolling_mean_6h"] = rolling_history.loc[ts - pd.Timedelta(hours=6):ts - pd.Timedelta(hours=1)].mean()
            features["rolling_mean_24h"] = previous_24h.mean()
            features["rolling_std_24h"] = previous_24h.std()
            for col in feature_cols:
                if col not in features:
                    features[col] = row[col]

            x = pd.DataFrame([features], columns=feature_cols)
            pred = float(model.predict(x)[0])
            rows.append({
                "datetime": ts,
                "issue_time": issue_time,
                "horizon_hour": horizon,
                "actual_kw": float(row[TARGET]),
                "forecast_kw": pred,
                "model": model_name,
            })
            rolling_history.loc[ts] = pred

        actual_history = pd.concat([actual_history, window[TARGET]])

    forecasts = pd.DataFrame(rows).set_index("datetime")
    return forecasts


def build_direct_day_ahead_frame(df: pd.DataFrame,
                                 split_df: pd.DataFrame,
                                 model_name: str | None = None):
    """Build non-leaky direct 24h training rows.

    Each target hour can use calendar features for the forecasted hour plus
    load history that would already be known at the day-ahead issue time.
    """
    rows = []
    for date, window in split_df.groupby(split_df.index.date):
        if len(window) != FORECAST_HOURS or window.index.min().hour != 0:
            continue

        issue_time = window.index.min()
        history = df.loc[:issue_time - pd.Timedelta(hours=1), TARGET]
        if len(history) < 168:
            continue

        pre_24h = history.iloc[-24:]
        common = {
            "pre_lag_1h": float(history.iloc[-1]),
            "pre_lag_2h": float(history.iloc[-2]),
            "pre_lag_3h": float(history.iloc[-3]),
            "pre_rolling_mean_24h": float(pre_24h.mean()),
            "pre_rolling_std_24h": float(pre_24h.std()),
        }

        for horizon, (ts, row) in enumerate(window.iterrows(), start=1):
            features = row[[
                "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "is_weekend", "month", "lag_24h", "lag_48h", "lag_168h",
            ]].to_dict()
            features.update(common)
            for col in WEATHER_FEATURE_COLS:
                if col in row.index:
                    features[col] = row[col]
            features["horizon_hour"] = horizon
            features["datetime"] = ts
            features["issue_time"] = issue_time
            features["actual_kw"] = float(row[TARGET])
            if model_name:
                features["model"] = model_name
            rows.append(features)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.set_index("datetime")


def predict_direct_day_ahead(model, df, test, model_name):
    direct = build_direct_day_ahead_frame(df, day_ahead_eval_slice(test), model_name=model_name)
    direct_feature_cols = available_direct_feature_cols(direct)
    preds = model.predict(direct[direct_feature_cols])
    forecasts = pd.DataFrame({
        "issue_time": direct["issue_time"],
        "horizon_hour": direct["horizon_hour"],
        "actual_kw": direct["actual_kw"],
        "forecast_kw": preds,
        "model": model_name,
    }, index=direct.index)
    forecasts.index.name = "datetime"
    return forecasts


def train_lgbm_model(n_estimators=400):
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="regression_l1",
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )


def train_per_horizon_lgbm(direct_train, direct_val, feature_cols):
    models = {}
    for horizon in range(1, FORECAST_HOURS + 1):
        train_h = direct_train[direct_train["horizon_hour"] == horizon]
        val_h = direct_val[direct_val["horizon_hour"] == horizon]
        model = train_lgbm_model(n_estimators=300)
        model.fit(
            train_h[feature_cols],
            train_h["actual_kw"],
            eval_set=[(val_h[feature_cols], val_h["actual_kw"])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        models[horizon] = model
    return models


def predict_per_horizon_models(models, df, test, model_name):
    direct = build_direct_day_ahead_frame(df, day_ahead_eval_slice(test), model_name=model_name)
    feature_cols = available_direct_feature_cols(direct)
    preds = np.zeros(len(direct), dtype=float)

    for horizon, model in models.items():
        mask = direct["horizon_hour"] == horizon
        if mask.any():
            preds[mask.to_numpy()] = model.predict(direct.loc[mask, feature_cols])

    forecasts = pd.DataFrame({
        "issue_time": direct["issue_time"],
        "horizon_hour": direct["horizon_hour"],
        "actual_kw": direct["actual_kw"],
        "forecast_kw": preds,
        "model": model_name,
    }, index=direct.index)
    forecasts.index.name = "datetime"
    return forecasts


if __name__ == "__main__":
    df = load_features()
    train, val, test = split(df)
    feature_cols = available_feature_cols(df)
    load_feature_cols = FEATURE_COLS

    X_train = train[feature_cols]
    y_train = train[TARGET]

    results = []
    print(f"\n{'Model':<20} {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}")
    print("-" * 50)

    # Naive: predict same hour yesterday
    eval_test = day_ahead_eval_slice(test)
    y_eval = eval_test[TARGET]

    y_naive = eval_test["lag_24h"]
    results.append(compute_metrics(y_eval, y_naive, "Naive day-ahead 24h"))

    # Moving Average: rolling mean of last 24h
    y_ma_true, y_ma_pred = predict_moving_average_day_ahead(df, test)
    results.append(compute_metrics(y_ma_true, y_ma_pred, "Moving Average day-ahead 24h"))

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_forecasts = predict_model_day_ahead(lr, df, test, "Linear Regression day-ahead 24h", feature_cols)
    results.append(compute_metrics(
        lr_forecasts["actual_kw"].values,
        lr_forecasts["forecast_kw"].values,
        "Linear Regression day-ahead 24h",
    ))

    # XGBoost
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8,
                       early_stopping_rounds=20, random_state=42,
                       eval_metric="mae")
    xgb.fit(train[load_feature_cols], y_train,
            eval_set=[(val[load_feature_cols], val[TARGET])],
            verbose=False)
    day_ahead_forecasts = predict_model_day_ahead(
        xgb,
        df,
        test,
        "XGBoost day-ahead 24h",
        load_feature_cols,
    )
    y_roll_true = day_ahead_forecasts["actual_kw"].values
    y_roll_pred = day_ahead_forecasts["forecast_kw"].values
    xgb_result = compute_metrics(y_roll_true, y_roll_pred, "XGBoost day-ahead 24h")
    results.append(xgb_result)
    best_forecasts = day_ahead_forecasts
    best_result = xgb_result

    # Refit with the early-stopped tree count on train+val, then evaluate once on test.
    best_n_estimators = getattr(xgb, "best_iteration", None)
    best_n_estimators = int(best_n_estimators + 1) if best_n_estimators is not None else 500
    train_val = pd.concat([train, val])
    xgb_refit = XGBRegressor(
        n_estimators=best_n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mae",
    )
    xgb_refit.fit(train_val[load_feature_cols], train_val[TARGET], verbose=False)
    refit_forecasts = predict_model_day_ahead(
        xgb_refit,
        df,
        test,
        "XGBoost refit train+val 24h",
        load_feature_cols,
    )
    refit_result = compute_metrics(
        refit_forecasts["actual_kw"].values,
        refit_forecasts["forecast_kw"].values,
        "XGBoost refit train+val 24h",
    )
    results.append(refit_result)
    if refit_result["MAE"] < best_result["MAE"]:
        best_forecasts = refit_forecasts
        best_result = refit_result

    # LightGBM recursive day-ahead baseline.
    lgbm = train_lgbm_model()
    lgbm.fit(
        train[load_feature_cols],
        y_train,
        eval_set=[(val[load_feature_cols], val[TARGET])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    lgbm_forecasts = predict_model_day_ahead(
        lgbm,
        df,
        test,
        "LightGBM day-ahead 24h",
        load_feature_cols,
    )
    lgbm_result = compute_metrics(
        lgbm_forecasts["actual_kw"].values,
        lgbm_forecasts["forecast_kw"].values,
        "LightGBM day-ahead 24h",
    )
    results.append(lgbm_result)
    if lgbm_result["MAE"] < best_result["MAE"]:
        best_forecasts = lgbm_forecasts
        best_result = lgbm_result

    # Direct XGBoost: one non-recursive model with horizon as a feature.
    direct_train = build_direct_day_ahead_frame(df, train)
    direct_val = build_direct_day_ahead_frame(df, val)
    direct_feature_cols = available_direct_feature_cols(direct_train)
    xgb_direct = XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        early_stopping_rounds=30,
        random_state=42,
        eval_metric="mae",
    )
    xgb_direct.fit(
        direct_train[direct_feature_cols],
        direct_train["actual_kw"],
        eval_set=[(direct_val[direct_feature_cols], direct_val["actual_kw"])],
        verbose=False,
    )
    direct_forecasts = predict_direct_day_ahead(
        xgb_direct,
        df,
        test,
        "XGBoost direct horizon 24h",
    )
    direct_result = compute_metrics(
        direct_forecasts["actual_kw"].values,
        direct_forecasts["forecast_kw"].values,
        "XGBoost direct horizon 24h",
    )
    results.append(direct_result)

    if direct_result["MAE"] < best_result["MAE"]:
        best_forecasts = direct_forecasts
        best_result = direct_result

    # Direct per-horizon LightGBM: one model per forecast hour.
    lgbm_horizon_models = train_per_horizon_lgbm(
        direct_train,
        direct_val,
        direct_feature_cols,
    )
    lgbm_horizon_forecasts = predict_per_horizon_models(
        lgbm_horizon_models,
        df,
        test,
        "LightGBM per-horizon 24h",
    )
    lgbm_horizon_result = compute_metrics(
        lgbm_horizon_forecasts["actual_kw"].values,
        lgbm_horizon_forecasts["forecast_kw"].values,
        "LightGBM per-horizon 24h",
    )
    results.append(lgbm_horizon_result)
    if lgbm_horizon_result["MAE"] < best_result["MAE"]:
        best_forecasts = lgbm_horizon_forecasts
        best_result = lgbm_horizon_result

    # Build forecast DataFrames for Naive and Moving Average (they return arrays, not DataFrames)
    ev_idx = eval_test.index
    n_complete = (len(ev_idx) // FORECAST_HOURS) * FORECAST_HOURS
    ev_idx = ev_idx[:n_complete]
    h_hours = list(range(1, FORECAST_HOURS + 1)) * (n_complete // FORECAST_HOURS)

    naive_forecasts = pd.DataFrame({
        "issue_time": ev_idx.floor("D"),
        "horizon_hour": h_hours,
        "actual_kw": y_eval.values[:n_complete],
        "forecast_kw": y_naive.values[:n_complete],
        "model": "Naive day-ahead 24h",
    }, index=ev_idx)
    naive_forecasts.index.name = "datetime"

    ma_n = min(len(y_ma_true), n_complete)
    ma_idx = ev_idx[:ma_n]
    ma_forecasts = pd.DataFrame({
        "issue_time": ma_idx.floor("D"),
        "horizon_hour": h_hours[:ma_n],
        "actual_kw": y_ma_true[:ma_n],
        "forecast_kw": y_ma_pred[:ma_n],
        "model": "Moving Average day-ahead 24h",
    }, index=ma_idx)
    ma_forecasts.index.name = "datetime"

    _cols = ["issue_time", "horizon_hour", "actual_kw", "forecast_kw", "model"]
    all_baseline_forecasts = pd.concat([
        naive_forecasts[_cols],
        ma_forecasts[_cols],
        lr_forecasts[_cols],
        day_ahead_forecasts[_cols],
        refit_forecasts[_cols],
        lgbm_forecasts[_cols],
        direct_forecasts[_cols],
        lgbm_horizon_forecasts[_cols],
    ])

    BASELINE_MODEL_NAMES = {f["model"] for f in results}

    # Preserve any neural model forecasts already saved
    if LOAD_FORECASTS_PATH.exists():
        existing = pd.read_csv(LOAD_FORECASTS_PATH, index_col="datetime", parse_dates=True)
        if "model" in existing.columns:
            neural = existing[~existing["model"].isin(BASELINE_MODEL_NAMES)]
            if not neural.empty:
                all_baseline_forecasts = pd.concat([all_baseline_forecasts, neural])

    # Save results
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(METRICS_PATH, index=False)
    all_baseline_forecasts.sort_index().to_csv(LOAD_FORECASTS_PATH)
    print(f"\nSaved: {METRICS_PATH}")
    print(f"Saved: {LOAD_FORECASTS_PATH} ({all_baseline_forecasts['model'].nunique()} models)")
