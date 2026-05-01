"""
Model training and evaluation.

- Chronos-2 (Amazon): optional zero-shot foundation model
 - N-HiTS / N-BEATS / TFT (Darts): trained neural models with rolling 24h evaluation
"""
import argparse
import importlib.util
import importlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
METRICS_PATH = Path(__file__).parent.parent / "outputs" / "metrics" / "model_results.csv"
LOAD_FORECASTS_PATH = Path(__file__).parent.parent / "outputs" / "metrics" / "load_forecasts.csv"
RUNS_PATH = Path(__file__).parent.parent / "outputs" / "metrics" / "experiment_runs.csv"
PLOTS_PATH = Path(__file__).parent.parent / "outputs" / "plots"
MODELS_PATH = Path(__file__).parent.parent / "outputs" / "models"
LOGS_PATH = Path(__file__).parent.parent / "outputs" / "logs"
DARTS_WORK_DIR = MODELS_PATH / "darts_checkpoints"

TARGET = "Global_active_power"
FORECAST_HOURS = 24

PAST_COV_COLS = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "Sub_metering_4",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "heating_degree_c",
    "cooling_degree_c",
    "temperature_lag_24h",
    "temperature_roll_24h",
]

COMPACT_COV_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "heating_degree_c",
    "cooling_degree_c",
    "temperature_lag_24h",
    "temperature_roll_24h",
]

KNOWN_FUTURE_COV_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "heating_degree_c",
    "cooling_degree_c",
]

NHITS_CONFIGS = [
    {
        "name": "N-HiTS compact lags MAE 168h",
        "model_name": "nhits_compact_lags_mae_168h",
        "input_chunk_length": 168,
        "num_stacks": 3,
        "num_blocks": 2,
        "num_layers": 2,
        "layer_widths": 256,
        "dropout": 0.1,
        "n_epochs": 80,
        "batch_size": 64,
        "lr": 3e-4,
        "loss": "mae",
        "scaler": "standard",
        "covariates": COMPACT_COV_COLS,
    },
    {
        "name": "N-HiTS tuned 168h",
        "model_name": "nhits_tuned_168h",
        "input_chunk_length": 168,
        "num_stacks": 3,
        "num_blocks": 2,
        "num_layers": 2,
        "layer_widths": 512,
        "dropout": 0.05,
        "n_epochs": 100,
        "batch_size": 128,
        "lr": 5e-4,
    },
    {
        "name": "N-HiTS tuned 336h",
        "model_name": "nhits_tuned_336h",
        "input_chunk_length": 336,
        "num_stacks": 3,
        "num_blocks": 2,
        "num_layers": 2,
        "layer_widths": 512,
        "dropout": 0.05,
        "n_epochs": 100,
        "batch_size": 128,
        "lr": 5e-4,
    },
]

NBEATS_CONFIGS = [
    {
        "name": "N-BEATS 168h",
        "model_name": "nbeats_168h",
        "model_type": "nbeats",
        "input_chunk_length": 168,
        "num_stacks": 12,
        "num_blocks": 1,
        "num_layers": 3,
        "layer_widths": 256,
        "dropout": 0.05,
        "n_epochs": 60,
        "batch_size": 64,
        "lr": 3e-4,
        "loss": "mae",
        "scaler": "standard",
        "progress_bar": False,
        "log_every_n_steps": 50,
    },
]

TFT_CONFIGS = [
    {
        "name": "TFT known covariates 168h",
        "model_name": "tft_known_covs_168h",
        "model_type": "tft",
        "input_chunk_length": 168,
        "hidden_size": 32,
        "lstm_layers": 1,
        "num_attention_heads": 4,
        "dropout": 0.1,
        "hidden_continuous_size": 16,
        "n_epochs": 60,
        "batch_size": 64,
        "lr": 3e-4,
        "loss": "mae",
        "scaler": "standard",
        "covariates": KNOWN_FUTURE_COV_COLS,
        "progress_bar": False,
        "log_every_n_steps": 50,
    },
]

MODEL_CONFIGS = {
    "nhits": [NHITS_CONFIGS[0]],
    "nbeats": NBEATS_CONFIGS,
    "tft": TFT_CONFIGS,
}


def missing_modules(module_names):
    missing = []
    for name in module_names:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name} ({exc.__class__.__name__}: {exc})")
    return missing


def require_modules(module_names, feature_name):
    missing = missing_modules(module_names)
    if not missing:
        return

    missing_text = ", ".join(missing)
    raise SystemExit(
        f"{feature_name} cannot run because these Python modules are missing: {missing_text}\n"
        "Install the project dependencies with:\n"
        "  python -m pip install -r requirements.txt"
    )


def load_splits():
    df = pd.read_csv(FEATURES_PATH, index_col="datetime", parse_dates=True).sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    if len(df) != len(full_idx) or not df.index.equals(full_idx):
        df = df.reindex(full_idx).interpolate(method="time", limit_direction="both").dropna()
        df.index.name = "datetime"
    df.index.freq = pd.tseries.frequencies.to_offset("h")

    n = len(df)
    train = df.iloc[:int(n * 0.70)]
    val = df.iloc[int(n * 0.70):int(n * 0.85)]
    test = df.iloc[int(n * 0.85):]
    return train, val, test


def compute_metrics(y_true, y_pred, name, extra=None):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<25} MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  R2={r2:.4f}")
    result = {
        "model": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
    }
    if extra:
        result.update(extra)
    return result


def safe_name(name):
    return name.lower().replace(" ", "_").replace("-", "_").replace("+", "plus")


def plot_forecast(actuals, preds, name):
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actuals, label="Actual", color="steelblue", linewidth=1.2)
    ax.plot(preds, label=f"{name} Forecast", color="tomato", linewidth=1.2, linestyle="--")
    ax.set_title(f"{name} Day-Ahead Forecast vs Actual (7 days)", fontsize=13)
    ax.set_xlabel("Hours")
    ax.set_ylabel("kW")
    ax.legend()
    plt.tight_layout()
    fname = PLOTS_PATH / f"forecast_{safe_name(name)}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Plot saved: {fname.name}")


def append_experiment_run(result, config):
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        **result,
        "config": json.dumps(config, sort_keys=True, default=str),
    }
    runs = pd.DataFrame([row])
    if RUNS_PATH.exists():
        previous = pd.read_csv(RUNS_PATH)
        runs = pd.concat([previous, runs], ignore_index=True)
    runs.to_csv(RUNS_PATH, index=False)
    print(f"Run logged: {RUNS_PATH}")


def save_model_results(results):
    """Update neural metrics without dropping previous model runs."""
    new_results = pd.DataFrame(results)
    if METRICS_PATH.exists():
        previous = pd.read_csv(METRICS_PATH)
        if "model" in previous.columns:
            previous = previous[~previous["model"].isin(new_results["model"])]
        combined = pd.concat([previous, new_results], ignore_index=True)
    else:
        combined = new_results

    combined = combined.sort_values("MAE", ascending=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(METRICS_PATH, index=False)
    return combined


def copy_best_neural_model(results):
    if results.empty or "model_path" not in results.columns:
        return None

    best = results.sort_values("MAE", ascending=True).iloc[0]
    best_path = Path(best["model_path"])
    if not best_path.exists():
        print(f"Best neural model file not found: {best_path}")
        return None

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    destination = MODELS_PATH / "neural_best.pt"
    shutil.copy2(best_path, destination)
    print(f"Best neural model copied to: {destination} ({best['model']})")
    return destination


def copy_epoch_log(csv_logger, model_name):
    metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
    if not metrics_csv.exists():
        print(f"Epoch log not found yet: {metrics_csv}")
        return None

    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    epoch_log_path = RUNS_PATH.parent / f"epochs_{model_name}.csv"
    shutil.copy2(metrics_csv, epoch_log_path)
    print(f"Epoch log saved: {epoch_log_path}")
    return epoch_log_path


def run_chronos(train, test):
    require_modules(["chronos"], "Chronos")

    from chronos import BaseChronosPipeline

    print("\n--- Chronos-2 zero-shot, amazon/chronos-t5-small ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        dtype=torch.float32,
    )
    print(f"Running on: {device.upper()}")

    input_hours = 168
    context = torch.tensor(train[TARGET].values, dtype=torch.float32)
    test_vals = test[TARGET].values
    n_windows = len(test_vals) // FORECAST_HOURS
    preds, actuals = [], []

    for i in range(n_windows):
        if i % 10 == 0:
            print(f"  Chronos window {i + 1}/{n_windows}...")
        forecast = pipeline.predict(inputs=context, prediction_length=FORECAST_HOURS)
        median = np.quantile(forecast[0].detach().cpu().numpy(), 0.5, axis=0)
        start = i * FORECAST_HOURS
        end = start + FORECAST_HOURS
        preds.extend(median.tolist())
        actuals.extend(test_vals[start:end].tolist())
        new_ctx = torch.tensor(test_vals[start:end], dtype=torch.float32)
        context = torch.cat([context, new_ctx])[-input_hours:]

    preds = np.array(preds[:len(actuals)])
    actuals = np.array(actuals)

    result = compute_metrics(actuals, preds, "Chronos-2 zero-shot")
    plot_forecast(actuals[:FORECAST_HOURS * 7], preds[:FORECAST_HOURS * 7], "Chronos-2")
    return result


def make_series(df, cols):
    from darts import TimeSeries

    return TimeSeries.from_dataframe(df[cols], fill_missing_dates=True, freq="h")


def flatten_forecasts(forecasts_scaled, target_scaler, actual_series):
    y_pred, y_true = [], []
    for forecast_scaled in forecasts_scaled:
        forecast = target_scaler.inverse_transform(forecast_scaled)
        actual = actual_series.slice_intersect(forecast)
        n = min(len(actual), len(forecast))
        y_pred.extend(forecast.values().flatten()[:n].tolist())
        y_true.extend(actual.values().flatten()[:n].tolist())
    return np.array(y_true), np.array(y_pred)


def build_neural_forecast_df(forecasts_scaled, target_scaler, actual_series, model_name):
    rows = []
    for forecast_scaled in forecasts_scaled:
        forecast = target_scaler.inverse_transform(forecast_scaled)
        actual = actual_series.slice_intersect(forecast)
        n = min(len(actual), len(forecast))
        times = forecast.time_index[:n]
        issue_time = times[0]
        for i in range(n):
            rows.append({
                "datetime": times[i],
                "issue_time": issue_time,
                "horizon_hour": i + 1,
                "actual_kw": float(actual.values().flatten()[i]),
                "forecast_kw": float(forecast.values().flatten()[i]),
                "model": model_name,
            })
    df = pd.DataFrame(rows).set_index("datetime")
    df.index.name = "datetime"
    return df


def save_neural_forecasts(forecast_df, model_name):
    if LOAD_FORECASTS_PATH.exists():
        existing = pd.read_csv(LOAD_FORECASTS_PATH, index_col="datetime", parse_dates=True)
        if "model" in existing.columns:
            existing = existing[existing["model"] != model_name]
        combined = pd.concat([existing, forecast_df])
    else:
        combined = forecast_df
    combined.sort_index().to_csv(LOAD_FORECASTS_PATH)
    print(f"Forecasts saved: {LOAD_FORECASTS_PATH} (added {model_name})")


def run_darts_model(train, val, test, config):
    require_modules(["darts.models", "pytorch_lightning"], "Darts neural training")

    from darts.dataprocessing.transformers import Scaler
    from darts.models import NBEATSModel, NHiTSModel, TFTModel
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import CSVLogger
    from sklearn.preprocessing import RobustScaler, StandardScaler

    print(f"\n--- {config['name']} ---")

    model_type = config.get("model_type", "nhits")
    cov_source = config.get("covariates", PAST_COV_COLS)
    cov_cols = [col for col in cov_source if col in train.columns]
    train_series = make_series(train, [TARGET])
    val_series = make_series(val, [TARGET])
    test_series = make_series(test, [TARGET])

    use_past_covariates = model_type == "nhits" and bool(cov_cols)
    use_future_covariates = model_type == "tft" and bool(cov_cols)
    train_cov = make_series(train, cov_cols) if cov_cols else None
    val_cov = make_series(val, cov_cols) if cov_cols else None
    test_cov = make_series(test, cov_cols) if cov_cols else None

    scaler_name = config.get("scaler", "standard")
    if scaler_name == "minmax":
        target_scaler = Scaler()
        cov_scaler = Scaler()
    elif scaler_name == "robust":
        target_scaler = Scaler(RobustScaler())
        cov_scaler = Scaler(RobustScaler())
    else:
        target_scaler = Scaler(StandardScaler())
        cov_scaler = Scaler(StandardScaler())
    train_scaled = target_scaler.fit_transform(train_series)
    val_scaled = target_scaler.transform(val_series)
    test_scaled = target_scaler.transform(test_series)

    if cov_cols:
        train_cov = cov_scaler.fit_transform(train_cov)
        val_cov = cov_scaler.transform(val_cov)
        test_cov = cov_scaler.transform(test_cov)

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=12,
        min_delta=1e-4,
        mode="min",
    )

    torch.set_float32_matmul_precision("medium")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    loss_name = config.get("loss", "smooth_l1")
    if loss_name == "mae":
        loss_fn = torch.nn.L1Loss()
    elif loss_name == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.SmoothL1Loss(beta=config.get("beta", 0.25))

    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(save_dir=str(LOGS_PATH), name=config["model_name"])

    common_kwargs = {
        "input_chunk_length": config["input_chunk_length"],
        "output_chunk_length": FORECAST_HOURS,
        "dropout": config["dropout"],
        "n_epochs": config["n_epochs"],
        "batch_size": config["batch_size"],
        "loss_fn": loss_fn,
        "optimizer_cls": torch.optim.AdamW,
        "optimizer_kwargs": {"lr": config["lr"], "weight_decay": 1e-4},
        "lr_scheduler_cls": torch.optim.lr_scheduler.CosineAnnealingLR,
        "lr_scheduler_kwargs": {"T_max": config["n_epochs"]},
        "random_state": 42,
        "model_name": config["model_name"],
        "work_dir": str(DARTS_WORK_DIR),
        "save_checkpoints": True,
        "force_reset": True,
        "pl_trainer_kwargs": {
            "accelerator": accelerator,
            "devices": 1,
            "enable_progress_bar": bool(config.get("progress_bar", True)),
            "log_every_n_steps": config.get("log_every_n_steps", 10),
            "callbacks": [early_stopper],
            "logger": csv_logger,
        },
    }

    if model_type == "nbeats":
        model_cls = NBEATSModel
        model = model_cls(
            **common_kwargs,
            generic_architecture=True,
            num_stacks=config["num_stacks"],
            num_blocks=config["num_blocks"],
            num_layers=config["num_layers"],
            layer_widths=config["layer_widths"],
        )
    elif model_type == "tft":
        model_cls = TFTModel
        model = model_cls(
            **common_kwargs,
            hidden_size=config["hidden_size"],
            lstm_layers=config["lstm_layers"],
            num_attention_heads=config["num_attention_heads"],
            hidden_continuous_size=config["hidden_continuous_size"],
            add_relative_index=True,
        )
    else:
        model_cls = NHiTSModel
        model = model_cls(
            **common_kwargs,
            num_stacks=config["num_stacks"],
            num_blocks=config["num_blocks"],
            num_layers=config["num_layers"],
            layer_widths=config["layer_widths"],
        )

    print(
        "Training "
        f"{config['name']} ({config['input_chunk_length']}h context, "
        f"{config['n_epochs']} max epochs, {len(cov_cols)} covariates, "
        f"{loss_name} loss, {scaler_name} scaler) on {accelerator.upper()}..."
    )
    fit_kwargs = {"val_series": val_scaled, "verbose": bool(config.get("progress_bar", True))}
    if use_past_covariates:
        fit_kwargs["past_covariates"] = train_cov
        fit_kwargs["val_past_covariates"] = val_cov
    if use_future_covariates:
        fit_kwargs["future_covariates"] = train_cov
        fit_kwargs["val_future_covariates"] = val_cov
    model.fit(train_scaled, **fit_kwargs)
    epoch_log_path = copy_epoch_log(csv_logger, config["model_name"])

    model = model_cls.load_from_checkpoint(
        model_name=config["model_name"],
        work_dir=str(DARTS_WORK_DIR),
        best=True,
    )

    print("Evaluating with non-overlapping rolling 24h forecasts...")
    full_target = train_scaled.append(val_scaled).append(test_scaled)
    full_cov = train_cov.append(val_cov).append(test_cov) if cov_cols else None

    forecast_kwargs = {}
    if use_past_covariates:
        forecast_kwargs["past_covariates"] = full_cov
    if use_future_covariates:
        forecast_kwargs["future_covariates"] = full_cov

    forecasts_scaled = model.historical_forecasts(
        full_target,
        start=test_scaled.start_time(),
        forecast_horizon=FORECAST_HOURS,
        stride=FORECAST_HOURS,
        retrain=False,
        last_points_only=False,
        verbose=False,
        **forecast_kwargs,
    )

    y_true, y_pred = flatten_forecasts(forecasts_scaled, target_scaler, test_series)
    model_path = MODELS_PATH / f"{config['model_name']}.pt"
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    forecast_df = build_neural_forecast_df(forecasts_scaled, target_scaler, test_series, config["name"])
    save_neural_forecasts(forecast_df, config["name"])

    result = compute_metrics(
        y_true,
        y_pred,
        config["name"],
        extra={
            "input_hours": config["input_chunk_length"],
            "covered_hours": len(y_true),
            "model_path": str(model_path),
            "epoch_log_path": str(epoch_log_path) if epoch_log_path else "",
        },
    )
    append_experiment_run(result, config)
    plot_forecast(y_true[:FORECAST_HOURS * 7], y_pred[:FORECAST_HOURS * 7], config["name"])
    return result, model_path


def run_darts_eval(train, val, test, config):
    """Load an already-trained model and regenerate per-hour forecasts without retraining."""
    require_modules(["darts.models", "pytorch_lightning"], "Darts eval")

    from darts.dataprocessing.transformers import Scaler
    from darts.models import NBEATSModel, NHiTSModel, TFTModel
    from sklearn.preprocessing import StandardScaler

    checkpoint_dir = DARTS_WORK_DIR / config["model_name"] / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"No checkpoint found for {config['name']}. Run training first.")
        return None

    model_type = config.get("model_type", "nhits")
    cov_source = config.get("covariates", PAST_COV_COLS)
    cov_cols = [col for col in cov_source if col in train.columns]

    train_series = make_series(train, [TARGET])
    val_series = make_series(val, [TARGET])
    test_series = make_series(test, [TARGET])

    target_scaler = Scaler(StandardScaler())
    cov_scaler = Scaler(StandardScaler())
    train_scaled = target_scaler.fit_transform(train_series)
    val_scaled = target_scaler.transform(val_series)
    test_scaled = target_scaler.transform(test_series)

    train_cov = val_cov = test_cov = None
    if cov_cols:
        train_cov = cov_scaler.fit_transform(make_series(train, cov_cols))
        val_cov = cov_scaler.transform(make_series(val, cov_cols))
        test_cov = cov_scaler.transform(make_series(test, cov_cols))

    use_past_covariates = model_type == "nhits" and bool(cov_cols)
    use_future_covariates = model_type == "tft" and bool(cov_cols)

    model_cls = {"nhits": NHiTSModel, "nbeats": NBEATSModel, "tft": TFTModel}.get(model_type, NHiTSModel)
    print(f"\n--- Eval-only: {config['name']} ---")

    # Reconstruct model from config and load only the PyTorch weights from the checkpoint,
    # bypassing the _model.pth.tar which embeds numpy random state from training and may
    # be incompatible across numpy versions.
    loss_name = config.get("loss", "smooth_l1")
    if loss_name == "mae":
        loss_fn = torch.nn.L1Loss()
    elif loss_name == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.SmoothL1Loss(beta=config.get("beta", 0.25))

    torch.set_float32_matmul_precision("medium")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    ckpt_dir = checkpoint_dir
    best_ckpts = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not best_ckpts:
        best_ckpts = sorted(ckpt_dir.glob("last-*.ckpt"))
    if not best_ckpts:
        print(f"No checkpoint files in {ckpt_dir}")
        return None
    ckpt_path = best_ckpts[0]
    print(f"Loading weights from: {ckpt_path.name}")
    ckpt_state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Build a fresh model (n_epochs=1) to initialize the architecture,
    # then immediately overwrite with the saved checkpoint weights.
    init_kwargs = {
        "input_chunk_length": config["input_chunk_length"],
        "output_chunk_length": FORECAST_HOURS,
        "dropout": config["dropout"],
        "n_epochs": 1,
        "batch_size": config["batch_size"],
        "loss_fn": loss_fn,
        "random_state": 42,
        "model_name": config["model_name"] + "_eval",
        "work_dir": str(DARTS_WORK_DIR),
        "save_checkpoints": False,
        "force_reset": True,
        "pl_trainer_kwargs": {"accelerator": accelerator, "devices": 1, "enable_progress_bar": False, "max_epochs": 1},
    }
    if model_type == "nbeats":
        model = model_cls(
            **init_kwargs,
            generic_architecture=True,
            num_stacks=config["num_stacks"],
            num_blocks=config["num_blocks"],
            num_layers=config["num_layers"],
            layer_widths=config["layer_widths"],
        )
    elif model_type == "tft":
        model = model_cls(
            **init_kwargs,
            hidden_size=config["hidden_size"],
            lstm_layers=config["lstm_layers"],
            num_attention_heads=config["num_attention_heads"],
            hidden_continuous_size=config["hidden_continuous_size"],
            add_relative_index=True,
        )
    else:
        model = model_cls(
            **init_kwargs,
            num_stacks=config["num_stacks"],
            num_blocks=config["num_blocks"],
            num_layers=config["num_layers"],
            layer_widths=config["layer_widths"],
        )

    # Fit 1 epoch to build the architecture, then load saved weights
    print("Initializing architecture (1 epoch)...")
    fit_kwargs_init = {"verbose": False}
    if use_past_covariates and train_cov is not None:
        fit_kwargs_init["past_covariates"] = train_cov
    if use_future_covariates and train_cov is not None:
        fit_kwargs_init["future_covariates"] = train_cov
    model.fit(train_scaled, **fit_kwargs_init)

    # Overwrite with checkpoint weights (keys may or may not have a "model." prefix)
    raw_sd = ckpt_state["state_dict"]
    if any(k.startswith("model.") for k in raw_sd):
        raw_sd = {k.removeprefix("model."): v for k, v in raw_sd.items() if k.startswith("model.")}
    model.model.load_state_dict(raw_sd, strict=True)
    model.model.eval()
    print(f"Weights loaded from checkpoint, running inference...")

    full_target = train_scaled.append(val_scaled).append(test_scaled)
    full_cov = train_cov.append(val_cov).append(test_cov) if cov_cols else None

    forecast_kwargs = {}
    if use_past_covariates:
        forecast_kwargs["past_covariates"] = full_cov
    if use_future_covariates:
        forecast_kwargs["future_covariates"] = full_cov

    print("Running historical forecasts...")
    forecasts_scaled = model.historical_forecasts(
        full_target,
        start=test_scaled.start_time(),
        forecast_horizon=FORECAST_HOURS,
        stride=FORECAST_HOURS,
        retrain=False,
        last_points_only=False,
        verbose=False,
        **forecast_kwargs,
    )

    y_true, y_pred = flatten_forecasts(forecasts_scaled, target_scaler, test_series)
    forecast_df = build_neural_forecast_df(forecasts_scaled, target_scaler, test_series, config["name"])
    save_neural_forecasts(forecast_df, config["name"])

    result = compute_metrics(y_true, y_pred, config["name"])
    print(f"Saved forecasts for {config['name']}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-chronos", action="store_true", help="Also run Chronos-2 zero-shot.")
    parser.add_argument("--tune", action="store_true", help="Train all configured N-HiTS variants.")
    parser.add_argument(
        "--model",
        choices=["nhits", "nbeats", "tft", "all"],
        default="nhits",
        help="Which Darts neural model family to train.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load saved .pt files and regenerate per-hour forecasts only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    require_modules(["darts.models", "pytorch_lightning"], "N-HiTS training")
    if args.include_chronos:
        require_modules(["chronos"], "Chronos")

    print("Loading data...")
    train, val, test = load_splits()
    print(
        f"Train: {len(train)}h  |  Val: {len(val)}h  |  Test: {len(test)}h\n"
        f"Test range: {test.index.min()} to {test.index.max()}"
    )

    results = []
    if args.include_chronos:
        results.append(run_chronos(train, test))

    if args.model == "all":
        configs = [NHITS_CONFIGS[0]] + NBEATS_CONFIGS + TFT_CONFIGS
    elif args.model == "nhits" and args.tune:
        configs = NHITS_CONFIGS
    else:
        configs = MODEL_CONFIGS[args.model]

    if args.eval_only:
        for config in configs:
            result = run_darts_eval(train, val, test, config)
            if result:
                results.append(result)
        if results:
            print("\n=== EVAL-ONLY RESULTS ===")
            print(pd.DataFrame(results).to_string(index=False))
    else:
        best_result, best_model_path = None, None
        for config in configs:
            result, model_path = run_darts_model(train, val, test, config)
            results.append(result)
            if best_result is None or result["MAE"] < best_result["MAE"]:
                best_result = result
                best_model_path = model_path

        combined_results = save_model_results(results)
        copy_best_neural_model(combined_results)

        print("\n=== FINAL RESULTS ===")
        print(pd.DataFrame(results).to_string(index=False))
        print("\n=== SAVED NEURAL COMPARISON ===")
        print(combined_results.to_string(index=False))
        print(f"\nSaved: {METRICS_PATH}")
