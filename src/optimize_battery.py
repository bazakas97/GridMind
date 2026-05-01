"""
Battery dispatch optimization.

Given a day-ahead load forecast and a time-of-use electricity price profile,
simulate a greedy battery charge/discharge strategy to minimize daily cost.
Also provides an LP-optimal dispatch using forecasted dynamic prices.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, replace
from pathlib import Path
from scipy.optimize import linprog

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
PLOTS_PATH    = Path(__file__).parent.parent / "outputs" / "plots"
METRICS_PATH  = Path(__file__).parent.parent / "outputs" / "metrics"
LOAD_FORECASTS_PATH = METRICS_PATH / "load_forecasts.csv"
PRICE_FORECASTS_PATH = METRICS_PATH / "price_forecasts.csv"

@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 10.0
    max_charge_rate_kw: float = 2.5
    max_discharge_rate_kw: float = 2.5
    initial_soc: float = 0.2
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95


DEFAULT_BATTERY_CONFIG = BatteryConfig()

# Backwards-compatible constants for scripts/imports that only read defaults.
BATTERY_CAPACITY = DEFAULT_BATTERY_CONFIG.capacity_kwh
MAX_CHARGE_RATE = DEFAULT_BATTERY_CONFIG.max_charge_rate_kw
MAX_DISCHARGE_RATE = DEFAULT_BATTERY_CONFIG.max_discharge_rate_kw
INITIAL_SOC = DEFAULT_BATTERY_CONFIG.initial_soc
CHARGE_EFFICIENCY = DEFAULT_BATTERY_CONFIG.charge_efficiency
DISCHARGE_EFFICIENCY = DEFAULT_BATTERY_CONFIG.discharge_efficiency

# time-of-use electricity price (€/kWh) — typical Greek grid tariff
# hour → price
def price_profile(hour: int) -> float:
    if 22 <= hour or hour < 7:
        return 0.08   # off-peak (night)
    elif 7 <= hour < 17:
        return 0.18   # mid-peak (day)
    else:
        return 0.28   # peak (17:00–22:00)


def tariff_prices(n: int) -> np.ndarray:
    return np.array([price_profile(h % 24) for h in range(n)], dtype=float)


def cost_summary(load_kw: np.ndarray, grid_draw: np.ndarray, prices: np.ndarray) -> dict:
    cost_without = float(np.sum(load_kw * prices))
    cost_with = float(np.sum(grid_draw * prices))
    savings = cost_without - cost_with
    savings_pct = savings / cost_without * 100 if cost_without != 0 else 0.0
    return {
        "cost_without": cost_without,
        "cost_with": cost_with,
        "savings": savings,
        "savings_pct": savings_pct,
    }


def realize_schedule(actual_kw: np.ndarray,
                     planned_charge: np.ndarray,
                     config: BatteryConfig | None = None) -> dict:
    """Apply a planned schedule to actual load with no grid export."""
    config = config or DEFAULT_BATTERY_CONFIG
    actual_kw = np.asarray(actual_kw, dtype=float)
    planned_charge = np.asarray(planned_charge, dtype=float)
    n = len(actual_kw)
    soc = np.zeros(n + 1)
    soc[0] = config.capacity_kwh * config.initial_soc
    charge = np.zeros(n)
    grid_draw = np.zeros(n)

    for t in range(n):
        if planned_charge[t] >= 0:
            charge_grid = min(
                planned_charge[t],
                config.max_charge_rate_kw,
                (config.capacity_kwh - soc[t]) / config.charge_efficiency,
            )
            charge[t] = max(charge_grid, 0.0)
            soc[t + 1] = soc[t] + charge[t] * config.charge_efficiency
        else:
            requested_discharge = -planned_charge[t]
            discharge_to_load = min(
                requested_discharge,
                config.max_discharge_rate_kw,
                soc[t] * config.discharge_efficiency,
                max(actual_kw[t], 0.0),
            )
            charge[t] = -max(discharge_to_load, 0.0)
            soc[t + 1] = soc[t] - (-charge[t]) / config.discharge_efficiency

        grid_draw[t] = max(actual_kw[t] + charge[t], 0.0)

    return {
        "realized_charge": charge,
        "realized_soc": soc[:-1],
        "realized_final_soc": soc[-1],
        "realized_grid_draw": grid_draw,
    }


def add_realized_costs(result: dict,
                       actual_kw: np.ndarray | None,
                       actual_prices: np.ndarray | None = None,
                       config: BatteryConfig | None = None) -> dict:
    if actual_kw is None:
        return result

    actual_kw = np.asarray(actual_kw, dtype=float)
    prices = result["prices"] if actual_prices is None else np.asarray(actual_prices, dtype=float)
    realized = realize_schedule(actual_kw, result["charge"], config=config)
    realized_costs = cost_summary(actual_kw, realized["realized_grid_draw"], prices)
    result.update(realized)
    result.update({
        "realized_cost_without": realized_costs["cost_without"],
        "realized_cost_with": realized_costs["cost_with"],
        "realized_savings": realized_costs["savings"],
        "realized_savings_pct": realized_costs["savings_pct"],
    })
    return result


def simulate_battery(forecast_kw: np.ndarray,
                     actual_kw: np.ndarray | None = None,
                     actual_prices: np.ndarray | None = None,
                     config: BatteryConfig | None = None) -> dict:
    """
    Greedy strategy:
      - Charge during cheap hours if battery not full
      - Discharge during expensive hours if battery not empty
    """
    config = config or DEFAULT_BATTERY_CONFIG
    forecast_kw = np.asarray(forecast_kw, dtype=float)
    n = len(forecast_kw)
    prices      = tariff_prices(n)
    soc         = np.zeros(n + 1)   # state of charge (kWh)
    soc[0]      = config.capacity_kwh * config.initial_soc
    charge      = np.zeros(n)       # positive = charging from grid, negative = discharging to load
    grid_draw   = np.zeros(n)       # actual power drawn from grid

    for t in range(n):
        p = prices[t]
        load = forecast_kw[t]

        if p <= 0.08:
            # cheap hour → charge as much as possible
            can_charge = min(
                config.max_charge_rate_kw,
                (config.capacity_kwh - soc[t]) / config.charge_efficiency,
            )
            charge[t]  = max(can_charge, 0.0)
            soc[t + 1] = soc[t] + charge[t] * config.charge_efficiency
        elif p >= 0.28:
            # expensive hour → discharge as much as possible
            can_discharge = min(
                config.max_discharge_rate_kw,
                soc[t] * config.discharge_efficiency,
                load,
            )
            charge[t]     = -max(can_discharge, 0.0)
            soc[t + 1] = soc[t] - (-charge[t]) / config.discharge_efficiency
        else:
            charge[t] = 0.0
            soc[t + 1] = soc[t]

        grid_draw[t] = load + charge[t]
        grid_draw[t] = max(grid_draw[t], 0.0)  # can't push back to grid (no export)

    costs = cost_summary(forecast_kw, grid_draw, prices)

    result = {
        "prices":        prices,
        "soc":           soc[:-1],
        "charge":        charge,
        "grid_draw":     grid_draw,
        "battery_capacity_kwh": config.capacity_kwh,
        **costs,
    }
    return add_realized_costs(result, actual_kw, actual_prices, config=config)


def optimize_battery_lp(forecast_kw: np.ndarray,
                        forecast_prices: np.ndarray,
                        actual_kw: np.ndarray | None = None,
                        actual_prices: np.ndarray | None = None,
                        require_terminal_soc: bool = True,
                        config: BatteryConfig | None = None) -> dict:
    """
    LP-optimal battery dispatch given forecasted dynamic prices.

    Formulation (variables = charge_from_grid[t], discharge_to_load[t]):
      minimize   sum( (load[t] + charge[t] - discharge[t]) * price[t] )
      subject to:
        SOC stays between 0 and capacity after charge/discharge efficiencies
        optional final SOC equals initial SOC
        no grid export: discharge[t] <= load[t]
    """
    config = config or DEFAULT_BATTERY_CONFIG
    forecast_kw = np.asarray(forecast_kw, dtype=float)
    forecast_prices = np.asarray(forecast_prices, dtype=float)
    n        = len(forecast_kw)
    soc_init = config.capacity_kwh * config.initial_soc

    # Objective excludes the constant load * price term.
    c_obj = np.concatenate([forecast_prices, -forecast_prices])

    # SOC constraints via cumulative sum matrix (lower triangular ones)
    L = np.tril(np.ones((n, n)))
    soc_change = np.hstack([
        config.charge_efficiency * L,
        -(1.0 / config.discharge_efficiency) * L,
    ])
    A_ub = np.vstack([soc_change, -soc_change])
    b_ub = np.concatenate([
        np.full(n, config.capacity_kwh - soc_init),
        np.full(n, soc_init),
    ])

    A_eq = None
    b_eq = None
    if require_terminal_soc:
        A_eq = np.concatenate([
            np.full(n, config.charge_efficiency),
            np.full(n, -1.0 / config.discharge_efficiency),
        ]).reshape(1, -1)
        b_eq = np.array([0.0])

    charge_bounds = [(0.0, config.max_charge_rate_kw)] * n
    discharge_bounds = [
        (0.0, min(config.max_discharge_rate_kw, max(load, 0.0)))
        for load in forecast_kw
    ]
    bounds = charge_bounds + discharge_bounds

    res = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return {"success": False, "message": res.message}

    charge_from_grid = res.x[:n]
    discharge_to_load = res.x[n:]
    charge    = charge_from_grid - discharge_to_load
    soc       = np.zeros(n + 1)
    soc[0]    = soc_init
    for t in range(n):
        soc[t + 1] = (
            soc[t]
            + charge_from_grid[t] * config.charge_efficiency
            - discharge_to_load[t] / config.discharge_efficiency
        )
    grid_draw = np.maximum(0.0, forecast_kw + charge_from_grid - discharge_to_load)

    costs = cost_summary(forecast_kw, grid_draw, forecast_prices)

    result = {
        "success":      True,
        "prices":       forecast_prices,
        "soc":          soc[:-1],
        "final_soc":    soc[-1],
        "charge":       charge,
        "charge_from_grid": charge_from_grid,
        "discharge_to_load": discharge_to_load,
        "grid_draw":    grid_draw,
        "battery_capacity_kwh": config.capacity_kwh,
        **costs,
    }
    return add_realized_costs(result, actual_kw, actual_prices, config=config)


def optimize_battery_rolling_horizon(forecast_kw: np.ndarray,
                                     forecast_prices: np.ndarray,
                                     actual_kw: np.ndarray | None = None,
                                     actual_prices: np.ndarray | None = None,
                                     horizon_hours: int = 72,
                                     step_hours: int = 24,
                                     config: BatteryConfig | None = None) -> dict:
    """Run LP dispatch over a multi-day rolling horizon.

    Each optimization sees horizon_hours ahead, applies only step_hours, then
    rolls forward with the realized state of charge. This avoids treating each
    day as an isolated island.
    """
    config = config or DEFAULT_BATTERY_CONFIG
    forecast_kw = np.asarray(forecast_kw, dtype=float)
    forecast_prices = np.asarray(forecast_prices, dtype=float)
    actual_kw_arr = None if actual_kw is None else np.asarray(actual_kw, dtype=float)
    actual_prices_arr = None if actual_prices is None else np.asarray(actual_prices, dtype=float)

    n = len(forecast_kw)
    current_soc = config.capacity_kwh * config.initial_soc
    charges = []
    socs = []
    grid_draws = []
    realized_charges = []
    realized_socs = []
    realized_grid_draws = []

    for start in range(0, n, step_hours):
        apply_end = min(start + step_hours, n)
        optimize_end = min(start + horizon_hours, n)
        apply_len = apply_end - start
        if apply_len <= 0:
            break

        window_config = replace(
            config,
            initial_soc=current_soc / config.capacity_kwh if config.capacity_kwh else 0.0,
        )
        window = optimize_battery_lp(
            forecast_kw[start:optimize_end],
            forecast_prices[start:optimize_end],
            require_terminal_soc=False,
            config=window_config,
        )
        if not window.get("success", False):
            return {"success": False, "message": window.get("message", "rolling LP failed")}

        planned_charge = window["charge"][:apply_len]
        planned_soc = window["soc"][:apply_len]
        planned_grid = window["grid_draw"][:apply_len]

        actual_segment = (
            forecast_kw[start:apply_end]
            if actual_kw_arr is None
            else actual_kw_arr[start:apply_end]
        )
        realized = realize_schedule(actual_segment, planned_charge, config=window_config)
        current_soc = realized["realized_final_soc"]

        charges.append(planned_charge)
        socs.append(planned_soc)
        grid_draws.append(planned_grid)
        realized_charges.append(realized["realized_charge"])
        realized_socs.append(realized["realized_soc"])
        realized_grid_draws.append(realized["realized_grid_draw"])

    charge = np.concatenate(charges) if charges else np.array([])
    soc = np.concatenate(socs) if socs else np.array([])
    grid_draw = np.concatenate(grid_draws) if grid_draws else np.array([])
    realized_charge = np.concatenate(realized_charges) if realized_charges else np.array([])
    realized_soc = np.concatenate(realized_socs) if realized_socs else np.array([])
    realized_grid_draw = np.concatenate(realized_grid_draws) if realized_grid_draws else np.array([])

    planned_costs = cost_summary(forecast_kw[:len(grid_draw)], grid_draw, forecast_prices[:len(grid_draw)])
    result = {
        "success": True,
        "prices": forecast_prices[:len(grid_draw)],
        "soc": soc,
        "final_soc": current_soc,
        "charge": charge,
        "grid_draw": grid_draw,
        "realized_charge": realized_charge,
        "realized_soc": realized_soc,
        "realized_final_soc": current_soc,
        "realized_grid_draw": realized_grid_draw,
        "battery_capacity_kwh": config.capacity_kwh,
        **planned_costs,
    }

    if actual_kw_arr is not None:
        prices = (
            forecast_prices[:len(realized_grid_draw)]
            if actual_prices_arr is None
            else actual_prices_arr[:len(realized_grid_draw)]
        )
        realized_costs = cost_summary(
            actual_kw_arr[:len(realized_grid_draw)],
            realized_grid_draw,
            prices,
        )
        result.update({
            "realized_cost_without": realized_costs["cost_without"],
            "realized_cost_with": realized_costs["cost_with"],
            "realized_savings": realized_costs["savings"],
            "realized_savings_pct": realized_costs["savings_pct"],
        })

    return result


def plot_battery(forecast_kw, result, date_label="Sample Day", actual_kw=None):
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    hours = np.arange(len(forecast_kw))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # price profile background colours
    price_colors = {0.08: "#d4edda", 0.18: "#fff3cd", 0.28: "#f8d7da"}
    for ax in axes:
        for h in hours:
            p = float(result["prices"][h])
            ax.axvspan(h, h + 1, alpha=0.3, color=price_colors.get(p, "#eef2f3"), linewidth=0)

    # plot 1: load vs grid draw
    if actual_kw is not None:
        axes[0].plot(hours, actual_kw, label="Actual load", color="gray", linewidth=1.2, alpha=0.8)
    axes[0].plot(hours, forecast_kw, label="Forecast load", color="steelblue", linewidth=1.5)
    grid_draw = result.get("realized_grid_draw", result["grid_draw"])
    axes[0].plot(hours, grid_draw, label="Grid draw (with battery)", color="tomato", linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("kW")
    axes[0].set_title(f"Battery Dispatch Simulation — {date_label}", fontsize=13)
    axes[0].legend()

    # plot 2: charge / discharge
    pos = np.clip(result["charge"],  0, None)
    neg = np.clip(result["charge"], None, 0)
    axes[1].bar(hours, pos, color="green",  alpha=0.7, label="Charging")
    axes[1].bar(hours, neg, color="orange", alpha=0.7, label="Discharging")
    axes[1].set_ylabel("kW")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].legend()

    # plot 3: state of charge
    soc = result.get("realized_soc", result["soc"])
    axes[2].fill_between(hours, soc, alpha=0.4, color="steelblue")
    axes[2].plot(hours, soc, color="steelblue", linewidth=1.5, label="State of Charge")
    battery_capacity = result.get("battery_capacity_kwh", BATTERY_CAPACITY)
    axes[2].axhline(battery_capacity, color="gray", linestyle="--", linewidth=0.8, label=f"Capacity ({battery_capacity} kWh)")
    axes[2].set_ylabel("kWh")
    axes[2].set_xlabel("Hour of day")
    axes[2].legend()

    # legend for price bands
    patches = [
        mpatches.Patch(color="#d4edda", alpha=0.6, label="Off-peak (0.08 €/kWh)"),
        mpatches.Patch(color="#fff3cd", alpha=0.6, label="Mid-peak (0.18 €/kWh)"),
        mpatches.Patch(color="#f8d7da", alpha=0.6, label="Peak (0.28 €/kWh)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fname = PLOTS_PATH / "battery_dispatch.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {fname.name}")


def full_day_dates(df: pd.DataFrame) -> list:
    return [
        date for date, group in df.groupby(df.index.date)
        if len(group) == 24
    ]


def load_sample_day():
    if LOAD_FORECASTS_PATH.exists():
        forecasts = pd.read_csv(LOAD_FORECASTS_PATH, index_col="datetime", parse_dates=True)
        dates = full_day_dates(forecasts)
        price_forecasts = None
        if PRICE_FORECASTS_PATH.exists():
            price_forecasts = pd.read_csv(PRICE_FORECASTS_PATH, index_col="datetime", parse_dates=True)
            dates = sorted(set(dates) & set(full_day_dates(price_forecasts)))
        else:
            dates = sorted(dates)

        mondays = [d for d in dates if pd.Timestamp(d).dayofweek == 0]
        sample_day = (mondays or dates)[0]
        day_load = forecasts[forecasts.index.date == sample_day]
        day_prices = None if price_forecasts is None else price_forecasts[price_forecasts.index.date == sample_day]
        return sample_day, day_load, day_prices

    df = pd.read_csv(FEATURES_PATH, index_col="datetime", parse_dates=True)
    n = len(df)
    test = df.iloc[int(n * 0.85):]
    mondays = test[test.index.dayofweek == 0]
    sample_day = mondays.index[0].date()
    day_data = test[test.index.date == sample_day].rename(columns={"Global_active_power": "actual_kw"})
    day_data["forecast_kw"] = day_data["actual_kw"]
    return sample_day, day_data, None


def load_sample_window(start_day, days: int = 3):
    if not LOAD_FORECASTS_PATH.exists() or not PRICE_FORECASTS_PATH.exists():
        return None, None

    forecasts = pd.read_csv(LOAD_FORECASTS_PATH, index_col="datetime", parse_dates=True)
    prices = pd.read_csv(PRICE_FORECASTS_PATH, index_col="datetime", parse_dates=True)
    dates = sorted(set(full_day_dates(forecasts)) & set(full_day_dates(prices)))
    if start_day not in dates:
        return None, None

    start_idx = dates.index(start_day)
    window_dates = dates[start_idx:start_idx + days]
    if len(window_dates) < days:
        return None, None

    window_load = forecasts[np.isin(forecasts.index.date, window_dates)].sort_index()
    window_prices = prices[np.isin(prices.index.date, window_dates)].sort_index()
    if len(window_load) != days * 24 or len(window_prices) != days * 24:
        return None, None
    return window_load, window_prices


if __name__ == "__main__":
    sample_day, day_load, day_prices = load_sample_day()

    forecast_kw = day_load["forecast_kw"].values
    actual_kw = day_load["actual_kw"].values
    print(f"Simulating battery dispatch for: {sample_day}")
    print(f"Hours in day: {len(forecast_kw)}")

    lp_result = None
    actual_prices = None
    forecast_prices = None
    if day_prices is not None and len(day_prices) >= len(forecast_kw):
        forecast_prices = day_prices["price_forecast"].values[:len(forecast_kw)]
        actual_prices = day_prices["price_actual"].values[:len(forecast_kw)]

    result = simulate_battery(forecast_kw, actual_kw=actual_kw, actual_prices=actual_prices)

    if forecast_prices is not None and actual_prices is not None:
        lp_result = optimize_battery_lp(
            forecast_kw,
            forecast_prices,
            actual_kw=actual_kw,
            actual_prices=actual_prices,
        )

    rolling_result = None
    window_load, window_prices = load_sample_window(sample_day, days=3)
    if window_load is not None and window_prices is not None:
        rolling_result = optimize_battery_rolling_horizon(
            window_load["forecast_kw"].values,
            window_prices["price_forecast"].values,
            actual_kw=window_load["actual_kw"].values,
            actual_prices=window_prices["price_actual"].values,
            horizon_hours=72,
            step_hours=24,
        )

    print(f"\n--- Results ---")
    print(f"Forecast-planned cost without battery: {result['cost_without']:.4f} €")
    print(f"Forecast-planned cost with battery:    {result['cost_with']:.4f} €")
    print(f"Realized cost without battery:         {result['realized_cost_without']:.4f} €")
    print(f"Realized cost with battery:            {result['realized_cost_with']:.4f} €")
    print(f"Realized daily savings:                {result['realized_savings']:.4f} € ({result['realized_savings_pct']:.1f}%)")
    if lp_result and lp_result.get("success", False):
        print(f"LP realized cost with battery:         {lp_result['realized_cost_with']:.4f} €")
        print(f"LP realized daily savings:             {lp_result['realized_savings']:.4f} € ({lp_result['realized_savings_pct']:.1f}%)")
    if rolling_result and rolling_result.get("success", False):
        print(f"Rolling 3-day LP realized cost:        {rolling_result['realized_cost_with']:.4f} €")
        print(f"Rolling 3-day LP realized savings:     {rolling_result['realized_savings']:.4f} € ({rolling_result['realized_savings_pct']:.1f}%)")

    plot_battery(forecast_kw, result, date_label=str(sample_day), actual_kw=actual_kw)

    # save summary
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    row = {
        "date":           str(sample_day),
        "greedy_planned_cost_without_€": round(result["cost_without"], 4),
        "greedy_planned_cost_with_€":    round(result["cost_with"],    4),
        "greedy_realized_cost_without_€": round(result["realized_cost_without"], 4),
        "greedy_realized_cost_with_€":    round(result["realized_cost_with"],    4),
        "greedy_realized_savings_€":      round(result["realized_savings"],      4),
        "greedy_realized_savings_%":      round(result["realized_savings_pct"],  1),
        "battery_kWh":    BATTERY_CAPACITY,
        "max_rate_kW":    MAX_CHARGE_RATE,
        "charge_efficiency": CHARGE_EFFICIENCY,
        "discharge_efficiency": DISCHARGE_EFFICIENCY,
    }
    if lp_result and lp_result.get("success", False):
        row.update({
            "lp_planned_cost_without_€": round(lp_result["cost_without"], 4),
            "lp_planned_cost_with_€":    round(lp_result["cost_with"],    4),
            "lp_realized_cost_without_€": round(lp_result["realized_cost_without"], 4),
            "lp_realized_cost_with_€":    round(lp_result["realized_cost_with"],    4),
            "lp_realized_savings_€":      round(lp_result["realized_savings"],      4),
            "lp_realized_savings_%":      round(lp_result["realized_savings_pct"],  1),
        })
    if rolling_result and rolling_result.get("success", False):
        row.update({
            "rolling_3d_realized_cost_without_€": round(rolling_result["realized_cost_without"], 4),
            "rolling_3d_realized_cost_with_€":    round(rolling_result["realized_cost_with"],    4),
            "rolling_3d_realized_savings_€":      round(rolling_result["realized_savings"],      4),
            "rolling_3d_realized_savings_%":      round(rolling_result["realized_savings_pct"],  1),
        })
    summary = pd.DataFrame([row])
    summary.to_csv(METRICS_PATH / "battery_results.csv", index=False)
    print(f"Saved: battery_results.csv")
