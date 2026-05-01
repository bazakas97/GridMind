import sys
from datetime import timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.optimize_battery import (
    BATTERY_CAPACITY,
    BatteryConfig,
    optimize_battery_lp,
    optimize_battery_rolling_horizon,
    simulate_battery,
)


st.set_page_config(
    page_title="GridMind",
    page_icon="⚡",
    layout="wide",
)


METRICS_DIR = Path(__file__).parent.parent / "outputs" / "metrics"
DISPLAY_START_DATE = pd.Timestamp("2026-04-27").date()

COLORS = {
    "actual": "#2563eb",
    "forecast": "#ef4444",
    "grid": "#0f766e",
    "lp": "#7c3aed",
    "muted": "#64748b",
    "good": "#16a34a",
    "bad": "#dc2626",
}


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    .section-note {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics():
    frames = []
    for filename in ["baseline_results.csv", "model_results.csv"]:
        path = METRICS_DIR / filename
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "R2"])

    combined = pd.concat(frames, ignore_index=True)
    required_cols = {"model", "MAE", "RMSE", "R2"}
    if not required_cols.issubset(combined.columns):
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "R2"])

    return combined.sort_values("MAE").reset_index(drop=True)


@st.cache_data
def load_price_forecasts():
    path = METRICS_DIR / "price_forecasts.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    required_cols = {"price_actual", "price_forecast"}
    return df.sort_index() if required_cols.issubset(df.columns) else None


@st.cache_data
def load_load_forecasts():
    path = METRICS_DIR / "load_forecasts.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="datetime", parse_dates=True).sort_index()
    required_cols = {"actual_kw", "forecast_kw"}
    return df if required_cols.issubset(df.columns) else None


@st.cache_data
def load_price_metrics():
    path = METRICS_DIR / "price_model_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def full_day_dates(df):
    if df is None:
        return []
    return [date for date, group in df.groupby(df.index.date) if len(group) == 24]


def available_models(load_df, metrics_df):
    if load_df is None or "model" not in load_df.columns:
        return []
    models = load_df["model"].unique().tolist()
    all_metrics = pd.concat([metrics_df, load_price_metrics()]) if not metrics_df.empty else pd.DataFrame()
    if not all_metrics.empty and "model" in all_metrics.columns:
        mae_map = all_metrics.set_index("model")["MAE"].to_dict()
        models.sort(key=lambda m: mae_map.get(m, 999))
    else:
        models.sort()
    return models


def filter_by_model(df, model):
    if df is None or model is None or "model" not in df.columns:
        return df
    return df[df["model"] == model]


def date_lookup(dates):
    return {
        source_date: DISPLAY_START_DATE + timedelta(days=i)
        for i, source_date in enumerate(sorted(dates))
    }


def display_date(source_date, lookup):
    return lookup.get(source_date, source_date)


def shifted_index(index, source_date, lookup):
    new_date = pd.Timestamp(display_date(source_date, lookup))
    return pd.DatetimeIndex([new_date + pd.Timedelta(hours=int(ts.hour)) for ts in index])


def shifted_multi_index(index, lookup):
    return pd.DatetimeIndex([
        pd.Timestamp(display_date(ts.date(), lookup)) + pd.Timedelta(hours=int(ts.hour))
        for ts in index
    ])


def realized_costs(result):
    return {
        "without": result.get("realized_cost_without", result["cost_without"]),
        "with": result.get("realized_cost_with", result["cost_with"]),
        "savings": result.get("realized_savings", result["savings"]),
        "savings_pct": result.get("realized_savings_pct", result["savings_pct"]),
    }


def plot_layout(fig, height=440, y_title=None, title=None):
    fig.update_layout(
        height=height,
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if y_title:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="day", step="day", stepmode="backward"),
                dict(step="all", label="all"),
            ]
        ),
    )
    return fig


def forecast_figure(day_data, source_date, lookup):
    display_idx = shifted_index(day_data.index, source_date, lookup)
    actual = day_data["actual_kw"].astype(float)
    forecast = day_data["forecast_kw"].astype(float)
    error = forecast.values - actual.values

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=display_idx,
            y=actual,
            name="Actual load",
            mode="lines+markers",
            line=dict(color=COLORS["actual"], width=2.5),
            hovertemplate="%{x|%H:%M}<br>Actual %{y:.3f} kW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display_idx,
            y=forecast,
            name="Forecast",
            mode="lines+markers",
            line=dict(color=COLORS["forecast"], width=2.5, dash="dash"),
            hovertemplate="%{x|%H:%M}<br>Forecast %{y:.3f} kW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=display_idx,
            y=error,
            name="Forecast error",
            marker_color=np.where(error >= 0, "rgba(239,68,68,0.35)", "rgba(37,99,235,0.28)"),
            yaxis="y2",
            hovertemplate="%{x|%H:%M}<br>Error %{y:.3f} kW<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis2=dict(
            title="Error",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
        )
    )
    return plot_layout(fig, title=f"Interactive Load Forecast - {display_date(source_date, lookup)}")


def price_figure(display_idx, actual_prices, forecast_prices, source_date, lookup):
    error = forecast_prices - actual_prices
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=display_idx,
            y=actual_prices,
            name="Actual price",
            mode="lines+markers",
            line=dict(color=COLORS["actual"], width=2.4),
            hovertemplate="%{x|%H:%M}<br>Actual %{y:.4f} EUR/kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display_idx,
            y=forecast_prices,
            name="Forecast price",
            mode="lines+markers",
            line=dict(color=COLORS["forecast"], width=2.4, dash="dash"),
            hovertemplate="%{x|%H:%M}<br>Forecast %{y:.4f} EUR/kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=display_idx,
            y=error,
            name="Price error",
            marker_color="rgba(100,116,139,0.28)",
            yaxis="y2",
            hovertemplate="%{x|%H:%M}<br>Error %{y:.4f} EUR/kWh<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis2=dict(title="Error", overlaying="y", side="right", showgrid=False),
    )
    return plot_layout(
        fig,
        height=360,
        y_title="EUR/kWh",
        title=f"Interactive Price Forecast - {display_date(source_date, lookup)}",
    )


def dispatch_figure(display_idx, actual_kw, forecast_kw, greedy, lp_res=None):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Load and Grid Draw", "Charge / Discharge", "State of Charge"),
    )

    fig.add_trace(go.Scatter(x=display_idx, y=actual_kw, name="Actual load", line=dict(color=COLORS["actual"], width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_idx, y=forecast_kw, name="Forecast load", line=dict(color=COLORS["muted"], width=1.8, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_idx, y=greedy["realized_grid_draw"], name="Grid draw - Greedy", line=dict(color=COLORS["grid"], width=2.2)), row=1, col=1)

    fig.add_trace(go.Bar(x=display_idx, y=np.clip(greedy["charge"], 0, None), name="Greedy charge", marker_color="rgba(15,118,110,0.65)"), row=2, col=1)
    fig.add_trace(go.Bar(x=display_idx, y=np.clip(greedy["charge"], None, 0), name="Greedy discharge", marker_color="rgba(15,118,110,0.35)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=display_idx, y=greedy["realized_soc"], name="SOC - Greedy", line=dict(color=COLORS["grid"], width=2.2)), row=3, col=1)

    if lp_res is not None:
        fig.add_trace(go.Scatter(x=display_idx, y=lp_res["realized_grid_draw"], name="Grid draw - Smart LP", line=dict(color=COLORS["lp"], width=2.2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Bar(x=display_idx, y=np.clip(lp_res["charge"], 0, None), name="LP charge", marker_color="rgba(124,58,237,0.62)"), row=2, col=1)
        fig.add_trace(go.Bar(x=display_idx, y=np.clip(lp_res["charge"], None, 0), name="LP discharge", marker_color="rgba(124,58,237,0.34)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=display_idx, y=lp_res["realized_soc"], name="SOC - Smart LP", line=dict(color=COLORS["lp"], width=2.2, dash="dash")), row=3, col=1)

    fig.update_layout(
        height=760,
        template="plotly_white",
        hovermode="x unified",
        barmode="relative",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider=dict(visible=True), row=3, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="kW", zeroline=True, row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    return fig


def model_comparison_figure(metrics):
    plot_df = metrics.copy()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("MAE - lower is better", "R2 - higher is better"))
    fig.add_trace(
        go.Bar(
            y=plot_df["model"],
            x=plot_df["MAE"],
            orientation="h",
            name="MAE",
            marker_color=COLORS["actual"],
            hovertemplate="%{y}<br>MAE %{x:.4f} kW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=plot_df["model"],
            x=plot_df["R2"],
            orientation="h",
            name="R2",
            marker_color=np.where(plot_df["R2"] >= 0, COLORS["good"], COLORS["bad"]),
            hovertemplate="%{y}<br>R2 %{x:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=430,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


metrics = load_metrics()
price_metrics = load_price_metrics()
price_forecasts = load_price_forecasts()
load_forecasts = load_load_forecasts()

st.sidebar.title("GridMind")
st.sidebar.caption("Interactive load, price, and battery dispatch lab")
st.sidebar.markdown("---")
tab_choice = st.sidebar.radio(
    "View",
    ["Forecast Lab", "Battery Dispatch", "Model Health"],
)
st.sidebar.markdown("---")

_model_options = available_models(load_forecasts, metrics)
if _model_options:
    selected_model = st.sidebar.selectbox(
        "Load forecast model",
        _model_options,
        index=0,
        help="Switch model for Forecast Lab and Battery Dispatch.",
    )
else:
    selected_model = None

filtered_load = filter_by_model(load_forecasts, selected_model)

st.sidebar.markdown("---")
st.sidebar.caption("Dates are shown on a 2026 simulation calendar. Source data remains unchanged.")


if tab_choice == "Forecast Lab":
    st.title("Forecast Lab")
    st.markdown('<div class="section-note">Zoom, pan, hover, and inspect the hours where the model misses.</div>', unsafe_allow_html=True)

    available_dates = full_day_dates(filtered_load)
    if filtered_load is None or not available_dates:
        st.error("Load forecast artifact not found. Run `python src/train_baseline.py` first.")
        st.stop()

    lookup = date_lookup(available_dates)
    selected_date = st.selectbox(
        "Simulation day",
        available_dates,
        format_func=lambda d: str(display_date(d, lookup)),
        key="forecast_date",
    )

    day_data = filtered_load[filtered_load.index.date == selected_date].copy()
    actual = day_data["actual_kw"].astype(float)
    pred = day_data["forecast_kw"].astype(float)
    abs_err = (actual - pred).abs()
    signed_err = pred - actual
    peak_error_hour = abs_err.idxmax()
    bias = float(signed_err.mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily MAE", f"{abs_err.mean():.3f} kW")
    c2.metric("Peak actual load", f"{actual.max():.2f} kW")
    c3.metric("Worst miss", f"{abs_err.max():.3f} kW", delta=peak_error_hour.strftime("%H:%M"))
    c4.metric("Mean bias", f"{bias:+.3f} kW")

    st.plotly_chart(forecast_figure(day_data, selected_date, lookup), use_container_width=True)

    insight_df = pd.DataFrame(
        {
            "time": shifted_index(day_data.index, selected_date, lookup).strftime("%H:%M"),
            "actual_kw": actual.values,
            "forecast_kw": pred.values,
            "error_kw": signed_err.values,
            "abs_error_kw": abs_err.values,
        }
    ).sort_values("abs_error_kw", ascending=False)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Highest-error hours")
        st.dataframe(
            insight_df.head(8).style.format(
                {"actual_kw": "{:.3f}", "forecast_kw": "{:.3f}", "error_kw": "{:+.3f}", "abs_error_kw": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Error distribution")
        hist = px.histogram(
            insight_df,
            x="error_kw",
            nbins=16,
            labels={"error_kw": "Forecast error (kW)"},
            color_discrete_sequence=[COLORS["muted"]],
        )
        hist.update_layout(template="plotly_white", height=290, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(hist, use_container_width=True)

    with st.expander("Source timestamp mapping"):
        st.write(f"Displayed day: `{display_date(selected_date, lookup)}`")
        st.write(f"Original source day: `{selected_date}`")


elif tab_choice == "Battery Dispatch":
    st.title("Battery Dispatch")
    st.markdown('<div class="section-note">Tune the battery, compare strategies, and inspect every charge/discharge decision.</div>', unsafe_allow_html=True)

    available_dates = full_day_dates(filtered_load)
    if price_forecasts is not None:
        common_dates = sorted(set(available_dates) & set(full_day_dates(price_forecasts)))
        if common_dates:
            available_dates = common_dates

    if filtered_load is None or not available_dates:
        st.error("Load forecast artifact not found. Run `python src/train_baseline.py` first.")
        st.stop()

    lookup = date_lookup(available_dates)

    control_cols = st.columns([1.2, 1, 1, 1])
    selected_date = control_cols[0].selectbox(
        "Simulation day",
        available_dates,
        format_func=lambda d: str(display_date(d, lookup)),
        key="battery_date",
    )
    battery_cap = control_cols[1].slider("Capacity (kWh)", 5.0, 20.0, BATTERY_CAPACITY, 0.5)
    max_rate = control_cols[2].slider("Max rate (kW)", 1.0, 5.0, 2.5, 0.5)
    initial_soc = control_cols[3].slider("Initial charge", 0, 100, 20, 5) / 100
    optimization_mode = st.radio(
        "Smart optimizer",
        ["Single-day LP", "Rolling 3-day LP"],
        horizontal=True,
    )

    selected_pos = available_dates.index(selected_date)
    if optimization_mode == "Rolling 3-day LP":
        selected_dates = available_dates[selected_pos:selected_pos + 3]
    else:
        selected_dates = [selected_date]

    day_data = filtered_load[np.isin(filtered_load.index.date, selected_dates)].copy()
    display_idx = shifted_multi_index(day_data.index, lookup)

    battery_config = BatteryConfig(
        capacity_kwh=battery_cap,
        max_charge_rate_kw=max_rate,
        max_discharge_rate_kw=max_rate,
        initial_soc=initial_soc,
    )

    forecast_kw = day_data["forecast_kw"].values.astype(float)
    actual_kw = day_data["actual_kw"].values.astype(float)
    dispatch_mae = float(np.mean(np.abs(actual_kw - forecast_kw)))

    fp = None
    ap = None
    if price_forecasts is not None:
        day_prices = price_forecasts[np.isin(price_forecasts.index.date, selected_dates)]
        if len(day_prices) >= len(forecast_kw):
            fp = day_prices["price_forecast"].values[: len(forecast_kw)].astype(float)
            ap = day_prices["price_actual"].values[: len(forecast_kw)].astype(float)

    greedy = simulate_battery(
        forecast_kw,
        actual_kw=actual_kw,
        actual_prices=ap,
        config=battery_config,
    )
    lp_res = None
    if fp is not None and ap is not None:
        if optimization_mode == "Rolling 3-day LP":
            candidate = optimize_battery_rolling_horizon(
                forecast_kw,
                fp,
                actual_kw=actual_kw,
                actual_prices=ap,
                horizon_hours=min(72, len(forecast_kw)),
                step_hours=24,
                config=battery_config,
            )
        else:
            candidate = optimize_battery_lp(
                forecast_kw,
                fp,
                actual_kw=actual_kw,
                actual_prices=ap,
                config=battery_config,
            )
        if candidate.get("success", False):
            lp_res = candidate

    g_costs = realized_costs(greedy)
    l_costs = realized_costs(lp_res) if lp_res else None

    st.subheader("Forecast quality for this dispatch window")
    q1, q2, q3 = st.columns(3)
    q1.metric("Window forecast MAE", f"{dispatch_mae:.3f} kW")
    q2.metric("Hours simulated", str(len(forecast_kw)))
    q3.metric("Optimization mode", optimization_mode)

    st.subheader("Battery strategy outcome")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("No battery cost", f"{g_costs['without']:.3f} EUR")
    c2.metric("Greedy savings", f"{g_costs['savings']:.3f} EUR", delta=f"{g_costs['savings_pct']:.1f}%")
    if l_costs:
        c3.metric("Smart LP savings", f"{l_costs['savings']:.3f} EUR", delta=f"{l_costs['savings_pct']:.1f}%")
        c4.metric("LP vs Greedy", f"{l_costs['savings'] - g_costs['savings']:+.3f} EUR")
    else:
        c3.metric("Smart LP", "No price data")
        c4.metric("LP vs Greedy", "-")

    if fp is not None and ap is not None:
        st.caption(
            "Price data is an OPSD 2018-2020 market series mapped onto the household load calendar. "
            "Use the price metrics as scenario-quality indicators, not as a true historical backtest."
        )
        st.plotly_chart(price_figure(display_idx, ap, fp, selected_date, lookup), use_container_width=True)

    st.plotly_chart(dispatch_figure(display_idx, actual_kw, forecast_kw, greedy, lp_res), use_container_width=True)

    dispatch_rows = {
        "time": display_idx.strftime("%Y-%m-%d %H:%M"),
        "actual_load_kw": actual_kw,
        "forecast_load_kw": forecast_kw,
        "greedy_charge_kw": greedy["charge"],
        "greedy_grid_kw": greedy["realized_grid_draw"],
        "greedy_soc_kwh": greedy["realized_soc"],
    }
    if lp_res:
        dispatch_rows.update(
            {
                "lp_charge_kw": lp_res["charge"],
                "lp_grid_kw": lp_res["realized_grid_draw"],
                "lp_soc_kwh": lp_res["realized_soc"],
            }
        )
    st.subheader("Dispatch table")
    st.dataframe(pd.DataFrame(dispatch_rows).style.format("{:.3f}", subset=list(dispatch_rows.keys())[1:]), use_container_width=True, hide_index=True)


elif tab_choice == "Model Health":
    st.title("Model Health")
    st.markdown('<div class="section-note">The scores are shown directly. The weaker models stay visible instead of being hidden.</div>', unsafe_allow_html=True)

    if metrics.empty:
        st.error("Model metrics not found. Run `python src/train_baseline.py` first.")
        st.stop()

    best = metrics.iloc[0]
    worst = metrics.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", best["model"])
    c2.metric("Best MAE", f"{best['MAE']:.4f} kW")
    c3.metric("Best R2", f"{best['R2']:.4f}")
    c4.metric("Worst MAE", f"{worst['MAE']:.4f} kW")

    st.plotly_chart(model_comparison_figure(metrics), use_container_width=True)

    st.subheader("Load model metrics")
    st.dataframe(
        metrics[["model", "MAE", "RMSE", "R2"]].style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    if not price_metrics.empty:
        st.subheader("Price model metrics")
        price_cols = ["model", "MAE", "RMSE", "R2"]
        if "nMAE" in price_metrics.columns:
            price_cols.append("nMAE")
        st.dataframe(
            price_metrics[price_cols].style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}", "nMAE": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These price scores use seasonally mapped OPSD prices, so they measure this synthetic scenario, "
            "not a strict same-calendar historical market forecast."
        )
