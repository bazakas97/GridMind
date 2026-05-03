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
    RL_MODEL_PATH,
    optimize_battery_lp,
    optimize_battery_mpc,
    optimize_battery_rolling_horizon,
    optimize_battery_stochastic,
    optimize_battery_rl,
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
    "actual": "#0ea5e9",
    "forecast": "#f97316",
    "grid": "#0ea5e9",
    "lp": "#8b5cf6",
    "muted": "#64748b",
    "good": "#10b981",
    "bad": "#ef4444",
    "bg": "#f0f4f8",
    "bg_secondary": "#ffffff",
    "grid_line": "rgba(0,0,0,0.06)",
}

DARK_LAYOUT = dict(
    paper_bgcolor="#f0f4f8",
    plot_bgcolor="#ffffff",
    font=dict(color="#1e293b", family="Inter, sans-serif", size=13),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.06)",
        zerolinecolor="rgba(0,0,0,0.12)",
        linecolor="rgba(0,0,0,0.12)",
        tickfont=dict(color="#64748b"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.06)",
        zerolinecolor="rgba(0,0,0,0.12)",
        linecolor="rgba(0,0,0,0.12)",
        tickfont=dict(color="#64748b"),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.08)",
        borderwidth=1,
        font=dict(color="#1e293b", size=12),
    ),
)


st.markdown(
    """
    <style>
    /* ── Light mode ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f0f4f8 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid rgba(0,0,0,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] caption {
        color: #64748b !important;
    }
    /* Main content */
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1500px;
        background-color: #f0f4f8 !important;
    }
    h1, h2, h3, h4 {
        color: #0f172a !important;
        letter-spacing: -0.01em;
    }
    p, li, label, .stMarkdown {
        color: #334155 !important;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
        color: #0ea5e9 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.9rem; }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 10px;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    /* Selectbox / dropdowns */
    [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
        border-color: rgba(0,0,0,0.12) !important;
        color: #1e293b !important;
    }
    /* Sliders */
    [data-testid="stSlider"] label { color: #64748b !important; }
    /* Dataframes */
    [data-testid="stDataFrame"] { background: #ffffff !important; }
    /* Section note */
    .section-note {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }
    /* Expander */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border-color: rgba(0,0,0,0.08) !important;
    }
    /* Dividers */
    hr { border-color: rgba(0,0,0,0.07) !important; }
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
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")) if title else None,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60 if title else 24, b=20),
        **DARK_LAYOUT,
    )
    if y_title:
        fig.update_yaxes(title_text=y_title, title_font=dict(color="#94a3b8"))
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor="#1a1f2e", bordercolor="rgba(255,255,255,0.1)"),
        rangeselector=dict(
            bgcolor="#1a1f2e",
            activecolor="#00d4aa",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#94a3b8"),
            buttons=[
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="day", step="day", stepmode="backward"),
                dict(step="all", label="all"),
            ],
        ),
    )
    return fig


def forecast_figure(day_data, source_date, lookup):
    display_idx = shifted_index(day_data.index, source_date, lookup)
    actual = day_data["actual_kw"].astype(float)
    forecast = day_data["forecast_kw"].astype(float)
    upper = np.maximum(actual.values, forecast.values)
    lower = np.minimum(actual.values, forecast.values)

    fig = go.Figure()

    # Shaded error band between actual and forecast
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([display_idx, display_idx[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255,107,107,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Error band",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    # Actual load
    fig.add_trace(
        go.Scatter(
            x=display_idx,
            y=actual,
            name="Actual load",
            mode="lines+markers",
            line=dict(color=COLORS["actual"], width=2.5),
            marker=dict(size=4, color=COLORS["actual"]),
            hovertemplate="%{x|%H:%M}  Actual: <b>%{y:.3f} kW</b><extra></extra>",
        )
    )
    # Forecast
    fig.add_trace(
        go.Scatter(
            x=display_idx,
            y=forecast,
            name="Forecast",
            mode="lines+markers",
            line=dict(color=COLORS["forecast"], width=2, dash="dot"),
            marker=dict(size=4, color=COLORS["forecast"]),
            hovertemplate="%{x|%H:%M}  Forecast: <b>%{y:.3f} kW</b><extra></extra>",
        )
    )

    return plot_layout(fig, title=f"Load Forecast — {display_date(source_date, lookup)}")


def price_figure(display_idx, actual_prices, forecast_prices, source_date, lookup):
    error = forecast_prices - actual_prices
    fig = go.Figure()
    upper = np.maximum(actual_prices, forecast_prices)
    lower = np.minimum(actual_prices, forecast_prices)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([display_idx, display_idx[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255,107,107,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Error band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display_idx,
            y=actual_prices,
            name="Actual price",
            mode="lines+markers",
            line=dict(color=COLORS["actual"], width=2.4),
            marker=dict(size=4),
            hovertemplate="%{x|%H:%M}  Actual: <b>%{y:.4f} EUR/kWh</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display_idx,
            y=forecast_prices,
            name="Forecast price",
            mode="lines+markers",
            line=dict(color=COLORS["forecast"], width=2, dash="dot"),
            marker=dict(size=4),
            hovertemplate="%{x|%H:%M}  Forecast: <b>%{y:.4f} EUR/kWh</b><extra></extra>",
        )
    )
    return plot_layout(
        fig,
        height=360,
        y_title="EUR/kWh",
        title=f"Price Forecast — {display_date(source_date, lookup)}",
    )


def _rgba(hex_color, alpha=0.6):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def dispatch_figure(display_idx, actual_kw, forecast_kw, greedy, lp_res=None, actual_prices=None, forecast_prices=None):
    has_price = actual_prices is not None and forecast_prices is not None
    n_rows = 4 if has_price else 3
    titles = (
        ("Electricity Price", "Load and Grid Draw", "Charge / Discharge", "State of Charge")
        if has_price else
        ("Load and Grid Draw", "Charge / Discharge", "State of Charge")
    )
    row_heights = [0.18, 0.32, 0.25, 0.25] if has_price else [0.38, 0.32, 0.30]
    load_row   = 2 if has_price else 1
    charge_row = 3 if has_price else 2
    soc_row    = 4 if has_price else 3

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=titles,
        row_heights=row_heights,
    )

    C = {
        "actual":           "#0ea5e9",
        "forecast":         "#94a3b8",
        "price_actual":     "#f97316",
        "price_forecast":   "#94a3b8",
        "greedy_grid":      "#f97316",
        "greedy_charge":    "#10b981",
        "greedy_discharge": "#ef4444",
        "greedy_soc":       "#0ea5e9",
        "lp_grid":          "#8b5cf6",
        "lp_charge":        "#06b6d4",
        "lp_discharge":     "#f43f5e",
        "lp_soc":           "#6366f1",
    }

    if has_price:
        upper_p = np.maximum(actual_prices, forecast_prices)
        lower_p = np.minimum(actual_prices, forecast_prices)
        fig.add_trace(go.Scatter(
            x=np.concatenate([display_idx, display_idx[::-1]]),
            y=np.concatenate([upper_p, lower_p[::-1]]),
            fill="toself", fillcolor=_rgba(C["price_actual"], 0.10),
            line=dict(color="rgba(0,0,0,0)"), name="Price error", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=display_idx, y=actual_prices,   name="Actual price",   line=dict(color=C["price_actual"],   width=2), mode="lines+markers", marker=dict(size=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=display_idx, y=forecast_prices, name="Forecast price", line=dict(color=C["price_forecast"], width=1.4, dash="dot"), mode="lines+markers", marker=dict(size=3)), row=1, col=1)

    # Grid draw traces first so they appear at the top of the legend
    fig.add_trace(go.Scatter(x=display_idx, y=greedy["realized_grid_draw"], name="Grid draw — Greedy",  line=dict(color=C["greedy_grid"], width=2.2)), row=load_row, col=1)
    if lp_res is not None:
        fig.add_trace(go.Scatter(x=display_idx, y=lp_res["realized_grid_draw"], name="Grid draw — Smart LP", line=dict(color=C["lp_grid"], width=2.2, dash="dash")), row=load_row, col=1)
    fig.add_trace(go.Scatter(x=display_idx, y=actual_kw,   name="Actual load",   line=dict(color=C["actual"],   width=2.2)), row=load_row, col=1)
    fig.add_trace(go.Scatter(x=display_idx, y=forecast_kw, name="Forecast load", line=dict(color=C["forecast"], width=1.4, dash="dot")), row=load_row, col=1)

    fig.add_trace(go.Bar(x=display_idx, y=np.clip(greedy["charge"], 0, None),  name="Greedy charge",    marker_color=_rgba(C["greedy_charge"])),    row=charge_row, col=1)
    fig.add_trace(go.Bar(x=display_idx, y=np.clip(greedy["charge"], None, 0),  name="Greedy discharge", marker_color=_rgba(C["greedy_discharge"])), row=charge_row, col=1)
    fig.add_trace(go.Scatter(x=display_idx, y=greedy["realized_soc"], name="SOC — Greedy",
                             line=dict(color=C["greedy_soc"], width=2.2), fill="tozeroy", fillcolor=_rgba(C["greedy_soc"], 0.08)), row=soc_row, col=1)

    if lp_res is not None:
        fig.add_trace(go.Bar(x=display_idx, y=np.clip(lp_res["charge"], 0, None),  name="LP charge",    marker_color=_rgba(C["lp_charge"])),    row=charge_row, col=1)
        fig.add_trace(go.Bar(x=display_idx, y=np.clip(lp_res["charge"], None, 0),  name="LP discharge", marker_color=_rgba(C["lp_discharge"])), row=charge_row, col=1)
        fig.add_trace(go.Scatter(x=display_idx, y=lp_res["realized_soc"], name="SOC — Smart LP",
                                 line=dict(color=C["lp_soc"], width=2.2, dash="dash"), fill="tozeroy", fillcolor=_rgba(C["lp_soc"], 0.08)), row=soc_row, col=1)

    fig.update_layout(
        height=920 if has_price else 780,
        hovermode="x unified",
        barmode="relative",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            orientation="h",
            x=0, y=1.02,
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            font=dict(color="#1e293b", size=11),
        ),
        **{k: v for k, v in DARK_LAYOUT.items() if k != "legend"},
    )
    for i in range(1, n_rows + 1):
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.06)", linecolor="rgba(0,0,0,0.1)", tickfont=dict(color="#64748b"), row=i, col=1)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)", linecolor="rgba(0,0,0,0.1)", tickfont=dict(color="#64748b"), row=i, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, bgcolor="#e2e8f0"), row=n_rows, col=1)
    if has_price:
        fig.update_yaxes(title_text="€/kWh", title_font=dict(color="#64748b"), row=1, col=1)
    fig.update_yaxes(title_text="kW",   title_font=dict(color="#64748b"), row=load_row, col=1)
    fig.update_yaxes(title_text="kW",   title_font=dict(color="#64748b"), zeroline=True, zerolinecolor="rgba(0,0,0,0.12)", row=charge_row, col=1)
    fig.update_yaxes(title_text="kWh",  title_font=dict(color="#64748b"), row=soc_row, col=1)
    return fig


def model_comparison_figure(metrics):
    plot_df = metrics.copy()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("MAE — lower is better", "R² — higher is better"),
    )
    fig.add_trace(
        go.Bar(
            y=plot_df["model"],
            x=plot_df["MAE"],
            orientation="h",
            name="MAE",
            marker=dict(
                color=plot_df["MAE"],
                colorscale=[[0, "#00d4aa"], [1, "#1a1f2e"]],
                showscale=False,
            ),
            hovertemplate="%{y}<br>MAE <b>%{x:.4f}</b> kW<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            y=plot_df["model"],
            x=plot_df["R2"],
            orientation="h",
            name="R²",
            marker=dict(
                color=plot_df["R2"],
                colorscale=[[0, "#f87171"], [0.5, "#1a1f2e"], [1, "#00d4aa"]],
                showscale=False,
            ),
            hovertemplate="%{y}<br>R² <b>%{x:.4f}</b><extra></extra>",
        ),
        row=1, col=2,
    )
    fig.update_layout(
        height=440,
        showlegend=False,
        margin=dict(l=20, r=20, t=45, b=20),
        **DARK_LAYOUT,
    )
    for i in range(1, 3):
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8"), row=1, col=i)
        fig.update_yaxes(tickfont=dict(color="#e2e8f0"), row=1, col=i)
    fig.update_yaxes(autorange="reversed")
    fig.update_annotations(font=dict(color="#94a3b8", size=12))
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
    ["Home", "Forecast Lab", "Battery Dispatch", "Model Health"],
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


if tab_choice == "Home":
    SLIDES = [
        {
            "icon": "⚡",
            "title": "Welcome to GridMind",
            "subtitle": "AI-powered energy forecasting & smart battery optimization",
            "body": """
GridMind combines **machine learning forecasting** with **intelligent battery control** to answer one question:

> *How can a household battery minimize electricity costs using smart predictions?*

The app has three main sections:
- **Forecast Lab** — explore how well AI predicts your energy consumption and electricity prices
- **Battery Dispatch** — simulate different battery strategies and compare savings
- **Model Health** — evaluate the accuracy of all trained models

Use the arrows below to learn how each part works.
""",
            "stats": [
                ("~4 years", "of household energy data"),
                ("11 models", "trained and compared"),
                ("5 strategies", "for battery optimization"),
            ],
            "color": "#0ea5e9",
        },
        {
            "icon": "📈",
            "title": "Energy Forecast",
            "subtitle": "Predicting hourly household electricity consumption",
            "body": """
We trained **11 different models** on 4 years of real household power consumption data, ranging from simple baselines to advanced neural networks.

**Baseline models:**
- Naive day-ahead (yesterday = today)
- Moving average

**Machine Learning:**
- XGBoost & LightGBM — gradient boosting on time features + lag values

**Neural Networks (Darts library):**
- **N-BEATS** — pure time-series decomposition network
- **NHiTS** — hierarchical interpolation for multi-scale patterns
- **TFT** (Temporal Fusion Transformer) — state-of-the-art with attention mechanism

The **Forecast Lab** lets you pick any day, any model, and zoom into the exact hours where predictions go wrong.
""",
            "stats": [
                ("2006–2010", "source data period"),
                ("1-minute", "original resolution → hourly"),
                ("24h ahead", "rolling day-ahead forecast"),
            ],
            "color": "#10b981",
        },
        {
            "icon": "💶",
            "title": "Price Forecast",
            "subtitle": "Predicting electricity market prices hour by hour",
            "body": """
Knowing **when electricity is cheap or expensive** is essential for smart battery decisions.

We trained an **XGBoost model** on real European day-ahead market prices (OPSD dataset, Germany/Luxembourg) to predict the price for the next 24 hours.

**Features used:**
- Hour of day, day of week, month
- Price 24h ago (same hour yesterday)
- Price 48h ago & 168h ago (same hour last week)
- Rolling 24h average price

The **Price Forecast chart** appears both in Forecast Lab (alongside the load forecast) and in Battery Dispatch (as the top subplot), so you can directly see the relationship between price and battery decisions.
""",
            "stats": [
                ("2018–2020", "OPSD market price data"),
                ("XGBoost", "forecasting model"),
                ("24h horizon", "day-ahead prediction"),
            ],
            "color": "#f97316",
        },
        {
            "icon": "🔋",
            "title": "Battery Optimization",
            "subtitle": "Five strategies — from simple rules to AI agents",
            "body": """
Given load and price forecasts, **when should the battery charge or discharge?**

We implemented 5 strategies of increasing sophistication:

| Strategy | How it works |
|---|---|
| **Greedy ToU** | Simple rule: charge at night (cheap), discharge at peak (expensive) |
| **Day-ahead LP** | Solves a linear program using all 24h of price forecasts at once |
| **MPC (6h horizon)** | Re-optimizes every hour using only the next 6h — more realistic |
| **Stochastic LP** | Averages 30 different price scenarios to hedge against uncertainty |
| **RL Agent (PPO)** | Neural network trained on historical data — learns by trial and error |

The **Battery Dispatch** tab lets you select any strategy, tune battery capacity and charge rate, and compare savings against the no-battery baseline.
""",
            "stats": [
                ("Greedy → RL", "5 strategies to compare"),
                ("10 kWh", "default battery capacity"),
                ("LP & RL", "trained on real market data"),
            ],
            "color": "#8b5cf6",
        },
    ]

    if "slide_idx" not in st.session_state:
        st.session_state.slide_idx = 0

    slide = SLIDES[st.session_state.slide_idx]
    n     = len(SLIDES)
    idx   = st.session_state.slide_idx

    st.markdown(f"""
    <div style="
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    ">
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.4rem;">
            <span style="font-size:2.6rem;">{slide['icon']}</span>
            <div>
                <div style="font-size:1.6rem; font-weight:700; color:#0f172a;">{slide['title']}</div>
                <div style="font-size:1rem; color:#64748b; margin-top:0.1rem;">{slide['subtitle']}</div>
            </div>
        </div>
        <hr style="border-color:rgba(0,0,0,0.07); margin:1.2rem 0;">
    </div>
    """, unsafe_allow_html=True)

    col_body, col_stats = st.columns([2, 1])
    with col_body:
        st.markdown(slide["body"])
    with col_stats:
        for val, label in slide["stats"]:
            st.markdown(f"""
            <div style="
                background:#f8fafc;
                border:1px solid rgba(0,0,0,0.06);
                border-left: 4px solid {slide['color']};
                border-radius:10px;
                padding:0.9rem 1.1rem;
                margin-bottom:0.7rem;
            ">
                <div style="font-size:1.5rem; font-weight:700; color:{slide['color']};">{val}</div>
                <div style="font-size:0.85rem; color:#64748b;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    nav_left, nav_mid, nav_right = st.columns([1, 3, 1])

    with nav_left:
        if idx > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.slide_idx -= 1
                st.rerun()

    with nav_mid:
        dots = " ".join(
            f"<span style='font-size:1.1rem; color:{'#0ea5e9' if i == idx else '#cbd5e1'};'>●</span>"
            for i in range(n)
        )
        st.markdown(f"<div style='text-align:center; padding-top:0.4rem;'>{dots}</div>", unsafe_allow_html=True)

    with nav_right:
        if idx < n - 1:
            if st.button("Next →", use_container_width=True):
                st.session_state.slide_idx += 1
                st.rerun()
        else:
            if st.button("Go to app →", use_container_width=True):
                st.session_state.slide_idx = 0
                st.rerun()

    st.markdown(f"<div style='text-align:center; color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;'>Slide {idx+1} of {n} — use the sidebar to jump to any section</div>", unsafe_allow_html=True)


elif tab_choice == "Forecast Lab":
    st.title("Forecast Lab")
    st.markdown('<div class="section-note">Zoom, pan, hover, and inspect the hours where the model misses.</div>', unsafe_allow_html=True)

    available_dates = full_day_dates(filtered_load)
    if filtered_load is None or not available_dates:
        st.error("Load forecast artifact not found. Run `python src/train_baseline.py` first.")
        st.stop()

    lookup = date_lookup(available_dates)
    selected_date = st.selectbox(
        "Select day",
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

    # ── Price forecast for the same day ──
    if price_forecasts is not None:
        price_dates = full_day_dates(price_forecasts)
        if selected_date in price_dates:
            st.markdown("---")
            st.subheader("Price Forecast")
            day_prices = price_forecasts[price_forecasts.index.date == selected_date]
            ap = day_prices["price_actual"].values.astype(float)
            fp = day_prices["price_forecast"].values.astype(float)
            display_idx = shifted_index(day_data.index, selected_date, lookup)

            price_err = np.abs(ap - fp)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Price MAE", f"{price_err.mean():.4f} €/kWh")
            p2.metric("Max actual price", f"{ap.max():.4f} €/kWh")
            p3.metric("Worst price miss", f"{price_err.max():.4f} €/kWh")
            p4.metric("Mean bias", f"{(fp - ap).mean():+.4f} €/kWh")

            st.plotly_chart(price_figure(display_idx, ap, fp, selected_date, lookup), use_container_width=True)

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
    st.markdown('<div class="section-note">Select an optimization strategy, tune the battery, and compare every charge/discharge decision.</div>', unsafe_allow_html=True)

    available_dates = full_day_dates(filtered_load)
    if price_forecasts is not None:
        common_dates = sorted(set(available_dates) & set(full_day_dates(price_forecasts)))
        if common_dates:
            available_dates = common_dates

    if filtered_load is None or not available_dates:
        st.error("Load forecast artifact not found.")
        st.stop()

    lookup = date_lookup(available_dates)

    # ── Strategy definitions ──────────────────────────────────────────────────
    rl_available = RL_MODEL_PATH.exists()
    STRATEGY_OPTIONS = [
        "Greedy ToU",
        "Day-ahead LP",
        "MPC (6h horizon)",
        "Stochastic LP",
    ]
    if rl_available:
        STRATEGY_OPTIONS.append("RL Agent (PPO)")

    STRATEGY_DESCRIPTIONS = {
        "Greedy ToU":        "Simple rule — charge when cheap, discharge when expensive",
        "Day-ahead LP":      "Optimal plan for the full day (uses all price forecasts at once)",
        "MPC (6h horizon)":  "Re-optimizes every hour with only 6h lookahead — more realistic",
        "Stochastic LP":     "Averages 30 price scenarios — robust against forecast uncertainty",
        "RL Agent (PPO)":    "Neural network trained on historical data — learns by experience",
    }

    # ── Controls ──────────────────────────────────────────────────────────────
    top_cols = st.columns([1.6, 1.6, 1, 1, 1])
    selected_date = top_cols[0].selectbox(
        "Select day",
        available_dates,
        format_func=lambda d: str(display_date(d, lookup)),
        key="battery_date",
    )
    strategy = top_cols[1].selectbox(
        "Optimization strategy",
        STRATEGY_OPTIONS,
        help="\n".join(f"**{k}**: {v}" for k, v in STRATEGY_DESCRIPTIONS.items()),
        key="battery_strategy",
    )
    battery_cap  = top_cols[2].slider("Capacity (kWh)", 5.0, 20.0, BATTERY_CAPACITY, 0.5)
    max_rate     = top_cols[3].slider("Max rate (kW)",  1.0,  5.0, 2.5, 0.5)
    initial_soc  = top_cols[4].slider("Initial charge", 0, 100, 20, 5) / 100

    # Rolling strategies need 3-day window
    needs_window = strategy in ("MPC (6h horizon)",)
    selected_pos   = available_dates.index(selected_date)
    selected_dates = available_dates[selected_pos:selected_pos + 3] if needs_window else [selected_date]

    day_data    = filtered_load[np.isin(filtered_load.index.date, selected_dates)].copy()
    display_idx = shifted_multi_index(day_data.index, lookup)

    battery_config = BatteryConfig(
        capacity_kwh=battery_cap,
        max_charge_rate_kw=max_rate,
        max_discharge_rate_kw=max_rate,
        initial_soc=initial_soc,
    )

    forecast_kw  = day_data["forecast_kw"].values.astype(float)
    actual_kw    = day_data["actual_kw"].values.astype(float)
    dispatch_mae = float(np.mean(np.abs(actual_kw - forecast_kw)))

    fp = ap = None
    if price_forecasts is not None:
        day_prices = price_forecasts[np.isin(price_forecasts.index.date, selected_dates)]
        if len(day_prices) >= len(forecast_kw):
            fp = day_prices["price_forecast"].values[:len(forecast_kw)].astype(float)
            ap = day_prices["price_actual"].values[:len(forecast_kw)].astype(float)

    # ── Run selected strategy ────────────────────────────────────────────────
    needs_price = strategy not in ("Greedy ToU",)
    if needs_price and fp is None:
        st.warning("Price forecast data is required for this strategy but not available.")
        st.stop()

    with st.spinner(f"Running {strategy}..."):
        if strategy == "Greedy ToU":
            primary = simulate_battery(forecast_kw, actual_kw=actual_kw, actual_prices=ap, config=battery_config)
        elif strategy == "Day-ahead LP":
            primary = optimize_battery_lp(forecast_kw, fp, actual_kw=actual_kw, actual_prices=ap, config=battery_config)
        elif strategy == "MPC (6h horizon)":
            primary = optimize_battery_mpc(forecast_kw, fp, actual_kw=actual_kw, actual_prices=ap, horizon_hours=6, config=battery_config)
        elif strategy == "Stochastic LP":
            primary = optimize_battery_stochastic(forecast_kw, fp, actual_kw=actual_kw, actual_prices=ap, config=battery_config)
        elif strategy == "RL Agent (PPO)":
            primary = optimize_battery_rl(forecast_kw, fp, actual_kw=actual_kw, actual_prices=ap, config=battery_config)

    if not primary.get("success", True) and "message" in primary:
        st.error(f"Strategy failed: {primary['message']}")
        st.stop()

    # Greedy as baseline for comparison (always fast)
    greedy = simulate_battery(forecast_kw, actual_kw=actual_kw, actual_prices=ap, config=battery_config)

    # ── Metrics ───────────────────────────────────────────────────────────────
    g_costs = realized_costs(greedy)
    p_costs = realized_costs(primary)

    st.caption(f"**{strategy}** — {STRATEGY_DESCRIPTIONS[strategy]}")
    st.subheader("Strategy outcome")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("No battery cost",       f"{p_costs['without']:.3f} €")
    m2.metric("Greedy savings",        f"{g_costs['savings']:.3f} €",  delta=f"{g_costs['savings_pct']:.1f}%")
    m3.metric(f"{strategy} savings",   f"{p_costs['savings']:.3f} €",  delta=f"{p_costs['savings_pct']:.1f}%")
    m4.metric("vs Greedy",             f"{p_costs['savings'] - g_costs['savings']:+.3f} €")
    m5.metric("Forecast MAE",          f"{dispatch_mae:.3f} kW")

    # ── Chart ─────────────────────────────────────────────────────────────────
    smart_res = None if strategy == "Greedy ToU" else (primary if primary.get("success", True) else None)
    st.plotly_chart(
        dispatch_figure(display_idx, actual_kw, forecast_kw, greedy, smart_res, actual_prices=ap, forecast_prices=fp),
        use_container_width=True,
    )

    # ── Dispatch table ────────────────────────────────────────────────────────
    dispatch_rows = {
        "time":             display_idx.strftime("%Y-%m-%d %H:%M"),
        "actual_load_kw":   actual_kw,
        "forecast_load_kw": forecast_kw,
        "greedy_charge_kw": greedy["charge"],
        "greedy_grid_kw":   greedy["realized_grid_draw"],
        "greedy_soc_kwh":   greedy["realized_soc"],
    }
    if smart_res:
        dispatch_rows.update({
            f"{strategy}_charge_kw": smart_res["charge"],
            f"{strategy}_grid_kw":   smart_res["realized_grid_draw"],
            f"{strategy}_soc_kwh":   smart_res["realized_soc"],
        })
    st.subheader("Dispatch table")
    st.dataframe(
        pd.DataFrame(dispatch_rows).style.format("{:.3f}", subset=list(dispatch_rows.keys())[1:]),
        use_container_width=True, hide_index=True,
    )

    if not rl_available:
        st.info("**RL Agent** not yet trained. Run `python src/train_rl_agent.py` to unlock it.")


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
