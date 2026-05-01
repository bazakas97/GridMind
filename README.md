# GridMind

**Household Load Forecasting & Battery Dispatch**

GridMind is an applied machine-learning project for hourly household electricity load forecasting and battery dispatch simulation. It uses the UCI Individual Household Electric Power Consumption dataset, builds day-ahead load forecasts, maps real European day-ahead market prices onto the same timeline, and compares battery strategies on realized cost.

## What It Does

- Cleans the UCI household power dataset from minute-level readings to hourly data
- Builds calendar, lag, and rolling-history features
- Evaluates classical day-ahead baselines and a Darts N-HiTS neural model
- Saves reusable load and price forecast artifacts for the dashboard
- Simulates battery dispatch using forecasted load and forecasted prices
- Reports realized battery cost against actual test-set load and actual mapped prices
- Presents the results in a Streamlit dashboard

## Current Results

Load forecasting, evaluated on complete 24-hour day-ahead windows:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| XGBoost day-ahead 24h | 0.4045 | 0.5639 | 0.3565 |
| N-HiTS compact lags MAE 168h | 0.4062 | 0.5872 | 0.3022 |
| Linear Regression day-ahead 24h | 0.4489 | 0.6059 | 0.2571 |
| Naive day-ahead 24h | 0.4867 | 0.7393 | -0.1060 |
| Moving Average day-ahead 24h | 0.5518 | 0.6990 | 0.0112 |

Price forecasting uses OPSD Germany-Luxembourg day-ahead prices, seasonally mapped to the UCI load index:

| Model | MAE EUR/kWh | RMSE EUR/kWh | R2 | nMAE |
|---|---:|---:|---:|---:|
| XGBoost Price Forecaster | 0.0084 | 0.0119 | 0.4852 | 0.040 |

Battery dispatch sample day, using forecasted load for planning and actual load for realized cost:

| Strategy | Realized Cost With Battery | Realized Savings |
|---|---:|---:|
| Greedy fixed-ToU planning | 0.7360 EUR | -0.1311 EUR |
| Smart LP forecast-price planning | 0.5248 EUR | 0.0801 EUR |

## Key Artifacts

- `outputs/metrics/load_forecasts.csv`: saved XGBoost 24-hour day-ahead load forecasts
- `outputs/metrics/price_forecasts.csv`: saved XGBoost price forecasts and actual mapped prices
- `outputs/metrics/baseline_results.csv`: classical day-ahead load metrics
- `outputs/metrics/model_results.csv`: neural model metrics
- `outputs/metrics/battery_results.csv`: realized battery dispatch results
- `outputs/plots/`: generated forecast, price, EDA, and battery plots

## Run Locally

```bash
pip install -r requirements.txt

python src/preprocessing.py
python src/download_weather.py
python src/features.py
python src/download_real_prices.py
python src/train_baseline.py
python src/train_price_model.py
python src/optimize_battery.py

python -m streamlit run app/streamlit_app.py
```

Optional neural model training:

```bash
python -m pip install -r requirements-neural.txt
python src/train_model.py
python src/train_model.py --tune
```

Install the neural stack in a separate virtual environment. Recent Darts
versions require newer numpy/pandas/scikit-learn releases than the Streamlit
dashboard environment uses.

If your environment has an old or broken `streamlit` console entry point,
`python -m streamlit run app/streamlit_app.py` uses the Streamlit installed in
the active Python interpreter and avoids that launcher issue.

## Methodology Notes

- Train, validation, and test sets use a chronological 70/15/15 split.
- The dashboard reads saved artifacts instead of training models inside the app.
- Battery schedules are planned from forecasts, then evaluated on actual load.
- The LP optimizer uses battery charge/discharge efficiency and requires final state of charge to match the initial state of charge.
- MAPE is not reported for price forecasting because European market prices can be near zero or negative.
- OPSD prices are seasonally mapped because the UCI load data covers 2006-2010 while the available OPSD DE-LU series used here covers 2018-2020.

## Project Structure

```text
GridMind/
├── app/                  # Streamlit dashboard
├── data/
│   ├── raw/              # original dataset, ignored by git
│   └── processed/        # cleaned load data, features, mapped prices
├── notebooks/            # exploratory notebooks
├── outputs/
│   ├── metrics/          # saved metrics and forecast artifacts
│   ├── plots/            # generated plots
│   └── models/           # trained model files, ignored by git
├── src/                  # reusable pipeline scripts
├── requirements.txt
└── README.md
```

## Data Sources

- UCI Machine Learning Repository: Individual Household Electric Power Consumption
- Open Power System Data: European day-ahead prices from ENTSO-E Transparency Platform
