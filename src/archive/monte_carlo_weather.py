# monte_carlo_weather.py
#
# Monte Carlo demand forecast driven by resampled historical weather.
# Loads the already-trained model (no retraining) and simulates many
# possible future demand paths by feeding it many possible future weather
# scenarios, built from real historical weather/storage blocks rather than
# an invented stochastic process.
#
#   python monte_carlo_weather.py

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

np.random.seed(42)

HORIZON_DAYS = 90
N_SIMULATIONS = 1000
DAY_OFFSET_JITTER = 10  # +/- days around the same calendar date, across years

df = pd.read_csv("../data/model_dataset.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

model = xgb.XGBRegressor()
model.load_model("../data/xgb_model.json")

feature_cols = [c for c in df.columns if c not in ["date", "demand_gwh"]]

# Forecast starts the day after the last real observation
forecast_start = df["date"].max() + pd.Timedelta(days=1)
forecast_dates = pd.date_range(forecast_start, periods=HORIZON_DAYS, freq="D")
print(f"Forecasting {HORIZON_DAYS} days from {forecast_dates[0].date()} to {forecast_dates[-1].date()}")

# --- Build the pool of candidate historical analog blocks ---
# For each past year that has a full HORIZON_DAYS block available starting
# near the same calendar date, record it as a candidate. We also jitter the
# start date by +/- DAY_OFFSET_JITTER days so the pool has far more than
# one path per historical year.
df_indexed = df.set_index("date")
available_years = sorted(df["date"].dt.year.unique())

candidates = []
for year in available_years:
    for offset in range(-DAY_OFFSET_JITTER, DAY_OFFSET_JITTER + 1):
        try:
            analog_start = forecast_start.replace(year=year) + pd.Timedelta(days=offset)
        except ValueError:
            continue  # Feb 29 in a non-leap year etc.
        analog_dates = pd.date_range(analog_start, periods=HORIZON_DAYS, freq="D")
        if analog_dates[0] in df_indexed.index and analog_dates[-1] in df_indexed.index:
            if analog_dates.isin(df_indexed.index).all():
                candidates.append(analog_dates)

print(f"Built {len(candidates)} candidate historical weather/storage blocks "
      f"across {len(available_years)} years (with +/-{DAY_OFFSET_JITTER}-day jitter)")

if len(candidates) == 0:
    raise RuntimeError("No valid historical analog blocks found -- need more history or a shorter horizon.")

weather_storage_cols = [
    "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "hdd",
    "hdd_lag1", "hdd_ma7",
    "storage_pct_full_lag1", "storage_twh_lag1",
    "withdrawal_gwh_lag1", "injection_gwh_lag1", "agsi_consumption_gwh_lag1",
]

# --- Seed history for the recursive autoregressive features ---
# demand_lag1/lag7/ma7/ma30 for day 1 of the forecast need real known demand
# history immediately preceding forecast_start.
seed_history = df["demand_gwh"].iloc[-30:].tolist()  # last 30 real days

# --- Run simulations ---
all_paths = np.zeros((N_SIMULATIONS, HORIZON_DAYS))

for sim in range(N_SIMULATIONS):
    analog_dates = candidates[np.random.randint(len(candidates))]
    analog_block = df_indexed.loc[analog_dates, weather_storage_cols].reset_index(drop=True)

    demand_history = list(seed_history)  # running buffer, updated with predictions

    for day in range(HORIZON_DAYS):
        target_date = forecast_dates[day]
        row = {}
        for col in weather_storage_cols:
            row[col] = analog_block.loc[day, col]

        row["day_of_week"] = target_date.dayofweek
        row["month"] = target_date.month
        row["is_weekend"] = int(target_date.dayofweek >= 5)
        row["day_of_year"] = target_date.dayofyear

        row["demand_lag1"] = demand_history[-1]
        row["demand_lag7"] = demand_history[-7]
        row["demand_ma7"] = np.mean(demand_history[-7:])
        row["demand_ma30"] = np.mean(demand_history[-30:])

        X_row = pd.DataFrame([row])[feature_cols]
        pred = model.predict(X_row)[0]

        all_paths[sim, day] = pred
        demand_history.append(pred)

print(f"\nRan {N_SIMULATIONS} simulations across {HORIZON_DAYS}-day horizon")

# --- Aggregate into percentiles ---
percentiles = [5, 25, 50, 75, 95]
pct_values = np.percentile(all_paths, percentiles, axis=0)
fan = pd.DataFrame(pct_values.T, columns=[f"p{p}" for p in percentiles])
fan.insert(0, "date", forecast_dates)

print("\nSample of forecast distribution (first 5 days):")
print(fan.head().to_string(index=False))

fan.to_csv("../data/monte_carlo_forecast.csv", index=False)
print("\nSaved data/monte_carlo_forecast.csv")

# --- Plot fan chart ---
plt.figure(figsize=(11, 5))
plt.plot(df["date"].iloc[-90:], df["demand_gwh"].iloc[-90:], color="black",
          linewidth=1, label="Recent actual")
plt.plot(fan["date"], fan["p50"], color="#1f6feb", linewidth=1.5, label="Median simulated demand")
plt.fill_between(fan["date"], fan["p5"], fan["p95"], color="#1f6feb", alpha=0.15, label="5th-95th percentile")
plt.fill_between(fan["date"], fan["p25"], fan["p75"], color="#1f6feb", alpha=0.3, label="25th-75th percentile")
plt.title(f"Monte Carlo demand forecast ({N_SIMULATIONS} weather-driven simulations)")
plt.ylabel("GWh/day")
plt.legend()
plt.tight_layout()
plt.savefig("../notebooks/monte_carlo_fan_chart.png", dpi=120)
print("Saved notebooks/monte_carlo_fan_chart.png")
