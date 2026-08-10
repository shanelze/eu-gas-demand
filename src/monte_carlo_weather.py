# monte_carlo_weather.py
#
# Monte Carlo demand forecast via XGBoost QUANTILE regression, driven by
# resampled historical scenarios -- not recursive re-prediction, and not
# pure historical replay either. Middle ground between the two approaches
# we tried before:
#
#   v1 (recursive point prediction): each simulated day's prediction fed
#   into the next day's lag features. A weak/noisy signal recursively
#   re-predicting itself for months compounded into a misleading drift.
#
#   v2 (pure historical bootstrap): dropped the model entirely and just
#   replayed real historical residual sequences. Fixed the drift, but
#   doesn't use XGBoost at all, and every simulation that lands on the same
#   historical block produces an identical path.
#
#   v3 (this version): for each simulated day, source ALL its features --
#   weather, storage, calendar, AND the resid_lag/ma autoregressive
#   features -- from a REAL historical analog day (never from a prior
#   SIMULATED day). That's what kills the compounding: nothing here ever
#   feeds a model's own earlier guess back into itself. Then run those real
#   feature vectors through 5 XGBoost quantile models (trained on
#   5th/25th/50th/75th/95th percentile of the residual) to get a predicted
#   distribution for that specific day, and draw one random sample from it.
#   That's the genuinely "Monte Carlo" part: two simulations landing on the
#   same historical block no longer produce identical paths, because each
#   draws its own random sample from the model's predicted distribution
#   for that day.
#
#   Reference for this style of approach: quantile/probabilistic gradient
#   boosting for energy demand and price forecasting, e.g. Gioia & Fabbiani
#   style probabilistic load forecasting using quantile regression forests
#   / gradient boosting, and the location-scale-shape probabilistic
#   forecasting literature discussed earlier for intraday power.
#
# Requires the 5 models from train_quantile_models.py to exist already:
#   python train_quantile_models.py
#   python monte_carlo_weather.py
 
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
 
np.random.seed(42)
 
HORIZON_DAYS = 365
N_SIMULATIONS = 1000
DAY_OFFSET_JITTER = 10
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
 
df = pd.read_csv("../data/model_dataset.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df_indexed = df.set_index("date")
 
feature_cols = [c for c in df.columns
                if c not in ["date", "demand_gwh", "stl_trend", "stl_seasonal", "stl_resid"]]
 
print("Loading 5 quantile models...")
models = {}
for q in QUANTILES:
    m = xgb.XGBRegressor()
    m.load_model(f"../data/xgb_quantile_{int(q*100):02d}.json")
    models[q] = m
 
forecast_start = df["date"].max() + pd.Timedelta(days=1)
forecast_dates = pd.date_range(forecast_start, periods=HORIZON_DAYS, freq="D")
print(f"Forecasting {HORIZON_DAYS} days from {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
 
# --- Deterministic trend + seasonal baseline, same as before ---
last_trend = df["stl_trend"].iloc[-1]
seasonal_lookup = df_indexed["stl_seasonal"]
 
def get_seasonal(date):
    for years_back in (365, 730, 1095):
        analog_date = date - pd.Timedelta(days=years_back)
        if analog_date in seasonal_lookup.index:
            return seasonal_lookup.loc[analog_date]
    raise ValueError(f"No seasonal analog found for {date}")
 
seasonal_future = pd.Series([get_seasonal(d) for d in forecast_dates], index=forecast_dates)
trend_future = pd.Series(last_trend, index=forecast_dates)
 
print(f"Trend held flat at last fitted value: {last_trend:.1f} GWh/day")
print(f"Seasonal baseline range over horizon: {seasonal_future.min():.1f} to {seasonal_future.max():.1f} GWh/day")
 
# --- Candidate historical blocks (same jittered-year logic as before) ---
available_years = sorted(df["date"].dt.year.unique())
candidates = []
for year in available_years:
    for offset in range(-DAY_OFFSET_JITTER, DAY_OFFSET_JITTER + 1):
        try:
            analog_start = forecast_start.replace(year=year) + pd.Timedelta(days=offset)
        except ValueError:
            continue
        analog_dates = pd.date_range(analog_start, periods=HORIZON_DAYS, freq="D")
        if analog_dates.isin(df_indexed.index).all():
            candidates.append(analog_dates)
 
print(f"Built {len(candidates)} candidate historical analog blocks")
if len(candidates) == 0:
    raise RuntimeError(f"No valid {HORIZON_DAYS}-day historical analog blocks found -- "
                        f"try a shorter HORIZON_DAYS given ~5 years of available history.")
 
# --- Pre-compute quantile predictions for every candidate block up front.
# Each block is HORIZON_DAYS long and uses only real historical features,
# so this is plain batch inference, not simulation -- do it once per block
# rather than once per simulation. ---
print("Running quantile predictions for each candidate block's real historical features...")
block_quantile_preds = []  # list of (HORIZON_DAYS, 5) arrays, one per candidate
for analog_dates in candidates:
    X_block = df_indexed.loc[analog_dates, feature_cols]
    preds = np.column_stack([models[q].predict(X_block) for q in QUANTILES])
    preds = np.sort(preds, axis=1)  # guard against any tiny quantile-crossing
    block_quantile_preds.append(preds)
 
# --- Run simulations: for each sim, pick a block, then draw one random
# sample per day from that day's predicted quantile distribution ---
all_demand_paths = np.zeros((N_SIMULATIONS, HORIZON_DAYS))
 
for sim in range(N_SIMULATIONS):
    block_idx = np.random.randint(len(candidates))
    preds = block_quantile_preds[block_idx]  # (HORIZON_DAYS, 5)
    u = np.random.uniform(0, 1, size=HORIZON_DAYS)
    sampled_resid = np.array([
        np.interp(u[d], QUANTILES, preds[d])
        for d in range(HORIZON_DAYS)
    ])
    all_demand_paths[sim, :] = trend_future.values + seasonal_future.values + sampled_resid
 
print(f"\nRan {N_SIMULATIONS} simulations across {HORIZON_DAYS}-day horizon")
 
# Gas demand can't be negative -- floor any simulated paths where a low
# quantile draw overshot below zero (happens occasionally on the q05 tail
# for winter days with a small training set).
n_clipped = (all_demand_paths < 0).sum()
if n_clipped > 0:
    print(f"Clipped {n_clipped} negative simulated values up to 0 "
          f"({n_clipped / all_demand_paths.size * 100:.2f}% of all simulated days)")
all_demand_paths = np.maximum(all_demand_paths, 0)
 
percentiles = [5, 25, 50, 75, 95]
pct_values = np.percentile(all_demand_paths, percentiles, axis=0)
fan = pd.DataFrame(pct_values.T, columns=[f"p{p}" for p in percentiles])
fan.insert(0, "date", forecast_dates)
 
print("\nSample of forecast distribution (first 5 days):")
print(fan.head().to_string(index=False))
print("\nSample of forecast distribution (last 5 days):")
print(fan.tail().to_string(index=False))
print("\nSample around the coming winter peak (mid-Dec to mid-Jan, if in range):")
winter_mask = (fan["date"] >= f"{forecast_dates[0].year}-12-10") & (fan["date"] <= f"{forecast_dates[0].year+1}-01-20")
print(fan[winter_mask].to_string(index=False))
 
fan.to_csv("../data/monte_carlo_forecast.csv", index=False)
print("\nSaved data/monte_carlo_forecast.csv")
 
plt.figure(figsize=(13, 5))
plt.plot(df["date"].iloc[-180:], df["demand_gwh"].iloc[-180:], color="black",
          linewidth=1, label="Recent actual")
plt.plot(fan["date"], fan["p50"], color="#1f6feb", linewidth=1.5, label="Median simulated demand")
plt.fill_between(fan["date"], fan["p5"], fan["p95"], color="#1f6feb", alpha=0.15, label="5th-95th percentile")
plt.fill_between(fan["date"], fan["p25"], fan["p75"], color="#1f6feb", alpha=0.3, label="25th-75th percentile")
plt.title(f"Monte Carlo demand forecast ({N_SIMULATIONS} sims, XGBoost quantile regression)")
plt.ylabel("GWh/day")
plt.legend()
plt.tight_layout()
plt.savefig("../notebooks/monte_carlo_fan_chart.png", dpi=120)
print("Saved notebooks/monte_carlo_fan_chart.png")

# # monte_carlo_weather.py
# #
# # Monte Carlo demand forecast via historical block bootstrap.
# #
# # Earlier version recursively re-predicted the STL residual day-by-day with
# # the trained model. That's dropped now: validation showed the residual
# # model doesn't clearly beat "assume no surprise" in most folds (naive
# # seasonal-only baseline won 4/5 folds), so asking it to recursively
# # re-predict itself hundreds of days in a row was exactly the situation
# # where a weak, low-confidence signal compounds into a misleading drift.
# #
# # Instead: keep our own deterministic trend + seasonal baseline (today's
# # best understanding of the expected shape), and for the "surprise"
# # component, directly reuse the ACTUAL historical residual sequence from a
# # randomly chosen historical analog period -- real day-to-day deviations
# # that really happened, not something re-predicted and prone to drifting.
# #
# #   python monte_carlo_weather.py

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)

# HORIZON_DAYS = 365  # a full year out -- change freely, e.g. 180 for just through winter
# N_SIMULATIONS = 1000
# DAY_OFFSET_JITTER = 10

# df = pd.read_csv("../data/model_dataset.csv", parse_dates=["date"])
# df = df.sort_values("date").reset_index(drop=True)
# df_indexed = df.set_index("date")

# forecast_start = df["date"].max() + pd.Timedelta(days=1)
# forecast_dates = pd.date_range(forecast_start, periods=HORIZON_DAYS, freq="D")
# print(f"Forecasting {HORIZON_DAYS} days from {forecast_dates[0].date()} to {forecast_dates[-1].date()}")

# # --- Deterministic seasonal + trend baseline (unchanged approach) ---
# last_trend = df["stl_trend"].iloc[-1]
# seasonal_lookup = df_indexed["stl_seasonal"]

# def get_seasonal(date):
#     for years_back in (365, 730, 1095):
#         analog_date = date - pd.Timedelta(days=years_back)
#         if analog_date in seasonal_lookup.index:
#             return seasonal_lookup.loc[analog_date]
#     raise ValueError(f"No seasonal analog found for {date}")

# seasonal_future = pd.Series([get_seasonal(d) for d in forecast_dates], index=forecast_dates)
# trend_future = pd.Series(last_trend, index=forecast_dates)

# print(f"Trend held flat at last fitted value: {last_trend:.1f} GWh/day")
# print(f"Seasonal baseline range over horizon: {seasonal_future.min():.1f} to {seasonal_future.max():.1f} GWh/day")

# # --- Candidate historical blocks: need HORIZON_DAYS of contiguous real
# # residual data starting near the same calendar date in some past year ---
# available_years = sorted(df["date"].dt.year.unique())
# candidates = []
# for year in available_years:
#     for offset in range(-DAY_OFFSET_JITTER, DAY_OFFSET_JITTER + 1):
#         try:
#             analog_start = forecast_start.replace(year=year) + pd.Timedelta(days=offset)
#         except ValueError:
#             continue
#         analog_dates = pd.date_range(analog_start, periods=HORIZON_DAYS, freq="D")
#         if analog_dates.isin(df_indexed.index).all():
#             candidates.append(analog_dates)

# print(f"Built {len(candidates)} candidate historical residual blocks "
#       f"(fewer than before, since a {HORIZON_DAYS}-day block needs a full "
#       f"real year of history to draw from)")
# if len(candidates) == 0:
#     raise RuntimeError(f"No valid {HORIZON_DAYS}-day historical analog blocks found -- "
#                         f"try a shorter HORIZON_DAYS given ~5 years of available history.")

# # --- Run simulations: pure resampling, no model calls needed ---
# all_demand_paths = np.zeros((N_SIMULATIONS, HORIZON_DAYS))

# for sim in range(N_SIMULATIONS):
#     analog_dates = candidates[np.random.randint(len(candidates))]
#     historical_resid = df_indexed.loc[analog_dates, "stl_resid"].values
#     all_demand_paths[sim, :] = trend_future.values + seasonal_future.values + historical_resid

# print(f"\nRan {N_SIMULATIONS} simulations across {HORIZON_DAYS}-day horizon")

# percentiles = [5, 25, 50, 75, 95]
# pct_values = np.percentile(all_demand_paths, percentiles, axis=0)
# fan = pd.DataFrame(pct_values.T, columns=[f"p{p}" for p in percentiles])
# fan.insert(0, "date", forecast_dates)

# print("\nSample of forecast distribution (first 5 days):")
# print(fan.head().to_string(index=False))
# print("\nSample of forecast distribution (last 5 days):")
# print(fan.tail().to_string(index=False))
# print("\nSample around the coming winter peak (mid-Dec to mid-Jan, if in range):")
# winter_mask = (fan["date"] >= f"{forecast_dates[0].year}-12-10") & (fan["date"] <= f"{forecast_dates[0].year+1}-01-20")
# print(fan[winter_mask].to_string(index=False))

# fan.to_csv("../data/monte_carlo_forecast.csv", index=False)
# print("\nSaved data/monte_carlo_forecast.csv")

# plt.figure(figsize=(13, 5))
# plt.plot(df["date"].iloc[-180:], df["demand_gwh"].iloc[-180:], color="black",
#           linewidth=1, label="Recent actual")
# plt.plot(fan["date"], fan["p50"], color="#1f6feb", linewidth=1.5, label="Median simulated demand")
# plt.fill_between(fan["date"], fan["p5"], fan["p95"], color="#1f6feb", alpha=0.15, label="5th-95th percentile")
# plt.fill_between(fan["date"], fan["p25"], fan["p75"], color="#1f6feb", alpha=0.3, label="25th-75th percentile")
# plt.title(f"Monte Carlo demand forecast ({N_SIMULATIONS} historical-bootstrap simulations)")
# plt.ylabel("GWh/day")
# plt.legend()
# plt.tight_layout()
# plt.savefig("../notebooks/monte_carlo_fan_chart.png", dpi=120)
# print("Saved notebooks/monte_carlo_fan_chart.png")

# # monte_carlo_weather.py
# #
# # Monte Carlo demand forecast, now simulating the STL RESIDUAL recursively

# #   python monte_carlo_weather.py

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import matplotlib.pyplot as plt

# np.random.seed(42)

# HORIZON_DAYS = 90
# N_SIMULATIONS = 1000
# DAY_OFFSET_JITTER = 10

# df = pd.read_csv("../data/model_dataset.csv", parse_dates=["date"])
# df = df.sort_values("date").reset_index(drop=True)

# model = xgb.XGBRegressor()
# model.load_model("../data/xgb_model.json")

# feature_cols = [c for c in df.columns
#                 if c not in ["date", "demand_gwh", "stl_trend", "stl_seasonal", "stl_resid"]]

# forecast_start = df["date"].max() + pd.Timedelta(days=1)
# forecast_dates = pd.date_range(forecast_start, periods=HORIZON_DAYS, freq="D")
# print(f"Forecasting {HORIZON_DAYS} days from {forecast_dates[0].date()} to {forecast_dates[-1].date()}")

# df_indexed = df.set_index("date")

# # --- Deterministic seasonal + trend baseline for the forecast horizon ---

# last_trend = df["stl_trend"].iloc[-1]
# seasonal_lookup = df_indexed["stl_seasonal"]

# def get_seasonal(date):
#     analog_date = date - pd.Timedelta(days=365)
#     if analog_date in seasonal_lookup.index:
#         return seasonal_lookup.loc[analog_date]
#     # fall back to 2 years back if needed (leap-year edge cases)
#     return seasonal_lookup.loc[date - pd.Timedelta(days=730)]

# seasonal_future = pd.Series([get_seasonal(d) for d in forecast_dates], index=forecast_dates)
# trend_future = pd.Series(last_trend, index=forecast_dates)

# print(f"Trend held flat at last fitted value: {last_trend:.1f} GWh/day")
# print(f"Seasonal baseline range over horizon: {seasonal_future.min():.1f} to {seasonal_future.max():.1f} GWh/day")

# # --- Candidate historical weather/storage analog blocks (unchanged approach) ---
# available_years = sorted(df["date"].dt.year.unique())
# candidates = []
# for year in available_years:
#     for offset in range(-DAY_OFFSET_JITTER, DAY_OFFSET_JITTER + 1):
#         try:
#             analog_start = forecast_start.replace(year=year) + pd.Timedelta(days=offset)
#         except ValueError:
#             continue
#         analog_dates = pd.date_range(analog_start, periods=HORIZON_DAYS, freq="D")
#         if analog_dates.isin(df_indexed.index).all():
#             candidates.append(analog_dates)

# print(f"Built {len(candidates)} candidate historical weather/storage blocks")
# if len(candidates) == 0:
#     raise RuntimeError("No valid historical analog blocks found.")

# weather_storage_cols = [
#     "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "hdd",
#     "hdd_lag1", "hdd_ma7",
#     "storage_pct_full_lag1", "storage_twh_lag1",
#     "withdrawal_gwh_lag1", "injection_gwh_lag1", "agsi_consumption_gwh_lag1",
# ]

# # --- Seed history for the recursive residual features (last 30 real days) ---
# seed_history = df["stl_resid"].iloc[-30:].tolist()
# print(f"Residual seed range (last 30 real days): {min(seed_history):.1f} to {max(seed_history):.1f} GWh/day "
#       f"(centered near zero, unlike raw demand -- this is the fix)")

# # --- Run simulations ---
# all_demand_paths = np.zeros((N_SIMULATIONS, HORIZON_DAYS))

# for sim in range(N_SIMULATIONS):
#     analog_dates = candidates[np.random.randint(len(candidates))]
#     analog_block = df_indexed.loc[analog_dates, weather_storage_cols].reset_index(drop=True)

#     resid_history = list(seed_history)

#     for day in range(HORIZON_DAYS):
#         target_date = forecast_dates[day]
#         row = {col: analog_block.loc[day, col] for col in weather_storage_cols}

#         row["day_of_week"] = target_date.dayofweek
#         row["month"] = target_date.month
#         row["is_weekend"] = int(target_date.dayofweek >= 5)
#         row["day_of_year"] = target_date.dayofyear

#         row["resid_lag1"] = resid_history[-1]
#         row["resid_lag7"] = resid_history[-7]
#         row["resid_ma7"] = np.mean(resid_history[-7:])
#         row["resid_ma30"] = np.mean(resid_history[-30:])

#         X_row = pd.DataFrame([row])[feature_cols]
#         resid_pred = model.predict(X_row)[0]
#         resid_history.append(resid_pred)

#         # Reconstruct actual demand for this simulated day
#         demand_pred = trend_future.iloc[day] + seasonal_future.iloc[day] + resid_pred
#         all_demand_paths[sim, day] = demand_pred

# print(f"\nRan {N_SIMULATIONS} simulations across {HORIZON_DAYS}-day horizon")

# percentiles = [5, 25, 50, 75, 95]
# pct_values = np.percentile(all_demand_paths, percentiles, axis=0)
# fan = pd.DataFrame(pct_values.T, columns=[f"p{p}" for p in percentiles])
# fan.insert(0, "date", forecast_dates)

# print("\nSample of forecast distribution (first 5 days):")
# print(fan.head().to_string(index=False))
# print("\nSample of forecast distribution (last 5 days):")
# print(fan.tail().to_string(index=False))

# fan.to_csv("../data/monte_carlo_forecast.csv", index=False)
# print("\nSaved data/monte_carlo_forecast.csv")

# plt.figure(figsize=(11, 5))
# plt.plot(df["date"].iloc[-90:], df["demand_gwh"].iloc[-90:], color="black",
#           linewidth=1, label="Recent actual")
# plt.plot(fan["date"], fan["p50"], color="#1f6feb", linewidth=1.5, label="Median simulated demand")
# plt.fill_between(fan["date"], fan["p5"], fan["p95"], color="#1f6feb", alpha=0.15, label="5th-95th percentile")
# plt.fill_between(fan["date"], fan["p25"], fan["p75"], color="#1f6feb", alpha=0.3, label="25th-75th percentile")
# plt.title(f"Monte Carlo demand forecast ({N_SIMULATIONS} weather-driven simulations, STL-residual method)")
# plt.ylabel("GWh/day")
# plt.legend()
# plt.tight_layout()
# plt.savefig("../notebooks/monte_carlo_fan_chart.png", dpi=120)
# print("Saved notebooks/monte_carlo_fan_chart.png")

