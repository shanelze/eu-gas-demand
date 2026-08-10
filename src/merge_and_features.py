# merge_and_features.py
#
# Now requires statsmodels for STL decomposition -- run locally:
#   pip install statsmodels
#   python merge_and_features.py

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

print("="*70)
print("MERGE")
print("="*70)

entsog = pd.read_csv("../data/entsog_germany_clean.csv", parse_dates=["date"])
agsi = pd.read_csv("../data/agsi_germany_clean.csv", parse_dates=["date"])
weather = pd.read_csv("../data/weather_germany_clean.csv", parse_dates=["date"])

print(f"ENTSOG:  {len(entsog):,} days ({entsog['date'].min().date()} to {entsog['date'].max().date()})")
print(f"AGSI:    {len(agsi):,} days ({agsi['date'].min().date()} to {agsi['date'].max().date()})")
print(f"Weather: {len(weather):,} days ({weather['date'].min().date()} to {weather['date'].max().date()})")

df = entsog.merge(agsi, on="date", how="inner").merge(weather, on="date", how="inner")
df = df.sort_values("date").reset_index(drop=True)
print(f"\nMerged: {len(df):,} days ({df['date'].min().date()} to {df['date'].max().date()})")

# Sanity check: STL needs a fully contiguous daily series, no gaps.
full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
missing = full_range.difference(df["date"])
if len(missing) > 0:
    raise RuntimeError(f"Merged series has {len(missing)} missing calendar days -- STL needs a contiguous series. "
                        f"First few gaps: {missing[:5].tolist()}")
print("Confirmed: no gaps in the daily date range.")

print("\n" + "="*70)
print("STL DECOMPOSITION")
print("="*70)

# Target is genuine final-consumer demand now, not total exit flow. Total
# exit flow (the old target) is ~70% cross-border transit + storage refill,
# neither of which is "demand" -- that combination is summer-peaked (transit
# arbitrage + storage injection season) which is the opposite of real
# heating-driven demand. consumer_exit_flow_kwh (ENTSOG's "Final Consumers" /
# "Letztverbraucher" points only) is winter-peaked as expected: ~561 GWh/day
# in January vs ~360-380 in summer.
df["demand_gwh"] = df["consumer_exit_flow_kwh"] / 1e6

# Keep the other flow categories too -- not used as model features (each one
# would leak into the target, since total = consumer + storage +
# interconnection), but useful as display/context series, e.g. a toggle in
# the dashboard between "final consumer demand", "total system offtake",
# "storage injection/withdrawal flow", and "cross-border transit flow".
df["total_system_offtake_gwh"] = df["total_exit_flow_kwh"] / 1e6
df["storage_exit_gwh"] = df["storage_exit_flow_kwh"] / 1e6
df["interconnection_exit_gwh"] = df["interconnection_exit_flow_kwh"] / 1e6

# Decompose into trend + annual seasonal + residual. period=365 -- daily
# data with annual seasonality. robust=True downweights outlier days so a
# handful of bad readings don't distort the trend/seasonal fit.
# NOTE: this is STL (Cleveland, Cleveland, McRae & Terpenning, 1990,
# Journal of Official Statistics vol. 6) -- not something invented for this
# project, it's a standard, citable time series decomposition method.
demand_series = df.set_index("date")["demand_gwh"]
stl_result = STL(demand_series, period=365, robust=True).fit()

df["stl_trend"] = stl_result.trend.values
df["stl_seasonal"] = stl_result.seasonal.values
df["stl_resid"] = stl_result.resid.values

print(f"Decomposed {len(df)} days into trend + seasonal + residual")
print(f"\nComponent stats (GWh/day):")
print(df[["stl_trend", "stl_seasonal", "stl_resid"]].describe())

# Sanity check: components should sum back to the original series
reconstruction_error = (df["stl_trend"] + df["stl_seasonal"] + df["stl_resid"] - df["demand_gwh"]).abs().max()
print(f"\nMax reconstruction error (trend+seasonal+resid vs actual): {reconstruction_error:.2e} GWh "
      f"(should be ~0, confirms the decomposition is additive and lossless)")

# Save the full merged + decomposed series now, before any leakage-safe
# lagging/dropping happens below. This is the file the dashboard reads from
# for the overview chart -- it has demand_gwh (final consumer) alongside
# total_system_offtake_gwh / storage_exit_gwh / interconnection_exit_gwh so
# a dashboard toggle can show all four series.
df.to_csv("../data/merged_daily.csv", index=False)
print("\nSaved data/merged_daily.csv (includes demand_gwh, total_system_offtake_gwh, "
      "storage_exit_gwh, interconnection_exit_gwh, stl_trend/seasonal/resid)")

print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# Calendar / seasonality (weekly patterns -- STL only captured ANNUAL
# seasonality above, so day-of-week effects are still handled here)
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["day_of_year"] = df["date"].dt.dayofyear

# Weather -- same-day values are safe (a forecast would substitute for an
# observation in production)
df["hdd_lag1"] = df["hdd"].shift(1)
df["hdd_ma7"] = df["hdd"].rolling(7).mean()

# Storage / AGSI-derived features -- lagged to D-1, same leakage reasoning
# as before.
df["storage_pct_full_lag1"] = df["storage_pct_full"].shift(1)
df["storage_twh_lag1"] = df["storage_twh"].shift(1)
df["withdrawal_gwh_lag1"] = df["withdrawal_gwh"].shift(1)
df["injection_gwh_lag1"] = df["injection_gwh"].shift(1)
df["agsi_consumption_gwh_lag1"] = df["agsi_consumption_gwh"].shift(1)

# Autoregressive features are now built on the STL RESIDUAL, not raw demand.
# This is the fix for the Monte Carlo anchoring bug: the residual hovers
# around zero by construction, so a recursive simulation seeded from it
# doesn't inherit a huge, misleading "recent demand level" bias -- the
# actual seasonal shape (which the residual excludes) gets added back
# deterministically at prediction time instead of being something the
# model has to reconstruct from its own lagged output.
df["resid_lag1"] = df["stl_resid"].shift(1)
df["resid_lag7"] = df["stl_resid"].shift(7)
df["resid_ma7"] = df["stl_resid"].shift(1).rolling(7).mean()
df["resid_ma30"] = df["stl_resid"].shift(1).rolling(30).mean()

# Drop raw same-day storage/AGSI columns (leakage), the raw per-category
# ENTSOG kWh columns (superseded by demand_gwh), and the other display-only
# flow series (total/storage/interconnection all overlap with or leak into
# today's target) -- keep stl_trend/stl_seasonal since we need them to
# reconstruct actual demand from a predicted residual later.
df_model = df.drop(columns=[
    "consumer_exit_flow_kwh", "storage_exit_flow_kwh", "interconnection_exit_flow_kwh",
    "total_exit_flow_kwh", "total_system_offtake_gwh", "storage_exit_gwh", "interconnection_exit_gwh",
    "storage_twh", "agsi_consumption_gwh", "injection_gwh", "withdrawal_gwh", "storage_pct_full",
])

before = len(df_model)
df_model = df_model.dropna().reset_index(drop=True)
print(f"\nRows before dropping warm-up NaNs: {before:,}")
print(f"Rows after: {len(df_model):,} (lost {before - len(df_model)} to the 30-day rolling warm-up window)")

feature_cols = [c for c in df_model.columns
                if c not in ["date", "demand_gwh", "stl_trend", "stl_seasonal", "stl_resid"]]
print(f"\nFinal feature set ({len(feature_cols)}): {feature_cols}")
print(f"Target: stl_resid (demand_gwh = stl_trend + stl_seasonal + stl_resid)")
print(f"\nDate range for modeling: {df_model['date'].min().date()} to {df_model['date'].max().date()}")

df_model.to_csv("../data/model_dataset.csv", index=False)
print("\nSaved data/model_dataset.csv")

# # merge_and_features.py
# #
# # Now requires statsmodels for STL decomposition -- run locally:
# #   pip install statsmodels
# #   python merge_and_features.py

# import pandas as pd
# import numpy as np
# from statsmodels.tsa.seasonal import STL

# print("="*70)
# print("MERGE")
# print("="*70)

# entsog = pd.read_csv("../data/entsog_germany_clean.csv", parse_dates=["date"])
# agsi = pd.read_csv("../data/agsi_germany_clean.csv", parse_dates=["date"])
# weather = pd.read_csv("../data/weather_germany_clean.csv", parse_dates=["date"])

# print(f"ENTSOG:  {len(entsog):,} days ({entsog['date'].min().date()} to {entsog['date'].max().date()})")
# print(f"AGSI:    {len(agsi):,} days ({agsi['date'].min().date()} to {agsi['date'].max().date()})")
# print(f"Weather: {len(weather):,} days ({weather['date'].min().date()} to {weather['date'].max().date()})")

# df = entsog.merge(agsi, on="date", how="inner").merge(weather, on="date", how="inner")
# df = df.sort_values("date").reset_index(drop=True)
# print(f"\nMerged: {len(df):,} days ({df['date'].min().date()} to {df['date'].max().date()})")

# # Sanity check: STL needs a fully contiguous daily series, no gaps.
# full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
# missing = full_range.difference(df["date"])
# if len(missing) > 0:
#     raise RuntimeError(f"Merged series has {len(missing)} missing calendar days -- STL needs a contiguous series. "
#                         f"First few gaps: {missing[:5].tolist()}")
# print("Confirmed: no gaps in the daily date range.")

# df.to_csv("../data/merged_daily.csv", index=False)
# print("Saved data/merged_daily.csv")

# print("\n" + "="*70)
# print("STL DECOMPOSITION")
# print("="*70)

# df["demand_gwh"] = df["entsog_exit_flow_kwh"] / 1e6

# # Decompose into trend + annual seasonal + residual. period=365 -- daily
# # data with annual seasonality. robust=True downweights outlier days so a
# # handful of bad readings don't distort the trend/seasonal fit.
# # NOTE: this is STL (Cleveland, Cleveland, McRae & Terpenning, 1990,
# # Journal of Official Statistics vol. 6) -- not something invented for this
# # project, it's a standard, citable time series decomposition method.
# demand_series = df.set_index("date")["demand_gwh"]
# stl_result = STL(demand_series, period=365, robust=True).fit()

# df["stl_trend"] = stl_result.trend.values
# df["stl_seasonal"] = stl_result.seasonal.values
# df["stl_resid"] = stl_result.resid.values

# print(f"Decomposed {len(df)} days into trend + seasonal + residual")
# print(f"\nComponent stats (GWh/day):")
# print(df[["stl_trend", "stl_seasonal", "stl_resid"]].describe())

# # Sanity check: components should sum back to the original series
# reconstruction_error = (df["stl_trend"] + df["stl_seasonal"] + df["stl_resid"] - df["demand_gwh"]).abs().max()
# print(f"\nMax reconstruction error (trend+seasonal+resid vs actual): {reconstruction_error:.2e} GWh "
#       f"(should be ~0, confirms the decomposition is additive and lossless)")

# print("\n" + "="*70)
# print("FEATURE ENGINEERING")
# print("="*70)

# # Calendar / seasonality (weekly patterns -- STL only captured ANNUAL
# # seasonality above, so day-of-week effects are still handled here)
# df["day_of_week"] = df["date"].dt.dayofweek
# df["month"] = df["date"].dt.month
# df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
# df["day_of_year"] = df["date"].dt.dayofyear

# # Weather -- same-day values are safe (a forecast would substitute for an
# # observation in production)
# df["hdd_lag1"] = df["hdd"].shift(1)
# df["hdd_ma7"] = df["hdd"].rolling(7).mean()

# # Storage / AGSI-derived features -- lagged to D-1, same leakage reasoning
# # as before.
# df["storage_pct_full_lag1"] = df["storage_pct_full"].shift(1)
# df["storage_twh_lag1"] = df["storage_twh"].shift(1)
# df["withdrawal_gwh_lag1"] = df["withdrawal_gwh"].shift(1)
# df["injection_gwh_lag1"] = df["injection_gwh"].shift(1)
# df["agsi_consumption_gwh_lag1"] = df["agsi_consumption_gwh"].shift(1)

# # Autoregressive features are now built on the STL RESIDUAL, not raw demand.
# # This is the fix for the Monte Carlo anchoring bug: the residual hovers
# # around zero by construction, so a recursive simulation seeded from it
# # doesn't inherit a huge, misleading "recent demand level" bias -- the
# # actual seasonal shape (which the residual excludes) gets added back
# # deterministically at prediction time instead of being something the
# # model has to reconstruct from its own lagged output.
# df["resid_lag1"] = df["stl_resid"].shift(1)
# df["resid_lag7"] = df["stl_resid"].shift(7)
# df["resid_ma7"] = df["stl_resid"].shift(1).rolling(7).mean()
# df["resid_ma30"] = df["stl_resid"].shift(1).rolling(30).mean()

# # Drop raw same-day storage/AGSI columns (leakage) and the old
# # unlagged demand-level columns -- keep stl_trend/stl_seasonal since we need
# # them to reconstruct actual demand from a predicted residual later.
# df_model = df.drop(columns=[
#     "entsog_exit_flow_kwh", "storage_twh", "agsi_consumption_gwh",
#     "injection_gwh", "withdrawal_gwh", "storage_pct_full",
# ])

# before = len(df_model)
# df_model = df_model.dropna().reset_index(drop=True)
# print(f"\nRows before dropping warm-up NaNs: {before:,}")
# print(f"Rows after: {len(df_model):,} (lost {before - len(df_model)} to the 30-day rolling warm-up window)")

# feature_cols = [c for c in df_model.columns
#                 if c not in ["date", "demand_gwh", "stl_trend", "stl_seasonal", "stl_resid"]]
# print(f"\nFinal feature set ({len(feature_cols)}): {feature_cols}")
# print(f"Target: stl_resid (demand_gwh = stl_trend + stl_seasonal + stl_resid)")
# print(f"\nDate range for modeling: {df_model['date'].min().date()} to {df_model['date'].max().date()}")

# df_model.to_csv("../data/model_dataset.csv", index=False)
# print("\nSaved data/model_dataset.csv")
