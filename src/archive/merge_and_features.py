import pandas as pd
import numpy as np

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
print(f"\nMerged: {len(df):,} days ({df['date'].min().date()} to {df['date'].max().date()})")
print(f"Nulls per column:\n{df.isna().sum()}")

df.to_csv("../data/merged_daily.csv", index=False)
print("\nSaved data/merged_daily.csv")

print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

df = df.sort_values("date").reset_index(drop=True)

# Target: national gas demand proxy (ENTSOG exit flow), in GWh for readability.
df["demand_gwh"] = df["entsog_exit_flow_kwh"] / 1e6

# Calendar / seasonality -- always knowable in advance, safe as same-day features.
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["day_of_year"] = df["date"].dt.dayofyear

# Weather -- same-day values are safe to use as-is, since in a real forecasting
# setup you'd use a weather FORECAST for the target day, not a same-day
# observation. hdd/temperature are treated as "known in advance" inputs.
df["hdd_lag1"] = df["hdd"].shift(1)
df["hdd_ma7"] = df["hdd"].rolling(7).mean()

# Storage / AGSI-derived features -- these must be lagged. Using today's own
# storage report or AGSI's own consumption estimate to predict today's ENTSOG
# demand would be circular (that data isn't actually available before the
# target day in a live forecast, and AGSI consumption is itself a competing
# proxy for the same underlying demand we're trying to model). Every one of
# these is shifted to D-1; the unlagged raw columns are dropped from the
# final feature set below.
df["storage_pct_full_lag1"] = df["storage_pct_full"].shift(1)
df["storage_twh_lag1"] = df["storage_twh"].shift(1)
df["withdrawal_gwh_lag1"] = df["withdrawal_gwh"].shift(1)
df["injection_gwh_lag1"] = df["injection_gwh"].shift(1)
df["agsi_consumption_gwh_lag1"] = df["agsi_consumption_gwh"].shift(1)

# Autoregressive demand features -- also inherently lagged (D-1, D-7, and a
# trailing 7/30-day average computed only from prior days).
df["demand_lag1"] = df["demand_gwh"].shift(1)
df["demand_lag7"] = df["demand_gwh"].shift(7)
df["demand_ma7"] = df["demand_gwh"].shift(1).rolling(7).mean()
df["demand_ma30"] = df["demand_gwh"].shift(1).rolling(30).mean()

# Drop the raw same-day storage/AGSI columns now that lagged versions exist,
# so there is no way for the model to accidentally see same-day information.
df_model = df.drop(columns=[
    "entsog_exit_flow_kwh", "storage_twh", "agsi_consumption_gwh",
    "injection_gwh", "withdrawal_gwh", "storage_pct_full",
])

before = len(df_model)
df_model = df_model.dropna().reset_index(drop=True)
print(f"\nRows before dropping warm-up NaNs (from lag/rolling features): {before:,}")
print(f"Rows after: {len(df_model):,} (lost {before - len(df_model)} to the 30-day rolling warm-up window)")

feature_cols = [c for c in df_model.columns if c not in ["date", "demand_gwh"]]
print(f"\nFinal feature set ({len(feature_cols)}): {feature_cols}")
print(f"\nDate range for modeling: {df_model['date'].min().date()} to {df_model['date'].max().date()}")

df_model.to_csv("../data/model_dataset.csv", index=False)
print("\nSaved data/model_dataset.csv")
