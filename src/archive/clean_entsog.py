import pandas as pd
import numpy as np

print("="*70)
print("ENTSOG CLEANING")
print("="*70)

df = pd.read_csv(
    "../data/entsog_germany.csv",
    usecols=["point_key","point_label","operator_key","direction_key",
             "period_from","period_to","value","unit","flow_status"],
    low_memory=False,
)
print(f"\nRaw rows: {len(df):,}")
print(f"Unique points: {df['point_key'].nunique()}")

# --- 1. Drop points that never report real data ---
null_frac = df.groupby("point_key")["value"].apply(lambda s: s.isna().mean())
all_null_points = null_frac[null_frac == 1.0].index.tolist()
print(f"\nDropping {len(all_null_points)} points with 100% null values:")
print(f"  {all_null_points}")
df = df[~df["point_key"].isin(all_null_points)]
print(f"Rows after dropping dead points: {len(df):,}")

# --- 2. Drop remaining nulls (partial gaps) ---
before = len(df)
df = df.dropna(subset=["value"])
print(f"\nDropped {before - len(df):,} remaining null rows (partial gaps)")

# --- 3. Remove implausible sensor/reporting glitches ---
# A single interconnection/consumer point in Germany cannot plausibly flow
# more than 10 billion kWh in a single day (that alone would be ~3x the
# entire country's actual daily consumption). This threshold is deliberately
# generous -- it exists only to catch glitches like the one found during
# inspection (FNC-00206 spiking to 248.9 billion kWh/d vs its typical
# ~375k kWh/d), not to trim normal day-to-day variance.
ABSOLUTE_CAP = 1e10
outliers = df[df["value"] > ABSOLUTE_CAP]
print(f"\nFound {len(outliers)} rows above the {ABSOLUTE_CAP:.0e} kWh/day sanity cap:")
print(outliers[["point_key","point_label","period_from","value"]].to_string(index=False))
df = df[df["value"] <= ABSOLUTE_CAP]
print(f"Rows after removing glitches: {len(df):,}")

# --- 4. Dedupe point-day combinations ---
before = len(df)
df = df.drop_duplicates()
print(f"\nDropped {before - len(df):,} exact duplicate rows")

# When both Provisional and Confirmed exist for the same point-day, keep Confirmed
df["flow_rank"] = df["flow_status"].map({"Confirmed": 1, "Provisional": 0}).fillna(0)
df = df.sort_values(["point_key","period_from","flow_rank"])
before = len(df)
df = df.drop_duplicates(subset=["point_key","period_from"], keep="last")
print(f"Dropped {before - len(df):,} Provisional rows superseded by a Confirmed value")
df = df.drop(columns=["flow_rank"])
print(f"Final row count: {len(df):,}")

# --- 5. Aggregate to national daily total ---
df["date"] = pd.to_datetime(df["period_from"], utc=True).dt.tz_convert("Europe/Berlin").dt.date
daily = df.groupby("date")["value"].sum().reset_index()
daily.columns = ["date", "entsog_exit_flow_kwh"]
daily["date"] = pd.to_datetime(daily["date"])

print(f"\nDaily national series: {len(daily)} days")
print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
print(f"\nSample:")
print(daily.head(3).to_string(index=False))
print(f"\nValue stats (kWh/day, national total):")
print(daily["entsog_exit_flow_kwh"].describe())

daily.to_csv("../data/entsog_germany_clean.csv", index=False)
print("\nSaved data/entsog_germany_clean.csv")
