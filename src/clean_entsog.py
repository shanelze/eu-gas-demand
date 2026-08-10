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

# --- 3. Remove implausible outlier readings ---
# A single interconnection/consumer point in Germany cannot plausibly flow
# more than 10 billion kWh in a single day (that alone would be ~3x the
# entire country's actual daily consumption). This threshold is deliberately
# generous -- it exists only to catch outliers like the one found during
# inspection (FNC-00206 spiking to 248.9 billion kWh/d vs its typical
# ~375k kWh/d), not to trim normal day-to-day variance. Root cause unknown
# (no public ENTSOG/operator notice found for it) -- flagged and removed
# based on the data pattern itself (a single day, ~700,000x that point's own
# typical value, back to normal immediately after), not a confirmed external
# explanation.
ABSOLUTE_CAP = 1e10
outliers = df[df["value"] > ABSOLUTE_CAP]
print(f"\nFound {len(outliers)} rows above the {ABSOLUTE_CAP:.0e} kWh/day sanity cap:")
print(outliers[["point_key","point_label","period_from","value"]].to_string(index=False))
df = df[df["value"] <= ABSOLUTE_CAP]
print(f"Rows after removing glitches: {len(df):,}")

# --- 3b. Catch smaller-magnitude outliers on Final Consumer points specifically ---
# The absolute cap above only catches truly enormous spikes. Smaller ones
# still distort the final consumer demand series -- found by inspection:
# FNC-00199 spiked to 4.1 billion kWh on 2025-11-17 (vs. its typical
# ~12-16 million kWh/day, a ~370x jump) and FNC-00030 spiked to 681 million
# on 2023-03-08 (vs. ~26 million typical). Both are large enough in
# absolute terms to visibly distort the national daily total. As with the
# cap above, the cause isn't independently confirmed (no public notice
# found) -- flagged purely because the value is wildly inconsistent with
# that point's own reporting history and reverts immediately the next day.
#
# A blanket ratio-to-median threshold doesn't work here: some consumer
# points (e.g. FNC-00045, DIS-00061) are small local networks with
# naturally high day-to-day *relative* swings that are real, not outliers --
# but because their absolute scale is tiny, those swings barely move the
# national total. So this only flags a point-day as an outlier if BOTH the
# ratio to that point's own median is extreme (>20x) AND the absolute
# excess is large enough to actually matter for the aggregate (>50,000
# MWh = 50 GWh) -- scoped to Final Consumer points only, since
# interconnection/storage points legitimately swing hard based on trading
# nominations and shouldn't be filtered this way.
consumer_mask = df["point_key"].astype(str).str.startswith(("FNC", "DIS"))
point_median = df.groupby("point_key")["value"].transform("median")
excess = df["value"] - point_median
ratio = df["value"] / point_median.replace(0, np.nan)
relative_outlier_mask = consumer_mask & (ratio > 20) & (excess > 50_000_000)

relative_outliers = df[relative_outlier_mask]
print(f"\nFound {len(relative_outliers)} additional Final Consumer point-days that are "
      f">20x that point's own median AND >50 GWh above it:")
print(relative_outliers[["point_key","point_label","period_from","value"]].to_string(index=False))
df = df[~relative_outlier_mask]
print(f"Rows after removing these: {len(df):,}")

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

# --- 5. Classify points and aggregate to a national daily series PER CATEGORY ---
# "Exit flow" from the German transmission network is not one homogeneous
# thing -- ENTSOG's own point labels split cleanly into:
#   FNC-* / DIS-*  "Final Consumers" / "Letztverbraucher" -- genuine domestic
#                  offtake (industrial + distribution to households)
#   UGS-*          Underground gas storage -- injection is booked as an
#                  "exit" from the transmission grid, and injection season
#                  runs April-October, i.e. the OPPOSITE seasonality of
#                  heating demand
#   ITP-*          Interconnection points -- cross-border transit/export to
#                  neighbouring countries, ~70% of total flow, driven by
#                  price arbitrage and transit routing, not German demand
# Summing all of these into one "demand" series (the original approach)
# produces a summer-peaked series because storage refill + transit exports
# dominate the total. Splitting them out gives a genuine, winter-peaked
# final-consumer demand series, plus the other categories as separate,
# still-useful context series (e.g. for a dashboard toggle).
df["date"] = pd.to_datetime(df["period_from"], utc=True).dt.tz_convert("Europe/Berlin").dt.date

def classify(point_key):
    if point_key.startswith(("FNC", "DIS")):
        return "consumer"
    if point_key.startswith("UGS"):
        return "storage"
    if point_key.startswith("ITP"):
        return "interconnection"
    return "other"

df["category"] = df["point_key"].astype(str).apply(classify)

print("\nUnique points per category:")
print(df.groupby("category")["point_key"].nunique())

pivot = df.groupby(["date", "category"])["value"].sum().unstack(fill_value=0)
pivot.columns = [f"{c}_exit_flow_kwh" for c in pivot.columns]
pivot["total_exit_flow_kwh"] = pivot[[c for c in pivot.columns]].sum(axis=1)
daily = pivot.reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

print(f"\nDaily national series: {len(daily)} days")
print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
print(f"Columns: {daily.columns.tolist()}")
print(f"\nSample:")
print(daily.head(3).to_string(index=False))

print(f"\nSanity check -- monthly average FINAL CONSUMER flow (GWh/day), should be winter-peaked:")
check = daily.copy()
check["month"] = check["date"].dt.month
monthly = (check.groupby("month")["consumer_exit_flow_kwh"].mean() / 1e6).round(1)
print(monthly)

daily.to_csv("../data/entsog_germany_clean.csv", index=False)
print("\nSaved data/entsog_germany_clean.csv")

# import pandas as pd
# import numpy as np

# print("="*70)
# print("ENTSOG CLEANING")
# print("="*70)

# df = pd.read_csv(
#     "../data/entsog_germany.csv",
#     usecols=["point_key","point_label","operator_key","direction_key",
#              "period_from","period_to","value","unit","flow_status"],
#     low_memory=False,
# )
# print(f"\nRaw rows: {len(df):,}")
# print(f"Unique points: {df['point_key'].nunique()}")

# # --- 1. Drop points that never report real data ---
# null_frac = df.groupby("point_key")["value"].apply(lambda s: s.isna().mean())
# all_null_points = null_frac[null_frac == 1.0].index.tolist()
# print(f"\nDropping {len(all_null_points)} points with 100% null values:")
# print(f"  {all_null_points}")
# df = df[~df["point_key"].isin(all_null_points)]
# print(f"Rows after dropping dead points: {len(df):,}")

# # --- 2. Drop remaining nulls (partial gaps) ---
# before = len(df)
# df = df.dropna(subset=["value"])
# print(f"\nDropped {before - len(df):,} remaining null rows (partial gaps)")

# # --- 3. Remove implausible sensor/reporting glitches ---
# # A single interconnection/consumer point in Germany cannot plausibly flow
# # more than 10 billion kWh in a single day (that alone would be ~3x the
# # entire country's actual daily consumption). This threshold is deliberately
# # generous -- it exists only to catch glitches like the one found during
# # inspection (FNC-00206 spiking to 248.9 billion kWh/d vs its typical
# # ~375k kWh/d), not to trim normal day-to-day variance.
# ABSOLUTE_CAP = 1e10
# outliers = df[df["value"] > ABSOLUTE_CAP]
# print(f"\nFound {len(outliers)} rows above the {ABSOLUTE_CAP:.0e} kWh/day sanity cap:")
# print(outliers[["point_key","point_label","period_from","value"]].to_string(index=False))
# df = df[df["value"] <= ABSOLUTE_CAP]
# print(f"Rows after removing glitches: {len(df):,}")

# # --- 4. Dedupe point-day combinations ---
# before = len(df)
# df = df.drop_duplicates()
# print(f"\nDropped {before - len(df):,} exact duplicate rows")

# # When both Provisional and Confirmed exist for the same point-day, keep Confirmed
# df["flow_rank"] = df["flow_status"].map({"Confirmed": 1, "Provisional": 0}).fillna(0)
# df = df.sort_values(["point_key","period_from","flow_rank"])
# before = len(df)
# df = df.drop_duplicates(subset=["point_key","period_from"], keep="last")
# print(f"Dropped {before - len(df):,} Provisional rows superseded by a Confirmed value")
# df = df.drop(columns=["flow_rank"])
# print(f"Final row count: {len(df):,}")

# # --- 5. Classify points and aggregate to a national daily series PER CATEGORY ---
# # "Exit flow" from the German transmission network is not one homogeneous
# # thing -- ENTSOG's own point labels split cleanly into:
# #   FNC-* / DIS-*  "Final Consumers" / "Letztverbraucher" -- genuine domestic
# #                  offtake (industrial + distribution to households)
# #   UGS-*          Underground gas storage -- injection is booked as an
# #                  "exit" from the transmission grid, and injection season
# #                  runs April-October, i.e. the OPPOSITE seasonality of
# #                  heating demand
# #   ITP-*          Interconnection points -- cross-border transit/export to
# #                  neighbouring countries, ~70% of total flow, driven by
# #                  price arbitrage and transit routing, not German demand
# # Summing all of these into one "demand" series (the original approach)
# # produces a summer-peaked series because storage refill + transit exports
# # dominate the total. Splitting them out gives a genuine, winter-peaked
# # final-consumer demand series, plus the other categories as separate,
# # still-useful context series (e.g. for a dashboard toggle).
# df["date"] = pd.to_datetime(df["period_from"], utc=True).dt.tz_convert("Europe/Berlin").dt.date

# def classify(point_key):
#     if point_key.startswith(("FNC", "DIS")):
#         return "consumer"
#     if point_key.startswith("UGS"):
#         return "storage"
#     if point_key.startswith("ITP"):
#         return "interconnection"
#     return "other"

# df["category"] = df["point_key"].astype(str).apply(classify)

# print("\nUnique points per category:")
# print(df.groupby("category")["point_key"].nunique())

# pivot = df.groupby(["date", "category"])["value"].sum().unstack(fill_value=0)
# pivot.columns = [f"{c}_exit_flow_kwh" for c in pivot.columns]
# pivot["total_exit_flow_kwh"] = pivot[[c for c in pivot.columns]].sum(axis=1)
# daily = pivot.reset_index()
# daily["date"] = pd.to_datetime(daily["date"])
# daily = daily.sort_values("date").reset_index(drop=True)

# print(f"\nDaily national series: {len(daily)} days")
# print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
# print(f"Columns: {daily.columns.tolist()}")
# print(f"\nSample:")
# print(daily.head(3).to_string(index=False))

# print(f"\nSanity check -- monthly average FINAL CONSUMER flow (GWh/day), should be winter-peaked:")
# check = daily.copy()
# check["month"] = check["date"].dt.month
# monthly = (check.groupby("month")["consumer_exit_flow_kwh"].mean() / 1e6).round(1)
# print(monthly)

# daily.to_csv("../data/entsog_germany_clean.csv", index=False)
# print("\nSaved data/entsog_germany_clean.csv")
