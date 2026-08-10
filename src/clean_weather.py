import pandas as pd

print("="*70)
print("WEATHER CLEANING")
print("="*70)

df = pd.read_csv("../data/weather_germany.csv")
print(f"\nRaw rows: {len(df):,}")

df["date_utc"] = pd.to_datetime(df["date"])

# The API call requested timezone=Europe/Berlin, so the returned timestamps
# already mark the start of each LOCAL day -- re-converting to Berlin time
# and taking .dt.date would double-shift across DST boundaries (confirmed
# during inspection: doing that produced 7 duplicate collisions, one per
# year, right at the October DST changeover). Just normalize the date part
# directly instead.
df["date"] = df["date_utc"].dt.tz_localize(None).dt.normalize().dt.date
df["date"] = pd.to_datetime(df["date"])

dupes = df["date"].duplicated().sum()
print(f"Duplicate dates after normalization: {dupes}")

df = df[["date","temperature_2m_mean","temperature_2m_min","temperature_2m_max","hdd"]]
df = df.sort_values("date").reset_index(drop=True)

print(f"\nFinal row count: {len(df):,}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"\nNulls per column:")
print(df.isna().sum())

df.to_csv("../data/weather_germany_clean.csv", index=False)
print("\nSaved data/weather_germany_clean.csv")
