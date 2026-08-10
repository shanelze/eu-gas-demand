import pandas as pd

print("="*70)
print("AGSI STORAGE CLEANING")
print("="*70)

df = pd.read_csv("../data/agsi_germany.csv", low_memory=False)
print(f"\nRaw rows: {len(df):,}")

df["updatedAt"] = pd.to_datetime(df["updatedAt"])
df["gasDayStart"] = pd.to_datetime(df["gasDayStart"])

before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df):,} exact duplicate rows (pagination overlap)")

# AGSI revises provisional values for ~1-2 days after publication.
# Keep only the most recently updated row per gas day.
df = df.sort_values("updatedAt")
before = len(df)
df = df.drop_duplicates(subset="gasDayStart", keep="last")
print(f"Dropped {before - len(df):,} superseded revisions (kept latest updatedAt per day)")

df = df[["gasDayStart","gasInStorage","consumption","injection","withdrawal","full"]]
df.columns = ["date","storage_twh","agsi_consumption_gwh","injection_gwh","withdrawal_gwh","storage_pct_full"]
df = df.sort_values("date").reset_index(drop=True)

print(f"\nFinal row count: {len(df):,}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"\nSample:")
print(df.head(3).to_string(index=False))
print(f"\nNulls per column:")
print(df.isna().sum())

df.to_csv("../data/agsi_germany_clean.csv", index=False)
print("\nSaved data/agsi_germany_clean.csv")
