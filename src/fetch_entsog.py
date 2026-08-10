# fetch_entsog.py
from entsog import EntsogPandasClient
import pandas as pd

client = EntsogPandasClient()

# ENTSOG's live API only retains a rolling ~5 years of data (older data is
# archived separately and not accessible via this endpoint). Keeping this
# window inside 5 years avoids "data too old" errors -- and conveniently
# still covers the 2022 European gas crisis, the most demand-relevant period.
start = pd.Timestamp("2021-07-28", tz="Europe/Brussels")
end = pd.Timestamp("2026-07-25", tz="Europe/Brussels")

print("Looking up German operator point directions...")
points = client.query_operator_point_directions()

# Exit points = gas leaving the German network (our demand-side proxy).
# has_data=True filters to TSOs actually publishing REG715 data.
de_exit = points[
    (points["t_so_country"] == "DE")
    & (points["direction_key"] == "exit")
    & (points["has_data"] == True)
]
print(f"Found {len(de_exit)} German exit point-directions")

point_directions = [
    f"{row.operator_key}{row.point_key}{row.direction_key}"
    for row in de_exit.itertuples()
]

print("Fetching physical flow data for these points, 2021-2026...")
print("(sending in batches of 20 -- all 174 points in one request blows past the URL length limit)")

BATCH_SIZE = 20
all_dfs = []
n_batches = -(-len(point_directions) // BATCH_SIZE)

for i in range(0, len(point_directions), BATCH_SIZE):
    batch = point_directions[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    print(f"Batch {batch_num}/{n_batches} ({len(batch)} points)...", end=" ")
    try:
        df_batch = client.query_operational_point_data(
            start=start,
            end=end,
            point_directions=batch,
            indicators=["physical_flow"],
            verbose=False,
        )
        print(f"got {len(df_batch)} rows")
        all_dfs.append(df_batch)
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        body = getattr(getattr(e, "response", None), "text", None)
        print(f"FAILED: {type(e).__name__} status={status}")
        if body:
            print("  response body (first 500 chars):", body[:500])

df = pd.concat(all_dfs)
print(f"\nTotal rows: {len(df)}")
df.to_csv("entsog_germany.csv")
print("Saved entsog_germany.csv")
