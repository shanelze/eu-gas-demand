# fetch_agsi.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["AGSI_API_KEY"]

headers = {"x-key": api_key}
params = {"country": "DE", "from": "2019-01-01", "to": "2026-07-25", "size": 300}

all_rows = []
page = 1  # AGSI pagination starts at 1, not 0
last_page = None

while True:
    r = requests.get(
        "https://agsi.gie.eu/api",
        headers=headers,
        params={**params, "page": page},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    rows = data.get("data", [])
    last_page = data.get("last_page", page)

    print(f"page {page}/{last_page}: got {len(rows)} rows, status {r.status_code}")

    if not rows:
        break

    all_rows.extend(rows)

    if page >= last_page:
        break

    page += 1

df = pd.DataFrame(all_rows)
df.to_csv("agsi_germany.csv", index=False)
print(f"Saved {len(df)} rows to agsi_germany.csv")
