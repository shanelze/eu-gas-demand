# European Natural Gas Demand Model — Germany

A data pipeline and gradient-boosted (XGBoost) model for German natural gas demand, built from public transmission, storage, and weather data. Combines a driver-analysis view (SHAP) with a walk-forward-validated forecast.

## Data sources (all free, no cost)

| Source | What | Access |
|---|---|---|
| [ENTSOG Transparency Platform](https://transparency.entsog.eu/) | Daily physical gas flow at ~130 German exit points (demand-side proxy) | Public REST API, no key. 5-year rolling retention. |
| [AGSI+ (GIE)](https://agsi.gie.eu/) | Daily underground storage levels, injection/withdrawal, storage-implied consumption | Free API, requires a free registered key |
| [Open-Meteo](https://open-meteo.com/) | Historical daily weather (Berlin), used to compute heating degree-days | Free, no key |

## Pipeline

```
src/
  fetch_entsog.py          # pulls ENTSOG exit-point flow data (entsog-py library)
  fetch_agsi.py             # pulls AGSI storage data
  fetch_weather.py          # pulls Open-Meteo weather data
  clean_entsog.py            # dedupe, drop dead points, remove outliers, aggregate to national daily total
  clean_agsi.py               # dedupe pagination overlap + provisional/confirmed revisions
  clean_weather.py            # fix DST double-shift bug, normalize dates
  merge_and_features.py        # merge all 3 sources, engineer leakage-safe features
  train_model.py                 # XGBoost + walk-forward validation + SHAP driver analysis

notebooks/
  01_data_cleaning_and_features.ipynb   # full cleaning walkthrough with before/after evidence

data/
  *_clean.csv                  # cleaned per-source daily series
  merged_daily.csv              # joined dataset before feature engineering
  model_dataset.csv              # final leakage-checked modeling dataset
```

## Key data quality findings

- **AGSI**: raw pull had a pagination bug producing ~15x duplicate rows (42,963 raw rows for what should be 2,763 daily observations), plus genuine same-day revisions as provisional values get confirmed over the following 1-2 days.
- **ENTSOG**: ~8% of requested exit points never reported real data and were dropped; one point (`FNC-00206`) had a single-day glitch spiking to 700,000x its typical value; remaining point-days had Provisional/Confirmed duplicate revisions, same pattern as AGSI.
- **Weather**: converting Open-Meteo's UTC-labeled timestamps back to local time and taking the date naively double-shifts across DST boundaries, producing one duplicate date per year at the October clock change. Fixed by normalizing the date directly instead of re-converting timezones.
- **Sanity check**: the cleaned demand series averages ~2,440 GWh/day, matching Germany's known annual consumption (~900 TWh/year ÷ 365 ≈ 2,460 GWh/day) almost exactly.
- **The data tells the 2022 crisis story on its own**: demand runs 5,000-7,000 GWh/day through 2021-2022, then drops to a structurally lower ~1,000-3,000 GWh/day baseline from 2023 onward — the well-documented price-driven demand destruction following the 2022 European gas supply shock.

## Methodology notes

- **ENTSOG's 5-year data retention** bounds the modeling window to Aug 2021 - Jul 2026 (~1,794 days after feature warm-up), even though AGSI/weather history goes back further.
- **Leakage check**: same-day storage and AGSI-consumption figures are excluded from the feature set (only D-1 lagged versions are used), since using today's own storage report to predict today's demand would be circular in a real forecasting setting. Weather is left same-day, since in production a forecast (not an observation) would be substituted for the target day.
- **Validation**: walk-forward (expanding-window) cross-validation, not a random train/test split, since a random split would leak future information into training for time series.

## Results

_Filled in after `train_model.py` is run (requires xgboost/scikit-learn/shap, not available in the environment this pipeline was scaffolded in — run locally)._

## Future work

- Extend to additional EU countries (France, Netherlands, Italy) for a cross-country comparison
- Add Eurostat's monthly sector-split data (`nrg_cb_gasm`) to decompose demand by end-use (industry / residential / power generation)
- Compare XGBoost against a SARIMA/Prophet baseline with proper backtesting
- Wrap in a Streamlit dashboard for interactive exploration
