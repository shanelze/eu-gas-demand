# main.py -- runs the full pipeline in order, from cleaned-but-unmerged raw
# data through to a ready-to-launch dashboard. Doesn't include the fetch_*.py
# scripts (those only need to be re-run when you want fresh raw data, not
# every time you re-process it) or the Streamlit app itself (that's a
# long-running server, not a one-shot step -- run it separately after this
# finishes).
#
#   python main.py
#
# Stops immediately if any step fails, so you don't waste time training a
# model on a broken merge.

import subprocess
import sys
import time

STEPS = [
    ("clean_entsog.py", "Clean ENTSOG flow data, split final-consumer/storage/interconnection"),
    ("clean_agsi.py", "Clean AGSI+ storage data"),
    ("clean_weather.py", "Clean Open-Meteo weather data"),
    ("merge_and_features.py", "Merge sources, STL decomposition, feature engineering"),
    ("train_model.py", "Train XGBoost point-prediction model, walk-forward validation, SHAP"),
    ("train_quantile_models.py", "Train 5 XGBoost quantile models for Monte Carlo"),
    ("monte_carlo_weather.py", "Run 1,000-simulation Monte Carlo forecast"),
]

start = time.time()
for i, (script, description) in enumerate(STEPS, start=1):
    print("\n" + "=" * 70)
    print(f"STEP {i}/{len(STEPS)}: {script}")
    print(f"  {description}")
    print("=" * 70)

    step_start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - step_start

    if result.returncode != 0:
        print(f"\nFAILED at step {i}/{len(STEPS)} ({script}) after {elapsed:.1f}s -- stopping.")
        sys.exit(1)

    print(f"\n[{script} done in {elapsed:.1f}s]")

total = time.time() - start
print("\n" + "=" * 70)
print(f"Pipeline complete in {total:.1f}s ({total/60:.1f} min)")
print("=" * 70)
print("\nNext: launch the dashboard with:")
print("  streamlit run app.py")
