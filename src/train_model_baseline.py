# train_model.py
#
#   pip install xgboost scikit-learn shap matplotlib
#   python train_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("../data/model_dataset.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in ["date", "demand_gwh"]]
X = df[feature_cols]
y = df["demand_gwh"]

# --- Naive baselines ---
# "predict tomorrow = today" and "predict tomorrow = trailing 7-day average"
# are both already columns in the dataset (demand_lag1, demand_ma7), since
# they're also used as model features. Computing their own error against
# the actual target tells us how much of XGBoost's accuracy is genuinely
# coming from the extra features (weather, storage, calendar) versus just
# restating recent history -- important given demand_lag1/ma7/ma30 dominate
# the SHAP ranking, which is expected for an autocorrelated series but is
# not, by itself, evidence the model learned anything beyond persistence.
def eval_baseline(name, preds, actual):
    mae = mean_absolute_error(actual, preds)
    rmse = root_mean_squared_error(actual, preds)
    mape = mean_absolute_percentage_error(actual, preds) * 100
    print(f"  [{name}] MAE: {mae:.1f} GWh/day | RMSE: {rmse:.1f} GWh/day | MAPE: {mape:.2f}%")
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape}

# --- Walk-forward validation ---
N_SPLITS = 5
TEST_SIZE = 60

results = []
n = len(df)
fold_starts = [n - TEST_SIZE * (N_SPLITS - i) for i in range(N_SPLITS)]

print(f"Dataset: {n} rows, {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Running {N_SPLITS}-fold walk-forward validation ({TEST_SIZE}-day test windows)\n")

for i, test_start in enumerate(fold_starts):
    test_end = min(test_start + TEST_SIZE, n)
    train_idx = range(0, test_start)
    test_idx = range(test_start, test_end)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    train_dates = df["date"].iloc[train_idx]
    test_dates = df["date"].iloc[test_idx]
    print(f"Fold {i+1}: train {train_dates.min().date()}..{train_dates.max().date()} "
          f"({len(train_idx)} rows) -> test {test_dates.min().date()}..{test_dates.max().date()} "
          f"({len(test_idx)} rows)")

    xgb_result = eval_baseline("XGBoost", preds, y_test)

    # Naive baselines on the same fold, same test window
    naive_persist = df["demand_lag1"].iloc[test_idx]
    naive_ma7 = df["demand_ma7"].iloc[test_idx]
    persist_result = eval_baseline("Naive: yesterday", naive_persist, y_test)
    ma7_result = eval_baseline("Naive: 7-day avg", naive_ma7, y_test)
    print()

    for r in (xgb_result, persist_result, ma7_result):
        r["fold"] = i + 1
        results.append(r)

results_df = pd.DataFrame(results)
print("=== Mean across folds, by model ===")
print(results_df.groupby("model")[["mae", "rmse", "mape"]].mean().round(2))

# --- Final holdout: last 60 days ---
final_test_start = n - TEST_SIZE
X_train, y_train = X.iloc[:final_test_start], y.iloc[:final_test_start]
X_test, y_test = X.iloc[final_test_start:], y.iloc[final_test_start:]

final_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

print("\n=== Final holdout (last 60 days) ===")
eval_baseline("XGBoost", preds, y_test)
eval_baseline("Naive: yesterday", df["demand_lag1"].iloc[final_test_start:], y_test)
eval_baseline("Naive: 7-day avg", df["demand_ma7"].iloc[final_test_start:], y_test)

# Actual vs predicted vs naive baseline plot
plt.figure(figsize=(11, 4))
plt.plot(df["date"].iloc[final_test_start:], y_test.values, label="Actual", linewidth=1.2)
plt.plot(df["date"].iloc[final_test_start:], preds, label="XGBoost", linewidth=1.2, linestyle="--")
plt.plot(df["date"].iloc[final_test_start:], df["demand_lag1"].iloc[final_test_start:],
          label="Naive (yesterday)", linewidth=1, linestyle=":", alpha=0.7)
plt.title("Holdout: actual vs. predicted vs. naive baseline (last 60 days)")
plt.ylabel("GWh/day")
plt.legend()
plt.tight_layout()
plt.savefig("../notebooks/holdout_actual_vs_predicted.png", dpi=120)
print("\nSaved notebooks/holdout_actual_vs_predicted.png")

# --- SHAP: driver analysis ---
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("../notebooks/shap_summary.png", dpi=120)
print("Saved notebooks/shap_summary.png")

mean_abs_shap = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

print("\n=== Top demand drivers (mean |SHAP value|) ===")
print(mean_abs_shap.to_string(index=False))
mean_abs_shap.to_csv("../data/shap_feature_importance.csv", index=False)

results_df.to_csv("../data/validation_results.csv", index=False)
final_model.save_model("../data/xgb_model.json")
print("\nSaved model to data/xgb_model.json")