# train_model.py
#
# Predicts the STL RESIDUAL, not raw demand -- stl_trend/stl_seasonal/
# stl_resid are excluded from the feature set (demand_gwh = their exact sum,
# so including any of them as an input feature would be leakage: the model
# would just learn to add them back up instead of learning anything real).
#
#   pip install xgboost scikit-learn shap statsmodels matplotlib
#   python train_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("../data/model_dataset.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in df.columns
                if c not in ["date", "demand_gwh", "stl_trend", "stl_seasonal", "stl_resid"]]
X = df[feature_cols]
y = df["stl_resid"]  # model predicts the RESIDUAL, not raw demand

def eval_baseline(name, predicted_demand, actual_demand):
    mae = mean_absolute_error(actual_demand, predicted_demand)
    rmse = root_mean_squared_error(actual_demand, predicted_demand)
    mape = mean_absolute_percentage_error(actual_demand, predicted_demand) * 100
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
    actual_demand = df["demand_gwh"].iloc[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    resid_preds = model.predict(X_test)

    # Reconstruct actual demand: trend + seasonal + predicted residual
    predicted_demand = (df["stl_trend"].iloc[test_idx].values
                         + df["stl_seasonal"].iloc[test_idx].values
                         + resid_preds)

    train_dates = df["date"].iloc[train_idx]
    test_dates = df["date"].iloc[test_idx]
    print(f"Fold {i+1}: train {train_dates.min().date()}..{train_dates.max().date()} "
          f"({len(train_idx)} rows) -> test {test_dates.min().date()}..{test_dates.max().date()} "
          f"({len(test_idx)} rows)")

    xgb_result = eval_baseline("XGBoost (trend+seasonal+resid)", predicted_demand, actual_demand)

    # Naive baselines, all evaluated in actual GWh terms
    seasonal_only = df["stl_trend"].iloc[test_idx] + df["stl_seasonal"].iloc[test_idx]
    naive_persist = df["demand_gwh"].shift(1).iloc[test_idx]
    naive_ma7 = df["demand_gwh"].shift(1).rolling(7).mean().iloc[test_idx]

    seasonal_result = eval_baseline("Naive: seasonal baseline only", seasonal_only, actual_demand)
    persist_result = eval_baseline("Naive: yesterday", naive_persist, actual_demand)
    ma7_result = eval_baseline("Naive: 7-day avg", naive_ma7, actual_demand)
    print()

    for r in (xgb_result, seasonal_result, persist_result, ma7_result):
        r["fold"] = i + 1
        results.append(r)

results_df = pd.DataFrame(results)
print("=== Mean across folds, by model ===")
print(results_df.groupby("model")[["mae", "rmse", "mape"]].mean().round(2))

# --- Final holdout: last 60 days ---
final_test_start = n - TEST_SIZE
X_train, y_train = X.iloc[:final_test_start], y.iloc[:final_test_start]
X_test = X.iloc[final_test_start:]
actual_demand = df["demand_gwh"].iloc[final_test_start:]

final_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
final_model.fit(X_train, y_train)
resid_preds = final_model.predict(X_test)
predicted_demand = (df["stl_trend"].iloc[final_test_start:].values
                     + df["stl_seasonal"].iloc[final_test_start:].values
                     + resid_preds)

print("\n=== Final holdout (last 60 days) ===")
eval_baseline("XGBoost", predicted_demand, actual_demand)
eval_baseline("Naive: seasonal baseline only",
              df["stl_trend"].iloc[final_test_start:] + df["stl_seasonal"].iloc[final_test_start:],
              actual_demand)
eval_baseline("Naive: yesterday", df["demand_gwh"].shift(1).iloc[final_test_start:], actual_demand)
eval_baseline("Naive: 7-day avg", df["demand_gwh"].shift(1).rolling(7).mean().iloc[final_test_start:], actual_demand)

plt.figure(figsize=(11, 4))
plt.plot(df["date"].iloc[final_test_start:], actual_demand.values, label="Actual", linewidth=1.2)
plt.plot(df["date"].iloc[final_test_start:], predicted_demand, label="XGBoost (trend+seasonal+resid)",
          linewidth=1.2, linestyle="--")
plt.plot(df["date"].iloc[final_test_start:],
          (df["stl_trend"].iloc[final_test_start:] + df["stl_seasonal"].iloc[final_test_start:]).values,
          label="Seasonal baseline only", linewidth=1, linestyle=":", alpha=0.7)
plt.title("Holdout: actual vs. predicted vs. seasonal baseline (last 60 days)")
plt.ylabel("GWh/day")
plt.legend()
plt.tight_layout()
plt.savefig("../notebooks/holdout_actual_vs_predicted.png", dpi=120)
print("\nSaved notebooks/holdout_actual_vs_predicted.png")

# --- SHAP: driver analysis on the RESIDUAL ---
# This explains what drives deviations from the expected seasonal pattern,
# not the raw demand level -- a more genuine "driver" story, since the
# obvious seasonal effect no longer soaks up all the credit.
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

print("\n=== Top drivers of deviation from seasonal norm (mean |SHAP value|) ===")
print(mean_abs_shap.to_string(index=False))
mean_abs_shap.to_csv("../data/shap_feature_importance.csv", index=False)

results_df.to_csv("../data/validation_results.csv", index=False)
final_model.save_model("../data/xgb_model.json")
print("\nSaved model to data/xgb_model.json")

# # train_model.py
# #
# # Run this locally -- xgboost/scikit-learn/shap aren't installable in the
# # sandbox this project was scaffolded in, so this step was written but never
# # executed on the pipeline's end. Everything upstream (data/model_dataset.csv)
# # is already real, cleaned, leakage-checked data.
# #
# #   pip install xgboost scikit-learn shap matplotlib
# #   python train_model.py

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
# import shap
# import matplotlib.pyplot as plt

# df = pd.read_csv("../data/model_dataset.csv", parse_dates=["date"])
# df = df.sort_values("date").reset_index(drop=True)

# feature_cols = [c for c in df.columns if c not in ["date", "demand_gwh"]]
# X = df[feature_cols]
# y = df["demand_gwh"]

# # --- Walk-forward validation ---
# # Standard random train/test splits leak future information into the past
# # for time series. Instead: train on an expanding window, always test on the
# # NEXT block the model hasn't seen. 5 folds, each testing ~60 days.
# N_SPLITS = 5
# TEST_SIZE = 60  # days per fold

# results = []
# n = len(df)
# fold_starts = [n - TEST_SIZE * (N_SPLITS - i) for i in range(N_SPLITS)]

# print(f"Dataset: {n} rows, {df['date'].min().date()} to {df['date'].max().date()}")
# print(f"Running {N_SPLITS}-fold walk-forward validation ({TEST_SIZE}-day test windows)\n")

# for i, test_start in enumerate(fold_starts):
#     test_end = test_start + TEST_SIZE
#     if test_end > n:
#         test_end = n

#     train_idx = range(0, test_start)
#     test_idx = range(test_start, test_end)

#     X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#     X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

#     model = xgb.XGBRegressor(
#         n_estimators=500,
#         max_depth=5,
#         learning_rate=0.03,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#     )
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     mae = mean_absolute_error(y_test, preds)
#     rmse = root_mean_squared_error(y_test, preds)
#     mape = mean_absolute_percentage_error(y_test, preds) * 100

#     train_dates = df["date"].iloc[train_idx]
#     test_dates = df["date"].iloc[test_idx]
#     print(f"Fold {i+1}: train {train_dates.min().date()}..{train_dates.max().date()} "
#           f"({len(train_idx)} rows) -> test {test_dates.min().date()}..{test_dates.max().date()} "
#           f"({len(test_idx)} rows)")
#     print(f"  MAE: {mae:.1f} GWh/day | RMSE: {rmse:.1f} GWh/day | MAPE: {mape:.2f}%\n")

#     results.append({"fold": i+1, "mae": mae, "rmse": rmse, "mape": mape})

# results_df = pd.DataFrame(results)
# print("=== Summary across folds ===")
# print(results_df.describe().loc[["mean", "std"]])

# # --- Final model: train on everything except the last 60 days, for SHAP + a holdout plot ---
# final_test_start = n - TEST_SIZE
# X_train, y_train = X.iloc[:final_test_start], y.iloc[:final_test_start]
# X_test, y_test = X.iloc[final_test_start:], y.iloc[final_test_start:]

# final_model = xgb.XGBRegressor(
#     n_estimators=500, max_depth=5, learning_rate=0.03,
#     subsample=0.8, colsample_bytree=0.8, random_state=42,
# )
# final_model.fit(X_train, y_train)
# preds = final_model.predict(X_test)

# print("\n=== Final holdout (last 60 days) ===")
# print(f"MAE: {mean_absolute_error(y_test, preds):.1f} GWh/day")
# print(f"RMSE: {root_mean_squared_error(y_test, preds):.1f} GWh/day")
# print(f"MAPE: {mean_absolute_percentage_error(y_test, preds) * 100:.2f}%")

# # Actual vs predicted plot
# plt.figure(figsize=(11, 4))
# plt.plot(df["date"].iloc[final_test_start:], y_test.values, label="Actual", linewidth=1.2)
# plt.plot(df["date"].iloc[final_test_start:], preds, label="Predicted", linewidth=1.2, linestyle="--")
# plt.title("Holdout: actual vs. predicted daily gas demand (last 60 days)")
# plt.ylabel("GWh/day")
# plt.legend()
# plt.tight_layout()
# plt.savefig("../notebooks/holdout_actual_vs_predicted.png", dpi=120)
# print("\nSaved notebooks/holdout_actual_vs_predicted.png")

# # --- SHAP: driver analysis ---
# print("\nComputing SHAP values...")
# explainer = shap.TreeExplainer(final_model)
# shap_values = explainer(X_test)

# plt.figure()
# shap.summary_plot(shap_values, X_test, show=False)
# plt.tight_layout()
# plt.savefig("../notebooks/shap_summary.png", dpi=120)
# print("Saved notebooks/shap_summary.png")

# # Feature importance table (mean absolute SHAP value per feature)
# mean_abs_shap = pd.DataFrame({
#     "feature": feature_cols,
#     "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
# }).sort_values("mean_abs_shap", ascending=False)

# print("\n=== Top demand drivers (mean |SHAP value|) ===")
# print(mean_abs_shap.to_string(index=False))
# mean_abs_shap.to_csv("../data/shap_feature_importance.csv", index=False)

# final_model.save_model("../data/xgb_model.json")
# print("\nSaved model to data/xgb_model.json")
