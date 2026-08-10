# train_quantile_models.py
#
# Trains 5 separate XGBoost quantile regressors (5th/25th/50th/75th/95th
# percentile of the STL residual), instead of a single point-prediction
# model. This gives each day a full predicted distribution rather than one
# number -- which is what the Monte Carlo simulation needs to sample from
# without collapsing every simulation of the same scenario into an
# identical outcome.
#
#   pip install xgboost scikit-learn statsmodels
#   python train_quantile_models.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_pinball_loss

df = pd.read_csv("../data/model_dataset.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in df.columns
                if c not in ["date", "demand_gwh", "stl_trend", "stl_seasonal", "stl_resid"]]
X = df[feature_cols]
y = df["stl_resid"]

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# Same train/holdout split as train_model.py, for a comparable check
TEST_SIZE = 60
n = len(df)
X_train, y_train = X.iloc[:n - TEST_SIZE], y.iloc[:n - TEST_SIZE]
X_test, y_test = X.iloc[n - TEST_SIZE:], y.iloc[n - TEST_SIZE:]

print(f"Training {len(QUANTILES)} quantile models on {len(X_train)} rows, "
      f"holding out the last {TEST_SIZE} days\n")

models = {}
for q in QUANTILES:
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=q,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    loss = mean_pinball_loss(y_test, preds, alpha=q)
    print(f"  q{q:.2f}: pinball loss = {loss:.2f}")
    models[q] = model
    model.save_model(f"../data/xgb_quantile_{int(q*100):02d}.json")

# Sanity check: quantiles should be properly ordered (q05 <= q25 <= ... <= q95)
# for every test row -- if they cross, something's wrong with training.
preds_by_q = np.column_stack([models[q].predict(X_test) for q in QUANTILES])
crossings = (np.diff(preds_by_q, axis=1) < 0).sum()
print(f"\nQuantile crossing check: {crossings} violations out of {preds_by_q.size} "
      f"comparisons (should be 0 or very close to it)")

print("\nSaved 5 quantile models to data/xgb_quantile_*.json")
