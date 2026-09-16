"""
train_xgboost.py — XGBoost baseline on lagged features for Poland CDS 5Y
forecasting (Section III.B of paper.tex).

For each horizon h in {1, 7, 30}, trains an XGBoost regressor on the full
lagged feature matrix produced by prepare_data.py, tunes hyperparameters
via grid search on the validation split, and reports RMSE/MAE/MAPE/
directional accuracy on the held-out test split.

Requires: pandas, numpy, xgboost, scikit-learn.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from metrics import all_metrics, rmse

TARGET_COL = "POLAND CDS USD SR 5Y Corp"
HORIZONS = [1, 7, 30]

PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 500],
    "subsample": [0.8, 1.0],
}


def get_feature_cols(df):
    return [c for c in df.columns if "_lag" in c]


def run_horizon(train, val, test, h):
    import xgboost as xgb

    feature_cols = get_feature_cols(train)
    target_col = f"target_h{h}"

    train_valid = train.dropna(subset=feature_cols + [target_col])
    val_valid = val.dropna(subset=feature_cols + [target_col])
    test_valid = test.dropna(subset=feature_cols + [target_col])

    X_train, y_train = train_valid[feature_cols], train_valid[target_col]
    X_val, y_val = val_valid[feature_cols], val_valid[target_col]
    X_test, y_test = test_valid[feature_cols], test_valid[target_col]

    best_rmse = np.inf
    best_params = None
    for params in ParameterGrid(PARAM_GRID):
        model = xgb.XGBRegressor(objective="reg:squarederror", **params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_rmse = rmse(y_val, val_pred)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = params

    print(f"[h={h}] best params: {best_params}, val RMSE={best_rmse:.4f}")

    final_model = xgb.XGBRegressor(objective="reg:squarederror", **best_params)
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_pred = final_model.predict(X_test)

    y_prev = test_valid[TARGET_COL]
    result = all_metrics(y_prev, y_test, test_pred)
    result["best_params"] = best_params
    return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="/home/jkanturski/CATO/experiments/data"
    )
    args = parser.parse_args()

    train = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    test = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))

    for h in HORIZONS:
        result = run_horizon(train, val, test, h)
        print(f"\n=== Horizon {h} day(s) (test set) ===")
        print(result)


if __name__ == "__main__":
    main()
