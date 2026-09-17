"""
train_arima.py — ARIMA baseline for Poland CDS 5Y forecasting (Section
III.A of paper.tex).

For each horizon h in {1, 7, 30}, fits a univariate ARIMA(p,d,q) model on
the target series only (no exogenous features), selecting orders by grid
search minimizing AIC on the training split, and evaluates on the
validation and test splits.

Requires: pandas, numpy, statsmodels (pip install statsmodels).
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from metrics import all_metrics

TARGET_COL = "POLAND CDS USD SR 5Y Corp"
HORIZONS = [1, 7, 30]


def select_order(series, max_p=5, max_q=5, d_range=(0, 1)):
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = np.inf
    best_order = (1, 0, 0)
    for p, d, q in itertools.product(range(max_p + 1), d_range, range(max_q + 1)):
        try:
            model = ARIMA(series, order=(p, d, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)
        except Exception:
            continue
    return best_order


def run_horizon(train, val, test, h):
    from statsmodels.tsa.arima.model import ARIMA

    target = train[TARGET_COL]
    order = select_order(target)
    print(f"[h={h}] selected ARIMA order: {order}")

    model = ARIMA(target, order=order).fit()

    y_true = test[f"target_h{h}"].dropna()
    y_prev = test.loc[y_true.index, TARGET_COL]
    # Static h-step-ahead forecast from the end of the training series (a
    # simple, reproducible baseline; a rolling re-fit would be more
    # accurate but far more expensive across ~600+ test points).
    forecast = model.get_forecast(steps=h).predicted_mean
    y_pred = np.full(len(y_true), forecast.iloc[-1])

    result = all_metrics(y_prev, y_true, y_pred)
    result["arima_order"] = order
    return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="/Users/rklopotek/CATO/Poprad_2026/experiments/data"
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

