"""
run_all_experiments.py — orchestrator running all 5 model configurations
(ARIMA, XGBoost, LSTM, TCN, Hybrid TCN->XGBoost) across all 3 forecasting
horizons (1, 7, 30 days) for Poland CDS 5Y forecasting, and writing the
collected test-set metrics to a JSON/CSV results file.

This is intended for a LOCAL PILOT run (e.g. on a MacBook Pro) to sanity
check the full pipeline end-to-end on real data before submitting the
full-scale run to the CATO cluster. Results from this script populate a
separate "local pilot" table in paper.tex, clearly distinguished from the
placeholder table reserved for the CATO cluster run.

Usage:
    conda activate poprad_cato
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python run_all_experiments.py
"""
import argparse
import json
import os
import time

import pandas as pd

HORIZONS = [1, 7, 30]


def run_arima(train, val, test, horizon):
    from train_arima import run_horizon as arima_run_horizon
    return arima_run_horizon(train, val, test, horizon)


def run_xgboost(train, val, test, horizon):
    from train_xgboost import run_horizon as xgb_run_horizon
    return xgb_run_horizon(train, val, test, horizon)


def run_lstm_or_tcn(model_name, horizon, data_dir, epochs):
    from train_lstm_tcn_ddp import run_horizon as dl_run_horizon

    args = argparse.Namespace(
        model=model_name, horizon=horizon, data_dir=data_dir,
        lookback=60, batch_size=64, epochs=epochs, lr=1e-3,
        out_dir=os.path.join(os.path.dirname(__file__), "checkpoints"),
    )
    return dl_run_horizon(args)


def run_hybrid(horizon, data_dir, epochs):
    from train_hybrid import run_horizon as hybrid_run_horizon

    args = argparse.Namespace(
        horizon=horizon, data_dir=data_dir, lookback=60,
        batch_size=64, epochs=epochs, lr=1e-3,
    )
    return hybrid_run_horizon(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=os.path.join(os.path.dirname(__file__), "data")
    )
    parser.add_argument("--epochs", type=int, default=100,
                         help="Max epochs for LSTM/TCN/Hybrid (early stopping applies).")
    parser.add_argument(
        "--out-json",
        default=os.path.join(os.path.dirname(__file__), "results_local_pilot.json"),
    )
    args = parser.parse_args()

    train = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    test = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))

    all_results = {}
    timings = {}

    for h in HORIZONS:
        all_results[h] = {}
        timings[h] = {}

        for model_name, fn in [
            ("ARIMA", lambda: run_arima(train, val, test, h)),
            ("XGBoost", lambda: run_xgboost(train, val, test, h)),
            ("LSTM", lambda: run_lstm_or_tcn("lstm", h, args.data_dir, args.epochs)),
            ("TCN", lambda: run_lstm_or_tcn("tcn", h, args.data_dir, args.epochs)),
            ("Hybrid TCN-XGB", lambda: run_hybrid(h, args.data_dir, args.epochs)),
        ]:
            print(f"\n{'=' * 60}\nRunning {model_name}, horizon={h}\n{'=' * 60}")
            t0 = time.time()
            result = fn()
            elapsed = time.time() - t0
            all_results[h][model_name] = result
            timings[h][model_name] = round(elapsed, 1)
            print(f"[{model_name} h={h}] done in {elapsed:.1f}s: {result}")

    output = {"results": all_results, "timings_seconds": timings}
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nAll experiments complete. Results written to {args.out_json}")
    total_time = sum(t for h in timings.values() for t in h.values())
    print(f"Total wall-clock time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
