"""
metrics.py — shared evaluation metrics for all forecasting models
(Section VI.C of paper.tex): RMSE, MAE, MAPE, directional accuracy.
"""
import numpy as np


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def directional_accuracy(y_prev, y_true, y_pred):
    y_prev, y_true, y_pred = np.asarray(y_prev), np.asarray(y_true), np.asarray(y_pred)
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return float(np.mean(true_dir == pred_dir))


def all_metrics(y_prev, y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_prev, y_true, y_pred),
        "n": int(len(np.asarray(y_true))),
    }
