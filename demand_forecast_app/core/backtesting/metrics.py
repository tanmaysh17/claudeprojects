"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (undefined when actual=0)."""
    mask = actual != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(actual) + np.abs(predicted))
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def mase(actual: np.ndarray, predicted: np.ndarray, seasonal_period: int = 52) -> float:
    """Mean Absolute Scaled Error (relative to seasonal naive)."""
    n = len(actual)
    if n <= seasonal_period:
        # Fall back to naive (1-step) baseline
        naive_errors = np.abs(np.diff(actual))
    else:
        naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])

    naive_mae = np.mean(naive_errors)
    if naive_mae == 0:
        return float("inf")

    forecast_mae = np.mean(np.abs(actual - predicted))
    return float(forecast_mae / naive_mae)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean bias (positive = over-forecasting)."""
    return float(np.mean(predicted - actual))


def coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Prediction interval coverage: fraction of actuals within bounds."""
    within = (actual >= lower) & (actual <= upper)
    return float(np.mean(within) * 100)


METRIC_FUNCTIONS = {
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "sMAPE": smape,
    "MASE": mase,
    "Bias": bias,
}

METRIC_DISPLAY_NAMES = {
    "MAE": "Mean Absolute Error",
    "RMSE": "Root Mean Squared Error",
    "MAPE": "Mean Absolute % Error",
    "sMAPE": "Symmetric Mean Absolute % Error",
    "MASE": "Mean Absolute Scaled Error",
    "Bias": "Mean Bias",
}


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute multiple metrics at once."""
    if metric_names is None:
        metric_names = list(METRIC_FUNCTIONS.keys())

    results = {}
    for name in metric_names:
        func = METRIC_FUNCTIONS.get(name)
        if func:
            try:
                results[name] = func(actual, predicted)
            except Exception:
                results[name] = float("nan")
    return results
