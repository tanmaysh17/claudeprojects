"""Outlier detection methods."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method. Returns boolean mask."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method. Returns boolean mask."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean).abs() / std
    return z > threshold


def detect_rolling_zscore(series: pd.Series, window: int = 13, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using rolling Z-score (better for trended data)."""
    rolling_mean = series.rolling(window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window, center=True, min_periods=1).std()
    rolling_std = rolling_std.replace(0, np.nan)
    z = (series - rolling_mean).abs() / rolling_std
    return z.fillna(0) > threshold


def get_outlier_summary(
    series: pd.Series,
    dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Get a summary of detected outliers across methods."""
    results = pd.DataFrame(index=series.index)
    results["value"] = series
    results["iqr_outlier"] = detect_iqr(series)
    results["zscore_outlier"] = detect_zscore(series)
    results["rolling_zscore_outlier"] = detect_rolling_zscore(series)
    results["n_methods_flagged"] = (
        results["iqr_outlier"].astype(int)
        + results["zscore_outlier"].astype(int)
        + results["rolling_zscore_outlier"].astype(int)
    )
    if dates is not None:
        results["date"] = dates.values

    return results[results["n_methods_flagged"] > 0].sort_values("n_methods_flagged", ascending=False)
