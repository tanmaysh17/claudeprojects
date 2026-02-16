"""Lag feature generation."""

from __future__ import annotations

import pandas as pd


def create_lag_features(series: pd.Series, lags: list[int] | None = None) -> pd.DataFrame:
    """Create lag features from a time series."""
    if lags is None:
        lags = [1, 2, 3, 4, 13, 26, 52]

    result = pd.DataFrame(index=series.index)
    for lag in lags:
        result[f"lag_{lag}"] = series.shift(lag)
    return result
