"""Seasonality index computation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_seasonal_indices(
    series: pd.Series,
    period: int = 52,
) -> pd.Series:
    """Compute seasonal indices using ratio-to-moving-average.

    Returns a Series of length `period` with the average seasonal index
    for each position in the cycle.
    """
    ma = series.rolling(window=period, center=True, min_periods=period // 2).mean()
    ratio = series / ma.replace(0, np.nan)

    positions = np.arange(len(series)) % period
    indices = pd.Series(dtype=float, index=range(period))

    for pos in range(period):
        mask = positions == pos
        vals = ratio.iloc[mask].dropna()
        if len(vals) > 0:
            indices.iloc[pos] = vals.median()

    # Normalize so indices average to 1.0
    mean_idx = indices.mean()
    if mean_idx > 0 and not np.isnan(mean_idx):
        indices = indices / mean_idx

    # Fill any remaining NaN with 1.0
    indices = indices.fillna(1.0)

    return indices


def apply_seasonal_indices(
    series: pd.Series,
    indices: pd.Series,
    period: int = 52,
) -> pd.Series:
    """Deseasonalize a series using precomputed indices."""
    positions = np.arange(len(series)) % period
    idx_values = indices.values[positions]
    idx_values = np.where(idx_values == 0, 1.0, idx_values)
    return series / idx_values
