"""Seasonality detection and analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 104,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Compute ACF and PACF with confidence bounds."""
    valid = series.dropna()
    nlags = min(nlags, len(valid) // 2 - 1)
    if nlags < 1:
        return {"acf": np.array([]), "pacf": np.array([]), "confint_acf": np.array([]), "confint_pacf": np.array([])}

    acf_vals, confint_acf = acf(valid, nlags=nlags, alpha=alpha)
    try:
        pacf_vals, confint_pacf = pacf(valid, nlags=min(nlags, len(valid) // 2 - 1), alpha=alpha)
    except Exception:
        pacf_vals = np.zeros(nlags + 1)
        confint_pacf = np.zeros((nlags + 1, 2))

    return {
        "acf": acf_vals,
        "pacf": pacf_vals,
        "confint_acf": confint_acf,
        "confint_pacf": confint_pacf,
    }


def detect_seasonal_period(series: pd.Series, max_period: int = 104) -> int:
    """Detect dominant seasonal period using ACF peaks."""
    valid = series.dropna()
    if len(valid) < 26:
        return 52  # default to annual for weekly data

    nlags = min(max_period, len(valid) // 2 - 1)
    if nlags < 10:
        return 52

    acf_vals = acf(valid, nlags=nlags)

    # Find local maxima in ACF beyond lag 2
    peaks = []
    for i in range(3, len(acf_vals) - 1):
        if acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1] and acf_vals[i] > 0.1:
            peaks.append((i, acf_vals[i]))

    if not peaks:
        return 52

    # Return the lag with highest ACF among plausible periods
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[0][0]


def seasonal_strength(trend: pd.Series, seasonal: pd.Series, residual: pd.Series) -> float:
    """Compute seasonal strength as 1 - Var(residual) / Var(seasonal + residual)."""
    seasonal_plus_resid = seasonal + residual
    var_sr = seasonal_plus_resid.var()
    var_r = residual.var()
    if var_sr == 0:
        return 0.0
    return max(0.0, 1.0 - var_r / var_sr)


def compute_seasonal_indices(series: pd.Series, period: int = 52) -> pd.Series:
    """Compute seasonal indices using ratio-to-moving-average method."""
    ma = series.rolling(window=period, center=True, min_periods=1).mean()
    ratio = series / ma.replace(0, np.nan)

    # Assign seasonal position (week of year for period=52)
    positions = np.arange(len(series)) % period
    indices = pd.Series(dtype=float, index=range(period))
    for pos in range(period):
        mask = positions == pos
        vals = ratio.iloc[mask].dropna()
        if len(vals) > 0:
            indices.iloc[pos] = vals.median()

    # Normalize so indices average to 1.0
    mean_idx = indices.mean()
    if mean_idx > 0:
        indices = indices / mean_idx

    return indices
