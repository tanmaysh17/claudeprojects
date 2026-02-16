"""Time series decomposition: STL and classical."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL, seasonal_decompose


@dataclass
class DecompositionResult:
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    method: str = "STL"


def decompose_stl(
    series: pd.Series,
    period: int = 52,
    robust: bool = True,
) -> DecompositionResult:
    """Decompose using STL (Seasonal and Trend decomposition using LOESS)."""
    valid = series.dropna()
    if len(valid) < 2 * period:
        # Fall back to classical for short series
        return decompose_classical(series, period, model="additive")

    stl = STL(valid, period=period, robust=robust)
    result = stl.fit()

    return DecompositionResult(
        observed=result.observed,
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        method="STL",
    )


def decompose_classical(
    series: pd.Series,
    period: int = 52,
    model: str = "additive",
) -> DecompositionResult:
    """Classical decomposition (additive or multiplicative)."""
    valid = series.dropna()

    # For multiplicative, all values must be positive
    if model == "multiplicative" and (valid <= 0).any():
        model = "additive"

    if len(valid) < 2 * period:
        period = max(2, len(valid) // 4)

    result = seasonal_decompose(valid, model=model, period=period, extrapolate_trend="freq")

    return DecompositionResult(
        observed=result.observed,
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        method=f"Classical ({model})",
    )
