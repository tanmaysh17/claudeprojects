"""Trend analysis: moving averages, polynomial fits, growth rates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend(series: pd.Series, window: int = 13) -> pd.Series:
    """Compute a centered moving average trend."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def fit_trend_line(series: pd.Series, degree: int = 1) -> tuple[np.ndarray, pd.Series]:
    """Fit a polynomial trend line. Returns (coefficients, fitted_values)."""
    valid = series.dropna()
    x = np.arange(len(valid))
    coeffs = np.polyfit(x, valid.values, degree)
    fitted = np.polyval(coeffs, x)
    return coeffs, pd.Series(fitted, index=valid.index, name="trend_fit")


def compute_growth_rates(series: pd.Series) -> dict[str, pd.Series]:
    """Compute various growth rate metrics."""
    results = {}

    # Week-over-week growth
    results["wow_growth"] = series.pct_change() * 100

    # Rolling 4-week growth (month-like)
    roll_4 = series.rolling(4).sum()
    results["rolling_4w_growth"] = roll_4.pct_change(4) * 100

    # Rolling 13-week growth (quarter-like)
    roll_13 = series.rolling(13).sum()
    results["rolling_13w_growth"] = roll_13.pct_change(13) * 100

    # Trailing 52-week total vs prior 52-week total
    roll_52 = series.rolling(52).sum()
    results["yoy_trailing_52w_growth"] = roll_52.pct_change(52) * 100

    # Week-level year-over-year growth
    results["yoy_weekly_growth"] = series.pct_change(52) * 100

    return results


def compute_rolling_totals(series: pd.Series) -> dict[str, pd.Series]:
    """Compute rolling totals for business metrics."""
    return {
        "rolling_4w_total": series.rolling(4, min_periods=1).sum(),
        "rolling_13w_total": series.rolling(13, min_periods=1).sum(),
        "rolling_26w_total": series.rolling(26, min_periods=1).sum(),
        "rolling_52w_total": series.rolling(52, min_periods=1).sum(),
    }


def compute_cagr(series: pd.Series, periods_per_year: int = 52) -> float | None:
    """Compute compound annual growth rate."""
    valid = series.dropna()
    if len(valid) < 2 or valid.iloc[0] <= 0 or valid.iloc[-1] <= 0:
        return None
    n_years = len(valid) / periods_per_year
    if n_years <= 0:
        return None
    cagr = (valid.iloc[-1] / valid.iloc[0]) ** (1 / n_years) - 1
    return float(cagr * 100)
