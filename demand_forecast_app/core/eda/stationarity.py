"""Stationarity tests and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    test_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    critical_values: dict
    interpretation: str


def adf_test(series: pd.Series, significance: float = 0.05) -> StationarityResult:
    """Augmented Dickey-Fuller test. Null: series has a unit root (non-stationary)."""
    valid = series.dropna()
    if len(valid) < 10:
        return StationarityResult(
            test_name="ADF",
            statistic=0,
            p_value=1,
            is_stationary=False,
            critical_values={},
            interpretation="Insufficient data for ADF test.",
        )

    result = adfuller(valid, autolag="AIC")
    stat, pval, _, _, crit, _ = result
    is_stat = pval < significance

    interp = (
        f"ADF statistic: {stat:.4f}, p-value: {pval:.4f}. "
        f"{'Reject' if is_stat else 'Fail to reject'} null hypothesis of unit root at {significance*100:.0f}% significance. "
        f"Series is {'likely stationary' if is_stat else 'likely non-stationary'}."
    )

    return StationarityResult(
        test_name="ADF",
        statistic=stat,
        p_value=pval,
        is_stationary=is_stat,
        critical_values=crit,
        interpretation=interp,
    )


def kpss_test(series: pd.Series, significance: float = 0.05) -> StationarityResult:
    """KPSS test. Null: series is stationary."""
    valid = series.dropna()
    if len(valid) < 10:
        return StationarityResult(
            test_name="KPSS",
            statistic=0,
            p_value=0,
            is_stationary=False,
            critical_values={},
            interpretation="Insufficient data for KPSS test.",
        )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, _, crit = kpss(valid, regression="c", nlags="auto")

    is_stat = pval > significance

    interp = (
        f"KPSS statistic: {stat:.4f}, p-value: {pval:.4f}. "
        f"{'Fail to reject' if is_stat else 'Reject'} null hypothesis of stationarity at {significance*100:.0f}% significance. "
        f"Series is {'likely stationary' if is_stat else 'likely non-stationary'}."
    )

    return StationarityResult(
        test_name="KPSS",
        statistic=stat,
        p_value=pval,
        is_stationary=is_stat,
        critical_values=crit,
        interpretation=interp,
    )


def suggest_differencing(series: pd.Series, max_d: int = 2) -> int:
    """Suggest the order of differencing needed for stationarity."""
    valid = series.dropna()
    for d in range(max_d + 1):
        result = adf_test(valid)
        if result.is_stationary:
            return d
        valid = valid.diff().dropna()
    return max_d
