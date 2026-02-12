"""Rolling statistics computation."""

from __future__ import annotations

import pandas as pd


def compute_rolling_statistics(
    series: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling mean, std, min, max for multiple window sizes."""
    if windows is None:
        windows = [4, 13, 26, 52]

    results = pd.DataFrame(index=series.index)
    for w in windows:
        results[f"rolling_{w}w_mean"] = series.rolling(w, min_periods=1).mean()
        results[f"rolling_{w}w_std"] = series.rolling(w, min_periods=1).std()
        results[f"rolling_{w}w_min"] = series.rolling(w, min_periods=1).min()
        results[f"rolling_{w}w_max"] = series.rolling(w, min_periods=1).max()

    return results
