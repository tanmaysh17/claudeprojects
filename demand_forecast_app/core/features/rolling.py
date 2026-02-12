"""Rolling window features for ML models."""

from __future__ import annotations

import pandas as pd


def create_rolling_features(
    series: pd.Series,
    windows: list[int] | None = None,
    agg_funcs: list[str] | None = None,
) -> pd.DataFrame:
    """Create rolling window aggregate features."""
    if windows is None:
        windows = [4, 13, 26, 52]
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max"]

    result = pd.DataFrame(index=series.index)
    for w in windows:
        rolled = series.shift(1).rolling(w, min_periods=1)
        for func_name in agg_funcs:
            result[f"rolling_{w}w_{func_name}"] = getattr(rolled, func_name)()
    return result
