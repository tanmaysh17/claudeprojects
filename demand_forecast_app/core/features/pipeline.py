"""Feature engineering orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .lags import create_lag_features
from .rolling import create_rolling_features
from .calendar import create_calendar_features
from .holidays import create_holiday_features


@dataclass
class FeatureConfig:
    use_lags: bool = True
    lag_values: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 13, 26, 52])
    use_rolling: bool = True
    rolling_windows: list[int] = field(default_factory=lambda: [4, 13, 26, 52])
    rolling_agg_funcs: list[str] = field(default_factory=lambda: ["mean", "std"])
    use_calendar: bool = True
    use_holidays: bool = True
    holiday_calendar: dict | None = None
    use_fourier: bool = True  # included in calendar features


def build_feature_matrix(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    config: FeatureConfig | None = None,
    exog_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a complete feature matrix for ML models.

    Returns a DataFrame with the date column, target column, and all
    engineered features. Rows with NaN from lagging are NOT dropped;
    the caller should handle that.
    """
    if config is None:
        config = FeatureConfig()

    result = df[[date_col, target_col]].copy()
    series = df[target_col]

    # Lag features
    if config.use_lags:
        lags_df = create_lag_features(series, lags=config.lag_values)
        result = pd.concat([result, lags_df], axis=1)

    # Rolling features
    if config.use_rolling:
        rolling_df = create_rolling_features(
            series,
            windows=config.rolling_windows,
            agg_funcs=config.rolling_agg_funcs,
        )
        result = pd.concat([result, rolling_df], axis=1)

    # Calendar features
    if config.use_calendar:
        cal_df = create_calendar_features(df[date_col])
        result = pd.concat([result, cal_df], axis=1)

    # Holiday features
    if config.use_holidays:
        hol_df = create_holiday_features(df[date_col], holiday_calendar=config.holiday_calendar)
        result = pd.concat([result, hol_df], axis=1)

    # Exogenous columns pass-through
    if exog_cols:
        for col in exog_cols:
            if col in df.columns and col not in result.columns:
                result[col] = df[col].values

    return result


def get_feature_names(
    feature_matrix: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> list[str]:
    """Get list of feature column names (excluding date and target)."""
    return [c for c in feature_matrix.columns if c not in (date_col, target_col)]
