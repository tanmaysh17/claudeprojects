"""Data preprocessing: frequency inference, resampling, missing value handling."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .column_mapping import ColumnMapping


def prepare_time_series(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    fill_method: str = "interpolate",
) -> pd.DataFrame:
    """Clean and prepare the dataframe as a weekly time series."""
    date_col = mapping.date_col
    target_col = mapping.target_col

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")

    # Drop full duplicates on date, keeping last
    out = out.drop_duplicates(subset=[date_col], keep="last")
    out = out.sort_values(date_col).reset_index(drop=True)

    # Set date as index for resampling
    out = out.set_index(date_col)

    # Detect frequency
    freq = infer_frequency(out.index)

    # Resample to fill gaps
    if freq:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in out.columns if c not in numeric_cols]

        resampled = out[numeric_cols].resample(freq).mean()

        # Bring back non-numeric columns via forward fill
        if non_numeric_cols:
            non_num = out[non_numeric_cols].resample(freq).first()
            non_num = non_num.ffill()
            resampled = pd.concat([resampled, non_num], axis=1)

        out = resampled

    # Handle missing target values
    if fill_method == "interpolate":
        out[target_col] = out[target_col].interpolate(method="linear", limit_direction="both")
    elif fill_method == "ffill":
        out[target_col] = out[target_col].ffill().bfill()
    elif fill_method == "zero":
        out[target_col] = out[target_col].fillna(0)

    # Fill remaining NaN in exog columns
    for col in mapping.exog_cols:
        if col in out.columns:
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
            out[col] = out[col].fillna(0)

    out = out.reset_index()
    return out


def infer_frequency(date_index: pd.DatetimeIndex) -> str | None:
    """Infer time series frequency from a DatetimeIndex."""
    if len(date_index) < 3:
        return None

    try:
        freq = pd.infer_freq(date_index)
        if freq:
            return freq
    except (ValueError, TypeError):
        pass

    # Fallback: compute median difference
    diffs = pd.Series(date_index).diff().dropna()
    median_days = diffs.dt.days.median()

    if 5 <= median_days <= 9:
        return "W"
    elif 27 <= median_days <= 33:
        return "MS"
    elif 85 <= median_days <= 95:
        return "QS"
    elif 1 <= median_days <= 2:
        return "D"
    else:
        return "W"  # Default to weekly for this application
