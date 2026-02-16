"""Calendar and date-based features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Create calendar features from date series."""
    dt = pd.to_datetime(dates)
    result = pd.DataFrame(index=dates.index)

    result["week_of_year"] = dt.dt.isocalendar().week.astype(int).values
    result["month"] = dt.dt.month
    result["quarter"] = dt.dt.quarter
    result["year"] = dt.dt.year
    result["day_of_year"] = dt.dt.dayofyear
    result["is_month_start"] = (dt.dt.day <= 7).astype(int)
    result["is_month_end"] = (dt.dt.day >= 24).astype(int)
    result["is_quarter_start"] = ((dt.dt.month.isin([1, 4, 7, 10])) & (dt.dt.day <= 7)).astype(int)
    result["is_quarter_end"] = ((dt.dt.month.isin([3, 6, 9, 12])) & (dt.dt.day >= 24)).astype(int)
    result["is_year_start"] = ((dt.dt.month == 1) & (dt.dt.day <= 7)).astype(int)
    result["is_year_end"] = ((dt.dt.month == 12) & (dt.dt.day >= 24)).astype(int)

    # Fourier terms for weekly seasonality (annual cycle)
    week_frac = result["week_of_year"] / 52.0
    for k in range(1, 4):  # 3 harmonics
        result[f"sin_{k}_annual"] = np.sin(2 * np.pi * k * week_frac)
        result[f"cos_{k}_annual"] = np.cos(2 * np.pi * k * week_frac)

    return result
