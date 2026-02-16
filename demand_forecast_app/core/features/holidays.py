"""Holiday feature generation."""

from __future__ import annotations

import pandas as pd
import numpy as np

# Built-in US holidays (approximate dates, week-level)
DEFAULT_HOLIDAYS = {
    "new_year": {"month": 1, "day_range": (1, 7)},
    "valentines": {"month": 2, "day_range": (8, 18)},
    "easter": {"month": 4, "day_range": (1, 21)},  # approximate
    "memorial_day": {"month": 5, "day_range": (24, 31)},
    "independence_day": {"month": 7, "day_range": (1, 7)},
    "labor_day": {"month": 9, "day_range": (1, 7)},
    "halloween": {"month": 10, "day_range": (25, 31)},
    "thanksgiving": {"month": 11, "day_range": (20, 30)},
    "christmas": {"month": 12, "day_range": (18, 31)},
    "black_friday": {"month": 11, "day_range": (23, 30)},
}


def create_holiday_features(
    dates: pd.Series,
    holiday_calendar: dict | None = None,
    custom_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Create binary holiday indicator features."""
    dt = pd.to_datetime(dates)
    holidays = holiday_calendar or DEFAULT_HOLIDAYS
    result = pd.DataFrame(index=dates.index)

    for name, spec in holidays.items():
        month = spec["month"]
        d_start, d_end = spec["day_range"]
        result[f"holiday_{name}"] = (
            (dt.dt.month == month) & (dt.dt.day >= d_start) & (dt.dt.day <= d_end)
        ).astype(int)

    # Aggregate: any holiday active
    holiday_cols = [c for c in result.columns if c.startswith("holiday_")]
    if holiday_cols:
        result["is_holiday_period"] = (result[holiday_cols].sum(axis=1) > 0).astype(int)

    # Custom holiday dates
    if custom_dates:
        custom_dt = pd.to_datetime(custom_dates)
        for i, cd in enumerate(custom_dt):
            week_start = cd - pd.Timedelta(days=cd.weekday())
            week_end = week_start + pd.Timedelta(days=6)
            result[f"custom_holiday_{i}"] = ((dt >= week_start) & (dt <= week_end)).astype(int)

    return result
