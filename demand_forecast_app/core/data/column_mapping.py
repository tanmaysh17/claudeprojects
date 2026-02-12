"""Auto-detect and manual mapping of date, target, and exogenous columns."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnMapping:
    date_col: str | None = None
    target_col: str | None = None
    exog_cols: list[str] = field(default_factory=list)
    group_col: str | None = None


_DATE_HINTS = [
    "date", "week", "week_start", "week_date", "period", "ds",
    "time", "timestamp", "week_ending", "week_beginning",
]
_TARGET_HINTS = [
    "sales", "demand", "quantity", "units", "volume", "revenue",
    "orders", "y", "value", "amount", "target",
]


def auto_detect_columns(df: pd.DataFrame) -> ColumnMapping:
    mapping = ColumnMapping()
    cols_lower = {c: c.strip().lower().replace(" ", "_") for c in df.columns}

    # Detect date column
    for col, norm in cols_lower.items():
        if norm in _DATE_HINTS:
            mapping.date_col = col
            break
    if mapping.date_col is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(20))
                mapping.date_col = col
                break
            except (ValueError, TypeError):
                continue

    # Detect target column
    for col, norm in cols_lower.items():
        if col == mapping.date_col:
            continue
        if norm in _TARGET_HINTS:
            mapping.target_col = col
            break
    if mapping.target_col is None:
        for col in df.columns:
            if col == mapping.date_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                mapping.target_col = col
                break

    # Detect exogenous numeric columns
    for col in df.columns:
        if col in (mapping.date_col, mapping.target_col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mapping.exog_cols.append(col)

    # Detect possible group column (low-cardinality string column)
    for col in df.columns:
        if col in (mapping.date_col, mapping.target_col):
            continue
        if col in mapping.exog_cols:
            continue
        if df[col].dtype == object and df[col].nunique() < 50:
            mapping.group_col = col
            break

    return mapping
