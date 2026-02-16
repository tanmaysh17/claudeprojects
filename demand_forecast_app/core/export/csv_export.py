"""CSV export utilities."""

from __future__ import annotations

from io import BytesIO, StringIO

import pandas as pd


def export_forecasts_csv(forecast_df: pd.DataFrame) -> BytesIO:
    """Export forecasts to CSV."""
    output = BytesIO()
    forecast_df.to_csv(output, index=False)
    output.seek(0)
    return output


def export_comparison_csv(comparison_df: pd.DataFrame) -> BytesIO:
    """Export model comparison to CSV."""
    output = BytesIO()
    comparison_df.to_csv(output, index=False)
    output.seek(0)
    return output
