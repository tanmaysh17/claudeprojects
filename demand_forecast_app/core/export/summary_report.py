"""Summary report builder."""

from __future__ import annotations

import pandas as pd


def build_text_report(
    data_summary: dict,
    model_comparison: pd.DataFrame,
    forecast_summary: str,
    methodology_note: str = "",
) -> str:
    """Build a full-text forecast report."""
    lines = []

    lines.append("=" * 70)
    lines.append("DEMAND FORECAST REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Data summary
    lines.append("DATA OVERVIEW")
    lines.append("-" * 40)
    for key, val in data_summary.items():
        lines.append(f"  {key}: {val}")
    lines.append("")

    # Methodology
    if methodology_note:
        lines.append("METHODOLOGY")
        lines.append("-" * 40)
        lines.append(f"  {methodology_note}")
        lines.append("")

    # Model comparison
    if len(model_comparison) > 0:
        lines.append("MODEL COMPARISON")
        lines.append("-" * 40)
        lines.append(model_comparison.to_string(index=False))
        lines.append("")

    # Forecast summary
    lines.append("FORECAST ANALYSIS")
    lines.append("-" * 40)
    lines.append(forecast_summary)
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Report")
    lines.append("=" * 70)

    return "\n".join(lines)
