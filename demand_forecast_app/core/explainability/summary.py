"""Plain-language forecast explanation generator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_summary(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    forecast_df: pd.DataFrame | None = None,
    data_stats: dict | None = None,
    n_folds: int = 3,
) -> str:
    """Generate a plain-language explanation of the forecast results."""
    sections = []

    # Model selection rationale
    best_row = comparison_df[comparison_df["Model"] == best_model_name]
    if len(best_row) > 0:
        best_row = best_row.iloc[0]
        metrics_str = []
        for col in ["MAPE", "RMSE", "MAE", "sMAPE"]:
            if col in best_row and not np.isnan(best_row[col]):
                metrics_str.append(f"{col}={best_row[col]:.2f}")
        metrics_text = ", ".join(metrics_str) if metrics_str else "evaluated metrics"

        sections.append(
            f"MODEL SELECTION: Based on {n_folds}-fold rolling cross-validation, "
            f"the {best_model_name} model was selected as the best performer "
            f"with {metrics_text}."
        )

    # Comparison to baselines
    baselines = comparison_df[comparison_df["Model"].str.contains("Naive|Moving Average", case=False, na=False)]
    if len(baselines) > 0 and len(best_row) > 0:
        best_mape = best_row.get("MAPE", None)
        baseline_mape = baselines["MAPE"].min() if "MAPE" in baselines.columns else None
        if best_mape is not None and baseline_mape is not None and baseline_mape > 0:
            improvement = (baseline_mape - best_mape) / baseline_mape * 100
            if improvement > 0:
                sections.append(
                    f"BASELINE COMPARISON: The selected model improves upon the best baseline "
                    f"by {improvement:.1f}% in MAPE terms."
                )
            else:
                sections.append(
                    "BASELINE COMPARISON: The selected model is a baseline method, "
                    "suggesting that more complex models did not provide meaningful improvement "
                    "for this data. This is common with short histories or highly regular patterns."
                )

    # Forecast overview
    if forecast_df is not None and "forecast" in forecast_df.columns:
        fc = forecast_df["forecast"]
        sections.append(
            f"FORECAST OVERVIEW: The forecast covers {len(fc)} periods. "
            f"Forecasted values range from {fc.min():.1f} to {fc.max():.1f} "
            f"with a mean of {fc.mean():.1f}."
        )

        # Growth outlook
        first_q = fc.iloc[:13].mean() if len(fc) >= 13 else fc.mean()
        last_q = fc.iloc[-13:].mean() if len(fc) >= 13 else fc.mean()
        if first_q > 0:
            growth = (last_q - first_q) / first_q * 100
            direction = "growth" if growth > 0 else "decline"
            sections.append(
                f"GROWTH OUTLOOK: The forecast suggests a {abs(growth):.1f}% {direction} "
                f"comparing the first quarter to the last quarter of the forecast horizon."
            )

    # Data quality notes
    if data_stats:
        notes = []
        n_points = data_stats.get("n_valid_points", 0)
        if n_points < 104:
            notes.append(
                f"Limited history ({n_points} data points) may reduce forecast reliability. "
                "Consider collecting more data for improved accuracy."
            )

        if notes:
            sections.append("RISKS AND ASSUMPTIONS: " + " ".join(notes))

    # Model reliability
    if len(best_row) > 0:
        mape_val = best_row.get("MAPE", None)
        if mape_val is not None:
            if mape_val < 10:
                reliability = "high"
                note = "The model demonstrates strong accuracy in backtesting."
            elif mape_val < 20:
                reliability = "moderate"
                note = "Accuracy is reasonable but forecast users should account for potential variation."
            else:
                reliability = "low"
                note = "High forecast error suggests significant uncertainty. Use forecasts as directional guidance rather than precise predictions."
            sections.append(f"RELIABILITY: Overall reliability is {reliability}. {note}")

    if not sections:
        return "No forecast results available for summary."

    return "\n\n".join(sections)


def generate_methodology_note(
    model_name: str,
    n_folds: int,
    test_size: int,
    data_points: int,
) -> str:
    """Generate a methodology description for the report."""
    return (
        f"Methodology: Forecasts were generated using the {model_name} model, "
        f"selected from a pool of statistical and machine learning methods. "
        f"Model selection was based on {n_folds}-fold rolling cross-validation "
        f"with {test_size}-week test windows, evaluated on {data_points} historical data points. "
        f"All models were evaluated using the same holdout strategy to ensure fair comparison. "
        f"Prediction intervals (80% and 95%) are provided to quantify forecast uncertainty."
    )
