"""Model ranking and selection logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Models considered simpler (prefer these when tied)
_SIMPLICITY_ORDER = [
    "Seasonal Naive",
    "Moving Average",
    "Holt-Winters",
    "STL Decomposition",
    "SARIMA",
    "ML (Ridge)",
    "ML (Random Forest)",
    "ML (Gradient Boosting)",
    "Ensemble (Average)",
    "Ensemble (Weighted)",
]


def rank_models(
    comparison_df: pd.DataFrame,
    primary_metric: str = "MAPE",
    stability_metric: str = "RMSE",
) -> pd.DataFrame:
    """Rank models by accuracy with simplicity tiebreaker."""
    df = comparison_df.copy()

    if primary_metric not in df.columns:
        # Use first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            df["Rank"] = range(1, len(df) + 1)
            return df
        primary_metric = numeric_cols[0]

    # Lower is better for all metrics
    df = df.sort_values(primary_metric, ascending=True).reset_index(drop=True)

    # Apply simplicity tiebreaker
    def simplicity_score(model_name: str) -> int:
        for i, name in enumerate(_SIMPLICITY_ORDER):
            if name.lower() in model_name.lower():
                return i
        return len(_SIMPLICITY_ORDER)

    # Check for ties (within 5% relative difference)
    if len(df) > 1:
        best_val = df[primary_metric].iloc[0]
        if best_val > 0:
            df["_pct_diff"] = (df[primary_metric] - best_val) / best_val * 100
            df["_simplicity"] = df["Model"].apply(simplicity_score)

            # Among models within 5% of best, prefer simpler
            close = df["_pct_diff"] <= 5.0
            if close.sum() > 1:
                close_models = df[close].sort_values("_simplicity")
                non_close = df[~close]
                df = pd.concat([close_models, non_close], ignore_index=True)

            df = df.drop(columns=["_pct_diff", "_simplicity"])

    df["Rank"] = range(1, len(df) + 1)
    return df


def select_best_model(
    comparison_df: pd.DataFrame,
    primary_metric: str = "MAPE",
) -> str:
    """Select the best model name based on ranking."""
    ranked = rank_models(comparison_df, primary_metric)
    if len(ranked) == 0:
        return ""
    return ranked["Model"].iloc[0]
