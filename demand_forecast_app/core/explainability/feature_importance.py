"""Feature importance computation for ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_model_feature_importance(model) -> pd.DataFrame | None:
    """Extract feature importance from an ML model.

    Works with MLForecaster instances that have _feature_importances.
    """
    if hasattr(model, "get_feature_importances"):
        imp = model.get_feature_importances()
        if imp:
            df = pd.DataFrame(
                {"feature": list(imp.keys()), "importance": list(imp.values())}
            )
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)
            df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
            return df
    return None
