"""ML regression forecasters using engineered features.

Implements recursive multi-step forecasting for models that use lag features.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

from .base import BaseForecaster


_MODEL_MAP = {
    "Ridge": Ridge,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
    "HistGradientBoosting": HistGradientBoostingRegressor,
}


class MLForecaster(BaseForecaster):
    name = "ML Regression"

    def __init__(
        self,
        model_type: str = "HistGradientBoosting",
        feature_cols: list[str] | None = None,
        lag_cols: list[str] | None = None,
        target_col: str = "y",
        date_col: str = "date",
        **model_kwargs,
    ):
        self.model_type = model_type
        self.name = f"ML ({model_type})"
        self.feature_cols = feature_cols or []
        self.lag_cols = lag_cols or []
        self.target_col = target_col
        self.date_col = date_col
        self.model_kwargs = model_kwargs
        self._model = None
        self._residuals: np.ndarray | None = None
        self._residual_std: float = 0.0
        self._y_train: pd.Series | None = None
        self._X_train: pd.DataFrame | None = None
        self._feature_importances: dict | None = None

    def _create_model(self):
        cls = _MODEL_MAP.get(self.model_type)
        if cls is None:
            raise ValueError(f"Unknown model type: {self.model_type}. Available: {list(_MODEL_MAP.keys())}")

        kwargs = dict(self.model_kwargs)
        if self.model_type == "RandomForest":
            kwargs.setdefault("n_estimators", 200)
            kwargs.setdefault("max_depth", 10)
            kwargs.setdefault("random_state", 42)
        elif self.model_type == "GradientBoosting":
            kwargs.setdefault("n_estimators", 200)
            kwargs.setdefault("max_depth", 5)
            kwargs.setdefault("random_state", 42)
        elif self.model_type == "HistGradientBoosting":
            kwargs.setdefault("max_iter", 200)
            kwargs.setdefault("max_depth", 6)
            kwargs.setdefault("random_state", 42)
        elif self.model_type == "Ridge":
            kwargs.setdefault("alpha", 1.0)

        return cls(**kwargs)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        if X_train is None or len(self.feature_cols) == 0:
            raise ValueError("ML models require a feature matrix (X_train) and feature_cols.")

        self._y_train = y_train.copy()
        self._X_train = X_train.copy()

        # Drop rows with NaN (from lags)
        cols = [c for c in self.feature_cols if c in X_train.columns]
        self.feature_cols = cols
        combined = X_train[cols].copy()
        combined["__target__"] = y_train.values
        combined = combined.dropna()

        X = combined[cols]
        y = combined["__target__"]

        self._model = self._create_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X, y)

        # In-sample residuals
        preds = self._model.predict(X)
        self._residuals = y.values - preds
        self._residual_std = float(np.std(self._residuals))

        # Feature importances
        if hasattr(self._model, "feature_importances_"):
            self._feature_importances = dict(zip(cols, self._model.feature_importances_))
        elif hasattr(self._model, "coef_"):
            self._feature_importances = dict(zip(cols, np.abs(self._model.coef_)))

    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        return_ci: bool = True,
        ci_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not fitted.")

        if ci_levels is None:
            ci_levels = [0.80, 0.95]

        if X_future is not None and len(X_future) >= horizon:
            # Direct prediction if future features are pre-computed
            cols = [c for c in self.feature_cols if c in X_future.columns]
            X = X_future[cols].iloc[:horizon].fillna(0)
            forecasts = self._model.predict(X)
        else:
            # Recursive prediction for models with lag features
            forecasts = self._recursive_predict(horizon, X_future)

        result = pd.DataFrame({"forecast": forecasts})

        if return_ci:
            from scipy import stats
            steps = np.arange(1, horizon + 1)
            for level in ci_levels:
                z = stats.norm.ppf(0.5 + level / 2)
                # Wider intervals for further horizons in recursive mode
                width = z * self._residual_std * np.sqrt(1 + 0.05 * steps)
                pct = int(level * 100)
                result[f"lower_{pct}"] = forecasts - width
                result[f"upper_{pct}"] = forecasts + width

        return result

    def _recursive_predict(self, horizon: int, X_future: pd.DataFrame | None) -> np.ndarray:
        """Recursive multi-step prediction."""
        forecasts = []
        history = self._y_train.values.tolist()

        for step in range(horizon):
            feature_row = {}

            # Rebuild lag features from history
            for col in self.feature_cols:
                if col.startswith("lag_"):
                    try:
                        lag_n = int(col.split("_")[1])
                        if lag_n <= len(history):
                            feature_row[col] = history[-lag_n]
                        else:
                            feature_row[col] = 0.0
                    except (ValueError, IndexError):
                        feature_row[col] = 0.0
                elif col.startswith("rolling_"):
                    # Approximate rolling features from history
                    parts = col.split("_")
                    try:
                        window = int(parts[1].replace("w", ""))
                        func = parts[2]
                        vals = history[-window:] if len(history) >= window else history
                        if func == "mean":
                            feature_row[col] = np.mean(vals)
                        elif func == "std":
                            feature_row[col] = np.std(vals) if len(vals) > 1 else 0
                        elif func == "min":
                            feature_row[col] = np.min(vals)
                        elif func == "max":
                            feature_row[col] = np.max(vals)
                        else:
                            feature_row[col] = 0.0
                    except (ValueError, IndexError):
                        feature_row[col] = 0.0
                elif X_future is not None and col in X_future.columns and step < len(X_future):
                    feature_row[col] = X_future[col].iloc[step]
                else:
                    # Use last known value or 0
                    if self._X_train is not None and col in self._X_train.columns:
                        feature_row[col] = self._X_train[col].iloc[-1]
                    else:
                        feature_row[col] = 0.0

            X_step = pd.DataFrame([feature_row])[self.feature_cols]
            X_step = X_step.fillna(0)
            pred = self._model.predict(X_step)[0]
            forecasts.append(pred)
            history.append(pred)

        return np.array(forecasts)

    def get_params(self) -> dict:
        return {
            "model_type": self.model_type,
            "n_features": len(self.feature_cols),
            "residual_std": round(self._residual_std, 2),
        }

    def get_feature_importances(self) -> dict | None:
        return self._feature_importances

    def summary(self) -> str:
        return (
            f"ML Forecaster ({self.model_type}) with {len(self.feature_cols)} features. "
            f"Residual std={self._residual_std:.2f}."
        )
