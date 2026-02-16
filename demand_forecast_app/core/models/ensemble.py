"""Ensemble forecasters: simple average, weighted average, stacking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster


class SimpleAverageEnsemble(BaseForecaster):
    name = "Ensemble (Simple Average)"

    def __init__(self, models: list[BaseForecaster] | None = None):
        self.models = models or []
        self._is_fitted = False

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        for model in self.models:
            model.fit(y_train, X_train)
        self._is_fitted = True

    def fit_from_pretrained(self, models: list[BaseForecaster]) -> None:
        """Use already-fitted models without re-fitting."""
        self.models = models
        self._is_fitted = True

    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        return_ci: bool = True,
        ci_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        if not self._is_fitted or not self.models:
            raise RuntimeError("Ensemble not fitted or has no models.")

        if ci_levels is None:
            ci_levels = [0.80, 0.95]

        all_forecasts = []
        for model in self.models:
            try:
                pred = model.predict(horizon, X_future, return_ci=False)
                all_forecasts.append(pred["forecast"].values)
            except Exception:
                continue

        if not all_forecasts:
            raise RuntimeError("No models produced forecasts.")

        stacked = np.array(all_forecasts)
        mean_forecast = stacked.mean(axis=0)

        result = pd.DataFrame({"forecast": mean_forecast})

        if return_ci:
            from scipy import stats
            std_forecast = stacked.std(axis=0)
            # Combine model uncertainty with spread across models
            for level in ci_levels:
                z = stats.norm.ppf(0.5 + level / 2)
                pct = int(level * 100)
                result[f"lower_{pct}"] = mean_forecast - z * std_forecast
                result[f"upper_{pct}"] = mean_forecast + z * std_forecast

        return result

    def get_params(self) -> dict:
        return {
            "n_models": len(self.models),
            "model_names": [m.name for m in self.models],
            "method": "simple_average",
        }

    def summary(self) -> str:
        names = ", ".join(m.name for m in self.models)
        return f"Simple Average Ensemble of {len(self.models)} models: [{names}]"


class WeightedEnsemble(BaseForecaster):
    name = "Ensemble (Weighted)"

    def __init__(
        self,
        models: list[BaseForecaster] | None = None,
        weights: list[float] | None = None,
    ):
        self.models = models or []
        self.weights = weights
        self._is_fitted = False

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        for model in self.models:
            model.fit(y_train, X_train)
        self._is_fitted = True

    def fit_from_pretrained(
        self,
        models: list[BaseForecaster],
        weights: list[float] | None = None,
    ) -> None:
        self.models = models
        self.weights = weights
        self._is_fitted = True

    def set_weights_from_errors(self, errors: list[float]) -> None:
        """Set weights as inverse of error (lower error = higher weight)."""
        errors = np.array(errors, dtype=float)
        errors = np.maximum(errors, 1e-10)
        inv_errors = 1.0 / errors
        self.weights = (inv_errors / inv_errors.sum()).tolist()

    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        return_ci: bool = True,
        ci_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        if not self._is_fitted or not self.models:
            raise RuntimeError("Ensemble not fitted.")

        if ci_levels is None:
            ci_levels = [0.80, 0.95]

        all_forecasts = []
        valid_models = []
        for model in self.models:
            try:
                pred = model.predict(horizon, X_future, return_ci=False)
                all_forecasts.append(pred["forecast"].values)
                valid_models.append(model)
            except Exception:
                continue

        if not all_forecasts:
            raise RuntimeError("No models produced forecasts.")

        stacked = np.array(all_forecasts)

        if self.weights and len(self.weights) == len(all_forecasts):
            w = np.array(self.weights)
        else:
            w = np.ones(len(all_forecasts)) / len(all_forecasts)

        w = w / w.sum()
        weighted_forecast = (stacked * w[:, np.newaxis]).sum(axis=0)

        result = pd.DataFrame({"forecast": weighted_forecast})

        if return_ci:
            from scipy import stats
            weighted_std = np.sqrt(((stacked - weighted_forecast) ** 2 * w[:, np.newaxis]).sum(axis=0))
            for level in ci_levels:
                z = stats.norm.ppf(0.5 + level / 2)
                pct = int(level * 100)
                result[f"lower_{pct}"] = weighted_forecast - z * weighted_std
                result[f"upper_{pct}"] = weighted_forecast + z * weighted_std

        return result

    def get_params(self) -> dict:
        return {
            "n_models": len(self.models),
            "model_names": [m.name for m in self.models],
            "weights": [round(w, 4) for w in (self.weights or [])],
            "method": "weighted_average",
        }

    def summary(self) -> str:
        parts = []
        w = self.weights or [1 / len(self.models)] * len(self.models)
        for m, weight in zip(self.models, w):
            parts.append(f"{m.name}({weight:.2%})")
        return f"Weighted Ensemble: {', '.join(parts)}"
