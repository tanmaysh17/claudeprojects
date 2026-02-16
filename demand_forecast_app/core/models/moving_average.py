"""Moving Average and Drift forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster


class MovingAverage(BaseForecaster):
    name = "Moving Average"

    def __init__(self, window: int = 13, use_drift: bool = True):
        self.window = window
        self.use_drift = use_drift
        self._ma_value: float = 0.0
        self._drift: float = 0.0
        self._residual_std: float = 0.0
        self._n_train: int = 0

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        self._n_train = len(y_train)
        self._ma_value = float(y_train.iloc[-self.window:].mean())

        if self.use_drift and len(y_train) > 1:
            self._drift = float((y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1))
        else:
            self._drift = 0.0

        # In-sample residuals for prediction intervals
        ma_insample = y_train.rolling(self.window, min_periods=1).mean().shift(1)
        residuals = (y_train - ma_insample).dropna()
        self._residual_std = float(residuals.std()) if len(residuals) > 0 else float(y_train.std())

    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        return_ci: bool = True,
        ci_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        if ci_levels is None:
            ci_levels = [0.80, 0.95]

        steps = np.arange(1, horizon + 1)
        if self.use_drift:
            forecasts = self._ma_value + self._drift * steps
        else:
            forecasts = np.full(horizon, self._ma_value)

        result = pd.DataFrame({"forecast": forecasts})

        if return_ci:
            from scipy import stats
            for level in ci_levels:
                z = stats.norm.ppf(0.5 + level / 2)
                width = z * self._residual_std * np.sqrt(steps)
                pct = int(level * 100)
                result[f"lower_{pct}"] = forecasts - width
                result[f"upper_{pct}"] = forecasts + width

        return result

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "use_drift": self.use_drift,
            "ma_value": round(self._ma_value, 2),
            "drift": round(self._drift, 4),
        }

    def summary(self) -> str:
        drift_str = f", drift={self._drift:.4f}/week" if self.use_drift else ""
        return f"Moving Average (window={self.window}{drift_str}): MA value={self._ma_value:.2f}."
