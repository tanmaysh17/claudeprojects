"""Additive decomposition forecaster (native Prophet-replacement).

Uses STL to decompose into trend + seasonal + residual, then independently
extrapolates each component and recombines.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .base import BaseForecaster


class DecompositionForecaster(BaseForecaster):
    name = "Decomposition (STL)"

    def __init__(
        self,
        seasonal_period: int = 52,
        trend_method: str = "linear",  # "linear", "quadratic", "holt"
        robust: bool = True,
    ):
        self.seasonal_period = seasonal_period
        self.trend_method = trend_method
        self.robust = robust
        self._trend: pd.Series | None = None
        self._seasonal: pd.Series | None = None
        self._residual: pd.Series | None = None
        self._trend_coeffs: np.ndarray | None = None
        self._seasonal_pattern: np.ndarray | None = None
        self._residual_std: float = 0.0
        self._n_train: int = 0
        self._holt_result = None

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        y = y_train.copy().astype(float)
        self._n_train = len(y)
        sp = self.seasonal_period

        if len(y) < 2 * sp:
            sp = max(4, len(y) // 4)

        # STL decomposition
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(y, period=sp, robust=self.robust)
            result = stl.fit()

        self._trend = result.trend
        self._seasonal = result.seasonal
        self._residual = result.resid

        # Fit trend extrapolation
        trend_vals = self._trend.values
        x = np.arange(len(trend_vals))

        if self.trend_method == "quadratic":
            self._trend_coeffs = np.polyfit(x, trend_vals, 2)
        elif self.trend_method == "holt":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    pd.Series(trend_vals),
                    trend="add",
                    seasonal=None,
                    damped_trend=True,
                )
                self._holt_result = model.fit(optimized=True)
        else:  # linear
            self._trend_coeffs = np.polyfit(x, trend_vals, 1)

        # Extract seasonal pattern (last full cycle)
        seasonal_vals = self._seasonal.values
        n_full_cycles = len(seasonal_vals) // sp
        if n_full_cycles >= 2:
            # Average last 2 full cycles for more stability
            last_two = seasonal_vals[-(2 * sp):]
            pattern = (last_two[:sp] + last_two[sp:]) / 2
        elif n_full_cycles >= 1:
            pattern = seasonal_vals[-sp:]
        else:
            pattern = seasonal_vals

        self._seasonal_pattern = pattern
        self._residual_std = float(self._residual.std())

    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        return_ci: bool = True,
        ci_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        if self._trend is None:
            raise RuntimeError("Model not fitted.")

        if ci_levels is None:
            ci_levels = [0.80, 0.95]

        n = self._n_train
        future_x = np.arange(n, n + horizon)

        # Trend forecast
        if self.trend_method == "holt" and self._holt_result is not None:
            trend_forecast = self._holt_result.forecast(horizon).values
        else:
            trend_forecast = np.polyval(self._trend_coeffs, future_x)

        # Seasonal forecast: tile the pattern
        sp = len(self._seasonal_pattern)
        # Determine where in the seasonal cycle we are
        start_pos = n % sp
        seasonal_forecast = np.array([
            self._seasonal_pattern[(start_pos + i) % sp]
            for i in range(horizon)
        ])

        # Combine
        forecasts = trend_forecast + seasonal_forecast

        result = pd.DataFrame({"forecast": forecasts})

        if return_ci:
            from scipy import stats
            steps = np.arange(1, horizon + 1)
            for level in ci_levels:
                z = stats.norm.ppf(0.5 + level / 2)
                width = z * self._residual_std * np.sqrt(1 + np.log1p(steps) * 0.1)
                pct = int(level * 100)
                result[f"lower_{pct}"] = forecasts - width
                result[f"upper_{pct}"] = forecasts + width

        return result

    def get_params(self) -> dict:
        return {
            "seasonal_period": self.seasonal_period,
            "trend_method": self.trend_method,
            "robust": self.robust,
            "residual_std": round(self._residual_std, 2),
        }

    def summary(self) -> str:
        return (
            f"STL Decomposition Forecaster (period={self.seasonal_period}, "
            f"trend={self.trend_method}, robust={self.robust}). "
            f"Residual std={self._residual_std:.2f}."
        )
