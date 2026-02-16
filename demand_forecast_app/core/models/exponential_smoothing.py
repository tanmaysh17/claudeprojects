"""ETS / Holt-Winters Exponential Smoothing forecaster."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .base import BaseForecaster


class HoltWinters(BaseForecaster):
    name = "Holt-Winters"

    def __init__(
        self,
        trend: str | None = "add",
        seasonal: str | None = "add",
        seasonal_periods: int = 52,
        damped_trend: bool = False,
        auto_select: bool = True,
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.auto_select = auto_select
        self._model = None
        self._result = None
        self._best_config: dict = {}
        self._residual_std: float = 0.0

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        y = y_train.copy().astype(float)

        # Ensure positive values for multiplicative models
        has_nonpositive = (y <= 0).any()

        if self.auto_select:
            self._auto_fit(y, has_nonpositive)
        else:
            trend = self.trend
            seasonal = self.seasonal
            if has_nonpositive:
                if seasonal == "mul":
                    seasonal = "add"
                if trend == "mul":
                    trend = "add"

            sp = self.seasonal_periods if seasonal else None
            if sp and len(y) < 2 * sp:
                seasonal = None
                sp = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = ExponentialSmoothing(
                    y,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=sp,
                    damped_trend=self.damped_trend if trend else False,
                )
                self._result = self._model.fit(optimized=True)
                self._best_config = {
                    "trend": trend,
                    "seasonal": seasonal,
                    "seasonal_periods": sp,
                    "damped_trend": self.damped_trend if trend else False,
                }

        if self._result is not None:
            self._residual_std = float(self._result.resid.std())

    def _auto_fit(self, y: pd.Series, has_nonpositive: bool):
        """Try multiple configurations and select by AIC."""
        configs = []
        trend_opts = ["add", None]
        seasonal_opts = ["add", None]
        if not has_nonpositive:
            trend_opts.append("mul")
            seasonal_opts.append("mul")

        sp = self.seasonal_periods
        if len(y) < 2 * sp:
            seasonal_opts = [None]
            sp = None

        for t in trend_opts:
            for s in seasonal_opts:
                for damped in [False, True] if t else [False]:
                    configs.append({
                        "trend": t,
                        "seasonal": s,
                        "seasonal_periods": sp if s else None,
                        "damped_trend": damped,
                    })

        best_aic = float("inf")
        for cfg in configs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        y,
                        trend=cfg["trend"],
                        seasonal=cfg["seasonal"],
                        seasonal_periods=cfg["seasonal_periods"],
                        damped_trend=cfg["damped_trend"],
                    )
                    result = model.fit(optimized=True)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        self._model = model
                        self._result = result
                        self._best_config = cfg
            except Exception:
                continue

        if self._result is None:
            # Fallback: simplest model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = ExponentialSmoothing(y, trend=None, seasonal=None)
                self._result = self._model.fit(optimized=True)
                self._best_config = {"trend": None, "seasonal": None, "seasonal_periods": None, "damped_trend": False}

    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        return_ci: bool = True,
        ci_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("Model not fitted.")

        if ci_levels is None:
            ci_levels = [0.80, 0.95]

        forecasts = self._result.forecast(horizon)
        result = pd.DataFrame({"forecast": forecasts.values})

        if return_ci:
            try:
                sim = self._result.simulate(horizon, repetitions=500, error="mul" if self._best_config.get("seasonal") == "mul" else "add")
                for level in ci_levels:
                    alpha = 1 - level
                    lower = sim.quantile(alpha / 2, axis=1).values
                    upper = sim.quantile(1 - alpha / 2, axis=1).values
                    pct = int(level * 100)
                    result[f"lower_{pct}"] = lower
                    result[f"upper_{pct}"] = upper
            except Exception:
                from scipy import stats
                steps = np.arange(1, horizon + 1)
                for level in ci_levels:
                    z = stats.norm.ppf(0.5 + level / 2)
                    width = z * self._residual_std * np.sqrt(steps)
                    pct = int(level * 100)
                    result[f"lower_{pct}"] = forecasts.values - width
                    result[f"upper_{pct}"] = forecasts.values + width

        return result

    def get_params(self) -> dict:
        params = dict(self._best_config)
        if self._result is not None:
            params["aic"] = round(self._result.aic, 2)
        return params

    def summary(self) -> str:
        cfg = self._best_config
        parts = []
        if cfg.get("trend"):
            parts.append(f"trend={cfg['trend']}")
        if cfg.get("seasonal"):
            parts.append(f"seasonal={cfg['seasonal']}(period={cfg.get('seasonal_periods')})")
        if cfg.get("damped_trend"):
            parts.append("damped")
        config_str = ", ".join(parts) if parts else "simple exponential smoothing"
        aic = f", AIC={self._result.aic:.1f}" if self._result else ""
        return f"Holt-Winters ({config_str}{aic})"
