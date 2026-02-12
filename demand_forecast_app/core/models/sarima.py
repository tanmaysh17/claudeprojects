"""SARIMA / SARIMAX forecaster."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseForecaster


class SARIMAForecaster(BaseForecaster):
    name = "SARIMA"

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        auto_select: bool = True,
        max_p: int = 2,
        max_q: int = 2,
        seasonal_period: int = 52,
    ):
        self.order = order or (1, 1, 1)
        self.seasonal_order = seasonal_order or (1, 1, 1, seasonal_period)
        self.auto_select = auto_select
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal_period = seasonal_period
        self._model = None
        self._result = None
        self._best_order = self.order
        self._best_seasonal_order = self.seasonal_order
        self._residual_std: float = 0.0
        self._uses_exog: bool = False

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        y = y_train.copy().astype(float)
        self._uses_exog = X_train is not None and len(X_train.columns) > 0

        # For weekly data with period=52, full SARIMA is very slow.
        # Use reduced seasonal if data is short.
        sp = self.seasonal_period
        if len(y) < 3 * sp:
            sp = 0  # disable seasonal component for short series

        if self.auto_select:
            self._auto_fit(y, X_train, sp)
        else:
            so = self.seasonal_order if sp > 0 else (0, 0, 0, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = SARIMAX(
                    y,
                    exog=X_train,
                    order=self.order,
                    seasonal_order=so,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self._result = self._model.fit(disp=False, maxiter=200)
                self._best_order = self.order
                self._best_seasonal_order = so

        if self._result is not None:
            self._residual_std = float(self._result.resid.std())

    def _auto_fit(self, y: pd.Series, X_train: pd.DataFrame | None, sp: int):
        """Grid search over a small parameter space."""
        p_range = range(self.max_p + 1)
        d_range = [0, 1]
        q_range = range(self.max_q + 1)

        if sp > 0:
            # Limited seasonal search for performance
            seasonal_configs = [
                (1, 1, 0, sp),
                (0, 1, 1, sp),
                (1, 1, 1, sp),
            ]
        else:
            seasonal_configs = [(0, 0, 0, 0)]

        best_aic = float("inf")
        best_result = None

        # First try non-seasonal ARIMA (faster)
        for p, d, q in itertools.product(p_range, d_range, q_range):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        y,
                        exog=X_train,
                        order=(p, d, q),
                        seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False, maxiter=100)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_result = result
                        self._model = model
                        self._best_order = (p, d, q)
                        self._best_seasonal_order = (0, 0, 0, 0)
            except Exception:
                continue

        # Then try seasonal (only if period > 0)
        if sp > 0:
            for so in seasonal_configs:
                for p, d, q in [(1, 1, 1), (1, 1, 0), (0, 1, 1)]:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = SARIMAX(
                                y,
                                exog=X_train,
                                order=(p, d, q),
                                seasonal_order=so,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            )
                            result = model.fit(disp=False, maxiter=100)
                            if result.aic < best_aic:
                                best_aic = result.aic
                                best_result = result
                                self._model = model
                                self._best_order = (p, d, q)
                                self._best_seasonal_order = so
                    except Exception:
                        continue

        if best_result is not None:
            self._result = best_result
        else:
            # Fallback: simple ARIMA(1,1,1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = SARIMAX(
                    y,
                    exog=X_train,
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self._result = self._model.fit(disp=False, maxiter=200)
                self._best_order = (1, 1, 1)
                self._best_seasonal_order = (0, 0, 0, 0)

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

        exog = X_future if self._uses_exog else None
        forecast_obj = self._result.get_forecast(steps=horizon, exog=exog)
        point_forecast = forecast_obj.predicted_mean.values

        result = pd.DataFrame({"forecast": point_forecast})

        if return_ci:
            for level in ci_levels:
                alpha = 1 - level
                ci = forecast_obj.conf_int(alpha=alpha)
                pct = int(level * 100)
                result[f"lower_{pct}"] = ci.iloc[:, 0].values
                result[f"upper_{pct}"] = ci.iloc[:, 1].values

        return result

    def get_params(self) -> dict:
        params = {
            "order": self._best_order,
            "seasonal_order": self._best_seasonal_order,
            "uses_exog": self._uses_exog,
        }
        if self._result is not None:
            params["aic"] = round(self._result.aic, 2)
        return params

    def summary(self) -> str:
        so = self._best_seasonal_order
        seasonal_str = f" x ({so[0]},{so[1]},{so[2]})[{so[3]}]" if so[3] > 0 else ""
        o = self._best_order
        aic = f", AIC={self._result.aic:.1f}" if self._result else ""
        exog_str = " with exog" if self._uses_exog else ""
        return f"SARIMA({o[0]},{o[1]},{o[2]}){seasonal_str}{exog_str}{aic}"
