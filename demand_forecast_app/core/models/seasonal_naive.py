"""Seasonal Naive forecaster: repeats the last observed seasonal cycle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster


class SeasonalNaive(BaseForecaster):
    name = "Seasonal Naive"

    def __init__(self, season_length: int = 52):
        self.season_length = season_length
        self._y_train: pd.Series | None = None
        self._residual_std: float = 0.0

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        self._y_train = y_train.copy()
        # Compute residual std from in-sample seasonal naive errors
        if len(y_train) > self.season_length:
            naive_pred = y_train.shift(self.season_length)
            residuals = (y_train - naive_pred).dropna()
            self._residual_std = float(residuals.std())
        else:
            self._residual_std = float(y_train.std())

    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        return_ci: bool = True,
        ci_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        if self._y_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if ci_levels is None:
            ci_levels = [0.80, 0.95]

        last_season = self._y_train.values[-self.season_length:]
        forecasts = np.tile(last_season, (horizon // self.season_length) + 1)[:horizon]

        result = pd.DataFrame({"forecast": forecasts})

        if return_ci:
            from scipy import stats
            for level in ci_levels:
                z = stats.norm.ppf(0.5 + level / 2)
                # Prediction interval widens with sqrt of number of seasons ahead
                steps = np.arange(1, horizon + 1)
                k = np.ceil(steps / self.season_length)
                width = z * self._residual_std * np.sqrt(k)
                pct = int(level * 100)
                result[f"lower_{pct}"] = forecasts - width
                result[f"upper_{pct}"] = forecasts + width

        return result

    def get_params(self) -> dict:
        return {"season_length": self.season_length}

    def summary(self) -> str:
        return (
            f"Seasonal Naive (period={self.season_length}): "
            f"Repeats last observed cycle. Residual std={self._residual_std:.2f}."
        )
