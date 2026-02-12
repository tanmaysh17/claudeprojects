"""Abstract base class for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Contract that every forecasting model must implement."""

    name: str = "BaseForecaster"

    @abstractmethod
    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> None:
        """Fit the model on training data.

        Args:
            y_train: Target time series (indexed by datetime or integer).
            X_train: Optional exogenous feature matrix aligned with y_train.
        """

    @abstractmethod
    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        return_ci: bool = True,
        ci_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for the given horizon.

        Args:
            horizon: Number of periods to forecast.
            X_future: Optional exogenous features for the forecast period.
            return_ci: Whether to include confidence intervals.
            ci_levels: Confidence levels (e.g., [0.80, 0.95]).

        Returns:
            DataFrame with columns: forecast, lower_80, upper_80, lower_95, upper_95
            (confidence interval columns present only if return_ci=True).
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Return model parameters as a dictionary."""

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model."""
        return f"{self.name}: {self.get_params()}"
