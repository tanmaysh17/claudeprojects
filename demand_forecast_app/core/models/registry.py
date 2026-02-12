"""Model registry: list and instantiate models by name."""

from __future__ import annotations

from .base import BaseForecaster
from .seasonal_naive import SeasonalNaive
from .moving_average import MovingAverage
from .exponential_smoothing import HoltWinters
from .sarima import SARIMAForecaster
from .decomposition_forecast import DecompositionForecaster
from .ml_regression import MLForecaster
from .ensemble import SimpleAverageEnsemble, WeightedEnsemble


MODEL_REGISTRY: dict[str, type[BaseForecaster]] = {
    "Seasonal Naive": SeasonalNaive,
    "Moving Average": MovingAverage,
    "Holt-Winters (ETS)": HoltWinters,
    "SARIMA": SARIMAForecaster,
    "STL Decomposition": DecompositionForecaster,
    "ML (Ridge)": MLForecaster,
    "ML (Random Forest)": MLForecaster,
    "ML (Gradient Boosting)": MLForecaster,
    "Ensemble (Average)": SimpleAverageEnsemble,
    "Ensemble (Weighted)": WeightedEnsemble,
}

# Default kwargs per model name
_DEFAULT_KWARGS: dict[str, dict] = {
    "ML (Ridge)": {"model_type": "Ridge"},
    "ML (Random Forest)": {"model_type": "RandomForest"},
    "ML (Gradient Boosting)": {"model_type": "HistGradientBoosting"},
}

# Models that are baseline (always included)
BASELINE_MODELS = ["Seasonal Naive", "Moving Average"]

# Models suitable for auto mode
AUTO_MODELS = [
    "Seasonal Naive",
    "Moving Average",
    "Holt-Winters (ETS)",
    "STL Decomposition",
]


def get_available_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_statistical_models() -> list[str]:
    return [
        "Seasonal Naive",
        "Moving Average",
        "Holt-Winters (ETS)",
        "SARIMA",
        "STL Decomposition",
    ]


def get_ml_models() -> list[str]:
    return [
        "ML (Ridge)",
        "ML (Random Forest)",
        "ML (Gradient Boosting)",
    ]


def create_model(name: str, **kwargs) -> BaseForecaster:
    """Create a model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {get_available_models()}")

    cls = MODEL_REGISTRY[name]
    default_kw = _DEFAULT_KWARGS.get(name, {})
    merged = {**default_kw, **kwargs}
    return cls(**merged)
