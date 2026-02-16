"""Time series cross-validation engine (rolling and expanding window)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models.base import BaseForecaster
from .metrics import compute_metrics, coverage

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    strategy: str = "rolling"  # "rolling" or "expanding"
    n_splits: int = 3
    test_size: int = 13  # weeks in each test window
    min_train_size: int = 104  # minimum training weeks
    step_size: int | None = None  # step between windows; defaults to test_size

    def __post_init__(self):
        if self.step_size is None:
            self.step_size = self.test_size


@dataclass
class FoldResult:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    actual: np.ndarray
    predicted: np.ndarray
    lower_80: np.ndarray | None = None
    upper_80: np.ndarray | None = None
    lower_95: np.ndarray | None = None
    upper_95: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    model_name: str
    config: BacktestConfig
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def aggregate_metrics(self) -> dict[str, float]:
        """Mean of metrics across folds."""
        if not self.folds:
            return {}
        all_metrics = [f.metrics for f in self.folds if f.metrics]
        if not all_metrics:
            return {}
        keys = all_metrics[0].keys()
        return {k: float(np.mean([m[k] for m in all_metrics if k in m])) for k in keys}

    def metrics_by_fold(self) -> pd.DataFrame:
        rows = []
        for fold in self.folds:
            row = {"fold": fold.fold}
            row.update(fold.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


def generate_splits(
    n_samples: int,
    config: BacktestConfig,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Generate train/test index ranges for time series cross-validation."""
    splits = []
    test_size = config.test_size
    step = config.step_size or test_size

    # Work backwards from the end
    for i in range(config.n_splits):
        test_end = n_samples - i * step
        test_start = test_end - test_size

        if test_start < config.min_train_size:
            break

        if config.strategy == "rolling":
            train_start = max(0, test_start - config.min_train_size)
        else:  # expanding
            train_start = 0

        train_end = test_start

        if train_end - train_start < config.min_train_size // 2:
            break

        splits.append(((train_start, train_end), (test_start, test_end)))

    splits.reverse()
    return splits


def run_backtest(
    model_factory,
    y: pd.Series,
    config: BacktestConfig,
    X: pd.DataFrame | None = None,
    model_kwargs: dict | None = None,
) -> BacktestResult:
    """Run time series cross-validation for a single model.

    Args:
        model_factory: Callable that creates a new model instance.
        y: Full target series.
        config: Backtest configuration.
        X: Optional feature matrix aligned with y.
        model_kwargs: Extra kwargs passed to model_factory.
    """
    model_kwargs = model_kwargs or {}
    splits = generate_splits(len(y), config)

    # Get model name from a temporary instance
    temp_model = model_factory(**model_kwargs)
    result = BacktestResult(model_name=temp_model.name, config=config)

    for fold_idx, ((train_start, train_end), (test_start, test_end)) in enumerate(splits):
        y_train = y.iloc[train_start:train_end]
        y_test = y.iloc[test_start:test_end]
        X_train = X.iloc[train_start:train_end] if X is not None else None
        X_test = X.iloc[test_start:test_end] if X is not None else None

        horizon = test_end - test_start

        try:
            model = model_factory(**model_kwargs)
            model.fit(y_train, X_train)
            preds = model.predict(horizon, X_test, return_ci=True)

            actual = y_test.values
            predicted = preds["forecast"].values

            fold_result = FoldResult(
                fold=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                actual=actual,
                predicted=predicted,
            )

            # Extract confidence intervals if available
            if "lower_80" in preds.columns:
                fold_result.lower_80 = preds["lower_80"].values
                fold_result.upper_80 = preds["upper_80"].values
            if "lower_95" in preds.columns:
                fold_result.lower_95 = preds["lower_95"].values
                fold_result.upper_95 = preds["upper_95"].values

            # Compute metrics
            fold_result.metrics = compute_metrics(actual, predicted)

            # Add coverage metrics if CI available
            if fold_result.lower_80 is not None:
                fold_result.metrics["Coverage_80"] = coverage(
                    actual, fold_result.lower_80, fold_result.upper_80
                )
            if fold_result.lower_95 is not None:
                fold_result.metrics["Coverage_95"] = coverage(
                    actual, fold_result.lower_95, fold_result.upper_95
                )

            result.folds.append(fold_result)

        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed for {temp_model.name}: {e}")
            continue

    return result


def compare_models(
    backtest_results: dict[str, BacktestResult],
) -> pd.DataFrame:
    """Create a comparison table across models."""
    rows = []
    for name, result in backtest_results.items():
        agg = result.aggregate_metrics()
        row = {"Model": name}
        row.update(agg)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "MAPE" in df.columns:
        df = df.sort_values("MAPE").reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
    return df
