"""Page 5: Backtesting and Cross-Validation."""

import streamlit as st
import pandas as pd
import numpy as np

from ui.layout import page_header
from ui.session import require_stage, set_stage, DATA_VALIDATED, MODELS_RUN, BACKTEST_DONE, log_action
from ui.widgets import cv_config_panel
from ui.charts import plot_backtest, plot_model_comparison
from core.backtesting.cross_validation import run_backtest, compare_models, BacktestConfig
from core.backtesting.metrics import METRIC_DISPLAY_NAMES
from core.models.registry import create_model, get_ml_models
from core.features.pipeline import get_feature_names

page_header(
    "Backtesting & Model Evaluation",
    "Evaluate model accuracy using time-series-aware cross-validation. No random splits.",
)

if not require_stage(DATA_VALIDATED, "Please upload and validate data first."):
    st.stop()

prepared_df = st.session_state["prepared_df"]
mapping = st.session_state["mapping"]
date_col = mapping.date_col
target_col = mapping.target_col
series = prepared_df[target_col]
dates = prepared_df[date_col]

selected_models = st.session_state.get("selected_models", [])
if not selected_models:
    st.warning("No models selected. Please go to the Forecasting page and select models first.")
    st.stop()

# --- CV Configuration ---
st.subheader("Cross-Validation Configuration")
config = cv_config_panel()

seasonal_period = st.session_state.get("forecast_horizon_sp", 52)
seasonal_period = st.number_input("Seasonal Period", min_value=2, max_value=104, value=52, key="bt_sp")

# --- Run Backtest ---
if st.button("Run Backtesting", type="primary"):
    backtest_results = {}
    progress = st.progress(0)
    status = st.empty()

    ml_model_names = set(get_ml_models())
    feature_matrix = st.session_state.get("feature_matrix")

    for i, model_name in enumerate(selected_models):
        if "Ensemble" in model_name:
            continue  # Skip ensembles in backtesting

        progress.progress(i / len(selected_models))
        status.text(f"Backtesting {model_name}...")

        try:
            # Define model factory
            if model_name in ml_model_names:
                if feature_matrix is None:
                    st.warning(f"Skipping {model_name}: Feature matrix not built.")
                    continue
                feature_names = get_feature_names(feature_matrix, date_col, target_col)
                lag_cols = [c for c in feature_names if c.startswith("lag_")]
                kwargs = {
                    "feature_cols": feature_names,
                    "lag_cols": lag_cols,
                    "target_col": target_col,
                    "date_col": date_col,
                }
                X = feature_matrix[feature_names]
            else:
                X = None
                kwargs = {}
                if model_name == "Holt-Winters (ETS)":
                    kwargs = {"seasonal_periods": seasonal_period, "auto_select": True}
                elif model_name == "SARIMA":
                    kwargs = {"seasonal_period": seasonal_period, "auto_select": True}
                elif model_name == "STL Decomposition":
                    kwargs = {"seasonal_period": seasonal_period}
                elif model_name == "Seasonal Naive":
                    kwargs = {"season_length": seasonal_period}
                elif model_name == "Moving Average":
                    kwargs = {"window": 13, "use_drift": True}

            def make_factory(name, kw):
                def factory(**extra):
                    merged = {**kw, **extra}
                    return create_model(name, **merged)
                return factory

            factory = make_factory(model_name, kwargs)

            result = run_backtest(
                model_factory=factory,
                y=series,
                config=config,
                X=X,
            )

            if result.folds:
                backtest_results[model_name] = result
                log_action(f"Backtest complete: {model_name} ({result.n_folds} folds)")

        except Exception as e:
            st.warning(f"Backtest failed for {model_name}: {e}")
            continue

    progress.progress(1.0)
    status.text("Backtesting complete!")

    if backtest_results:
        st.session_state["backtest_results"] = backtest_results

        # Compare models
        comparison_df = compare_models(backtest_results)
        st.session_state["comparison_df"] = comparison_df
        set_stage(BACKTEST_DONE)

        st.subheader("Model Comparison")
        st.dataframe(comparison_df, use_container_width=True)

        # Visual comparison
        metric_to_show = st.selectbox(
            "Compare by metric:",
            [c for c in comparison_df.columns if c not in ("Model", "Rank")],
        )
        fig = plot_model_comparison(comparison_df, metric_to_show)
        st.plotly_chart(fig, use_container_width=True)

        # Per-fold details
        st.subheader("Per-Fold Results")
        for model_name, result in backtest_results.items():
            with st.expander(f"{model_name} ({result.n_folds} folds)"):
                fold_df = result.metrics_by_fold()
                st.dataframe(fold_df, use_container_width=True)

                # Backtest visualization
                fig = plot_backtest(dates, series, result.folds, model_name)
                st.plotly_chart(fig, use_container_width=True)

        # Prediction interval evaluation
        st.subheader("Prediction Interval Coverage")
        coverage_data = []
        for model_name, result in backtest_results.items():
            agg = result.aggregate_metrics()
            cov80 = agg.get("Coverage_80")
            cov95 = agg.get("Coverage_95")
            if cov80 is not None or cov95 is not None:
                coverage_data.append({
                    "Model": model_name,
                    "80% CI Coverage": f"{cov80:.1f}%" if cov80 else "N/A",
                    "95% CI Coverage": f"{cov95:.1f}%" if cov95 else "N/A",
                    "80% Target": "80%",
                    "95% Target": "95%",
                })
        if coverage_data:
            st.dataframe(pd.DataFrame(coverage_data), use_container_width=True)
            st.caption("Good coverage: actual coverage close to the nominal level (80% or 95%).")

    else:
        st.error("No models completed backtesting successfully.")

elif st.session_state.get("comparison_df") is not None:
    # Show existing results
    comparison_df = st.session_state["comparison_df"]
    st.success("Backtest results loaded from previous run.")
    st.dataframe(comparison_df, use_container_width=True)
