"""Reusable Streamlit widget patterns."""

from __future__ import annotations

import streamlit as st

from core.models.registry import get_available_models, get_statistical_models, get_ml_models, BASELINE_MODELS, AUTO_MODELS
from core.backtesting.cross_validation import BacktestConfig


def model_selector() -> list[str]:
    """Model selection widget with Auto mode."""
    mode = st.radio(
        "Model Selection Mode",
        ["Auto (Recommended)", "Manual"],
        horizontal=True,
        help="Auto mode runs baselines plus best statistical models. Manual lets you pick specific models.",
    )

    if mode == "Auto (Recommended)":
        st.info(f"Auto mode will run: {', '.join(AUTO_MODELS)}")
        return list(AUTO_MODELS)
    else:
        all_models = get_available_models()
        # Separate into categories
        stat_models = get_statistical_models()
        ml_models = get_ml_models()
        ensemble_models = [m for m in all_models if "Ensemble" in m]

        selected = []

        st.write("**Statistical Models:**")
        for m in stat_models:
            default = m in BASELINE_MODELS
            if st.checkbox(m, value=default, key=f"model_{m}"):
                selected.append(m)

        st.write("**ML Models:**")
        for m in ml_models:
            if st.checkbox(m, value=False, key=f"model_{m}"):
                selected.append(m)

        st.write("**Ensemble:**")
        for m in ensemble_models:
            if st.checkbox(m, value=False, key=f"model_{m}"):
                selected.append(m)

        if not selected:
            st.warning("Please select at least one model.")

        return selected


def horizon_slider(max_val: int = 204) -> int:
    """Forecast horizon selection slider."""
    return st.slider(
        "Forecast Horizon (weeks)",
        min_value=52,
        max_value=max_val,
        value=52,
        step=1,
        help="Number of weeks to forecast ahead (52 = 1 year, 104 = 2 years, 156 = 3 years, 204 = ~4 years).",
    )


def cv_config_panel() -> BacktestConfig:
    """Cross-validation configuration panel."""
    with st.expander("Cross-Validation Settings", expanded=False):
        strategy = st.selectbox(
            "CV Strategy",
            ["rolling", "expanding"],
            help="Rolling: fixed-size training window. Expanding: growing training window.",
        )
        n_splits = st.slider("Number of CV Folds", 2, 10, 3)
        test_size = st.selectbox(
            "Test Window Size (weeks)",
            [13, 26, 52],
            index=0,
            help="13 weeks (~1 quarter), 26 weeks (~6 months), 52 weeks (~1 year).",
        )
        min_train = st.number_input(
            "Minimum Training Size (weeks)",
            min_value=52,
            max_value=520,
            value=104,
            step=13,
        )

    return BacktestConfig(
        strategy=strategy,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train,
    )
