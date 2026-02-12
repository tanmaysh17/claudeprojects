"""Page 3: Feature Engineering Configuration and Preview."""

import streamlit as st
import pandas as pd

from ui.layout import page_header, show_dataframe_summary
from ui.session import require_stage, set_stage, DATA_VALIDATED, FEATURES_BUILT, log_action
from core.features.pipeline import FeatureConfig, build_feature_matrix, get_feature_names
from core.features.holidays import DEFAULT_HOLIDAYS

page_header(
    "Feature Engineering",
    "Configure engineered features for ML-based forecasting models. Statistical models (ETS, SARIMA) do not use these features.",
)

if not require_stage(DATA_VALIDATED, "Please upload and validate data first."):
    st.stop()

prepared_df = st.session_state["prepared_df"]
mapping = st.session_state["mapping"]
date_col = mapping.date_col
target_col = mapping.target_col

st.info(
    "Feature engineering is used by ML models (Ridge, Random Forest, Gradient Boosting). "
    "Statistical models like Holt-Winters and SARIMA use their own internal feature extraction. "
    "You can skip this step if you only plan to use statistical models."
)

# --- Feature Configuration ---
st.subheader("Feature Configuration")

col1, col2 = st.columns(2)

with col1:
    use_lags = st.checkbox("Lag Features", value=True, help="Past values of the target as predictors.")
    if use_lags:
        lag_values = st.multiselect(
            "Lag Values (weeks back)",
            options=[1, 2, 3, 4, 8, 13, 26, 39, 52],
            default=[1, 2, 4, 13, 26, 52],
        )
    else:
        lag_values = []

    use_rolling = st.checkbox("Rolling Window Features", value=True, help="Rolling statistics (mean, std) as features.")
    if use_rolling:
        rolling_windows = st.multiselect(
            "Rolling Window Sizes (weeks)",
            options=[4, 8, 13, 26, 52],
            default=[4, 13, 52],
        )
    else:
        rolling_windows = []

with col2:
    use_calendar = st.checkbox("Calendar Features", value=True, help="Week of year, month, quarter, Fourier terms.")
    use_holidays = st.checkbox("Holiday Features", value=True, help="Approximate holiday period flags.")
    if use_holidays:
        st.write("**Included Holidays:**")
        for name in list(DEFAULT_HOLIDAYS.keys())[:6]:
            st.text(f"  - {name.replace('_', ' ').title()}")
        if len(DEFAULT_HOLIDAYS) > 6:
            st.text(f"  + {len(DEFAULT_HOLIDAYS) - 6} more")

config = FeatureConfig(
    use_lags=use_lags,
    lag_values=lag_values,
    use_rolling=use_rolling,
    rolling_windows=rolling_windows,
    rolling_agg_funcs=["mean", "std"],
    use_calendar=use_calendar,
    use_holidays=use_holidays,
)

# --- Build Features ---
if st.button("Build Feature Matrix", type="primary"):
    with st.spinner("Engineering features..."):
        feature_matrix = build_feature_matrix(
            prepared_df,
            date_col=date_col,
            target_col=target_col,
            config=config,
            exog_cols=mapping.exog_cols,
        )

        st.session_state["feature_config"] = config
        st.session_state["feature_matrix"] = feature_matrix
        set_stage(FEATURES_BUILT)
        log_action(f"Feature matrix built: {feature_matrix.shape[1] - 2} features")

        # Display results
        feature_names = get_feature_names(feature_matrix, date_col, target_col)

        st.success(f"Feature matrix built: {len(feature_names)} features across {len(feature_matrix)} rows.")

        # Feature summary
        col1, col2, col3 = st.columns(3)
        lag_feats = [f for f in feature_names if f.startswith("lag_")]
        rolling_feats = [f for f in feature_names if f.startswith("rolling_")]
        other_feats = [f for f in feature_names if f not in lag_feats and f not in rolling_feats]
        col1.metric("Lag Features", len(lag_feats))
        col2.metric("Rolling Features", len(rolling_feats))
        col3.metric("Calendar/Holiday/Other", len(other_feats))

        # Preview
        st.write("**Feature Matrix Preview (first 10 rows):**")
        st.dataframe(feature_matrix.head(10), use_container_width=True)

        # NaN analysis
        nan_counts = feature_matrix[feature_names].isna().sum()
        if nan_counts.sum() > 0:
            with st.expander("NaN Analysis"):
                nan_df = nan_counts[nan_counts > 0].reset_index()
                nan_df.columns = ["Feature", "NaN Count"]
                nan_df["NaN %"] = (nan_df["NaN Count"] / len(feature_matrix) * 100).round(1)
                st.dataframe(nan_df, use_container_width=True)
                st.caption("NaN values in lag/rolling features are expected for the first rows. These rows will be excluded during ML model training.")

        with st.expander("All Feature Names"):
            for i, name in enumerate(feature_names, 1):
                st.text(f"  {i}. {name}")

elif st.session_state.get("feature_matrix") is not None:
    feature_matrix = st.session_state["feature_matrix"]
    feature_names = get_feature_names(feature_matrix, date_col, target_col)
    st.success(f"Feature matrix loaded: {len(feature_names)} features.")
    show_dataframe_summary(feature_matrix, "Feature Matrix Summary")
