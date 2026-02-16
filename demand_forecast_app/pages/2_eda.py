"""Page 2: Exploratory Data Analysis."""

import streamlit as st
import pandas as pd
import numpy as np

from ui.layout import page_header, metric_row
from ui.session import require_stage, set_stage, DATA_VALIDATED, EDA_COMPLETE, log_action
from ui.charts import (
    plot_time_series,
    plot_decomposition,
    plot_acf,
    plot_growth_rates,
    plot_seasonal_indices,
    plot_residual_diagnostics,
)
from core.eda.trend import compute_trend, compute_growth_rates, compute_rolling_totals, compute_cagr
from core.eda.seasonality import compute_acf_pacf, detect_seasonal_period, compute_seasonal_indices, seasonal_strength
from core.eda.decomposition import decompose_stl, decompose_classical
from core.eda.rolling_stats import compute_rolling_statistics
from core.eda.outliers import get_outlier_summary
from core.eda.stationarity import adf_test, kpss_test, suggest_differencing

page_header(
    "Exploratory Data Analysis",
    "Analyze trends, seasonality, outliers, and data health before forecasting.",
)

if not require_stage(DATA_VALIDATED, "Please upload and validate data on the Upload page first."):
    st.stop()

prepared_df = st.session_state["prepared_df"]
mapping = st.session_state["mapping"]
date_col = mapping.date_col
target_col = mapping.target_col
dates = prepared_df[date_col]
series = prepared_df[target_col]

# Tabs for different analysis sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Trend & Overview",
    "Seasonality",
    "Decomposition",
    "Growth Rates",
    "Outliers",
    "Stationarity",
])

# --- Tab 1: Trend & Overview ---
with tab1:
    st.subheader("Time Series Overview")

    # Basic metrics
    metric_row({
        "Total Points": len(series),
        "Mean": round(series.mean(), 1),
        "Std Dev": round(series.std(), 1),
        "Min": round(series.min(), 1),
        "Max": round(series.max(), 1),
        "CV (%)": round(series.std() / series.mean() * 100, 1) if series.mean() != 0 else 0,
    }, columns=6)

    # Time series with trend
    trend = compute_trend(series, window=13)
    plot_df = prepared_df.copy()
    plot_df["Trend (13w MA)"] = trend.values
    fig = plot_time_series(plot_df, date_col, [target_col, "Trend (13w MA)"], "Time Series with Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Rolling statistics
    st.subheader("Rolling Statistics")
    roll_stats = compute_rolling_statistics(series, windows=[4, 13, 26, 52])
    roll_df = prepared_df[[date_col]].copy()
    roll_df[target_col] = series.values
    for col in ["rolling_13w_mean", "rolling_52w_mean"]:
        if col in roll_stats.columns:
            roll_df[col] = roll_stats[col].values
    fig = plot_time_series(roll_df, date_col, [target_col, "rolling_13w_mean", "rolling_52w_mean"], "Rolling Averages")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Seasonality ---
with tab2:
    st.subheader("Seasonality Analysis")

    # Detect seasonal period
    sp = detect_seasonal_period(series)
    st.info(f"Detected dominant seasonal period: **{sp} weeks** (approximately {'annual' if sp >= 48 and sp <= 56 else 'non-annual'} cycle)")

    # ACF / PACF
    acf_result = compute_acf_pacf(series, nlags=min(104, len(series) // 2 - 1))
    if len(acf_result["acf"]) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_acf(acf_result["acf"], "Autocorrelation (ACF)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_acf(acf_result["pacf"], "Partial Autocorrelation (PACF)")
            st.plotly_chart(fig, use_container_width=True)

    # Seasonal indices
    st.subheader("Seasonal Indices by Week")
    indices = compute_seasonal_indices(series, period=min(sp, 52))
    fig = plot_seasonal_indices(indices, f"Seasonal Indices (Period={min(sp, 52)})")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Seasonal Index Table"):
        idx_df = pd.DataFrame({
            "Week": range(1, len(indices) + 1),
            "Index": indices.values.round(3),
        })
        st.dataframe(idx_df, use_container_width=True)

# --- Tab 3: Decomposition ---
with tab3:
    st.subheader("Time Series Decomposition")

    decomp_method = st.radio("Decomposition Method", ["STL (Recommended)", "Classical"], horizontal=True)
    decomp_period = st.number_input("Seasonal Period", min_value=2, max_value=104, value=min(52, len(series) // 3))

    if st.button("Run Decomposition"):
        with st.spinner("Decomposing time series..."):
            if decomp_method.startswith("STL"):
                decomp = decompose_stl(series, period=decomp_period)
            else:
                model_type = st.selectbox("Model Type", ["additive", "multiplicative"])
                decomp = decompose_classical(series, period=decomp_period, model=model_type)

            st.session_state["decomposition"] = decomp

            # Seasonal strength
            ss = seasonal_strength(decomp.trend, decomp.seasonal, decomp.residual)
            st.metric("Seasonal Strength", f"{ss:.2%}", help="0 = no seasonality, 1 = pure seasonal pattern")

            # Decomposition plot
            fig = plot_decomposition(decomp.observed, decomp.trend, decomp.seasonal, decomp.residual)
            st.plotly_chart(fig, use_container_width=True)

            # Residual diagnostics
            fig = plot_residual_diagnostics(decomp.residual.dropna().values)
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Growth Rates ---
with tab4:
    st.subheader("Growth Rate Analysis")

    growth = compute_growth_rates(series)
    rolling_totals = compute_rolling_totals(series)

    # CAGR
    cagr = compute_cagr(series, periods_per_year=52)
    if cagr is not None:
        st.metric("CAGR (Compound Annual Growth Rate)", f"{cagr:.1f}%")

    # Growth rate charts
    growth_to_plot = {k: v for k, v in growth.items() if k in ["wow_growth", "rolling_13w_growth", "yoy_weekly_growth"]}
    if growth_to_plot:
        fig = plot_growth_rates(dates, growth_to_plot, "Growth Rates")
        st.plotly_chart(fig, use_container_width=True)

    # Rolling totals
    st.subheader("Rolling Totals")
    totals_df = prepared_df[[date_col]].copy()
    for name, vals in rolling_totals.items():
        totals_df[name] = vals.values
    fig = plot_time_series(totals_df, date_col, list(rolling_totals.keys()), "Rolling Totals")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Outliers ---
with tab5:
    st.subheader("Outlier Detection")

    outlier_summary = get_outlier_summary(series, dates)

    if len(outlier_summary) > 0:
        st.warning(f"Detected {len(outlier_summary)} potential outlier(s).")
        display_df = outlier_summary.copy()
        if "date" in display_df.columns:
            display_cols = ["date", "value", "iqr_outlier", "zscore_outlier", "rolling_zscore_outlier", "n_methods_flagged"]
        else:
            display_cols = ["value", "iqr_outlier", "zscore_outlier", "rolling_zscore_outlier", "n_methods_flagged"]
        st.dataframe(display_df[display_cols], use_container_width=True)
    else:
        st.success("No outliers detected.")

    # Visualize outliers on time series
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=series, name="Data", line=dict(color="#1f77b4")))
    if len(outlier_summary) > 0 and "date" in outlier_summary.columns:
        fig.add_trace(go.Scatter(
            x=outlier_summary["date"],
            y=outlier_summary["value"],
            mode="markers",
            name="Outliers",
            marker=dict(color="red", size=10, symbol="x"),
        ))
    fig.update_layout(title="Time Series with Outliers Highlighted", template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 6: Stationarity ---
with tab6:
    st.subheader("Stationarity Tests")

    adf_result = adf_test(series)
    kpss_result = kpss_test(series)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Augmented Dickey-Fuller Test**")
        st.write(f"Statistic: {adf_result.statistic:.4f}")
        st.write(f"P-value: {adf_result.p_value:.4f}")
        if adf_result.is_stationary:
            st.success("Series is likely stationary (ADF).")
        else:
            st.warning("Series may be non-stationary (ADF).")
        st.caption(adf_result.interpretation)

    with col2:
        st.write("**KPSS Test**")
        st.write(f"Statistic: {kpss_result.statistic:.4f}")
        st.write(f"P-value: {kpss_result.p_value:.4f}")
        if kpss_result.is_stationary:
            st.success("Series is likely stationary (KPSS).")
        else:
            st.warning("Series may be non-stationary (KPSS).")
        st.caption(kpss_result.interpretation)

    diff_order = suggest_differencing(series)
    st.info(f"Suggested differencing order: d={diff_order}")

# Mark EDA as complete
set_stage(EDA_COMPLETE)
log_action("EDA analysis completed")
