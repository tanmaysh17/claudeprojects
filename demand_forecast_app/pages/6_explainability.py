"""Page 6: Model Explainability and Recommendations."""

import streamlit as st
import pandas as pd

from ui.layout import page_header
from ui.session import require_stage, BACKTEST_DONE, MODELS_RUN, log_action
from ui.charts import plot_model_comparison
from core.explainability.ranking import rank_models, select_best_model
from core.explainability.summary import generate_summary, generate_methodology_note
from core.explainability.feature_importance import get_model_feature_importance

page_header(
    "Explainability & Recommendations",
    "Understand why a model was selected and review forecast reliability.",
)

has_backtest = require_stage(BACKTEST_DONE, "Run backtesting first for full explainability. Partial results available if models are fitted.")

comparison_df = st.session_state.get("comparison_df")
forecasts = st.session_state.get("forecasts", {})
fitted_models = st.session_state.get("fitted_models", {})
backtest_results = st.session_state.get("backtest_results", {})
quality_report = st.session_state.get("quality_report")

if comparison_df is None or len(comparison_df) == 0:
    st.info("No model comparison data available. Please run backtesting first.")
    st.stop()

# --- Model Ranking ---
st.subheader("Model Ranking")

primary_metric = st.selectbox(
    "Rank by metric:",
    [c for c in comparison_df.columns if c not in ("Model", "Rank")],
    index=0,
)

ranked_df = rank_models(comparison_df, primary_metric=primary_metric)
st.dataframe(ranked_df, use_container_width=True)

best_model_name = ranked_df["Model"].iloc[0] if len(ranked_df) > 0 else ""
st.session_state["best_model"] = best_model_name

if best_model_name:
    st.success(f"Recommended model: **{best_model_name}**")

# Visual comparison
fig = plot_model_comparison(ranked_df, primary_metric)
st.plotly_chart(fig, use_container_width=True)

# --- Plain-Language Summary ---
st.subheader("Forecast Explanation")

data_stats = {}
if quality_report and hasattr(quality_report, "stats"):
    data_stats = quality_report.stats

n_folds = 3
if backtest_results and best_model_name in backtest_results:
    n_folds = backtest_results[best_model_name].n_folds

forecast_df = forecasts.get(best_model_name)

summary_text = generate_summary(
    comparison_df=comparison_df,
    best_model_name=best_model_name,
    forecast_df=forecast_df,
    data_stats=data_stats,
    n_folds=n_folds,
)

st.session_state["summary_text"] = summary_text

for paragraph in summary_text.split("\n\n"):
    if paragraph.startswith("MODEL SELECTION:"):
        st.info(paragraph)
    elif paragraph.startswith("BASELINE COMPARISON:"):
        st.write(paragraph)
    elif paragraph.startswith("FORECAST OVERVIEW:"):
        st.write(paragraph)
    elif paragraph.startswith("GROWTH OUTLOOK:"):
        st.write(paragraph)
    elif paragraph.startswith("RISKS"):
        st.warning(paragraph)
    elif paragraph.startswith("RELIABILITY:"):
        if "high" in paragraph.lower():
            st.success(paragraph)
        elif "low" in paragraph.lower():
            st.error(paragraph)
        else:
            st.write(paragraph)
    else:
        st.write(paragraph)

# Methodology note
methodology = generate_methodology_note(
    model_name=best_model_name,
    n_folds=n_folds,
    test_size=st.session_state.get("backtest_config", {}).get("test_size", 13) if isinstance(st.session_state.get("backtest_config"), dict) else 13,
    data_points=len(st.session_state.get("prepared_df", [])),
)
with st.expander("Methodology Note"):
    st.write(methodology)

# --- Feature Importance (for ML models) ---
if fitted_models:
    ml_models_with_importance = []
    for name, model in fitted_models.items():
        imp = get_model_feature_importance(model)
        if imp is not None:
            ml_models_with_importance.append((name, imp))

    if ml_models_with_importance:
        st.subheader("Feature Importance")
        for name, imp_df in ml_models_with_importance:
            st.write(f"**{name}:**")
            top_n = min(15, len(imp_df))
            st.dataframe(imp_df.head(top_n), use_container_width=True)

# --- Individual Model Summaries ---
st.subheader("Model Details")
for name, model in fitted_models.items():
    with st.expander(name):
        st.write(model.summary())
        params = model.get_params()
        st.json(params)

log_action("Explainability analysis viewed")
