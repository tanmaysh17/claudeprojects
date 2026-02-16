"""Page 4: Forecasting - Model Selection and Forecast Generation."""

import streamlit as st
import pandas as pd
import numpy as np

from ui.layout import page_header
from ui.session import (
    require_stage, set_stage, DATA_VALIDATED, FEATURES_BUILT, MODELS_RUN, log_action,
)
from ui.widgets import model_selector, horizon_slider
from ui.charts import plot_forecast, plot_time_series
from core.models.registry import create_model, get_ml_models, BASELINE_MODELS
from core.models.ensemble import SimpleAverageEnsemble, WeightedEnsemble
from core.features.pipeline import get_feature_names

page_header(
    "Forecasting",
    "Select models and generate demand forecasts with prediction intervals.",
)

if not require_stage(DATA_VALIDATED, "Please upload and validate data first."):
    st.stop()

prepared_df = st.session_state["prepared_df"]
mapping = st.session_state["mapping"]
date_col = mapping.date_col
target_col = mapping.target_col
series = prepared_df[target_col]
dates = prepared_df[date_col]

# --- Settings ---
st.subheader("Forecast Settings")

col1, col2 = st.columns(2)
with col1:
    horizon = horizon_slider()
    st.session_state["forecast_horizon"] = horizon
with col2:
    seasonal_period = st.number_input(
        "Seasonal Period (weeks)",
        min_value=2,
        max_value=104,
        value=52,
        help="52 for annual seasonality in weekly data.",
    )

# Model selection
st.subheader("Model Selection")
selected_models = model_selector()
st.session_state["selected_models"] = selected_models

# --- Run Forecasts ---
if st.button("Generate Forecasts", type="primary") and selected_models:
    forecasts = {}
    fitted_models = {}
    progress = st.progress(0)
    status = st.empty()

    ml_model_names = set(get_ml_models())
    feature_matrix = st.session_state.get("feature_matrix")

    for i, model_name in enumerate(selected_models):
        progress.progress((i) / len(selected_models))
        status.text(f"Fitting {model_name}...")

        try:
            # Handle ensemble models differently
            if "Ensemble" in model_name:
                continue  # Handle after individual models

            # Create model with appropriate kwargs
            kwargs = {"seasonal_period": seasonal_period}
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
            elif model_name in ml_model_names:
                if feature_matrix is None:
                    st.warning(f"Skipping {model_name}: Feature matrix not built. Go to Feature Engineering first.")
                    continue
                feature_names = get_feature_names(feature_matrix, date_col, target_col)
                lag_cols = [c for c in feature_names if c.startswith("lag_")]
                kwargs = {
                    "feature_cols": feature_names,
                    "lag_cols": lag_cols,
                    "target_col": target_col,
                    "date_col": date_col,
                }

            model = create_model(model_name, **kwargs)

            # Fit
            if model_name in ml_model_names:
                X_train = feature_matrix[feature_names] if feature_matrix is not None else None
                model.fit(series, X_train)
            else:
                exog = prepared_df[mapping.exog_cols] if mapping.exog_cols and model_name == "SARIMA" else None
                model.fit(series, exog)

            # Predict
            X_future = None
            if model_name in ml_model_names and feature_matrix is not None:
                # For ML models, we need future features - use recursive prediction
                pass
            forecast_df = model.predict(horizon, X_future, return_ci=True)

            forecasts[model_name] = forecast_df
            fitted_models[model_name] = model
            log_action(f"Generated forecast: {model_name} (horizon={horizon})")

        except Exception as e:
            st.warning(f"Error fitting {model_name}: {e}")
            continue

    # Handle ensemble models
    ensemble_models = [m for m in selected_models if "Ensemble" in m]
    if ensemble_models and len(fitted_models) >= 2:
        non_ensemble = [m for m in fitted_models.values() if "Ensemble" not in m.name]
        for ens_name in ensemble_models:
            try:
                status.text(f"Building {ens_name}...")
                if "Weighted" in ens_name:
                    ens = WeightedEnsemble()
                    ens.fit_from_pretrained(non_ensemble)
                else:
                    ens = SimpleAverageEnsemble()
                    ens.fit_from_pretrained(non_ensemble)

                forecast_df = ens.predict(horizon, return_ci=True)
                forecasts[ens_name] = forecast_df
                fitted_models[ens_name] = ens
            except Exception as e:
                st.warning(f"Error building {ens_name}: {e}")

    progress.progress(1.0)
    status.text("Done!")

    if forecasts:
        st.session_state["forecasts"] = forecasts
        st.session_state["fitted_models"] = fitted_models
        set_stage(MODELS_RUN)

        st.success(f"Generated {len(forecasts)} forecast(s) for {horizon} weeks.")

        # Forecast visualization
        st.subheader("Forecast Results")
        fig = plot_forecast(dates, series, forecasts, "Demand Forecast")
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.subheader("Forecast Summary")
        summary_rows = []
        last_actual = series.iloc[-1]
        for name, fc in forecasts.items():
            fc_vals = fc["forecast"]
            summary_rows.append({
                "Model": name,
                "Last Actual": round(last_actual, 1),
                "First Forecast": round(fc_vals.iloc[0], 1),
                "Mean Forecast": round(fc_vals.mean(), 1),
                "Min Forecast": round(fc_vals.min(), 1),
                "Max Forecast": round(fc_vals.max(), 1),
                "Forecast Growth (%)": round((fc_vals.mean() - last_actual) / last_actual * 100, 1) if last_actual != 0 else 0,
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        # Individual model details
        with st.expander("Model Details"):
            for name, model in fitted_models.items():
                st.write(f"**{name}:** {model.summary()}")

    else:
        st.error("No models produced forecasts. Check warnings above.")

elif st.session_state.get("forecasts"):
    # Show existing forecasts
    forecasts = st.session_state["forecasts"]
    st.success(f"{len(forecasts)} forecast(s) loaded from previous run.")

    fig = plot_forecast(dates, series, forecasts, "Demand Forecast")
    st.plotly_chart(fig, use_container_width=True)
