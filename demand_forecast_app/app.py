"""Demand Forecast Application - Main Entry Point.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Demand Forecasting Tool",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.session import init_session_state
from ui.layout import pipeline_progress_sidebar

# Initialize session state
init_session_state()

# Define pages
pages = {
    "Data": [
        st.Page("pages/1_data_upload.py", title="Upload & Validate", icon=":material/upload_file:"),
    ],
    "Analysis": [
        st.Page("pages/2_eda.py", title="Exploratory Analysis", icon=":material/insights:"),
        st.Page("pages/3_feature_engineering.py", title="Feature Engineering", icon=":material/settings:"),
    ],
    "Modeling": [
        st.Page("pages/4_forecasting.py", title="Forecasting", icon=":material/trending_up:"),
        st.Page("pages/5_backtesting.py", title="Backtesting", icon=":material/fact_check:"),
    ],
    "Results": [
        st.Page("pages/6_explainability.py", title="Explainability", icon=":material/lightbulb:"),
        st.Page("pages/7_export.py", title="Export & Report", icon=":material/download:"),
    ],
}

# Navigation
pg = st.navigation(pages)

# Sidebar: pipeline progress
pipeline_progress_sidebar()

# Sidebar: app info
with st.sidebar:
    st.divider()
    st.caption("Demand Forecasting Tool v1.0")
    st.caption("Upload weekly time series data to generate demand forecasts.")

# Run the selected page
pg.run()
