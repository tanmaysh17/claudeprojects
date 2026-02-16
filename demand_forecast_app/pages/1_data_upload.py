"""Page 1: Data Upload, Column Mapping, and Validation."""

import streamlit as st
import pandas as pd
from io import BytesIO

from ui.layout import page_header, show_dataframe_summary
from ui.session import set_stage, log_action, DATA_LOADED, DATA_VALIDATED
from core.data.ingestion import load_file
from core.data.column_mapping import auto_detect_columns, ColumnMapping
from core.data.validation import validate_schema, check_data_quality
from core.data.preprocessing import prepare_time_series

page_header(
    "Data Upload & Validation",
    "Upload an Excel or CSV file with weekly time series data. The system will auto-detect columns and validate data quality.",
)

# --- Template Download ---
with st.expander("Download Template"):
    st.write("Use this template format for best results:")
    template_df = pd.DataFrame({
        "date": pd.date_range("2020-01-06", periods=5, freq="W-MON"),
        "sales": [100, 120, 115, 130, 125],
        "price": [9.99, 9.99, 8.99, 8.99, 9.49],
        "promotion": [0, 0, 1, 1, 0],
    })
    st.dataframe(template_df, use_container_width=True)

    buf = BytesIO()
    template_df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    st.download_button(
        "Download Template (.xlsx)",
        data=buf,
        file_name="demand_forecast_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --- File Upload ---
st.subheader("1. Upload Data File")
uploaded_file = st.file_uploader(
    "Choose an Excel (.xlsx) or CSV file",
    type=["xlsx", "xls", "csv"],
    help="The file must contain at least a date column and a numeric sales/demand column.",
)

if uploaded_file is not None:
    try:
        result = load_file(uploaded_file, uploaded_file.name)
        raw_df = result.df
        st.session_state["raw_df"] = raw_df
        set_stage(DATA_LOADED)
        log_action(f"Uploaded file: {uploaded_file.name} ({result.shape[0]} rows, {result.shape[1]} cols)")

        show_dataframe_summary(raw_df, "Uploaded Data Summary")

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # --- Column Mapping ---
    st.subheader("2. Column Mapping")
    auto_mapping = auto_detect_columns(raw_df)

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox(
            "Date Column",
            options=raw_df.columns.tolist(),
            index=raw_df.columns.tolist().index(auto_mapping.date_col) if auto_mapping.date_col in raw_df.columns else 0,
        )
    with col2:
        non_date_cols = [c for c in raw_df.columns if c != date_col]
        target_default = 0
        if auto_mapping.target_col and auto_mapping.target_col in non_date_cols:
            target_default = non_date_cols.index(auto_mapping.target_col)
        target_col = st.selectbox(
            "Target (Sales/Demand) Column",
            options=non_date_cols,
            index=target_default,
        )

    # Optional exogenous columns
    remaining_cols = [c for c in raw_df.columns if c not in (date_col, target_col)]
    numeric_remaining = [c for c in remaining_cols if pd.api.types.is_numeric_dtype(raw_df[c])]
    exog_cols = st.multiselect(
        "External Regressors (optional)",
        options=numeric_remaining,
        default=[],
        help="Select numeric columns to use as external features (e.g., price, promotions).",
    )

    mapping = ColumnMapping(
        date_col=date_col,
        target_col=target_col,
        exog_cols=exog_cols,
    )
    st.session_state["mapping"] = mapping

    # --- Validation ---
    st.subheader("3. Data Validation")

    if st.button("Validate Data", type="primary"):
        schema_report = validate_schema(raw_df, mapping)
        quality_report = check_data_quality(raw_df, mapping)

        st.session_state["schema_report"] = schema_report
        st.session_state["quality_report"] = quality_report

        # Display schema validation
        if schema_report.is_valid:
            st.success("Schema validation passed.")
        else:
            for issue in schema_report.errors:
                st.error(f"[{issue.category}] {issue.message}")
            st.stop()

        # Display quality report
        st.write("**Data Quality Report:**")

        if quality_report.stats:
            stats = quality_report.stats
            cols = st.columns(4)
            cols[0].metric("Valid Data Points", f"{stats.get('n_valid_points', 'N/A'):,}")
            if stats.get("date_range") and stats["date_range"][0] != "N/A":
                cols[1].metric("Date Range", f"{stats['date_range'][0]} to {stats['date_range'][1]}")
            cols[2].metric("Mean", f"{stats.get('target_mean', 0):,.1f}")
            cols[3].metric("Std Dev", f"{stats.get('target_std', 0):,.1f}")

        if quality_report.issues:
            for issue in quality_report.issues:
                if issue.severity == "error":
                    st.error(f"[{issue.category}] {issue.message}")
                elif issue.severity == "warning":
                    st.warning(f"[{issue.category}] {issue.message}")
                else:
                    st.info(f"[{issue.category}] {issue.message}")
                if issue.details:
                    st.caption(f"  Detail: {issue.details}")
        else:
            st.success("No data quality issues detected.")

        # Preprocess data
        try:
            fill_method = st.selectbox(
                "Missing Value Strategy",
                ["interpolate", "ffill", "zero"],
                index=0,
                help="How to fill missing values in the target column.",
            )
            prepared_df = prepare_time_series(raw_df, mapping, fill_method=fill_method)
            st.session_state["prepared_df"] = prepared_df
            set_stage(DATA_VALIDATED)
            log_action(f"Data validated and preprocessed ({len(prepared_df)} rows)")

            st.success(f"Data prepared: {len(prepared_df)} rows ready for analysis.")

            st.write("**Prepared Data Preview:**")
            st.dataframe(prepared_df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error during preprocessing: {e}")

else:
    st.info("Please upload a file to get started.")
