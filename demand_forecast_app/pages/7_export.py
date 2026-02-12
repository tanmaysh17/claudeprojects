"""Page 7: Export and Reporting."""

import streamlit as st
import pandas as pd
from datetime import datetime

from ui.layout import page_header
from ui.session import require_stage, MODELS_RUN, log_action
from core.export.excel_report import create_forecast_workbook
from core.export.csv_export import export_forecasts_csv, export_comparison_csv
from core.export.summary_report import build_text_report

page_header(
    "Export & Report",
    "Download forecasts, model comparison, and summary reports.",
)

if not require_stage(MODELS_RUN, "Please generate forecasts first."):
    st.stop()

forecasts = st.session_state.get("forecasts", {})
comparison_df = st.session_state.get("comparison_df")
summary_text = st.session_state.get("summary_text", "")
best_model = st.session_state.get("best_model", "")
prepared_df = st.session_state.get("prepared_df")
mapping = st.session_state.get("mapping")
quality_report = st.session_state.get("quality_report")

if not forecasts:
    st.info("No forecasts available to export.")
    st.stop()

date_col = mapping.date_col
target_col = mapping.target_col
dates = prepared_df[date_col]

# --- Build Forecast Table ---
st.subheader("Forecast Data")

# Combine all forecasts into one table
last_date = pd.to_datetime(dates.iloc[-1])
horizon = len(list(forecasts.values())[0])
future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")

export_df = pd.DataFrame({"date": future_dates})

for model_name, fc_df in forecasts.items():
    prefix = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    export_df[f"{prefix}_forecast"] = fc_df["forecast"].values
    if "lower_80" in fc_df.columns:
        export_df[f"{prefix}_lower_80"] = fc_df["lower_80"].values
        export_df[f"{prefix}_upper_80"] = fc_df["upper_80"].values
    if "lower_95" in fc_df.columns:
        export_df[f"{prefix}_lower_95"] = fc_df["lower_95"].values
        export_df[f"{prefix}_upper_95"] = fc_df["upper_95"].values

# Add actuals for context
actuals_df = pd.DataFrame({
    "date": dates,
    "actual": prepared_df[target_col].values,
})
full_df = pd.concat([actuals_df, export_df], ignore_index=True).sort_values("date")

st.write(f"**{len(export_df)} forecast rows, {len(actuals_df)} historical rows**")

with st.expander("Preview Forecast Table"):
    st.dataframe(export_df.head(20), use_container_width=True)

# --- Export Options ---
st.subheader("Download Options")

col1, col2, col3 = st.columns(3)

# CSV Export
with col1:
    st.write("**CSV Export**")
    csv_data = export_forecasts_csv(full_df)
    st.download_button(
        "Download Forecasts (CSV)",
        data=csv_data,
        file_name=f"demand_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    if comparison_df is not None:
        csv_comp = export_comparison_csv(comparison_df)
        st.download_button(
            "Download Model Comparison (CSV)",
            data=csv_comp,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# Excel Export
with col2:
    st.write("**Excel Report**")

    # Build quality issues dataframe
    quality_df = None
    if quality_report and quality_report.issues:
        quality_df = pd.DataFrame([
            {"Severity": i.severity, "Category": i.category, "Message": i.message, "Details": i.details}
            for i in quality_report.issues
        ])

    excel_data = create_forecast_workbook(
        forecast_df=full_df,
        comparison_df=comparison_df,
        summary_text=summary_text,
        data_quality_df=quality_df,
    )

    st.download_button(
        "Download Full Report (Excel)",
        data=excel_data,
        file_name=f"demand_forecast_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Text Report
with col3:
    st.write("**Text Summary**")

    data_summary = {}
    if quality_report and quality_report.stats:
        data_summary = {
            "Data Points": quality_report.stats.get("n_valid_points", "N/A"),
            "Date Range": f"{quality_report.stats.get('date_range', ('N/A', 'N/A'))[0]} to {quality_report.stats.get('date_range', ('N/A', 'N/A'))[1]}",
            "Target Mean": f"{quality_report.stats.get('target_mean', 0):.1f}",
            "Forecast Horizon": f"{horizon} weeks",
            "Selected Model": best_model,
        }

    text_report = build_text_report(
        data_summary=data_summary,
        model_comparison=comparison_df if comparison_df is not None else pd.DataFrame(),
        forecast_summary=summary_text,
    )

    st.download_button(
        "Download Summary (Text)",
        data=text_report,
        file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )

# --- Summary Preview ---
st.subheader("Report Preview")
if summary_text:
    st.text(summary_text)
else:
    st.info("Run explainability analysis to generate a summary.")

# --- Audit Trail ---
st.subheader("Run Log")
run_log = st.session_state.get("run_log", [])
if run_log:
    log_df = pd.DataFrame(run_log)
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No actions logged yet.")

log_action("Export page viewed")
