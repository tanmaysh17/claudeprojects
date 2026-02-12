"""Reusable layout helpers for Streamlit pages."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from .session import STAGES, STAGE_LABELS, is_stage_complete


def page_header(title: str, description: str = ""):
    """Render a standardized page header."""
    st.title(title)
    if description:
        st.caption(description)
    st.divider()


def pipeline_progress_sidebar():
    """Show pipeline progress in the sidebar."""
    with st.sidebar:
        st.subheader("Pipeline Progress")
        for stage in STAGES:
            label = STAGE_LABELS.get(stage, stage)
            done = is_stage_complete(stage)
            icon = "+" if done else " "
            st.text(f"[{icon}] {label}")

        st.divider()
        if st.button("Reset Pipeline", type="secondary"):
            from .session import reset_pipeline
            reset_pipeline()
            st.rerun()


def show_dataframe_summary(df: pd.DataFrame, title: str = "Data Summary"):
    """Show a compact dataframe summary in an expander."""
    with st.expander(title, expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{len(df.columns):,}")
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        st.write("**Column Types:**")
        type_counts = df.dtypes.value_counts()
        type_str = ", ".join(f"{v} {k}" for k, v in type_counts.items())
        st.text(type_str)

        st.write("**First 5 Rows:**")
        st.dataframe(df.head(), use_container_width=True)


def metric_row(metrics: dict[str, float | str], columns: int = 4):
    """Display metrics in a row of columns."""
    cols = st.columns(columns)
    for i, (label, value) in enumerate(metrics.items()):
        col = cols[i % columns]
        if isinstance(value, float):
            col.metric(label, f"{value:.2f}")
        else:
            col.metric(label, str(value))
