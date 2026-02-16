"""Session state management for the pipeline."""

from __future__ import annotations

import streamlit as st

# Pipeline stages
DATA_LOADED = "data_loaded"
DATA_VALIDATED = "data_validated"
EDA_COMPLETE = "eda_complete"
FEATURES_BUILT = "features_built"
MODELS_RUN = "models_run"
BACKTEST_DONE = "backtest_done"

STAGES = [DATA_LOADED, DATA_VALIDATED, EDA_COMPLETE, FEATURES_BUILT, MODELS_RUN, BACKTEST_DONE]

STAGE_LABELS = {
    DATA_LOADED: "Data Uploaded",
    DATA_VALIDATED: "Data Validated",
    EDA_COMPLETE: "EDA Complete",
    FEATURES_BUILT: "Features Built",
    MODELS_RUN: "Forecasts Generated",
    BACKTEST_DONE: "Backtesting Done",
}


def init_session_state():
    """Initialize default session state values."""
    defaults = {
        "pipeline_stages": {s: False for s in STAGES},
        "raw_df": None,
        "mapping": None,
        "schema_report": None,
        "quality_report": None,
        "prepared_df": None,
        "eda_results": {},
        "decomposition": None,
        "feature_config": None,
        "feature_matrix": None,
        "selected_models": [],
        "forecasts": {},
        "fitted_models": {},
        "forecast_horizon": 52,
        "backtest_results": {},
        "comparison_df": None,
        "best_model": None,
        "summary_text": "",
        "run_log": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def set_stage(stage: str, value: bool = True):
    """Mark a pipeline stage as complete or incomplete."""
    if "pipeline_stages" not in st.session_state:
        st.session_state["pipeline_stages"] = {s: False for s in STAGES}
    st.session_state["pipeline_stages"][stage] = value


def is_stage_complete(stage: str) -> bool:
    """Check if a pipeline stage is complete."""
    if "pipeline_stages" not in st.session_state:
        return False
    return st.session_state["pipeline_stages"].get(stage, False)


def require_stage(stage: str, message: str | None = None) -> bool:
    """Check if a prerequisite stage is complete. Shows warning and returns False if not."""
    if not is_stage_complete(stage):
        label = STAGE_LABELS.get(stage, stage)
        msg = message or f"Please complete the '{label}' step first."
        st.warning(msg)
        return False
    return True


def get_completed_stages() -> list[str]:
    """Get list of completed stage names."""
    if "pipeline_stages" not in st.session_state:
        return []
    return [s for s in STAGES if st.session_state["pipeline_stages"].get(s, False)]


def reset_pipeline():
    """Reset all pipeline state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


def log_action(message: str):
    """Add an entry to the run log."""
    if "run_log" not in st.session_state:
        st.session_state["run_log"] = []
    import datetime
    st.session_state["run_log"].append(
        {"timestamp": datetime.datetime.now().isoformat(), "message": message}
    )
