"""Plotly chart builders for the forecast application."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Consistent color palette
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    title: str = "Time Series",
) -> go.Figure:
    """Plot one or more time series."""
    fig = go.Figure()
    for i, col in enumerate(value_cols):
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[col],
            name=col,
            line=dict(color=COLORS[i % len(COLORS)]),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_decomposition(
    observed: pd.Series,
    trend: pd.Series,
    seasonal: pd.Series,
    residual: pd.Series,
    title: str = "Time Series Decomposition",
) -> go.Figure:
    """Plot 4-panel decomposition chart."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.06,
    )
    fig.add_trace(go.Scatter(y=observed, name="Observed", line=dict(color=COLORS[0])), row=1, col=1)
    fig.add_trace(go.Scatter(y=trend, name="Trend", line=dict(color=COLORS[1])), row=2, col=1)
    fig.add_trace(go.Scatter(y=seasonal, name="Seasonal", line=dict(color=COLORS[2])), row=3, col=1)
    fig.add_trace(go.Scatter(y=residual, name="Residual", line=dict(color=COLORS[3])), row=4, col=1)

    fig.update_layout(
        title=title,
        height=700,
        showlegend=False,
        template="plotly_white",
    )
    return fig


def plot_forecast(
    dates_actual: pd.Series,
    actual: pd.Series,
    forecasts: dict[str, pd.DataFrame],
    title: str = "Forecast",
    show_ci: bool = True,
) -> go.Figure:
    """Plot actuals + multiple model forecasts with CI ribbons."""
    fig = go.Figure()

    # Actuals
    fig.add_trace(go.Scatter(
        x=dates_actual,
        y=actual,
        name="Actual",
        line=dict(color="black", width=2),
    ))

    # Forecast start date
    last_actual_date = dates_actual.iloc[-1]

    for i, (model_name, fc_df) in enumerate(forecasts.items()):
        color = COLORS[i % len(COLORS)]

        # Create future dates
        n_fc = len(fc_df)
        future_dates = pd.date_range(
            start=last_actual_date + pd.Timedelta(weeks=1),
            periods=n_fc,
            freq="W",
        )

        # Forecast line
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=fc_df["forecast"],
            name=model_name,
            line=dict(color=color, width=2),
        ))

        # Confidence intervals
        if show_ci and "lower_95" in fc_df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                y=pd.concat([fc_df["upper_95"].reset_index(drop=True), fc_df["lower_95"].iloc[::-1].reset_index(drop=True)]),
                fill="toself",
                fillcolor=f"rgba({_hex_to_rgb(color)}, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{model_name} 95% CI",
                showlegend=False,
            ))

        if show_ci and "lower_80" in fc_df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                y=pd.concat([fc_df["upper_80"].reset_index(drop=True), fc_df["lower_80"].iloc[::-1].reset_index(drop=True)]),
                fill="toself",
                fillcolor=f"rgba({_hex_to_rgb(color)}, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{model_name} 80% CI",
                showlegend=False,
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_backtest(
    dates: pd.Series,
    actual: pd.Series,
    fold_results: list,
    model_name: str = "Model",
) -> go.Figure:
    """Plot backtest: train/test splits with predictions vs actuals."""
    fig = go.Figure()

    # Full actuals
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        name="Actual",
        line=dict(color="black", width=1.5),
    ))

    for fold in fold_results:
        color = COLORS[fold.fold % len(COLORS)]
        test_dates = dates.iloc[fold.test_start:fold.test_end]

        # Predicted
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=fold.predicted,
            name=f"Fold {fold.fold + 1} Predicted",
            line=dict(color=color, dash="dash", width=2),
        ))

        # Highlight test region
        fig.add_vrect(
            x0=test_dates.iloc[0],
            x1=test_dates.iloc[-1],
            fillcolor=color,
            opacity=0.05,
            line_width=0,
        )

    fig.update_layout(
        title=f"Backtest Results: {model_name}",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_acf(acf_vals: np.ndarray, title: str = "Autocorrelation (ACF)") -> go.Figure:
    """Plot autocorrelation function."""
    n = len(acf_vals)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(n)),
        y=acf_vals,
        marker_color=COLORS[0],
        width=0.3,
    ))

    # Significance bounds (approximate)
    if n > 0:
        bound = 1.96 / np.sqrt(n * 5)  # rough approximation
        fig.add_hline(y=bound, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=-bound, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Lag",
        yaxis_title="ACF",
        template="plotly_white",
        height=350,
    )
    return fig


def plot_residual_diagnostics(residuals: np.ndarray, title: str = "Residual Diagnostics") -> go.Figure:
    """Plot residual histogram and time series."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Residuals Over Time", "Residual Distribution"),
    )

    fig.add_trace(
        go.Scatter(y=residuals, mode="lines", name="Residuals", line=dict(color=COLORS[0])),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(
        go.Histogram(x=residuals, name="Distribution", marker_color=COLORS[1], nbinsx=30),
        row=1, col=2,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=350,
        showlegend=False,
    )
    return fig


def plot_growth_rates(
    dates: pd.Series,
    growth_data: dict[str, pd.Series],
    title: str = "Growth Rates",
) -> go.Figure:
    """Plot multiple growth rate series."""
    fig = go.Figure()
    for i, (name, series) in enumerate(growth_data.items()):
        fig.add_trace(go.Scatter(
            x=dates,
            y=series,
            name=name,
            line=dict(color=COLORS[i % len(COLORS)]),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth (%)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_seasonal_indices(indices: pd.Series, title: str = "Seasonal Indices by Week") -> go.Figure:
    """Plot seasonal indices as a bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(indices) + 1)),
        y=indices.values,
        marker_color=[COLORS[0] if v >= 1 else COLORS[3] for v in indices.values],
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Week of Year",
        yaxis_title="Seasonal Index",
        template="plotly_white",
        height=350,
    )
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = "MAPE") -> go.Figure:
    """Horizontal bar chart comparing models on a metric."""
    if metric not in comparison_df.columns:
        return go.Figure()

    df = comparison_df.sort_values(metric, ascending=True)
    colors = [COLORS[2] if i == 0 else COLORS[0] for i in range(len(df))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[metric],
        y=df["Model"],
        orientation="h",
        marker_color=colors,
        text=df[metric].round(2),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Model Comparison: {metric}",
        xaxis_title=metric,
        template="plotly_white",
        height=max(300, len(df) * 50),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to rgb string for rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
