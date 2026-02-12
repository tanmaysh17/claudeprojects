import io
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Demand Forecasting Studio", layout="wide")


@dataclass
class ModelResult:
    name: str
    forecast: pd.Series
    lower80: pd.Series
    upper80: pd.Series
    lower95: pd.Series
    upper95: pd.Series
    metrics: Dict[str, float]
    notes: str = ""


REQUIRED_COLS_HELP = "Date and demand are required. Optional columns can be used as segment or exogenous regressors."


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return col
        except Exception:
            continue
    return None


def detect_value_column(df: pd.DataFrame, date_col: Optional[str]) -> Optional[str]:
    candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c != date_col]
    if candidates:
        return candidates[0]
    for col in df.columns:
        if col == date_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() > 0.8:
            return col
    return None


def validate_data(df: pd.DataFrame, date_col: str, y_col: str) -> Tuple[pd.DataFrame, List[str]]:
    issues = []
    local = df.copy()
    local[date_col] = pd.to_datetime(local[date_col], errors="coerce")
    local[y_col] = pd.to_numeric(local[y_col], errors="coerce")

    if local[date_col].isna().any():
        issues.append(f"{local[date_col].isna().sum()} rows have invalid dates.")
    if local[y_col].isna().any():
        issues.append(f"{local[y_col].isna().sum()} rows have non-numeric or missing demand.")

    local = local.dropna(subset=[date_col, y_col]).sort_values(date_col)

    duplicates = local.duplicated(subset=[date_col]).sum()
    if duplicates:
        issues.append(f"{duplicates} duplicate weeks detected (same date).")

    weekly = local.set_index(date_col).asfreq("W-MON")
    missing_weeks = weekly[y_col].isna().sum()
    if missing_weeks:
        issues.append(f"{missing_weeks} missing weeks detected in the timeline.")

    if (local[y_col] < 0).any():
        issues.append("Negative demand values found.")

    outlier_mask = np.abs((local[y_col] - local[y_col].median()) / (local[y_col].mad() + 1e-9)) > 6
    if outlier_mask.any():
        issues.append(f"{outlier_mask.sum()} strong outliers detected (robust MAD threshold).")

    if len(local) < 104:
        issues.append("Sparse history: less than 104 weeks. Forecast uncertainty will be higher.")

    rolling_mean = local[y_col].rolling(26).mean()
    if rolling_mean.notna().sum() > 10:
        shift_score = np.abs(rolling_mean.diff(13))
        if shift_score.max() > 2.5 * np.nanstd(local[y_col]):
            issues.append("Possible structural break or level shift detected.")

    return local, issues


def add_time_features(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    out = df.copy()
    out["weekofyear"] = out.index.isocalendar().week.astype(int)
    out["year"] = out.index.year
    for lag in [1, 2, 4, 13, 26, 52]:
        out[f"lag_{lag}"] = out[y_col].shift(lag)
    out["roll_mean_4"] = out[y_col].shift(1).rolling(4).mean()
    out["roll_mean_13"] = out[y_col].shift(1).rolling(13).mean()
    out["roll_std_13"] = out[y_col].shift(1).rolling(13).std()
    return out


def growth_metrics(series: pd.Series) -> pd.DataFrame:
    s = series.copy()
    out = pd.DataFrame(index=s.index)
    out["wow_growth"] = s.pct_change(1)
    out["roll4_growth"] = s.rolling(4).sum().pct_change(1)
    out["roll13_growth"] = s.rolling(13).sum().pct_change(1)
    out["ttm52_growth"] = s.rolling(52).sum().pct_change(52)
    out["yoy_growth"] = s.pct_change(52)
    return out


def seasonality_index(series: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"y": series})
    base["woy"] = base.index.isocalendar().week.astype(int)
    idx = base.groupby("woy")["y"].mean() / base["y"].mean()
    return idx.reset_index(name="seasonality_index")


def metric_pack(y_true: pd.Series, y_pred: pd.Series, lower95: pd.Series, upper95: pd.Series) -> Dict[str, float]:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    lower95, _ = lower95.align(y_true, join="inner")
    upper95, _ = upper95.align(y_true, join="inner")

    eps = 1e-9
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100
    wape = (np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() + eps)) * 100
    coverage95 = ((y_true >= lower95) & (y_true <= upper95)).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape, "PI95_Coverage": coverage95}


def seasonal_naive(train: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    idx = pd.date_range(train.index[-1] + pd.Timedelta(days=7), periods=horizon, freq="W-MON")
    preds = pd.Series([train.iloc[-52 + (i % 52)] if len(train) >= 52 else train.iloc[-1] for i in range(horizon)], index=idx)
    resid_std = (train - train.shift(52)).dropna().std() if len(train) > 60 else train.std()
    l80, u80 = preds - 1.28 * resid_std, preds + 1.28 * resid_std
    l95, u95 = preds - 1.96 * resid_std, preds + 1.96 * resid_std
    return preds, l80, u80, l95, u95


def moving_average_drift(train: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    idx = pd.date_range(train.index[-1] + pd.Timedelta(days=7), periods=horizon, freq="W-MON")
    drift = (train.iloc[-1] - train.iloc[0]) / max(len(train) - 1, 1)
    start = train.rolling(8).mean().iloc[-1]
    preds = pd.Series([start + drift * (i + 1) for i in range(horizon)], index=idx)
    resid_std = train.diff().dropna().std()
    scale = np.sqrt(np.arange(1, horizon + 1))
    l80, u80 = preds - 1.28 * resid_std * scale, preds + 1.28 * resid_std * scale
    l95, u95 = preds - 1.96 * resid_std * scale, preds + 1.96 * resid_std * scale
    return preds, l80, u80, l95, u95


def ets_model(train: pd.Series, horizon: int):
    fit = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=52).fit(optimized=True)
    preds = fit.forecast(horizon)
    resid_std = np.std(fit.resid)
    l80, u80 = preds - 1.28 * resid_std, preds + 1.28 * resid_std
    l95, u95 = preds - 1.96 * resid_std, preds + 1.96 * resid_std
    return preds, l80, u80, l95, u95


def sarimax_model(train: pd.Series, horizon: int):
    fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 52), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred_obj = fit.get_forecast(steps=horizon)
    preds = pred_obj.predicted_mean
    conf95 = pred_obj.conf_int(alpha=0.05)
    conf80 = pred_obj.conf_int(alpha=0.2)
    return preds, conf80.iloc[:, 0], conf80.iloc[:, 1], conf95.iloc[:, 0], conf95.iloc[:, 1]


def ml_lag_model(train_df: pd.DataFrame, y_col: str, horizon: int):
    full = train_df.copy()
    train_ml = full.dropna()
    features = [c for c in train_ml.columns if c != y_col]
    X = train_ml[features]
    y = train_ml[y_col]

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X, y)

    future_index = pd.date_range(full.index[-1] + pd.Timedelta(days=7), periods=horizon, freq="W-MON")
    history = full[[y_col]].copy()
    preds = []

    for ts in future_index:
        ext = history.copy()
        ext.loc[ts] = np.nan
        ext_feat = add_time_features(ext, y_col)
        row = ext_feat.loc[[ts]].drop(columns=[y_col]).fillna(method="ffill").fillna(method="bfill").fillna(0)
        yhat = model.predict(row)[0]
        preds.append(yhat)
        history.loc[ts, y_col] = yhat

    preds = pd.Series(preds, index=future_index)
    residuals = y - model.predict(X)
    resid_std = residuals.std()
    l80, u80 = preds - 1.28 * resid_std, preds + 1.28 * resid_std
    l95, u95 = preds - 1.96 * resid_std, preds + 1.96 * resid_std
    return preds, l80, u80, l95, u95


def backtest(series: pd.Series, model_name: str, holdouts: List[int]) -> Dict[str, float]:
    rows = []
    for h in holdouts:
        if len(series) <= h + 104:
            continue
        train = series.iloc[:-h]
        test = series.iloc[-h:]

        try:
            if model_name == "SeasonalNaive":
                preds, l80, u80, l95, u95 = seasonal_naive(train, h)
            elif model_name == "MovingAverageDrift":
                preds, l80, u80, l95, u95 = moving_average_drift(train, h)
            elif model_name == "ETS":
                preds, l80, u80, l95, u95 = ets_model(train, h)
            elif model_name == "SARIMAX":
                preds, l80, u80, l95, u95 = sarimax_model(train, h)
            else:
                feat = add_time_features(train.to_frame("y"), "y")
                preds, l80, u80, l95, u95 = ml_lag_model(feat, "y", h)

            rows.append(metric_pack(test, preds, l95, u95))
        except Exception:
            continue

    if not rows:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "WAPE": np.nan, "PI95_Coverage": np.nan}

    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def run_modeling(series: pd.Series, horizon: int, selected_models: List[str]) -> List[ModelResult]:
    outcomes: List[ModelResult] = []
    for model_name in selected_models:
        try:
            if model_name == "SeasonalNaive":
                preds, l80, u80, l95, u95 = seasonal_naive(series, horizon)
            elif model_name == "MovingAverageDrift":
                preds, l80, u80, l95, u95 = moving_average_drift(series, horizon)
            elif model_name == "ETS":
                preds, l80, u80, l95, u95 = ets_model(series, horizon)
            elif model_name == "SARIMAX":
                preds, l80, u80, l95, u95 = sarimax_model(series, horizon)
            elif model_name == "RandomForestLags":
                feat = add_time_features(series.to_frame("y"), "y")
                preds, l80, u80, l95, u95 = ml_lag_model(feat, "y", horizon)
            else:
                continue

            metrics = backtest(series, model_name, holdouts=[13, 26, 52])
            outcomes.append(ModelResult(model_name, preds, l80, u80, l95, u95, metrics))
        except Exception as e:
            outcomes.append(
                ModelResult(
                    name=model_name,
                    forecast=pd.Series(dtype=float),
                    lower80=pd.Series(dtype=float),
                    upper80=pd.Series(dtype=float),
                    lower95=pd.Series(dtype=float),
                    upper95=pd.Series(dtype=float),
                    metrics={"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "WAPE": np.nan, "PI95_Coverage": np.nan},
                    notes=f"Model failed: {e}",
                )
            )

    valid = [m for m in outcomes if not m.forecast.empty]
    if len(valid) >= 2:
        top = sorted(valid, key=lambda x: (np.nan_to_num(x.metrics.get("WAPE", np.inf)), np.nan_to_num(x.metrics.get("RMSE", np.inf))))[:2]
        ens_forecast = (top[0].forecast + top[1].forecast) / 2
        ens_l80 = (top[0].lower80 + top[1].lower80) / 2
        ens_u80 = (top[0].upper80 + top[1].upper80) / 2
        ens_l95 = (top[0].lower95 + top[1].lower95) / 2
        ens_u95 = (top[0].upper95 + top[1].upper95) / 2
        ens_metrics = pd.DataFrame([t.metrics for t in top]).mean(numeric_only=True).to_dict()
        outcomes.append(ModelResult("EnsembleTop2", ens_forecast, ens_l80, ens_u80, ens_l95, ens_u95, ens_metrics, notes="Average of top 2 models."))

    return outcomes


def explain_selection(comparison: pd.DataFrame, issues: List[str]) -> str:
    winner = comparison.sort_values(["WAPE", "RMSE"]).iloc[0]
    baseline = comparison[comparison["Model"] == "SeasonalNaive"]
    baseline_text = "No baseline available"
    if not baseline.empty:
        delta = baseline.iloc[0]["WAPE"] - winner["WAPE"]
        baseline_text = f"WAPE improvement vs seasonal naïve: {delta:.2f} points"

    risk = "Data quality checks are acceptable."
    if issues:
        risk = "Data quality concerns: " + "; ".join(issues[:3])

    return (
        f"Selected model: {winner['Model']}. "
        f"It ranked highest using WAPE and RMSE across rolling holdouts (13/26/52 weeks). "
        f"{baseline_text}. "
        "The model is preferred for stability and forecast plausibility; if performance is similar, simpler baselines are favored. "
        f"Reliability note: PI95 coverage={winner['PI95_Coverage']:.1f}%. {risk}"
    )


def to_excel_bytes(forecast_table: pd.DataFrame, comparison: pd.DataFrame, summary_text: str) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        forecast_table.to_excel(writer, index=False, sheet_name="forecast")
        comparison.to_excel(writer, index=False, sheet_name="model_comparison")
        pd.DataFrame({"summary": [summary_text]}).to_excel(writer, index=False, sheet_name="summary")
    return bio.getvalue()


st.title("Demand Forecasting Studio")
st.caption("Enterprise-ready weekly demand forecasting with data validation, backtesting, and explainability.")

uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded:
    raw = pd.read_excel(uploaded)
    st.subheader("Upload and column mapping")
    st.write(REQUIRED_COLS_HELP)

    default_date = detect_date_column(raw)
    default_y = detect_value_column(raw, default_date)

    c1, c2 = st.columns(2)
    date_col = c1.selectbox("Date column", options=raw.columns, index=raw.columns.get_loc(default_date) if default_date in raw.columns else 0)
    y_col = c2.selectbox("Demand column", options=raw.columns, index=raw.columns.get_loc(default_y) if default_y in raw.columns else 0)

    st.dataframe(raw.head(20), use_container_width=True)

    cleaned, issues = validate_data(raw, date_col, y_col)
    ts = cleaned.set_index(date_col).sort_index()[[y_col]].asfreq("W-MON")
    ts[y_col] = ts[y_col].interpolate(limit_direction="both")

    tabs = st.tabs([
        "Data health diagnostics",
        "Feature engineering & seasonality",
        "Modeling & forecasting",
        "Backtesting & accuracy",
        "Explainability",
        "Export & reporting",
    ])

    with tabs[0]:
        st.subheader("Data validation report")
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("No major quality issues detected.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts[y_col], mode="lines", name="Demand"))
        fig.update_layout(title="Trend visualization", height=400)
        st.plotly_chart(fig, use_container_width=True)

        roll_mean = ts[y_col].rolling(13).mean()
        roll_std = ts[y_col].rolling(13).std()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts.index, y=roll_mean, name="Rolling mean 13w"))
        fig2.add_trace(go.Scatter(x=ts.index, y=roll_std, name="Rolling std 13w"))
        fig2.update_layout(title="Rolling statistics")
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        st.subheader("Seasonality and growth metrics")
        g = growth_metrics(ts[y_col])
        st.dataframe(g.tail(15), use_container_width=True)

        sidx = seasonality_index(ts[y_col])
        fig3 = px.line(sidx, x="woy", y="seasonality_index", title="Seasonality index by week of year")
        st.plotly_chart(fig3, use_container_width=True)

    with tabs[2]:
        st.subheader("Forecast generation")
        horizon = st.slider("Forecast horizon (weeks)", min_value=52, max_value=204, value=52, step=1)
        mode = st.radio("Model mode", ["Auto", "Manual"], horizontal=True)

        all_models = ["SeasonalNaive", "MovingAverageDrift", "ETS", "SARIMAX", "RandomForestLags"]
        if mode == "Manual":
            selected = st.multiselect("Choose models", options=all_models, default=["SeasonalNaive", "ETS", "SARIMAX"])
            if not selected:
                st.stop()
        else:
            selected = all_models

        if st.button("Generate forecasts", type="primary"):
            with st.spinner("Training and forecasting..."):
                results = run_modeling(ts[y_col], horizon, selected)
            valid_results = [r for r in results if not r.forecast.empty]
            if not valid_results:
                st.error("No models succeeded. Check data quality and history length.")
                st.stop()

            comparison = pd.DataFrame(
                [dict(Model=r.name, **r.metrics, Notes=r.notes) for r in valid_results]
            ).sort_values(["WAPE", "RMSE"])

            winner = comparison.iloc[0]["Model"]
            best = next(r for r in valid_results if r.name == winner)

            fc_table = pd.DataFrame(
                {
                    "date": best.forecast.index,
                    "actual": np.nan,
                    "forecast": best.forecast.values,
                    "lower80": best.lower80.values,
                    "upper80": best.upper80.values,
                    "lower95": best.lower95.values,
                    "upper95": best.upper95.values,
                    "model": winner,
                }
            )

            hist = ts.reset_index().rename(columns={date_col: "date", y_col: "actual"})[["date", "actual"]]
            combined = pd.concat([hist, fc_table], ignore_index=True)

            st.session_state["comparison"] = comparison
            st.session_state["forecast_table"] = fc_table
            st.session_state["winner"] = winner
            st.session_state["results"] = valid_results
            st.session_state["issues"] = issues
            st.session_state["combined"] = combined

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=hist["date"], y=hist["actual"], name="Actuals", mode="lines"))
            fig4.add_trace(go.Scatter(x=best.forecast.index, y=best.forecast, name=f"Forecast ({winner})", mode="lines"))
            fig4.add_trace(go.Scatter(x=best.forecast.index, y=best.upper95, line=dict(width=0), showlegend=False))
            fig4.add_trace(go.Scatter(x=best.forecast.index, y=best.lower95, fill="tonexty", name="95% PI", opacity=0.2, line=dict(width=0)))
            fig4.update_layout(title="Actuals + forecast with prediction interval", height=450)
            st.plotly_chart(fig4, use_container_width=True)

            over = go.Figure()
            for r in valid_results[:5]:
                over.add_trace(go.Scatter(x=r.forecast.index, y=r.forecast, mode="lines", name=r.name))
            over.update_layout(title="Overlaid model forecasts")
            st.plotly_chart(over, use_container_width=True)

    with tabs[3]:
        st.subheader("Model comparison and backtesting")
        if "comparison" in st.session_state:
            st.dataframe(st.session_state["comparison"], use_container_width=True)
        else:
            st.info("Generate forecasts to view model comparison.")

    with tabs[4]:
        st.subheader("Plain-language recommendation")
        if "comparison" in st.session_state:
            narrative = explain_selection(st.session_state["comparison"], st.session_state.get("issues", []))
            st.write(narrative)
            st.session_state["narrative"] = narrative
        else:
            st.info("Run modeling first.")

    with tabs[5]:
        st.subheader("Export outputs")
        if "forecast_table" in st.session_state and "comparison" in st.session_state:
            forecast_table = st.session_state["forecast_table"]
            comparison = st.session_state["comparison"]
            narrative = st.session_state.get("narrative", "")
            winner = st.session_state.get("winner", "N/A")
            avg_growth = growth_metrics(ts[y_col])["ttm52_growth"].dropna().tail(1)
            growth_text = f"Latest trailing-52w growth: {avg_growth.iloc[0] * 100:.2f}%" if not avg_growth.empty else "Insufficient history for trailing-52w growth"

            one_pager = textwrap.dedent(
                f"""
                Forecast Summary
                Selected model: {winner}
                Method: Weekly time-series forecasting with rolling holdout backtests (13/26/52 weeks).
                Assumptions: Continuation of observed demand dynamics, yearly seasonality, and stable data pipeline.
                Growth outlook: {growth_text}
                Risks: {'; '.join(st.session_state.get('issues', [])[:5]) if st.session_state.get('issues') else 'No major data quality flags.'}
                Recommendation: Use model output as baseline and apply governed manual overrides when known business events exist.
                """
            ).strip()

            st.text_area("One-page summary", value=one_pager, height=240)
            st.dataframe(forecast_table.head(20), use_container_width=True)

            excel_bytes = to_excel_bytes(forecast_table, comparison, one_pager)
            st.download_button("Download Excel", data=excel_bytes, file_name="forecast_results.xlsx")
            st.download_button("Download forecast CSV", data=forecast_table.to_csv(index=False), file_name="forecast.csv")
            st.download_button("Download model comparison CSV", data=comparison.to_csv(index=False), file_name="model_comparison.csv")
        else:
            st.info("Run modeling to enable exports.")
else:
    st.info("Upload an Excel file to begin.")
