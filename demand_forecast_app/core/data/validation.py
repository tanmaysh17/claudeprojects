"""Data validation: schema checks and quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .column_mapping import ColumnMapping


@dataclass
class ValidationIssue:
    severity: str  # "error", "warning", "info"
    category: str
    message: str
    details: str = ""


@dataclass
class ValidationReport:
    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, severity: str, category: str, message: str, details: str = ""):
        issue = ValidationIssue(severity=severity, category=category, message=message, details=details)
        self.issues.append(issue)
        if severity == "error":
            self.is_valid = False

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


def validate_schema(df: pd.DataFrame, mapping: ColumnMapping) -> ValidationReport:
    report = ValidationReport()

    if mapping.date_col is None:
        report.add("error", "schema", "No date column identified. Please map a date column.")
        return report

    if mapping.target_col is None:
        report.add("error", "schema", "No target (demand/sales) column identified.")
        return report

    if mapping.date_col not in df.columns:
        report.add("error", "schema", f"Date column '{mapping.date_col}' not found in data.")
        return report

    if mapping.target_col not in df.columns:
        report.add("error", "schema", f"Target column '{mapping.target_col}' not found in data.")
        return report

    # Try parsing dates
    try:
        pd.to_datetime(df[mapping.date_col])
    except (ValueError, TypeError):
        report.add("error", "schema", f"Column '{mapping.date_col}' cannot be parsed as dates.")

    # Check target is numeric
    if not pd.api.types.is_numeric_dtype(df[mapping.target_col]):
        try:
            pd.to_numeric(df[mapping.target_col], errors="raise")
        except (ValueError, TypeError):
            report.add("error", "schema", f"Target column '{mapping.target_col}' contains non-numeric values.")

    report.stats["n_rows"] = len(df)
    report.stats["n_cols"] = len(df.columns)
    return report


def check_data_quality(df: pd.DataFrame, mapping: ColumnMapping) -> ValidationReport:
    report = ValidationReport()
    date_col = mapping.date_col
    target_col = mapping.target_col

    if date_col is None or target_col is None:
        report.add("error", "prerequisite", "Column mapping incomplete.")
        return report

    dates = pd.to_datetime(df[date_col], errors="coerce")
    target = pd.to_numeric(df[target_col], errors="coerce")

    # Duplicate dates
    n_dups = dates.duplicated().sum()
    if n_dups > 0:
        report.add("warning", "duplicates", f"{n_dups} duplicate date(s) found.", "Consider aggregating or removing duplicates.")

    # Missing values in target
    n_missing_target = target.isna().sum()
    pct_missing = n_missing_target / len(target) * 100 if len(target) > 0 else 0
    if n_missing_target > 0:
        report.add(
            "warning" if pct_missing < 10 else "error",
            "missing_values",
            f"{n_missing_target} missing value(s) in target ({pct_missing:.1f}%).",
        )

    # Missing dates (gaps)
    sorted_dates = dates.dropna().sort_values().reset_index(drop=True)
    if len(sorted_dates) > 1:
        diffs = sorted_dates.diff().dropna()
        median_diff = diffs.median()
        expected_gaps = diffs[diffs > median_diff * 1.5]
        n_gaps = len(expected_gaps)
        if n_gaps > 0:
            report.add("warning", "missing_weeks", f"{n_gaps} gap(s) detected in the date sequence.", f"Median interval: {median_diff.days} days.")

    # Negative values
    n_neg = (target < 0).sum()
    if n_neg > 0:
        report.add("warning", "negative_values", f"{n_neg} negative value(s) in target column.")

    # Outliers (IQR method)
    q1, q3 = target.quantile(0.25), target.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        outlier_mask = (target < q1 - 3 * iqr) | (target > q3 + 3 * iqr)
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            report.add("info", "outliers", f"{n_outliers} potential outlier(s) detected (IQR method).")

    # Short history
    n_valid = target.notna().sum()
    if n_valid < 52:
        report.add("warning", "sparse_history", f"Only {n_valid} valid data points. At least 52 weeks recommended for reliable forecasting.")
    elif n_valid < 104:
        report.add("info", "sparse_history", f"{n_valid} data points available. 104+ weeks recommended for models with annual seasonality.")

    # Constant or near-constant target
    if target.std() == 0:
        report.add("warning", "constant", "Target column has zero variance (constant value).")
    elif target.std() / (target.mean() + 1e-9) < 0.01:
        report.add("info", "low_variance", "Target has very low coefficient of variation.")

    # Structural break detection (simple: compare first-half mean vs second-half mean)
    if n_valid >= 52:
        mid = n_valid // 2
        sorted_target = target.dropna().sort_index()
        first_half_mean = sorted_target.iloc[:mid].mean()
        second_half_mean = sorted_target.iloc[mid:].mean()
        if first_half_mean > 0:
            shift_ratio = abs(second_half_mean - first_half_mean) / first_half_mean
            if shift_ratio > 0.5:
                report.add("warning", "structural_break", f"Possible level shift detected: first-half mean={first_half_mean:.1f}, second-half mean={second_half_mean:.1f}.")

    report.stats["n_valid_points"] = int(n_valid)
    report.stats["date_range"] = (str(sorted_dates.iloc[0].date()), str(sorted_dates.iloc[-1].date())) if len(sorted_dates) > 0 else ("N/A", "N/A")
    report.stats["target_mean"] = float(target.mean()) if n_valid > 0 else 0
    report.stats["target_std"] = float(target.std()) if n_valid > 0 else 0
    report.stats["target_min"] = float(target.min()) if n_valid > 0 else 0
    report.stats["target_max"] = float(target.max()) if n_valid > 0 else 0

    return report
