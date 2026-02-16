"""Multi-sheet Excel workbook export."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def create_forecast_workbook(
    forecast_df: pd.DataFrame,
    comparison_df: pd.DataFrame | None = None,
    summary_text: str = "",
    data_quality_df: pd.DataFrame | None = None,
) -> BytesIO:
    """Create a multi-sheet Excel workbook with all forecast results."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Header format
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#1f77b4",
            "font_color": "#ffffff",
            "border": 1,
        })
        number_fmt = workbook.add_format({"num_format": "#,##0.0"})
        pct_fmt = workbook.add_format({"num_format": "0.0%"})

        # Sheet 1: Forecasts
        forecast_df.to_excel(writer, sheet_name="Forecasts", index=False)
        ws = writer.sheets["Forecasts"]
        for i, col in enumerate(forecast_df.columns):
            ws.write(0, i, col, header_fmt)
            ws.set_column(i, i, max(15, len(col) + 5))

        # Sheet 2: Model Comparison
        if comparison_df is not None and len(comparison_df) > 0:
            comparison_df.to_excel(writer, sheet_name="Model Comparison", index=False)
            ws = writer.sheets["Model Comparison"]
            for i, col in enumerate(comparison_df.columns):
                ws.write(0, i, col, header_fmt)
                ws.set_column(i, i, max(15, len(col) + 5))

            # Conditional formatting for metric columns
            n_rows = len(comparison_df)
            for i, col in enumerate(comparison_df.columns):
                if col not in ("Model", "Rank"):
                    ws.conditional_format(
                        1, i, n_rows, i,
                        {
                            "type": "3_color_scale",
                            "min_color": "#63BE7B",
                            "mid_color": "#FFEB84",
                            "max_color": "#F8696B",
                        },
                    )

        # Sheet 3: Summary
        if summary_text:
            ws = workbook.add_worksheet("Summary")
            writer.sheets["Summary"] = ws
            title_fmt = workbook.add_format({"bold": True, "font_size": 14})
            text_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

            ws.set_column(0, 0, 100)
            ws.write(0, 0, "Forecast Summary Report", title_fmt)
            for i, paragraph in enumerate(summary_text.split("\n\n")):
                ws.write(i + 2, 0, paragraph, text_fmt)

        # Sheet 4: Data Quality
        if data_quality_df is not None and len(data_quality_df) > 0:
            data_quality_df.to_excel(writer, sheet_name="Data Quality", index=False)
            ws = writer.sheets["Data Quality"]
            for i, col in enumerate(data_quality_df.columns):
                ws.write(0, i, col, header_fmt)
                ws.set_column(i, i, max(15, len(col) + 5))

    output.seek(0)
    return output
