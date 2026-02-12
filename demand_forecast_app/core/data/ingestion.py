"""Data ingestion: Excel and CSV file loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import pandas as pd


@dataclass
class IngestionResult:
    df: pd.DataFrame
    sheet_names: list[str] = field(default_factory=list)
    source_name: str = ""
    shape: tuple[int, int] = (0, 0)
    dtypes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.shape = self.df.shape
        self.dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}


def load_excel(
    file_buffer: BytesIO | Any,
    sheet_name: str | int = 0,
) -> IngestionResult:
    xls = pd.ExcelFile(file_buffer, engine="openpyxl")
    sheet_names = xls.sheet_names
    df = pd.read_excel(xls, sheet_name=sheet_name)
    source = getattr(file_buffer, "name", "uploaded_file.xlsx")
    return IngestionResult(df=df, sheet_names=sheet_names, source_name=source)


def load_csv(file_buffer: BytesIO | Any) -> IngestionResult:
    df = pd.read_csv(file_buffer)
    source = getattr(file_buffer, "name", "uploaded_file.csv")
    return IngestionResult(df=df, source_name=source)


def load_file(file_buffer: BytesIO | Any, filename: str, sheet_name: str | int = 0) -> IngestionResult:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("xlsx", "xls"):
        return load_excel(file_buffer, sheet_name=sheet_name)
    elif ext == "csv":
        return load_csv(file_buffer)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use .xlsx, .xls, or .csv.")
