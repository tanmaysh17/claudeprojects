# Demand Forecasting Studio

A production-oriented Streamlit web application for weekly demand forecasting with:

- Excel upload and column mapping
- data quality validation (missing weeks, duplicates, non-numeric values, outliers, sparse history, structural shifts)
- exploratory diagnostics and seasonality analysis
- multiple forecast model families (baseline statistical + SARIMAX + ML lag model + ensemble)
- rolling-holdout backtesting (13, 26, 52 weeks)
- prediction intervals (80% and 95%)
- model ranking and plain-language explainability
- business growth metrics and seasonality indices
- export to Excel and CSV with one-page summary

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit (typically `http://localhost:8501`).
