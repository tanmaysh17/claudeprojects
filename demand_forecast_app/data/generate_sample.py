"""Generate sample demand data for testing the application.

Run: python data/generate_sample.py
Creates: data/sample_demand.xlsx
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_demand(
    start_date: str = "2019-01-07",
    n_weeks: int = 260,  # 5 years
    base_demand: float = 1000,
    trend_slope: float = 2.0,
    seasonal_amplitude: float = 200,
    noise_std: float = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic weekly demand data with trend, seasonality, and noise."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    # Trend
    trend = base_demand + trend_slope * t

    # Annual seasonality (52-week cycle)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 52)

    # Holiday effects
    holiday_boost = np.zeros(n_weeks)
    for i, d in enumerate(dates):
        # Christmas/holiday season (weeks 48-52)
        week = d.isocalendar()[1]
        if week >= 48 or week <= 1:
            holiday_boost[i] = 150
        # Back-to-school (weeks 32-36)
        elif 32 <= week <= 36:
            holiday_boost[i] = 80
        # Summer dip (weeks 24-30)
        elif 24 <= week <= 30:
            holiday_boost[i] = -60

    # Noise
    noise = rng.normal(0, noise_std, n_weeks)

    # Combined demand
    demand = trend + seasonal + holiday_boost + noise
    demand = np.maximum(demand, 10)  # Floor at 10

    # Price (inversely correlated with promotions)
    base_price = 19.99
    price = base_price + rng.normal(0, 1, n_weeks)
    promotion = (rng.random(n_weeks) < 0.15).astype(int)
    price[promotion == 1] *= 0.85  # 15% discount during promotions

    # Add a few outliers
    outlier_idx = rng.choice(n_weeks, size=5, replace=False)
    demand[outlier_idx] *= rng.uniform(1.5, 2.5, size=5)

    df = pd.DataFrame({
        "date": dates,
        "sales": np.round(demand, 0).astype(int),
        "price": np.round(price, 2),
        "promotion": promotion,
        "region": rng.choice(["North", "South", "East", "West"], n_weeks),
    })

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "sample_demand.xlsx"
    df = generate_sample_demand()
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"Sample data generated: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Mean sales: {df['sales'].mean():.0f}")
    print(df.head())
