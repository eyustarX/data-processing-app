"""Data processing module powered by NumPy for vectorized operations and statistical analysis."""

from typing import Any
import numpy as np
import pandas as pd


def compute_financial_metrics(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute gross and net revenues using vectorized NumPy array operations."""
    unit_prices: np.ndarray = df["unit_price"].to_numpy(dtype=np.float64)
    quantities: np.ndarray = df["quantity"].to_numpy(dtype=np.float64)
    discounts: np.ndarray = df["discount_pct"].to_numpy(dtype=np.float64)

    # Vectorized gross revenue: Price * Quantity
    gross_revenue: np.ndarray = unit_prices * quantities

    # Vectorized discount amount: Gross Revenue * Discount Percentage
    discount_amount: np.ndarray = gross_revenue * discounts

    # Vectorized net revenue: Gross Revenue - Discount Amount
    net_revenue: np.ndarray = gross_revenue - discount_amount

    return {
        "unit_prices": unit_prices,
        "quantities": quantities,
        "discounts": discounts,
        "gross_revenue": gross_revenue,
        "discount_amount": discount_amount,
        "net_revenue": net_revenue,
    }


def calculate_statistics(arr: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics on a 1D NumPy array."""
    if arr.size == 0:
        return {}

    return {
        "count": float(arr.size),
        "total": float(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std_dev": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def normalize_array_minmax(arr: np.ndarray) -> np.ndarray:
    """Normalize array values between 0.0 and 1.0 using NumPy vectorization."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def calculate_z_scores(arr: np.ndarray) -> np.ndarray:
    """Calculate Z-score standardization (mean=0, std=1) with NumPy."""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Comprehensive processing pipeline aggregating NumPy calculations."""
    metrics = compute_financial_metrics(df)
    net_stats = calculate_statistics(metrics["net_revenue"])
    qty_stats = calculate_statistics(metrics["quantities"])
    normalized_rev = normalize_array_minmax(metrics["net_revenue"])

    return {
        "metrics": metrics,
        "net_revenue_statistics": net_stats,
        "quantity_statistics": qty_stats,
        "normalized_net_revenue": normalized_rev,
    }
