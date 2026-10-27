"""Week 01: Small Python Data-Processing App.

This module demonstrates:
- Loading raw datasets (CSV with missing values, duplicates, and type mismatches)
- Modular data sanitization with pure Python functions
- Vectorized numerical processing and financial metrics with NumPy
- Descriptive statistical distributions and normalization
- Clean export of processed data
"""

from io import StringIO
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Fallback Dataset (ensures standalone execution anywhere)
# ---------------------------------------------------------
RAW_DATA_FALLBACK = """order_id,date,product,category,unit_price,quantity,discount_pct,city
1001,2024-01-05,Laptop,Electronics,1200,2,0.1,New York
1002,2024-01-06,Wireless Mouse,Electronics,25,10,0.05,Los Angeles
1003,2024-01-08,Desk Chair,Furniture,150,3,,Chicago
1004,2024-01-10,USB Cable,Electronics,12,-2,0,Houston
1005,2024-01-12,Mechanical Keyboard,Electronics,85,4,0.15,
1006,2024-01-15,Coffee Mug,Kitchen,15,5,0,New York
1007,2024-01-18,Gaming Monitor,Electronics,300,N/A,0.1,San Francisco
1008,2024-01-20,Office Desk,Furniture,,1,0.05,Austin
1009,2024-01-22,Desk Lamp,Furniture,45,3,0,Seattle
1010,2024-01-25,Water Bottle,Kitchen,20,8,0.2,Los Angeles
1001,2024-01-05,Laptop,Electronics,1200,2,0.1,New York
1011,2024-01-28,Headphones,Electronics,90,5,0.1,Chicago
1012,2024-01-30,Standing Desk,Furniture,400,2,,Houston"""


# ---------------------------------------------------------
# 1. Data Ingestion & Sanitization Functions
# ---------------------------------------------------------
def load_dataset(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load dataset from CSV file path, or fallback if file does not exist."""
    if file_path:
        p = Path(file_path)
        if p.exists():
            return pd.read_csv(p)
    return pd.read_csv(StringIO(RAW_DATA_FALLBACK))


def remove_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> pd.DataFrame:
    """Drop duplicate records from the DataFrame."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def sanitize_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numerical columns to float, handle errors, and filter out invalid rows."""
    cleaned = df.copy()
    for col in ["unit_price", "quantity", "discount_pct"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Filter out invalid records (e.g. negative or non-positive quantities and prices)
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] > 0)]
    return cleaned.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute sensible defaults for missing fields and drop unusable records."""
    cleaned = df.copy()
    if "discount_pct" in cleaned.columns:
        cleaned["discount_pct"] = cleaned["discount_pct"].fillna(0.0)
    if "city" in cleaned.columns:
        cleaned["city"] = cleaned["city"].fillna("Unknown")

    cleaned = cleaned.dropna(subset=["order_id", "product", "unit_price", "quantity"])
    cleaned["order_id"] = cleaned["order_id"].astype(int)
    cleaned["quantity"] = cleaned["quantity"].astype(int)
    return cleaned.reset_index(drop=True)


def clean_dataset_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute complete cleaning workflow in sequential order."""
    step1 = remove_duplicates(df, subset=["order_id"])
    step2 = sanitize_numerical_columns(step1)
    step3 = handle_missing_values(step2)
    return step3


# ---------------------------------------------------------
# 2. NumPy Processing Functions
# ---------------------------------------------------------
def compute_financial_metrics(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute financial metrics using vectorized NumPy array operations."""
    unit_prices: np.ndarray = df["unit_price"].to_numpy(dtype=np.float64)
    quantities: np.ndarray = df["quantity"].to_numpy(dtype=np.float64)
    discounts: np.ndarray = df["discount_pct"].to_numpy(dtype=np.float64)

    # Vectorized arithmetic
    gross_revenue: np.ndarray = unit_prices * quantities
    discount_amount: np.ndarray = gross_revenue * discounts
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


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Normalize array values between 0.0 and 1.0 using NumPy vectorization."""
    min_val, max_val = float(np.min(arr)), float(np.max(arr))
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def calculate_z_scores(arr: np.ndarray) -> np.ndarray:
    """Calculate Z-score standardization (mean=0, std=1) with NumPy."""
    mean, std = float(np.mean(arr)), float(np.std(arr))
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


# ---------------------------------------------------------
# 3. Main Pipeline Runner
# ---------------------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parent
    csv_file = base_dir / "data" / "raw_sales_data.csv"

    print("=" * 65)
    print("   AWS SBG TASKS - WEEK 01: Python Data-Processing App   ")
    print("=" * 65)

    # Step 1: Load raw data
    print(f"\n[1] Loading raw dataset from: {csv_file.name if csv_file.exists() else 'Embedded Fallback'}")
    raw_df = load_dataset(csv_file)
    print(f"    Raw records loaded: {len(raw_df)} rows, {len(raw_df.columns)} columns")
    print(raw_df.head())

    # Step 2: Clean data
    print(f"\n[2] Cleaning dataset with modular functions...")
    cleaned_df = clean_dataset_pipeline(raw_df)
    print(f"    Cleaned records remaining: {len(cleaned_df)} rows (duplicates and invalid records removed)")
    print(cleaned_df)

    # Step 3: NumPy Vectorized Processing
    print(f"\n[3] Processing numerical data with NumPy vectorization...")
    metrics = compute_financial_metrics(cleaned_df)
    cleaned_df["gross_revenue"] = metrics["gross_revenue"]
    cleaned_df["net_revenue"] = metrics["net_revenue"]
    cleaned_df["normalized_revenue"] = np.round(normalize_minmax(metrics["net_revenue"]), 3)
    cleaned_df["z_score"] = np.round(calculate_z_scores(metrics["net_revenue"]), 3)

    print("\n--- Enriched Dataset with NumPy Calculations ---")
    print(cleaned_df[["order_id", "product", "unit_price", "quantity", "discount_pct", "net_revenue", "normalized_revenue"]])

    # Step 4: Descriptive Statistics via NumPy
    stats = calculate_statistics(metrics["net_revenue"])
    print(f"\n[4] Net Revenue Statistics (computed via NumPy):")
    print(f"    - Total Revenue : ${stats['total']:,.2f}")
    print(f"    - Mean Revenue  : ${stats['mean']:,.2f}")
    print(f"    - Median Revenue: ${stats['median']:,.2f}")
    print(f"    - Std Deviation : ${stats['std_dev']:,.2f}")
    print(f"    - Min Revenue   : ${stats['min']:,.2f}")
    print(f"    - Max Revenue   : ${stats['max']:,.2f}")
    print(f"    - 25th / 75th % : ${stats['p25']:,.2f} / ${stats['p75']:,.2f}")

    # Step 5: Export clean data
    out_dir = base_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "processed_sales_data.csv"
    cleaned_df.to_csv(out_path, index=False)
    print(f"\n[5] Exported cleaned dataset to: {out_path.name}")

    print("\n" + "=" * 65)
    print("Week 01 Task pipeline executed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
