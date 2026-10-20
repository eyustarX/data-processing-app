"""Data cleaning module with reusable functions for dataset sanitization."""

from typing import Union
import pandas as pd
from pathlib import Path


def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw dataset from a CSV file path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")
    return pd.read_csv(path)


def remove_duplicates(df: pd.DataFrame, subset: list[str] = None) -> pd.DataFrame:
    """Remove duplicate records from the DataFrame."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def sanitize_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numerical fields, coerce invalid strings to NaN, and filter invalid records."""
    cleaned = df.copy()

    # Coerce numeric values (e.g. strings like 'N/A' become NaN)
    cleaned["unit_price"] = pd.to_numeric(cleaned["unit_price"], errors="coerce")
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce")
    cleaned["discount_pct"] = pd.to_numeric(cleaned["discount_pct"], errors="coerce")

    # Filter out impossible or negative values (e.g. negative quantities)
    cleaned = cleaned[cleaned["quantity"] > 0]
    cleaned = cleaned[cleaned["unit_price"] > 0]

    return cleaned.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing or null values across columns."""
    cleaned = df.copy()

    # Default missing discount to 0.0
    cleaned["discount_pct"] = cleaned["discount_pct"].fillna(0.0)

    # Impute missing city with 'Unknown'
    cleaned["city"] = cleaned["city"].fillna("Unknown")

    # Drop any remaining records missing key identifiers
    cleaned = cleaned.dropna(subset=["order_id", "product", "unit_price", "quantity"])

    # Cast integer columns appropriately
    cleaned["order_id"] = cleaned["order_id"].astype(int)
    cleaned["quantity"] = cleaned["quantity"].astype(int)

    return cleaned.reset_index(drop=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Master pipeline function applying all cleaning steps sequentially."""
    step1 = remove_duplicates(df, subset=["order_id"])
    step2 = sanitize_numerical_columns(step1)
    step3 = handle_missing_values(step2)
    return step3
