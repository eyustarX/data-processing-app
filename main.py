"""Main entry point for running the Data-Processing App from the command line."""

from pathlib import Path
from src.data_cleaning import load_dataset, clean_dataset
from src.data_processing import summarize_dataset


def main():
    base_dir = Path(__file__).resolve().parent
    raw_data_path = base_dir / "data" / "raw_sales_data.csv"

    print("=" * 60)
    print("      SMALL PYTHON DATA-PROCESSING APP (NumPy & Pandas)      ")
    print("=" * 60)

    # 1. Load Raw Dataset
    print(f"\n[1] Loading raw dataset from: {raw_data_path.name}")
    raw_df = load_dataset(raw_data_path)
    print(f"    Raw records loaded: {len(raw_df)} rows, {len(raw_df.columns)} columns")
    print("\n--- Raw Data Preview ---")
    print(raw_df.head())

    # 2. Clean Dataset
    print("\n[2] Cleaning dataset using modular cleaning functions...")
    cleaned_df = clean_dataset(raw_df)
    print(f"    Cleaned records remaining: {len(cleaned_df)} rows")
    print("\n--- Cleaned Data Preview ---")
    print(cleaned_df)

    # 3. Process with NumPy
    print("\n[3] Processing data with NumPy vectorized operations...")
    summary = summarize_dataset(cleaned_df)
    metrics = summary["metrics"]

    # Append computed metrics back to DataFrame for display
    cleaned_df["gross_revenue"] = metrics["gross_revenue"]
    cleaned_df["net_revenue"] = metrics["net_revenue"]
    cleaned_df["normalized_score"] = summary["normalized_net_revenue"]

    print("\n--- Enriched Dataset with NumPy Computed Fields ---")
    print(cleaned_df[["order_id", "product", "unit_price", "quantity", "discount_pct", "net_revenue", "normalized_score"]])

    # 4. NumPy Statistical Summary
    stats = summary["net_revenue_statistics"]
    print("\n[4] Net Revenue Statistics (computed with NumPy):")
    print(f"    - Total Revenue : ${stats['total']:,.2f}")
    print(f"    - Mean Revenue  : ${stats['mean']:,.2f}")
    print(f"    - Median Revenue: ${stats['median']:,.2f}")
    print(f"    - Std Deviation : ${stats['std_dev']:,.2f}")
    print(f"    - Min Revenue   : ${stats['min']:,.2f}")
    print(f"    - Max Revenue   : ${stats['max']:,.2f}")
    print(f"    - 25th / 75th % : ${stats['p25']:,.2f} / ${stats['p75']:,.2f}")

    print("\n" + "=" * 60)
    print("Data processing pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
