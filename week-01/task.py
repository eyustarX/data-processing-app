"""Week 01 Task Runner: Small Python Data-Processing App.

Entry point conforming to standard weekly assignment submission guidelines:
Runs the complete data loading, cleaning, NumPy processing, and export pipeline.
"""

from data_processing import (
    calculate_statistics,
    calculate_z_scores,
    clean_dataset_pipeline,
    compute_financial_metrics,
    handle_missing_values,
    load_dataset,
    main,
    normalize_minmax,
    remove_duplicates,
    sanitize_numerical_columns,
)

if __name__ == "__main__":
    main()
