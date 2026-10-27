"""Unit tests verifying Week 01 data cleaning and NumPy processing pipeline."""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure week-01 directory is on python path for direct imports
WEEK_01_DIR = Path(__file__).resolve().parent.parent
if str(WEEK_01_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK_01_DIR))

from data_processing import (
    calculate_statistics,
    calculate_z_scores,
    clean_dataset_pipeline,
    compute_financial_metrics,
    handle_missing_values,
    load_dataset,
    normalize_minmax,
    remove_duplicates,
    sanitize_numerical_columns,
)


class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        self.sample_raw = pd.DataFrame({
            "order_id": [1, 1, 2, 3],
            "product": ["Item A", "Item A", "Item B", "Item C"],
            "unit_price": ["100", "100", "-50", "200"],
            "quantity": [2, 2, "N/A", 3],
            "discount_pct": [0.1, 0.1, np.nan, np.nan],
            "city": ["NY", "NY", "LA", np.nan],
        })

    def test_remove_duplicates(self):
        deduped = remove_duplicates(self.sample_raw, subset=["order_id"])
        self.assertEqual(len(deduped), 3)

    def test_sanitize_numerical_columns(self):
        sanitized = sanitize_numerical_columns(self.sample_raw)
        # Should drop negative prices and invalid 'N/A' quantities
        self.assertTrue(all(sanitized["quantity"] > 0))
        self.assertTrue(all(sanitized["unit_price"] > 0))

    def test_handle_missing_values(self):
        sanitized = sanitize_numerical_columns(self.sample_raw)
        cleaned = handle_missing_values(sanitized)
        # Missing discount should default to 0.0
        self.assertFalse(cleaned["discount_pct"].isna().any())
        # Missing city should default to 'Unknown'
        self.assertIn("Unknown", cleaned["city"].values)

    def test_clean_dataset_pipeline(self):
        cleaned = clean_dataset_pipeline(self.sample_raw)
        self.assertEqual(len(cleaned), 2)  # only valid orders 1 and 3 survive


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.cleaned_df = pd.DataFrame({
            "order_id": [101, 102, 103],
            "unit_price": [100.0, 50.0, 20.0],
            "quantity": [2, 4, 5],
            "discount_pct": [0.1, 0.0, 0.2],
        })

    def test_compute_financial_metrics(self):
        metrics = compute_financial_metrics(self.cleaned_df)
        # gross: [200, 200, 100]
        np.testing.assert_array_equal(metrics["gross_revenue"], np.array([200.0, 200.0, 100.0]))
        # net: [180, 200, 80]
        np.testing.assert_array_equal(metrics["net_revenue"], np.array([180.0, 200.0, 80.0]))

    def test_calculate_statistics(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        stats = calculate_statistics(arr)
        self.assertEqual(stats["total"], 150.0)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["median"], 30.0)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 50.0)

    def test_normalize_minmax(self):
        arr = np.array([0.0, 50.0, 100.0])
        normalized = normalize_minmax(arr)
        np.testing.assert_array_almost_equal(normalized, np.array([0.0, 0.5, 1.0]))

    def test_calculate_z_scores(self):
        arr = np.array([10.0, 20.0, 30.0])
        z = calculate_z_scores(arr)
        self.assertAlmostEqual(float(np.mean(z)), 0.0, places=5)
        self.assertAlmostEqual(float(np.std(z)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
