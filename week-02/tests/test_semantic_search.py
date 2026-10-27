"""Unit tests verifying Week 02 Mini Semantic Search pipeline."""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Ensure week-02 directory is on python path for imports
WEEK_02_DIR = Path(__file__).resolve().parent.parent
if str(WEEK_02_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK_02_DIR))

from semantic_search import (
    clean_and_chunk_faqs,
    clean_text,
    chunk_text,
    init_vector_store,
    load_faq_data,
    semantic_search,
)


class TestSemanticSearch(unittest.TestCase):
    def setUp(self):
        self.data_dir = WEEK_02_DIR / "data"
        self.sample_df = pd.DataFrame([
            {
                "id": "faq-test-1",
                "category": "AWS Storage",
                "question": "What is Amazon S3?",
                "answer": "Amazon S3 is scalable object storage designed for 99.999999999% durability.",
            },
            {
                "id": "faq-test-2",
                "category": "AWS Compute",
                "question": "What is Amazon EC2?",
                "answer": "Amazon EC2 provides resizable virtual compute capacity in the cloud.",
            },
            {
                "id": "faq-test-3",
                "category": "AWS AI/ML",
                "question": "What is Amazon SageMaker?",
                "answer": "Amazon SageMaker is a managed service for machine learning model training and inference.",
            },
        ])

    def test_load_faq_data_json(self):
        json_file = self.data_dir / "faqs.json"
        df = load_faq_data(json_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 10)
        self.assertIn("question", df.columns)
        self.assertIn("answer", df.columns)

    def test_load_faq_data_csv(self):
        csv_file = self.data_dir / "faqs.csv"
        df = load_faq_data(csv_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 10)
        self.assertIn("question", df.columns)

    def test_clean_and_chunk_faqs(self):
        chunks = clean_and_chunk_faqs(self.sample_df, max_words=30)
        self.assertGreaterEqual(len(chunks), 3)
        first = chunks[0]
        self.assertIn("id", first)
        self.assertIn("question", first)
        self.assertIn("answer", first)
        self.assertIn("text", first)
        self.assertTrue(first["text"].startswith("Question: What is Amazon S3?"))

    def test_vector_store_indexing_and_query(self):
        chunks = clean_and_chunk_faqs(self.sample_df)
        store = init_vector_store(chunks)

        # Query for storage
        results = semantic_search(store, "How to store objects durably?", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[1]["rank"], 2)
        self.assertEqual(results[2]["rank"], 3)
        # Top result should match Amazon S3
        self.assertIn("S3", results[0]["question"])

    def test_empty_query_handling(self):
        chunks = clean_and_chunk_faqs(self.sample_df)
        store = init_vector_store(chunks)
        results = semantic_search(store, "", top_k=3)
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
