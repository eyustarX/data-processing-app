# 📊 Week 01: Small Python Data-Processing App

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eyustarX/data-processing-app/blob/main/week-01/task.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Task Objectives
Build a high-performance, modular data processing application that:
1. **Ingests Raw Datasets**: Loads structured records (`data/raw_sales_data.csv`) featuring common real-world data quality issues (duplicates, negative prices, null values, type mismatches). Includes an embedded fallback dataset so the pipeline can execute standalone in any environment.
2. **Modular Sanitization**: Applies functional, testable data sanitization steps (deduplication, numeric coercion, invalid record filtering, and intelligent null imputation).
3. **NumPy Vectorized Processing**: Replaces slow iteration loops with high-performance NumPy array operations to compute gross revenue, discount deductions, net revenue, min-max scaling ($[0, 1]$), and Z-score standardization ($\mu=0, \sigma=1$).
4. **Statistical Summaries**: Calculates key distribution metrics (sum, mean, median, standard deviation, min, max, 25th and 75th percentiles).
5. **Interactive Notebook with Visualizations**: Includes an executed Jupyter / Colab notebook with generated charts, cell outputs, and markdown narrative.
6. **Data Export**: Saves the cleaned and enriched dataset to `data/processed_sales_data.csv`.

---

## 📁 File Structure

```text
week-01/
├── README.md               # Week 01 task overview, instructions & Colab badge
├── task.py                 # Primary task entry point
├── data_processing.py      # Core data cleaning & NumPy processing module
├── task.ipynb              # Executed Google Colab notebook with charts & outputs
├── data_processing.ipynb   # Executed notebook
├── data/
│   ├── raw_sales_data.csv       # Raw dirty input data
│   └── processed_sales_data.csv # Exported clean & enriched dataset
└── tests/
    ├── __init__.py
    └── test_pipeline.py    # Automated unit tests (cleaning + NumPy operations)
```

---

## ✅ Submission Rules Compliance

| Rule | Requirement | Status |
| :--- | :--- | :---: |
| **Rule 1** | Submit `.py` source code and `.ipynb` notebook | ✅ Completed (`task.py`, `task.ipynb`) |
| **Rule 2** | Separate folder for each week (`week-01/`) | ✅ Completed (`week-01/`) |
| **Rule 3** | Clear file naming & runnable code | ✅ Tested and verified |
| **Rule 4** | Notebook includes code, outputs & brief explanations | ✅ Full narrative, plots & cell outputs |
| **Rule 5** | Pushed cleanly to GitHub with active history | ✅ Version controlled |

---

## 🚀 How to Run

### Option 1: Run Python Script via CLI
From repository root:
```bash
python week-01/task.py
```
Or from inside `week-01/`:
```bash
cd week-01
python task.py
# or:
python data_processing.py
```

### Option 2: Run Automated Unit Tests
```bash
python -m unittest discover -s week-01/tests -v
```

### Option 3: Run Locally in Jupyter Notebook
```bash
jupyter notebook week-01/task.ipynb
```

### Option 4: Run in Google Colab (Zero-Install)
Click the badge above or navigate to:
[Open in Google Colab](https://colab.research.google.com/github/eyustarX/data-processing-app/blob/main/week-01/task.ipynb)

---

## 🔬 Implementation Highlights

### 1. Data Cleaning Pipeline
- `remove_duplicates()`: Identifies and removes duplicate orders by `order_id`.
- `sanitize_numerical_columns()`: Safely converts string values to float and filters out non-positive quantities and prices.
- `handle_missing_values()`: Defaults empty discount values to `0.0`, missing cities to `'Unknown'`, and drops unusable records with missing identifiers.
- `clean_dataset_pipeline()`: Chains all steps into a single idempotent cleaning pipeline.

### 2. NumPy Vectorized Processing
- **Gross Revenue**:
  $$\text{Gross} = \text{unit\_price} \times \text{quantity}$$
- **Net Revenue**:
  $$\text{Net} = \text{Gross} \times (1 - \text{discount\_pct})$$
- **Min-Max Scaling**:
  $$\text{Normalized} = \frac{\text{Net} - \min(\text{Net})}{\max(\text{Net}) - \min(\text{Net})}$$
- **Z-Score Standardization**:
  $$z = \frac{\text{Net} - \mu}{\sigma}$$
- **Descriptive Statistics**:
  Calculated with `np.sum()`, `np.mean()`, `np.median()`, `np.std()`, and `np.percentile()`.
