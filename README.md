# 📊 Small Python Data-Processing App

A clean, modular Python application that loads raw datasets, performs multi-stage sanitization using pure functions, and processes calculations with high-performance **NumPy** vectorization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/data-processing-app/blob/main/notebooks/data_analysis.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Do We Need a Google Colab Link?

**Short answer:** It is **not strictly mandatory** if you only run the project locally via Jupyter Notebook inside your `venv`, but it is **highly recommended**!

### Why include a Colab link?
1. **Zero-install execution:** Anyone reviewing your repository (evaluators, teammates, recruiters) can click the badge and run your entire pipeline in their browser without setting up Python, virtual environments, or dependencies.
2. **Best of both worlds:** You satisfy local development standards (`venv` + `requirements.txt`) **and** cloud notebook presentation.

### How the Colab URL Works:
Once pushed to GitHub, your notebook URL follows this exact pattern:
```text
https://colab.research.google.com/github/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>/blob/main/notebooks/data_analysis.ipynb
```
Replace `<YOUR_GITHUB_USERNAME>` and `<YOUR_REPOSITORY_NAME>` with your actual GitHub username and repository name.

---

## 📁 Project Architecture

```text
data_processing_app/
│
├── data/
│   └── raw_sales_data.csv       # Raw dataset containing dirty records, NaNs, duplicates
│
├── notebooks/
│   └── data_analysis.ipynb      # Interactive Jupyter / Colab notebook with charts & steps
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py         # Dedicated cleaning functions (dedup, null imputation, etc.)
│   └── data_processing.py       # NumPy vectorized calculations and statistical metrics
│
├── .gitignore                   # Ignore .venv, __pycache__, checkpoints
├── main.py                      # CLI runner script
├── requirements.txt             # Project dependencies (numpy, pandas, jupyter, matplotlib)
└── README.md                    # Project documentation & GitHub guide
```

---

## 🚀 Local Quickstart Guide

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/data-processing-app.git
cd data-processing-app
```

### 2. Create and activate a Virtual Environment (`venv`)

- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(If PowerShell scripts are restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application

- **Run via Terminal (CLI):**
  ```bash
  python main.py
  ```

- **Run via Jupyter Notebook:**
  ```bash
  jupyter notebook notebooks/data_analysis.ipynb
  ```

---

## 🛠️ Step-by-Step GitHub Setup Guide

Follow these steps to publish this project to GitHub from scratch:

### Step 1: Create a New GitHub Repository
1. Log in to [GitHub](https://github.com).
2. Click the **`+`** icon in the top right corner and select **New repository**.
3. Name your repository (e.g., `data-processing-app`).
4. Keep it **Public** (required for the Colab badge to open without login prompts).
5. **Do not** initialize with a README, .gitignore, or license (we already created them).
6. Click **Create repository**.

### Step 2: Initialize Git and Push Local Code
Open your terminal in the project directory (`data_processing_app`):

```bash
# Initialize local git repository
git init

# Stage all project files
git add .

# Create the initial commit
git commit -m "feat: initial commit with data cleaning, numpy processing and jupyter notebook"

# Rename default branch to main
git branch -M main

# Link to your remote GitHub repository (replace with your URL)
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/data-processing-app.git

# Push code to GitHub
git push -u origin main
```

### Step 3: Update the Colab Link in README and Notebook
1. Open `README.md` and `notebooks/data_analysis.ipynb`.
2. Replace `YOUR_GITHUB_USERNAME` and `data-processing-app` with your actual GitHub username and repository name.
3. Commit and push the update:
   ```bash
   git add README.md notebooks/data_analysis.ipynb
   git commit -m "docs: update Colab badge URL with actual repository path"
   git push
   ```
4. Click the badge on your GitHub repository page to test running in Google Colab!

---

## 🔬 What This Project Implements

1. **Raw Data Ingestion**: Handles realistic dirty data (negative quantities, missing prices/discounts, invalid string inputs).
2. **Modular Data Cleaning (`src/data_cleaning.py`)**:
   - `remove_duplicates()`: Removes duplicate transactions.
   - `sanitize_numerical_columns()`: Coerces invalid formats and discards invalid values.
   - `handle_missing_values()`: Imputes default discount and city values; removes incomplete rows.
3. **NumPy Vectorized Processing (`src/data_processing.py`)**:
   - High-speed array computations: Gross & Net Revenue calculation.
   - Normalization: Vectorized Min-Max scaling ($[0, 1]$) and Z-score standardization ($\mu=0, \sigma=1$).
   - Statistical Analysis: `np.mean()`, `np.median()`, `np.std()`, `np.percentile()`.
4. **Interactive Notebook & Visualization**:
   - Bar charts & scatter plots via `matplotlib`.
   - Fallback dataset mechanism allowing zero-config Colab runs.
