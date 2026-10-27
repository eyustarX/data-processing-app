# 🚀 AWS SBG Tasks

Weekly practical coding tasks, assignments, and data engineering projects for the **AWS Student Builder Group (SBG)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eyustarX/data-processing-app/blob/main/week-01/task.ipynb)
[![CI Pipeline](https://github.com/eyustarX/data-processing-app/actions/workflows/ci.yml/badge.svg)](https://github.com/eyustarX/data-processing-app/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"GitHub repo activity and the quality of projects matter when you look for a job, so make sure you are consistent for your own sake."*

---

## 📁 Repository Structure

```text
aws-sbg-tasks/
├── README.md                 # Main repository overview & weekly roadmap
├── requirements.txt          # Global dependencies for all weekly tasks
├── .gitignore                # Ignored virtual environments, checkpoints, and caches
├── LICENSE                   # MIT Open Source License
├── .github/
│   └── workflows/ci.yml      # Automated GitHub Actions test pipeline
│
├── week-01/                  # Week 01: Small Python Data-Processing App
│   ├── README.md             # Week 01 documentation & Colab badge
│   ├── task.py               # Primary task entry point
│   ├── data_processing.py    # Core data cleaning & NumPy processing module
│   ├── task.ipynb            # Executed Google Colab / Jupyter notebook with plots
│   ├── data_processing.ipynb # Executed notebook
│   ├── data/
│   │   ├── raw_sales_data.csv       # Raw dirty input dataset
│   │   └── processed_sales_data.csv # Output cleaned & enriched data
│   └── tests/
│       ├── __init__.py
│       └── test_pipeline.py  # Automated unit tests for week 01
│
├── week-02/                  # Week 02: (Upcoming)
│   ├── README.md
│   ├── task.py
│   └── task.ipynb
│
└── ...
```

---

## 📅 Weekly Task Index

| Week | Task Title | Source Code | Notebook / Colab | Status |
| :---: | :--- | :---: | :---: | :---: |
| **01** | **Small Python Data-Processing App** (Cleaning + NumPy) | [`task.py`](./week-01/task.py) | [Open in Colab](https://colab.research.google.com/github/eyustarX/data-processing-app/blob/main/week-01/task.ipynb) | ✅ Completed |
| **02** | *Week 02 Task* | `week-02/task.py` | `week-02/task.ipynb` | ⏳ Upcoming |
| **03** | *Week 03 Task* | `week-03/task.py` | `week-03/task.ipynb` | ⏳ Upcoming |
| **04** | *Week 04 Task* | `week-04/task.py` | `week-04/task.ipynb` | ⏳ Upcoming |

---

## 📌 Submission Standards & Best Practices

1. **Dual Submissions Every Week**: Each week includes both runnable Python source files (`task.py` / `<task_name>.py`) and an executed notebook (`task.ipynb` / `<task_name>.ipynb`) containing cell outputs and data visualizations.
2. **Dedicated Modular Directories**: Every task is housed in its own isolated directory (`week-01/`, `week-02/`, etc.) with its own documentation and sample data.
3. **Automated Testing & CI**: All logic is backed by automated unit tests running on Python 3.10, 3.11, and 3.12 via GitHub Actions.
4. **Cloud-Ready (Google Colab)**: Colab badges are placed in every weekly task README for instant, zero-install browser execution.
5. **Clean Version Control**: Meaningful commit messages and active, consistent commits showing incremental progress.

---

## 🚀 Local Quickstart Guide

### 1. Clone the repository
```bash
git clone https://github.com/eyustarX/data-processing-app.git
cd data-processing-app
```

### 2. Create and activate a Virtual Environment (`venv`)

- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(If script execution is restricted: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Weekly Tasks & Tests

- **Run Week 01 Python Application:**
  ```bash
  python week-01/task.py
  ```

- **Run Automated Unit Tests:**
  ```bash
  python -m unittest discover -s week-01/tests -v
  ```

- **Launch Interactive Jupyter Notebook:**
  ```bash
  jupyter notebook week-01/task.ipynb
  ```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
