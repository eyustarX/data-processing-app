# 🔍 Week 02: Mini Semantic Search Engine over AWS FAQs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eyustarX/data-processing-app/blob/main/week-02/task.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-red.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AI/ML AWS AASTU Cohort · Week 2 Hands-on Lab & Deliverable**

An end-to-end Mini Semantic Search pipeline that ingests a curated knowledge base of AWS Cloud and AI/ML FAQs, sanitizes and chunks documents into contextual passages, indexes them into a **ChromaDB** vector database, and executes semantic similarity searches returning the top 3 ranked matches.

---

## 🎯 Lab Objectives & Deliverables

| Requirement | Implementation Details | Status |
| :--- | :--- | :---: |
| **Load CSV/JSON of FAQs with Pandas** | Ingests `data/faqs.json` and `data/faqs.csv` using Pandas with fallback support | ✅ Completed |
| **Clean it and chunk the text** | Whitespace normalization, deduplication, and overlapping contextual passage chunking | ✅ Completed |
| **Embed and store it in Chroma** | Ingests embeddings and metadata into Chroma vector store collection | ✅ Completed |
| **Query it and print top 3 matches** | Ranked similarity retrieval with cosine distance, categories, questions, & answers | ✅ Completed |
| **Notebook (Colab or Jupyter)** | Fully executed `task.ipynb` with embedded outputs, visualizations, and Colab badge | ✅ Completed |
| **Clear README with example queries** | Comprehensive documentation with real search queries and top 3 results | ✅ Completed |
| **requirements.txt** | Isolated dependencies file in `week-02/` and root repository | ✅ Completed |
| **Commit history showing progress** | Granular git commits tracking development progress | ✅ Completed |

---

## 📁 File Structure

```text
week-02/
├── README.md                 # Lab documentation, query benchmarks & setup instructions
├── requirements.txt          # Dedicated dependencies (chromadb, pandas, numpy, etc.)
├── task.py                   # Standard weekly entry point runner
├── semantic_search.py        # Core modular pipeline (cleaning, chunking, Chroma ingestion, search)
├── task.ipynb                # Executed Google Colab / Jupyter notebook with outputs
├── semantic_search.ipynb     # Executed notebook
├── data/
│   ├── faqs.json             # Curated AWS & AI/ML FAQ dataset in JSON format
│   └── faqs.csv              # Curated AWS & AI/ML FAQ dataset in CSV format
└── tests/
    ├── __init__.py
    └── test_semantic_search.py # 5 automated unit tests for loading, chunking, & search
```

---

## 🏗️ Architecture & Pipeline Flow

```text
 ┌──────────────────────┐
 │ AWS & AI/ML FAQs     │  (faqs.json / faqs.csv)
 └──────────┬───────────┘
            │ 1. Load via Pandas
            ▼
 ┌──────────────────────┐
 │ Text Preprocessing   │  Clean whitespace, strip anomalies, extract Q&A pairs
 └──────────┬───────────┘
            │ 2. Contextual Chunking
            ▼
 ┌──────────────────────┐
 │ Document Chunking    │  Generate overlapping word chunks preserving context
 └──────────┬───────────┘
            │ 3. Vector Embedding
            ▼
 ┌──────────────────────┐
 │ ChromaDB Collection  │  Stores dense vector embeddings, metadata, and chunk IDs
 └──────────┬───────────┘
            │ 4. User Semantic Query
            ▼
 ┌──────────────────────┐
 │ Vector Search Engine │  Approximate Nearest Neighbor (ANN) search via Cosine Distance
 └──────────┬───────────┘
            │ 5. Ranked Output
            ▼
 ┌──────────────────────┐
 │ Top 3 Ranked Matches │  Rank, Category, Similarity Score %, Question, Answer
 └──────────────────────┘
```

---

## 🔎 Example Queries & Top 3 Matches

Below are actual execution benchmarks demonstrating the semantic retrieval capabilities:

### Query 1: *"How is pricing calculated for S3 bucket storage?"*

```text
[Rank #1] | Category: AWS Storage | Similarity: 48.59% (Distance: 0.5141)
  * Question: How does Amazon S3 storage classes and pricing work?
  * Answer  : Amazon Simple Storage Service (Amazon S3) is an object storage service offering 99.999999999% (11 9s) data durability. Pricing is based on storage volume (GB/month), requests (GET, PUT), and data transfer out. Classes include S3 Standard for frequent access, S3 Intelligent-Tiering for automatic cost optimization, and S3 Glacier for long-term archiving.

[Rank #2] | Category: AWS Storage | Similarity: 25.14% (Distance: 0.7486)
  * Question: How does Amazon S3 storage classes and pricing work?
  * Answer  : Amazon Simple Storage Service (Amazon S3) is an object storage service offering 99.999999999% (11 9s) data durability.

[Rank #3] | Category: AWS Serverless | Similarity: 8.46% (Distance: 0.9154)
  * Question: What is AWS Lambda and how does serverless execution work?
  * Answer  : AWS Lambda is a serverless, event-driven compute service that lets you run code without provisioning or managing servers. You pay only for the compute time you consume...
```

---

### Query 2: *"What tools exist for deploying machine learning models on AWS?"*

```text
[Rank #1] | Category: AWS AI/ML | Similarity: 25.68% (Distance: 0.7432)
  * Question: What is Amazon SageMaker and what are its key capabilities?
  * Answer  : Amazon SageMaker is a fully managed machine learning platform that enables data scientists and developers to build, train, tune, and deploy machine learning and deep learning models at scale. It includes SageMaker Studio, automated model tuning, distributed training, and managed real-time or serverless endpoints.

[Rank #2] | Category: AWS Storage | Similarity: 11.42% (Distance: 0.8858)
  * Question: How does Amazon S3 storage classes and pricing work?
  * Answer  : Amazon Simple Storage Service (Amazon S3) is an object storage service...

[Rank #3] | Category: AWS AI/ML | Similarity: 9.09% (Distance: 0.9091)
  * Question: What is Amazon Kendra and how does it implement enterprise search?
  * Answer  : Amazon Kendra is an intelligent enterprise search service powered by machine learning...
```

---

### Query 3: *"How do I apply the principle of least privilege in cloud security?"*

```text
[Rank #1] | Category: AWS Security | Similarity: 37.49% (Distance: 0.6251)
  * Question: How do I secure my AWS infrastructure using IAM policies and least privilege?
  * Answer  : AWS Identity and Access Management (IAM) controls authentication and authorization. Follow the principle of least privilege by granting only the minimum permissions required for specific roles or users. Use IAM roles for EC2 instances and Lambda functions instead of embedding permanent access keys.

[Rank #2] | Category: AWS Networking | Similarity: 11.13% (Distance: 0.8887)
  * Question: What is an Amazon VPC and how do subnets provide network isolation?
  * Answer  : Amazon Virtual Private Cloud (Amazon VPC) lets you provision a logically isolated section...

[Rank #3] | Category: AWS Compute | Similarity: 8.40% (Distance: 0.9160)
  * Question: What is Amazon EC2 and when should I use it?
  * Answer  : Amazon Elastic Compute Cloud (Amazon EC2) provides scalable virtual servers...
```

---

## 🚀 How to Run

### 1. Run Python CLI Application
```bash
# From repository root
python week-02/task.py

# Or inside week-02
cd week-02
python semantic_search.py
```

### 2. Run Automated Unit Tests
```bash
python -m unittest discover -s week-02/tests -v
```

### 3. Launch Locally in Jupyter Notebook
```bash
jupyter notebook week-02/task.ipynb
```

### 4. Run in Google Colab (Zero-Install)
Click the badge above or navigate to:
[Open in Google Colab](https://colab.research.google.com/github/eyustarX/data-processing-app/blob/main/week-02/task.ipynb)
