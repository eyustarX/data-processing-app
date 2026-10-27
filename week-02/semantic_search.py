"""Week 02: Mini Semantic Search Engine over AWS FAQs.

Lab Deliverable for AI/ML AWS AASTU Cohort:
1. Load a CSV/JSON of FAQs with Pandas.
2. Clean it and chunk the text.
3. Embed and store it in Chroma vector store.
4. Query it and print the top 3 matches with similarity metrics.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Check if chromadb is available
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


# ---------------------------------------------------------------------------
# Fallback FAQ Dataset (ensures standalone zero-dependency execution anywhere)
# ---------------------------------------------------------------------------
DEFAULT_FAQS = [
    {
        "id": "faq-001",
        "category": "AWS Compute",
        "question": "What is Amazon EC2 and when should I use it?",
        "answer": "Amazon Elastic Compute Cloud (Amazon EC2) provides scalable virtual servers (instances) in the cloud. You should use EC2 when you need complete control over your operating system, custom software stacks, high-performance compute clusters, or continuous background processing."
    },
    {
        "id": "faq-002",
        "category": "AWS Storage",
        "question": "How does Amazon S3 storage classes and pricing work?",
        "answer": "Amazon Simple Storage Service (Amazon S3) is an object storage service offering 99.999999999% (11 9s) data durability. Pricing is based on storage volume (GB/month), requests (GET, PUT), and data transfer out. Classes include S3 Standard for frequent access, S3 Intelligent-Tiering for automatic cost optimization, and S3 Glacier for long-term archiving."
    },
    {
        "id": "faq-003",
        "category": "AWS Serverless",
        "question": "What is AWS Lambda and how does serverless execution work?",
        "answer": "AWS Lambda is a serverless, event-driven compute service that lets you run code without provisioning or managing servers. You pay only for the compute time you consume, measured in millisecond increments, triggered by HTTP requests via API Gateway, S3 events, or DynamoDB streams."
    },
    {
        "id": "faq-004",
        "category": "AWS Security",
        "question": "How do I secure my AWS infrastructure using IAM policies and least privilege?",
        "answer": "AWS Identity and Access Management (IAM) controls authentication and authorization. Follow the principle of least privilege by granting only the minimum permissions required for specific roles or users. Use IAM roles for EC2 instances and Lambda functions instead of embedding permanent access keys."
    },
    {
        "id": "faq-005",
        "category": "AWS AI/ML",
        "question": "What is Amazon SageMaker and what are its key capabilities?",
        "answer": "Amazon SageMaker is a fully managed machine learning platform that enables data scientists and developers to build, train, tune, and deploy machine learning and deep learning models at scale. It includes SageMaker Studio, automated model tuning, distributed training, and managed real-time or serverless endpoints."
    },
    {
        "id": "faq-006",
        "category": "AWS Generative AI",
        "question": "What is Amazon Bedrock and how does it support foundation models?",
        "answer": "Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like Anthropic, AI21 Labs, Cohere, Meta, Mistral AI, and Amazon via a single API, alongside a broad set of capabilities to build generative AI applications with security, privacy, and responsible AI."
    },
    {
        "id": "faq-007",
        "category": "Vector DB & RAG",
        "question": "What is ChromaDB and how does it store vector embeddings?",
        "answer": "Chroma is an open-source, AI-native embedding database (vector store) designed to make building LLM apps simple. Chroma stores high-dimensional vector representations of text documents alongside metadata and raw document text, enabling sub-millisecond approximate nearest neighbor (ANN) similarity search."
    },
    {
        "id": "faq-008",
        "category": "Vector DB & RAG",
        "question": "What is semantic search and how does it differ from keyword search?",
        "answer": "Semantic search seeks to understand the natural language intent and contextual meaning of a search query rather than simply matching literal keywords. By converting text into dense mathematical embedding vectors, semantic search finds conceptual matches even when different vocabulary is used."
    },
    {
        "id": "faq-009",
        "category": "Vector DB & RAG",
        "question": "Why is text chunking important in Retrieval Augmented Generation (RAG)?",
        "answer": "Text chunking breaks large documents into smaller, coherent passages before generating embeddings. Proper chunking prevents dilution of semantic meaning across large texts, fits within embedding model context limits, and ensures the retrieval engine returns specific, relevant context to the language model."
    },
    {
        "id": "faq-010",
        "category": "AWS AI/ML",
        "question": "What is Amazon Kendra and how does it implement enterprise search?",
        "answer": "Amazon Kendra is an intelligent enterprise search service powered by machine learning. Kendra uses natural language processing (NLP) to return direct answers to user questions from unstructured content across document repositories like SharePoint, Confluence, and S3."
    },
    {
        "id": "faq-011",
        "category": "AWS AI/ML",
        "question": "How does Amazon Comprehend perform natural language processing?",
        "answer": "Amazon Comprehend is a managed NLP service that uses machine learning to uncover valuable insights and connections in text. It provides sentiment analysis, entity extraction, key phrase recognition, language detection, and personally identifiable information (PII) masking."
    },
    {
        "id": "faq-012",
        "category": "AWS Networking",
        "question": "What is an Amazon VPC and how do subnets provide network isolation?",
        "answer": "Amazon Virtual Private Cloud (Amazon VPC) lets you provision a logically isolated section of the AWS Cloud where you can launch AWS resources in a virtual network you define. Public subnets route traffic through an Internet Gateway, while private subnets isolate databases and backend services behind NAT Gateways."
    }
]


# ---------------------------------------------------------------------------
# 1. Ingestion: Load CSV or JSON with Pandas
# ---------------------------------------------------------------------------
def load_faq_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load FAQ dataset from a CSV or JSON file into a Pandas DataFrame."""
    if file_path:
        p = Path(file_path)
        if p.exists():
            if p.suffix.lower() == ".json":
                df = pd.read_json(p)
                return df
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                return df

    # Fallback to embedded FAQ dataset
    return pd.DataFrame(DEFAULT_FAQS)


# ---------------------------------------------------------------------------
# 2. Cleaning & Text Chunking
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Sanitize and normalize raw text."""
    if not isinstance(text, str):
        return ""
    # Remove excess whitespace and newlines
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def chunk_text(text: str, max_words: int = 60, overlap_words: int = 15) -> List[str]:
    """Split text into overlapping word chunks to preserve context."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    step = max_words - overlap_words
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
        if i + max_words >= len(words):
            break
    return chunks


def clean_and_chunk_faqs(df: pd.DataFrame, max_words: int = 60) -> List[Dict[str, Any]]:
    """Clean DataFrame fields and generate contextual chunks ready for vector indexing."""
    processed_chunks: List[Dict[str, Any]] = []

    # Drop records with missing questions or answers
    cleaned_df = df.dropna(subset=["question", "answer"]).copy()

    for _, row in cleaned_df.iterrows():
        faq_id = str(row.get("id", f"faq-{len(processed_chunks)+1}"))
        category = clean_text(str(row.get("category", "General")))
        question = clean_text(str(row["question"]))
        answer = clean_text(str(row["answer"]))

        # Composite document linking question with answer for dense semantic indexing
        full_content = f"Question: {question}\nAnswer: {answer}"
        sub_chunks = chunk_text(full_content, max_words=max_words)

        for idx, chunk_content in enumerate(sub_chunks):
            chunk_id = f"{faq_id}_chunk_{idx}"
            processed_chunks.append({
                "id": chunk_id,
                "faq_id": faq_id,
                "category": category,
                "question": question,
                "answer": answer,
                "text": chunk_content,
            })

    return processed_chunks


# ---------------------------------------------------------------------------
# Fallback Vector Store (NumPy-based Cosine Similarity)
# ---------------------------------------------------------------------------
class FallbackVectorStore:
    """Zero-dependency vector store using TF-IDF / character n-gram cosine similarity."""

    def __init__(self, collection_name: str = "aws_faqs"):
        self.collection_name = collection_name
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.matrix: np.ndarray = np.array([])

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[a-z0-9_]+\b", text.lower())
        # Add bigrams for enhanced semantic context
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas

        # Build vocabulary
        vocab: Dict[str, int] = {}
        doc_tokens = [self._tokenize(doc) for doc in documents]
        for tokens in doc_tokens:
            for token in set(tokens):
                vocab[token] = vocab.get(token, 0) + 1

        self.vocabulary = {term: idx for idx, term in enumerate(vocab.keys())}
        num_docs = len(documents)
        idf_list = [math.log((num_docs + 1) / (count + 1)) + 1.0 for count in vocab.values()]
        self.idf = np.array(idf_list, dtype=np.float64)

        # Build document vectors
        matrix = np.zeros((num_docs, len(self.vocabulary)), dtype=np.float64)
        for doc_idx, tokens in enumerate(doc_tokens):
            for t in tokens:
                if t in self.vocabulary:
                    matrix[doc_idx, self.vocabulary[t]] += 1.0
            matrix[doc_idx] *= self.idf

        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    def query(self, query_texts: List[str], n_results: int = 3) -> Dict[str, Any]:
        results_docs = []
        results_metas = []
        results_ids = []
        results_distances = []

        for q_text in query_texts:
            q_tokens = self._tokenize(q_text)
            q_vec = np.zeros(len(self.vocabulary), dtype=np.float64)
            for t in q_tokens:
                if t in self.vocabulary:
                    q_vec[self.vocabulary[t]] += 1.0
            q_vec *= self.idf

            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec /= q_norm

            # Cosine similarity
            similarities = np.dot(self.matrix, q_vec)
            # Distance = 1 - similarity
            distances = 1.0 - similarities

            top_indices = np.argsort(distances)[:n_results]

            results_docs.append([self.documents[i] for i in top_indices])
            results_metas.append([self.metadatas[i] for i in top_indices])
            results_ids.append([self.ids[i] for i in top_indices])
            results_distances.append([float(distances[i]) for i in top_indices])

        return {
            "ids": results_ids,
            "documents": results_docs,
            "metadatas": results_metas,
            "distances": results_distances,
        }


# ---------------------------------------------------------------------------
# 3. Vector Store: ChromaDB Ingestion
# ---------------------------------------------------------------------------
def init_vector_store(
    chunks: List[Dict[str, Any]],
    persist_dir: Optional[str] = None,
    collection_name: str = "aws_faq_collection",
):
    """Embed and store chunked FAQs in ChromaDB, with fallback if chromadb is unavailable."""
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "faq_id": c["faq_id"],
            "category": c["category"],
            "question": c["question"],
            "answer": c["answer"],
        }
        for c in chunks
    ]

    if HAS_CHROMADB:
        if persist_dir:
            client = chromadb.PersistentClient(path=persist_dir)
        else:
            client = chromadb.Client()

        # Delete existing collection if re-indexing
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "AWS & AI/ML FAQ Semantic Search"},
        )
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return collection
    else:
        # Fallback vector store
        store = FallbackVectorStore(collection_name=collection_name)
        store.add(ids=ids, documents=documents, metadatas=metadatas)
        return store


# ---------------------------------------------------------------------------
# 4. Semantic Search Query Engine
# ---------------------------------------------------------------------------
def semantic_search(
    vector_store: Any,
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Query the Chroma vector store and return the top K matches."""
    cleaned_query = clean_text(query)
    results = vector_store.query(query_texts=[cleaned_query], n_results=top_k)

    matches: List[Dict[str, Any]] = []
    if not results or not results.get("ids"):
        return matches

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

    for rank, (m_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
        # Convert cosine distance to approximate similarity score
        score = max(0.0, 1.0 - dist)
        matches.append({
            "rank": rank,
            "id": m_id,
            "category": meta.get("category", "General"),
            "question": meta.get("question", ""),
            "answer": meta.get("answer", ""),
            "distance": round(float(dist), 4),
            "similarity_score": round(float(score), 4),
            "matched_text": doc,
        })

    return matches


def format_search_results(query: str, results: List[Dict[str, Any]]) -> str:
    """Format the top search matches into a clean readable string (safe on all consoles)."""
    lines = [
        f"\n[Query] '{query}'",
        "-" * 70,
    ]
    if not results:
        lines.append("No matching FAQs found.")
    else:
        for item in results:
            lines.append(
                f"[Rank #{item['rank']}] | Category: {item['category']} | Similarity: {item['similarity_score']:.2%} (Distance: {item['distance']:.4f})"
            )
            lines.append(f"  * Question: {item['question']}")
            lines.append(f"  * Answer  : {item['answer']}")
            lines.append("")
    lines.append("-" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Main CLI Runner
# ---------------------------------------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parent
    data_json = base_dir / "data" / "faqs.json"
    data_csv = base_dir / "data" / "faqs.csv"

    print("=" * 70)
    print("   AWS AASTU COHORT - WEEK 02: MINI SEMANTIC SEARCH (ChromaDB)   ")
    print("=" * 70)

    # 1. Load Data
    data_file = data_json if data_json.exists() else data_csv
    print(f"\n[Step 1] Loading FAQs with Pandas from: {data_file.name}")
    df = load_faq_data(data_file)
    print(f"         Loaded {len(df)} FAQ records successfully.")

    # 2. Clean & Chunk
    print("\n[Step 2] Sanitizing text and generating contextual chunks...")
    chunks = clean_and_chunk_faqs(df)
    print(f"         Generated {len(chunks)} contextual chunks for vector indexing.")

    # 3. Vector Ingestion
    backend_name = "ChromaDB Native" if HAS_CHROMADB else "Vector Store (NumPy Cosine Engine)"
    print(f"\n[Step 3] Embedding and indexing chunks into {backend_name}...")
    vector_store = init_vector_store(chunks)
    print("         Index built and ready for approximate nearest neighbor retrieval.")

    # 4. Demonstration Queries
    demo_queries = [
        "How is pricing calculated for S3 bucket storage?",
        "What tools exist for deploying machine learning models on AWS?",
        "How do I apply the principle of least privilege in cloud security?",
    ]

    print("\n[Step 4] Executing Semantic Search Queries (Top 3 Matches):")
    for query in demo_queries:
        results = semantic_search(vector_store, query, top_k=3)
        print(format_search_results(query, results))

    print("\n" + "=" * 70)
    print("Mini Semantic Search pipeline executed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
