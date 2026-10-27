"""Week 02 Task Runner: Mini Semantic Search over AWS FAQs.

Entry point conforming to standard weekly assignment submission guidelines:
Executes the FAQ data ingestion, text cleaning & chunking, Chroma vector storage,
and top-3 semantic search retrieval pipeline.
"""

from semantic_search import (
    clean_and_chunk_faqs,
    format_search_results,
    init_vector_store,
    load_faq_data,
    main,
    semantic_search,
)

if __name__ == "__main__":
    main()
