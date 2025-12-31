"""
RAG (Retrieval-Augmented Generation) Module.

Provides RAG capabilities using OCI AI Studio (Generative AI service)
for embeddings and retrieval-augmented context enrichment.

Features:
- OCI GenAI embeddings (Cohere embed-v3)
- Redis-based vector store for development
- Document chunking and indexing
- Semantic retrieval for agent context
- Multiple document loaders (Markdown, JSON, YAML, directories)

Usage:
    from src.rag import RAGRetriever, get_retriever

    # Get a retriever for a namespace
    retriever = await get_retriever("oci-docs")

    # Ingest documents
    await retriever.ingest_file("docs/ARCHITECTURE.md")
    await retriever.ingest_text("Some documentation text", source="manual")

    # Retrieve relevant context
    result = await retriever.retrieve("How do I create a VCN?")
    print(result.context)  # Combined context from retrieved docs

Environment Variables:
    OCI_COMPARTMENT_ID: OCI compartment for GenAI embeddings
    OCI_GENAI_ENDPOINT: GenAI service endpoint (default: us-chicago-1)
    REDIS_URL: Redis connection for vector store
"""

from src.rag.embeddings import MockEmbeddings, OCIEmbeddings, get_embeddings
from src.rag.loaders import (
    DirectoryLoader,
    LoaderConfig,
    MarkdownLoader,
    OCIDocumentationLoader,
    StructuredDataLoader,
)
from src.rag.retriever import (
    ChunkConfig,
    RAGRetriever,
    RetrievalResult,
    close_all_retrievers,
    get_retriever,
)
from src.rag.vector_store import Document, RedisVectorStore, SearchResult

__all__ = [
    # Core classes
    "OCIEmbeddings",
    "MockEmbeddings",
    "RedisVectorStore",
    "RAGRetriever",
    # Data types
    "Document",
    "SearchResult",
    "RetrievalResult",
    "ChunkConfig",
    "LoaderConfig",
    # Loaders
    "DirectoryLoader",
    "MarkdownLoader",
    "OCIDocumentationLoader",
    "StructuredDataLoader",
    # Factory functions
    "get_embeddings",
    "get_retriever",
    "close_all_retrievers",
]
