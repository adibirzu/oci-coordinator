"""
RAG Retriever.

High-level retrieval interface for agents to use RAG capabilities.
Handles document ingestion, chunking, and semantic retrieval.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from src.rag.embeddings import get_embeddings
from src.rag.vector_store import Document, RedisVectorStore, SearchResult

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = structlog.get_logger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""

    query: str
    documents: list[SearchResult]
    context: str  # Combined context from retrieved documents
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_results(self) -> bool:
        return len(self.documents) > 0


class RAGRetriever:
    """
    RAG Retriever for agent context enrichment.

    Provides a high-level interface for:
    - Ingesting documents with automatic chunking
    - Semantic retrieval based on queries
    - Context formatting for LLM prompts

    Example:
        retriever = RAGRetriever(namespace="oci-docs")
        await retriever.ingest_text(doc_content, metadata={"source": "oci"})
        result = await retriever.retrieve("How do I create a VCN?")
        # Use result.context in agent prompt
    """

    def __init__(
        self,
        namespace: str = "default",
        embeddings: Embeddings | None = None,
        redis_url: str | None = None,
        chunk_config: ChunkConfig | None = None,
        default_k: int = 5,
        min_score: float = 0.3,
    ):
        """
        Initialize RAG Retriever.

        Args:
            namespace: Namespace for document isolation
            embeddings: Embeddings model (defaults to OCI GenAI)
            redis_url: Redis URL for vector store
            chunk_config: Text chunking configuration
            default_k: Default number of results to retrieve
            min_score: Minimum similarity score to include
        """
        self.namespace = namespace
        self.embeddings = embeddings or get_embeddings()
        self.chunk_config = chunk_config or ChunkConfig()
        self.default_k = default_k
        self.min_score = min_score

        self.vector_store = RedisVectorStore(
            redis_url=redis_url,
            namespace=namespace,
        )

        logger.info(
            "Initialized RAG Retriever",
            namespace=namespace,
            embeddings_type=type(self.embeddings).__name__,
        )

    async def initialize(self) -> None:
        """Initialize the retriever (create indexes, etc.)."""
        await self.vector_store.create_index()

    async def close(self) -> None:
        """Close connections."""
        await self.vector_store.close()

    def _chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Split text into chunks with overlap.

        Uses recursive splitting with configurable separators.
        """
        chunks = []
        config = self.chunk_config
        metadata = metadata or {}

        def split_recursive(text: str, separators: list[str]) -> list[str]:
            """Recursively split text using separators."""
            if not text:
                return []

            if len(text) <= config.chunk_size:
                return [text] if len(text) >= config.min_chunk_size else []

            if not separators:
                # No more separators, force split
                result = []
                for i in range(0, len(text), config.chunk_size - config.chunk_overlap):
                    chunk = text[i : i + config.chunk_size]
                    if len(chunk) >= config.min_chunk_size:
                        result.append(chunk)
                return result

            separator = separators[0]
            remaining_separators = separators[1:]

            if separator not in text:
                return split_recursive(text, remaining_separators)

            parts = text.split(separator)
            result = []
            current_chunk = ""

            for part in parts:
                test_chunk = current_chunk + (separator if current_chunk else "") + part

                if len(test_chunk) <= config.chunk_size:
                    current_chunk = test_chunk
                else:
                    if len(current_chunk) >= config.min_chunk_size:
                        result.append(current_chunk)

                    # Add overlap from end of previous chunk
                    if result and config.chunk_overlap > 0:
                        overlap = result[-1][-config.chunk_overlap :]
                        current_chunk = overlap + separator + part
                    else:
                        current_chunk = part

                    # If part is too large, recursively split
                    if len(current_chunk) > config.chunk_size:
                        sub_chunks = split_recursive(current_chunk, remaining_separators)
                        if sub_chunks:
                            result.extend(sub_chunks[:-1])
                            current_chunk = sub_chunks[-1]

            if len(current_chunk) >= config.min_chunk_size:
                result.append(current_chunk)

            return result

        text_chunks = split_recursive(text, config.separators)

        # Create documents with unique IDs
        for i, chunk in enumerate(text_chunks):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
            doc_id = f"{metadata.get('source', 'doc')}_{chunk_hash}_{i}"

            chunks.append(
                Document(
                    id=doc_id,
                    content=chunk,
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                )
            )

        return chunks

    async def ingest_text(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Ingest a text document into the vector store.

        Args:
            text: Document text content
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Number of chunks ingested
        """
        metadata = metadata or {}
        metadata["source"] = source

        chunks = self._chunk_text(text, metadata)

        if not chunks:
            logger.warning("No chunks generated from text", source=source)
            return 0

        count = await self.vector_store.add_documents(chunks, self.embeddings)
        logger.info("Ingested document", source=source, chunks=count)

        return count

    async def ingest_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Ingest a file into the vector store.

        Supports: .txt, .md, .json, .yaml, .py

        Args:
            file_path: Path to file
            metadata: Additional metadata

        Returns:
            Number of chunks ingested
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        metadata = metadata or {}
        metadata["file_path"] = file_path
        metadata["file_name"] = os.path.basename(file_path)

        return await self.ingest_text(
            content,
            source=os.path.basename(file_path),
            metadata=metadata,
        )

    async def ingest_documents(
        self,
        documents: list[Document],
    ) -> int:
        """
        Ingest pre-created documents.

        Args:
            documents: List of Document objects

        Returns:
            Number of documents ingested
        """
        return await self.vector_store.add_documents(documents, self.embeddings)

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        include_context: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of results (defaults to default_k)
            filter_metadata: Metadata filters
            include_context: Whether to build combined context

        Returns:
            RetrievalResult with documents and context
        """
        k = k or self.default_k

        # Generate query embedding
        query_embedding = await self.embeddings.aembed_query(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding,
            k=k,
            filter_metadata=filter_metadata,
        )

        # Filter by minimum score
        results = [r for r in results if r.score >= self.min_score]

        # Build context string
        context = ""
        if include_context and results:
            context = self._build_context(results)

        return RetrievalResult(
            query=query,
            documents=results,
            context=context,
            metadata={
                "k": k,
                "filter": filter_metadata,
                "min_score": self.min_score,
                "result_count": len(results),
            },
        )

    def _build_context(
        self,
        results: list[SearchResult],
        max_tokens: int = 4000,
    ) -> str:
        """
        Build context string from search results.

        Formats results for inclusion in LLM prompt.
        """
        if not results:
            return ""

        # Estimate ~4 chars per token
        max_chars = max_tokens * 4

        context_parts = []
        total_chars = 0

        for result in results:
            source = result.document.metadata.get("source", "unknown")
            content = result.document.content

            part = f"[Source: {source}]\n{content}\n"

            if total_chars + len(part) > max_chars:
                # Truncate if needed
                remaining = max_chars - total_chars
                if remaining > 100:
                    part = part[:remaining] + "..."
                    context_parts.append(part)
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n---\n".join(context_parts)

    async def get_stats(self) -> dict[str, Any]:
        """Get retriever statistics."""
        store_stats = await self.vector_store.get_stats()
        return {
            "namespace": self.namespace,
            "embeddings_type": type(self.embeddings).__name__,
            "chunk_size": self.chunk_config.chunk_size,
            "chunk_overlap": self.chunk_config.chunk_overlap,
            "default_k": self.default_k,
            "min_score": self.min_score,
            **store_stats,
        }

    async def clear(self) -> int:
        """Clear all documents from the retriever."""
        return await self.vector_store.clear()


# Convenience function for creating retrievers
_retrievers: dict[str, RAGRetriever] = {}


async def get_retriever(
    namespace: str = "default",
    use_mock: bool = False,
) -> RAGRetriever:
    """
    Get or create a RAG retriever for a namespace.

    Args:
        namespace: Namespace for document isolation
        use_mock: Use mock embeddings (for testing)

    Returns:
        RAGRetriever instance
    """
    key = f"{namespace}:{use_mock}"

    if key not in _retrievers:
        embeddings = get_embeddings(use_mock=use_mock)
        redis_url = os.getenv("REDIS_URL") or os.getenv("MCP_REDIS_URL")
        default_k = int(os.getenv("RAG_DEFAULT_K", "5"))
        min_score = float(os.getenv("RAG_MIN_SCORE", "0.3"))
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        chunk_config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        retriever = RAGRetriever(
            namespace=namespace,
            embeddings=embeddings,
            redis_url=redis_url,
            chunk_config=chunk_config,
            default_k=default_k,
            min_score=min_score,
        )
        await retriever.initialize()
        _retrievers[key] = retriever

    return _retrievers[key]


async def close_all_retrievers() -> None:
    """Close all cached retrievers."""
    for retriever in _retrievers.values():
        await retriever.close()
    _retrievers.clear()
