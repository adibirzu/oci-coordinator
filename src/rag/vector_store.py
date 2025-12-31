"""
Vector Store Implementation.

Provides Redis-based vector storage for RAG retrieval.
Uses Redis Vector Similarity Search (VSS) or falls back to
brute-force cosine similarity for development.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from src.rag.embeddings import Embeddings

logger = structlog.get_logger(__name__)

# Redis key prefixes
VECTOR_PREFIX = "rag:vector:"
DOCUMENT_PREFIX = "rag:doc:"
INDEX_KEY = "rag:index"


@dataclass
class Document:
    """A document with content and metadata."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """Result from a similarity search."""

    document: Document
    score: float  # Cosine similarity (higher is better)
    rank: int


class RedisVectorStore:
    """
    Redis-based vector store for RAG.

    Stores document embeddings in Redis and provides similarity search.
    Uses Redis VSS if available, otherwise falls back to brute-force search.

    Attributes:
        namespace: Namespace prefix for isolation
        dimensions: Embedding vector dimensions
        index_name: Name of the vector index
    """

    def __init__(
        self,
        redis_url: str | None = None,
        namespace: str = "default",
        dimensions: int = 1024,
        distance_metric: str = "COSINE",
    ):
        """
        Initialize Redis Vector Store.

        Args:
            redis_url: Redis connection URL
            namespace: Namespace for key isolation
            dimensions: Embedding dimensions (1024 for Cohere embed-v3)
            distance_metric: Distance metric (COSINE, L2, IP)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.namespace = namespace
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self._redis: Redis | None = None
        self._vss_available: bool | None = None

    async def get_redis(self) -> "Redis":
        """Get or create Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis

            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            logger.info("Connected to Redis", url=self.redis_url[:30] + "...")

        return self._redis

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _key(self, suffix: str) -> str:
        """Generate namespaced key."""
        return f"rag:{self.namespace}:{suffix}"

    async def _check_vss(self) -> bool:
        """Check if Redis VSS is available."""
        if self._vss_available is None:
            try:
                redis = await self.get_redis()
                await redis.execute_command("FT._LIST")
                self._vss_available = True
                logger.info("Redis VSS available")
            except Exception:
                self._vss_available = False
                logger.info("Redis VSS not available, using brute-force search")

        return self._vss_available

    async def create_index(self, recreate: bool = False) -> bool:
        """
        Create vector index if using Redis VSS.

        Args:
            recreate: If True, drop and recreate index

        Returns:
            True if index was created/exists
        """
        if not await self._check_vss():
            return False

        redis = await self.get_redis()
        index_name = self._key("idx")

        try:
            if recreate:
                try:
                    await redis.execute_command("FT.DROPINDEX", index_name)
                except Exception:
                    pass

            # Create index with vector field
            await redis.execute_command(
                "FT.CREATE",
                index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self._key("doc:"),
                "SCHEMA",
                "content",
                "TEXT",
                "embedding",
                "VECTOR",
                "FLAT",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.dimensions),
                "DISTANCE_METRIC",
                self.distance_metric,
            )
            logger.info("Created vector index", name=index_name)
            return True

        except Exception as e:
            if "Index already exists" in str(e):
                logger.debug("Index already exists", name=index_name)
                return True
            logger.error("Failed to create index", error=str(e))
            return False

    async def add_documents(
        self,
        documents: list[Document],
        embeddings_model: "Embeddings",
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: Documents to add
            embeddings_model: Embeddings model for vectorization

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        redis = await self.get_redis()

        # Generate embeddings for documents without them
        texts_to_embed = []
        docs_needing_embeddings = []

        for doc in documents:
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                docs_needing_embeddings.append(doc)

        if texts_to_embed:
            embeddings = await embeddings_model.aembed_documents(texts_to_embed)
            for doc, emb in zip(docs_needing_embeddings, embeddings):
                doc.embedding = emb

        # Store in Redis
        use_vss = await self._check_vss()
        pipeline = redis.pipeline()

        for doc in documents:
            doc_key = self._key(f"doc:{doc.id}")

            if use_vss:
                # Store as hash for VSS
                embedding_bytes = np.array(doc.embedding, dtype=np.float32).tobytes()
                pipeline.hset(
                    doc_key,
                    mapping={
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": json.dumps(doc.metadata),
                        "embedding": embedding_bytes,
                    },
                )
            else:
                # Store embedding separately for brute-force
                pipeline.hset(
                    doc_key,
                    mapping={
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": json.dumps(doc.metadata),
                    },
                )
                # Store embedding as JSON list
                emb_key = self._key(f"emb:{doc.id}")
                pipeline.set(emb_key, json.dumps(doc.embedding))

            # Add to index set
            pipeline.sadd(self._key("docs"), doc.id)

        await pipeline.execute()
        logger.info("Added documents", count=len(documents), namespace=self.namespace)

        return len(documents)

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results sorted by similarity
        """
        if await self._check_vss():
            return await self._search_vss(query_embedding, k, filter_metadata)
        else:
            return await self._search_brute_force(query_embedding, k, filter_metadata)

    async def _search_vss(
        self,
        query_embedding: list[float],
        k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Search using Redis VSS."""
        redis = await self.get_redis()
        index_name = self._key("idx")

        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        # Build query
        query = f"*=>[KNN {k} @embedding $vec AS score]"

        try:
            results = await redis.execute_command(
                "FT.SEARCH",
                index_name,
                query,
                "PARAMS",
                "2",
                "vec",
                query_bytes,
                "RETURN",
                "4",
                "id",
                "content",
                "metadata",
                "score",
                "SORTBY",
                "score",
                "LIMIT",
                "0",
                str(k),
                "DIALECT",
                "2",
            )

            search_results = []
            # Parse results (format: [count, key1, fields1, key2, fields2, ...])
            if results and len(results) > 1:
                for i in range(1, len(results), 2):
                    if i + 1 < len(results):
                        fields = dict(zip(results[i + 1][::2], results[i + 1][1::2]))
                        doc = Document(
                            id=fields.get("id", ""),
                            content=fields.get("content", ""),
                            metadata=json.loads(fields.get("metadata", "{}")),
                        )
                        # Convert distance to similarity (1 - distance for cosine)
                        score = 1.0 - float(fields.get("score", 1.0))
                        search_results.append(
                            SearchResult(document=doc, score=score, rank=len(search_results))
                        )

            return search_results

        except Exception as e:
            logger.error("VSS search failed", error=str(e))
            return await self._search_brute_force(query_embedding, k, filter_metadata)

    async def _search_brute_force(
        self,
        query_embedding: list[float],
        k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Brute-force cosine similarity search."""
        redis = await self.get_redis()

        # Get all document IDs
        doc_ids = await redis.smembers(self._key("docs"))
        if not doc_ids:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)

        scores = []
        for doc_id in doc_ids:
            doc_key = self._key(f"doc:{doc_id}")
            emb_key = self._key(f"emb:{doc_id}")

            doc_data = await redis.hgetall(doc_key)
            emb_data = await redis.get(emb_key)

            if not doc_data or not emb_data:
                continue

            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=json.loads(doc_data.get("metadata", "{}")),
            )

            # Apply metadata filter
            if filter_metadata:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue

            doc_vec = np.array(json.loads(emb_data))
            doc_norm = np.linalg.norm(doc_vec)

            # Cosine similarity
            if query_norm > 0 and doc_norm > 0:
                similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
            else:
                similarity = 0.0

            scores.append((doc, similarity))

        # Sort by similarity (descending) and take top k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [
            SearchResult(document=doc, score=score, rank=i)
            for i, (doc, score) in enumerate(scores[:k])
        ]

        return results

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store."""
        redis = await self.get_redis()
        doc_key = self._key(f"doc:{doc_id}")
        emb_key = self._key(f"emb:{doc_id}")

        pipeline = redis.pipeline()
        pipeline.delete(doc_key)
        pipeline.delete(emb_key)
        pipeline.srem(self._key("docs"), doc_id)
        results = await pipeline.execute()

        deleted = results[0] > 0
        if deleted:
            logger.debug("Deleted document", doc_id=doc_id)

        return deleted

    async def clear(self) -> int:
        """Clear all documents from the namespace."""
        redis = await self.get_redis()

        doc_ids = await redis.smembers(self._key("docs"))
        if not doc_ids:
            return 0

        pipeline = redis.pipeline()
        for doc_id in doc_ids:
            pipeline.delete(self._key(f"doc:{doc_id}"))
            pipeline.delete(self._key(f"emb:{doc_id}"))

        pipeline.delete(self._key("docs"))
        await pipeline.execute()

        logger.info("Cleared vector store", namespace=self.namespace, count=len(doc_ids))
        return len(doc_ids)

    async def count(self) -> int:
        """Get number of documents in the store."""
        redis = await self.get_redis()
        return await redis.scard(self._key("docs"))

    async def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        return {
            "namespace": self.namespace,
            "dimensions": self.dimensions,
            "document_count": await self.count(),
            "vss_available": await self._check_vss(),
            "distance_metric": self.distance_metric,
        }
