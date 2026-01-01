"""
OCI Generative AI Embeddings.

Provides embedding generation using OCI AI Studio's Generative AI service.
Supports Cohere embed models for high-quality document and query embeddings.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import structlog
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    import oci

logger = structlog.get_logger(__name__)

# Default embedding model in OCI GenAI
DEFAULT_EMBEDDING_MODEL = "cohere.embed-english-v3.0"
DEFAULT_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
DEFAULT_SERVICE_ENDPOINT = os.getenv(
    "OCI_GENAI_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)


class OCIEmbeddings(Embeddings):
    """
    OCI Generative AI Embeddings.

    Uses OCI AI Studio's embedding models for generating vector representations
    of text. Supports both document and query embeddings with optimized input types.

    Attributes:
        compartment_id: OCI compartment OCID
        model_id: Embedding model ID (default: cohere.embed-english-v3.0)
        service_endpoint: OCI GenAI service endpoint
        truncate: How to handle text exceeding max length (NONE, START, END)
    """

    def __init__(
        self,
        compartment_id: str | None = None,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        service_endpoint: str | None = None,
        truncate: str = "END",
        batch_size: int = 96,
        config_profile: str = "DEFAULT",
    ):
        """
        Initialize OCI Embeddings.

        Args:
            compartment_id: OCI compartment OCID (defaults to OCI_COMPARTMENT_ID env var)
            model_id: Embedding model ID
            service_endpoint: OCI GenAI endpoint (defaults to us-chicago-1)
            truncate: Truncation strategy for long texts (NONE, START, END)
            batch_size: Maximum texts per API call (max 96 for Cohere)
            config_profile: OCI config profile name
        """
        self.compartment_id = compartment_id or DEFAULT_COMPARTMENT_ID
        self.model_id = model_id
        self.service_endpoint = service_endpoint or DEFAULT_SERVICE_ENDPOINT
        self.truncate = truncate
        self.batch_size = min(batch_size, 96)  # Cohere max is 96
        self.config_profile = config_profile

        # Initialize OCI client lazily
        self._client: oci.generative_ai_inference.GenerativeAiInferenceClient | None = None

        if not self.compartment_id:
            logger.warning(
                "No compartment ID configured for OCI embeddings. "
                "Set OCI_COMPARTMENT_ID environment variable."
            )

    @property
    def client(self) -> oci.generative_ai_inference.GenerativeAiInferenceClient:
        """Get or create the OCI GenAI client."""
        if self._client is None:
            try:
                import oci

                config = oci.config.from_file(profile_name=self.config_profile)
                self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config=config,
                    service_endpoint=self.service_endpoint,
                )
                logger.info(
                    "Initialized OCI GenAI client",
                    endpoint=self.service_endpoint,
                    model=self.model_id,
                )
            except Exception as e:
                logger.error("Failed to initialize OCI GenAI client", error=str(e))
                raise

        return self._client

    def embed_documents(
        self, texts: list[str], input_type: str = "search_document"
    ) -> list[list[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed
            input_type: Cohere input type (search_document, search_query, classification, clustering)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self.compartment_id:
            logger.warning("No compartment ID - returning zero embeddings")
            return [[0.0] * 1024] * len(texts)  # Cohere embed-v3 is 1024 dims

        try:
            import oci

            all_embeddings = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                embed_text_details = oci.generative_ai_inference.models.EmbedTextDetails(
                    inputs=batch,
                    serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                        model_id=self.model_id
                    ),
                    compartment_id=self.compartment_id,
                    input_type=input_type.upper(),
                    truncate=self.truncate,
                )

                response = self.client.embed_text(embed_text_details)
                all_embeddings.extend(response.data.embeddings)

                logger.debug(
                    "Embedded batch",
                    batch_size=len(batch),
                    total_processed=len(all_embeddings),
                )

            return all_embeddings

        except Exception as e:
            logger.error("Failed to embed documents", error=str(e), count=len(texts))
            raise

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Uses 'search_query' input type for optimal retrieval performance.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents([text], input_type="search_query")
        return embeddings[0] if embeddings else []

    async def aembed_documents(
        self, texts: list[str], input_type: str = "search_document"
    ) -> list[list[float]]:
        """Async version of embed_documents."""
        # OCI SDK doesn't have native async, so we run in executor
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.embed_documents(texts, input_type)
        )

    async def aembed_query(self, text: str) -> list[float]:
        """Async version of embed_query."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed_query(text))


class MockEmbeddings(Embeddings):
    """
    Mock embeddings for testing without OCI.

    Generates deterministic pseudo-embeddings based on text hash.
    """

    def __init__(self, dimensions: int = 1024):
        self.dimensions = dimensions

    def _hash_to_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Use hash bytes to seed random-like values
        embedding = []
        for i in range(self.dimensions):
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1] range
            val = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(val)
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_to_embedding(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def get_embeddings(use_mock: bool = False) -> Embeddings:
    """
    Get an embeddings instance.

    Args:
        use_mock: If True, return MockEmbeddings for testing

    Returns:
        Embeddings instance (OCI or Mock)
    """
    if use_mock or not os.getenv("OCI_COMPARTMENT_ID"):
        logger.info("Using mock embeddings")
        return MockEmbeddings()

    return OCIEmbeddings()
