import os
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.llm.health import validate_provider_config
from src.llm.oca import ChatOCA, is_oca_authenticated
from src.llm.rate_limiter import wrap_with_rate_limiter

logger = structlog.get_logger(__name__)


class LLMProviderError(Exception):
    """Raised when LLM provider configuration or connection fails."""

    def __init__(self, provider: str, message: str, suggestion: str | None = None):
        self.provider = provider
        self.suggestion = suggestion
        super().__init__(f"[{provider}] {message}")


class LLMFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_llm(
        config: dict[str, Any],
        enable_rate_limiting: bool | None = None,
        max_concurrent: int = 5,
        validate: bool = True,
    ) -> BaseChatModel:
        """Create an LLM instance based on configuration.

        Args:
            config: Configuration dictionary containing 'provider' and other options.
            enable_rate_limiting: Whether to wrap the LLM with rate limiting.
                If None, uses LLM_RATE_LIMITING env var (default: True in production).
            max_concurrent: Maximum concurrent LLM calls when rate limiting (default: 5).
            validate: Whether to validate provider config before creating (default: True).

        Returns:
            BaseChatModel instance, optionally wrapped with rate limiting.

        Raises:
            ValueError: If the provider is unknown.
            LLMProviderError: If provider configuration is invalid.
        """
        provider = config.get("provider", "").lower()

        # Validate configuration
        if validate:
            is_valid, error_msg = validate_provider_config(provider, config)
            if not is_valid:
                raise LLMProviderError(
                    provider=provider,
                    message=error_msg or "Configuration validation failed",
                    suggestion=f"Check your .env.local configuration for {provider}",
                )

        # Create the base LLM
        if provider == "oca":
            # OCA uses shared token cache from ~/.oca/token.json
            # Authentication is handled by oca-langchain-client OAuth flow
            if not is_oca_authenticated():
                import structlog
                structlog.get_logger().warning(
                    "OCA not authenticated - requests will fail until login"
                )
            llm = ChatOCA(
                model=config.get("model_name", "oca/gpt5"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4096),
            )
        elif provider == "anthropic":
            llm = ChatAnthropic(
                api_key=config.get("api_key"),
                model_name=config.get("model_name", "claude-sonnet-4-20250514"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4096),
            )
        elif provider == "openai":
            kwargs = {
                "api_key": config.get("api_key"),
                "model_name": config.get("model_name", "gpt-4-turbo-preview"),
                "temperature": config.get("temperature", 0.7),
                "max_tokens": config.get("max_tokens", 4096),
            }
            # Support custom base URL for LM Studio and other OpenAI-compatible APIs
            if config.get("base_url"):
                kwargs["base_url"] = config.get("base_url")
            llm = ChatOpenAI(**kwargs)
        elif provider == "oci_genai":
            # OCI Generative AI Service using langchain-oci
            try:
                from langchain_community.chat_models.oci_generative_ai import (
                    ChatOCIGenAI,
                )
            except ImportError:
                raise ImportError(
                    "OCI GenAI requires langchain-community package. "
                    "Install with: pip install langchain-community"
                )

            import structlog

            # OCI GenAI requires compartment ID and uses OCI SDK authentication
            compartment_id = config.get("compartment_id") or os.getenv(
                "OCI_GENAI_COMPARTMENT_ID"
            )
            if not compartment_id:
                raise ValueError(
                    "OCI GenAI requires compartment_id in config or "
                    "OCI_GENAI_COMPARTMENT_ID environment variable"
                )

            llm = ChatOCIGenAI(
                model_id=config.get("model_name", "cohere.command-r-plus"),
                compartment_id=compartment_id,
                service_endpoint=config.get("endpoint") or os.getenv(
                    "OCI_GENAI_ENDPOINT",
                    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                ),
                model_kwargs={
                    "temperature": config.get("temperature", 0.7),
                    "max_tokens": config.get("max_tokens", 4096),
                },
            )
            structlog.get_logger().info(
                "OCI GenAI initialized",
                model=config.get("model_name", "cohere.command-r-plus"),
                compartment_id=compartment_id[:30] + "...",
            )
        elif provider == "lm_studio":
            # LM Studio uses OpenAI-compatible API
            import structlog

            base_url = config.get("base_url") or os.getenv(
                "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
            )
            model_name = config.get("model_name", "local-model")

            structlog.get_logger().info(
                "LM Studio initialized",
                model=model_name,
                base_url=base_url,
            )

            llm = ChatOpenAI(
                api_key="lm-studio",  # LM Studio doesn't require a real API key
                model_name=model_name,
                base_url=base_url,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4096),
            )
        elif provider == "ollama":
            # Ollama uses OpenAI-compatible API
            import structlog

            base_url = config.get("base_url") or os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            model_name = config.get("model_name", "llama3.1")

            structlog.get_logger().info(
                "Ollama initialized",
                model=model_name,
                base_url=base_url,
            )

            llm = ChatOpenAI(
                api_key="ollama",  # Ollama doesn't require a real API key
                model_name=model_name,
                base_url=base_url,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4096),
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        # Determine if rate limiting should be applied
        if enable_rate_limiting is None:
            # Default: enabled in production, can be disabled via env var
            enable_rate_limiting = os.getenv("LLM_RATE_LIMITING", "true").lower() == "true"

        if enable_rate_limiting:
            import structlog
            structlog.get_logger().info(
                "LLM rate limiting enabled",
                provider=provider,
                max_concurrent=max_concurrent,
            )
            return wrap_with_rate_limiter(llm, max_concurrent=max_concurrent)

        return llm
