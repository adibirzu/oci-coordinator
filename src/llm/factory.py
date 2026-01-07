import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.llm.oca import ChatOCA, is_oca_authenticated
from src.llm.rate_limiter import wrap_with_rate_limiter


class LLMFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_llm(
        config: dict[str, Any],
        enable_rate_limiting: bool | None = None,
        max_concurrent: int = 5,
    ) -> BaseChatModel:
        """Create an LLM instance based on configuration.

        Args:
            config: Configuration dictionary containing 'provider' and other options.
            enable_rate_limiting: Whether to wrap the LLM with rate limiting.
                If None, uses LLM_RATE_LIMITING env var (default: True in production).
            max_concurrent: Maximum concurrent LLM calls when rate limiting (default: 5).

        Returns:
            BaseChatModel instance, optionally wrapped with rate limiting.

        Raises:
            ValueError: If the provider is unknown.
        """
        provider = config.get("provider", "").lower()

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
                model_name=config.get("model_name", "claude-3-sonnet-20240229"),
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
