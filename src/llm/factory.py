from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.llm.oca import ChatOCA, is_oca_authenticated


class LLMFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_llm(config: dict[str, Any]) -> BaseChatModel:
        """Create an LLM instance based on configuration.

        Args:
            config: Configuration dictionary containing 'provider' and other options.

        Returns:
            BaseChatModel instance.

        Raises:
            ValueError: If the provider is unknown.
        """
        provider = config.get("provider", "").lower()

        if provider == "oca":
            # OCA uses shared token cache from ~/.oca/token.json
            # Authentication is handled by oca-langchain-client OAuth flow
            if not is_oca_authenticated():
                import structlog
                structlog.get_logger().warning(
                    "OCA not authenticated - requests will fail until login"
                )
            return ChatOCA(
                model=config.get("model_name", "oca/gpt5"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4096),
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                api_key=config.get("api_key"),
                model_name=config.get("model_name", "claude-3-sonnet-20240229"),
                temperature=config.get("temperature", 0.7),
            )
        elif provider == "openai":
            kwargs = {
                "api_key": config.get("api_key"),
                "model_name": config.get("model_name", "gpt-4-turbo-preview"),
                "temperature": config.get("temperature", 0.7),
            }
            # Support custom base URL for LM Studio and other OpenAI-compatible APIs
            if config.get("base_url"):
                kwargs["base_url"] = config.get("base_url")
            return ChatOpenAI(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
