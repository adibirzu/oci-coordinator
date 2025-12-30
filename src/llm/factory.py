from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.llm.providers.oca import ChatOCA


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
            return ChatOCA(
                api_url=config.get("api_url"),
                idcs_url=config.get("idcs_url"),
                client_id=config.get("client_id"),
                redirect_uri=config.get(
                    "redirect_uri", "http://localhost:3001/api/oca/callback"
                ),
                model_name=config.get(
                    "model_name", "oci.generativeai.cohere.command-r-plus"
                ),
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
            return ChatOpenAI(
                api_key=config.get("api_key"),
                model_name=config.get("model_name", "gpt-4-turbo-preview"),
                temperature=config.get("temperature", 0.7),
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
