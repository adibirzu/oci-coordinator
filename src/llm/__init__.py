"""LLM module - Multi-provider LLM factory and utilities.

This module provides a unified interface for creating LLM instances
from various providers configured via environment variables.

Usage:
    from src.llm import get_llm

    llm = get_llm()  # Gets LLM based on .env.local configuration
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = structlog.get_logger(__name__)

# Re-export factory
from src.llm.factory import LLMFactory

# Re-export OCA utilities
from src.llm.oca import (
    OCA_CONFIG,
    ChatOCA,
    get_oca_health,
    get_oca_llm,
    is_oca_authenticated,
    oca_token_manager,
    verify_oca_integration,
)

__all__ = [
    "LLMFactory",
    "get_llm",
    "get_llm_config",
    # OCA-specific exports
    "ChatOCA",
    "OCA_CONFIG",
    "get_oca_llm",
    "get_oca_health",
    "is_oca_authenticated",
    "verify_oca_integration",
    "oca_token_manager",
]


def _env_with_channel(key: str, channel_type: str | None) -> str | None:
    """Resolve channel-specific env var, falling back to the default key."""
    if channel_type:
        channel_key = f"{key}_{channel_type.upper()}"
        value = os.getenv(channel_key)
        if value is not None:
            return value
    return os.getenv(key)


def get_llm_config(channel_type: str | None = None) -> dict:
    """Get LLM configuration from environment variables.

    Reads LLM_PROVIDER and related environment variables to build
    a configuration dictionary.

    Args:
        channel_type: Optional channel hint (e.g., "slack", "api") used to
            resolve channel-specific overrides like LLM_PROVIDER_SLACK.

    Returns:
        Configuration dictionary for LLMFactory.create_llm()
    """
    provider = (_env_with_channel("LLM_PROVIDER", channel_type) or "anthropic").lower()

    config = {"provider": provider}

    if provider == "oracle_code_assist" or provider == "oca":
        # Oracle Code Assist - uses shared token cache from ~/.oca/token.json
        # Authentication is handled externally via oca-langchain-client OAuth flow
        config["provider"] = "oca"
        config["model_name"] = _env_with_channel("OCA_MODEL", channel_type) or "oca/gpt5"
        config["temperature"] = float(_env_with_channel("LLM_TEMPERATURE", channel_type) or "0.7")
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")

    elif provider == "anthropic":
        config["api_key"] = _env_with_channel("ANTHROPIC_API_KEY", channel_type)
        config["model_name"] = _env_with_channel("ANTHROPIC_MODEL", channel_type) or "claude-sonnet-4-20250514"
        config["temperature"] = float(_env_with_channel("LLM_TEMPERATURE", channel_type) or "0.7")
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")

    elif provider == "openai":
        config["api_key"] = os.getenv("OPENAI_API_KEY")
        config["model_name"] = _env_with_channel("OPENAI_MODEL", channel_type) or "gpt-4o-mini"
        config["temperature"] = float(_env_with_channel("LLM_TEMPERATURE", channel_type) or "0.7")
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")
        base_url = _env_with_channel("OPENAI_BASE_URL", channel_type)
        if base_url:
            config["base_url"] = base_url

    elif provider == "lm_studio":
        # LM Studio uses OpenAI-compatible API
        config["provider"] = "openai"
        config["api_key"] = _env_with_channel("LLM_API_KEY", channel_type) or "lm-studio"
        config["model_name"] = _env_with_channel("LLM_MODEL", channel_type) or "local-model"
        config["base_url"] = _env_with_channel("LLM_BASE_URL", channel_type) or "http://localhost:1234/v1"
        config["temperature"] = float(_env_with_channel("LLM_TEMPERATURE", channel_type) or "0.7")
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")

    else:
        # Default to Anthropic
        config["provider"] = "anthropic"
        config["api_key"] = _env_with_channel("ANTHROPIC_API_KEY", channel_type)
        config["model_name"] = _env_with_channel("ANTHROPIC_MODEL", channel_type) or "claude-sonnet-4-20250514"
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")

    return config


def get_llm(channel_type: str | None = None) -> BaseChatModel:
    """Get an LLM instance based on environment configuration.

    This is the primary entry point for getting an LLM in the application.
    It reads configuration from environment variables and creates the
    appropriate LLM provider.

    Environment Variables:
        LLM_PROVIDER: Provider name (oracle_code_assist, anthropic, openai, lm_studio)
        OCA_MODEL: Oracle Code Assist model (e.g., oca/gpt5)
        ANTHROPIC_API_KEY: API key for Anthropic
        OPENAI_API_KEY: API key for OpenAI
        See get_llm_config() for full list of supported variables.

    Returns:
        BaseChatModel instance configured for the selected provider

    Raises:
        ValueError: If provider is unknown or configuration is invalid

    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("Hello, how are you?")
    """
    config = get_llm_config(channel_type)

    logger.info(
        "Creating LLM instance",
        provider=config.get("provider"),
        model=config.get("model_name"),
    )

    return LLMFactory.create_llm(config)
