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
from src.llm.factory import LLMFactory, LLMProviderError

# Re-export health checks
from src.llm.health import (
    DEFAULT_LLM_PRIORITY,
    check_all_providers_health,
    check_anthropic_health,
    check_lm_studio_health,
    check_oci_genai_health,
    check_ollama_health,
    check_openai_health,
    check_provider_health,
    get_first_available_provider,
    get_llm_with_fallback,
    print_llm_availability_report,
    print_startup_status,
    validate_llm_startup,
    validate_provider_config,
)

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
    "LLMProviderError",
    "get_llm",
    "get_llm_config",
    "get_llm_with_auto_fallback",
    # Health checks
    "check_provider_health",
    "check_all_providers_health",
    "check_lm_studio_health",
    "check_ollama_health",
    "check_openai_health",
    "check_anthropic_health",
    "check_oci_genai_health",
    "validate_provider_config",
    "validate_llm_startup",
    "print_startup_status",
    # Fallback / Priority
    "DEFAULT_LLM_PRIORITY",
    "get_first_available_provider",
    "get_llm_with_fallback",
    "print_llm_availability_report",
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
        config["model_name"] = _env_with_channel("OCA_MODEL", channel_type) or "oca/gpt-4.1"
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
        # LM Studio uses OpenAI-compatible API via factory's lm_studio handler
        config["provider"] = "lm_studio"
        config["model_name"] = (
            _env_with_channel("LM_STUDIO_MODEL", channel_type)
            or os.getenv("LLM_MODEL")
            or "local-model"
        )
        config["base_url"] = (
            _env_with_channel("LM_STUDIO_BASE_URL", channel_type)
            or os.getenv("LLM_BASE_URL")
            or "http://localhost:1234/v1"
        )
        config["temperature"] = float(_env_with_channel("LLM_TEMPERATURE", channel_type) or "0.7")
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")

    elif provider == "ollama":
        # Ollama uses OpenAI-compatible API via factory's ollama handler
        config["provider"] = "ollama"
        config["model_name"] = (
            _env_with_channel("OLLAMA_MODEL", channel_type)
            or os.getenv("LLM_MODEL")
            or "llama3.1"
        )
        config["base_url"] = (
            _env_with_channel("OLLAMA_BASE_URL", channel_type)
            or os.getenv("LLM_BASE_URL")
            or "http://localhost:11434/v1"
        )
        config["temperature"] = float(_env_with_channel("LLM_TEMPERATURE", channel_type) or "0.7")
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")

    elif provider == "oci_genai":
        # OCI Generative AI Service
        config["provider"] = "oci_genai"
        config["model_name"] = (
            _env_with_channel("OCI_GENAI_MODEL_ID", channel_type)
            or os.getenv("OCI_GENAI_MODEL")
            or "cohere.command-r-plus"
        )
        config["compartment_id"] = os.getenv("OCI_GENAI_COMPARTMENT_ID")
        config["endpoint"] = os.getenv("OCI_GENAI_ENDPOINT")
        config["temperature"] = float(_env_with_channel("LLM_TEMPERATURE", channel_type) or "0.7")
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")

    else:
        # Default to Anthropic
        config["provider"] = "anthropic"
        config["api_key"] = _env_with_channel("ANTHROPIC_API_KEY", channel_type)
        config["model_name"] = _env_with_channel("ANTHROPIC_MODEL", channel_type) or "claude-sonnet-4-20250514"
        config["max_tokens"] = int(_env_with_channel("LLM_MAX_TOKENS", channel_type) or "4096")

    return config


async def get_llm_with_auto_fallback(
    channel_type: str | None = None,
    timeout: float = 5.0,
) -> BaseChatModel:
    """Get an LLM instance with automatic fallback to available providers.

    This function first checks if the configured provider is available,
    and if not, falls back to the next available provider based on priority.

    Priority order (configurable via LLM_PROVIDER_PRIORITY env var):
    1. lm_studio (local)
    2. ollama (local)
    3. oca (Oracle Code Assist)
    4. oci_genai (OCI GenAI)
    5. anthropic (Anthropic Claude)
    6. openai (OpenAI)

    Args:
        channel_type: Optional channel hint for channel-specific configuration
        timeout: Health check timeout per provider in seconds

    Returns:
        BaseChatModel instance from the first available provider

    Raises:
        RuntimeError: If no LLM providers are available

    Example:
        >>> llm = await get_llm_with_auto_fallback()
        >>> response = llm.invoke("Hello!")
    """
    from src.llm.health import get_llm_with_fallback

    # Get the first available provider
    provider, health = await get_llm_with_fallback(timeout=timeout)

    logger.info(
        "Creating LLM with auto-fallback",
        selected_provider=provider,
        status=health.get("status"),
    )

    # Build config for the selected provider
    config = get_llm_config(channel_type)
    config["provider"] = provider

    # If fallback provider differs from configured, update relevant config
    if provider != config.get("provider"):
        # Get default config for the fallback provider
        original_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        if provider != original_provider:
            # Reset provider-specific config for fallback
            if provider == "lm_studio":
                config["base_url"] = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
                config["model_name"] = os.getenv("LM_STUDIO_MODEL", "local-model")
            elif provider == "ollama":
                config["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
                config["model_name"] = os.getenv("OLLAMA_MODEL", "llama3.1")
            elif provider == "oca":
                config["model_name"] = os.getenv("OCA_MODEL", "oca/gpt-4.1")
            elif provider == "anthropic":
                config["api_key"] = os.getenv("ANTHROPIC_API_KEY")
                config["model_name"] = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            elif provider == "openai":
                config["api_key"] = os.getenv("OPENAI_API_KEY")
                config["model_name"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            elif provider == "oci_genai":
                config["compartment_id"] = os.getenv("OCI_GENAI_COMPARTMENT_ID")
                config["model_name"] = os.getenv("OCI_GENAI_MODEL_ID", "cohere.command-r-plus")

    return LLMFactory.create_llm(config)


def get_llm(channel_type: str | None = None) -> BaseChatModel:
    """Get an LLM instance based on environment configuration.

    This is the primary entry point for getting an LLM in the application.
    It reads configuration from environment variables and creates the
    appropriate LLM provider.

    Environment Variables:
        LLM_PROVIDER: Provider name (oca, anthropic, openai, lm_studio, ollama, oci_genai)
        OCA_MODEL: Oracle Code Assist model (e.g., oca/gpt-4.1)
        ANTHROPIC_API_KEY: API key for Anthropic
        OPENAI_API_KEY: API key for OpenAI
        LM_STUDIO_BASE_URL: LM Studio API endpoint (default: http://localhost:1234/v1)
        OLLAMA_BASE_URL: Ollama API endpoint (default: http://localhost:11434/v1)
        OCI_GENAI_COMPARTMENT_ID: OCI compartment for GenAI
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
