"""
LLM Provider Health Checks.

Provides connectivity validation for all supported LLM providers.
Use these functions to verify provider availability before starting services.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


async def check_openai_compatible_health(
    base_url: str,
    api_key: str = "test",
    provider_name: str = "OpenAI",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check health of an OpenAI-compatible API endpoint.

    Args:
        base_url: The base URL of the API (e.g., http://localhost:1234/v1)
        api_key: API key (optional for local providers)
        provider_name: Human-readable provider name for logging
        timeout: Connection timeout in seconds

    Returns:
        Dictionary with status, available models, and any errors
    """
    result = {
        "provider": provider_name,
        "base_url": base_url,
        "status": "unknown",
        "connected": False,
        "models": [],
        "error": None,
    }

    # Normalize base URL
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    models_url = f"{base_url}/models"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = await client.get(models_url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id", "unknown") for m in models]

                result["status"] = "healthy"
                result["connected"] = True
                result["models"] = model_ids[:10]  # Limit to first 10

                logger.info(
                    f"{provider_name} health check passed",
                    base_url=base_url,
                    model_count=len(models),
                )
            else:
                result["status"] = "unhealthy"
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning(
                    f"{provider_name} health check failed",
                    base_url=base_url,
                    status_code=response.status_code,
                )

    except httpx.ConnectError as e:
        result["status"] = "unreachable"
        result["error"] = f"Connection failed: {provider_name} server not running at {base_url}"
        logger.error(
            f"{provider_name} connection failed",
            base_url=base_url,
            error=str(e),
        )

    except httpx.TimeoutException:
        result["status"] = "timeout"
        result["error"] = f"Connection timed out after {timeout}s"
        logger.error(
            f"{provider_name} connection timeout",
            base_url=base_url,
            timeout=timeout,
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(
            f"{provider_name} health check error",
            base_url=base_url,
            error=str(e),
        )

    return result


async def check_lm_studio_health(
    base_url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check LM Studio server health.

    Args:
        base_url: LM Studio API URL (default: from env or localhost:1234/v1)
        timeout: Connection timeout

    Returns:
        Health check result dictionary
    """
    url = base_url or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    return await check_openai_compatible_health(
        base_url=url,
        api_key="lm-studio",
        provider_name="LM Studio",
        timeout=timeout,
    )


async def check_ollama_health(
    base_url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check Ollama server health.

    Args:
        base_url: Ollama API URL (default: from env or localhost:11434/v1)
        timeout: Connection timeout

    Returns:
        Health check result dictionary
    """
    url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return await check_openai_compatible_health(
        base_url=url,
        api_key="ollama",
        provider_name="Ollama",
        timeout=timeout,
    )


async def check_openai_health(
    api_key: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check OpenAI API health.

    Args:
        api_key: OpenAI API key (default: from env)
        timeout: Connection timeout

    Returns:
        Health check result dictionary
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return {
            "provider": "OpenAI",
            "status": "not_configured",
            "connected": False,
            "error": "OPENAI_API_KEY not set",
        }

    return await check_openai_compatible_health(
        base_url="https://api.openai.com/v1",
        api_key=key,
        provider_name="OpenAI",
        timeout=timeout,
    )


async def check_anthropic_health(
    api_key: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check Anthropic API health.

    Args:
        api_key: Anthropic API key (default: from env)
        timeout: Connection timeout

    Returns:
        Health check result dictionary
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return {
            "provider": "Anthropic",
            "status": "not_configured",
            "connected": False,
            "error": "ANTHROPIC_API_KEY not set",
        }

    result = {
        "provider": "Anthropic",
        "status": "unknown",
        "connected": False,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Anthropic doesn't have a models endpoint, so we ping the API
            # with an invalid request to check connectivity
            response = await client.get(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
            )

            # 405 Method Not Allowed is expected (GET not allowed on /messages)
            # This confirms the API is reachable and key is recognized
            if response.status_code in (405, 401, 200):
                result["status"] = "healthy" if response.status_code != 401 else "invalid_key"
                result["connected"] = response.status_code != 401
                if response.status_code == 401:
                    result["error"] = "Invalid API key"
                logger.info("Anthropic health check passed")
            else:
                result["status"] = "unhealthy"
                result["error"] = f"HTTP {response.status_code}"

    except httpx.ConnectError as e:
        result["status"] = "unreachable"
        result["error"] = f"Connection failed: {e}"
        logger.error("Anthropic connection failed", error=str(e))

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def check_oci_genai_health(
    compartment_id: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check OCI GenAI service health.

    Args:
        compartment_id: OCI compartment ID (default: from env)
        timeout: Connection timeout

    Returns:
        Health check result dictionary
    """
    comp_id = compartment_id or os.getenv("OCI_GENAI_COMPARTMENT_ID")
    if not comp_id:
        return {
            "provider": "OCI GenAI",
            "status": "not_configured",
            "connected": False,
            "error": "OCI_GENAI_COMPARTMENT_ID not set",
        }

    result = {
        "provider": "OCI GenAI",
        "compartment_id": comp_id[:30] + "..." if len(comp_id) > 30 else comp_id,
        "status": "unknown",
        "connected": False,
        "error": None,
    }

    try:
        # OCI GenAI uses OCI SDK authentication
        import oci

        config = oci.config.from_file()
        endpoint = os.getenv(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        )

        client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config, service_endpoint=endpoint
        )

        # Just verify the client can be created - actual call would require model ID
        result["status"] = "configured"
        result["connected"] = True
        result["endpoint"] = endpoint
        logger.info("OCI GenAI health check passed")

    except ImportError:
        result["status"] = "not_installed"
        result["error"] = "OCI SDK not installed"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error("OCI GenAI health check failed", error=str(e))

    return result


async def check_provider_health(
    provider: str,
    config: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check health of a specific LLM provider.

    Args:
        provider: Provider name (oca, anthropic, openai, lm_studio, ollama, oci_genai)
        config: Optional configuration overrides
        timeout: Connection timeout

    Returns:
        Health check result dictionary
    """
    config = config or {}
    provider = provider.lower()

    if provider == "oca":
        from src.llm.oca import get_oca_health
        return await get_oca_health()

    elif provider == "anthropic":
        return await check_anthropic_health(
            api_key=config.get("api_key"),
            timeout=timeout,
        )

    elif provider == "openai":
        return await check_openai_health(
            api_key=config.get("api_key"),
            timeout=timeout,
        )

    elif provider == "lm_studio":
        return await check_lm_studio_health(
            base_url=config.get("base_url"),
            timeout=timeout,
        )

    elif provider == "ollama":
        return await check_ollama_health(
            base_url=config.get("base_url"),
            timeout=timeout,
        )

    elif provider == "oci_genai":
        return await check_oci_genai_health(
            compartment_id=config.get("compartment_id"),
            timeout=timeout,
        )

    else:
        return {
            "provider": provider,
            "status": "unknown_provider",
            "connected": False,
            "error": f"Unknown provider: {provider}",
        }


async def check_all_providers_health(timeout: float = 10.0) -> dict[str, dict[str, Any]]:
    """Check health of all configured LLM providers.

    Returns:
        Dictionary mapping provider names to health check results
    """
    providers = ["oca", "anthropic", "openai", "lm_studio", "ollama", "oci_genai"]
    results = {}

    for provider in providers:
        try:
            results[provider] = await check_provider_health(provider, timeout=timeout)
        except Exception as e:
            results[provider] = {
                "provider": provider,
                "status": "error",
                "connected": False,
                "error": str(e),
            }

    return results


def validate_provider_config(provider: str, config: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate provider configuration before creating LLM.

    Args:
        provider: Provider name
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    provider = provider.lower()

    if provider == "anthropic":
        if not config.get("api_key"):
            return False, "ANTHROPIC_API_KEY is required"

    elif provider == "openai":
        if not config.get("api_key"):
            return False, "OPENAI_API_KEY is required"

    elif provider == "oci_genai":
        if not config.get("compartment_id") and not os.getenv("OCI_GENAI_COMPARTMENT_ID"):
            return False, "OCI_GENAI_COMPARTMENT_ID is required"

    elif provider == "lm_studio":
        # LM Studio doesn't require API key, just warn about connectivity
        base_url = config.get("base_url") or os.getenv(
            "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
        )
        logger.info(
            "LM Studio configured",
            base_url=base_url,
            note="Ensure LM Studio server is running",
        )

    elif provider == "ollama":
        # Ollama doesn't require API key, just warn about connectivity
        base_url = config.get("base_url") or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        logger.info(
            "Ollama configured",
            base_url=base_url,
            note="Ensure Ollama server is running",
        )

    return True, None


async def validate_llm_startup(
    provider: str | None = None,
    config: dict[str, Any] | None = None,
    timeout: float = 10.0,
    fail_on_unreachable: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate LLM provider at startup.

    This is a comprehensive check that should be called before starting
    services that depend on an LLM. It checks both configuration and
    connectivity.

    Args:
        provider: Provider name (default: from LLM_PROVIDER env var)
        config: Optional configuration overrides
        timeout: Connection timeout in seconds
        fail_on_unreachable: If True, treat unreachable local servers as failure

    Returns:
        Tuple of (success, message, details)

    Example:
        success, msg, details = await validate_llm_startup()
        if not success:
            print(f"LLM startup failed: {msg}")
            sys.exit(1)
    """
    import os as _os

    provider = provider or _os.getenv("LLM_PROVIDER", "anthropic").lower()
    config = config or {}
    details: dict[str, Any] = {"provider": provider, "checks": []}

    # Step 1: Validate configuration
    is_valid, error_msg = validate_provider_config(provider, config)
    if not is_valid:
        details["checks"].append({"name": "config", "passed": False, "error": error_msg})
        return False, f"Configuration error: {error_msg}", details

    details["checks"].append({"name": "config", "passed": True})

    # Step 2: Check connectivity
    try:
        health = await check_provider_health(provider, config, timeout=timeout)
        details["health"] = health

        if health.get("connected", False):
            details["checks"].append({"name": "connectivity", "passed": True})
            return True, f"{provider} is ready", details

        elif health.get("status") == "unreachable":
            error = health.get("error", "Server not reachable")
            details["checks"].append({"name": "connectivity", "passed": False, "error": error})

            # For local providers, provide helpful guidance
            if provider == "lm_studio":
                suggestion = (
                    f"LM Studio not running. Start LM Studio and load a model, "
                    f"or switch to a different provider (e.g., LLM_PROVIDER=oca)"
                )
            elif provider == "ollama":
                suggestion = (
                    f"Ollama not running. Start with 'ollama serve', "
                    f"or switch to a different provider"
                )
            else:
                suggestion = f"Check {provider} configuration and connectivity"

            details["suggestion"] = suggestion

            if fail_on_unreachable:
                return False, f"{provider} is unreachable: {error}", details
            else:
                logger.warning(
                    f"{provider} is unreachable - LLM calls will fail",
                    error=error,
                    suggestion=suggestion,
                )
                return True, f"{provider} configured but unreachable (will retry on use)", details

        elif health.get("status") == "not_configured":
            error = health.get("error", "Not configured")
            details["checks"].append({"name": "connectivity", "passed": False, "error": error})
            return False, f"{provider}: {error}", details

        else:
            # Unknown status - treat as warning
            details["checks"].append({
                "name": "connectivity",
                "passed": False,
                "error": health.get("error", "Unknown status"),
            })
            return True, f"{provider} status: {health.get('status')}", details

    except Exception as e:
        error_msg = str(e)
        details["checks"].append({"name": "connectivity", "passed": False, "error": error_msg})
        logger.error(f"Health check failed for {provider}", error=error_msg)
        return False, f"Health check failed: {error_msg}", details


# Default LLM provider priority order (lower number = higher priority)
# Users can override by setting LLM_PROVIDER_PRIORITY env var as comma-separated list
DEFAULT_LLM_PRIORITY = [
    "lm_studio",   # 1. Local LM Studio (lowest latency, no cost)
    "ollama",      # 2. Local Ollama (lowest latency, no cost)
    "oca",         # 3. Oracle Code Assist (enterprise)
    "oci_genai",   # 4. OCI GenAI (enterprise)
    "anthropic",   # 5. Anthropic Claude (cloud)
    "openai",      # 6. OpenAI (cloud)
]


async def get_first_available_provider(
    priority: list[str] | None = None,
    timeout: float = 5.0,
) -> tuple[str | None, dict[str, Any]]:
    """Find the first available LLM provider based on priority order.

    Checks providers in priority order and returns the first healthy one.

    Args:
        priority: List of provider names in priority order.
                  Default: LLM_PROVIDER_PRIORITY env var or DEFAULT_LLM_PRIORITY
        timeout: Health check timeout per provider in seconds

    Returns:
        Tuple of (provider_name, health_details) or (None, empty_dict) if none available
    """
    # Get priority from env var or use default
    if priority is None:
        env_priority = os.getenv("LLM_PROVIDER_PRIORITY", "")
        if env_priority:
            priority = [p.strip().lower() for p in env_priority.split(",") if p.strip()]
        else:
            priority = DEFAULT_LLM_PRIORITY

    logger.info(
        "Checking LLM providers in priority order",
        priority=priority,
    )

    for provider in priority:
        try:
            health = await check_provider_health(provider, timeout=timeout)

            if health.get("connected", False):
                logger.info(
                    "Found available LLM provider",
                    provider=provider,
                    status=health.get("status"),
                )
                return provider, health

            logger.debug(
                "LLM provider not available",
                provider=provider,
                status=health.get("status"),
                error=health.get("error"),
            )

        except Exception as e:
            logger.debug(
                "LLM provider check failed",
                provider=provider,
                error=str(e),
            )
            continue

    logger.warning("No LLM providers available from priority list", priority=priority)
    return None, {}


async def get_llm_with_fallback(
    preferred_provider: str | None = None,
    priority: list[str] | None = None,
    timeout: float = 5.0,
) -> tuple[str, dict[str, Any]]:
    """Get the best available LLM provider with fallback.

    First tries the preferred provider (from config/env), then falls back
    to the priority list if the preferred provider is unavailable.

    Args:
        preferred_provider: Provider to try first (default: from LLM_PROVIDER env)
        priority: Fallback priority list (default: LLM_PROVIDER_PRIORITY or DEFAULT)
        timeout: Health check timeout per provider

    Returns:
        Tuple of (provider_name, health_details)

    Raises:
        RuntimeError: If no LLM providers are available
    """
    preferred = preferred_provider or os.getenv("LLM_PROVIDER", "").lower()

    # First, try the preferred provider
    if preferred:
        logger.info("Checking preferred LLM provider", provider=preferred)
        try:
            health = await check_provider_health(preferred, timeout=timeout)
            if health.get("connected", False):
                logger.info(
                    "Preferred LLM provider is available",
                    provider=preferred,
                    status=health.get("status"),
                )
                return preferred, health

            logger.warning(
                "Preferred LLM provider unavailable, will try fallbacks",
                provider=preferred,
                status=health.get("status"),
                error=health.get("error"),
            )
        except Exception as e:
            logger.warning(
                "Preferred LLM provider check failed, will try fallbacks",
                provider=preferred,
                error=str(e),
            )

    # Fallback to priority list
    provider, health = await get_first_available_provider(priority, timeout=timeout)

    if provider:
        if preferred and provider != preferred:
            logger.warning(
                "Using fallback LLM provider instead of preferred",
                preferred=preferred,
                fallback=provider,
            )
        return provider, health

    raise RuntimeError(
        f"No LLM providers available. Tried: {preferred or 'none'} (preferred), "
        f"then {priority or DEFAULT_LLM_PRIORITY} (fallback). "
        "Please ensure at least one LLM provider is running and configured."
    )


async def print_llm_availability_report(timeout: float = 5.0) -> dict[str, Any]:
    """Print a report of all LLM provider availability.

    Useful for debugging and startup diagnostics.

    Args:
        timeout: Health check timeout per provider

    Returns:
        Dictionary mapping provider names to health status
    """
    results = await check_all_providers_health(timeout=timeout)

    print("\n" + "=" * 60)
    print("LLM Provider Availability Report")
    print("=" * 60)

    available = []
    unavailable = []

    for provider, health in results.items():
        connected = health.get("connected", False)
        status = health.get("status", "unknown")
        error = health.get("error")

        icon = "✓" if connected else "✗"
        status_str = f"{icon} {provider.upper():12} | {status}"

        if connected:
            available.append(provider)
            if health.get("models"):
                status_str += f" | Models: {', '.join(health['models'][:3])}"
        else:
            unavailable.append(provider)
            if error:
                status_str += f" | {error[:50]}"

        print(f"  {status_str}")

    print("-" * 60)
    print(f"  Available: {', '.join(available) if available else 'None'}")
    print(f"  Configured priority: {', '.join(DEFAULT_LLM_PRIORITY)}")
    print("=" * 60 + "\n")

    return results


def print_startup_status(success: bool, message: str, details: dict[str, Any]) -> None:
    """Print formatted startup status for LLM provider.

    Args:
        success: Whether validation passed
        message: Status message
        details: Validation details from validate_llm_startup
    """
    provider = details.get("provider", "unknown")
    checks = details.get("checks", [])

    status_icon = "\u2713" if success else "\u2717"  # checkmark or X
    print(f"\n{'='*60}")
    print(f"LLM Provider Status: {provider.upper()}")
    print(f"{'='*60}")

    for check in checks:
        check_icon = "\u2713" if check.get("passed") else "\u2717"
        check_name = check.get("name", "unknown")
        if check.get("passed"):
            print(f"  {check_icon} {check_name}: OK")
        else:
            print(f"  {check_icon} {check_name}: {check.get('error', 'Failed')}")

    print(f"\n  {status_icon} Result: {message}")

    if details.get("suggestion"):
        print(f"\n  Suggestion: {details['suggestion']}")

    if details.get("health", {}).get("models"):
        models = details["health"]["models"][:5]  # Show first 5
        print(f"\n  Available models: {', '.join(models)}")

    print(f"{'='*60}\n")
