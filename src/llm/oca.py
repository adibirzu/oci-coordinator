"""
Oracle Code Assist (OCA) LLM Provider.

Implements OCA integration via LiteLLM endpoint with PKCE OAuth authentication.
Token caching and refresh is handled automatically.

Available Models (as of 2026-01):
    - oca/gpt-4.1 (default, supports vision)
    - oca/gpt-oss-120b (Oracle's 120B model)
    - oca/llama4
    - oca/openai-o3

Usage:
    from src.llm.oca import ChatOCA, get_oca_llm

    # Create LLM instance
    llm = get_oca_llm(model="oca/gpt-4.1")

    # Or use directly
    llm = ChatOCA(model="oca/gpt-4.1")
    response = await llm.ainvoke([HumanMessage(content="Hello!")])
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import structlog
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

logger = structlog.get_logger(__name__)


# OCA Configuration
class OCAConfig:
    """OCA OAuth and API configuration."""

    # IDCS OAuth endpoints
    IDCS_CLIENT_ID = "a8331954c0cf48ba99b5dd223a14c6ea"
    IDCS_OAUTH_URL = "https://idcs-9dc693e80d9b469480d7afe00e743931.identity.oraclecloud.com"

    @property
    def token_endpoint(self) -> str:
        return f"{self.IDCS_OAUTH_URL}/oauth2/v1/token"

    @property
    def authz_endpoint(self) -> str:
        return f"{self.IDCS_OAUTH_URL}/oauth2/v1/authorize"

    # OCA LiteLLM endpoint
    OCA_ENDPOINT = os.getenv(
        "OCA_ENDPOINT",
        "https://code-internal.aiservice.us-chicago-1.oci.oraclecloud.com",
    )
    OCA_API_VERSION = "20250206"

    @property
    def litellm_url(self) -> str:
        return f"{self.OCA_ENDPOINT}/{self.OCA_API_VERSION}/app/litellm"

    # Token cache
    CACHE_DIR = Path(os.getenv("OCA_CACHE_DIR", Path.home() / ".oca"))

    @property
    def token_cache_path(self) -> Path:
        return self.CACHE_DIR / "token.json"

    # Token refresh buffer (3 minutes)
    RENEW_BUFFER_SEC = 3 * 60

    # Default model - use oca/gpt-4.1 (most capable, supports vision)
    # Available models: oca/gpt-4.1, oca/gpt-oss-120b, oca/llama4, oca/openai-o3
    DEFAULT_MODEL = os.getenv("OCA_MODEL", "oca/gpt-4.1")

    # Fallback models (in order of preference)
    FALLBACK_MODELS = ["oca/gpt-4.1", "oca/gpt-oss-120b", "oca/llama4", "oca/openai-o3"]


OCA_CONFIG = OCAConfig()


class OCATokenManager:
    """Manages OCA access tokens with caching and refresh."""

    def __init__(self):
        self._cached_token: dict | None = None
        self._token_expires_at: float = 0
        self._refresh_expires_at: float = 0
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        OCA_CONFIG.CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _load_from_file(self) -> bool:
        """Load cached token from file."""
        try:
            if OCA_CONFIG.token_cache_path.exists():
                data = json.loads(OCA_CONFIG.token_cache_path.read_text())
                self._cached_token = data
                self._token_expires_at = data.get("_expires_at", 0)
                self._refresh_expires_at = data.get("_refresh_expires_at", 0)
                logger.debug("Loaded OCA token from cache")
                return True
        except Exception as e:
            logger.warning("Failed to load OCA token cache", error=str(e))
        return False

    def _save_to_file(self, token: dict) -> None:
        """Save token to cache file."""
        try:
            self._ensure_cache_dir()
            OCA_CONFIG.token_cache_path.write_text(json.dumps(token, indent=2))
            OCA_CONFIG.token_cache_path.chmod(0o600)
            logger.debug("Saved OCA token to cache")
        except Exception as e:
            logger.error("Failed to save OCA token", error=str(e))

    def cache_token(self, token: dict) -> None:
        """Cache a new token."""
        import time

        now = time.time()
        token["_cached_at"] = now
        token["_expires_at"] = now + token.get("expires_in", 3600)
        token["_refresh_expires_at"] = now + 8 * 60 * 60  # 8 hours

        self._cached_token = token
        self._token_expires_at = token["_expires_at"]
        self._refresh_expires_at = token["_refresh_expires_at"]
        self._save_to_file(token)
        logger.info("OCA token cached successfully")

    def has_valid_token(self) -> bool:
        """Check if we have a valid access token."""
        import time

        if not self._cached_token:
            if not self._load_from_file():
                return False

        now = time.time()
        if now >= self._token_expires_at - OCA_CONFIG.RENEW_BUFFER_SEC:
            logger.debug("OCA token expired or expiring soon")
            return False
        return True

    def can_refresh(self) -> bool:
        """Check if we can refresh the token."""
        import time

        if not self._cached_token:
            if not self._load_from_file():
                return False

        if not self._cached_token.get("refresh_token"):
            return False

        now = time.time()
        if now >= self._refresh_expires_at:
            logger.debug("OCA refresh token expired")
            return False
        return True

    def get_access_token(self) -> str | None:
        """Get valid access token or None."""
        if self.has_valid_token():
            return self._cached_token["access_token"]
        return None

    async def refresh_token(self) -> str | None:
        """Refresh the access token."""
        if not self.can_refresh():
            logger.warning("Cannot refresh OCA token")
            return None

        try:
            logger.info("Refreshing OCA access token...")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OCA_CONFIG.token_endpoint,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self._cached_token["refresh_token"],
                        "client_id": OCA_CONFIG.IDCS_CLIENT_ID,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code != 200:
                    raise Exception(f"Token refresh failed: {response.status_code}")

                new_token = response.json()
                self.cache_token(new_token)
                logger.info("OCA token refreshed successfully")
                return self._cached_token["access_token"]

        except Exception as e:
            logger.error("OCA token refresh failed", error=str(e))
            self.clear_token()
            return None

    def clear_token(self) -> None:
        """Clear cached token."""
        self._cached_token = None
        self._token_expires_at = 0
        self._refresh_expires_at = 0
        if OCA_CONFIG.token_cache_path.exists():
            OCA_CONFIG.token_cache_path.unlink()
        logger.info("OCA token cleared")

    def force_reload_from_disk(self) -> bool:
        """Force reload token from disk, clearing in-memory cache first.

        This is useful when the user has re-authenticated in the browser
        and the new token was saved to disk while the bot is still running.

        Returns:
            True if a valid token was loaded from disk.
        """
        # Clear in-memory cache to force disk reload
        self._cached_token = None
        self._token_expires_at = 0
        self._refresh_expires_at = 0

        # Try to load from disk
        if self._load_from_file():
            logger.info(
                "Reloaded OCA token from disk",
                has_token=self._cached_token is not None,
                is_valid=self.has_valid_token(),
            )
            return self.has_valid_token()

        logger.debug("No token file found on disk to reload")
        return False

    def get_token_info(self) -> dict:
        """Get token status information."""
        import time

        if not self._cached_token:
            self._load_from_file()

        now = time.time()
        return {
            "has_token": self._cached_token is not None,
            "is_valid": self.has_valid_token(),
            "can_refresh": self.can_refresh(),
            "expires_in_seconds": max(0, self._token_expires_at - now),
            "refresh_expires_in_seconds": max(0, self._refresh_expires_at - now),
            "cached_at": self._cached_token.get("_cached_at") if self._cached_token else None,
        }

    async def verify_token_with_endpoint(self) -> tuple[bool, str]:
        """Verify token is valid by making a test API call.

        Returns:
            Tuple of (is_valid, message)
        """
        token = self.get_access_token()
        if not token:
            return False, "No valid access token available"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try to list models as a lightweight health check
                response = await client.get(
                    f"{OCA_CONFIG.litellm_url}/models",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "client": "oci-coordinator",
                    },
                )

                if response.status_code == 200:
                    return True, "Token verified successfully"
                elif response.status_code == 401:
                    self.clear_token()
                    return False, "Token rejected by server (401 Unauthorized)"
                elif response.status_code == 403:
                    return False, "Token lacks required permissions (403 Forbidden)"
                else:
                    return False, f"Unexpected status: {response.status_code}"

        except httpx.ConnectError as e:
            return False, f"Connection failed: {e}"
        except Exception as e:
            return False, f"Verification error: {type(e).__name__}: {e}"


# Global token manager
oca_token_manager = OCATokenManager()


def is_oca_authenticated() -> bool:
    """Check if OCA authentication is valid.

    This function checks:
    1. In-memory cached token
    2. Token on disk (in case user just authenticated)
    """
    # Check in-memory first
    if oca_token_manager.has_valid_token() or oca_token_manager.can_refresh():
        return True

    # Try reloading from disk (user may have just authenticated)
    if oca_token_manager.force_reload_from_disk():
        return True

    return False


class ChatOCA(BaseChatModel):
    """LangChain Chat Model for Oracle Code Assist."""

    model: str = Field(default=OCA_CONFIG.DEFAULT_MODEL)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)
    client_name: str = Field(default="oci-coordinator")
    client_version: str = Field(default="1.0")
    timeout: float = Field(default=300.0)  # 5 minutes for long responses

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "oca"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    async def _get_access_token(self) -> str:
        """Get valid access token, refreshing or reloading from disk if needed.

        This method tries multiple strategies to get a valid token:
        1. Use the in-memory cached token if valid
        2. Try to refresh the token using the refresh_token
        3. Force reload from disk (in case user re-authenticated in browser)

        Raises:
            ValueError: If no valid token is available after all attempts.
        """
        # Strategy 1: Use cached token
        token = oca_token_manager.get_access_token()
        if token:
            return token

        # Strategy 2: Try to refresh
        token = await oca_token_manager.refresh_token()
        if token:
            return token

        # Strategy 3: Reload from disk (user may have re-authenticated)
        logger.info("Attempting to reload OCA token from disk...")
        if oca_token_manager.force_reload_from_disk():
            token = oca_token_manager.get_access_token()
            if token:
                logger.info("Successfully loaded new token from disk after re-authentication")
                return token

        raise ValueError(
            "Oracle Code Assist requires authentication. "
            "Please complete the OAuth login flow first."
        )

    def _convert_messages(
        self, messages: list[BaseMessage]
    ) -> list[dict[str, Any]]:
        """Convert LangChain messages to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            else:
                # Default to user message
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _parse_sse_response(self, response_text: str) -> tuple[str, dict]:
        """Parse SSE streaming response from OCA."""
        content_parts = []
        token_usage = {}

        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])

                    # Extract token usage
                    if "usage" in chunk:
                        token_usage = {
                            "prompt_tokens": chunk["usage"].get("prompt_tokens"),
                            "completion_tokens": chunk["usage"].get("completion_tokens"),
                            "total_tokens": chunk["usage"].get("total_tokens"),
                        }

                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]

                        # Streaming delta
                        if "delta" in choice and "content" in choice["delta"]:
                            content_parts.append(choice["delta"]["content"])

                        # Non-streaming message
                        if "message" in choice and "content" in choice["message"]:
                            content_parts.append(choice["message"]["content"])

                except json.JSONDecodeError:
                    continue

        return "".join(content_parts), token_usage

    def _parse_json_response(self, response_text: str) -> tuple[str, dict] | None:
        """Try to parse as standard JSON response."""
        try:
            data = json.loads(response_text)
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                token_usage = {}
                if "usage" in data:
                    token_usage = {
                        "prompt_tokens": data["usage"].get("prompt_tokens"),
                        "completion_tokens": data["usage"].get("completion_tokens"),
                        "total_tokens": data["usage"].get("total_tokens"),
                    }
                return content, token_usage
        except json.JSONDecodeError:
            pass
        return None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation - wraps async."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop, run_manager, **kwargs)
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion from OCA with full GenAI semantic conventions.

        Uses OracleCodeAssistInstrumentor for comprehensive tracing compatible
        with OCI APM, viewapp LLM Observability, and DataDog.
        """
        # Import the enhanced instrumentor
        from src.observability import OracleCodeAssistInstrumentor

        # Use the OracleCodeAssistInstrumentor for comprehensive GenAI tracing
        with OracleCodeAssistInstrumentor.chat_span(
            model=self.model,
            endpoint=OCA_CONFIG.OCA_ENDPOINT,
        ) as llm_ctx:
            # Set request parameters for observability
            llm_ctx.set_request_params(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Record the prompt (last user message for context)
            if messages:
                # Combine all messages for prompt context
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        llm_ctx.set_prompt(str(msg.content), role="system")
                    elif isinstance(msg, HumanMessage):
                        llm_ctx.set_prompt(str(msg.content), role="user")
                    elif isinstance(msg, AIMessage):
                        llm_ctx.set_prompt(str(msg.content), role="assistant")

            # Get trace context for logging
            trace_id = ""
            try:
                from opentelemetry import trace
                current_span = trace.get_current_span()
                if current_span:
                    ctx = current_span.get_span_context()
                    trace_id = format(ctx.trace_id, "032x")
            except Exception:
                pass

            try:
                access_token = await self._get_access_token()

                converted_messages = self._convert_messages(messages)

                payload = {
                    "model": self.model,
                    "messages": converted_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }

                if stop:
                    payload["stop"] = stop

                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "client": self.client_name,
                    "client-version": self.client_version,
                }

                logger.info("Sending OCA request", model=self.model)

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{OCA_CONFIG.litellm_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                    logger.debug("OCA response status", status=response.status_code)

                    if response.status_code == 401:
                        oca_token_manager.clear_token()
                        llm_ctx.set_error(
                            ValueError("OCA authentication expired"),
                            error_type="auth_expired",
                        )
                        raise ValueError("OCA authentication expired. Please log in again.")

                    if response.status_code != 200:
                        llm_ctx.set_error(
                            ValueError(f"OCA API error: {response.status_code}"),
                            error_type="api_error",
                        )
                        raise ValueError(f"OCA API error: {response.status_code} - {response.text[:500]}")

                    response_text = response.text

                    # Try JSON first, then SSE
                    parsed = self._parse_json_response(response_text)
                    if parsed is None:
                        parsed = self._parse_sse_response(response_text)
                    else:
                        content, token_usage = parsed

                    content, token_usage = parsed if parsed else ("", {})

                    # Record completion content (for viewapp LLM observability)
                    llm_ctx.set_completion(content)
                    llm_ctx.set_response_model(self.model)
                    llm_ctx.set_finish_reason("stop")

                    # Record token usage using GenAI semantic conventions
                    if token_usage:
                        prompt_tokens = token_usage.get("prompt_tokens", 0)
                        completion_tokens = token_usage.get("completion_tokens", 0)

                        llm_ctx.set_tokens(
                            input=prompt_tokens,
                            output=completion_tokens,
                        )

                        # Estimate cost
                        from src.observability import LLMInstrumentor
                        cost = LLMInstrumentor.estimate_cost(
                            self.model, prompt_tokens, completion_tokens
                        )
                        llm_ctx.set_cost_estimate(cost)

                    logger.info(
                        "OCA response received",
                        chars=len(content),
                        tokens=token_usage.get("total_tokens"),
                        trace_id=trace_id,
                    )

                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(content=content),
                                generation_info=token_usage,
                            )
                        ],
                        llm_output={
                            "model": self.model,
                            "token_usage": token_usage,
                        },
                    )

            except Exception as e:
                # Record error with GenAI semantic conventions
                error_type = "unknown"
                if "authentication" in str(e).lower():
                    error_type = "auth_error"
                elif "timeout" in str(e).lower():
                    error_type = "timeout"
                elif "connection" in str(e).lower():
                    error_type = "connection_error"
                elif "rate" in str(e).lower():
                    error_type = "rate_limit"

                llm_ctx.set_error(e, error_type=error_type)

                logger.error(
                    "OCA LLM call failed",
                    model=self.model,
                    error=str(e)[:200],
                    error_type=error_type,
                    trace_id=trace_id,
                )
                raise


def get_oca_llm(
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> ChatOCA:
    """Create an OCA LLM instance.

    Args:
        model: Model to use (default: oca/gpt-4.1). Available models:
               oca/gpt-4.1, oca/gpt-oss-120b, oca/llama4, oca/openai-o3
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        ChatOCA instance
    """
    if not is_oca_authenticated():
        logger.warning("OCA not authenticated - requests will fail until login")

    return ChatOCA(
        model=model or OCA_CONFIG.DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def get_oca_health() -> dict:
    """Get comprehensive OCA health status.

    Returns:
        Dictionary with health status including:
        - provider: Provider name ('oca' or 'oracle_code_assist')
        - connected: Whether the provider is available
        - authentication: Token status
        - endpoint: API connectivity
        - models: Available models (if authenticated)
        - trace_context: Current trace info for debugging
    """
    from opentelemetry import trace

    health = {
        "provider": "oracle_code_assist",
        "status": "unknown",
        "connected": False,  # Will be set based on checks
        "authentication": {},
        "endpoint": {},
        "models": [],
        "trace_context": {},
        "config": {
            "endpoint": OCA_CONFIG.OCA_ENDPOINT,
            "model": OCA_CONFIG.DEFAULT_MODEL,
            "api_version": OCA_CONFIG.OCA_API_VERSION,
        },
    }

    # Get trace context for debugging
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        ctx = current_span.get_span_context()
        health["trace_context"] = {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "is_recording": True,
        }
    else:
        health["trace_context"] = {"is_recording": False}

    # Check authentication
    token_info = oca_token_manager.get_token_info()
    health["authentication"] = {
        "has_token": token_info["has_token"],
        "is_valid": token_info["is_valid"],
        "can_refresh": token_info["can_refresh"],
        "expires_in_minutes": round(token_info["expires_in_seconds"] / 60, 1),
    }

    # Add helpful error message if not authenticated
    if not token_info["has_token"]:
        health["error"] = "OCA not authenticated. Run 'oca-login' or authenticate via browser."
    elif not token_info["is_valid"] and not token_info["can_refresh"]:
        health["error"] = "OCA token expired and cannot be refreshed. Please re-authenticate."

    # Verify with endpoint if we have a token
    if token_info["is_valid"]:
        verified, message = await oca_token_manager.verify_token_with_endpoint()
        health["authentication"]["verified"] = verified
        health["authentication"]["verify_message"] = message

    # Check endpoint connectivity and get models
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            token = oca_token_manager.get_access_token()
            headers = {"client": "oci-coordinator"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            response = await client.get(
                f"{OCA_CONFIG.litellm_url}/models",
                headers=headers,
            )

            health["endpoint"] = {
                "reachable": True,
                "status_code": response.status_code,
            }

            if response.status_code == 200:
                try:
                    models_data = response.json()
                    if "data" in models_data:
                        health["models"] = [m.get("id") for m in models_data["data"]]
                except Exception:
                    pass

    except httpx.ConnectError as e:
        health["endpoint"] = {
            "reachable": False,
            "error": f"Connection failed: {e}",
        }
    except Exception as e:
        health["endpoint"] = {
            "reachable": False,
            "error": f"{type(e).__name__}: {e}",
        }

    # Determine overall status
    if (
        health["authentication"].get("verified", False)
        and health["endpoint"].get("reachable", False)
    ):
        health["status"] = "healthy"
        health["connected"] = True
    elif health["endpoint"].get("reachable", False):
        health["status"] = "degraded"
        # Still connected but auth may be expired
        health["connected"] = health["authentication"].get("is_valid", False)
    else:
        health["status"] = "unhealthy"
        health["connected"] = False

    return health


async def verify_oca_integration() -> tuple[bool, str, dict]:
    """Verify complete OCA integration including authentication and tracing.

    This is a comprehensive check that:
    1. Verifies token is valid
    2. Tests endpoint connectivity
    3. Makes a minimal LLM call (if authenticated)
    4. Verifies tracing is working

    Returns:
        Tuple of (success, message, details)
    """
    from opentelemetry.trace import SpanKind

    details = {
        "checks": [],
        "trace_id": None,
    }

    try:
        from src.observability import get_tracer, is_otel_enabled

        tracer = get_tracer("coordinator")

        with tracer.start_as_current_span(
            "oca.integration_verification", kind=SpanKind.INTERNAL
        ) as span:
            ctx = span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")
            details["trace_id"] = trace_id

            span.set_attribute("verification.type", "oca_integration")
            span.set_attribute("verification.otel_enabled", is_otel_enabled())

            # Check 1: Authentication
            auth_ok = is_oca_authenticated()
            details["checks"].append({
                "name": "authentication",
                "passed": auth_ok,
            })
            span.set_attribute("verification.auth_ok", auth_ok)

            if not auth_ok:
                span.set_attribute("verification.result", "auth_failed")
                return False, "OCA authentication invalid", details

            # Check 2: Token verification with endpoint
            verified, verify_msg = await oca_token_manager.verify_token_with_endpoint()
            details["checks"].append({
                "name": "token_verification",
                "passed": verified,
                "message": verify_msg,
            })
            span.set_attribute("verification.token_verified", verified)

            if not verified:
                span.set_attribute("verification.result", "token_rejected")
                return False, f"Token verification failed: {verify_msg}", details

            # Check 3: Minimal LLM call
            try:
                from langchain_core.messages import HumanMessage

                llm = ChatOCA(model="oca/gpt-4.1", max_tokens=10, temperature=0)
                response = await llm.ainvoke([HumanMessage(content="Say 'ok'")])

                llm_ok = len(response.content) > 0
                details["checks"].append({
                    "name": "llm_inference",
                    "passed": llm_ok,
                    "response_length": len(response.content),
                })
                span.set_attribute("verification.llm_ok", llm_ok)

            except Exception as e:
                details["checks"].append({
                    "name": "llm_inference",
                    "passed": False,
                    "error": str(e),
                })
                span.set_attribute("verification.llm_ok", False)
                span.set_attribute("verification.llm_error", str(e)[:100])
                span.record_exception(e)
                return False, f"LLM inference failed: {e}", details

            # Check 4: Tracing
            tracing_ok = is_otel_enabled() and span.is_recording()
            details["checks"].append({
                "name": "tracing",
                "passed": tracing_ok,
                "otel_enabled": is_otel_enabled(),
                "span_recording": span.is_recording(),
            })
            span.set_attribute("verification.tracing_ok", tracing_ok)

            # All checks passed
            span.set_attribute("verification.result", "success")
            span.set_attribute("verification.all_checks_passed", True)

            return True, "OCA integration verified successfully", details

    except Exception as e:
        logger.error("OCA verification failed", error=str(e))
        details["checks"].append({
            "name": "verification_error",
            "passed": False,
            "error": str(e),
        })
        return False, f"Verification error: {e}", details
