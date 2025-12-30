import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class OCAAuth:
    """Oracle Code Assist PKCE OAuth handler."""

    TOKEN_CACHE_FILE = Path.home() / ".oca_token_cache.json"

    def __init__(self, idcs_url: str, client_id: str, redirect_uri: str) -> None:
        """Initialize OCA auth."""
        self.idcs_url = idcs_url.rstrip("/")
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self._token: dict | None = None
        self._load_cached_token()

    def _load_cached_token(self) -> None:
        """Load token from cache file."""
        if self.TOKEN_CACHE_FILE.exists():
            try:
                with self.TOKEN_CACHE_FILE.open() as f:
                    self._token = json.load(f)
                    if self._is_token_expired():
                        self._token = None
            except Exception:
                pass

    def _save_token(self, token: dict) -> None:
        """Save token to cache file."""
        self._token = token
        try:
            with self.TOKEN_CACHE_FILE.open("w") as f:
                json.dump(token, f)
            self.TOKEN_CACHE_FILE.chmod(0o600)
        except Exception:
            pass

    def _is_token_expired(self) -> bool:
        """Check if current token is expired."""
        if not self._token:
            return True
        expires_at = self._token.get("expires_at", 0)
        return time.time() >= expires_at - 300  # 5 min buffer

    async def refresh_token(self) -> dict | None:
        """Refresh the access token."""
        if not self._token or "refresh_token" not in self._token:
            return None

        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self._token["refresh_token"],
                "client_id": self.client_id,
            }
            async with session.post(
                f"{self.idcs_url}/oauth2/v1/token",
                data=data,
            ) as resp:
                if resp.status == 200:
                    token = await resp.json()
                    token["expires_at"] = time.time() + token.get("expires_in", 3600)
                    self._save_token(token)
                    return token
                return None

    async def get_valid_token(self) -> str | None:
        """Get a valid access token, refreshing if needed."""
        if self._is_token_expired():
            await self.refresh_token()

        if self._token and "access_token" in self._token:
            return self._token["access_token"]
        return None


class ChatOCA(BaseChatModel):
    """Oracle Code Assist Chat Model."""

    api_url: str
    idcs_url: str
    client_id: str
    redirect_uri: str = "http://localhost:3001/api/oca/callback"
    model_name: str = "oci.generativeai.cohere.command-r-plus"
    temperature: float = 0.7
    max_tokens: int = 4096

    _auth: OCAAuth | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._auth = OCAAuth(self.idcs_url, self.client_id, self.redirect_uri)

    @property
    def _llm_type(self) -> str:
        return "oracle_code_assist"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """Internal generate method."""
        return asyncio.run(self._agenerate(messages, stop, None))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """Async generate method."""
        token = await self._auth.get_valid_token()
        if not token:
            raise RuntimeError("OCA authentication required")

        payload = {
            "model": self.model_name,
            "messages": [self._convert_message_to_dict(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            async with session.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"OCA error: {error}")

                data = await resp.json()
                choice = data.get("choices", [{}])[0]
                message_dict = choice.get("message", {})
                content = message_dict.get("content", "")

                ai_message = AIMessage(
                    content=content,
                    additional_kwargs={"usage": data.get("usage", {})},
                )

                return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _convert_message_to_dict(self, message: BaseMessage) -> dict[str, Any]:
        if isinstance(message, ChatMessage):
            return {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        else:
            raise ValueError(f"Unknown message type: {message}")
