"""
Shared Memory Manager for OCI Coordinator.

Provides tiered storage architecture:
- Redis: Hot cache for session state, tool results (TTL: 1 hour)
- Neo4j: Persistent storage for conversation history, audit logs (optional)
- LangGraph: Graph state checkpoints (managed separately)
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MemoryStore(ABC):
    """Abstract base for memory storage backends."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(
        self, key: str, value: Any, ttl: timedelta | None = None
    ) -> None:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value by key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    async def close(self) -> None:
        """Close connection (override if needed)."""
        pass


class RedisMemoryStore(MemoryStore):
    """
    Redis-based hot cache for fast access.

    Used for:
    - Session state
    - Tool results cache
    - Agent status
    - Recent conversation context

    Default TTL: 1 hour
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis connection.

        Args:
            redis_url: Redis connection URL
        """
        import redis.asyncio as redis

        self.client = redis.from_url(redis_url, decode_responses=True)
        self._logger = logger.bind(store="redis")

    async def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except json.JSONDecodeError as e:
            self._logger.error("Failed to decode JSON", key=key, error=str(e))
            return None
        except Exception as e:
            self._logger.error("Redis get failed", key=key, error=str(e))
            return None

    async def set(
        self, key: str, value: Any, ttl: timedelta | None = None
    ) -> None:
        """Set value in Redis with optional TTL."""
        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                await self.client.setex(key, int(ttl.total_seconds()), serialized)
            else:
                await self.client.set(key, serialized)
        except Exception as e:
            self._logger.error("Redis set failed", key=key, error=str(e))
            raise

    async def delete(self, key: str) -> None:
        """Delete value from Redis."""
        try:
            await self.client.delete(key)
        except Exception as e:
            self._logger.error("Redis delete failed", key=key, error=str(e))
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            self._logger.error("Redis exists check failed", key=key, error=str(e))
            return False

    async def get_keys(self, pattern: str) -> list[str]:
        """Get all keys matching pattern."""
        try:
            return await self.client.keys(pattern)
        except Exception as e:
            self._logger.error("Redis keys failed", pattern=pattern, error=str(e))
            return []

    async def close(self) -> None:
        """Close Redis connection."""
        await self.client.close()


class Neo4jMemoryStore(MemoryStore):
    """
    Neo4j-backed persistent storage.

    Stores key/value entries as nodes for shared memory persistence.
    """

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._logger = logger.bind(store="neo4j")

    async def get(self, key: str) -> Any | None:
        """Get value from Neo4j."""
        query = (
            "MATCH (m:MemoryEntry {key: $key}) "
            "RETURN m.value AS value, m.expires_at AS expires_at"
        )
        try:
            async with self._driver.session() as session:
                result = await session.run(query, key=key)
                record = await result.single()
                if not record:
                    return None
                expires_at = record.get("expires_at")
                if expires_at and expires_at < time.time():
                    await session.run(
                        "MATCH (m:MemoryEntry {key: $key}) DELETE m", key=key
                    )
                    return None
                value = record.get("value")
                if value is None:
                    return None
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
        except Exception as e:
            self._logger.error("Neo4j get failed", key=key, error=str(e))
            return None

    async def set(
        self, key: str, value: Any, ttl: timedelta | None = None
    ) -> None:
        """Set value in Neo4j with optional TTL."""
        expires_at = None
        if ttl:
            expires_at = time.time() + ttl.total_seconds()
        payload = json.dumps(value, default=str)
        query = (
            "MERGE (m:MemoryEntry {key: $key}) "
            "SET m.value = $value, m.updated_at = timestamp(), m.expires_at = $expires_at"
        )
        try:
            async with self._driver.session() as session:
                await session.run(
                    query,
                    key=key,
                    value=payload,
                    expires_at=expires_at,
                )
        except Exception as e:
            self._logger.error("Neo4j set failed", key=key, error=str(e))
            raise

    async def delete(self, key: str) -> None:
        """Delete value from Neo4j."""
        try:
            async with self._driver.session() as session:
                await session.run("MATCH (m:MemoryEntry {key: $key}) DELETE m", key=key)
        except Exception as e:
            self._logger.error("Neo4j delete failed", key=key, error=str(e))
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists in Neo4j."""
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    "MATCH (m:MemoryEntry {key: $key}) RETURN m.key AS key",
                    key=key,
                )
                record = await result.single()
                return record is not None
        except Exception as e:
            self._logger.error("Neo4j exists failed", key=key, error=str(e))
            return False

    async def close(self) -> None:
        """Close Neo4j connection."""
        await self._driver.close()


class InMemoryStore(MemoryStore):
    """
    In-memory store for testing and development.

    Not persistent across restarts.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._logger = logger.bind(store="inmemory")

    async def get(self, key: str) -> Any | None:
        return self._data.get(key)

    async def set(
        self, key: str, value: Any, ttl: timedelta | None = None
    ) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()


class SharedMemoryManager:
    """
    Unified memory manager with tiered storage.

    Storage Tiers:
    - Cache (Redis): Fast access, TTL-based expiration
    - Persistent (Neo4j): Long-term storage, audit trails (optional)

    Key Patterns:
    - session:{session_id} - Session state
    - conversation:{thread_id} - Conversation history
    - agent:{agent_id}:{memory_type} - Agent-specific memory
    - tool_result:{hash} - Cached tool results

    Usage:
        memory = SharedMemoryManager(
            redis_url="redis://localhost:6379",
            neo4j_uri="bolt://localhost:7687"
        )

        # Session state (cache only)
        await memory.set_session_state(session_id, {"user_id": "..."})

        # Conversation (both cache and persistent)
        await memory.append_conversation(thread_id, message)

        # Agent memory (both cache and persistent)
        await memory.set_agent_memory(agent_id, "last_query", {...})
    """

    def __init__(
        self,
        redis_url: str | None = "redis://localhost:6379",
        neo4j_uri: str | None = None,
        use_in_memory: bool = False,
    ):
        """
        Initialize memory manager with storage backends.

        Args:
            redis_url: Redis connection URL (None to skip Redis)
            neo4j_uri: Neo4j URI (optional persistent backend)
            use_in_memory: Use in-memory store instead of Redis (for testing)
        """
        self._logger = logger.bind(component="SharedMemoryManager")

        # Initialize cache
        if use_in_memory:
            self.cache: MemoryStore = InMemoryStore()
            self._logger.info("Using in-memory cache")
        elif redis_url:
            self.cache = RedisMemoryStore(redis_url)
            self._logger.info("Using Redis cache", url=redis_url)
        else:
            self.cache = InMemoryStore()
            self._logger.warning("No cache configured, using in-memory fallback")

        # Initialize persistent store (Neo4j only)
        persistent_backend = os.getenv("MEMORY_PERSISTENT_BACKEND", "").lower()
        disable_persistent = persistent_backend in {"none", "disabled", "off", "false"}
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if persistent_backend == "neo4j":
            neo4j_uri = neo4j_uri or "bolt://localhost:7687"
            neo4j_user = neo4j_user or "neo4j"
            neo4j_password = neo4j_password or "neo4j"
        neo4j_configured = bool(neo4j_uri and neo4j_user and neo4j_password)

        if disable_persistent:
            self.persistent = None
            self._logger.info("Persistent storage disabled via MEMORY_PERSISTENT_BACKEND")
        elif neo4j_configured:
            self.persistent = Neo4jMemoryStore(neo4j_uri, neo4j_user, neo4j_password)
            self._logger.info("Using Neo4j persistent storage", uri=neo4j_uri)
        else:
            self.persistent = None
            self._logger.info("No persistent storage configured (cache-only mode)")

        # Default TTLs
        self.default_cache_ttl = timedelta(hours=1)
        self.session_ttl = timedelta(hours=4)
        self.tool_result_ttl = timedelta(minutes=15)

    async def close(self) -> None:
        """Close all connections."""
        await self.cache.close()
        if self.persistent:
            await self.persistent.close()

    async def _disable_persistent(self, operation: str, error: Exception) -> None:
        """Disable persistent storage after an error to prevent repeated failures."""
        backend = type(self.persistent).__name__ if self.persistent else "none"
        self._logger.warning(
            "Disabling persistent storage after error",
            operation=operation,
            backend=backend,
            error=str(error),
        )
        try:
            if self.persistent:
                await self.persistent.close()
        except Exception as close_err:
            self._logger.debug(
                "Failed to close persistent storage cleanly",
                backend=backend,
                error=str(close_err),
            )
        self.persistent = None

    async def ensure_persistent_schema(self) -> bool:
        """Ensure persistent storage schema exists.

        Returns:
            True if schema was initialized, False otherwise.
        """
        # No schema initialization needed for current backends
        # (Redis is schemaless, Neo4j creates nodes dynamically)
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Session State (Cache only - ephemeral)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Get session state from cache."""
        return await self.cache.get(f"session:{session_id}")

    async def set_session_state(
        self, session_id: str, state: dict[str, Any]
    ) -> None:
        """Set session state in cache."""
        await self.cache.set(f"session:{session_id}", state, self.session_ttl)

    async def update_session_state(
        self, session_id: str, updates: dict[str, Any]
    ) -> None:
        """Update session state (merge with existing)."""
        current = await self.get_session_state(session_id) or {}
        current.update(updates)
        await self.set_session_state(session_id, current)

    async def delete_session(self, session_id: str) -> None:
        """Delete session state."""
        await self.cache.delete(f"session:{session_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Conversation History (Cache + Persistent)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_conversation_history(
        self, thread_id: str
    ) -> list[dict[str, Any]] | None:
        """
        Get full conversation history.

        Tries cache first, falls back to persistent storage.
        """
        # Try cache first
        history = await self.cache.get(f"conversation:{thread_id}")

        if history is None and self.persistent:
            # Fall back to persistent
            history = await self.persistent.get(f"conversation:{thread_id}")
            if history:
                # Warm cache
                await self.cache.set(
                    f"conversation:{thread_id}", history, self.default_cache_ttl
                )

        return history

    async def append_conversation(
        self, thread_id: str, message: dict[str, Any]
    ) -> None:
        """Append message to conversation history."""
        history = await self.get_conversation_history(thread_id) or []
        history.append(message)

        # Update cache
        await self.cache.set(
            f"conversation:{thread_id}", history, self.default_cache_ttl
        )

        # Update persistent
        if self.persistent:
            try:
                await self.persistent.set(f"conversation:{thread_id}", history)
            except Exception as e:
                await self._disable_persistent("append_conversation", e)

    async def get_recent_messages(
        self, thread_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get last N messages from conversation."""
        history = await self.get_conversation_history(thread_id) or []
        return history[-limit:]

    # ─────────────────────────────────────────────────────────────────────────
    # Agent Memory (Cache + Persistent)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_agent_memory(
        self, agent_id: str, memory_type: str
    ) -> Any | None:
        """
        Get agent-specific memory.

        Args:
            agent_id: Agent identifier
            memory_type: Type of memory (e.g., "preferences", "learned_patterns")
        """
        key = f"agent:{agent_id}:{memory_type}"

        # Try cache first
        result = await self.cache.get(key)

        if result is None and self.persistent:
            # Fall back to persistent
            result = await self.persistent.get(key)
            if result:
                # Warm cache
                await self.cache.set(key, result, self.default_cache_ttl)

        return result

    async def set_agent_memory(
        self, agent_id: str, memory_type: str, value: Any
    ) -> None:
        """Set agent-specific memory."""
        key = f"agent:{agent_id}:{memory_type}"

        # Update cache
        await self.cache.set(key, value, self.default_cache_ttl)

        # Update persistent
        if self.persistent:
            try:
                await self.persistent.set(key, value)
            except Exception as e:
                await self._disable_persistent("set_agent_memory", e)

    async def delete_agent_memory(self, agent_id: str, memory_type: str) -> None:
        """Delete agent-specific memory."""
        key = f"agent:{agent_id}:{memory_type}"
        await self.cache.delete(key)
        if self.persistent:
            try:
                await self.persistent.delete(key)
            except Exception as e:
                await self._disable_persistent("delete_agent_memory", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Tool Result Cache (Cache only - short TTL)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_tool_result(
        self, tool_name: str, args_hash: str
    ) -> Any | None:
        """Get cached tool result."""
        return await self.cache.get(f"tool_result:{tool_name}:{args_hash}")

    async def set_tool_result(
        self,
        tool_name: str,
        args_hash: str,
        result: Any,
        ttl: timedelta | None = None,
    ) -> None:
        """Cache tool result with optional TTL override."""
        cache_ttl = ttl or self.tool_result_ttl
        await self.cache.set(
            f"tool_result:{tool_name}:{args_hash}",
            result,
            cache_ttl,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Feedback Loop (Cache + Persistent)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_feedback_entries(self, scope: str = "global") -> list[dict[str, Any]]:
        """Get feedback entries for a scope."""
        key = f"feedback:{scope}"
        data = await self.cache.get(key)

        if data is None and self.persistent:
            data = await self.persistent.get(key)
            if data is not None:
                await self.cache.set(key, data, self.default_cache_ttl)

        if isinstance(data, dict) and "entries" in data:
            return data["entries"]
        if isinstance(data, list):
            return data
        if isinstance(data, str):
            return [{"text": data, "source": "manual", "timestamp": time.time()}]
        return []

    async def get_feedback_text(self, scope: str = "global") -> str:
        """Get feedback text formatted for prompt injection."""
        entries = await self.get_feedback_entries(scope)
        if not entries:
            return ""
        lines = []
        for entry in entries[-10:]:
            text = entry.get("text", "")
            source = entry.get("source", "feedback")
            lines.append(f"- ({source}) {text}")
        return "\n".join(lines)

    async def set_feedback(
        self, scope: str, text: str, source: str = "operator"
    ) -> None:
        """Replace feedback for a scope."""
        payload = {
            "entries": [
                {"text": text, "source": source, "timestamp": time.time()}
            ],
            "updated_at": time.time(),
        }
        key = f"feedback:{scope}"
        await self.cache.set(key, payload, self.default_cache_ttl)
        if self.persistent:
            try:
                await self.persistent.set(key, payload)
            except Exception as e:
                await self._disable_persistent("set_feedback", e)

    async def append_feedback(
        self, scope: str, text: str, source: str = "operator"
    ) -> None:
        """Append a feedback entry to a scope."""
        entries = await self.get_feedback_entries(scope)
        entries.append({"text": text, "source": source, "timestamp": time.time()})
        payload = {"entries": entries, "updated_at": time.time()}
        key = f"feedback:{scope}"
        await self.cache.set(key, payload, self.default_cache_ttl)
        if self.persistent:
            try:
                await self.persistent.set(key, payload)
            except Exception as e:
                await self._disable_persistent("append_feedback", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Audit Logging (Persistent only)
    # ─────────────────────────────────────────────────────────────────────────

    async def log_audit(
        self,
        agent_id: str,
        action: str,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        """Log action to audit trail."""
        # Log audit events via structlog
        # (For persistent audit storage, integrate with OCI Logging or external system)
        self._logger.info(
            "Audit log",
            agent_id=agent_id,
            action=action,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message if status != "success" else None,
        )
