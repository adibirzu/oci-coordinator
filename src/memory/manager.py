"""
Shared Memory Manager for OCI Coordinator.

Provides tiered storage architecture:
- Redis: Hot cache for session state, tool results (TTL: 1 hour)
- OCI ATP: Persistent storage for conversation history, audit logs
- LangGraph: Graph state checkpoints (managed separately)
"""

from __future__ import annotations

import json
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


class ATPMemoryStore(MemoryStore):
    """
    OCI ATP-based persistent storage for long-term memory.

    Used for:
    - Conversation history
    - Agent memory (learned patterns)
    - Audit logs
    - Agent registry

    Requires tables: agent_memory, conversation_history, agent_audit_log
    """

    def __init__(self, connection_string: str):
        """
        Initialize ATP connection pool.

        Args:
            connection_string: Oracle connection string
                Format: user/password@host:port/service_name
        """
        self._connection_string = connection_string
        self._pool = None
        self._logger = logger.bind(store="atp")

    async def _get_pool(self) -> Any:
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import oracledb

                self._pool = await oracledb.create_pool_async(
                    dsn=self._connection_string,
                    min=2,
                    max=10,
                )
                self._logger.info("ATP connection pool created")
            except Exception as e:
                self._logger.error("Failed to create ATP pool", error=str(e))
                raise
        return self._pool

    async def get(self, key: str) -> Any | None:
        """Get value from ATP agent_memory table."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "SELECT value FROM agent_memory WHERE key = :key",
                        {"key": key},
                    )
                    row = await cursor.fetchone()
                    if row:
                        return json.loads(row[0])
                    return None
        except Exception as e:
            self._logger.error("ATP get failed", key=key, error=str(e))
            return None

    async def set(
        self, key: str, value: Any, ttl: timedelta | None = None
    ) -> None:
        """Set value in ATP using MERGE (upsert)."""
        try:
            pool = await self._get_pool()
            serialized = json.dumps(value, default=str)
            ttl_seconds = int(ttl.total_seconds()) if ttl else None

            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        """
                        MERGE INTO agent_memory t
                        USING (SELECT :key as key FROM dual) s
                        ON (t.key = s.key)
                        WHEN MATCHED THEN
                            UPDATE SET value = :value, updated_at = SYSTIMESTAMP, ttl_seconds = :ttl
                        WHEN NOT MATCHED THEN
                            INSERT (key, value, ttl_seconds)
                            VALUES (:key, :value, :ttl)
                        """,
                        {"key": key, "value": serialized, "ttl": ttl_seconds},
                    )
                    await conn.commit()
        except Exception as e:
            self._logger.error("ATP set failed", key=key, error=str(e))
            raise

    async def delete(self, key: str) -> None:
        """Delete value from ATP."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "DELETE FROM agent_memory WHERE key = :key",
                        {"key": key},
                    )
                    await conn.commit()
        except Exception as e:
            self._logger.error("ATP delete failed", key=key, error=str(e))
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists in ATP."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "SELECT 1 FROM agent_memory WHERE key = :key",
                        {"key": key},
                    )
                    row = await cursor.fetchone()
                    return row is not None
        except Exception as e:
            self._logger.error("ATP exists check failed", key=key, error=str(e))
            return False

    async def append_audit_log(
        self,
        agent_id: str,
        action: str,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        """Append entry to audit log."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        """
                        INSERT INTO agent_audit_log
                        (agent_id, action, request, response, duration_ms, status, error_message)
                        VALUES (:agent_id, :action, :request, :response, :duration_ms, :status, :error_message)
                        """,
                        {
                            "agent_id": agent_id,
                            "action": action,
                            "request": json.dumps(request) if request else None,
                            "response": json.dumps(response) if response else None,
                            "duration_ms": duration_ms,
                            "status": status,
                            "error_message": error_message,
                        },
                    )
                    await conn.commit()
        except Exception as e:
            self._logger.error(
                "Failed to append audit log",
                agent_id=agent_id,
                action=action,
                error=str(e),
            )

    async def close(self) -> None:
        """Close ATP connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


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
    - Persistent (ATP): Long-term storage, audit trails

    Key Patterns:
    - session:{session_id} - Session state
    - conversation:{thread_id} - Conversation history
    - agent:{agent_id}:{memory_type} - Agent-specific memory
    - tool_result:{hash} - Cached tool results

    Usage:
        memory = SharedMemoryManager(
            redis_url="redis://localhost:6379",
            atp_connection="user/pass@host:1521/service"
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
        atp_connection: str | None = None,
        use_in_memory: bool = False,
    ):
        """
        Initialize memory manager with storage backends.

        Args:
            redis_url: Redis connection URL (None to skip Redis)
            atp_connection: OCI ATP connection string (None for cache-only)
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

        # Initialize persistent store
        if atp_connection:
            self.persistent: MemoryStore | None = ATPMemoryStore(atp_connection)
            self._logger.info("Using ATP persistent storage")
        else:
            self.persistent = None
            self._logger.warning("No persistent storage configured")

        # Default TTLs
        self.default_cache_ttl = timedelta(hours=1)
        self.session_ttl = timedelta(hours=4)
        self.tool_result_ttl = timedelta(minutes=15)

    async def close(self) -> None:
        """Close all connections."""
        await self.cache.close()
        if self.persistent:
            await self.persistent.close()

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
            await self.persistent.set(f"conversation:{thread_id}", history)

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
            await self.persistent.set(key, value)

    async def delete_agent_memory(self, agent_id: str, memory_type: str) -> None:
        """Delete agent-specific memory."""
        key = f"agent:{agent_id}:{memory_type}"
        await self.cache.delete(key)
        if self.persistent:
            await self.persistent.delete(key)

    # ─────────────────────────────────────────────────────────────────────────
    # Tool Result Cache (Cache only - short TTL)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_tool_result(
        self, tool_name: str, args_hash: str
    ) -> Any | None:
        """Get cached tool result."""
        return await self.cache.get(f"tool_result:{tool_name}:{args_hash}")

    async def set_tool_result(
        self, tool_name: str, args_hash: str, result: Any
    ) -> None:
        """Cache tool result."""
        await self.cache.set(
            f"tool_result:{tool_name}:{args_hash}",
            result,
            self.tool_result_ttl,
        )

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
        if self.persistent and isinstance(self.persistent, ATPMemoryStore):
            await self.persistent.append_audit_log(
                agent_id=agent_id,
                action=action,
                request=request,
                response=response,
                duration_ms=duration_ms,
                status=status,
                error_message=error_message,
            )
        else:
            # Log to structlog if no persistent storage
            self._logger.info(
                "Audit log",
                agent_id=agent_id,
                action=action,
                status=status,
                duration_ms=duration_ms,
            )
