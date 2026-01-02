"""
Dead Letter Queue for Failed Operations.

Persists failed operations to Redis for analysis and retry.
Provides failure pattern tracking and alerting capabilities.

Features:
- Redis-backed persistent storage
- Failure categorization and pattern tracking
- Configurable retention and max queue size
- Async-safe with connection pooling
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


class FailureType(str, Enum):
    """Categories of failures for analysis."""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    MCP_CONNECTION = "mcp_connection"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class FailedOperation:
    """A failed operation stored in the dead letter queue."""

    id: str
    failure_type: FailureType
    operation: str
    error: str
    params: dict[str, Any]
    context: dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    last_retry: datetime | None = None
    resolved: bool = False
    resolution_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "failure_type": self.failure_type.value,
            "operation": self.operation,
            "error": self.error,
            "params": self.params,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "last_retry": self.last_retry.isoformat() if self.last_retry else None,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailedOperation:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            failure_type=FailureType(data["failure_type"]),
            operation=data["operation"],
            error=data["error"],
            params=data.get("params", {}),
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            retry_count=data.get("retry_count", 0),
            last_retry=datetime.fromisoformat(data["last_retry"]) if data.get("last_retry") else None,
            resolved=data.get("resolved", False),
            resolution_notes=data.get("resolution_notes"),
        )


@dataclass
class FailureStats:
    """Statistics about failures in the queue."""

    total_failures: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_operation: dict[str, int] = field(default_factory=dict)
    resolved_count: int = 0
    pending_count: int = 0
    avg_retry_count: float = 0.0
    oldest_failure: datetime | None = None
    newest_failure: datetime | None = None


class DeadLetterQueue:
    """
    Redis-backed dead letter queue for failed operations.

    Stores failed operations with metadata for analysis and retry.
    Supports failure pattern tracking and alerting.

    Example:
        dlq = DeadLetterQueue(redis_url="redis://localhost:6379")

        # Enqueue a failed operation
        await dlq.enqueue(
            failure_type=FailureType.TOOL_CALL,
            operation="oci_cost_get_summary",
            error="Timeout after 60s",
            params={"days": 30},
            context={"user_id": "U123", "thread_id": "T456"},
        )

        # Get pending failures
        failures = await dlq.get_pending(limit=10)

        # Retry a failed operation
        await dlq.mark_retrying(failure_id)

        # Mark as resolved
        await dlq.resolve(failure_id, "Fixed by increasing timeout")
    """

    QUEUE_KEY = "oci:deadletter:queue"
    STATS_KEY = "oci:deadletter:stats"
    MAX_QUEUE_SIZE = 10000
    RETENTION_DAYS = 30

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_size: int = MAX_QUEUE_SIZE,
        retention_days: int = RETENTION_DAYS,
    ):
        """Initialize dead letter queue.

        Args:
            redis_url: Redis connection URL
            max_size: Maximum queue size before eviction
            retention_days: Days to retain resolved failures
        """
        self._redis_url = redis_url
        self._max_size = max_size
        self._retention_days = retention_days
        self._redis: Any = None
        self._logger = logger.bind(component="DeadLetterQueue")

    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except ImportError:
                self._logger.warning("Redis not available, using in-memory fallback")
                self._redis = InMemoryFallback()
        return self._redis

    async def enqueue(
        self,
        failure_type: FailureType | str,
        operation: str,
        error: str,
        params: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> FailedOperation:
        """
        Add a failed operation to the queue.

        Args:
            failure_type: Category of failure
            operation: Name of the failed operation
            error: Error message or details
            params: Parameters passed to the operation
            context: Additional context (user_id, thread_id, etc.)

        Returns:
            The created FailedOperation
        """
        if isinstance(failure_type, str):
            failure_type = FailureType(failure_type)

        failed_op = FailedOperation(
            id=str(uuid4()),
            failure_type=failure_type,
            operation=operation,
            error=error,
            params=params or {},
            context=context or {},
            timestamp=datetime.utcnow(),
        )

        redis = await self._get_redis()

        # Store in sorted set by timestamp
        score = time.time()
        await redis.zadd(
            self.QUEUE_KEY,
            {json.dumps(failed_op.to_dict()): score},
        )

        # Update stats
        await self._update_stats(failure_type, operation)

        # Trim queue if needed
        await self._trim_queue()

        self._logger.info(
            "Failed operation enqueued",
            id=failed_op.id,
            type=failure_type.value,
            operation=operation,
        )

        return failed_op

    async def _update_stats(self, failure_type: FailureType, operation: str) -> None:
        """Update failure statistics."""
        redis = await self._get_redis()
        await redis.hincrby(self.STATS_KEY, "total", 1)
        await redis.hincrby(self.STATS_KEY, f"type:{failure_type.value}", 1)
        await redis.hincrby(self.STATS_KEY, f"op:{operation}", 1)

    async def _trim_queue(self) -> None:
        """Trim queue to max size, removing oldest resolved items first."""
        redis = await self._get_redis()
        size = await redis.zcard(self.QUEUE_KEY)

        if size > self._max_size:
            # Remove oldest entries beyond max size
            to_remove = size - self._max_size
            await redis.zremrangebyrank(self.QUEUE_KEY, 0, to_remove - 1)
            self._logger.warning(
                "Dead letter queue trimmed",
                removed=to_remove,
                max_size=self._max_size,
            )

    async def get_pending(
        self,
        limit: int = 100,
        failure_type: FailureType | None = None,
        operation: str | None = None,
    ) -> list[FailedOperation]:
        """
        Get pending (unresolved) failed operations.

        Args:
            limit: Maximum number to return
            failure_type: Filter by failure type
            operation: Filter by operation name

        Returns:
            List of pending FailedOperation objects
        """
        redis = await self._get_redis()
        raw_items = await redis.zrange(self.QUEUE_KEY, 0, -1)

        results = []
        for item in raw_items:
            try:
                data = json.loads(item)
                failed_op = FailedOperation.from_dict(data)

                if failed_op.resolved:
                    continue

                if failure_type and failed_op.failure_type != failure_type:
                    continue

                if operation and failed_op.operation != operation:
                    continue

                results.append(failed_op)

                if len(results) >= limit:
                    break

            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning("Invalid queue entry", error=str(e))

        return results

    async def get_by_id(self, failure_id: str) -> FailedOperation | None:
        """Get a specific failed operation by ID."""
        redis = await self._get_redis()
        raw_items = await redis.zrange(self.QUEUE_KEY, 0, -1)

        for item in raw_items:
            try:
                data = json.loads(item)
                if data.get("id") == failure_id:
                    return FailedOperation.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    async def mark_retrying(self, failure_id: str) -> bool:
        """
        Mark a failed operation as being retried.

        Args:
            failure_id: ID of the failed operation

        Returns:
            True if operation was found and updated
        """
        redis = await self._get_redis()
        raw_items = await redis.zrange(self.QUEUE_KEY, 0, -1, withscores=True)

        for item, score in raw_items:
            try:
                data = json.loads(item)
                if data.get("id") == failure_id:
                    data["retry_count"] = data.get("retry_count", 0) + 1
                    data["last_retry"] = datetime.utcnow().isoformat()

                    # Remove old entry and add updated one
                    await redis.zrem(self.QUEUE_KEY, item)
                    await redis.zadd(
                        self.QUEUE_KEY,
                        {json.dumps(data): score},
                    )

                    self._logger.info(
                        "Failed operation marked for retry",
                        id=failure_id,
                        retry_count=data["retry_count"],
                    )
                    return True

            except (json.JSONDecodeError, KeyError):
                continue

        return False

    async def resolve(
        self,
        failure_id: str,
        resolution_notes: str | None = None,
    ) -> bool:
        """
        Mark a failed operation as resolved.

        Args:
            failure_id: ID of the failed operation
            resolution_notes: Optional notes about the resolution

        Returns:
            True if operation was found and resolved
        """
        redis = await self._get_redis()
        raw_items = await redis.zrange(self.QUEUE_KEY, 0, -1, withscores=True)

        for item, score in raw_items:
            try:
                data = json.loads(item)
                if data.get("id") == failure_id:
                    data["resolved"] = True
                    data["resolution_notes"] = resolution_notes

                    # Remove old entry and add updated one
                    await redis.zrem(self.QUEUE_KEY, item)
                    await redis.zadd(
                        self.QUEUE_KEY,
                        {json.dumps(data): score},
                    )

                    await redis.hincrby(self.STATS_KEY, "resolved", 1)

                    self._logger.info(
                        "Failed operation resolved",
                        id=failure_id,
                        notes=resolution_notes,
                    )
                    return True

            except (json.JSONDecodeError, KeyError):
                continue

        return False

    async def get_stats(self) -> FailureStats:
        """Get statistics about failures in the queue."""
        redis = await self._get_redis()

        stats = FailureStats()

        # Get all stats from hash
        raw_stats = await redis.hgetall(self.STATS_KEY)

        stats.total_failures = int(raw_stats.get("total", 0))
        stats.resolved_count = int(raw_stats.get("resolved", 0))
        stats.pending_count = stats.total_failures - stats.resolved_count

        # Parse type stats
        for key, value in raw_stats.items():
            if key.startswith("type:"):
                stats.by_type[key[5:]] = int(value)
            elif key.startswith("op:"):
                stats.by_operation[key[3:]] = int(value)

        # Get queue items for additional stats
        raw_items = await redis.zrange(self.QUEUE_KEY, 0, -1)
        total_retries = 0
        for item in raw_items:
            try:
                data = json.loads(item)
                total_retries += data.get("retry_count", 0)

                ts = datetime.fromisoformat(data["timestamp"])
                if stats.oldest_failure is None or ts < stats.oldest_failure:
                    stats.oldest_failure = ts
                if stats.newest_failure is None or ts > stats.newest_failure:
                    stats.newest_failure = ts

            except (json.JSONDecodeError, KeyError):
                continue

        if stats.total_failures > 0:
            stats.avg_retry_count = total_retries / stats.total_failures

        return stats

    async def clear_resolved(self, older_than_days: int | None = None) -> int:
        """
        Remove resolved failures from the queue.

        Args:
            older_than_days: Only clear resolved failures older than this

        Returns:
            Number of entries removed
        """
        redis = await self._get_redis()
        raw_items = await redis.zrange(self.QUEUE_KEY, 0, -1)

        removed = 0
        cutoff = None
        if older_than_days:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=older_than_days)

        for item in raw_items:
            try:
                data = json.loads(item)
                if data.get("resolved"):
                    if cutoff:
                        ts = datetime.fromisoformat(data["timestamp"])
                        if ts > cutoff:
                            continue

                    await redis.zrem(self.QUEUE_KEY, item)
                    removed += 1

            except (json.JSONDecodeError, KeyError):
                continue

        if removed > 0:
            self._logger.info("Cleared resolved failures", count=removed)

        return removed


class InMemoryFallback:
    """In-memory fallback when Redis is not available."""

    def __init__(self):
        self._data: dict[str, dict] = {}
        self._sorted_sets: dict[str, list[tuple[str, float]]] = {}
        self._hashes: dict[str, dict[str, str]] = {}

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if key not in self._sorted_sets:
            self._sorted_sets[key] = []

        for member, score in mapping.items():
            self._sorted_sets[key].append((member, score))

        self._sorted_sets[key].sort(key=lambda x: x[1])
        return len(mapping)

    async def zrange(self, key: str, start: int, end: int, withscores: bool = False):
        items = self._sorted_sets.get(key, [])
        if end == -1:
            end = len(items)
        else:
            end = end + 1

        slice_items = items[start:end]

        if withscores:
            return slice_items
        return [item[0] for item in slice_items]

    async def zrem(self, key: str, member: str) -> int:
        if key not in self._sorted_sets:
            return 0

        original_len = len(self._sorted_sets[key])
        self._sorted_sets[key] = [
            (m, s) for m, s in self._sorted_sets[key] if m != member
        ]
        return original_len - len(self._sorted_sets[key])

    async def zcard(self, key: str) -> int:
        return len(self._sorted_sets.get(key, []))

    async def zremrangebyrank(self, key: str, start: int, end: int) -> int:
        if key not in self._sorted_sets:
            return 0

        original_len = len(self._sorted_sets[key])
        self._sorted_sets[key] = (
            self._sorted_sets[key][:start] + self._sorted_sets[key][end + 1:]
        )
        return original_len - len(self._sorted_sets[key])

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        if key not in self._hashes:
            self._hashes[key] = {}

        current = int(self._hashes[key].get(field, "0"))
        self._hashes[key][field] = str(current + amount)
        return current + amount

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hashes.get(key, {})
