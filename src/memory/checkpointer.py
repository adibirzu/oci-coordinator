"""
ATP-backed Checkpointer for LangGraph.

Provides persistent checkpointing to OCI ATP for fault-tolerant
multi-agent orchestration. Checkpoints are stored in ATP and
cached in Redis for fast access.

Uses JSON serialization for safety (no pickle).

Usage:
    from src.memory.checkpointer import ATPCheckpointer, create_checkpointer

    # Create checkpointer
    checkpointer = await create_checkpointer()

    # Use with LangGraph
    graph = StateGraph(MyState)
    compiled = graph.compile(checkpointer=checkpointer)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import structlog
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = structlog.get_logger(__name__)


@dataclass
class CheckpointRecord:
    """Checkpoint record for persistence."""

    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    parent_checkpoint_id: str | None
    checkpoint_data: str  # JSON string
    metadata: dict
    created_at: datetime


class ATPCheckpointer(BaseCheckpointSaver):
    """
    LangGraph checkpointer with ATP persistence and Redis caching.

    Features:
    - Persistent checkpoints in OCI ATP
    - Redis hot cache for fast retrieval
    - Automatic cleanup of old checkpoints
    - Cross-agent state sharing
    - JSON serialization (safe, no pickle)

    Schema in ATP:
    - langgraph_checkpoints: Main checkpoint storage
    - langgraph_checkpoint_writes: Pending writes for batch operations
    """

    def __init__(
        self,
        atp_pool: Any,
        redis_client: Any | None = None,
        cache_ttl: timedelta = timedelta(hours=4),
        retention_days: int = 7,
    ):
        """
        Initialize ATP Checkpointer.

        Args:
            atp_pool: oracledb async connection pool
            redis_client: Optional Redis client for caching
            cache_ttl: Cache TTL for Redis
            retention_days: Days to keep checkpoints in ATP
        """
        super().__init__(serde=JsonPlusSerializer())
        self._pool = atp_pool
        self._redis = redis_client
        self._cache_ttl = cache_ttl
        self._retention_days = retention_days
        self._logger = logger.bind(component="ATPCheckpointer")

    @classmethod
    async def create(
        cls,
        atp_config: dict | None = None,
        redis_url: str | None = None,
    ) -> ATPCheckpointer:
        """
        Factory method to create and initialize checkpointer.

        Args:
            atp_config: ATP configuration dict
            redis_url: Redis connection URL

        Returns:
            Initialized ATPCheckpointer
        """
        from src.memory.atp_config import ATPConfig, create_atp_pool

        # Create ATP pool
        config = (
            ATPConfig(**atp_config) if atp_config else ATPConfig.from_env()
        )
        pool = await create_atp_pool(config)

        # Create Redis client if URL provided
        redis_client = None
        if redis_url:
            import redis.asyncio as redis_lib

            redis_client = redis_lib.from_url(redis_url, decode_responses=True)

        checkpointer = cls(atp_pool=pool, redis_client=redis_client)

        # Initialize schema
        await checkpointer._init_schema()

        return checkpointer

    async def _init_schema(self) -> None:
        """Initialize ATP schema for checkpoints."""
        ddl_statements = [
            """
            CREATE TABLE langgraph_checkpoints (
                thread_id VARCHAR2(256) NOT NULL,
                checkpoint_ns VARCHAR2(256) DEFAULT '' NOT NULL,
                checkpoint_id VARCHAR2(256) NOT NULL,
                parent_checkpoint_id VARCHAR2(256),
                checkpoint_data CLOB NOT NULL,
                metadata CLOB,
                created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
            """,
            """
            CREATE INDEX idx_checkpoints_thread
            ON langgraph_checkpoints(thread_id, checkpoint_ns, created_at DESC)
            """,
            """
            CREATE TABLE langgraph_checkpoint_writes (
                thread_id VARCHAR2(256) NOT NULL,
                checkpoint_ns VARCHAR2(256) DEFAULT '' NOT NULL,
                checkpoint_id VARCHAR2(256) NOT NULL,
                task_id VARCHAR2(256) NOT NULL,
                channel VARCHAR2(256) NOT NULL,
                type_name VARCHAR2(64) NOT NULL,
                value_json CLOB,
                created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, channel)
            )
            """,
        ]

        async with self._pool.acquire() as conn, conn.cursor() as cursor:
            for ddl in ddl_statements:
                try:
                    await cursor.execute(ddl)
                except Exception as e:
                    # ORA-00955: table already exists
                    # ORA-01408: index already exists
                    if hasattr(e, "args") and len(e.args) > 0:
                        err = e.args[0]
                        if hasattr(err, "code") and err.code in (955, 1408):
                            continue
                    self._logger.debug("DDL already exists", error=str(e))
            await conn.commit()

        self._logger.info("ATP checkpoint schema initialized")

    def _cache_key(
        self, thread_id: str, checkpoint_ns: str = "", checkpoint_id: str = ""
    ) -> str:
        """Generate cache key."""
        return f"checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    async def _get_from_cache(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str = ""
    ) -> str | None:
        """Get checkpoint from Redis cache."""
        if not self._redis:
            return None

        try:
            key = self._cache_key(thread_id, checkpoint_ns, checkpoint_id)
            return await self._redis.get(key)
        except Exception as e:
            self._logger.warning("Cache get failed", error=str(e))
            return None

    async def _set_cache(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        data: str,
    ) -> None:
        """Set checkpoint in Redis cache."""
        if not self._redis:
            return

        try:
            key = self._cache_key(thread_id, checkpoint_ns, checkpoint_id)
            await self._redis.setex(
                key,
                int(self._cache_ttl.total_seconds()),
                data,
            )
        except Exception as e:
            self._logger.warning("Cache set failed", error=str(e))

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """Serialize checkpoint to JSON string."""
        # dumps_typed returns (type_name, value_data)
        type_name, value_data = self.serde.dumps_typed(checkpoint)
        return json.dumps({"type": type_name, "data": value_data}, default=str)

    def _deserialize_checkpoint(self, data: str) -> Checkpoint:
        """Deserialize checkpoint from JSON string."""
        parsed = json.loads(data)
        # loads_typed takes (type_name, value_data)
        return self.serde.loads_typed((parsed["type"], parsed["data"]))

    async def aget_tuple(self, config: dict) -> CheckpointTuple | None:
        """
        Get checkpoint tuple for a thread.

        Args:
            config: Configuration with thread_id and optional checkpoint_ns

        Returns:
            CheckpointTuple or None if not found
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        # Try cache first
        if checkpoint_id:
            cached = await self._get_from_cache(
                thread_id, checkpoint_ns, checkpoint_id
            )
            if cached:
                checkpoint = self._deserialize_checkpoint(cached)
                return CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint,
                    metadata=CheckpointMetadata(source="cache", step=-1, writes=None, parents={}),
                    parent_config=None,
                    pending_writes=[],
                )

        # Query ATP
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if checkpoint_id:
                    # Get specific checkpoint
                    await cursor.execute(
                        """
                        SELECT checkpoint_id, parent_checkpoint_id,
                               checkpoint_data, metadata
                        FROM langgraph_checkpoints
                        WHERE thread_id = :thread_id
                          AND checkpoint_ns = :checkpoint_ns
                          AND checkpoint_id = :checkpoint_id
                        """,
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        },
                    )
                else:
                    # Get latest checkpoint
                    await cursor.execute(
                        """
                        SELECT checkpoint_id, parent_checkpoint_id,
                               checkpoint_data, metadata
                        FROM langgraph_checkpoints
                        WHERE thread_id = :thread_id
                          AND checkpoint_ns = :checkpoint_ns
                        ORDER BY created_at DESC
                        FETCH FIRST 1 ROW ONLY
                        """,
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                        },
                    )

                row = await cursor.fetchone()

                if not row:
                    return None

                ckpt_id, parent_id, ckpt_data, metadata_json = row

                # Read CLOB data
                checkpoint_str = ckpt_data.read() if hasattr(ckpt_data, "read") else ckpt_data
                metadata_str = metadata_json.read() if metadata_json and hasattr(metadata_json, "read") else metadata_json
                metadata = json.loads(metadata_str) if metadata_str else {}

                # Cache the result
                await self._set_cache(
                    thread_id, checkpoint_ns, ckpt_id, checkpoint_str
                )

                # Get pending writes for this checkpoint
                await cursor.execute(
                    """
                    SELECT task_id, channel, type_name, value_json
                    FROM langgraph_checkpoint_writes
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                      AND checkpoint_id = :checkpoint_id
                    ORDER BY created_at
                    """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": ckpt_id,
                    },
                )

                pending_writes = []
                async for write_row in cursor:
                    task_id, channel, type_name, value_json_clob = write_row
                    value_str = (
                        value_json_clob.read()
                        if value_json_clob and hasattr(value_json_clob, "read")
                        else value_json_clob
                    )
                    if value_str:
                        value_data = json.loads(value_str)
                        value = self.serde.loads_typed((type_name, value_data))
                        pending_writes.append((task_id, channel, value))

                # Deserialize checkpoint
                checkpoint = self._deserialize_checkpoint(checkpoint_str)

                # Build parent config
                parent_config = None
                if parent_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_id,
                        }
                    }

                return CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": ckpt_id,
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=CheckpointMetadata(**metadata) if metadata else CheckpointMetadata(
                        source="atp", step=-1, writes=None, parents={}
                    ),
                    parent_config=parent_config,
                    pending_writes=pending_writes,
                )

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> dict:
        """
        Save a checkpoint.

        Args:
            config: Configuration with thread_id
            checkpoint: The checkpoint to save
            metadata: Checkpoint metadata
            new_versions: Channel versions

        Returns:
            Updated config with checkpoint_id
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # Generate new checkpoint ID
        checkpoint_id = checkpoint["id"]

        # Serialize checkpoint using JSON
        checkpoint_json = self._serialize_checkpoint(checkpoint)

        # Serialize metadata
        metadata_json = json.dumps(
            {
                "source": metadata.source,
                "step": metadata.step,
                "writes": metadata.writes,
                "parents": metadata.parents,
            },
            default=str,
        )

        # Save to ATP
        async with self._pool.acquire() as conn, conn.cursor() as cursor:
            await cursor.execute(
                """
                    MERGE INTO langgraph_checkpoints t
                    USING (SELECT :thread_id as thread_id,
                                  :checkpoint_ns as checkpoint_ns,
                                  :checkpoint_id as checkpoint_id FROM dual) s
                    ON (t.thread_id = s.thread_id
                        AND t.checkpoint_ns = s.checkpoint_ns
                        AND t.checkpoint_id = s.checkpoint_id)
                    WHEN MATCHED THEN
                        UPDATE SET checkpoint_data = :checkpoint_data,
                                   metadata = :metadata,
                                   parent_checkpoint_id = :parent_id
                    WHEN NOT MATCHED THEN
                        INSERT (thread_id, checkpoint_ns, checkpoint_id,
                                parent_checkpoint_id, checkpoint_data, metadata)
                        VALUES (:thread_id, :checkpoint_ns, :checkpoint_id,
                                :parent_id, :checkpoint_data, :metadata)
                    """,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "parent_id": parent_checkpoint_id,
                    "checkpoint_data": checkpoint_json,
                    "metadata": metadata_json,
                },
            )
            await conn.commit()

        # Update cache
        await self._set_cache(
            thread_id, checkpoint_ns, checkpoint_id, checkpoint_json
        )

        self._logger.debug(
            "Checkpoint saved",
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Save pending writes for a checkpoint.

        Args:
            config: Configuration with checkpoint info
            writes: List of (channel, value) writes
            task_id: Task identifier
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        async with self._pool.acquire() as conn, conn.cursor() as cursor:
            for channel, value in writes:
                type_name, value_data = self.serde.dumps_typed(value)
                value_json = json.dumps(value_data, default=str)

                await cursor.execute(
                    """
                        MERGE INTO langgraph_checkpoint_writes t
                        USING (SELECT :thread_id as thread_id,
                                      :checkpoint_ns as checkpoint_ns,
                                      :checkpoint_id as checkpoint_id,
                                      :task_id as task_id,
                                      :channel as channel FROM dual) s
                        ON (t.thread_id = s.thread_id
                            AND t.checkpoint_ns = s.checkpoint_ns
                            AND t.checkpoint_id = s.checkpoint_id
                            AND t.task_id = s.task_id
                            AND t.channel = s.channel)
                        WHEN MATCHED THEN
                            UPDATE SET type_name = :type_name, value_json = :value_json
                        WHEN NOT MATCHED THEN
                            INSERT (thread_id, checkpoint_ns, checkpoint_id,
                                    task_id, channel, type_name, value_json)
                            VALUES (:thread_id, :checkpoint_ns, :checkpoint_id,
                                    :task_id, :channel, :type_name, :value_json)
                        """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "channel": channel,
                        "type_name": type_name,
                        "value_json": value_json,
                    },
                )
            await conn.commit()

    async def alist(
        self,
        config: dict | None,
        *,
        filter: dict | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> list[CheckpointTuple]:
        """
        List checkpoints for a thread.

        Args:
            config: Configuration with thread_id
            filter: Optional filter criteria
            before: Get checkpoints before this config
            limit: Maximum number to return

        Returns:
            List of CheckpointTuples
        """
        if not config:
            return []

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        results = []
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                query = """
                    SELECT checkpoint_id, parent_checkpoint_id,
                           checkpoint_data, metadata
                    FROM langgraph_checkpoints
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                """
                params = {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                }

                if before:
                    before_id = before["configurable"]["checkpoint_id"]
                    query += """
                        AND created_at < (
                            SELECT created_at FROM langgraph_checkpoints
                            WHERE thread_id = :thread_id
                              AND checkpoint_ns = :checkpoint_ns
                              AND checkpoint_id = :before_id
                        )
                    """
                    params["before_id"] = before_id

                query += " ORDER BY created_at DESC"

                if limit:
                    query += f" FETCH FIRST {limit} ROWS ONLY"

                await cursor.execute(query, params)

                async for row in cursor:
                    ckpt_id, parent_id, ckpt_data, metadata_clob = row
                    checkpoint_str = (
                        ckpt_data.read()
                        if hasattr(ckpt_data, "read")
                        else ckpt_data
                    )
                    metadata_str = (
                        metadata_clob.read()
                        if metadata_clob and hasattr(metadata_clob, "read")
                        else metadata_clob
                    )
                    metadata = json.loads(metadata_str) if metadata_str else {}

                    checkpoint = self._deserialize_checkpoint(checkpoint_str)

                    parent_config = None
                    if parent_id:
                        parent_config = {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_id,
                            }
                        }

                    results.append(
                        CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": ckpt_id,
                                }
                            },
                            checkpoint=checkpoint,
                            metadata=CheckpointMetadata(**metadata) if metadata else CheckpointMetadata(
                                source="atp", step=-1, writes=None, parents={}
                            ),
                            parent_config=parent_config,
                            pending_writes=[],
                        )
                    )

        return results

    async def cleanup_old_checkpoints(self, days: int | None = None) -> int:
        """
        Remove checkpoints older than retention period.

        Args:
            days: Days to keep (uses retention_days if not specified)

        Returns:
            Number of checkpoints deleted
        """
        retention = days or self._retention_days

        async with self._pool.acquire() as conn, conn.cursor() as cursor:
            # Delete old writes first (FK constraint)
            await cursor.execute(
                """
                    DELETE FROM langgraph_checkpoint_writes
                    WHERE created_at < SYSTIMESTAMP - INTERVAL :days DAY
                    """,
                {"days": str(retention)},
            )

            # Delete old checkpoints
            await cursor.execute(
                """
                    DELETE FROM langgraph_checkpoints
                    WHERE created_at < SYSTIMESTAMP - INTERVAL :days DAY
                    """,
                {"days": str(retention)},
            )

            deleted = cursor.rowcount
            await conn.commit()

        self._logger.info(
            "Cleaned up old checkpoints",
            deleted=deleted,
            retention_days=retention,
        )

        return deleted

    # Synchronous methods required by LangGraph (delegate to async)
    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        """Sync wrapper for aget_tuple."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.aget_tuple(config)
                )
                return future.result()
        else:
            return asyncio.run(self.aget_tuple(config))

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> dict:
        """Sync wrapper for aput."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.aput(config, checkpoint, metadata, new_versions),
                )
                return future.result()
        else:
            return asyncio.run(
                self.aput(config, checkpoint, metadata, new_versions)
            )

    def put_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Sync wrapper for aput_writes."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.aput_writes(config, writes, task_id)
                )
                future.result()
        else:
            asyncio.run(self.aput_writes(config, writes, task_id))

    def list(
        self,
        config: dict | None,
        *,
        filter: dict | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Sync wrapper for alist."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.alist(config, filter=filter, before=before, limit=limit),
                )
                results = future.result()
        else:
            results = asyncio.run(
                self.alist(config, filter=filter, before=before, limit=limit)
            )

        yield from results


async def create_checkpointer(
    use_atp: bool = True,
    redis_url: str | None = None,
) -> BaseCheckpointSaver:
    """
    Factory to create appropriate checkpointer.

    Args:
        use_atp: If True, use ATP persistence; else use in-memory
        redis_url: Redis URL for caching

    Returns:
        Configured checkpointer instance
    """
    if use_atp:
        try:
            return await ATPCheckpointer.create(redis_url=redis_url)
        except Exception as e:
            logger.warning(
                "ATP checkpointer creation failed, falling back to memory",
                error=str(e),
            )

    # Fall back to in-memory
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()
