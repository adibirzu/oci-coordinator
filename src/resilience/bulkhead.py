"""
Bulkhead Pattern for Resource Isolation.

Isolates different operation types into separate resource pools,
preventing failures in one area from exhausting resources for others.

Features:
- Configurable partitions with max concurrent limits
- Automatic partition selection based on operation type
- Queue management with timeout
- Metrics tracking per partition
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BulkheadPartition(str, Enum):
    """Predefined resource partitions for OCI operations."""

    DATABASE = "database"           # Max 3 concurrent (SQL, OPSI operations)
    INFRASTRUCTURE = "infrastructure"  # Max 5 concurrent (compute, network)
    COST = "cost"                   # Max 2 concurrent (Usage API is slow)
    SECURITY = "security"           # Max 3 concurrent (Cloud Guard, VSS)
    DISCOVERY = "discovery"         # Max 6 concurrent (resource discovery)
    LLM = "llm"                     # Max 5 concurrent (LLM calls)
    DEFAULT = "default"             # Max 10 concurrent (catch-all)


# Default limits per partition
# Tuned for typical concurrent request patterns:
# - Database operations are most common, need headroom for parallel workflows
#   (each DB troubleshooting workflow can make 3-5 sequential tool calls)
# - Discovery operations do bulk listing and need high concurrency
# - LLM calls are expensive and should be bounded
DEFAULT_PARTITION_LIMITS = {
    BulkheadPartition.DATABASE: 15,     # Increased: DB workflows use 3-5 tools each, concurrent users
    BulkheadPartition.INFRASTRUCTURE: 10,  # Increased for parallel compute/network operations
    BulkheadPartition.COST: 8,          # Increased: cost queries are slow but parallelizable
    BulkheadPartition.SECURITY: 8,      # Increased for concurrent Cloud Guard queries
    BulkheadPartition.DISCOVERY: 12,    # Increased for parallel list/search operations
    BulkheadPartition.LLM: 8,           # Increased: LLM calls for multiple concurrent users
    BulkheadPartition.DEFAULT: 20,      # Increased for catch-all partition
}

# Map tool prefixes to partitions
TOOL_TO_PARTITION = {
    "oci_database_": BulkheadPartition.DATABASE,
    "oci_opsi_": BulkheadPartition.DATABASE,
    "execute_sql": BulkheadPartition.DATABASE,
    "oci_compute_": BulkheadPartition.INFRASTRUCTURE,
    "oci_network_": BulkheadPartition.INFRASTRUCTURE,
    "oci_cost_": BulkheadPartition.COST,
    "oci_security_": BulkheadPartition.SECURITY,
    "oci_search_": BulkheadPartition.DISCOVERY,
    "oci_list_": BulkheadPartition.DISCOVERY,
}

# Partition-specific acquire timeouts (seconds)
# These timeouts should match or exceed the longest tool timeout in each partition
# to prevent bulkhead starvation when tools are waiting for slow MCP responses
PARTITION_ACQUIRE_TIMEOUTS = {
    BulkheadPartition.DATABASE: 120.0,     # OPSI/DBMGMT tools can take 60-240s (see mcp_servers.yaml)
    BulkheadPartition.INFRASTRUCTURE: 30.0,  # Compute/network can take up to 180s for large compartments
    BulkheadPartition.COST: 120.0,         # Usage API is notoriously slow (60-180s)
    BulkheadPartition.SECURITY: 60.0,      # Cloud Guard queries can take 180s
    BulkheadPartition.DISCOVERY: 30.0,     # List/search can be slow with many resources
    BulkheadPartition.LLM: 90.0,           # LLM calls can be very slow (60-90s typical)
    BulkheadPartition.DEFAULT: 30.0,       # Default: reasonable wait before fail
}


def get_partition_timeout(partition: BulkheadPartition) -> float:
    """Get recommended acquire timeout for a partition.

    Args:
        partition: The bulkhead partition

    Returns:
        Recommended timeout in seconds
    """
    return PARTITION_ACQUIRE_TIMEOUTS.get(partition, 10.0)


@dataclass
class PartitionMetrics:
    """Metrics for a single bulkhead partition."""

    total_acquired: int = 0
    total_rejected: int = 0
    total_timeouts: int = 0
    current_active: int = 0
    peak_active: int = 0
    total_wait_time_ms: float = 0
    total_hold_time_ms: float = 0

    @property
    def avg_wait_time_ms(self) -> float:
        """Average time waiting for semaphore."""
        if self.total_acquired == 0:
            return 0
        return self.total_wait_time_ms / self.total_acquired

    @property
    def avg_hold_time_ms(self) -> float:
        """Average time holding the semaphore."""
        if self.total_acquired == 0:
            return 0
        return self.total_hold_time_ms / self.total_acquired

    @property
    def rejection_rate(self) -> float:
        """Rate of rejected acquisitions."""
        total = self.total_acquired + self.total_rejected
        if total == 0:
            return 0
        return self.total_rejected / total

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_acquired": self.total_acquired,
            "total_rejected": self.total_rejected,
            "total_timeouts": self.total_timeouts,
            "current_active": self.current_active,
            "peak_active": self.peak_active,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "avg_hold_time_ms": round(self.avg_hold_time_ms, 2),
            "rejection_rate": round(self.rejection_rate, 4),
        }


@dataclass
class PartitionState:
    """State for a bulkhead partition."""

    partition: BulkheadPartition
    max_concurrent: int
    semaphore: asyncio.Semaphore
    metrics: PartitionMetrics = field(default_factory=PartitionMetrics)
    created_at: datetime = field(default_factory=datetime.utcnow)


class BulkheadHandle:
    """Context manager handle for a bulkhead acquisition."""

    def __init__(
        self,
        partition_state: PartitionState,
        bulkhead: Bulkhead,
    ):
        self._state = partition_state
        self._bulkhead = bulkhead
        self._acquired = False
        self._acquire_time: float | None = None

    async def __aenter__(self) -> BulkheadHandle:
        """Acquire the bulkhead slot."""
        self._acquired = True
        self._acquire_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the bulkhead slot."""
        if self._acquired:
            hold_time_ms = (time.perf_counter() - self._acquire_time) * 1000
            self._bulkhead._release(self._state, hold_time_ms)
            self._acquired = False


class Bulkhead:
    """
    Bulkhead pattern implementation for resource isolation.

    Partitions resources into isolated pools to prevent cascading failures.
    Each partition has its own semaphore and metrics.

    Example:
        bulkhead = Bulkhead()

        # Use context manager
        async with bulkhead.acquire("database"):
            result = await execute_database_operation()

        # Or manually
        handle = await bulkhead.try_acquire("cost", timeout=5.0)
        if handle:
            try:
                result = await get_cost_data()
            finally:
                await handle.__aexit__(None, None, None)

        # Auto-detect partition from tool name
        partition = bulkhead.get_partition_for_tool("oci_database_execute_sql")
        async with bulkhead.acquire(partition):
            ...
    """

    _instance: Bulkhead | None = None

    def __init__(
        self,
        partition_limits: dict[BulkheadPartition, int] | None = None,
    ):
        """Initialize bulkhead with partition limits.

        Args:
            partition_limits: Override default limits per partition
        """
        self._limits = partition_limits or DEFAULT_PARTITION_LIMITS.copy()
        self._partitions: dict[BulkheadPartition, PartitionState] = {}
        self._logger = logger.bind(component="Bulkhead")

        # Initialize all partitions
        for partition, limit in self._limits.items():
            self._partitions[partition] = PartitionState(
                partition=partition,
                max_concurrent=limit,
                semaphore=asyncio.Semaphore(limit),
            )

    @classmethod
    def get_instance(cls) -> Bulkhead:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def get_partition_for_tool(self, tool_name: str) -> BulkheadPartition:
        """
        Determine the appropriate partition for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Appropriate BulkheadPartition
        """
        for prefix, partition in TOOL_TO_PARTITION.items():
            if tool_name.startswith(prefix) or tool_name == prefix:
                return partition
        return BulkheadPartition.DEFAULT

    async def acquire(
        self,
        partition: BulkheadPartition | str,
        timeout: float = 30.0,
    ) -> BulkheadHandle:
        """
        Acquire a slot in a partition.

        Args:
            partition: Partition to acquire from
            timeout: Maximum time to wait (seconds)

        Returns:
            BulkheadHandle context manager

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            ValueError: If partition is full and non-blocking
        """
        if isinstance(partition, str):
            partition = BulkheadPartition(partition)

        state = self._partitions.get(partition)
        if not state:
            self._logger.warning(
                "Unknown partition, using default",
                partition=partition,
            )
            state = self._partitions[BulkheadPartition.DEFAULT]

        wait_start = time.perf_counter()

        try:
            await asyncio.wait_for(
                state.semaphore.acquire(),
                timeout=timeout,
            )
        except TimeoutError:
            state.metrics.total_timeouts += 1
            self._logger.warning(
                "Bulkhead acquisition timeout",
                partition=partition.value,
                timeout=timeout,
                current_active=state.metrics.current_active,
            )
            raise

        # Update metrics
        wait_time_ms = (time.perf_counter() - wait_start) * 1000
        state.metrics.total_acquired += 1
        state.metrics.total_wait_time_ms += wait_time_ms
        state.metrics.current_active += 1
        state.metrics.peak_active = max(
            state.metrics.peak_active,
            state.metrics.current_active,
        )

        self._logger.debug(
            "Bulkhead acquired",
            partition=partition.value,
            current_active=state.metrics.current_active,
            max_concurrent=state.max_concurrent,
        )

        return BulkheadHandle(state, self)

    async def try_acquire(
        self,
        partition: BulkheadPartition | str,
        timeout: float = 0.0,
    ) -> BulkheadHandle | None:
        """
        Try to acquire a slot, returning None if unavailable.

        Args:
            partition: Partition to acquire from
            timeout: Maximum time to wait (0 = non-blocking)

        Returns:
            BulkheadHandle if acquired, None otherwise
        """
        try:
            return await self.acquire(partition, timeout)
        except TimeoutError:
            if isinstance(partition, str):
                partition = BulkheadPartition(partition)
            state = self._partitions.get(partition, self._partitions[BulkheadPartition.DEFAULT])
            state.metrics.total_rejected += 1
            return None

    def _release(self, state: PartitionState, hold_time_ms: float) -> None:
        """Release a slot back to the partition."""
        state.semaphore.release()
        state.metrics.current_active -= 1
        state.metrics.total_hold_time_ms += hold_time_ms

        self._logger.debug(
            "Bulkhead released",
            partition=state.partition.value,
            current_active=state.metrics.current_active,
            hold_time_ms=round(hold_time_ms, 2),
        )

    def get_metrics(
        self,
        partition: BulkheadPartition | str | None = None,
    ) -> dict[str, Any]:
        """
        Get metrics for partitions.

        Args:
            partition: Specific partition or None for all

        Returns:
            Metrics dictionary
        """
        if partition:
            if isinstance(partition, str):
                partition = BulkheadPartition(partition)
            state = self._partitions.get(partition)
            if state:
                return {
                    "partition": partition.value,
                    "max_concurrent": state.max_concurrent,
                    **state.metrics.to_dict(),
                }
            return {}

        # Return all partitions
        return {
            p.value: {
                "max_concurrent": s.max_concurrent,
                **s.metrics.to_dict(),
            }
            for p, s in self._partitions.items()
        }

    def get_available_slots(self, partition: BulkheadPartition | str) -> int:
        """Get number of available slots in a partition."""
        if isinstance(partition, str):
            partition = BulkheadPartition(partition)

        state = self._partitions.get(partition)
        if not state:
            return 0

        return state.max_concurrent - state.metrics.current_active

    def is_partition_full(self, partition: BulkheadPartition | str) -> bool:
        """Check if a partition has no available slots."""
        return self.get_available_slots(partition) <= 0

    def update_limit(self, partition: BulkheadPartition | str, new_limit: int) -> None:
        """
        Update the concurrent limit for a partition.

        Note: This creates a new semaphore, so existing waiters may be affected.

        Args:
            partition: Partition to update
            new_limit: New maximum concurrent limit
        """
        if isinstance(partition, str):
            partition = BulkheadPartition(partition)

        if partition in self._partitions:
            old_limit = self._partitions[partition].max_concurrent
            self._partitions[partition] = PartitionState(
                partition=partition,
                max_concurrent=new_limit,
                semaphore=asyncio.Semaphore(new_limit),
                metrics=self._partitions[partition].metrics,
            )
            self._logger.info(
                "Bulkhead limit updated",
                partition=partition.value,
                old_limit=old_limit,
                new_limit=new_limit,
            )
