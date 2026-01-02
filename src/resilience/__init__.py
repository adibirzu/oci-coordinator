"""
Resilience Infrastructure.

Provides production-grade resilience patterns for the OCI Coordinator:

- **DeadLetterQueue**: Persist failed operations for analysis and retry
- **Bulkhead**: Resource isolation between operation types
- **HealthMonitor**: Component-level health with auto-restart

Usage:
    from src.resilience import DeadLetterQueue, Bulkhead, HealthMonitor

    # Dead letter queue for failed operations
    dlq = DeadLetterQueue(redis_url="redis://localhost:6379")
    await dlq.enqueue("tool_call", "oci_cost_get_summary", error, params)

    # Bulkhead for resource isolation
    async with Bulkhead.get_instance().acquire("database"):
        await execute_database_operation()

    # Health monitor with auto-restart
    monitor = HealthMonitor.get_instance()
    monitor.register_check(HealthCheck(
        name="mcp_server",
        check_func=check_fn,
        restart_func=restart_fn,
    ))
    await monitor.start()
"""

from src.resilience.deadletter import (
    DeadLetterQueue,
    FailedOperation,
    FailureType,
    FailureStats,
)
from src.resilience.bulkhead import (
    Bulkhead,
    BulkheadPartition,
    BulkheadHandle,
    PartitionMetrics,
)
from src.resilience.health import (
    HealthMonitor,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    ComponentHealth,
    create_mcp_server_check,
    create_mcp_server_restart,
    create_redis_check,
    create_llm_check,
)

__all__ = [
    # Deadletter Queue
    "DeadLetterQueue",
    "FailedOperation",
    "FailureType",
    "FailureStats",
    # Bulkhead
    "Bulkhead",
    "BulkheadPartition",
    "BulkheadHandle",
    "PartitionMetrics",
    # Health Monitor
    "HealthMonitor",
    "HealthCheck",
    "HealthCheckResult",
    "HealthStatus",
    "ComponentHealth",
    # Health Check Factories
    "create_mcp_server_check",
    "create_mcp_server_restart",
    "create_redis_check",
    "create_llm_check",
]
