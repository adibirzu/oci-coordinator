"""
Health Monitor with Auto-Restart.

Provides component-level health monitoring with automatic recovery.
Coordinates with circuit breakers and deadletter queues for
comprehensive failure management.

Features:
- Configurable health check registration
- Periodic monitoring with configurable intervals
- Automatic restart of unhealthy components
- Event callbacks for status changes
- Metrics tracking per component
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status of a component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    status: HealthStatus
    message: str | None = None
    latency_ms: float = 0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthCheck:
    """Configuration for a health check."""

    name: str
    check_func: Callable[[], Awaitable[HealthCheckResult]]
    restart_func: Callable[[], Awaitable[bool]] | None = None
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    failure_threshold: int = 3
    success_threshold: int = 1
    enabled: bool = True
    critical: bool = False  # Critical components block other operations when unhealthy


@dataclass
class ComponentHealth:
    """Health state for a single component."""

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    total_restarts: int = 0
    last_check: datetime | None = None
    last_healthy: datetime | None = None
    last_failure_reason: str | None = None
    avg_latency_ms: float = 0
    _latency_sum: float = 0

    def record_success(self, latency_ms: float) -> None:
        """Record a successful check."""
        self.status = HealthStatus.HEALTHY
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.total_checks += 1
        self.last_check = datetime.utcnow()
        self.last_healthy = self.last_check
        self._latency_sum += latency_ms
        self.avg_latency_ms = self._latency_sum / self.total_checks

    def record_failure(self, reason: str | None = None) -> None:
        """Record a failed check."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.total_checks += 1
        self.total_failures += 1
        self.last_check = datetime.utcnow()
        self.last_failure_reason = reason

        if self.consecutive_failures >= 3:
            self.status = HealthStatus.UNHEALTHY
        else:
            self.status = HealthStatus.DEGRADED

    def record_restart(self) -> None:
        """Record a restart attempt."""
        self.status = HealthStatus.RESTARTING
        self.total_restarts += 1

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_checks": self.total_checks,
            "total_failures": self.total_failures,
            "total_restarts": self.total_restarts,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_healthy": self.last_healthy.isoformat() if self.last_healthy else None,
            "last_failure_reason": self.last_failure_reason,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


# Event types for health status changes
EVENT_HEALTH_CHECK_PASSED = "health_check_passed"
EVENT_HEALTH_CHECK_FAILED = "health_check_failed"
EVENT_STATUS_CHANGED = "status_changed"
EVENT_RESTART_STARTED = "restart_started"
EVENT_RESTART_COMPLETED = "restart_completed"
EVENT_RESTART_FAILED = "restart_failed"


class HealthMonitor:
    """
    Component-level health monitor with auto-restart.

    Monitors registered components, tracks health status, and
    automatically restarts unhealthy components.

    Example:
        monitor = HealthMonitor()

        # Register MCP server check
        async def check_mcp():
            try:
                await registry.get_client("database-observatory")
                return HealthCheckResult(HealthStatus.HEALTHY)
            except Exception as e:
                return HealthCheckResult(HealthStatus.UNHEALTHY, str(e))

        async def restart_mcp():
            await registry.reconnect("database-observatory")
            return True

        monitor.register_check(HealthCheck(
            name="mcp_database",
            check_func=check_mcp,
            restart_func=restart_mcp,
            failure_threshold=3,
        ))

        # Start monitoring
        await monitor.start()

        # Get health status
        status = monitor.get_status("mcp_database")
        all_healthy = monitor.is_healthy()

        # Stop monitoring
        await monitor.stop()
    """

    _instance: HealthMonitor | None = None

    def __init__(
        self,
        default_interval: float = 30.0,
        restart_cooldown: float = 60.0,
        max_restarts_per_hour: int = 5,
    ):
        """Initialize health monitor.

        Args:
            default_interval: Default check interval in seconds
            restart_cooldown: Minimum seconds between restarts
            max_restarts_per_hour: Maximum restart attempts per hour
        """
        self._checks: dict[str, HealthCheck] = {}
        self._health: dict[str, ComponentHealth] = {}
        self._callbacks: list[Callable[[str, str, dict], Awaitable[None] | None]] = []
        self._running = False
        self._tasks: dict[str, asyncio.Task] = {}
        self._default_interval = default_interval
        self._restart_cooldown = restart_cooldown
        self._max_restarts_per_hour = max_restarts_per_hour
        self._last_restart_times: dict[str, list[float]] = {}
        self._logger = logger.bind(component="HealthMonitor")

    @classmethod
    def get_instance(cls) -> HealthMonitor:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        if cls._instance:
            asyncio.create_task(cls._instance.stop())
        cls._instance = None

    def register_check(self, check: HealthCheck) -> None:
        """
        Register a health check.

        Args:
            check: HealthCheck configuration
        """
        self._checks[check.name] = check
        self._health[check.name] = ComponentHealth(name=check.name)
        self._last_restart_times[check.name] = []

        self._logger.info(
            "Health check registered",
            name=check.name,
            interval=check.interval_seconds,
            critical=check.critical,
        )

    def unregister_check(self, name: str) -> bool:
        """
        Unregister a health check.

        Args:
            name: Name of the check to remove

        Returns:
            True if check was found and removed
        """
        if name in self._checks:
            del self._checks[name]
            del self._health[name]
            del self._last_restart_times[name]

            # Cancel running task if any
            if name in self._tasks:
                self._tasks[name].cancel()
                del self._tasks[name]

            self._logger.info("Health check unregistered", name=name)
            return True
        return False

    def on_event(
        self,
        callback: Callable[[str, str, dict], Awaitable[None] | None],
    ) -> None:
        """
        Register an event callback.

        Args:
            callback: Function called with (event_type, component_name, data)
        """
        self._callbacks.append(callback)

    async def _trigger_event(
        self,
        event_type: str,
        component: str,
        data: dict[str, Any],
    ) -> None:
        """Trigger an event to all callbacks."""
        for callback in self._callbacks:
            try:
                result = callback(event_type, component, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._logger.warning(
                    "Event callback error",
                    event=event_type,
                    component=component,
                    error=str(e),
                )

    async def start(self) -> None:
        """Start health monitoring for all registered checks."""
        if self._running:
            return

        self._running = True
        self._logger.info(
            "Health monitor starting",
            checks=list(self._checks.keys()),
        )

        # Start a task for each check
        for name, check in self._checks.items():
            if check.enabled:
                self._tasks[name] = asyncio.create_task(
                    self._run_check_loop(name),
                    name=f"health_check_{name}",
                )

    async def stop(self) -> None:
        """Stop health monitoring."""
        if not self._running:
            return

        self._running = False

        # Cancel all tasks
        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        self._logger.info("Health monitor stopped")

    async def _run_check_loop(self, name: str) -> None:
        """Run the health check loop for a component."""
        check = self._checks.get(name)
        if not check:
            return

        while self._running and check.enabled:
            try:
                await self._perform_check(name)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(
                    "Health check loop error",
                    name=name,
                    error=str(e),
                )

            await asyncio.sleep(check.interval_seconds)

    async def _perform_check(self, name: str) -> HealthCheckResult:
        """
        Perform a single health check.

        Args:
            name: Name of the component to check

        Returns:
            HealthCheckResult
        """
        check = self._checks.get(name)
        health = self._health.get(name)

        if not check or not health:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="Check not found",
            )

        start_time = time.perf_counter()
        old_status = health.status

        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_func(),
                timeout=check.timeout_seconds,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if result.status == HealthStatus.HEALTHY:
                health.record_success(latency_ms)
                await self._trigger_event(
                    EVENT_HEALTH_CHECK_PASSED,
                    name,
                    {"latency_ms": latency_ms},
                )
            else:
                health.record_failure(result.message)
                await self._trigger_event(
                    EVENT_HEALTH_CHECK_FAILED,
                    name,
                    {"message": result.message, "status": result.status.value},
                )

        except asyncio.TimeoutError:
            health.record_failure("Health check timeout")
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Timeout after {check.timeout_seconds}s",
            )
            await self._trigger_event(
                EVENT_HEALTH_CHECK_FAILED,
                name,
                {"message": "timeout"},
            )

        except Exception as e:
            health.record_failure(str(e))
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
            await self._trigger_event(
                EVENT_HEALTH_CHECK_FAILED,
                name,
                {"message": str(e)},
            )

        # Check for status change
        if health.status != old_status:
            await self._trigger_event(
                EVENT_STATUS_CHANGED,
                name,
                {"old_status": old_status.value, "new_status": health.status.value},
            )

        # Auto-restart if unhealthy and restart function available
        if (
            health.status == HealthStatus.UNHEALTHY
            and health.consecutive_failures >= check.failure_threshold
            and check.restart_func
        ):
            await self._attempt_restart(name)

        return result

    async def _attempt_restart(self, name: str) -> bool:
        """
        Attempt to restart an unhealthy component.

        Args:
            name: Name of the component to restart

        Returns:
            True if restart was successful
        """
        check = self._checks.get(name)
        health = self._health.get(name)

        if not check or not health or not check.restart_func:
            return False

        # Check restart rate limiting
        now = time.time()
        restart_times = self._last_restart_times.get(name, [])

        # Clean old restart times (older than 1 hour)
        restart_times = [t for t in restart_times if now - t < 3600]
        self._last_restart_times[name] = restart_times

        if len(restart_times) >= self._max_restarts_per_hour:
            self._logger.warning(
                "Restart rate limit reached",
                name=name,
                restarts_last_hour=len(restart_times),
            )
            return False

        # Check cooldown
        if restart_times and now - restart_times[-1] < self._restart_cooldown:
            self._logger.debug(
                "Restart cooldown active",
                name=name,
                seconds_remaining=self._restart_cooldown - (now - restart_times[-1]),
            )
            return False

        # Perform restart
        health.record_restart()
        await self._trigger_event(EVENT_RESTART_STARTED, name, {})

        self._logger.info(
            "Attempting restart",
            name=name,
            consecutive_failures=health.consecutive_failures,
        )

        try:
            success = await check.restart_func()

            if success:
                restart_times.append(now)
                self._last_restart_times[name] = restart_times

                # Reset failure count on successful restart
                health.consecutive_failures = 0
                health.status = HealthStatus.UNKNOWN  # Will be updated on next check

                await self._trigger_event(
                    EVENT_RESTART_COMPLETED,
                    name,
                    {"success": True},
                )

                self._logger.info("Restart successful", name=name)
                return True
            else:
                await self._trigger_event(
                    EVENT_RESTART_FAILED,
                    name,
                    {"reason": "restart function returned false"},
                )
                self._logger.warning("Restart failed", name=name)
                return False

        except Exception as e:
            await self._trigger_event(
                EVENT_RESTART_FAILED,
                name,
                {"reason": str(e)},
            )
            self._logger.error("Restart error", name=name, error=str(e))
            return False

    async def check_now(self, name: str) -> HealthCheckResult:
        """
        Perform an immediate health check.

        Args:
            name: Name of the component to check

        Returns:
            HealthCheckResult
        """
        return await self._perform_check(name)

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """
        Check all registered components immediately.

        Returns:
            Dictionary of component name to HealthCheckResult
        """
        results = {}
        tasks = []

        for name in self._checks:
            tasks.append(self._perform_check(name))

        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(self._checks.keys(), check_results):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                )
            else:
                results[name] = result

        return results

    def get_status(self, name: str) -> ComponentHealth | None:
        """
        Get health status for a component.

        Args:
            name: Component name

        Returns:
            ComponentHealth or None if not found
        """
        return self._health.get(name)

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """
        Get health status for all components.

        Returns:
            Dictionary of component name to health status dict
        """
        return {name: health.to_dict() for name, health in self._health.items()}

    def is_healthy(self, include_degraded: bool = True) -> bool:
        """
        Check if all components are healthy.

        Args:
            include_degraded: Consider DEGRADED as healthy

        Returns:
            True if all components are healthy
        """
        for health in self._health.values():
            if health.status == HealthStatus.UNHEALTHY:
                return False
            if not include_degraded and health.status == HealthStatus.DEGRADED:
                return False
        return True

    def is_critical_healthy(self) -> bool:
        """
        Check if all critical components are healthy.

        Returns:
            True if all critical components are healthy
        """
        for name, check in self._checks.items():
            if check.critical:
                health = self._health.get(name)
                if health and health.status == HealthStatus.UNHEALTHY:
                    return False
        return True

    def get_unhealthy_components(self) -> list[str]:
        """Get list of unhealthy component names."""
        return [
            name
            for name, health in self._health.items()
            if health.status == HealthStatus.UNHEALTHY
        ]

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of health status.

        Returns:
            Summary dictionary
        """
        healthy = 0
        degraded = 0
        unhealthy = 0
        unknown = 0

        for health in self._health.values():
            if health.status == HealthStatus.HEALTHY:
                healthy += 1
            elif health.status == HealthStatus.DEGRADED:
                degraded += 1
            elif health.status == HealthStatus.UNHEALTHY:
                unhealthy += 1
            else:
                unknown += 1

        return {
            "total_components": len(self._health),
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "unknown": unknown,
            "is_healthy": self.is_healthy(),
            "is_critical_healthy": self.is_critical_healthy(),
            "unhealthy_components": self.get_unhealthy_components(),
        }


# Pre-built health check factories for common components


async def create_mcp_server_check(
    registry,
    server_id: str,
) -> Callable[[], Awaitable[HealthCheckResult]]:
    """
    Create a health check function for an MCP server.

    Args:
        registry: ServerRegistry instance
        server_id: Server ID to check

    Returns:
        Health check function
    """
    async def check() -> HealthCheckResult:
        try:
            client = await registry.get_client(server_id)
            if client:
                # Try to list tools as a health check
                tools = await client.list_tools()
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    details={"tool_count": len(tools)},
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="No client available",
                )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check


async def create_mcp_server_restart(
    registry,
    server_id: str,
) -> Callable[[], Awaitable[bool]]:
    """
    Create a restart function for an MCP server.

    Args:
        registry: ServerRegistry instance
        server_id: Server ID to restart

    Returns:
        Restart function
    """
    async def restart() -> bool:
        try:
            await registry.disconnect(server_id)
            await asyncio.sleep(2)  # Brief pause before reconnect
            await registry.connect(server_id)
            return True
        except Exception:
            return False

    return restart


async def create_redis_check(
    redis_url: str,
) -> Callable[[], Awaitable[HealthCheckResult]]:
    """
    Create a health check function for Redis.

    Args:
        redis_url: Redis connection URL

    Returns:
        Health check function
    """
    async def check() -> HealthCheckResult:
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(redis_url)
            await client.ping()
            await client.close()
            return HealthCheckResult(status=HealthStatus.HEALTHY)
        except ImportError:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message="Redis library not available",
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check


async def create_llm_check(
    llm,
) -> Callable[[], Awaitable[HealthCheckResult]]:
    """
    Create a health check function for an LLM.

    Args:
        llm: LangChain LLM instance

    Returns:
        Health check function
    """
    async def check() -> HealthCheckResult:
        try:
            # Simple invocation to test connectivity
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content="ping")])
            if response:
                return HealthCheckResult(status=HealthStatus.HEALTHY)
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="No response from LLM",
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check
