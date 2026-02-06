#!/usr/bin/env python
"""OCI AI Agent Coordinator - Main Entry Point.

Starts the coordinator with Slack and API integration.

Usage:
    # Start both Slack and API (default)
    poetry run python -m src.main

    # Explicit modes
    poetry run python -m src.main --mode both      # Slack + API (default)
    poetry run python -m src.main --mode slack     # Slack only
    poetry run python -m src.main --mode api       # API only
    poetry run python -m src.main --mode api --port 8080  # Custom port
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Load environment from .env.local
# Search multiple locations to support worktrees and different working directories
from dotenv import load_dotenv

def _find_and_load_env():
    """Find and load .env.local from multiple possible locations.

    Search order:
    1. Project root (relative to this file): ../../../.env.local or ../.env.local
    2. Current working directory: ./.env.local
    3. Main project directory (for worktrees): ~/dev/oci-coordinator/.env.local
    4. Environment variable OCI_COORDINATOR_ENV_FILE
    """
    import os
    from pathlib import Path

    # Candidate paths to search
    candidates = [
        # Relative to this file (standard layout: src/main.py -> ../.env.local)
        Path(__file__).parent.parent / ".env.local",
        # Current working directory
        Path.cwd() / ".env.local",
        # Home-relative main project (for worktrees)
        Path.home() / "dev" / "oci-coordinator" / ".env.local",
        # Explicit override via environment variable
        Path(os.environ.get("OCI_COORDINATOR_ENV_FILE", "")) if os.environ.get("OCI_COORDINATOR_ENV_FILE") else None,
    ]

    # Try each candidate
    for env_file in candidates:
        if env_file and env_file.exists():
            load_dotenv(env_file, override=True)
            # Log which file was loaded (before structlog is configured)
            print(f"[startup] Loaded environment from: {env_file}")
            return str(env_file)

    # No .env.local found - warn but continue (might be using system env vars)
    print("[startup] Warning: No .env.local found. Using system environment variables only.")
    print(f"[startup] Searched: {[str(p) for p in candidates if p]}")
    return None

_loaded_env_file = _find_and_load_env()

import structlog

from src.agents.catalog import AgentCatalog
from src.mcp.catalog import ToolCatalog
from src.mcp.config import initialize_mcp_from_config, load_mcp_config
from src.mcp.registry import (
    EVENT_SERVER_CONNECTED,
    EVENT_SERVER_DISCONNECTED,
    EVENT_TOOLS_UPDATED,
    ServerRegistry,
)
from src.observability import init_observability, shutdown_observability
from src.resilience import (
    Bulkhead,
    DeadLetterQueue,
    HealthCheck,
    HealthCheckResult,
    HealthMonitor,
    HealthStatus,
)

# Global references for cleanup
_registry: ServerRegistry | None = None
_catalog: ToolCatalog | None = None
_health_monitor: HealthMonitor | None = None
_bulkhead: Bulkhead | None = None
_deadletter_queue: DeadLetterQueue | None = None
_coordinator = None  # LangGraph coordinator (pre-warmed at startup)

# Initialization lock to prevent race conditions in concurrent startup
_init_lock: asyncio.Lock | None = None
_initialized: bool = False


def _get_init_lock() -> asyncio.Lock:
    """Get or create the initialization lock (must be created within event loop)."""
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


def configure_logging() -> None:
    """Configure structlog to route through Python logging for OCI Logging integration."""
    # Shared processors for structlog
    # Note: format_exc_info is NOT included because ConsoleRenderer handles exceptions
    # Including it would cause the warning:
    # "Remove format_exc_info from your processor chain if you want pretty exceptions"
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog to use stdlib and render to log_kwargs
    # This passes logs through Python's logging system where OCI handler is attached
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up Python root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler with structlog formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Use structlog's ProcessorFormatter for nice console output
    # ConsoleRenderer handles exception formatting automatically
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(formatter)

    # Add console handler to root logger
    root_logger.addHandler(console_handler)


logger = structlog.get_logger(__name__)


def is_coordinator_initialized() -> bool:
    """Check if the coordinator has been initialized.

    Used by API lifespan to avoid redundant initialization when running
    in combined mode (start_both.py) where initialize_coordinator() is
    called before the API server starts.

    Returns:
        True if initialize_coordinator() has completed successfully.
    """
    return _initialized


def get_mcp_registry():
    """Get the MCP server registry if initialized.

    Returns:
        The ServerRegistry instance, or None if not yet initialized.
    """
    return _registry


async def initialize_coordinator() -> None:
    """Initialize the coordinator and all components.

    Uses an async lock to prevent race conditions if multiple startup
    events occur concurrently (e.g., multiple Slack reconnects).
    """
    global _initialized, _registry, _catalog, _health_monitor, _bulkhead, _deadletter_queue

    async with _get_init_lock():
        if _initialized:
            logger.debug("Coordinator already initialized, skipping")
            return

        logger.info("Initializing OCI AI Agent Coordinator")

        # Initialize observability first (tracing + logging)
        # This sets up the tracer provider for all components
        init_observability(agent_name="coordinator")

        # Initialize OCI Logging handlers for ALL agents
        # Each agent gets its own log stream for filtering in OCI Logging
        from src.observability.oci_logging import init_oci_logging
        agent_names = [
            "db-troubleshoot-agent",
            "log-analytics-agent",
            "security-threat-agent",
            "finops-agent",
            "infrastructure-agent",
            "slack-handler",
        ]
        for agent_name in agent_names:
            handler = init_oci_logging(agent_name=agent_name)
            if handler:
                logger.debug(f"OCI Logging initialized for {agent_name}")

        logger.info("Observability initialized", agents=len(agent_names) + 1)

        # Start OCA callback server for OAuth authentication
        # This handles redirects from Oracle SSO when users log in via Slack
        try:
            from src.llm.oca_callback_server import start_callback_server

            # Callback to notify Slack users when auth succeeds
            def on_token_received(token: dict) -> None:
                """Notify Slack users when OCA authentication succeeds."""
                try:
                    from src.channels.slack import notify_auth_success
                    notify_auth_success()
                    logger.info("Sent auth success notification to Slack")
                except Exception as e:
                    logger.warning("Failed to send auth notification", error=str(e))

            await start_callback_server(on_token_received=on_token_received)
            logger.info("OCA callback server started (OAuth redirects enabled)")
        except Exception as e:
            logger.warning("OCA callback server failed to start", error=str(e))

        # Initialize OCI discovery service for compartment caching
        try:
            from src.oci.discovery import initialize_discovery

            discovery = await initialize_discovery()
            status = await discovery.get_status()
            logger.info(
                "OCI discovery initialized",
                tenancies=status["tenancies"],
                compartments=status["cached_compartments"],
            )
        except Exception as e:
            logger.warning("OCI discovery initialization failed", error=str(e))

        # Initialize MCP infrastructure
        config = load_mcp_config()
        enabled_servers = config.get_enabled_servers()
        logger.info("MCP configuration loaded", servers=list(enabled_servers.keys()))

        if enabled_servers:
            try:
                registry, catalog = await initialize_mcp_from_config(config)
                _registry = registry
                _catalog = catalog

                # Connect registry events to catalog for dynamic updates
                def on_registry_event(event_type: str, server_id: str, data: dict):
                    """Handle registry events to keep catalog in sync."""
                    if event_type in (
                        EVENT_SERVER_CONNECTED,
                        EVENT_SERVER_DISCONNECTED,
                        EVENT_TOOLS_UPDATED,
                    ):
                        # Invalidate catalog cache - tools will refresh on next access
                        registry.invalidate_tool_cache()
                        logger.debug(
                            "Catalog invalidated due to registry event",
                            event=event_type,
                            server=server_id,
                        )

                registry.on_event(on_registry_event)

                # Start background health check loop
                await registry.start_health_checks(interval_seconds=30)
                logger.info("MCP health check loop started")

                # Count connected servers
                connected = sum(
                    1 for server_id in registry.list_servers()
                    if registry.get_status(server_id) == "connected"
                )
                logger.info(
                    "MCP servers connected",
                    connected=connected,
                    total=len(registry.list_servers()),
                    tools=len(catalog.list_tools()),
                )

                # Optional validation (disable with COORDINATOR_VALIDATE_* env vars)
                validate_catalog = os.getenv(
                    "COORDINATOR_VALIDATE_CATALOG", "true"
                ).lower() in ("true", "1", "yes", "on")
                validate_manifest = os.getenv(
                    "COORDINATOR_VALIDATE_MANIFEST", "true"
                ).lower() in ("true", "1", "yes", "on")

                if validate_catalog:
                    try:
                        from src.mcp.validation import validate_tool_catalog

                        result = await validate_tool_catalog(
                            catalog, registry, include_health_check=False
                        )
                        logger.info(
                            "Tool catalog validation complete",
                            valid=result.valid,
                            errors=result.by_severity.get("error", 0),
                            warnings=result.by_severity.get("warning", 0),
                        )
                    except Exception as e:
                        logger.warning(
                            "Tool catalog validation failed",
                            error=str(e),
                        )

                if validate_manifest:
                    try:
                        from src.mcp.validation import validate_server_manifests

                        manifest_result = await validate_server_manifests(registry)
                        logger.info("Manifest validation complete", **manifest_result)
                    except Exception as e:
                        logger.warning(
                            "Manifest validation failed",
                            error=str(e),
                        )
            except Exception as e:
                logger.warning("MCP initialization failed", error=str(e))

        # ─────────────────────────────────────────────────────────────────────
        # Initialize Resilience Infrastructure
        # ─────────────────────────────────────────────────────────────────────
        try:
            # Initialize Bulkhead for resource isolation
            _bulkhead = Bulkhead.get_instance()
            logger.info("Bulkhead initialized", partitions=list(_bulkhead._partitions.keys()))

            # Initialize Deadletter Queue for failed operations
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            _deadletter_queue = DeadLetterQueue(redis_url=redis_url)
            logger.info("Deadletter queue initialized")

            # Connect resilience to catalog
            if _catalog:
                _catalog._deadletter_queue = _deadletter_queue
                _catalog._bulkhead = _bulkhead

            # Initialize Health Monitor with auto-restart
            _health_monitor = HealthMonitor.get_instance()

            # Register MCP server health checks
            if _registry:
                for server_id in _registry.list_servers():
                    # Create health check for each MCP server
                    async def create_mcp_check(sid: str):
                        async def check_mcp() -> HealthCheckResult:
                            try:
                                client = _registry.get_client(sid)
                                if client:
                                    # Quick tool list as health check
                                    tools = client.list_tools() if hasattr(client, 'list_tools') else []
                                    return HealthCheckResult(
                                        status=HealthStatus.HEALTHY,
                                        details={"tool_count": len(tools)},
                                    )
                                return HealthCheckResult(
                                    status=HealthStatus.UNHEALTHY,
                                    message="No client available",
                                )
                            except Exception as e:
                                return HealthCheckResult(
                                    status=HealthStatus.UNHEALTHY,
                                    message=str(e),
                                )
                        return check_mcp

                    async def create_mcp_restart(sid: str):
                        async def restart_mcp() -> bool:
                            try:
                                await _registry.disconnect(sid)
                                await asyncio.sleep(2)
                                await _registry.connect(sid)
                                return True
                            except Exception:
                                return False
                        return restart_mcp

                    check_func = await create_mcp_check(server_id)
                    restart_func = await create_mcp_restart(server_id)

                    _health_monitor.register_check(HealthCheck(
                        name=f"mcp_{server_id}",
                        check_func=check_func,
                        restart_func=restart_func,
                        interval_seconds=60.0,
                        failure_threshold=3,
                        critical=True,
                    ))

            # Start health monitoring
            await _health_monitor.start()
            logger.info(
                "Health monitor started",
                checks=list(_health_monitor._checks.keys()),
            )

        except Exception as e:
            logger.warning("Resilience initialization failed", error=str(e))

        # Initialize agent catalog with auto-discovery
        agent_catalog = AgentCatalog.get_instance(tool_catalog=_catalog)
        agent_catalog.auto_discover()
        if _catalog:
            agent_catalog.sync_mcp_tools(_catalog)
        agents = agent_catalog.list_all()
        logger.info("Agents discovered", count=len(agents))

        # Pre-warm LangGraph coordinator to eliminate first-request latency
        # This creates and compiles the graph so the first Slack message is fast
        await prewarm_coordinator()

        # Initialize ShowOCI cache if enabled
        if os.getenv("SHOWOCI_CACHE_ENABLED", "false").lower() == "true":
            try:
                from src.showoci.cache_loader import ShowOCICacheLoader

                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                profiles = os.getenv("OCI_PROFILES", "DEFAULT").split(",")
                profiles = [p.strip() for p in profiles if p.strip()]

                cache_loader = ShowOCICacheLoader(
                    redis_url=redis_url,
                    profiles=profiles,
                )

                # Run initial cache load in background
                asyncio.create_task(_load_showoci_cache(cache_loader))

                # Start periodic refresh scheduler if configured
                refresh_interval = float(os.getenv("SHOWOCI_REFRESH_HOURS", "4"))
                if refresh_interval > 0:
                    await cache_loader.start_scheduler(interval_hours=refresh_interval)
                    logger.info(
                        "ShowOCI cache scheduler started",
                        interval_hours=refresh_interval,
                        profiles=profiles,
                    )
            except Exception as e:
                logger.warning("ShowOCI cache initialization failed", error=str(e))

        # Mark initialization complete
        _initialized = True
        logger.info("Coordinator initialization complete")


async def _load_showoci_cache(cache_loader) -> None:
    """Background task to load ShowOCI cache on startup."""
    try:
        logger = structlog.get_logger(__name__)
        logger.info("Starting initial ShowOCI cache load")
        result = await cache_loader.run_full_load()
        logger.info(
            "ShowOCI cache loaded",
            profiles_loaded=result.get("profiles_loaded", 0),
            total_resources=result.get("total_resources", {}),
        )
    except Exception as e:
        logger = structlog.get_logger(__name__)
        logger.error("ShowOCI cache load failed", error=str(e))


async def _check_oca_auth_status(log) -> None:
    """Check OCA authentication status and prompt for re-auth if needed.

    This is called at startup when LLM_PROVIDER is set to oca/oracle_code_assist.
    If the token is expired and cannot be refreshed, it prints clear instructions
    for the user to re-authenticate.
    """
    try:
        from src.llm.oca import oca_token_manager

        token_info = oca_token_manager.get_token_info()

        if not token_info.get("has_token"):
            log.warning(
                "OCA not authenticated - no token found",
                hint="Run: poetry run python scripts/oca_login.py",
            )
            _print_oca_auth_instructions("No OCA token found")
            return

        if token_info.get("is_valid"):
            # Token is valid, nothing to do
            expires_in = token_info.get("expires_in_seconds", 0)
            log.info(
                "OCA token valid",
                expires_in_minutes=round(expires_in / 60, 1),
            )
            return

        # Token expired - check if we can refresh
        if token_info.get("can_refresh"):
            log.info("OCA token expired, attempting refresh...")
            new_token = await oca_token_manager.refresh_token()
            if new_token:
                log.info("OCA token refreshed successfully")
                return
            else:
                log.warning("OCA token refresh failed")

        # Cannot refresh - prompt for re-authentication
        log.error(
            "OCA token expired and cannot be refreshed",
            hint="Run: poetry run python scripts/oca_login.py",
        )
        _print_oca_auth_instructions("OCA token expired")

    except ImportError:
        log.warning("OCA module not available")
    except Exception as e:
        log.warning("OCA auth check failed", error=str(e))


def _print_oca_auth_instructions(reason: str) -> None:
    """Print clear instructions for OCA re-authentication."""
    print("\n" + "=" * 70)
    print("  ORACLE CODE ASSIST AUTHENTICATION REQUIRED")
    print("=" * 70)
    print(f"\n  Reason: {reason}")
    print("\n  To authenticate, run:")
    print("\n    poetry run python scripts/oca_login.py")
    print("\n  This will open a browser for OAuth login.")
    print("  After authentication, restart the service.")
    print("\n" + "=" * 70 + "\n")


async def prewarm_coordinator() -> None:
    """Pre-warm the LangGraph coordinator to eliminate first-request latency.

    Creates and initializes the LangGraph coordinator at startup so the first
    Slack message doesn't incur initialization overhead (5-15s saved).

    The coordinator is stored in the global _coordinator variable and can be
    accessed via get_coordinator() by Slack handlers and other consumers.

    LLM Fallback:
        Uses automatic fallback if configured provider is unavailable.
        Priority order (configurable via LLM_PROVIDER_PRIORITY):
        1. lm_studio (local)
        2. ollama (local)
        3. oca (Oracle Code Assist)
        4. oci_genai (OCI GenAI)
        5. anthropic
        6. openai
    """
    global _coordinator

    import time

    start_time = time.time()
    log = structlog.get_logger(__name__)
    log.info("Pre-warming LangGraph coordinator...")

    try:
        from src.agents.catalog import AgentCatalog
        from src.agents.coordinator.graph import LangGraphCoordinator
        from src.agents.coordinator.workflows import get_workflow_registry
        from src.llm import get_llm_with_auto_fallback, print_llm_availability_report
        from src.mcp.catalog import ToolCatalog
        from src.memory.manager import SharedMemoryManager

        # Check OCA authentication if configured
        llm_provider = os.getenv("LLM_PROVIDER", "").lower()
        if llm_provider in ("oca", "oracle_code_assist"):
            await _check_oca_auth_status(log)

        # Print LLM availability report at startup
        log.info("Checking LLM provider availability...")
        await print_llm_availability_report(timeout=5.0)

        # Get LLM with automatic fallback if configured provider is unavailable
        # Priority: local LLMs first (lowest latency), then cloud providers
        try:
            llm = await get_llm_with_auto_fallback()
        except RuntimeError as e:
            log.error(
                "No LLM providers available - coordinator will fail on first request",
                error=str(e),
            )
            # Fall back to non-fallback version for graceful degradation
            from src.llm import get_llm
            llm = get_llm()

        # Use existing singleton catalogs initialized by MCPConnectionManager
        tool_catalog = ToolCatalog.get_instance()
        agent_catalog = AgentCatalog.get_instance()

        if not tool_catalog:
            log.warning("Tool catalog not available for pre-warm, skipping")
            return

        # Initialize memory manager
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        memory = SharedMemoryManager(redis_url=redis_url)

        # Load workflow registry
        workflow_registry = get_workflow_registry()

        # Create and initialize coordinator
        _coordinator = LangGraphCoordinator(
            llm=llm,
            tool_catalog=tool_catalog,
            agent_catalog=agent_catalog,
            memory=memory,
            workflow_registry=workflow_registry,
            max_iterations=10,
        )

        # Initialize the graph - this is the expensive operation we're pre-warming
        await _coordinator.initialize()

        duration_ms = int((time.time() - start_time) * 1000)
        log.info(
            "LangGraph coordinator pre-warmed",
            duration_ms=duration_ms,
            tool_count=len(tool_catalog.list_tools()) if tool_catalog else 0,
            workflow_count=len(workflow_registry),
            agent_count=len(agent_catalog.list_all()) if agent_catalog else 0,
        )

    except Exception as e:
        log.warning("Coordinator pre-warm failed (will lazy-init on first request)", error=str(e))


def get_coordinator():
    """Get the pre-warmed LangGraph coordinator.

    Returns:
        LangGraphCoordinator if pre-warmed, None otherwise.

    Usage:
        from src.main import get_coordinator
        coordinator = get_coordinator()
        if coordinator:
            result = await coordinator.invoke(query, thread_id, user_id)
    """
    return _coordinator


def run_slack_mode_sync() -> None:
    """Run Slack handler in blocking sync mode.

    Only use this for standalone Slack-only mode.
    For concurrent execution with API, use run_slack_mode_async().
    """
    from src.channels.slack import create_slack_app

    app_token = os.getenv("SLACK_APP_TOKEN")
    slack_handler = create_slack_app()

    logger.info("Starting Slack handler in blocking sync mode")
    slack_handler.start(socket_mode=bool(app_token))


async def run_slack_mode_async() -> None:
    """Run Slack handler in async mode for concurrent execution.

    Uses AsyncSocketModeHandler which integrates properly with the
    asyncio event loop, avoiding BrokenPipeError when running alongside
    the API server.
    """
    from src.channels.slack import create_slack_app

    slack_handler = create_slack_app()

    logger.info("Starting Slack handler in async mode")
    await slack_handler.start_async()


async def run_slack_mode(blocking: bool = True) -> None:
    """Run in Slack integration mode.

    Args:
        blocking: If True, runs in blocking sync mode (standalone).
                  If False, runs in async mode for concurrent execution with API.
    """
    logger.info("Starting Slack integration mode", blocking=blocking, mode="sync" if blocking else "async")

    # Verify Slack tokens are configured
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    if not bot_token:
        logger.error("SLACK_BOT_TOKEN not configured")
        raise RuntimeError("SLACK_BOT_TOKEN not configured - cannot start Slack mode")

    if not app_token:
        logger.warning("SLACK_APP_TOKEN not configured - Socket Mode disabled")
        return

    if blocking:
        # Handle shutdown gracefully for standalone mode
        loop = asyncio.get_event_loop()

        def handle_shutdown(sig):
            logger.info("Shutdown signal received", signal=sig.name)
            shutdown_observability()
            loop.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, handle_shutdown, sig)

        # Run in blocking sync mode (standalone)
        run_slack_mode_sync()
    else:
        # Run Slack in async mode - shares event loop with API server
        # This avoids BrokenPipeError from competing event loops
        await run_slack_mode_async()


async def run_api_mode(port: int = 3001) -> None:
    """Run in API server mode."""
    import uvicorn

    from src.api.main import app

    logger.info("Starting API server mode", port=port)

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OCI AI Agent Coordinator",
    )
    parser.add_argument(
        "--mode",
        choices=["slack", "api", "both"],
        default="both",
        help="Run mode: slack, api, or both (default: both)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3001,
        help="API server port (for api or both mode)",
    )

    args = parser.parse_args()

    # Configure logging FIRST - routes structlog through Python logging for OCI integration
    configure_logging()

    async def run():
        await initialize_coordinator()

        if args.mode == "slack":
            # Use async Socket Mode even in Slack-only mode to keep a single event loop.
            await run_slack_mode(blocking=False)
        elif args.mode == "api":
            await run_api_mode(args.port)
        elif args.mode == "both":
            # Run both concurrently using async Slack handler
            logger.info("Starting both Slack and API modes concurrently", port=args.port)
            await asyncio.gather(
                run_slack_mode(blocking=False),
                run_api_mode(args.port),
            )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        _cleanup()
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        _cleanup()
        sys.exit(1)


async def _async_cleanup() -> None:
    """Async cleanup for resources that need await."""
    global _registry

    # Stop MCP health checks and disconnect servers
    if _registry:
        try:
            await _registry.stop_health_checks()
            await _registry.disconnect_all()
            logger.info("MCP servers disconnected")
        except Exception as e:
            logger.warning("MCP cleanup error", error=str(e))


def _cleanup() -> None:
    """Clean up resources on shutdown."""
    # Run async cleanup
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_async_cleanup())
        else:
            loop.run_until_complete(_async_cleanup())
    except Exception:
        pass

    # Stop OCA callback server
    try:
        from src.llm.oca_callback_server import stop_callback_server
        stop_callback_server()
    except Exception:
        pass

    # Shutdown observability
    shutdown_observability()


if __name__ == "__main__":
    main()
