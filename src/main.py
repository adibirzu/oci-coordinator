#!/usr/bin/env python
"""OCI AI Agent Coordinator - Main Entry Point.

Starts the coordinator with Slack integration and observability.

Usage:
    # Load environment from .env.local
    poetry run python -m src.main

    # Or with explicit mode
    poetry run python -m src.main --mode slack
    poetry run python -m src.main --mode api
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
from dotenv import load_dotenv

env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file)

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

# Global references for cleanup
_registry: ServerRegistry | None = None
_catalog: ToolCatalog | None = None


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


async def initialize_coordinator() -> None:
    """Initialize the coordinator and all components."""
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

        await start_callback_server()
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
    global _registry, _catalog
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
        except Exception as e:
            logger.warning("MCP initialization failed", error=str(e))

    # Initialize agent catalog with auto-discovery
    agent_catalog = AgentCatalog.get_instance()
    agent_catalog.auto_discover()
    agents = agent_catalog.list_all()
    logger.info("Agents discovered", count=len(agents))

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


async def run_slack_mode() -> None:
    """Run in Slack integration mode."""
    from src.channels.slack import create_slack_app

    logger.info("Starting Slack integration mode")

    # Verify Slack tokens are configured
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    if not bot_token:
        logger.error("SLACK_BOT_TOKEN not configured")
        sys.exit(1)

    if not app_token:
        logger.warning("SLACK_APP_TOKEN not configured - Socket Mode disabled")

    # Create and start Slack handler
    slack_handler = create_slack_app()

    # Handle shutdown gracefully
    loop = asyncio.get_event_loop()

    def handle_shutdown(sig):
        logger.info("Shutdown signal received", signal=sig.name)
        shutdown_observability()
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown, sig)

    logger.info(
        "Starting Slack handler",
        socket_mode=bool(app_token),
    )

    # Start in blocking mode
    slack_handler.start(socket_mode=bool(app_token))


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
        default="slack",
        help="Run mode: slack, api, or both",
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
            await run_slack_mode()
        elif args.mode == "api":
            await run_api_mode(args.port)
        elif args.mode == "both":
            # Run both concurrently
            await asyncio.gather(
                run_slack_mode(),
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
