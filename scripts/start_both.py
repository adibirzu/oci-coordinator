#!/usr/bin/env python
"""OCI AI Agent Coordinator - Combined Slack + API Startup.

This script starts both Slack Socket Mode and the API server concurrently.
It works around a subtle asyncio interaction issue in main.py by creating
uvicorn directly instead of going through run_api_mode().

Usage:
    poetry run python scripts/start_both.py
    # or from project root:
    cd /path/to/oci-coordinator && poetry run python scripts/start_both.py
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Track shutdown state
_shutdown_event: asyncio.Event | None = None
_tasks: list[asyncio.Task] = []


async def run_slack():
    """Start Slack Socket Mode handler."""
    print("Slack: Starting...")
    from src.main import run_slack_mode
    try:
        await run_slack_mode(blocking=False)
        print("Slack: Done (should not reach here normally)")
    except asyncio.CancelledError:
        print("Slack: Shutting down gracefully...")
        raise


async def run_api():
    """Start uvicorn API server directly."""
    print("API: Starting uvicorn...")
    import uvicorn
    from src.api.main import app

    config = uvicorn.Config(app, host="0.0.0.0", port=3001, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
        print("API: Done (should not reach here normally)")
    except asyncio.CancelledError:
        print("API: Shutting down gracefully...")
        raise


async def graceful_shutdown(loop: asyncio.AbstractEventLoop) -> None:
    """Handle graceful shutdown of all services."""
    global _tasks

    print("\nInitiating graceful shutdown...")

    # Cancel all running tasks
    for task in _tasks:
        if not task.done():
            task.cancel()

    # Wait for tasks to complete with timeout
    if _tasks:
        print("Waiting for services to stop...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*_tasks, return_exceptions=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            print("Warning: Shutdown timed out, forcing exit")

    # Run cleanup
    try:
        from src.main import _cleanup
        _cleanup()
    except Exception as e:
        print(f"Cleanup error (non-fatal): {e}")

    print("Shutdown complete")


def handle_signal(sig: signal.Signals, loop: asyncio.AbstractEventLoop) -> None:
    """Handle shutdown signals."""
    print(f"\nReceived signal {sig.name}")
    # Schedule graceful shutdown
    loop.create_task(graceful_shutdown(loop))


async def main():
    """Initialize and start both services."""
    global _tasks

    # Get event loop for signal handling
    loop = asyncio.get_running_loop()

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s, loop))

    print("Step 1: Importing initialize_coordinator...")
    from src.main import initialize_coordinator

    print("Step 2: Running initialize_coordinator...")
    await initialize_coordinator()
    print("Step 3: Initialize completed")

    print("Step 4: Starting both modes with asyncio.gather...")

    # Create tasks for better control
    slack_task = asyncio.create_task(run_slack(), name="slack")
    api_task = asyncio.create_task(run_api(), name="api")
    _tasks = [slack_task, api_task]

    try:
        # Wait for either task to complete (or fail)
        done, pending = await asyncio.wait(
            _tasks,
            return_when=asyncio.FIRST_EXCEPTION
        )

        # Check if any task raised an exception
        for task in done:
            if task.exception() is not None:
                exc = task.exception()
                print(f"Task {task.get_name()} failed: {exc}")
                # Cancel remaining tasks
                for p in pending:
                    p.cancel()
                raise exc

    except asyncio.CancelledError:
        print("Main task cancelled")
    except Exception as e:
        print(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested via keyboard interrupt")
    except SystemExit:
        # Don't log SystemExit as an error - it's expected during shutdown
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
