"""
Async Runtime for Channel Integrations.

Provides a shared event loop for Slack and other channel handlers,
avoiding the need to create new event loops per event which causes
MCP connection issues.

Usage:
    from src.channels.async_runtime import AsyncRuntime

    runtime = AsyncRuntime.get_instance()
    result = runtime.run_coroutine(my_async_function())
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class AsyncRuntime:
    """
    Shared async runtime for channel integrations.

    Maintains a single event loop in a background thread,
    allowing sync code to run coroutines without creating
    new event loops.

    This fixes issues where:
    - Slack Bolt creates new event loops per handler
    - MCP connections are bound to specific event loops
    - asyncio.run() fails in nested contexts
    """

    _instance: AsyncRuntime | None = None
    _lock = threading.RLock()  # RLock allows reentrant acquisition (get_instance -> start)

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._started = False
        self._logger = logger.bind(component="AsyncRuntime")

    @classmethod
    def get_instance(cls) -> AsyncRuntime:
        """Get the singleton instance, creating if necessary."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance.start()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        if cls._instance is not None:
            cls._instance.stop()
            cls._instance = None

    def start(self) -> None:
        """Start the background event loop."""
        if self._started:
            return

        with self._lock:
            if self._started:
                return

            # Create a new event loop
            self._loop = asyncio.new_event_loop()

            # Start the loop in a background thread
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="async-runtime-loop",
            )
            self._thread.start()
            self._started = True

            self._logger.info("Async runtime started")

    def _run_loop(self) -> None:
        """Run the event loop forever in the background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def stop(self) -> None:
        """Stop the background event loop."""
        if not self._started or not self._loop:
            return

        with self._lock:
            if not self._started:
                return

            # Stop the loop
            self._loop.call_soon_threadsafe(self._loop.stop)

            # Wait for thread to finish
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)

            # Cleanup
            self._loop.close()
            self._loop = None
            self._thread = None
            self._started = False

            self._logger.info("Async runtime stopped")

    def run_coroutine(self, coro: Coroutine[Any, Any, T], timeout: float = 300) -> T:
        """
        Run a coroutine in the shared event loop.

        This is the main entry point for sync code to run async code.

        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds (default 300 = 5 minutes)

        Returns:
            Result of the coroutine

        Raises:
            RuntimeError: If the runtime is not started
            TimeoutError: If the coroutine times out
            Exception: Any exception raised by the coroutine
        """
        if not self._started or not self._loop:
            raise RuntimeError("Async runtime not started")

        # Submit the coroutine to the event loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        # Wait for result with timeout
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            # Cancel the future and log
            future.cancel()
            self._logger.warning(
                "Coroutine timed out",
                timeout=timeout,
            )
            raise TimeoutError(f"Coroutine timed out after {timeout}s")
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            self._logger.warning("Coroutine was cancelled")
            raise TimeoutError("Coroutine was cancelled")
        except Exception as e:
            # Cancel if still running and re-raise
            if not future.done():
                future.cancel()
            # Log the error
            self._logger.error(
                "Coroutine failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def run_coroutine_nowait(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """
        Run a coroutine without waiting for the result.

        Returns a Future that can be used to get the result later.

        Args:
            coro: Coroutine to execute

        Returns:
            Future for the result
        """
        if not self._started or not self._loop:
            raise RuntimeError("Async runtime not started")

        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def create_task(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """
        Create a task in the shared event loop.

        Args:
            coro: Coroutine to execute

        Returns:
            Async task
        """
        if not self._started or not self._loop:
            raise RuntimeError("Async runtime not started")

        return self._loop.create_task(coro)

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the event loop (use with caution)."""
        return self._loop

    @property
    def is_running(self) -> bool:
        """Check if the runtime is running."""
        return self._started and self._loop is not None and self._loop.is_running()


# Convenience functions
def run_async(coro: Coroutine[Any, Any, T], timeout: float = 300) -> T:
    """
    Run an async function from sync code.

    Uses the shared AsyncRuntime to execute the coroutine.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds (default 300 = 5 minutes)

    Returns:
        Result of the coroutine

    Raises:
        TimeoutError: If the coroutine times out or is cancelled
        RuntimeError: If the runtime cannot be started
    """
    runtime = AsyncRuntime.get_instance()

    # Ensure runtime is started
    if not runtime.is_running:
        logger.warning("AsyncRuntime not running, attempting to start...")
        runtime.start()
        # Wait for loop to be ready
        import time
        for _ in range(10):
            if runtime.is_running:
                break
            time.sleep(0.1)

    if not runtime.is_running:
        raise RuntimeError("Failed to start AsyncRuntime")

    try:
        return runtime.run_coroutine(coro, timeout=timeout)
    except TimeoutError:
        # Re-raise with more context
        raise
    except asyncio.CancelledError:
        # Convert to TimeoutError for consistent handling
        raise TimeoutError("Request was cancelled")


def get_shared_loop() -> asyncio.AbstractEventLoop:
    """
    Get the shared event loop.

    Use this when you need to schedule work on the shared loop.

    Returns:
        The shared event loop

    Raises:
        RuntimeError: If the runtime is not started
    """
    runtime = AsyncRuntime.get_instance()
    if not runtime.loop:
        raise RuntimeError("Async runtime not started")
    return runtime.loop
