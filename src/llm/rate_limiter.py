"""
LLM Rate Limiter.

Provides concurrency control for LLM calls to prevent overwhelming
the provider when multiple users query simultaneously.

Features:
- Semaphore-based concurrency limiting
- Configurable max concurrent calls
- Queue-based waiting with timeout
- Metrics tracking for monitoring
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableConfig
from pydantic import PrivateAttr

logger = structlog.get_logger(__name__)


@dataclass
class RateLimiterMetrics:
    """Metrics for rate limiter monitoring."""

    total_calls: int = 0
    queued_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    total_wait_time_ms: float = 0
    total_execution_time_ms: float = 0
    peak_queue_size: int = 0
    timeouts: int = 0

    @property
    def avg_wait_time_ms(self) -> float:
        """Average time spent waiting for semaphore."""
        if self.completed_calls == 0:
            return 0
        return self.total_wait_time_ms / self.completed_calls

    @property
    def avg_execution_time_ms(self) -> float:
        """Average LLM execution time."""
        if self.completed_calls == 0:
            return 0
        return self.total_execution_time_ms / self.completed_calls

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_calls": self.total_calls,
            "queued_calls": self.queued_calls,
            "completed_calls": self.completed_calls,
            "failed_calls": self.failed_calls,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "peak_queue_size": self.peak_queue_size,
            "timeouts": self.timeouts,
        }


class RateLimitedLLM(BaseChatModel):
    """
    Wrapper around BaseChatModel that limits concurrent calls.

    Uses asyncio.Semaphore to control concurrency and prevent
    overwhelming the LLM provider during high load.

    Example:
        llm = ChatAnthropic(...)
        rate_limited_llm = RateLimitedLLM(llm, max_concurrent=5)
        response = await rate_limited_llm.ainvoke(messages)
    """

    # Pydantic private attributes (must use PrivateAttr for BaseChatModel compatibility)
    _llm: BaseChatModel = PrivateAttr()
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _max_concurrent: int = PrivateAttr(default=5)
    _timeout_seconds: float = PrivateAttr(default=300.0)
    _metrics: RateLimiterMetrics = PrivateAttr()
    _current_queue_size: int = PrivateAttr(default=0)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        llm: BaseChatModel,
        max_concurrent: int = 5,
        timeout_seconds: float = 300.0,
        **kwargs: Any,
    ):
        """Initialize rate limited LLM wrapper.

        Args:
            llm: The underlying LLM to wrap
            max_concurrent: Maximum concurrent calls (default 5)
            timeout_seconds: Maximum wait time for semaphore (default 300s)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._max_concurrent = max_concurrent
        self._timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._metrics = RateLimiterMetrics()
        self._current_queue_size = 0

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return f"rate_limited_{self._llm._llm_type}"

    @property
    def metrics(self) -> RateLimiterMetrics:
        """Get rate limiter metrics."""
        return self._metrics

    @property
    def current_queue_size(self) -> int:
        """Get current queue size (waiting calls)."""
        return self._current_queue_size

    @property
    def available_slots(self) -> int:
        """Get available concurrent slots."""
        return self._max_concurrent - (self._max_concurrent - self._semaphore._value)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation (passthrough to underlying LLM)."""
        return self._llm._generate(messages, stop, run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation with rate limiting."""
        self._metrics.total_calls += 1
        self._current_queue_size += 1
        self._metrics.queued_calls = self._current_queue_size
        self._metrics.peak_queue_size = max(
            self._metrics.peak_queue_size, self._current_queue_size
        )

        wait_start = time.perf_counter()

        try:
            # Wait for semaphore with timeout
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self._timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._metrics.timeouts += 1
                self._current_queue_size -= 1
                logger.warning(
                    "Rate limiter timeout",
                    timeout=self._timeout_seconds,
                    queue_size=self._current_queue_size,
                )
                raise TimeoutError(
                    f"LLM rate limit timeout after {self._timeout_seconds}s. "
                    f"Queue size: {self._current_queue_size}"
                )

            wait_end = time.perf_counter()
            self._metrics.total_wait_time_ms += (wait_end - wait_start) * 1000
            self._current_queue_size -= 1

            try:
                exec_start = time.perf_counter()
                result = await self._llm._agenerate(messages, stop, run_manager, **kwargs)
                exec_end = time.perf_counter()

                self._metrics.total_execution_time_ms += (exec_end - exec_start) * 1000
                self._metrics.completed_calls += 1

                return result

            except Exception as e:
                self._metrics.failed_calls += 1
                raise

        finally:
            self._semaphore.release()

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Async invoke with rate limiting."""
        # Delegate to parent which calls _agenerate
        return await super().ainvoke(input, config, **kwargs)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Bind tools to the underlying LLM."""
        bound = self._llm.bind_tools(tools, **kwargs)
        # Wrap the bound LLM with rate limiting
        return RateLimitedLLM(
            bound,
            max_concurrent=self._max_concurrent,
            timeout_seconds=self._timeout_seconds,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> BaseChatModel:
        """Get structured output from the underlying LLM."""
        structured = self._llm.with_structured_output(schema, **kwargs)
        return RateLimitedLLM(
            structured,
            max_concurrent=self._max_concurrent,
            timeout_seconds=self._timeout_seconds,
        )


# Global rate limiter instance (singleton pattern)
_global_rate_limiter: RateLimitedLLM | None = None


def wrap_with_rate_limiter(
    llm: BaseChatModel,
    max_concurrent: int = 5,
    timeout_seconds: float = 300.0,
) -> RateLimitedLLM:
    """
    Wrap an LLM with rate limiting.

    Args:
        llm: The LLM to wrap
        max_concurrent: Maximum concurrent calls
        timeout_seconds: Maximum wait time

    Returns:
        Rate limited LLM wrapper
    """
    return RateLimitedLLM(
        llm,
        max_concurrent=max_concurrent,
        timeout_seconds=timeout_seconds,
    )


def get_rate_limiter_metrics() -> dict[str, Any] | None:
    """Get global rate limiter metrics if available."""
    if _global_rate_limiter:
        return _global_rate_limiter.metrics.to_dict()
    return None
