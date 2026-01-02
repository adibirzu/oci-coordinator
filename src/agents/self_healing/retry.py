"""
Retry Strategy for Self-Healing Agents.

Implements intelligent retry logic that:
1. Decides whether to retry based on error analysis
2. Applies exponential backoff with jitter
3. Modifies parameters between retries
4. Tracks retry patterns for learning
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, TypeVar, ParamSpec

import structlog

from src.agents.self_healing.analyzer import ErrorAnalysis, RecoveryAction

if TYPE_CHECKING:
    from src.agents.self_healing.corrector import ParameterCorrector

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RetryOutcome(str, Enum):
    """Outcome of a retry attempt."""

    SUCCESS = "success"  # Retry succeeded
    FAILED = "failed"  # All retries exhausted
    ABORTED = "aborted"  # Decided not to retry
    SKIPPED = "skipped"  # Error analysis said to skip


@dataclass
class RetryDecision:
    """Decision on whether and how to retry."""

    should_retry: bool
    wait_seconds: float = 0.0
    modified_params: dict[str, Any] | None = None
    reason: str = ""
    attempt: int = 0
    max_attempts: int = 3
    fallback_tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "should_retry": self.should_retry,
            "wait_seconds": self.wait_seconds,
            "modified_params": self.modified_params,
            "reason": self.reason,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "fallback_tool": self.fallback_tool,
        }


@dataclass
class RetryRecord:
    """Record of a retry attempt."""

    tool_name: str
    attempt: int
    success: bool
    error: str | None
    wait_time: float
    params_modified: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RetryStrategy:
    """
    Intelligent retry strategy with learning capabilities.

    Features:
    - Exponential backoff with jitter
    - Parameter modification between retries
    - Error-aware retry decisions
    - Success/failure tracking
    - Adaptive max retries based on error type
    """

    # Default retry settings by error type
    DEFAULT_SETTINGS = {
        "timeout": {"max_retries": 3, "base_delay": 2.0, "max_delay": 30.0},
        "rate_limit": {"max_retries": 5, "base_delay": 5.0, "max_delay": 60.0},
        "network_error": {"max_retries": 3, "base_delay": 1.0, "max_delay": 15.0},
        "service_error": {"max_retries": 2, "base_delay": 3.0, "max_delay": 20.0},
        "parameter_error": {"max_retries": 2, "base_delay": 0.5, "max_delay": 5.0},
        "default": {"max_retries": 2, "base_delay": 1.0, "max_delay": 10.0},
    }

    def __init__(
        self,
        corrector: "ParameterCorrector | None" = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: float = 0.1,
    ):
        """
        Initialize retry strategy.

        Args:
            corrector: Parameter corrector for retry modifications
            max_retries: Default maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Random jitter factor (0-1)
        """
        self.corrector = corrector
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._retry_history: list[RetryRecord] = []
        self._tool_success_rates: dict[str, dict[str, int]] = {}
        self._logger = logger.bind(component="RetryStrategy")

    def should_retry(
        self,
        analysis: ErrorAnalysis,
        attempt: int,
        tool_name: str | None = None,
    ) -> RetryDecision:
        """
        Decide whether to retry based on error analysis.

        Args:
            analysis: Error analysis result
            attempt: Current attempt number (0-indexed)
            tool_name: Name of the tool that failed

        Returns:
            RetryDecision with retry instructions
        """
        # Get settings for this error type
        settings = self.DEFAULT_SETTINGS.get(
            analysis.category.value,
            self.DEFAULT_SETTINGS["default"],
        )
        max_retries = min(analysis.max_retries, settings["max_retries"])

        # Check if we've exceeded max retries
        if attempt >= max_retries:
            return RetryDecision(
                should_retry=False,
                reason=f"Exceeded max retries ({max_retries})",
                attempt=attempt,
                max_attempts=max_retries,
                fallback_tool=analysis.fallback_tool,
            )

        # Check recovery action
        if analysis.recovery_action == RecoveryAction.ABORT:
            return RetryDecision(
                should_retry=False,
                reason="Error analysis recommends aborting",
                attempt=attempt,
                max_attempts=max_retries,
            )

        if analysis.recovery_action == RecoveryAction.ESCALATE:
            return RetryDecision(
                should_retry=False,
                reason="Error requires human intervention",
                attempt=attempt,
                max_attempts=max_retries,
            )

        if analysis.recovery_action == RecoveryAction.SKIP:
            return RetryDecision(
                should_retry=False,
                reason="Error analysis recommends skipping",
                attempt=attempt,
                max_attempts=max_retries,
            )

        if analysis.recovery_action == RecoveryAction.FALLBACK:
            return RetryDecision(
                should_retry=False,
                reason="Should use fallback tool",
                attempt=attempt,
                max_attempts=max_retries,
                fallback_tool=analysis.fallback_tool,
            )

        # Check if retry is worthwhile
        if not analysis.retry_worthwhile:
            return RetryDecision(
                should_retry=False,
                reason="Error analysis says retry unlikely to help",
                attempt=attempt,
                max_attempts=max_retries,
            )

        # Calculate wait time with exponential backoff
        base = settings.get("base_delay", self.base_delay)
        max_d = settings.get("max_delay", self.max_delay)
        wait_time = self._calculate_delay(attempt, base, max_d)

        return RetryDecision(
            should_retry=True,
            wait_seconds=wait_time,
            modified_params=analysis.modified_params,
            reason=f"Retry attempt {attempt + 1}/{max_retries}",
            attempt=attempt,
            max_attempts=max_retries,
            fallback_tool=analysis.fallback_tool,
        )

    def _calculate_delay(
        self, attempt: int, base_delay: float, max_delay: float
    ) -> float:
        """Calculate delay with exponential backoff and jitter."""
        # Exponential backoff: base * 2^attempt
        delay = base_delay * (2 ** attempt)

        # Cap at max delay
        delay = min(delay, max_delay)

        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)  # Minimum 100ms

    async def execute_with_retry(
        self,
        func: Callable[P, T],
        *args: P.args,
        tool_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        analyze_error: Callable[[Exception], Any] | None = None,
        **kwargs: P.kwargs,
    ) -> tuple[T | None, RetryOutcome, list[RetryRecord]]:
        """
        Execute a function with automatic retry handling.

        Args:
            func: Async function to execute
            *args: Function arguments
            tool_name: Tool name for tracking
            parameters: Parameters for correction
            analyze_error: Function to analyze errors
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, outcome, retry_records)
        """
        records: list[RetryRecord] = []
        current_params = dict(parameters) if parameters else {}
        attempt = 0

        while True:
            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Success!
                self._record_success(tool_name or "unknown", attempt)
                records.append(
                    RetryRecord(
                        tool_name=tool_name or "unknown",
                        attempt=attempt,
                        success=True,
                        error=None,
                        wait_time=0,
                        params_modified=attempt > 0,
                    )
                )

                return result, RetryOutcome.SUCCESS, records

            except Exception as e:
                error_str = str(e)
                self._logger.warning(
                    "Execution failed",
                    tool=tool_name,
                    attempt=attempt,
                    error=error_str,
                )

                # Analyze the error
                analysis = None
                if analyze_error:
                    try:
                        analysis = await analyze_error(e)
                    except Exception:
                        pass

                # Use basic analysis if custom analyzer failed
                if not analysis:
                    from src.agents.self_healing.analyzer import (
                        ErrorAnalysis,
                        ErrorCategory,
                        RecoveryAction,
                    )
                    analysis = ErrorAnalysis(
                        error_message=error_str,
                        category=ErrorCategory.UNKNOWN,
                        root_cause="Unknown",
                        recovery_action=RecoveryAction.RETRY_SAME,
                        retry_worthwhile=True,
                    )

                # Decide whether to retry
                decision = self.should_retry(analysis, attempt, tool_name)

                records.append(
                    RetryRecord(
                        tool_name=tool_name or "unknown",
                        attempt=attempt,
                        success=False,
                        error=error_str,
                        wait_time=decision.wait_seconds,
                        params_modified=bool(decision.modified_params),
                    )
                )

                if not decision.should_retry:
                    self._record_failure(tool_name or "unknown", attempt)
                    outcome = (
                        RetryOutcome.SKIPPED
                        if decision.reason == "Error analysis recommends skipping"
                        else RetryOutcome.FAILED
                    )
                    return None, outcome, records

                # Wait before retry
                if decision.wait_seconds > 0:
                    self._logger.debug(
                        "Waiting before retry",
                        wait_seconds=decision.wait_seconds,
                    )
                    await asyncio.sleep(decision.wait_seconds)

                # Apply parameter corrections
                if decision.modified_params:
                    current_params.update(decision.modified_params)
                    if "parameters" in kwargs:
                        kwargs["parameters"] = current_params  # type: ignore

                attempt += 1

    def _record_success(self, tool_name: str, attempts: int) -> None:
        """Record successful execution."""
        if tool_name not in self._tool_success_rates:
            self._tool_success_rates[tool_name] = {"success": 0, "failure": 0}
        self._tool_success_rates[tool_name]["success"] += 1

    def _record_failure(self, tool_name: str, attempts: int) -> None:
        """Record failed execution (after all retries)."""
        if tool_name not in self._tool_success_rates:
            self._tool_success_rates[tool_name] = {"success": 0, "failure": 0}
        self._tool_success_rates[tool_name]["failure"] += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get retry statistics."""
        total_retries = len(self._retry_history)
        successful_retries = sum(1 for r in self._retry_history if r.success)
        modified_retries = sum(1 for r in self._retry_history if r.params_modified)

        tool_stats = {}
        for tool_name, counts in self._tool_success_rates.items():
            total = counts["success"] + counts["failure"]
            tool_stats[tool_name] = {
                "total": total,
                "success_rate": counts["success"] / total if total > 0 else 0,
            }

        return {
            "total_retry_attempts": total_retries,
            "successful_retries": successful_retries,
            "retry_success_rate": successful_retries / total_retries if total_retries > 0 else 0,
            "parameter_modifications": modified_retries,
            "tool_statistics": tool_stats,
        }

    def get_tool_success_rate(self, tool_name: str) -> float:
        """Get success rate for a specific tool."""
        if tool_name not in self._tool_success_rates:
            return 1.0  # Assume success for unknown tools

        counts = self._tool_success_rates[tool_name]
        total = counts["success"] + counts["failure"]
        return counts["success"] / total if total > 0 else 1.0
