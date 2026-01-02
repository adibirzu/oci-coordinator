"""
LLM-based Error Analysis for Self-Healing Agents.

Analyzes tool failures and agent errors to:
1. Identify root cause
2. Suggest corrections
3. Determine if retry is worthwhile
4. Learn patterns from failures
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = structlog.get_logger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for analysis."""

    PARAMETER_ERROR = "parameter_error"  # Wrong/missing parameters
    TIMEOUT = "timeout"  # Operation timed out
    AUTH_ERROR = "auth_error"  # Authentication/authorization failure
    RESOURCE_NOT_FOUND = "resource_not_found"  # Resource doesn't exist
    RATE_LIMIT = "rate_limit"  # API rate limiting
    NETWORK_ERROR = "network_error"  # Network connectivity issues
    SERVICE_ERROR = "service_error"  # OCI service error
    LOGIC_ERROR = "logic_error"  # Agent logic/reasoning error
    DATA_ERROR = "data_error"  # Invalid data format
    UNKNOWN = "unknown"  # Unknown error type


class RecoveryAction(str, Enum):
    """Recommended recovery actions."""

    RETRY_SAME = "retry_same"  # Retry with same parameters
    RETRY_MODIFIED = "retry_modified"  # Retry with modified parameters
    SKIP = "skip"  # Skip this step and continue
    FALLBACK = "fallback"  # Use fallback tool/approach
    ESCALATE = "escalate"  # Escalate to human/coordinator
    ABORT = "abort"  # Abort the workflow


@dataclass
class ErrorAnalysis:
    """Result of error analysis."""

    error_message: str
    category: ErrorCategory
    root_cause: str
    recovery_action: RecoveryAction
    suggested_fix: str | None = None
    modified_params: dict[str, Any] | None = None
    confidence: float = 0.0
    reasoning: str = ""
    retry_worthwhile: bool = False
    max_retries: int = 3
    backoff_seconds: float = 1.0
    fallback_tool: str | None = None
    learned_pattern: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "error_message": self.error_message,
            "category": self.category.value,
            "root_cause": self.root_cause,
            "recovery_action": self.recovery_action.value,
            "suggested_fix": self.suggested_fix,
            "modified_params": self.modified_params,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "retry_worthwhile": self.retry_worthwhile,
            "max_retries": self.max_retries,
            "backoff_seconds": self.backoff_seconds,
            "fallback_tool": self.fallback_tool,
            "learned_pattern": self.learned_pattern,
        }


@dataclass
class ErrorPattern:
    """A learned error pattern for quick matching."""

    pattern: str  # Regex pattern
    category: ErrorCategory
    recovery_action: RecoveryAction
    fix_template: str | None = None
    hit_count: int = 0
    last_seen: datetime | None = None


class ErrorAnalyzer:
    """
    LLM-powered error analyzer for self-healing agents.

    Analyzes errors to determine:
    - What went wrong (root cause)
    - How to fix it (recovery action)
    - Whether to retry (with what modifications)
    - What patterns to learn from
    """

    # Pre-defined error patterns for quick matching (no LLM needed)
    KNOWN_PATTERNS: list[ErrorPattern] = [
        # OCI-specific patterns
        ErrorPattern(
            pattern=r"InvalidParameter|invalid.*(parameter|argument|value)",
            category=ErrorCategory.PARAMETER_ERROR,
            recovery_action=RecoveryAction.RETRY_MODIFIED,
            fix_template="Check parameter types and required fields",
        ),
        ErrorPattern(
            pattern=r"NotAuthenticated|NotAuthorized|401|403|InvalidCredentials",
            category=ErrorCategory.AUTH_ERROR,
            recovery_action=RecoveryAction.ESCALATE,
            fix_template="Authentication required - check credentials",
        ),
        ErrorPattern(
            pattern=r"NotFound|404|does not exist|no.*found",
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            recovery_action=RecoveryAction.SKIP,
            fix_template="Resource not found - may need different identifier",
        ),
        ErrorPattern(
            pattern=r"timeout|timed out|TimeoutError|deadline exceeded",
            category=ErrorCategory.TIMEOUT,
            recovery_action=RecoveryAction.RETRY_SAME,
            fix_template="Increase timeout or retry",
        ),
        ErrorPattern(
            pattern=r"rate.?limit|429|too many requests|throttl",
            category=ErrorCategory.RATE_LIMIT,
            recovery_action=RecoveryAction.RETRY_SAME,
            fix_template="Wait and retry with exponential backoff",
        ),
        ErrorPattern(
            pattern=r"connection.*refused|network.*unreachable|DNS",
            category=ErrorCategory.NETWORK_ERROR,
            recovery_action=RecoveryAction.RETRY_SAME,
            fix_template="Check network connectivity",
        ),
        ErrorPattern(
            pattern=r"ORA-\d{5}",
            category=ErrorCategory.SERVICE_ERROR,
            recovery_action=RecoveryAction.FALLBACK,
            fix_template="Oracle database error - check database status",
        ),
        ErrorPattern(
            pattern=r"ServiceError|InternalError|500|502|503",
            category=ErrorCategory.SERVICE_ERROR,
            recovery_action=RecoveryAction.RETRY_SAME,
            fix_template="OCI service error - retry or check service status",
        ),
        ErrorPattern(
            pattern=r"JSON|parse|decode|serialize|deserialize",
            category=ErrorCategory.DATA_ERROR,
            recovery_action=RecoveryAction.RETRY_MODIFIED,
            fix_template="Data format error - check input format",
        ),
    ]

    def __init__(self, llm: "BaseChatModel | None" = None):
        """
        Initialize error analyzer.

        Args:
            llm: LangChain LLM for complex error analysis
        """
        self.llm = llm
        self._learned_patterns: list[ErrorPattern] = []
        self._error_history: list[dict[str, Any]] = []
        self._logger = logger.bind(component="ErrorAnalyzer")

    async def analyze(
        self,
        error: Exception | str,
        tool_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> ErrorAnalysis:
        """
        Analyze an error and provide recovery recommendations.

        Args:
            error: The error (exception or message string)
            tool_name: Name of the tool that failed
            parameters: Parameters that were passed to the tool
            context: Additional context (agent state, history, etc.)
            use_llm: Whether to use LLM for complex analysis

        Returns:
            ErrorAnalysis with diagnosis and recovery recommendations
        """
        error_msg = str(error)

        # 1. Try quick pattern matching first
        pattern_match = self._match_known_pattern(error_msg)
        if pattern_match:
            self._logger.debug(
                "Error matched known pattern",
                pattern=pattern_match.pattern,
                category=pattern_match.category.value,
            )
            return self._create_analysis_from_pattern(
                error_msg, pattern_match, tool_name, parameters
            )

        # 2. Check learned patterns
        learned_match = self._match_learned_pattern(error_msg)
        if learned_match:
            self._logger.debug(
                "Error matched learned pattern",
                pattern=learned_match.pattern,
            )
            return self._create_analysis_from_pattern(
                error_msg, learned_match, tool_name, parameters
            )

        # 3. Use LLM for complex analysis
        if use_llm and self.llm:
            return await self._analyze_with_llm(
                error_msg, tool_name, parameters, context
            )

        # 4. Fallback to basic analysis
        return self._basic_analysis(error_msg, tool_name)

    def _match_known_pattern(self, error_msg: str) -> ErrorPattern | None:
        """Match error against known patterns."""
        for pattern in self.KNOWN_PATTERNS:
            if re.search(pattern.pattern, error_msg, re.IGNORECASE):
                pattern.hit_count += 1
                pattern.last_seen = datetime.utcnow()
                return pattern
        return None

    def _match_learned_pattern(self, error_msg: str) -> ErrorPattern | None:
        """Match error against learned patterns."""
        for pattern in self._learned_patterns:
            if re.search(pattern.pattern, error_msg, re.IGNORECASE):
                pattern.hit_count += 1
                pattern.last_seen = datetime.utcnow()
                return pattern
        return None

    def _create_analysis_from_pattern(
        self,
        error_msg: str,
        pattern: ErrorPattern,
        tool_name: str | None,
        parameters: dict[str, Any] | None,
    ) -> ErrorAnalysis:
        """Create analysis from matched pattern."""
        # Determine retry settings based on category
        retry_settings = self._get_retry_settings(pattern.category)

        return ErrorAnalysis(
            error_message=error_msg,
            category=pattern.category,
            root_cause=f"Matched pattern: {pattern.pattern}",
            recovery_action=pattern.recovery_action,
            suggested_fix=pattern.fix_template,
            confidence=0.85,  # Pattern match is fairly confident
            reasoning=f"Error matched known pattern for {pattern.category.value}",
            retry_worthwhile=pattern.recovery_action in (
                RecoveryAction.RETRY_SAME,
                RecoveryAction.RETRY_MODIFIED,
            ),
            **retry_settings,
        )

    def _get_retry_settings(self, category: ErrorCategory) -> dict[str, Any]:
        """Get retry settings based on error category."""
        settings = {
            ErrorCategory.TIMEOUT: {"max_retries": 3, "backoff_seconds": 2.0},
            ErrorCategory.RATE_LIMIT: {"max_retries": 5, "backoff_seconds": 5.0},
            ErrorCategory.NETWORK_ERROR: {"max_retries": 3, "backoff_seconds": 1.0},
            ErrorCategory.SERVICE_ERROR: {"max_retries": 2, "backoff_seconds": 3.0},
            ErrorCategory.PARAMETER_ERROR: {"max_retries": 2, "backoff_seconds": 0.5},
        }
        return settings.get(category, {"max_retries": 2, "backoff_seconds": 1.0})

    async def _analyze_with_llm(
        self,
        error_msg: str,
        tool_name: str | None,
        parameters: dict[str, Any] | None,
        context: dict[str, Any] | None,
    ) -> ErrorAnalysis:
        """Use LLM to analyze complex errors."""
        prompt = f"""Analyze this error and provide recovery recommendations.

ERROR MESSAGE:
{error_msg}

TOOL: {tool_name or "Unknown"}

PARAMETERS:
{json.dumps(parameters, indent=2, default=str) if parameters else "None"}

CONTEXT:
{json.dumps(context, indent=2, default=str) if context else "None"}

Analyze this error and respond in JSON format:
{{
    "category": "parameter_error|timeout|auth_error|resource_not_found|rate_limit|network_error|service_error|logic_error|data_error|unknown",
    "root_cause": "Brief explanation of what went wrong",
    "recovery_action": "retry_same|retry_modified|skip|fallback|escalate|abort",
    "suggested_fix": "Specific fix recommendation",
    "modified_params": {{"param_name": "new_value"}} or null,
    "confidence": 0.0-1.0,
    "reasoning": "Why this diagnosis and recovery action",
    "retry_worthwhile": true|false,
    "fallback_tool": "alternative tool name" or null,
    "learned_pattern": "regex pattern to match similar errors" or null
}}

Be specific about parameter corrections if needed. Consider OCI-specific error codes."""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                result = json.loads(json_match.group())

                # Learn new pattern if provided
                if result.get("learned_pattern"):
                    self._learn_pattern(
                        result["learned_pattern"],
                        ErrorCategory(result.get("category", "unknown")),
                        RecoveryAction(result.get("recovery_action", "retry_same")),
                        result.get("suggested_fix"),
                    )

                return ErrorAnalysis(
                    error_message=error_msg,
                    category=ErrorCategory(result.get("category", "unknown")),
                    root_cause=result.get("root_cause", "Unknown"),
                    recovery_action=RecoveryAction(result.get("recovery_action", "retry_same")),
                    suggested_fix=result.get("suggested_fix"),
                    modified_params=result.get("modified_params"),
                    confidence=result.get("confidence", 0.7),
                    reasoning=result.get("reasoning", ""),
                    retry_worthwhile=result.get("retry_worthwhile", False),
                    fallback_tool=result.get("fallback_tool"),
                    learned_pattern=result.get("learned_pattern"),
                )

        except Exception as e:
            self._logger.warning("LLM analysis failed", error=str(e))

        # Fallback to basic analysis
        return self._basic_analysis(error_msg, tool_name)

    def _basic_analysis(self, error_msg: str, tool_name: str | None) -> ErrorAnalysis:
        """Basic analysis when patterns don't match and LLM unavailable."""
        return ErrorAnalysis(
            error_message=error_msg,
            category=ErrorCategory.UNKNOWN,
            root_cause="Unable to determine root cause",
            recovery_action=RecoveryAction.RETRY_SAME,
            suggested_fix="Try again or check logs for more details",
            confidence=0.3,
            reasoning="No pattern match, basic analysis only",
            retry_worthwhile=True,
            max_retries=2,
            backoff_seconds=1.0,
        )

    def _learn_pattern(
        self,
        pattern: str,
        category: ErrorCategory,
        recovery_action: RecoveryAction,
        fix_template: str | None,
    ) -> None:
        """Learn a new error pattern from LLM analysis."""
        try:
            # Validate regex pattern
            re.compile(pattern)

            # Check if pattern already exists
            for existing in self._learned_patterns:
                if existing.pattern == pattern:
                    return

            self._learned_patterns.append(
                ErrorPattern(
                    pattern=pattern,
                    category=category,
                    recovery_action=recovery_action,
                    fix_template=fix_template,
                    hit_count=1,
                    last_seen=datetime.utcnow(),
                )
            )

            self._logger.info(
                "Learned new error pattern",
                pattern=pattern,
                category=category.value,
            )

        except re.error:
            self._logger.warning("Invalid regex pattern from LLM", pattern=pattern)

    def record_error(
        self,
        error: str,
        tool_name: str | None,
        analysis: ErrorAnalysis,
        resolution: str | None = None,
    ) -> None:
        """Record error for history and learning."""
        self._error_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "error": error,
            "tool_name": tool_name,
            "category": analysis.category.value,
            "recovery_action": analysis.recovery_action.value,
            "resolution": resolution,
        })

        # Keep last 100 errors
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-100:]

    def get_error_statistics(self) -> dict[str, Any]:
        """Get statistics on error patterns."""
        category_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}

        for error in self._error_history:
            cat = error["category"]
            action = error["recovery_action"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "total_errors": len(self._error_history),
            "by_category": category_counts,
            "by_action": action_counts,
            "known_patterns": len(self.KNOWN_PATTERNS),
            "learned_patterns": len(self._learned_patterns),
        }
