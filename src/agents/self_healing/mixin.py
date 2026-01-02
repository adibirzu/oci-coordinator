"""
Self-Healing Mixin for OCI Agents.

Provides self-healing capabilities to any agent:
1. Automatic error analysis and recovery
2. Parameter correction on failures
3. Logic validation before execution
4. Smart retry with learning
5. Fallback tool selection
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.agents.self_healing.analyzer import (
    ErrorAnalyzer,
    ErrorAnalysis,
    ErrorCategory,
    RecoveryAction,
)
from src.agents.self_healing.corrector import ParameterCorrector, CorrectionResult
from src.agents.self_healing.validator import LogicValidator, ValidationResult
from src.agents.self_healing.retry import RetryStrategy, RetryDecision, RetryOutcome

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from src.mcp.catalog import ToolCatalog

logger = structlog.get_logger(__name__)


class SelfHealingMixin:
    """
    Mixin class that adds self-healing capabilities to agents.

    Usage:
        class MyAgent(BaseAgent, SelfHealingMixin):
            def __init__(self, llm, tool_catalog, ...):
                super().__init__(...)
                self.init_self_healing(llm)

            async def invoke(self, query):
                # Use self-healing tool call
                result = await self.healing_call_tool(
                    "oci_database_list_autonomous",
                    {},
                    user_intent=query,
                )

    Features:
    - healing_call_tool: Tool call with automatic retry and correction
    - validate_before_call: Pre-execution logic validation
    - analyze_error: LLM-powered error diagnosis
    - correct_parameters: Automatic parameter fixing
    """

    # Self-healing components (initialized in init_self_healing)
    _error_analyzer: ErrorAnalyzer | None = None
    _param_corrector: ParameterCorrector | None = None
    _logic_validator: LogicValidator | None = None
    _retry_strategy: RetryStrategy | None = None
    _self_healing_enabled: bool = False
    _healing_logger: Any = None

    def init_self_healing(
        self,
        llm: "BaseChatModel | None" = None,
        max_retries: int = 3,
        enable_validation: bool = True,
        enable_correction: bool = True,
    ) -> None:
        """
        Initialize self-healing capabilities.

        Args:
            llm: LangChain LLM for analysis (can use agent's LLM)
            max_retries: Maximum retry attempts
            enable_validation: Enable pre-call validation
            enable_correction: Enable parameter correction
        """
        self._error_analyzer = ErrorAnalyzer(llm)
        self._param_corrector = ParameterCorrector(llm) if enable_correction else None
        self._logic_validator = LogicValidator(llm) if enable_validation else None
        self._retry_strategy = RetryStrategy(
            corrector=self._param_corrector,
            max_retries=max_retries,
        )
        self._self_healing_enabled = True

        # Get agent role from definition (handle classmethod case)
        agent_role = "unknown"
        if hasattr(self.__class__, "get_definition"):
            try:
                definition = self.__class__.get_definition()
                agent_role = getattr(definition, "role", "unknown")
            except Exception:
                pass

        self._healing_logger = logger.bind(
            component="SelfHealingMixin",
            agent=agent_role,
        )
        self._healing_logger.info("Self-healing capabilities initialized")

    async def healing_call_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        user_intent: str | None = None,
        validate: bool = True,
        correct_on_failure: bool = True,
        max_retries: int | None = None,
    ) -> Any:
        """
        Call a tool with self-healing capabilities.

        This method:
        1. Validates the tool call before execution (optional)
        2. Executes the tool
        3. On failure, analyzes the error
        4. Corrects parameters if possible
        5. Retries with exponential backoff
        6. Falls back to alternative tools if available

        Args:
            tool_name: Name of the MCP tool to call
            parameters: Tool parameters
            user_intent: Original user query for context
            validate: Whether to validate before calling
            correct_on_failure: Whether to correct parameters on failure
            max_retries: Override default max retries

        Returns:
            Tool result or None if all attempts fail

        Raises:
            ValueError: If tool catalog is not available
        """
        if not self._self_healing_enabled:
            # Fall back to regular call
            return await self._direct_tool_call(tool_name, parameters)

        # Get tool catalog from parent class
        tools = getattr(self, "tools", None)
        if not tools:
            raise ValueError("Tool catalog not available - call_tool requires tools")

        current_params = dict(parameters)
        current_tool = tool_name
        attempt = 0
        max_attempts = max_retries or (self._retry_strategy.max_retries if self._retry_strategy else 3)
        last_error: Exception | None = None
        fallback_tools: list[str] = []

        # 1. Pre-execution validation
        if validate and self._logic_validator:
            validation = await self._logic_validator.validate(
                tool_name=current_tool,
                parameters=current_params,
                user_intent=user_intent,
                available_tools=self._get_available_tools(),
            )

            if not validation.should_proceed:
                self._healing_logger.warning(
                    "Validation failed",
                    tool=current_tool,
                    issues=[i.message for i in validation.issues],
                )

                # Use suggested tool/params if provided
                if validation.suggested_tool:
                    current_tool = validation.suggested_tool
                    self._healing_logger.info(
                        "Switched to suggested tool",
                        original=tool_name,
                        suggested=current_tool,
                    )

                if validation.suggested_params:
                    current_params = validation.suggested_params

        # 2. Execute with retry loop
        while attempt < max_attempts:
            try:
                # Execute tool call
                result = await self._direct_tool_call(current_tool, current_params)

                # Success!
                self._healing_logger.debug(
                    "Tool call succeeded",
                    tool=current_tool,
                    attempt=attempt,
                )

                # Record success if we had previous failures
                if attempt > 0 and self._error_analyzer:
                    self._error_analyzer.record_error(
                        str(last_error) if last_error else "Unknown",
                        current_tool,
                        ErrorAnalysis(
                            error_message=str(last_error) if last_error else "",
                            category=ErrorCategory.UNKNOWN,
                            root_cause="Resolved on retry",
                            recovery_action=RecoveryAction.RETRY_SAME,
                        ),
                        resolution=f"Succeeded on attempt {attempt + 1}",
                    )

                return result

            except Exception as e:
                last_error = e
                error_msg = str(e)

                self._healing_logger.warning(
                    "Tool call failed",
                    tool=current_tool,
                    attempt=attempt,
                    error=error_msg,
                )

                # 3. Analyze the error
                analysis = await self._analyze_error(e, current_tool, current_params)

                # 4. Decide whether to retry
                if self._retry_strategy:
                    decision = self._retry_strategy.should_retry(
                        analysis, attempt, current_tool
                    )
                else:
                    decision = RetryDecision(
                        should_retry=attempt < max_attempts - 1,
                        wait_seconds=1.0 * (2 ** attempt),
                        reason="Default retry logic",
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )

                if not decision.should_retry:
                    # Check for fallback
                    if decision.fallback_tool and decision.fallback_tool not in fallback_tools:
                        fallback_tools.append(current_tool)
                        current_tool = decision.fallback_tool
                        self._healing_logger.info(
                            "Switching to fallback tool",
                            fallback=current_tool,
                        )
                        attempt = 0  # Reset attempts for fallback
                        continue
                    else:
                        # No more options
                        break

                # 5. Correct parameters if enabled
                if correct_on_failure and self._param_corrector:
                    correction = await self._param_corrector.correct(
                        current_tool, current_params, error_msg
                    )
                    if correction.corrected:
                        current_params = correction.corrected_params
                        self._healing_logger.info(
                            "Parameters corrected",
                            changes=correction.changes_made,
                        )

                # Apply any corrections from error analysis
                if decision.modified_params:
                    current_params.update(decision.modified_params)

                # 6. Wait before retry
                if decision.wait_seconds > 0:
                    self._healing_logger.debug(
                        "Waiting before retry",
                        wait_seconds=decision.wait_seconds,
                    )
                    await asyncio.sleep(decision.wait_seconds)

                attempt += 1

        # All attempts exhausted
        self._healing_logger.error(
            "All retry attempts exhausted",
            tool=current_tool,
            original_tool=tool_name,
            attempts=attempt,
            last_error=str(last_error),
        )

        # Record failure for learning
        if self._error_analyzer and last_error:
            self._error_analyzer.record_error(
                str(last_error),
                current_tool,
                await self._analyze_error(last_error, current_tool, current_params),
                resolution="Failed after all retries",
            )

        return None

    async def _direct_tool_call(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> Any:
        """Direct tool call using parent class method."""
        # Try call_tool first (BaseAgent method)
        if hasattr(self, "call_tool"):
            return await self.call_tool(tool_name, parameters)

        # Fall back to tools.execute
        tools = getattr(self, "tools", None)
        if tools:
            result = await tools.execute(tool_name, parameters)
            return result.content if hasattr(result, "content") else result

        raise ValueError("No tool execution method available")

    def _get_available_tools(self) -> list[str]:
        """Get list of available tools from catalog."""
        tools = getattr(self, "tools", None)
        if not tools:
            return []

        # Try to get tool list from catalog
        if hasattr(tools, "list_tools"):
            try:
                return [t.name for t in tools.list_tools()]
            except Exception:
                pass

        # Fall back to MCP_TOOLS class variable
        mcp_tools = getattr(self.__class__, "MCP_TOOLS", [])
        return list(mcp_tools)

    async def _analyze_error(
        self,
        error: Exception,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ErrorAnalysis:
        """Analyze an error using the error analyzer."""
        if self._error_analyzer:
            # Get agent role from definition (handle classmethod case)
            agent_role = "unknown"
            if hasattr(self.__class__, "get_definition"):
                try:
                    definition = self.__class__.get_definition()
                    agent_role = getattr(definition, "role", "unknown")
                except Exception:
                    pass

            return await self._error_analyzer.analyze(
                error=error,
                tool_name=tool_name,
                parameters=parameters,
                context={"agent": agent_role},
            )

        # Basic fallback analysis
        return ErrorAnalysis(
            error_message=str(error),
            category=ErrorCategory.UNKNOWN,
            root_cause="Unknown - no error analyzer available",
            recovery_action=RecoveryAction.RETRY_SAME,
            retry_worthwhile=True,
        )

    async def validate_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        user_intent: str | None = None,
    ) -> ValidationResult:
        """
        Validate a tool call before execution.

        Args:
            tool_name: Tool to validate
            parameters: Parameters to validate
            user_intent: User's original query

        Returns:
            ValidationResult with issues and suggestions
        """
        if not self._logic_validator:
            from src.agents.self_healing.validator import ValidationResult
            return ValidationResult(
                valid=True,
                confidence=0.5,
                reasoning="No validator available",
                should_proceed=True,
            )

        return await self._logic_validator.validate(
            tool_name=tool_name,
            parameters=parameters,
            user_intent=user_intent,
            available_tools=self._get_available_tools(),
        )

    async def correct_parameters(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        error_message: str | None = None,
    ) -> CorrectionResult:
        """
        Correct parameters for a tool call.

        Args:
            tool_name: Tool name
            parameters: Current parameters
            error_message: Error from previous attempt

        Returns:
            CorrectionResult with corrections
        """
        if not self._param_corrector:
            return CorrectionResult(
                original_params=parameters,
                corrected_params=parameters,
                changes_made=[],
                confidence=0.5,
                reasoning="No corrector available",
                corrected=False,
            )

        return await self._param_corrector.correct(
            tool_name=tool_name,
            parameters=parameters,
            error_message=error_message,
        )

    def get_healing_statistics(self) -> dict[str, Any]:
        """Get statistics on self-healing operations."""
        stats: dict[str, Any] = {
            "enabled": self._self_healing_enabled,
        }

        if self._error_analyzer:
            stats["error_analysis"] = self._error_analyzer.get_error_statistics()

        if self._param_corrector:
            stats["parameter_correction"] = self._param_corrector.get_correction_statistics()

        if self._logic_validator:
            stats["validation"] = self._logic_validator.get_validation_statistics()

        if self._retry_strategy:
            stats["retry"] = self._retry_strategy.get_statistics()

        return stats

    def disable_self_healing(self) -> None:
        """Disable self-healing (for debugging)."""
        self._self_healing_enabled = False
        if self._healing_logger:
            self._healing_logger.info("Self-healing disabled")

    def enable_self_healing(self) -> None:
        """Re-enable self-healing."""
        self._self_healing_enabled = True
        if self._healing_logger:
            self._healing_logger.info("Self-healing enabled")
