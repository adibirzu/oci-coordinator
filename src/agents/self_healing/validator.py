"""
Logic Validator for Self-Healing Agents.

Uses LLM to validate agent decisions before execution:
1. Check if tool choice is appropriate for the task
2. Validate parameter completeness and correctness
3. Detect potential logic errors in reasoning
4. Suggest alternatives when current plan is flawed

IMPORTANT: This validator uses the actual ToolCatalog to validate tool names
and suggestions. It does NOT rely on hardcoded tool lists which can become
stale and cause the LLM to suggest non-existent tools.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from src.mcp.catalog import ToolCatalog

logger = structlog.get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""

    INFO = "info"  # Informational, proceed anyway
    WARNING = "warning"  # Might cause issues, but can proceed
    ERROR = "error"  # Likely to fail, should correct
    CRITICAL = "critical"  # Will definitely fail, must correct


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: ValidationSeverity
    message: str
    suggestion: str | None = None
    affected_param: str | None = None


@dataclass
class ValidationResult:
    """Result of logic validation."""

    valid: bool
    confidence: float
    issues: list[ValidationIssue] = field(default_factory=list)
    suggested_tool: str | None = None
    suggested_params: dict[str, Any] | None = None
    reasoning: str = ""
    should_proceed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "valid": self.valid,
            "confidence": self.confidence,
            "issues": [
                {
                    "severity": i.severity.value,
                    "message": i.message,
                    "suggestion": i.suggestion,
                    "affected_param": i.affected_param,
                }
                for i in self.issues
            ],
            "suggested_tool": self.suggested_tool,
            "suggested_params": self.suggested_params,
            "reasoning": self.reasoning,
            "should_proceed": self.should_proceed,
        }

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in self.issues
        )

    @property
    def error_count(self) -> int:
        """Count error-level issues."""
        return sum(
            1
            for i in self.issues
            if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )


# Tool capability mappings for quick validation
TOOL_CAPABILITIES = {
    # Database tools
    "oci_database_list_autonomous": {
        "capability": "list_databases",
        "domains": ["database"],
        "returns": "list of databases",
        "requires_auth": True,
    },
    "oci_opsi_search_databases": {
        "capability": "search_databases",
        "domains": ["database", "opsi"],
        "returns": "database search results",
        "requires_auth": True,
        "faster_than": ["oci_database_list_autonomous"],
    },
    "oci_database_execute_sql": {
        "capability": "execute_sql",
        "domains": ["database"],
        "returns": "query results",
        "requires_auth": True,
        "requires_connection": True,
    },
    # Cost tools
    "oci_cost_get_summary": {
        "capability": "get_cost_summary",
        "domains": ["finops", "cost"],
        "returns": "cost breakdown by service",
        "requires_auth": True,
        "slow_api": True,
    },
    # Compute tools
    "oci_compute_list_instances": {
        "capability": "list_instances",
        "domains": ["compute", "infrastructure"],
        "returns": "list of compute instances",
        "requires_auth": True,
    },
    # Logging tools
    "oci_logging_search_logs": {
        "capability": "search_logs",
        "domains": ["logging", "observability"],
        "returns": "log entries",
        "requires_auth": True,
    },
    "oci_logan_execute_query": {
        "capability": "query_logs",
        "domains": ["logan", "observability"],
        "returns": "log analytics results",
        "requires_auth": True,
    },
    # Security tools
    "oci_security_cloudguard_list_problems": {
        "capability": "list_security_problems",
        "domains": ["security", "cloudguard"],
        "returns": "cloud guard problems",
        "requires_auth": True,
    },
    "oci_security_list_cloud_guard_problems": {
        "capability": "list_security_problems",
        "domains": ["security", "cloudguard"],
        "returns": "cloud guard problems",
        "requires_auth": True,
    },
}

# Query intent to tool mapping
INTENT_TOOL_MAPPING = {
    "list_databases": ["oci_opsi_search_databases", "oci_database_list_autonomous"],
    "database_cost": ["oci_cost_get_summary"],
    "list_instances": ["oci_compute_list_instances"],
    "search_logs": ["oci_logging_search_logs", "oci_logan_execute_query"],
    "cost_summary": ["oci_cost_get_summary"],
    "security_problems": [
        "oci_security_cloudguard_list_problems",
        "oci_security_list_cloud_guard_problems",
    ],
}


class LogicValidator:
    """
    Validates agent logic and decisions before execution.

    Features:
    - Tool-to-intent matching
    - Parameter completeness checking
    - Logic flow validation
    - Alternative suggestion
    - LLM-powered reasoning validation

    IMPORTANT: Uses actual ToolCatalog to validate tool names and prevent
    the LLM from suggesting non-existent tools.
    """

    def __init__(
        self,
        llm: "BaseChatModel | None" = None,
        tool_catalog: "ToolCatalog | None" = None,
    ):
        """
        Initialize logic validator.

        Args:
            llm: LangChain LLM for complex validation
            tool_catalog: MCP ToolCatalog for validating tool names
        """
        self.llm = llm
        self._tool_catalog = tool_catalog
        self._validation_history: list[ValidationResult] = []
        self._logger = logger.bind(component="LogicValidator")
        self._cached_tool_names: set[str] | None = None

    def _get_available_tool_names(self) -> set[str]:
        """Get set of actual available tool names from catalog."""
        if self._cached_tool_names is not None:
            return self._cached_tool_names

        if self._tool_catalog is None:
            # Fallback to hardcoded list if no catalog provided
            self._cached_tool_names = set(TOOL_CAPABILITIES.keys())
            return self._cached_tool_names

        # Get actual tools from catalog
        try:
            tools = self._tool_catalog.list_tools()
            self._cached_tool_names = {t.name for t in tools}
            self._logger.debug(
                "Loaded tool names from catalog",
                tool_count=len(self._cached_tool_names),
            )
        except Exception as e:
            self._logger.warning(
                "Failed to load tools from catalog, using fallback",
                error=str(e),
            )
            self._cached_tool_names = set(TOOL_CAPABILITIES.keys())

        return self._cached_tool_names

    def clear_tool_cache(self) -> None:
        """Clear cached tool names to force refresh on next validation."""
        self._cached_tool_names = None

    async def validate(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        user_intent: str | None = None,
        available_tools: list[str] | None = None,
        context: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> ValidationResult:
        """
        Validate a tool call decision.

        Args:
            tool_name: Proposed tool to call
            parameters: Proposed parameters
            user_intent: Original user query/intent
            available_tools: List of available tools
            context: Additional context
            use_llm: Whether to use LLM for validation

        Returns:
            ValidationResult with issues and suggestions
        """
        issues: list[ValidationIssue] = []

        # Get actual available tool names from catalog
        actual_tool_names = self._get_available_tool_names()

        # Use provided available_tools or fall back to catalog
        effective_available_tools = (
            set(available_tools) if available_tools else actual_tool_names
        )

        # 1. Check if tool exists in actual catalog (not just hardcoded list)
        tool_exists_in_catalog = tool_name in actual_tool_names
        tool_caps = TOOL_CAPABILITIES.get(tool_name)

        if not tool_exists_in_catalog and not tool_caps:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Tool does not exist: {tool_name}",
                    suggestion="Check ToolCatalog for available tools",
                )
            )
        elif not tool_exists_in_catalog:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Tool {tool_name} not found in MCP catalog",
                    suggestion="Tool may be unavailable or misspelled",
                )
            )

        # 2. Validate tool-intent match
        if user_intent and tool_caps:
            intent_issues = self._validate_intent_match(tool_name, user_intent, tool_caps)
            issues.extend(intent_issues)

        # 3. Check parameter requirements
        param_issues = self._validate_parameters(tool_name, parameters)
        issues.extend(param_issues)

        # 4. Check for better alternatives
        if effective_available_tools and tool_caps:
            alt_issues = self._check_alternatives(
                tool_name, list(effective_available_tools), tool_caps
            )
            issues.extend(alt_issues)

        # 5. Use LLM for complex validation (with actual tool list)
        llm_result = None
        if use_llm and self.llm and user_intent:
            llm_result = await self._validate_with_llm(
                tool_name,
                parameters,
                user_intent,
                context,
                effective_available_tools,
            )
            if llm_result:
                issues.extend(llm_result.issues)

        # Determine if we should proceed
        has_errors = any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in issues
        )

        result = ValidationResult(
            valid=not has_errors,
            confidence=0.9 if not issues else (0.5 if has_errors else 0.7),
            issues=issues,
            suggested_tool=llm_result.suggested_tool if llm_result else None,
            suggested_params=llm_result.suggested_params if llm_result else None,
            reasoning=self._build_reasoning(issues, llm_result),
            should_proceed=not has_errors,
        )

        self._validation_history.append(result)
        return result

    def _validate_intent_match(
        self,
        tool_name: str,
        user_intent: str,
        tool_caps: dict[str, Any],
    ) -> list[ValidationIssue]:
        """Validate tool matches user intent."""
        issues = []
        intent_lower = user_intent.lower()
        tool_capability = tool_caps.get("capability", "")

        # Check for obvious mismatches
        cost_keywords = ["cost", "spend", "spending", "price", "billing"]
        db_keywords = ["database", "databases", "db", "dbs", "autonomous"]
        log_keywords = ["log", "logs", "error", "errors", "audit"]

        is_cost_intent = any(kw in intent_lower for kw in cost_keywords)
        is_db_intent = any(kw in intent_lower for kw in db_keywords)
        is_log_intent = any(kw in intent_lower for kw in log_keywords)

        tool_domains = tool_caps.get("domains", [])

        if is_cost_intent and "cost" not in tool_capability and "finops" not in tool_domains:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Tool {tool_name} may not be appropriate for cost queries",
                    suggestion="Consider using oci_cost_get_summary for cost data",
                )
            )

        if is_db_intent and "database" not in tool_domains and "opsi" not in tool_domains:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Tool {tool_name} may not be appropriate for database queries",
                    suggestion="Consider using oci_opsi_search_databases or oci_database_list_autonomous",
                )
            )

        if is_log_intent and "logging" not in tool_domains and "observability" not in tool_domains:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Tool {tool_name} may not be appropriate for log queries",
                    suggestion="Consider using oci_logging_search_logs or oci_logan_execute_query",
                )
            )

        return issues

    def _validate_parameters(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Validate parameter completeness and format."""
        issues = []

        # Check for empty parameters
        if not parameters:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="No parameters provided - using defaults",
                )
            )

        # Check OCID formats
        for param_name, value in parameters.items():
            if value is None:
                continue

            if "id" in param_name.lower() and isinstance(value, str):
                if not value.startswith("ocid1."):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Parameter {param_name} may not be a valid OCID",
                            suggestion="OCI IDs should start with 'ocid1.'",
                            affected_param=param_name,
                        )
                    )

            # Check for placeholder values
            if isinstance(value, str):
                if value in ("<value>", "{{value}}", "TODO", "PLACEHOLDER"):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Parameter {param_name} has a placeholder value",
                            suggestion="Replace with actual value",
                            affected_param=param_name,
                        )
                    )

        return issues

    def _check_alternatives(
        self,
        tool_name: str,
        available_tools: list[str],
        tool_caps: dict[str, Any],
    ) -> list[ValidationIssue]:
        """Check if there are better alternatives."""
        issues = []

        # Check if there's a faster alternative
        faster_than = tool_caps.get("faster_than", [])
        if tool_name in faster_than:
            # Current tool might be slower
            for faster_tool in [t for t in TOOL_CAPABILITIES if tool_name in TOOL_CAPABILITIES.get(t, {}).get("faster_than", [])]:
                if faster_tool in available_tools:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"{faster_tool} might be faster than {tool_name}",
                            suggestion=f"Consider using {faster_tool} for better performance",
                        )
                    )

        # Check for slow API warning
        if tool_caps.get("slow_api"):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Tool {tool_name} uses a slow API - may timeout",
                    suggestion="Be prepared to retry or increase timeout",
                )
            )

        return issues

    async def _validate_with_llm(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        user_intent: str,
        context: dict[str, Any] | None,
        available_tools: set[str] | None = None,
    ) -> ValidationResult | None:
        """
        Use LLM for complex validation.

        Args:
            tool_name: Proposed tool name
            parameters: Proposed parameters
            user_intent: Original user intent
            context: Additional context
            available_tools: Set of actually available tool names from catalog

        Returns:
            ValidationResult with LLM assessment, or None on failure
        """
        # Format available tools for prompt (limit to relevant ones for context)
        tools_context = ""
        if available_tools:
            # Group by prefix for readability
            tool_list = sorted(available_tools)
            # Limit to first 100 tools to avoid overwhelming the prompt
            if len(tool_list) > 100:
                tools_context = f"AVAILABLE TOOLS (first 100 of {len(tool_list)}):\n" + "\n".join(f"- {t}" for t in tool_list[:100])
            else:
                tools_context = f"AVAILABLE TOOLS ({len(tool_list)} total):\n" + "\n".join(f"- {t}" for t in tool_list)

        prompt = f"""Validate this tool call decision for an OCI agent.

USER INTENT: {user_intent}

PROPOSED TOOL: {tool_name}

PROPOSED PARAMETERS:
{json.dumps(parameters, indent=2, default=str)}

CONTEXT:
{json.dumps(context, indent=2, default=str) if context else "None"}

{tools_context}

Analyze if this is the right tool and parameters for the user's intent.

CRITICAL: If you suggest an alternative tool (suggested_tool), it MUST be one from the AVAILABLE TOOLS list above.
Do NOT suggest tools that are not in the list - they will not work.

Respond in JSON format:
{{
    "valid": true|false,
    "issues": [
        {{
            "severity": "info|warning|error|critical",
            "message": "description of issue",
            "suggestion": "how to fix",
            "affected_param": "param name or null"
        }}
    ],
    "suggested_tool": "tool name from AVAILABLE TOOLS list" or null,
    "suggested_params": {{"corrected params"}} or null,
    "reasoning": "why this decision is good/bad"
}}

Consider:
- Does the tool match what the user is asking for?
- Are the parameters correct and complete?
- Is there a better tool for this task from the AVAILABLE TOOLS?
- Will this likely succeed or fail?"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                result = json.loads(json_match.group())

                issues = []
                for issue_data in result.get("issues", []):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity(issue_data.get("severity", "info")),
                            message=issue_data.get("message", ""),
                            suggestion=issue_data.get("suggestion"),
                            affected_param=issue_data.get("affected_param"),
                        )
                    )

                # Validate suggested_tool against actual available tools
                suggested_tool = result.get("suggested_tool")
                if suggested_tool and available_tools:
                    if suggested_tool not in available_tools:
                        self._logger.warning(
                            "LLM suggested non-existent tool, ignoring",
                            suggested=suggested_tool,
                            available_count=len(available_tools),
                        )
                        # Add warning issue and clear the invalid suggestion
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"LLM suggested unavailable tool: {suggested_tool}",
                                suggestion="Using original tool instead",
                            )
                        )
                        suggested_tool = None

                return ValidationResult(
                    valid=result.get("valid", True),
                    confidence=0.8,
                    issues=issues,
                    suggested_tool=suggested_tool,
                    suggested_params=result.get("suggested_params"),
                    reasoning=result.get("reasoning", ""),
                    should_proceed=result.get("valid", True),
                )

        except Exception as e:
            self._logger.warning("LLM validation failed", error=str(e))

        return None

    def _build_reasoning(
        self,
        issues: list[ValidationIssue],
        llm_result: ValidationResult | None,
    ) -> str:
        """Build reasoning summary from validation results."""
        parts = []

        if not issues:
            parts.append("No issues detected - tool call appears valid")
        else:
            error_count = sum(1 for i in issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL))
            warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)

            if error_count:
                parts.append(f"Found {error_count} error(s) that should be addressed")
            if warning_count:
                parts.append(f"Found {warning_count} warning(s) to consider")

        if llm_result and llm_result.reasoning:
            parts.append(f"LLM assessment: {llm_result.reasoning}")

        return ". ".join(parts)

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get validation statistics."""
        total = len(self._validation_history)
        valid_count = sum(1 for r in self._validation_history if r.valid)

        severity_counts: dict[str, int] = {}
        for result in self._validation_history:
            for issue in result.issues:
                sev = issue.severity.value
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_validations": total,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "validation_rate": valid_count / total if total > 0 else 1.0,
            "issues_by_severity": severity_counts,
        }
