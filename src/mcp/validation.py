"""
MCP Tool Catalog Validation.

Validates that all tools in the catalog follow naming conventions,
have proper tier assignments, and are correctly configured.

Features:
- Naming convention validation (oci_{domain}_{action})
- Tool tier classification check
- Timeout configuration validation
- Startup health verification
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.mcp.registry import ServerRegistry

logger = structlog.get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Must fix before production
    WARNING = "warning"  # Should fix, but not blocking
    INFO = "info"  # Informational, best practice


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: ValidationSeverity
    category: str
    tool_name: str
    message: str
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "tool_name": self.tool_name,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of tool catalog validation."""

    valid: bool
    total_tools: int
    issues: list[ValidationIssue] = field(default_factory=list)
    by_severity: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.utcnow)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)

        # Update counts
        sev = issue.severity.value
        self.by_severity[sev] = self.by_severity.get(sev, 0) + 1

        cat = issue.category
        self.by_category[cat] = self.by_category.get(cat, 0) + 1

        # Mark invalid if any errors
        if issue.severity == ValidationSeverity.ERROR:
            self.valid = False

    def get_errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "valid": self.valid,
            "total_tools": self.total_tools,
            "issues": [i.to_dict() for i in self.issues],
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "validated_at": self.validated_at.isoformat(),
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Tool Catalog Validation: {'VALID' if self.valid else 'INVALID'}",
            f"  Total tools: {self.total_tools}",
            f"  Errors: {self.by_severity.get('error', 0)}",
            f"  Warnings: {self.by_severity.get('warning', 0)}",
            f"  Info: {self.by_severity.get('info', 0)}",
        ]

        if self.issues:
            lines.append("\nIssues by category:")
            for cat, count in sorted(self.by_category.items()):
                lines.append(f"  {cat}: {count}")

        return "\n".join(lines)


# Standard tool naming pattern: oci_{domain}_{action}
TOOL_NAME_PATTERN = re.compile(r"^oci_([a-z]+)_([a-z_]+)$")

# Known domains (used for validation)
KNOWN_DOMAINS = {
    "compute",
    "network",
    "database",
    "security",
    "cost",
    "observability",
    "identity",
    "opsi",
    "logan",
    "search",
    "list",
    "get",
    "sqlwatch",
}

# Tools that are exempt from naming validation
EXEMPT_TOOLS = {
    "oci_ping",  # Simple health check
    "execute_sql",  # Legacy name allowed
}

# Tools that should have extended timeouts
SLOW_TOOLS = {
    "oci_cost_get_summary",
    "oci_cost_get_usage_report",
    "oci_cost_get_forecast",
    "oci_search_resources",
    "oci_list_all_resources",
    "oci_opsi_get_fleet_summary",
    "oci_opsi_analyze_cpu",
    "oci_opsi_analyze_memory",
}


class ToolCatalogValidator:
    """
    Validates MCP tool catalog for production readiness.

    Checks:
    - Tool naming conventions (oci_{domain}_{action})
    - Tool tier assignments
    - Timeout configurations
    - Server connectivity
    """

    def __init__(
        self,
        catalog: "ToolCatalog",
        registry: "ServerRegistry | None" = None,
    ):
        """Initialize validator.

        Args:
            catalog: Tool catalog to validate
            registry: Optional server registry for connectivity checks
        """
        self._catalog = catalog
        self._registry = registry
        self._logger = logger.bind(component="ToolCatalogValidator")

    async def validate(self, include_health_check: bool = False) -> ValidationResult:
        """
        Run full validation on the tool catalog.

        Args:
            include_health_check: Also verify tool connectivity

        Returns:
            ValidationResult with all issues found
        """
        tools = self._catalog.list_tools()
        result = ValidationResult(valid=True, total_tools=len(tools))

        self._logger.info("Starting tool catalog validation", tool_count=len(tools))

        # Run all validations
        for tool_def in tools:
            self._validate_naming(tool_def.name, result)
            self._validate_tier(tool_def.name, result)
            self._validate_timeout(tool_def.name, result)
            self._validate_description(tool_def, result)

        # Health check if requested
        if include_health_check and self._registry:
            await self._validate_connectivity(result)

        self._logger.info(
            "Tool catalog validation complete",
            valid=result.valid,
            errors=result.by_severity.get("error", 0),
            warnings=result.by_severity.get("warning", 0),
        )

        return result

    def _validate_naming(self, tool_name: str, result: ValidationResult) -> None:
        """Validate tool follows naming convention."""
        # Skip exempt tools
        if tool_name in EXEMPT_TOOLS:
            return

        # Check standard naming pattern
        match = TOOL_NAME_PATTERN.match(tool_name)
        if not match:
            # Check if it's a legacy name with alias
            from src.mcp.catalog import TOOL_ALIASES
            if tool_name in TOOL_ALIASES:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="naming",
                    tool_name=tool_name,
                    message=f"Legacy tool name has alias: {TOOL_ALIASES[tool_name]}",
                    suggestion=f"Use standardized name: {TOOL_ALIASES[tool_name]}",
                ))
            else:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="naming",
                    tool_name=tool_name,
                    message="Tool name doesn't follow oci_{domain}_{action} pattern",
                    suggestion="Rename to oci_{domain}_{action} format or add to TOOL_ALIASES",
                ))
            return

        # Check domain is known
        domain = match.group(1)
        if domain not in KNOWN_DOMAINS:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="naming",
                tool_name=tool_name,
                message=f"Unknown domain '{domain}' in tool name",
                suggestion=f"Add '{domain}' to KNOWN_DOMAINS if this is a valid domain",
            ))

    def _validate_tier(self, tool_name: str, result: ValidationResult) -> None:
        """Validate tool has a tier assignment."""
        from src.mcp.catalog import TOOL_TIERS, _tier_key

        tier_key = _tier_key(tool_name)
        if tier_key not in TOOL_TIERS:
            # Check if there's a prefix match
            has_prefix_match = any(
                tool_name.startswith(prefix.rstrip("_"))
                for prefix in TOOL_TIERS.keys()
                if prefix.endswith("_")
            )

            if not has_prefix_match:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="tier",
                    tool_name=tool_name,
                    message="Tool has no explicit tier assignment (using default tier 2)",
                    suggestion="Add explicit tier in TOOL_TIERS for better latency estimation",
                ))

    def _validate_timeout(self, tool_name: str, result: ValidationResult) -> None:
        """Validate slow tools have appropriate timeout configuration."""
        if tool_name in SLOW_TOOLS:
            # Check if timeout is configured
            # For now, just log info that it's a slow tool
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="timeout",
                tool_name=tool_name,
                message="Tool is marked as slow - verify timeout is configured appropriately",
                suggestion="Ensure extended timeout (30s+) is configured for this tool",
            ))

    def _validate_description(self, tool_def: Any, result: ValidationResult) -> None:
        """Validate tool has a meaningful description."""
        description = getattr(tool_def, "description", "") or ""

        if len(description) < 10:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="documentation",
                tool_name=tool_def.name,
                message="Tool has missing or very short description",
                suggestion="Add a meaningful description for agent understanding",
            ))

    async def _validate_connectivity(self, result: ValidationResult) -> None:
        """Validate tool connectivity by attempting simple operations."""
        if not self._registry:
            return

        self._logger.info("Running connectivity checks")

        # Group tools by server
        tools_by_server: dict[str, list[str]] = {}
        for tool_def in self._catalog.list_tools():
            server_id = tool_def.server_id
            if server_id not in tools_by_server:
                tools_by_server[server_id] = []
            tools_by_server[server_id].append(tool_def.name)

        # Check each server
        for server_id, tools in tools_by_server.items():
            status = self._registry.get_status(server_id)
            if status != "connected":
                for tool_name in tools:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="connectivity",
                        tool_name=tool_name,
                        message=f"Server {server_id} is {status}",
                        suggestion="Ensure MCP server is running and accessible",
                    ))


async def validate_tool_catalog(
    catalog: "ToolCatalog",
    registry: "ServerRegistry | None" = None,
    include_health_check: bool = False,
) -> ValidationResult:
    """
    Convenience function to validate a tool catalog.

    Args:
        catalog: Tool catalog to validate
        registry: Optional server registry for connectivity checks
        include_health_check: Also verify tool connectivity

    Returns:
        ValidationResult with all issues found
    """
    validator = ToolCatalogValidator(catalog, registry)
    return await validator.validate(include_health_check)


async def validate_server_manifests(
    registry: "ServerRegistry",
    required_fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate that connected servers expose a conforming manifest.

    Args:
        registry: Server registry with connected clients
        required_fields: Required top-level manifest fields

    Returns:
        Validation summary with missing/invalid manifests
    """
    required_fields = required_fields or [
        "schema_version",
        "server_id",
        "server_version",
        "tools",
        "domains",
    ]

    results = {
        "total_servers": len(registry.list_servers()),
        "checked": 0,
        "missing_manifest": [],
        "invalid_manifest": [],
    }

    for server_id in registry.list_servers():
        client = registry.get_client(server_id)
        if not client or not client.connected:
            continue

        results["checked"] += 1
        try:
            resources = client.resources or {}
            if "server://manifest" not in resources:
                results["missing_manifest"].append(server_id)
                continue

            raw = await client.read_resource("server://manifest")
            if not raw:
                results["invalid_manifest"].append(
                    {"server_id": server_id, "error": "empty_manifest"}
                )
                continue

            try:
                manifest = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError as exc:
                results["invalid_manifest"].append(
                    {"server_id": server_id, "error": f"invalid_json: {exc}"}
                )
                continue

            missing = [f for f in required_fields if f not in manifest]
            if missing:
                results["invalid_manifest"].append(
                    {"server_id": server_id, "error": f"missing_fields: {missing}"}
                )

        except Exception as exc:
            results["invalid_manifest"].append(
                {"server_id": server_id, "error": str(exc)}
            )

    return results


async def verify_startup_health(
    catalog: "ToolCatalog",
    test_categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    Verify tool health on startup by testing sample tools.

    Args:
        catalog: Tool catalog to test
        test_categories: Categories to test (None = all)

    Returns:
        Health check results by category
    """
    test_categories = test_categories or [
        "compute",
        "database",
        "cost",
        "security",
        "identity",
    ]

    results: dict[str, Any] = {}
    logger = structlog.get_logger(__name__)

    for category in test_categories:
        # Find a list tool for this category
        test_tool = f"oci_{category}_list_"
        matching_tools = [
            t.name for t in catalog.list_tools()
            if t.name.startswith(test_tool) or t.name == f"oci_list_{category}s"
        ]

        if not matching_tools:
            results[category] = {
                "status": "skipped",
                "message": f"No list tool found for category",
            }
            continue

        # Try the first matching tool
        tool_name = matching_tools[0]
        try:
            result = await catalog.execute(tool_name, {"limit": 1})
            if result.success:
                results[category] = {
                    "status": "healthy",
                    "tool": tool_name,
                    "duration_ms": result.duration_ms,
                }
                logger.info(f"Category {category} OK", tool=tool_name)
            else:
                results[category] = {
                    "status": "error",
                    "tool": tool_name,
                    "error": result.error,
                }
                logger.warning(f"Category {category} FAILED", error=result.error)

        except Exception as e:
            results[category] = {
                "status": "error",
                "tool": tool_name,
                "error": str(e),
            }
            logger.error(f"Category {category} exception", error=str(e))

    return results
