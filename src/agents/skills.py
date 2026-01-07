"""
Skill Execution Framework.

This module provides a reusable framework for defining and executing
agent skills (workflows). Skills are deterministic sequences of steps
that can be shared across agents.

Architecture:
    SkillDefinition -> SkillExecutor -> StepResults

Key Concepts:
- Skills are reusable workflows with defined steps
- Each skill declares required MCP tools
- Skills can be executed by any agent with the necessary tools
- Step results are accumulated and passed to subsequent steps
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog

logger = structlog.get_logger(__name__)


class SkillStatus(str, Enum):
    """Skill execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SkillStep:
    """Definition of a single step in a skill workflow."""

    name: str
    description: str
    required_tools: list[str] = field(default_factory=list)
    optional: bool = False  # If True, failure doesn't stop the workflow
    timeout_seconds: int = 60


@dataclass
class StepResult:
    """Result from executing a skill step."""

    step_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_name": self.step_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tool_calls": self.tool_calls,
        }


@dataclass
class SkillDefinition:
    """
    Definition of a reusable agent skill/workflow.

    Skills encapsulate deterministic workflows that can be:
    - Reused across multiple agents
    - Validated for tool availability
    - Executed with consistent error handling

    Example:
        rca_skill = SkillDefinition(
            name="rca_workflow",
            description="Root cause analysis for database performance",
            steps=[
                SkillStep("detect_symptom", "Identify performance symptoms"),
                SkillStep("check_blocking", "Check for blocking sessions"),
                SkillStep("analyze_wait_events", "Analyze wait event data"),
            ],
            required_tools=["oci_database_get_metrics", "oci_database_query_logs"],
            estimated_duration_seconds=60
        )
    """

    name: str
    description: str
    steps: list[SkillStep]
    required_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    estimated_duration_seconds: int = 60
    max_retries: int = 2

    def validate_tools(self, catalog: ToolCatalog) -> tuple[bool, list[str]]:
        """
        Verify all required tools are available.

        Args:
            catalog: Tool catalog to check against

        Returns:
            Tuple of (is_valid, missing_tools)
        """
        missing = []
        for tool in self.required_tools:
            if not catalog.get_tool(tool):
                missing.append(tool)

        # Also check step-level required tools
        for step in self.steps:
            for tool in step.required_tools:
                if tool not in self.required_tools and not catalog.get_tool(tool):
                    if tool not in missing:
                        missing.append(tool)

        return len(missing) == 0, missing

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "name": s.name,
                    "description": s.description,
                    "required_tools": s.required_tools,
                    "optional": s.optional,
                    "timeout_seconds": s.timeout_seconds,
                }
                for s in self.steps
            ],
            "required_tools": self.required_tools,
            "tags": self.tags,
            "estimated_duration_seconds": self.estimated_duration_seconds,
        }


@dataclass
class SkillExecutionResult:
    """Result from executing a complete skill."""

    skill_name: str
    status: SkillStatus
    step_results: list[StepResult] = field(default_factory=list)
    final_output: Any = None
    total_duration_ms: float = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if skill completed successfully."""
        return self.status == SkillStatus.COMPLETED

    def get_step_result(self, step_name: str) -> StepResult | None:
        """Get result for a specific step."""
        for result in self.step_results:
            if result.step_name == step_name:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skill_name": self.skill_name,
            "status": self.status.value,
            "step_results": [r.to_dict() for r in self.step_results],
            "final_output": self.final_output,
            "total_duration_ms": self.total_duration_ms,
            "error": self.error,
        }


# Type alias for step handlers
StepHandler = Callable[
    [dict[str, Any], list[StepResult]],
    Coroutine[Any, Any, StepResult],
]


class SkillExecutor:
    """
    Executes defined skills/workflows.

    The executor manages:
    - Skill registration and lookup
    - Step-by-step execution with error handling
    - Result accumulation across steps
    - Retry logic for failed steps

    Usage:
        executor = SkillExecutor(catalog)
        executor.register(rca_skill, {
            "detect_symptom": my_detect_handler,
            "check_blocking": my_blocking_handler,
        })

        result = await executor.execute("rca_workflow", context={"db_ocid": "..."})
    """

    def __init__(self, catalog: ToolCatalog):
        """
        Initialize skill executor.

        Args:
            catalog: Tool catalog for tool execution
        """
        self._catalog = catalog
        self._skills: dict[str, SkillDefinition] = {}
        self._handlers: dict[str, dict[str, StepHandler]] = {}
        self._logger = logger.bind(component="SkillExecutor")

    def register(
        self,
        skill: SkillDefinition,
        handlers: dict[str, StepHandler] | None = None,
    ) -> bool:
        """
        Register a skill definition.

        Args:
            skill: Skill definition to register
            handlers: Optional dict of step_name -> handler function

        Returns:
            True if registration successful, False if tools unavailable
        """
        # Validate tools are available
        is_valid, missing = skill.validate_tools(self._catalog)
        if not is_valid:
            self._logger.warning(
                "Skill registration failed: missing tools",
                skill=skill.name,
                missing_tools=missing,
            )
            return False

        self._skills[skill.name] = skill
        if handlers:
            self._handlers[skill.name] = handlers

        self._logger.info(
            "Skill registered",
            skill=skill.name,
            steps=len(skill.steps),
            handlers=list(handlers.keys()) if handlers else [],
        )
        return True

    def unregister(self, skill_name: str) -> None:
        """Unregister a skill."""
        self._skills.pop(skill_name, None)
        self._handlers.pop(skill_name, None)

    def get_skill(self, name: str) -> SkillDefinition | None:
        """Get a registered skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """List all registered skills."""
        return list(self._skills.values())

    def get_skills_by_tag(self, tag: str) -> list[SkillDefinition]:
        """Get skills with a specific tag."""
        return [s for s in self._skills.values() if tag in s.tags]

    async def execute(
        self,
        skill_name: str,
        context: dict[str, Any] | None = None,
        stop_on_failure: bool = True,
    ) -> SkillExecutionResult:
        """
        Execute a registered skill.

        Args:
            skill_name: Name of skill to execute
            context: Initial context data for the skill
            stop_on_failure: Stop execution on first step failure (respects step.optional)

        Returns:
            SkillExecutionResult with all step results
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return SkillExecutionResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED,
                error=f"Skill not found: {skill_name}",
            )

        handlers = self._handlers.get(skill_name, {})
        context = context or {}
        step_results: list[StepResult] = []
        start_time = time.time()

        self._logger.info(
            "Starting skill execution",
            skill=skill_name,
            steps=len(skill.steps),
        )

        status = SkillStatus.RUNNING

        for step in skill.steps:
            # Get handler for this step
            handler = handlers.get(step.name)

            if handler:
                # Execute custom handler
                try:
                    step_result = await asyncio.wait_for(
                        handler(context, step_results),
                        timeout=step.timeout_seconds,
                    )
                except TimeoutError:
                    step_result = StepResult(
                        step_name=step.name,
                        success=False,
                        error=f"Step timed out after {step.timeout_seconds}s",
                    )
                except Exception as e:
                    step_result = StepResult(
                        step_name=step.name,
                        success=False,
                        error=str(e),
                    )
            else:
                # No handler - execute default (tools only)
                step_result = await self._execute_default_step(step, context, step_results)

            step_results.append(step_result)

            # Update context with step result
            context[f"{step.name}_result"] = step_result.result

            self._logger.info(
                "Step completed",
                skill=skill_name,
                step=step.name,
                success=step_result.success,
                duration_ms=step_result.duration_ms,
            )

            # Check if we should stop
            if not step_result.success and stop_on_failure and not step.optional:
                status = SkillStatus.FAILED
                break

        # Determine final status
        if status != SkillStatus.FAILED:
            # Check if all required steps succeeded
            all_required_success = all(
                r.success
                for r, s in zip(step_results, skill.steps)
                if not s.optional
            )
            status = SkillStatus.COMPLETED if all_required_success else SkillStatus.FAILED

        total_duration = (time.time() - start_time) * 1000

        self._logger.info(
            "Skill execution complete",
            skill=skill_name,
            status=status.value,
            duration_ms=total_duration,
        )

        return SkillExecutionResult(
            skill_name=skill_name,
            status=status,
            step_results=step_results,
            total_duration_ms=total_duration,
        )

    async def _execute_default_step(
        self,
        step: SkillStep,
        context: dict[str, Any],
        previous_results: list[StepResult],
    ) -> StepResult:
        """
        Execute a step using its required tools.

        Default execution calls each required tool with context.
        """
        start_time = time.time()
        tool_calls: list[dict[str, Any]] = []
        results: list[Any] = []
        error: str | None = None

        for tool_name in step.required_tools:
            try:
                # Build arguments from context
                arguments = self._build_tool_arguments(tool_name, context)

                result = await self._catalog.execute(tool_name, arguments)
                tool_calls.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "success": result.success,
                })

                if result.success:
                    results.append(result.result)
                else:
                    error = f"Tool {tool_name} failed: {result.error}"
                    break

            except Exception as e:
                error = f"Tool {tool_name} error: {e}"
                tool_calls.append({
                    "tool": tool_name,
                    "error": str(e),
                    "success": False,
                })
                break

        duration_ms = (time.time() - start_time) * 1000

        return StepResult(
            step_name=step.name,
            success=error is None,
            result=results if len(results) > 1 else (results[0] if results else None),
            error=error,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
        )

    def _build_tool_arguments(
        self,
        tool_name: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build tool arguments from context.

        Maps common context keys to tool parameters.
        """
        tool_def = self._catalog.get_tool(tool_name)
        if not tool_def:
            return {}

        schema = tool_def.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        arguments = {}

        # Map context keys to tool parameters
        context_mappings = {
            "db_ocid": ["database_id", "db_ocid", "ocid", "autonomous_database_id"],
            "compartment_id": ["compartment_id", "compartment_ocid"],
            "start_time": ["start_time", "begin_time", "from_time"],
            "end_time": ["end_time", "to_time", "until_time"],
            "instance_ocid": ["instance_id", "instance_ocid", "compute_id"],
        }

        for context_key, possible_params in context_mappings.items():
            if context_key in context:
                for param in possible_params:
                    if param in properties:
                        arguments[param] = context[context_key]
                        break

        # Add any directly matching keys
        for key, value in context.items():
            if key in properties and key not in arguments:
                arguments[key] = value

        return arguments


class SkillRegistry:
    """
    Global registry of available skills.

    Provides skill discovery and lookup across the system.
    """

    _instance: SkillRegistry | None = None
    _skills: dict[str, SkillDefinition] = {}

    @classmethod
    def get_instance(cls) -> SkillRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._skills = {}

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill globally."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_all(self) -> list[SkillDefinition]:
        """List all registered skills."""
        return list(self._skills.values())

    def get_by_tag(self, tag: str) -> list[SkillDefinition]:
        """Get skills with a specific tag."""
        return [s for s in self._skills.values() if tag in s.tags]


# ─────────────────────────────────────────────────────────────────────────────
# Pre-defined Skills
# ─────────────────────────────────────────────────────────────────────────────

# Database Observatory RCA Workflow
# Uses tools from the database-observatory MCP server
DB_RCA_WORKFLOW = SkillDefinition(
    name="db_rca_workflow",
    description="7-step root cause analysis using Database Observatory MCP server",
    steps=[
        SkillStep(
            name="discover_database",
            description="Find and validate the target database",
            required_tools=["search_databases", "get_cached_database"],
            timeout_seconds=15,
        ),
        SkillStep(
            name="get_performance_overview",
            description="Get combined CPU, memory, I/O performance summary",
            required_tools=["get_performance_summary"],
            timeout_seconds=60,
        ),
        SkillStep(
            name="analyze_cpu_usage",
            description="Deep-dive into CPU usage trends",
            required_tools=["analyze_cpu_usage"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="analyze_memory_usage",
            description="Analyze memory utilization patterns",
            required_tools=["analyze_memory_usage"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="analyze_io_performance",
            description="Check I/O throughput and bottlenecks",
            required_tools=["analyze_io_performance"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="check_wait_events",
            description="Query database for wait events (requires SQL access)",
            required_tools=["execute_sql"],
            optional=True,  # May fail if no SQLcl connection
            timeout_seconds=60,
        ),
        SkillStep(
            name="generate_recommendations",
            description="Generate findings report with recommendations",
            required_tools=[],  # Pure analysis step
            timeout_seconds=15,
        ),
    ],
    required_tools=[
        "search_databases",
        "get_performance_summary",
        "analyze_cpu_usage",
    ],
    tags=["database", "performance", "troubleshooting", "database-observatory"],
    estimated_duration_seconds=180,
)

# Quick Database Health Check Workflow
DB_HEALTH_CHECK_WORKFLOW = SkillDefinition(
    name="db_health_check_workflow",
    description="Fast health check using cached data and OPSI metrics",
    steps=[
        SkillStep(
            name="refresh_cache",
            description="Ensure database cache is up-to-date",
            required_tools=["refresh_cache_if_needed"],
            timeout_seconds=60,
        ),
        SkillStep(
            name="get_fleet_overview",
            description="Get fleet summary with database counts",
            required_tools=["get_fleet_summary"],
            timeout_seconds=10,
        ),
        SkillStep(
            name="identify_issues",
            description="Find databases with potential issues",
            required_tools=["find_cost_opportunities"],
            timeout_seconds=30,
        ),
    ],
    required_tools=["get_fleet_summary", "refresh_cache_if_needed"],
    tags=["database", "health", "quick-check"],
    estimated_duration_seconds=60,
)

# Database SQL Analysis Workflow
DB_SQL_ANALYSIS_WORKFLOW = SkillDefinition(
    name="db_sql_analysis_workflow",
    description="SQL-level analysis using SQLcl for deep database inspection",
    steps=[
        SkillStep(
            name="check_connections",
            description="List available database connections",
            required_tools=["list_connections"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_instance_status",
            description="Check database instance status",
            required_tools=["database_status"],
            timeout_seconds=60,
        ),
        SkillStep(
            name="query_top_sql",
            description="Find top SQL by elapsed time",
            required_tools=["execute_sql"],
            timeout_seconds=90,
        ),
        SkillStep(
            name="analyze_wait_events",
            description="Query active session history for wait events",
            required_tools=["execute_sql"],
            timeout_seconds=90,
        ),
        SkillStep(
            name="check_blocking",
            description="Find blocking sessions",
            required_tools=["execute_sql"],
            timeout_seconds=60,
        ),
    ],
    required_tools=["execute_sql", "list_connections"],
    tags=["database", "sql", "deep-analysis"],
    estimated_duration_seconds=300,
)

# AWR Report Workflow - generates AWR HTML reports
DB_AWR_REPORT_WORKFLOW = SkillDefinition(
    name="db_awr_report_workflow",
    description="Generate AWR report from DB Management or SQLcl",
    steps=[
        SkillStep(
            name="find_database",
            description="Find database by name using cache or DB Management",
            required_tools=["search_databases", "search_managed_databases"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_snapshots",
            description="List available AWR snapshots",
            required_tools=["list_awr_db_snapshots"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="generate_report",
            description="Generate AWR HTML report",
            required_tools=["get_awr_db_report_auto"],
            timeout_seconds=120,
        ),
    ],
    required_tools=[
        "search_databases",
        "search_managed_databases",
        "list_awr_db_snapshots",
        "get_awr_db_report_auto",
    ],
    tags=["database", "awr", "performance", "report"],
    estimated_duration_seconds=180,
)

# Legacy RCA Workflow (for compatibility)
RCA_WORKFLOW = SkillDefinition(
    name="rca_workflow",
    description="7-step root cause analysis for database performance issues",
    steps=[
        SkillStep(
            name="detect_symptom",
            description="Identify the primary performance symptom",
            required_tools=["get_performance_summary"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="check_blocking",
            description="Check for blocking sessions and lock contention",
            required_tools=["execute_sql"],
            optional=True,
            timeout_seconds=30,
        ),
        SkillStep(
            name="analyze_wait_events",
            description="Analyze wait event distribution",
            required_tools=["execute_sql"],
            optional=True,
            timeout_seconds=45,
        ),
        SkillStep(
            name="check_sql_performance",
            description="Identify problematic SQL statements",
            required_tools=["execute_sql"],
            optional=True,
            timeout_seconds=45,
        ),
        SkillStep(
            name="check_longops",
            description="Check for long-running operations",
            required_tools=["execute_sql"],
            optional=True,
            timeout_seconds=30,
        ),
        SkillStep(
            name="check_parallel_queries",
            description="Analyze parallel query execution",
            required_tools=["execute_sql"],
            optional=True,
            timeout_seconds=30,
        ),
        SkillStep(
            name="generate_report",
            description="Generate findings report with recommendations",
            required_tools=[],
            timeout_seconds=30,
        ),
    ],
    required_tools=["get_performance_summary"],
    tags=["database", "performance", "troubleshooting"],
    estimated_duration_seconds=180,
)

# Cost Analysis Workflow
COST_ANALYSIS_WORKFLOW = SkillDefinition(
    name="cost_analysis_workflow",
    description="Analyze cloud spending and identify optimization opportunities",
    steps=[
        SkillStep(
            name="get_cost_summary",
            description="Retrieve cost summary for the period",
            required_tools=["oci_cost_get_summary"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="identify_top_spenders",
            description="Identify top cost contributors",
            required_tools=[],
            timeout_seconds=20,
        ),
        SkillStep(
            name="analyze_trends",
            description="Analyze spending trends over time",
            required_tools=[],
            timeout_seconds=30,
        ),
        SkillStep(
            name="generate_recommendations",
            description="Generate cost optimization recommendations",
            required_tools=[],
            timeout_seconds=30,
        ),
    ],
    required_tools=["oci_cost_get_summary"],
    tags=["finops", "cost", "optimization"],
    estimated_duration_seconds=90,
)

# Security Assessment Workflow
SECURITY_ASSESSMENT_WORKFLOW = SkillDefinition(
    name="security_assessment_workflow",
    description="Assess security posture and identify vulnerabilities",
    steps=[
        SkillStep(
            name="list_security_problems",
            description="List active security problems from Cloud Guard",
            required_tools=["oci_security_cloudguard_list_problems"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="prioritize_findings",
            description="Prioritize findings by severity and impact",
            required_tools=[],
            timeout_seconds=20,
        ),
        SkillStep(
            name="map_to_mitre",
            description="Map findings to MITRE ATT&CK framework",
            required_tools=[],
            timeout_seconds=30,
        ),
        SkillStep(
            name="generate_remediation",
            description="Generate remediation recommendations",
            required_tools=[],
            timeout_seconds=30,
        ),
    ],
    required_tools=["oci_security_cloudguard_list_problems"],
    tags=["security", "compliance", "assessment"],
    estimated_duration_seconds=90,
)


SECURITY_POSTURE_WORKFLOW = SkillDefinition(
    name="security_posture_workflow",
    description="Summarize Cloud Guard posture with recommendations",
    steps=[
        SkillStep(
            name="list_cloud_guard_problems",
            description="List Cloud Guard security problems",
            required_tools=["oci_security_cloudguard_list_problems"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_security_score",
            description="Retrieve Cloud Guard security score",
            required_tools=["oci_security_cloudguard_get_security_score"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_recommendations",
            description="List Cloud Guard recommendations",
            required_tools=["oci_security_cloudguard_list_recommendations"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="posture_summary",
            description="Generate posture summary using built-in skill",
            required_tools=["oci_security_skill_posture_summary"],
            timeout_seconds=60,
        ),
    ],
    required_tools=[
        "oci_security_cloudguard_list_problems",
        "oci_security_cloudguard_get_security_score",
        "oci_security_cloudguard_list_recommendations",
        "oci_security_skill_posture_summary",
    ],
    tags=["security", "cloudguard", "posture"],
    estimated_duration_seconds=120,
)


SECURITY_CLOUDGUARD_INVESTIGATION_WORKFLOW = SkillDefinition(
    name="security_cloudguard_investigation_workflow",
    description="Investigate Cloud Guard findings and responder context",
    steps=[
        SkillStep(
            name="get_problem_details",
            description="Get detailed Cloud Guard problem information",
            required_tools=["oci_security_cloudguard_get_problem"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_detectors",
            description="List detector recipes for context",
            required_tools=["oci_security_cloudguard_list_detectors"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_responders",
            description="List responder recipes for context",
            required_tools=["oci_security_cloudguard_list_responders"],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_cloudguard_get_problem",
        "oci_security_cloudguard_list_detectors",
        "oci_security_cloudguard_list_responders",
    ],
    tags=["security", "cloudguard", "investigation"],
    estimated_duration_seconds=120,
)


SECURITY_CLOUDGUARD_REMEDIATION_WORKFLOW = SkillDefinition(
    name="security_cloudguard_remediation_workflow",
    description="Remediate Cloud Guard problems (requires explicit confirmation)",
    steps=[
        SkillStep(
            name="remediate_problem",
            description="Apply remediation to a Cloud Guard problem",
            required_tools=["oci_security_cloudguard_remediate_problem"],
            timeout_seconds=60,
        ),
    ],
    required_tools=["oci_security_cloudguard_remediate_problem"],
    tags=["security", "cloudguard", "remediation"],
    estimated_duration_seconds=60,
)


SECURITY_VULNERABILITY_WORKFLOW = SkillDefinition(
    name="security_vulnerability_workflow",
    description="Review host and container vulnerability scans",
    steps=[
        SkillStep(
            name="list_host_scans",
            description="List host vulnerability scans",
            required_tools=["oci_security_vss_list_host_scans"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_host_scan",
            description="Get host scan details",
            required_tools=["oci_security_vss_get_host_scan"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_container_scans",
            description="List container vulnerability scans",
            required_tools=["oci_security_vss_list_container_scans"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_vulnerabilities",
            description="List detected vulnerabilities",
            required_tools=["oci_security_vss_list_vulnerabilities"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="vulnerability_summary",
            description="Generate vulnerability overview using built-in skill",
            required_tools=["oci_security_skill_vulnerability_overview"],
            timeout_seconds=60,
        ),
    ],
    required_tools=[
        "oci_security_vss_list_host_scans",
        "oci_security_vss_get_host_scan",
        "oci_security_vss_list_container_scans",
        "oci_security_vss_list_vulnerabilities",
        "oci_security_skill_vulnerability_overview",
    ],
    tags=["security", "vss", "vulnerability"],
    estimated_duration_seconds=150,
)


SECURITY_ZONE_COMPLIANCE_WORKFLOW = SkillDefinition(
    name="security_zone_compliance_workflow",
    description="Evaluate Security Zones compliance and policies",
    steps=[
        SkillStep(
            name="list_security_zones",
            description="List Security Zones",
            required_tools=["oci_security_zones_list_zones"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_security_zone",
            description="Get Security Zone details",
            required_tools=["oci_security_zones_get_zone"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_zone_policies",
            description="List Security Zone policies",
            required_tools=["oci_security_zones_list_policies"],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_zones_list_zones",
        "oci_security_zones_get_zone",
        "oci_security_zones_list_policies",
    ],
    tags=["security", "zones", "compliance"],
    estimated_duration_seconds=120,
)


SECURITY_BASTION_AUDIT_WORKFLOW = SkillDefinition(
    name="security_bastion_audit_workflow",
    description="Audit bastion hosts and active sessions",
    steps=[
        SkillStep(
            name="list_bastions",
            description="List bastion resources",
            required_tools=["oci_security_bastion_list"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_bastion_details",
            description="Get bastion details",
            required_tools=["oci_security_bastion_get"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_bastion_sessions",
            description="List active bastion sessions",
            required_tools=["oci_security_bastion_list_sessions"],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_bastion_list",
        "oci_security_bastion_get",
        "oci_security_bastion_list_sessions",
    ],
    tags=["security", "bastion", "access"],
    estimated_duration_seconds=120,
)


SECURITY_BASTION_SESSION_CLEANUP_WORKFLOW = SkillDefinition(
    name="security_bastion_session_cleanup_workflow",
    description="Terminate bastion sessions (requires explicit confirmation)",
    steps=[
        SkillStep(
            name="terminate_session",
            description="Terminate a bastion session",
            required_tools=["oci_security_bastion_terminate_session"],
            timeout_seconds=60,
        ),
    ],
    required_tools=["oci_security_bastion_terminate_session"],
    tags=["security", "bastion", "remediation"],
    estimated_duration_seconds=60,
)


SECURITY_DATASAFE_ASSESSMENT_WORKFLOW = SkillDefinition(
    name="security_datasafe_assessment_workflow",
    description="Review Data Safe targets, assessments, and findings",
    steps=[
        SkillStep(
            name="list_datasafe_targets",
            description="List Data Safe target databases",
            required_tools=["oci_security_datasafe_list_targets"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_datasafe_assessments",
            description="List Data Safe assessments",
            required_tools=["oci_security_datasafe_list_assessments"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_datasafe_assessment",
            description="Get Data Safe assessment details",
            required_tools=["oci_security_datasafe_get_assessment"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_datasafe_findings",
            description="List Data Safe findings",
            required_tools=["oci_security_datasafe_list_findings"],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_datasafe_list_targets",
        "oci_security_datasafe_list_assessments",
        "oci_security_datasafe_get_assessment",
        "oci_security_datasafe_list_findings",
    ],
    tags=["security", "datasafe", "database"],
    estimated_duration_seconds=150,
)


SECURITY_WAF_POLICY_WORKFLOW = SkillDefinition(
    name="security_waf_policy_workflow",
    description="Review WAF firewalls and policies",
    steps=[
        SkillStep(
            name="list_waf_firewalls",
            description="List WAF firewalls",
            required_tools=["oci_security_waf_list_firewalls"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_waf_firewall",
            description="Get WAF firewall details",
            required_tools=["oci_security_waf_get_firewall"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_waf_policies",
            description="List WAF policies",
            required_tools=["oci_security_waf_list_policies"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_waf_policy",
            description="Get WAF policy details",
            required_tools=["oci_security_waf_get_policy"],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_waf_list_firewalls",
        "oci_security_waf_get_firewall",
        "oci_security_waf_list_policies",
        "oci_security_waf_get_policy",
    ],
    tags=["security", "waf", "policy"],
    estimated_duration_seconds=150,
)


SECURITY_AUDIT_ACTIVITY_WORKFLOW = SkillDefinition(
    name="security_audit_activity_workflow",
    description="Review audit events and configuration",
    steps=[
        SkillStep(
            name="list_audit_events",
            description="List audit events",
            required_tools=["oci_security_audit_list_events"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_audit_configuration",
            description="Get audit configuration",
            required_tools=["oci_security_audit_get_configuration"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="audit_digest",
            description="Generate audit activity digest",
            required_tools=["oci_security_skill_audit_digest"],
            timeout_seconds=60,
        ),
    ],
    required_tools=[
        "oci_security_audit_list_events",
        "oci_security_audit_get_configuration",
        "oci_security_skill_audit_digest",
    ],
    tags=["security", "audit", "compliance"],
    estimated_duration_seconds=120,
)


SECURITY_ACCESS_GOVERNANCE_WORKFLOW = SkillDefinition(
    name="security_access_governance_workflow",
    description="Review access governance instances",
    steps=[
        SkillStep(
            name="list_access_governance_instances",
            description="List access governance instances",
            required_tools=["oci_security_accessgov_list_instances"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_access_governance_instance",
            description="Get access governance instance details",
            required_tools=["oci_security_accessgov_get_instance"],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_accessgov_list_instances",
        "oci_security_accessgov_get_instance",
    ],
    tags=["security", "access-governance", "identity"],
    estimated_duration_seconds=90,
)


SECURITY_KMS_INVENTORY_WORKFLOW = SkillDefinition(
    name="security_kms_inventory_workflow",
    description="Inventory KMS vaults and keys",
    steps=[
        SkillStep(
            name="list_kms_vaults",
            description="List KMS vaults",
            required_tools=["oci_security_kms_list_vaults"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_kms_vault",
            description="Get KMS vault details",
            required_tools=["oci_security_kms_get_vault"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_kms_keys",
            description="List KMS keys",
            required_tools=["oci_security_kms_list_keys"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_kms_key",
            description="Get KMS key details",
            required_tools=["oci_security_kms_get_key"],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_kms_list_vaults",
        "oci_security_kms_get_vault",
        "oci_security_kms_list_keys",
        "oci_security_kms_get_key",
    ],
    tags=["security", "kms", "encryption"],
    estimated_duration_seconds=120,
)


SECURITY_IAM_REVIEW_WORKFLOW = SkillDefinition(
    name="security_iam_review_workflow",
    description="Review IAM users, groups, and policies",
    steps=[
        SkillStep(
            name="list_users",
            description="List IAM users",
            required_tools=["oci_security_list_users"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_groups",
            description="List IAM groups",
            required_tools=["oci_security_list_groups"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_policies",
            description="List IAM policies",
            required_tools=["oci_security_list_policies"],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_list_users",
        "oci_security_list_groups",
        "oci_security_list_policies",
    ],
    tags=["security", "iam", "policy"],
    estimated_duration_seconds=120,
)

# =============================================================================
# Infrastructure Agent Workflows (oci-infrastructure MCP server)
# =============================================================================

INFRA_INVENTORY_WORKFLOW = SkillDefinition(
    name="infra_inventory_workflow",
    description="Gather comprehensive infrastructure inventory (compute, network, security)",
    steps=[
        SkillStep(
            name="list_instances",
            description="List all compute instances in compartment",
            required_tools=["list_instances"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_vcns",
            description="List Virtual Cloud Networks",
            required_tools=["oci_network_list_vcns"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_subnets",
            description="List subnets across VCNs",
            required_tools=["oci_network_list_subnets"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="summarize_inventory",
            description="Summarize infrastructure resources",
            required_tools=[],
            timeout_seconds=10,
        ),
    ],
    required_tools=["list_instances", "oci_network_list_vcns", "oci_network_list_subnets"],
    tags=["infrastructure", "inventory", "compute", "network"],
    estimated_duration_seconds=120,
)


INFRA_INSTANCE_MANAGEMENT_WORKFLOW = SkillDefinition(
    name="infra_instance_management_workflow",
    description="Manage compute instance lifecycle (start, stop, restart)",
    steps=[
        SkillStep(
            name="get_instance_status",
            description="Get current instance state and metrics",
            required_tools=["list_instances", "get_instance_metrics"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="validate_action",
            description="Validate requested action is safe",
            required_tools=[],
            timeout_seconds=10,
        ),
        SkillStep(
            name="execute_action",
            description="Execute instance action (requires ALLOW_MUTATIONS=true)",
            required_tools=["start_instance", "stop_instance", "restart_instance"],
            optional=True,  # May fail if mutations disabled
            timeout_seconds=60,
        ),
        SkillStep(
            name="verify_result",
            description="Verify action completed successfully",
            required_tools=["list_instances"],
            timeout_seconds=30,
        ),
    ],
    required_tools=["list_instances"],
    tags=["infrastructure", "compute", "instance", "lifecycle"],
    estimated_duration_seconds=150,
)


INFRA_NETWORK_ANALYSIS_WORKFLOW = SkillDefinition(
    name="infra_network_analysis_workflow",
    description="Analyze network topology and security configuration",
    steps=[
        SkillStep(
            name="list_vcns",
            description="List Virtual Cloud Networks",
            required_tools=["oci_network_list_vcns"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="get_vcn_details",
            description="Get VCN details including routing",
            required_tools=["oci_network_get_vcn"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_subnets",
            description="List subnets and their configurations",
            required_tools=["oci_network_list_subnets"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_security_lists",
            description="List security lists and rules",
            required_tools=["oci_network_list_security_lists"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="analyze_security",
            description="Analyze network security posture",
            required_tools=["oci_network_analyze_security"],
            timeout_seconds=60,
        ),
    ],
    required_tools=[
        "oci_network_list_vcns",
        "oci_network_list_subnets",
        "oci_network_list_security_lists",
    ],
    tags=["infrastructure", "network", "vcn", "security"],
    estimated_duration_seconds=180,
)


INFRA_SECURITY_AUDIT_WORKFLOW = SkillDefinition(
    name="infra_security_audit_workflow",
    description="Comprehensive security audit including IAM, Cloud Guard, and policies",
    steps=[
        SkillStep(
            name="list_cloud_guard_problems",
            description="List active Cloud Guard security problems",
            required_tools=["oci_security_list_cloud_guard_problems"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="list_policies",
            description="List IAM policies",
            required_tools=["oci_security_list_policies"],
            timeout_seconds=30,
        ),
        SkillStep(
            name="analyze_network_security",
            description="Analyze network security configuration",
            required_tools=["oci_network_analyze_security"],
            timeout_seconds=60,
        ),
        SkillStep(
            name="run_security_audit",
            description="Run comprehensive security audit",
            required_tools=["oci_security_audit"],
            optional=True,  # May be slow
            timeout_seconds=120,
        ),
        SkillStep(
            name="generate_report",
            description="Generate security audit report",
            required_tools=[],
            timeout_seconds=30,
        ),
    ],
    required_tools=[
        "oci_security_list_cloud_guard_problems",
        "oci_security_list_policies",
    ],
    tags=["infrastructure", "security", "audit", "compliance", "cloudguard"],
    estimated_duration_seconds=300,
)


def register_default_skills() -> None:
    """Register all default skills to the global registry."""
    registry = SkillRegistry.get_instance()

    # Database Observatory skills (primary)
    registry.register(DB_RCA_WORKFLOW)
    registry.register(DB_HEALTH_CHECK_WORKFLOW)
    registry.register(DB_SQL_ANALYSIS_WORKFLOW)
    registry.register(DB_AWR_REPORT_WORKFLOW)

    # Infrastructure skills (oci-infrastructure MCP server)
    registry.register(INFRA_INVENTORY_WORKFLOW)
    registry.register(INFRA_INSTANCE_MANAGEMENT_WORKFLOW)
    registry.register(INFRA_NETWORK_ANALYSIS_WORKFLOW)
    registry.register(INFRA_SECURITY_AUDIT_WORKFLOW)

    # Legacy skills (for compatibility)
    registry.register(RCA_WORKFLOW)
    registry.register(COST_ANALYSIS_WORKFLOW)
    registry.register(SECURITY_ASSESSMENT_WORKFLOW)
    registry.register(SECURITY_POSTURE_WORKFLOW)
    registry.register(SECURITY_CLOUDGUARD_INVESTIGATION_WORKFLOW)
    registry.register(SECURITY_CLOUDGUARD_REMEDIATION_WORKFLOW)
    registry.register(SECURITY_VULNERABILITY_WORKFLOW)
    registry.register(SECURITY_ZONE_COMPLIANCE_WORKFLOW)
    registry.register(SECURITY_BASTION_AUDIT_WORKFLOW)
    registry.register(SECURITY_BASTION_SESSION_CLEANUP_WORKFLOW)
    registry.register(SECURITY_DATASAFE_ASSESSMENT_WORKFLOW)
    registry.register(SECURITY_WAF_POLICY_WORKFLOW)
    registry.register(SECURITY_AUDIT_ACTIVITY_WORKFLOW)
    registry.register(SECURITY_ACCESS_GOVERNANCE_WORKFLOW)
    registry.register(SECURITY_KMS_INVENTORY_WORKFLOW)
    registry.register(SECURITY_IAM_REVIEW_WORKFLOW)
