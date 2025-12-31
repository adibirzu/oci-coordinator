"""
Database Troubleshooting Agent with Database Observatory MCP Integration.

Specialized agent for Oracle Database performance analysis and troubleshooting
using the Database Observatory MCP server (OPSI, SQLcl, Logan Analytics).

Key Features:
- Uses tiered MCP tools (cache-based, OPSI API, SQL execution)
- Full observability integration (OCI APM + OCI Logging)
- Deterministic workflow skills with LLM analysis
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
from langgraph.graph import END, StateGraph
from opentelemetry import trace

from src.agents.base import (
    AgentDefinition,
    AgentMetadata,
    BaseAgent,
    KafkaTopics,
)
from src.agents.skills import (
    DB_RCA_WORKFLOW,
    DB_HEALTH_CHECK_WORKFLOW,
    SkillExecutionResult,
    SkillStatus,
    StepResult,
)
from src.observability import get_trace_id

if TYPE_CHECKING:
    from src.memory.manager import SharedMemoryManager
    from src.mcp.catalog import ToolCatalog

logger = structlog.get_logger(__name__)


class AnalysisSeverity(str, Enum):
    """Severity levels for database issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    HEALTHY = "healthy"


@dataclass
class ProblemArea:
    """A specific problem area in the database."""

    area: str  # CPU, Memory, I/O, Locks, etc.
    status: str  # critical, warning, ok
    value: str  # Current value
    threshold: str  # Threshold value
    details: str  # Additional details


@dataclass
class TopIssue:
    """A top issue found during analysis."""

    rank: int
    issue: str
    impact: str
    wait_event: str | None = None
    sql_id: str | None = None
    sql_preview: str | None = None


@dataclass
class Recommendation:
    """A recommendation for fixing an issue."""

    priority: int
    action: str
    type: str  # sql_tuning, scaling, configuration, etc.
    impact: str
    effort: str  # low, medium, high
    command: str | None = None


@dataclass
class DbAnalysisResult:
    """Result of database analysis."""

    summary: str
    health_score: int  # 0-100
    severity: AnalysisSeverity
    problem_areas: list[ProblemArea] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    top_issues: list[TopIssue] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.summary,
            "health_score": self.health_score,
            "severity": self.severity.value,
            "problem_areas": [
                {
                    "area": p.area,
                    "status": p.status,
                    "value": p.value,
                    "threshold": p.threshold,
                    "details": p.details,
                }
                for p in self.problem_areas
            ],
            "metrics": self.metrics,
            "top_issues": [
                {
                    "rank": i.rank,
                    "issue": i.issue,
                    "impact": i.impact,
                    "wait_event": i.wait_event,
                    "sql_id": i.sql_id,
                    "sql_preview": i.sql_preview,
                }
                for i in self.top_issues
            ],
            "recommendations": [
                {
                    "priority": r.priority,
                    "action": r.action,
                    "type": r.type,
                    "impact": r.impact,
                    "effort": r.effort,
                    "command": r.command,
                }
                for r in self.recommendations
            ],
            "next_steps": self.next_steps,
            "trace_id": self.trace_id,
        }


@dataclass
class TroubleshootState:
    """State for the troubleshooting workflow."""

    query: str = ""
    database_id: str | None = None
    database_name: str | None = None
    database_type: str | None = None  # ADB-S, ADB-D, DBCS, ExaCS
    compartment_id: str | None = None
    region: str | None = None

    # Phase tracking
    phase: str = "discover"
    iteration: int = 0
    max_iterations: int = 10

    # Collected data from MCP tools
    fleet_summary: dict[str, Any] = field(default_factory=dict)
    performance_summary: dict[str, Any] = field(default_factory=dict)
    cpu_analysis: dict[str, Any] = field(default_factory=dict)
    memory_analysis: dict[str, Any] = field(default_factory=dict)
    io_analysis: dict[str, Any] = field(default_factory=dict)

    # SQL-level data (if SQLcl available)
    wait_events: list[dict] = field(default_factory=list)
    top_sql: list[dict] = field(default_factory=list)
    blocking_sessions: list[dict] = field(default_factory=list)

    # Skill execution tracking
    skill_results: list[StepResult] = field(default_factory=list)

    # Analysis result
    result: DbAnalysisResult | None = None
    error: str | None = None


# System prompt for the DB Troubleshoot Agent
DB_TROUBLESHOOT_SYSTEM_PROMPT = """You are the OCI Database Troubleshooting Agent, a specialized AI expert in Oracle Database performance analysis.

Your expertise includes:
- Oracle Database internals and wait events
- Autonomous Database (ADB) performance
- DB Systems (DBCS) troubleshooting
- AWR and ASH analysis
- SQL performance tuning
- Resource bottleneck identification

## Available MCP Tools (Database Observatory)

### Tier 1: Cache-Based (Instant, <100ms)
- `get_fleet_summary`: Get fleet overview with database counts
- `search_databases`: Find databases by name/type/compartment
- `get_cached_database`: Get database details from cache
- `get_cached_statistics`: Get cache statistics

### Tier 2: OPSI API (1-5s)
- `analyze_cpu_usage`: CPU usage trends with recommendations
- `analyze_memory_usage`: Memory utilization analysis
- `analyze_io_performance`: I/O throughput analysis
- `get_performance_summary`: Combined CPU/memory/I/O summary
- `find_cost_opportunities`: Identify cost savings opportunities

### Tier 3: SQL Execution (5-30s)
- `execute_sql`: Run SQL queries via SQLcl
- `get_schema_info`: Get schema metadata
- `database_status`: Check connection status

## Troubleshooting Methodology

1. **Discovery**: Find the target database using cache-based tools
2. **Overview**: Get performance summary (CPU, memory, I/O)
3. **Deep-dive**: Analyze specific metrics with issues
4. **SQL Analysis**: Query wait events, top SQL if needed
5. **Recommendations**: Generate prioritized action items

## Critical Thresholds
- CPU Utilization > 90% for 5+ minutes → Scale OCPU
- Memory Usage > 85% → Review SGA/PGA allocation
- I/O Spikes > 3x average → Storage optimization needed
- Wait Time % > 50% → Wait event deep-dive required

Always provide structured analysis with health scores (0-100) and severity levels."""


class DbTroubleshootAgent(BaseAgent):
    """
    Database Troubleshooting Agent with MCP Integration.

    Uses the Database Observatory MCP server for:
    - OPSI operations insights (CPU, memory, I/O analysis)
    - SQLcl for direct database queries
    - Logan Analytics for log correlation

    Workflow:
    1. Discover database using cache
    2. Get performance overview from OPSI
    3. Deep-dive into problem areas
    4. Query database if SQLcl available
    5. Generate recommendations
    """

    # MCP tools from Database Observatory server
    MCP_TOOLS = [
        # Tier 1: Cache-based
        "get_fleet_summary",
        "search_databases",
        "get_cached_database",
        "get_cached_statistics",
        "refresh_cache_if_needed",
        # Tier 2: OPSI API
        "analyze_cpu_usage",
        "analyze_memory_usage",
        "analyze_io_performance",
        "get_performance_summary",
        "find_cost_opportunities",
        "list_database_insights",
        # Tier 3: SQLcl
        "execute_sql",
        "get_schema_info",
        "list_connections",
        "database_status",
        "check_network_access",
    ]

    @classmethod
    def get_definition(cls) -> AgentDefinition:
        """Return agent definition for catalog registration."""
        return AgentDefinition(
            agent_id="db-troubleshoot-agent",
            role="db-troubleshoot-agent",
            capabilities=[
                "database-analysis",
                "performance-diagnostics",
                "sql-tuning",
                "blocking-analysis",
                "wait-event-analysis",
                "cost-optimization",
            ],
            skills=[
                "db_rca_workflow",
                "db_health_check_workflow",
                "db_sql_analysis_workflow",
                "rca_workflow",  # Legacy
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.db-troubleshoot-agent"],
                produce=["results.db-troubleshoot-agent"],
            ),
            health_endpoint="http://localhost:8010/health",
            metadata=AgentMetadata(
                version="2.0.0",  # Updated for MCP integration
                namespace="oci-coordinator",
                max_iterations=15,
                timeout_seconds=120,
            ),
            description=(
                "Database Expert Agent for Oracle performance analysis using "
                "Database Observatory MCP server. Provides RCA using OPSI metrics, "
                "SQLcl queries, and Logan log analysis."
            ),
            mcp_tools=cls.MCP_TOOLS,
            mcp_servers=["database-observatory"],
        )

    def __init__(
        self,
        memory_manager: "SharedMemoryManager | None" = None,
        tool_catalog: "ToolCatalog | None" = None,
        config: dict[str, Any] | None = None,
        llm: Any = None,
    ):
        """
        Initialize DB Troubleshoot Agent.

        Args:
            memory_manager: Shared memory manager
            tool_catalog: Tool catalog for MCP tools
            config: Agent configuration
            llm: LangChain LLM for analysis
        """
        super().__init__(memory_manager, tool_catalog, config)
        self.llm = llm
        self._graph: StateGraph | None = None
        self._tracer = trace.get_tracer("oci-db-troubleshoot-agent")

    def build_graph(self) -> StateGraph:
        """
        Build the troubleshooting workflow graph.

        Graph structure uses Database Observatory tools:
        discover → performance_overview → analyze_metrics →
        sql_analysis (optional) → generate_recommendations → output
        """
        graph = StateGraph(TroubleshootState)

        # Add nodes using MCP tools
        graph.add_node("discover", self._discover_node)
        graph.add_node("performance_overview", self._performance_overview_node)
        graph.add_node("analyze_metrics", self._analyze_metrics_node)
        graph.add_node("sql_analysis", self._sql_analysis_node)
        graph.add_node("generate_recommendations", self._generate_recommendations_node)
        graph.add_node("output", self._output_node)

        # Set entry point
        graph.set_entry_point("discover")

        # Add edges
        graph.add_edge("discover", "performance_overview")
        graph.add_edge("performance_overview", "analyze_metrics")
        graph.add_edge("analyze_metrics", "sql_analysis")
        graph.add_edge("sql_analysis", "generate_recommendations")
        graph.add_edge("generate_recommendations", "output")
        graph.add_edge("output", END)

        self._graph = graph.compile()
        return self._graph

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call an MCP tool with tracing and error handling.

        Args:
            tool_name: Name of the MCP tool
            arguments: Tool arguments

        Returns:
            Tool result or error dict
        """
        with self._tracer.start_as_current_span(f"mcp.tool.{tool_name}") as span:
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.tool.args", str(arguments)[:200])

            start_time = time.time()
            try:
                if not self.tools:
                    return {"success": False, "error": "Tool catalog not initialized"}

                result = await self.call_tool(tool_name, arguments)

                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("mcp.tool.duration_ms", duration_ms)
                span.set_attribute("mcp.tool.success", True)

                self._logger.info(
                    "MCP tool call completed",
                    tool=tool_name,
                    duration_ms=duration_ms,
                    trace_id=get_trace_id(),
                )

                return result if isinstance(result, dict) else {"result": result}

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("mcp.tool.success", False)
                span.set_attribute("mcp.tool.error", str(e)[:200])

                self._logger.error(
                    "MCP tool call failed",
                    tool=tool_name,
                    error=str(e),
                    duration_ms=duration_ms,
                    trace_id=get_trace_id(),
                )

                return {"success": False, "error": str(e)}

    async def _discover_node(self, state: TroubleshootState) -> dict[str, Any]:
        """Discover database using cache-based tools."""
        with self._tracer.start_as_current_span("discover_database") as span:
            self._logger.info(
                "Discovering database",
                query=state.query[:100],
                trace_id=get_trace_id(),
            )

            # Extract database name from query
            db_name = self._extract_db_name(state.query)
            span.set_attribute("db.name_hint", db_name or "none")

            # If database_id provided, get details from cache
            if state.database_id:
                result = await self._call_mcp_tool(
                    "get_cached_database",
                    {"database_id": state.database_id},
                )
                if result.get("success"):
                    db = result.get("database", {})
                    return {
                        "database_name": db.get("database_name"),
                        "database_type": db.get("database_type"),
                        "phase": "performance_overview",
                        "iteration": state.iteration + 1,
                    }

            # Search by name if provided
            if db_name:
                result = await self._call_mcp_tool(
                    "search_databases",
                    {"name": db_name, "limit": 5},
                )
                if result.get("success") and result.get("databases"):
                    db = result["databases"][0]
                    return {
                        "database_id": db.get("id"),
                        "database_name": db.get("database_name"),
                        "database_type": db.get("database_type"),
                        "phase": "performance_overview",
                        "iteration": state.iteration + 1,
                    }

            # Get fleet summary for context
            fleet_result = await self._call_mcp_tool(
                "get_fleet_summary",
                {"use_cache": True},
            )

            return {
                "fleet_summary": fleet_result if fleet_result.get("success") else {},
                "phase": "performance_overview",
                "iteration": state.iteration + 1,
            }

    async def _performance_overview_node(self, state: TroubleshootState) -> dict[str, Any]:
        """Get performance overview using OPSI."""
        with self._tracer.start_as_current_span("performance_overview") as span:
            self._logger.info(
                "Getting performance overview",
                database_id=state.database_id,
                trace_id=get_trace_id(),
            )

            if not state.database_id:
                # No specific database, return fleet-level summary
                return {
                    "performance_summary": state.fleet_summary,
                    "phase": "analyze_metrics",
                }

            # Get combined performance summary
            result = await self._call_mcp_tool(
                "get_performance_summary",
                {
                    "database_id": state.database_id,
                    "hours_back": 24,
                },
            )

            span.set_attribute("db.overall_health", result.get("overall_health", "unknown"))

            return {
                "performance_summary": result if result.get("success") else {},
                "phase": "analyze_metrics",
            }

    async def _analyze_metrics_node(self, state: TroubleshootState) -> dict[str, Any]:
        """Deep-dive into specific metrics based on performance summary."""
        with self._tracer.start_as_current_span("analyze_metrics") as span:
            self._logger.info(
                "Analyzing metrics",
                database_id=state.database_id,
                trace_id=get_trace_id(),
            )

            cpu_analysis = {}
            memory_analysis = {}
            io_analysis = {}

            if state.database_id:
                # Analyze CPU if there are issues
                perf = state.performance_summary
                if perf.get("cpu", {}).get("avg_percent", 0) > 50:
                    cpu_analysis = await self._call_mcp_tool(
                        "analyze_cpu_usage",
                        {"database_id": state.database_id, "hours_back": 24},
                    )

                # Analyze memory if there are issues
                if perf.get("memory", {}).get("avg_percent", 0) > 50:
                    memory_analysis = await self._call_mcp_tool(
                        "analyze_memory_usage",
                        {"database_id": state.database_id, "hours_back": 24},
                    )

                # Analyze I/O if there are issues
                if perf.get("io", {}).get("avg_throughput_mbps", 0) > 100:
                    io_analysis = await self._call_mcp_tool(
                        "analyze_io_performance",
                        {"database_id": state.database_id, "hours_back": 24},
                    )

            return {
                "cpu_analysis": cpu_analysis,
                "memory_analysis": memory_analysis,
                "io_analysis": io_analysis,
                "phase": "sql_analysis",
            }

    async def _sql_analysis_node(self, state: TroubleshootState) -> dict[str, Any]:
        """SQL-level analysis using SQLcl (optional)."""
        with self._tracer.start_as_current_span("sql_analysis") as span:
            self._logger.info(
                "Attempting SQL analysis",
                database_id=state.database_id,
                trace_id=get_trace_id(),
            )

            wait_events = []
            top_sql = []
            blocking_sessions = []

            # Check if SQLcl is available
            connections = await self._call_mcp_tool("list_connections", {})

            if not connections.get("success") or not connections.get("connections"):
                self._logger.info("SQLcl not available, skipping SQL analysis")
                span.set_attribute("sql.available", False)
                return {
                    "wait_events": [],
                    "top_sql": [],
                    "blocking_sessions": [],
                    "phase": "generate_recommendations",
                }

            span.set_attribute("sql.available", True)

            # Query wait events
            wait_result = await self._call_mcp_tool(
                "execute_sql",
                {
                    "sql": """
                        SELECT event, total_waits, time_waited_micro/1000000 as time_waited_sec
                        FROM v$system_event
                        WHERE wait_class != 'Idle'
                        ORDER BY time_waited_micro DESC
                        FETCH FIRST 10 ROWS ONLY
                    """,
                },
            )
            if wait_result.get("success"):
                wait_events = wait_result.get("result", [])

            # Query top SQL
            top_sql_result = await self._call_mcp_tool(
                "execute_sql",
                {
                    "sql": """
                        SELECT sql_id, elapsed_time/1000000 as elapsed_sec,
                               executions, buffer_gets, disk_reads
                        FROM v$sqlarea
                        WHERE elapsed_time > 0
                        ORDER BY elapsed_time DESC
                        FETCH FIRST 10 ROWS ONLY
                    """,
                },
            )
            if top_sql_result.get("success"):
                top_sql = top_sql_result.get("result", [])

            # Query blocking sessions
            blocking_result = await self._call_mcp_tool(
                "execute_sql",
                {
                    "sql": """
                        SELECT blocking_session, sid, serial#, username, sql_id
                        FROM v$session
                        WHERE blocking_session IS NOT NULL
                    """,
                },
            )
            if blocking_result.get("success"):
                blocking_sessions = blocking_result.get("result", [])

            return {
                "wait_events": wait_events,
                "top_sql": top_sql,
                "blocking_sessions": blocking_sessions,
                "phase": "generate_recommendations",
            }

    async def _generate_recommendations_node(
        self, state: TroubleshootState
    ) -> dict[str, Any]:
        """Generate recommendations based on all collected data."""
        with self._tracer.start_as_current_span("generate_recommendations") as span:
            self._logger.info(
                "Generating recommendations",
                database_id=state.database_id,
                trace_id=get_trace_id(),
            )

            problem_areas = []
            recommendations = []
            top_issues = []

            # Analyze performance summary
            perf = state.performance_summary
            overall_health = perf.get("overall_health", "unknown")

            # CPU Analysis
            cpu_stats = perf.get("cpu", {})
            cpu_avg = cpu_stats.get("avg_percent", 0)

            if cpu_avg > 90:
                problem_areas.append(
                    ProblemArea(
                        area="CPU",
                        status="critical",
                        value=f"{cpu_avg}%",
                        threshold="90%",
                        details="CPU utilization is critically high",
                    )
                )
                recommendations.append(
                    Recommendation(
                        priority=1,
                        action="Scale OCPU count immediately",
                        type="scaling",
                        impact="Immediate CPU relief",
                        effort="low",
                    )
                )
            elif cpu_avg > 80:
                problem_areas.append(
                    ProblemArea(
                        area="CPU",
                        status="warning",
                        value=f"{cpu_avg}%",
                        threshold="80%",
                        details="CPU utilization approaching threshold",
                    )
                )

            # Memory Analysis
            mem_stats = perf.get("memory", {})
            mem_avg = mem_stats.get("avg_percent", 0)

            if mem_avg > 85:
                problem_areas.append(
                    ProblemArea(
                        area="Memory",
                        status="critical",
                        value=f"{mem_avg}%",
                        threshold="85%",
                        details="Memory pressure detected",
                    )
                )
                recommendations.append(
                    Recommendation(
                        priority=2,
                        action="Review SGA/PGA allocation",
                        type="configuration",
                        impact="Reduce memory pressure",
                        effort="medium",
                    )
                )

            # I/O Analysis
            io_stats = perf.get("io", {})
            io_max = io_stats.get("max_throughput_mbps", 0)
            io_avg = io_stats.get("avg_throughput_mbps", 0)

            if io_max > io_avg * 3 and io_max > 100:
                problem_areas.append(
                    ProblemArea(
                        area="I/O",
                        status="warning",
                        value=f"{io_max} MB/s peak",
                        threshold="3x average",
                        details="I/O spikes detected",
                    )
                )
                recommendations.append(
                    Recommendation(
                        priority=3,
                        action="Investigate I/O spike periods",
                        type="investigation",
                        impact="Identify I/O bottlenecks",
                        effort="medium",
                    )
                )

            # Blocking sessions
            if state.blocking_sessions:
                problem_areas.append(
                    ProblemArea(
                        area="Concurrency",
                        status="critical",
                        value=f"{len(state.blocking_sessions)} blockers",
                        threshold="0",
                        details="Active blocking sessions detected",
                    )
                )
                recommendations.append(
                    Recommendation(
                        priority=1,
                        action="Resolve blocking sessions",
                        type="concurrency",
                        impact="Unblock waiting sessions",
                        effort="low",
                    )
                )

            # Process recommendations from OPSI tools
            for rec in perf.get("recommendations", []):
                recommendations.append(
                    Recommendation(
                        priority=len(recommendations) + 1,
                        action=rec.get("message", ""),
                        type=rec.get("type", "other"),
                        impact=rec.get("severity", "medium"),
                        effort="medium",
                    )
                )

            # Calculate health score
            health_score = 100
            for pa in problem_areas:
                if pa.status == "critical":
                    health_score -= 30
                elif pa.status == "warning":
                    health_score -= 15
            health_score = max(0, health_score)

            # Determine severity
            if health_score < 30:
                severity = AnalysisSeverity.CRITICAL
            elif health_score < 50:
                severity = AnalysisSeverity.HIGH
            elif health_score < 70:
                severity = AnalysisSeverity.MEDIUM
            elif health_score < 90:
                severity = AnalysisSeverity.LOW
            else:
                severity = AnalysisSeverity.HEALTHY

            span.set_attribute("db.health_score", health_score)
            span.set_attribute("db.severity", severity.value)

            result = DbAnalysisResult(
                summary=self._generate_summary(state, problem_areas),
                health_score=health_score,
                severity=severity,
                problem_areas=problem_areas,
                metrics={
                    "cpu_utilization": cpu_avg,
                    "memory_utilization": mem_avg,
                    "io_throughput_mbps": io_avg,
                    "blocked_sessions": len(state.blocking_sessions),
                    "overall_health": overall_health,
                },
                top_issues=top_issues,
                recommendations=recommendations,
                next_steps=[
                    "Monitor metrics after implementing recommendations",
                    "Check related dependent queries",
                    "Review application connection pooling",
                ],
                trace_id=get_trace_id(),
            )

            return {
                "result": result,
                "phase": "output",
            }

    def _generate_summary(
        self, state: TroubleshootState, problem_areas: list[ProblemArea]
    ) -> str:
        """Generate a human-readable summary."""
        db_name = state.database_name or state.database_id or "database"

        if not problem_areas:
            return f"Database '{db_name}' is healthy with no significant issues detected."

        critical = [p for p in problem_areas if p.status == "critical"]
        warnings = [p for p in problem_areas if p.status == "warning"]

        summary_parts = []
        if critical:
            summary_parts.append(
                f"{len(critical)} critical issue(s): {', '.join(p.area for p in critical)}"
            )
        if warnings:
            summary_parts.append(
                f"{len(warnings)} warning(s): {', '.join(p.area for p in warnings)}"
            )

        return f"Database '{db_name}' analysis: {'; '.join(summary_parts)}."

    async def _output_node(self, state: TroubleshootState) -> dict[str, Any]:
        """Prepare output and log completion."""
        self._logger.info(
            "Analysis complete",
            health_score=state.result.health_score if state.result else None,
            severity=state.result.severity.value if state.result else None,
            trace_id=get_trace_id(),
        )
        return {}

    def _extract_db_name(self, query: str) -> str | None:
        """Extract database name from natural language query."""
        query_lower = query.lower()

        # Common patterns
        patterns = [
            "database ", "db ", "for ", "analyze ", "check ",
            "troubleshoot ", "investigate ",
        ]

        for pattern in patterns:
            if pattern in query_lower:
                idx = query_lower.find(pattern) + len(pattern)
                remaining = query[idx:].strip()
                # Get first word after pattern
                if remaining:
                    parts = remaining.split()
                    if parts:
                        return parts[0].strip("'\"")

        return None

    async def invoke(self, query: str, context: dict[str, Any] | None = None) -> str:
        """
        Execute the troubleshooting workflow.

        Args:
            query: User query describing the database issue
            context: Additional context (database_id, compartment_id, etc.)

        Returns:
            Analysis result as formatted string
        """
        context = context or {}

        with self._tracer.start_as_current_span("db_troubleshoot_invoke") as span:
            span.set_attribute("query", query[:100])
            span.set_attribute("database_id", context.get("database_id", "none"))

            # Build graph if not already built
            if not self._graph:
                self.build_graph()

            # Create initial state
            initial_state = TroubleshootState(
                query=query,
                database_id=context.get("database_id"),
                database_name=context.get("database_name"),
                database_type=context.get("database_type"),
                compartment_id=context.get("compartment_id"),
                region=context.get("region"),
            )

            self._logger.info(
                "Starting database troubleshooting",
                query=query[:100],
                database_id=initial_state.database_id,
                trace_id=get_trace_id(),
            )

            try:
                # Run the workflow
                result = await self._graph.ainvoke(initial_state)

                # Format output
                if result.get("result"):
                    analysis = result["result"]
                    span.set_attribute("health_score", analysis.health_score)
                    return self._format_response(analysis)
                elif result.get("error"):
                    return f"Error during analysis: {result['error']}"
                else:
                    return "Analysis completed but no results available."

            except Exception as e:
                self._logger.error(
                    "Troubleshooting failed",
                    error=str(e),
                    trace_id=get_trace_id(),
                )
                span.set_attribute("error", True)
                return f"Database troubleshooting failed: {e}"

    async def quick_health_check(
        self,
        database_id: str | None = None,
    ) -> DbAnalysisResult:
        """
        Perform a quick health check using cached data.

        Args:
            database_id: Optional database ID to check

        Returns:
            Quick analysis result
        """
        with self._tracer.start_as_current_span("quick_health_check"):
            # Use skill executor for health check workflow
            result = await self.execute_skill(
                "db_health_check_workflow",
                context={"database_id": database_id},
            )

            if result.success:
                # Convert to DbAnalysisResult
                fleet = result.get_step_result("get_fleet_overview")
                if fleet and fleet.result:
                    return DbAnalysisResult(
                        summary=f"Fleet has {fleet.result.get('total_databases', 0)} databases",
                        health_score=85,
                        severity=AnalysisSeverity.HEALTHY,
                        metrics=fleet.result,
                        trace_id=get_trace_id(),
                    )

            return DbAnalysisResult(
                summary="Health check failed",
                health_score=0,
                severity=AnalysisSeverity.CRITICAL,
                trace_id=get_trace_id(),
            )

    def _format_response(self, analysis: DbAnalysisResult) -> str | dict:
        """
        Format analysis result using structured response.

        Returns markdown or Slack Block Kit depending on output_format config.
        """
        from src.formatting import (
            ListItem,
            MetricValue,
            ResponseFooter,
            Severity as FmtSeverity,
            StatusIndicator,
        )

        # Map severity
        severity_map = {
            AnalysisSeverity.CRITICAL: "critical",
            AnalysisSeverity.HIGH: "high",
            AnalysisSeverity.MEDIUM: "medium",
            AnalysisSeverity.LOW: "low",
            AnalysisSeverity.HEALTHY: "success",
        }

        # Create structured response
        response = self.create_response(
            title="Database Analysis Results",
            subtitle=analysis.summary,
            severity=severity_map.get(analysis.severity, "info"),
        )

        # Add health score as metric
        response.add_metrics(
            "Health Status",
            [
                MetricValue(
                    label="Health Score",
                    value=analysis.health_score,
                    unit="/100",
                    threshold=70,
                    severity=FmtSeverity(severity_map.get(analysis.severity, "info")),
                ),
            ],
        )

        # Add problem areas
        if analysis.problem_areas:
            problem_fields = []
            for pa in analysis.problem_areas:
                pa_severity = FmtSeverity.CRITICAL if pa.status == "critical" else FmtSeverity.MEDIUM
                problem_fields.append(
                    StatusIndicator(
                        label=pa.area,
                        value=pa.value,
                        severity=pa_severity,
                        description=f"{pa.details} (threshold: {pa.threshold})",
                    )
                )
            response.add_status_list("Problem Areas", problem_fields, divider_after=True)

        # Add recommendations
        if analysis.recommendations:
            rec_items = []
            for rec in analysis.recommendations:
                details = f"Type: {rec.type} | Impact: {rec.impact} | Effort: {rec.effort}"
                if rec.command:
                    details += f"\nCommand: `{rec.command}`"
                rec_items.append(
                    ListItem(
                        text=f"**{rec.priority}.** {rec.action}",
                        details=details,
                        severity=FmtSeverity.INFO,
                    )
                )
            response.add_recommendations(rec_items, divider_after=True)

        # Add footer with trace info
        next_steps = analysis.next_steps.copy()
        if analysis.trace_id:
            next_steps.append(f"Trace ID: `{analysis.trace_id}`")

        if next_steps:
            response.footer = ResponseFooter(
                next_steps=next_steps,
                help_text="Run `/oci db analyze <db_name>` for detailed analysis",
            )

        return self.format_response(response)
