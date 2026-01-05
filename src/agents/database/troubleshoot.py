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
    StepResult,
)
from src.agents.self_healing import SelfHealingMixin
from src.observability import get_trace_id

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

from src.skills.troubleshoot_database import DBTroubleshootSkill


logger = structlog.get_logger(__name__)

# Minimal tool baseline for catalog registration and self-healing fallback.
MCP_TOOLS = [
    "oci_opsi_get_fleet_summary",
    "oci_opsi_search_databases",
    "oci_opsi_get_database",
    "oci_opsi_get_performance_summary",
    "oci_opsi_analyze_cpu",
    "oci_opsi_analyze_memory",
    "oci_opsi_analyze_io",
    "oci_database_execute_sql",
    "oci_dbmgmt_get_metrics",
    "oci_dbmgmt_get_awr_report",
]


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
class AwrReportData:
    """AWR report data for file attachment."""

    html_content: str
    database_name: str
    begin_snapshot_id: int
    end_snapshot_id: int
    source: str  # sqlcl, dbmgmt, opsi
    generated_at: str | None = None


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
    awr_report: AwrReportData | None = None  # AWR HTML report for attachment

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
            "awr_report": {
                "database_name": self.awr_report.database_name,
                "begin_snapshot_id": self.awr_report.begin_snapshot_id,
                "end_snapshot_id": self.awr_report.end_snapshot_id,
                "source": self.awr_report.source,
                "generated_at": self.awr_report.generated_at,
            } if self.awr_report else None,
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

    # AWR report data (for attachment)
    awr_report: AwrReportData | None = None
    awr_requested: bool = False

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

## Available MCP Tools (Database Observatory - oci_{domain}_{action}_{resource} convention)

### Tier 1: Cache-Based (Instant, <100ms)
- `oci_opsi_get_fleet_summary`: Get fleet overview with database counts
- `oci_opsi_search_databases`: Find databases by name/type/compartment
- `oci_opsi_get_database`: Get database details from cache
- `oci_opsi_get_statistics`: Get cache statistics
- `oci_opsi_refresh_cache`: Refresh cache if needed

### Tier 2: OPSI API (1-5s)
- `oci_opsi_analyze_cpu`: CPU usage trends with recommendations
- `oci_opsi_analyze_memory`: Memory utilization analysis
- `oci_opsi_analyze_io`: I/O throughput analysis
- `oci_opsi_get_performance_summary`: Combined CPU/memory/I/O summary
- `oci_opsi_find_cost_opportunities`: Identify cost savings opportunities
- `oci_opsi_list_insights`: List database insights

### Tier 3: SQL Execution (5-30s)
- `oci_database_execute_sql`: Run SQL queries via SQLcl
- `oci_database_get_schema`: Get schema metadata
- `oci_database_get_status`: Check connection status
- `oci_database_list_connections`: List available connections

### Tier 4: Tenancy Management
- `oci_tenancy_list_profiles`: List OCI profiles
- `oci_tenancy_list_compartments`: List compartments

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


class DbTroubleshootAgent(BaseAgent, SelfHealingMixin):
    """
    Database Troubleshooting Agent with MCP Integration and Self-Healing.

    Uses the Database Observatory MCP server for:
    - OPSI operations insights (CPU, memory, I/O analysis)
    - SQLcl for direct database queries
    - Logan Analytics for log correlation

    Self-Healing Capabilities:
    - Automatic retry on tool failures with exponential backoff
    - LLM-powered error analysis and root cause detection
    - Parameter correction based on error messages
    - Fallback to alternative tools (OPSI → Autonomous DB API)

    Workflow:
    1. Discover database using cache
    2. Get performance overview from OPSI
    3. Deep-dive into problem areas
    4. Query database if SQLcl available
    5. Generate recommendations
    """

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
                "db_awr_report_workflow",
                "db_fleet_health_workflow",
                "db_addm_analysis_workflow",
                "db_capacity_planning_workflow",
                "rca_workflow",  # Legacy
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.db-troubleshoot-agent"],
                produce=["results.db-troubleshoot-agent"],
            ),
            health_endpoint="http://localhost:8010/health",
            metadata=AgentMetadata(
                version="3.0.0",  # Updated for oci-unified integration
                namespace="oci-coordinator",
                max_iterations=15,
                timeout_seconds=180,  # DB operations via MCP can be slow
            ),
            description=(
                "Database Expert Agent for Oracle performance analysis using "
                "OCI DB Management and Operations Insights (OPSI) via oci-unified MCP server. "
                "Provides fleet health, AWR reports, ADDM findings, capacity planning, and SQL tuning."
            ),
            mcp_tools=list(MCP_TOOLS),
            mcp_servers=["oci-unified"],
        )

    def __init__(
        self,
        memory_manager: SharedMemoryManager | None = None,
        tool_catalog: ToolCatalog | None = None,
        config: dict[str, Any] | None = None,
        llm: Any = None,
    ):
        """
        Initialize DB Troubleshoot Agent with self-healing capabilities.

        Args:
            memory_manager: Shared memory manager
            tool_catalog: Tool catalog for MCP tools
            config: Agent configuration
            llm: LangChain LLM for analysis and self-healing
        """
        super().__init__(memory_manager, tool_catalog, config)
        self.llm = llm
        self._graph: StateGraph | None = None
        self._tracer = trace.get_tracer("oci-db-troubleshoot-agent")

        # Initialize self-healing capabilities
        self.init_self_healing(
            llm=llm,
            max_retries=3,
            enable_validation=True,
            enable_correction=True,
        )

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
        graph.add_node("enhanced_rca", self._enhanced_rca_node) # New Node
        graph.add_node("generate_recommendations", self._generate_recommendations_node)
        graph.add_node("output", self._output_node)

        # Set entry point
        graph.set_entry_point("discover")

        # Add edges
        graph.add_edge("discover", "performance_overview")
        graph.add_edge("performance_overview", "analyze_metrics")
        
        # Branching: If RCA requested, go to enhanced_rca
        graph.add_conditional_edges(
            "analyze_metrics",
            self._route_analysis,
            {
                "standard": "sql_analysis",
                "advanced_rca": "enhanced_rca"
            }
        )
        
        graph.add_edge("sql_analysis", "generate_recommendations")
        graph.add_edge("enhanced_rca", "output") # RCA generates its own report for now
        graph.add_edge("generate_recommendations", "output")
        graph.add_edge("output", END)

        self._graph = graph.compile()
        return self._graph

    def _route_analysis(self, state: TroubleshootState) -> str:
        """Route to standard or advanced analysis."""
        if "advanced" in state.query.lower() or "rca" in state.query.lower():
            return "advanced_rca"
        return "standard"

    async def _enhanced_rca_node(self, state: TroubleshootState) -> dict[str, Any]:
        """Execute the Enhanced Database Troubleshooting Skill."""
        skill = DBTroubleshootSkill(self)
        workflow = skill.build_graph()
        
        # Map agent state to skill state
        skill_input = {
            "query": state.query,
            "database_id": state.database_id,
            "compartment_id": state.compartment_id
        }
        
        result = await workflow.ainvoke(skill_input)
        
        # Map skill output back to agent result
        return {
            "result": DbAnalysisResult(
                summary=result.get("final_report", "No report generated."),
                health_score=50, # Placeholder
                severity=AnalysisSeverity.HIGH if "Critical" in result.get("final_report", "") else AnalysisSeverity.MEDIUM
            ),
            "phase": "output"
        }

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_intent: str | None = None,
        use_self_healing: bool = True,
    ) -> dict[str, Any]:
        """
        Call an MCP tool with tracing, self-healing, and error handling.

        Self-healing features:
        - Automatic retry on transient failures
        - LLM-powered parameter correction
        - Fallback to alternative tools

        Args:
            tool_name: Name of the MCP tool
            arguments: Tool arguments
            user_intent: User query for context (helps with error correction)
            use_self_healing: Whether to use self-healing (default True)

        Returns:
            Tool result or error dict
        """
        with self._tracer.start_as_current_span(f"mcp.tool.{tool_name}") as span:
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.tool.args", str(arguments)[:200])
            span.set_attribute("mcp.tool.self_healing", use_self_healing)

            start_time = time.time()
            try:
                if not self.tools:
                    return {"success": False, "error": "Tool catalog not initialized"}

                # Use self-healing if enabled and initialized
                if use_self_healing and self._self_healing_enabled:
                    result = await self.healing_call_tool(
                        tool_name,
                        arguments,
                        user_intent=user_intent,
                        validate=True,
                        correct_on_failure=True,
                    )
                    # healing_call_tool returns None on complete failure
                    if result is None:
                        duration_ms = int((time.time() - start_time) * 1000)
                        span.set_attribute("mcp.tool.success", False)
                        span.set_attribute("mcp.tool.self_healed", True)
                        return {"success": False, "error": "All retry attempts failed"}
                else:
                    # Direct call without self-healing
                    result = await self.call_tool(tool_name, arguments)

                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("mcp.tool.duration_ms", duration_ms)
                span.set_attribute("mcp.tool.success", True)

                self._logger.info(
                    "MCP tool call completed",
                    tool=tool_name,
                    duration_ms=duration_ms,
                    self_healing=use_self_healing,
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
        """Discover database using cache-based tools and name resolution."""
        with self._tracer.start_as_current_span("discover_database") as span:
            self._logger.info(
                "Discovering database",
                query=state.query[:100],
                trace_id=get_trace_id(),
            )

            updates: dict[str, Any] = {
                "phase": "performance_overview",
                "iteration": state.iteration + 1,
            }

            # If database_id already provided, get details from cache
            if state.database_id:
                span.set_attribute("db.id_provided", True)
                result = await self._call_mcp_tool(
                    "oci_opsi_get_database",
                    {"database_id": state.database_id},
                )
                if result.get("success"):
                    db = result.get("database", {})
                    updates["database_name"] = db.get("database_name")
                    updates["database_type"] = db.get("database_type")
                    return updates

            # Try to resolve database name from query
            database_id, database_name = await self._resolve_database_from_query(state.query)
            span.set_attribute("db.name_hint", database_name or "none")
            span.set_attribute("db.resolved_id", bool(database_id))

            if database_id:
                # Get full details from cache
                result = await self._call_mcp_tool(
                    "oci_opsi_get_database",
                    {"database_id": database_id},
                )
                if result.get("success"):
                    db = result.get("database", {})
                    updates["database_id"] = database_id
                    updates["database_name"] = db.get("database_name") or database_name
                    updates["database_type"] = db.get("database_type")
                    return updates
                else:
                    # Use resolved info even if cache lookup failed
                    updates["database_id"] = database_id
                    updates["database_name"] = database_name
                    return updates
            elif database_name:
                # Have name but no ID - store name for context
                updates["database_name"] = database_name
                self._logger.info(
                    "Database name found but OCID not resolved",
                    name=database_name,
                )

            # Try to resolve compartment from query for context
            compartment_id = await self._resolve_compartment_from_query(state.query)
            if compartment_id:
                updates["compartment_id"] = compartment_id
                span.set_attribute("compartment.resolved", True)

            # Get fleet summary for context if no specific database
            fleet_result = await self._call_mcp_tool(
                "oci_opsi_get_fleet_summary",
                {"use_cache": True},
            )

            updates["fleet_summary"] = fleet_result if fleet_result.get("success") else {}
            return updates

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
                "oci_opsi_get_performance_summary",
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
                        "oci_opsi_analyze_cpu",
                        {"database_id": state.database_id, "hours_back": 24},
                    )

                # Analyze memory if there are issues
                if perf.get("memory", {}).get("avg_percent", 0) > 50:
                    memory_analysis = await self._call_mcp_tool(
                        "oci_opsi_analyze_memory",
                        {"database_id": state.database_id, "hours_back": 24},
                    )

                # Analyze I/O if there are issues
                if perf.get("io", {}).get("avg_throughput_mbps", 0) > 100:
                    io_analysis = await self._call_mcp_tool(
                        "oci_opsi_analyze_io",
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
            connections = await self._call_mcp_tool("oci_database_list_connections", {})

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
                "oci_database_execute_sql",
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
                "oci_database_execute_sql",
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
                "oci_database_execute_sql",
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
        import re

        query_lower = query.lower()

        # Regex patterns for database name extraction
        db_patterns = [
            r"(?:database|db)\s+['\"]?([\w_-]+)['\"]?",
            r"(?:for|on|analyze|check|troubleshoot|investigate)\s+['\"]?([\w_-]+)['\"]?\s*(?:database|db)?",
            r"['\"]?([\w_-]+)['\"]?\s+(?:database|db)",
            r"performance\s+(?:of|for)\s+['\"]?([\w_-]+)['\"]?",
        ]

        for pattern in db_patterns:
            match = re.search(pattern, query_lower)
            if match:
                name = match.group(1)
                # Filter out common words that aren't database names
                if name not in {"the", "a", "my", "our", "this", "that", "all", "any"}:
                    return name

        return None

    async def _resolve_database_from_query(self, query: str) -> tuple[str | None, str | None]:
        """
        Extract and resolve database name from user query.

        Returns:
            Tuple of (database_id, database_name) or (None, None) if not found
        """
        db_name = self._extract_db_name(query)

        if not db_name:
            return None, None

        self._logger.debug("Attempting to resolve database", name=db_name)

        # Try to resolve using MCP search_databases tool
        try:
            result = await self._call_mcp_tool(
                "oci_opsi_search_databases",
                {"name": db_name, "limit": 5},
            )

            if result.get("success") and result.get("databases"):
                databases = result["databases"]
                # Exact match first
                for db in databases:
                    if db.get("database_name", "").lower() == db_name.lower():
                        self._logger.info(
                            "Resolved database name to OCID",
                            name=db_name,
                            database_id=db.get("id", "")[:50],
                        )
                        return db.get("id"), db.get("database_name")

                # Otherwise use first match
                db = databases[0]
                self._logger.info(
                    "Resolved database name to OCID (partial match)",
                    name=db_name,
                    database_id=db.get("id", "")[:50],
                )
                return db.get("id"), db.get("database_name")

        except Exception as e:
            self._logger.warning(
                "Database resolution failed",
                name=db_name,
                error=str(e),
            )

        # Try OCIResourceCache as fallback
        try:
            from src.cache.oci_resource_cache import OCIResourceCache
            import os

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            cache = OCIResourceCache.get_instance(redis_url)

            # Search in cached databases
            databases = await cache.search_resources(
                resource_type="database",
                name_pattern=db_name,
                limit=5,
            )

            if databases:
                db = databases[0]
                self._logger.info(
                    "Resolved database from cache",
                    name=db_name,
                    database_id=db.get("id", "")[:50],
                )
                return db.get("id"), db.get("display_name")

        except Exception as e:
            self._logger.debug("Cache lookup failed", error=str(e))

        return None, db_name  # Return name even if ID not found

    async def _resolve_compartment_from_query(self, query: str) -> str | None:
        """
        Extract and resolve compartment name from user query.

        Returns:
            Compartment OCID or None if not found
        """
        import re

        query_lower = query.lower()

        # Regex patterns for compartment name extraction
        patterns = [
            r"(?:in|from|for)\s+(?:the\s+)?['\"]?([\w_-]+)['\"]?\s+compartment",
            r"compartment\s+['\"]?([\w_-]+)['\"]?",
            r"(?:in|from|for)\s+compartment\s+['\"]?([\w_-]+)['\"]?",
        ]

        compartment_name = None
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                compartment_name = match.group(1)
                break

        if not compartment_name:
            return None

        self._logger.debug("Attempting to resolve compartment", name=compartment_name)

        try:
            from src.oci.tenancy_manager import TenancyManager

            manager = TenancyManager.get_instance()
            if not manager._initialized:
                await manager.initialize()

            # Try exact match first
            ocid = await manager.get_compartment_ocid(compartment_name)
            if ocid:
                self._logger.info(
                    "Resolved compartment name to OCID",
                    name=compartment_name,
                    ocid=ocid[:50],
                )
                return ocid

            # Try partial match
            matches = await manager.search_compartments(compartment_name)
            if matches:
                self._logger.info(
                    "Resolved compartment (partial match)",
                    name=compartment_name,
                    ocid=matches[0].id[:50],
                )
                return matches[0].id

        except Exception as e:
            self._logger.warning(
                "Compartment resolution failed",
                name=compartment_name,
                error=str(e),
            )

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

            # Pre-resolve database and compartment from query if not provided
            database_id = context.get("database_id")
            database_name = context.get("database_name")
            compartment_id = context.get("compartment_id")

            if not database_id:
                resolved_id, resolved_name = await self._resolve_database_from_query(query)
                if resolved_id:
                    database_id = resolved_id
                    database_name = database_name or resolved_name
                    span.set_attribute("db.pre_resolved", True)
                    self._logger.info(
                        "Pre-resolved database from query",
                        database_id=database_id[:50] if database_id else None,
                        database_name=database_name,
                    )

            if not compartment_id:
                resolved_compartment = await self._resolve_compartment_from_query(query)
                if resolved_compartment:
                    compartment_id = resolved_compartment
                    span.set_attribute("compartment.pre_resolved", True)

            # Create initial state with resolved values
            initial_state = TroubleshootState(
                query=query,
                database_id=database_id,
                database_name=database_name,
                database_type=context.get("database_type"),
                compartment_id=compartment_id,
                region=context.get("region"),
            )

            self._logger.info(
                "Starting database troubleshooting",
                query=query[:100],
                database_id=initial_state.database_id[:50] if initial_state.database_id else None,
                database_name=initial_state.database_name,
                compartment_id=initial_state.compartment_id[:50] if initial_state.compartment_id else None,
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

    async def get_awr_report(
        self,
        database_name: str | None = None,
        database_id: str | None = None,
        managed_database_id: str | None = None,
        hours_back: int = 1,
        source: str = "auto",  # auto, sqlcl, dbmgmt, opsi
        profile: str = "EMDEMO",
    ) -> AwrReportData | None:
        """
        Generate AWR report from the best available source.

        Source priority (per user preference):
        1. SQLcl Direct (if connection available) - PRIMARY
        2. DB Management (if managed)
        3. OpsInsights (if insight enabled)

        Args:
            database_name: Database display name (for resolution)
            database_id: OPSI database OCID
            managed_database_id: DB Management database OCID
            hours_back: Hours of data for report (default 1)
            source: Force specific source or auto-detect
            profile: OCI profile to use (default EMDEMO)

        Returns:
            AwrReportData with HTML content or None if failed
        """
        import base64
        from datetime import datetime

        with self._tracer.start_as_current_span("get_awr_report") as span:
            span.set_attribute("awr.source_preference", source)
            span.set_attribute("awr.hours_back", hours_back)
            span.set_attribute("awr.profile", profile)

            self._logger.info(
                "Generating AWR report",
                database_name=database_name,
                source=source,
                hours_back=hours_back,
                trace_id=get_trace_id(),
            )

            # Try to resolve database if only name provided
            if database_name and not database_id and not managed_database_id:
                resolved_id, _ = await self._resolve_database_from_query(database_name)
                database_id = resolved_id

            # Source 1: SQLcl Direct (if source is auto or sqlcl)
            if source in ("auto", "sqlcl"):
                awr_data = await self._get_awr_via_sqlcl(database_name, hours_back)
                if awr_data:
                    span.set_attribute("awr.source_used", "sqlcl")
                    return awr_data

            # Source 2: DB Management (if source is auto or dbmgmt)
            if source in ("auto", "dbmgmt") and managed_database_id:
                awr_data = await self._get_awr_via_dbmgmt(
                    managed_database_id, hours_back, profile
                )
                if awr_data:
                    span.set_attribute("awr.source_used", "dbmgmt")
                    return awr_data

            # Source 2b: Try to find managed database by name
            if source in ("auto", "dbmgmt") and database_name:
                # Search for managed database
                search_result = await self._call_mcp_tool(
                    "search_managed_databases",
                    {"name": database_name, "profile": profile, "limit": 5},
                )
                if search_result.get("success") and search_result.get("databases"):
                    db = search_result["databases"][0]
                    awr_data = await self._get_awr_via_dbmgmt(
                        db["id"], hours_back, profile
                    )
                    if awr_data:
                        awr_data.database_name = db.get("name", database_name)
                        span.set_attribute("awr.source_used", "dbmgmt")
                        return awr_data

            self._logger.warning(
                "AWR report generation failed - no source available",
                database_name=database_name,
                source=source,
            )
            span.set_attribute("awr.source_used", "none")
            return None

    async def _get_awr_via_sqlcl(
        self, database_name: str | None, hours_back: int
    ) -> AwrReportData | None:
        """Get AWR report via SQLcl direct connection."""
        # Check if SQLcl is available for this database
        connections = await self._call_mcp_tool("oci_database_list_connections", {})

        if not connections.get("success") or not connections.get("connections"):
            self._logger.debug("SQLcl not available for AWR")
            return None

        # Find matching connection
        conn_name = None
        for conn in connections.get("connections", []):
            if database_name and database_name.lower() in conn.get("name", "").lower():
                conn_name = conn.get("name")
                break

        if not conn_name:
            self._logger.debug("No SQLcl connection for database", name=database_name)
            return None

        # Generate AWR via SQLcl SQL command
        # This uses the DBMS_WORKLOAD_REPOSITORY package
        awr_sql = f"""
            DECLARE
                l_report CLOB;
                l_dbid   NUMBER;
                l_inst   NUMBER := 1;
                l_bid    NUMBER;
                l_eid    NUMBER;
            BEGIN
                SELECT dbid INTO l_dbid FROM v$database;

                SELECT MIN(snap_id), MAX(snap_id)
                INTO l_bid, l_eid
                FROM dba_hist_snapshot
                WHERE end_interval_time > SYSDATE - {hours_back}/24;

                SELECT XMLAGG(
                    XMLELEMENT(e, output || CHR(10))
                ).getClobVal()
                INTO l_report
                FROM TABLE(
                    DBMS_WORKLOAD_REPOSITORY.AWR_REPORT_HTML(
                        l_dbid, l_inst, l_bid, l_eid
                    )
                );

                DBMS_OUTPUT.PUT_LINE('AWR_BEGIN');
                DBMS_OUTPUT.PUT_LINE(l_report);
                DBMS_OUTPUT.PUT_LINE('AWR_END');
            END;
        """

        result = await self._call_mcp_tool(
            "oci_database_execute_sql",
            {"sql": awr_sql, "connection": conn_name},
        )

        if result.get("success"):
            output = result.get("output", "")
            # Extract HTML between markers
            if "AWR_BEGIN" in output and "AWR_END" in output:
                start = output.find("AWR_BEGIN") + len("AWR_BEGIN")
                end = output.find("AWR_END")
                html_content = output[start:end].strip()

                return AwrReportData(
                    html_content=html_content,
                    database_name=database_name or "unknown",
                    begin_snapshot_id=0,  # Not easily available from output
                    end_snapshot_id=0,
                    source="sqlcl",
                    generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )

        return None

    async def _get_awr_via_dbmgmt(
        self,
        managed_database_id: str,
        hours_back: int,
        profile: str,
    ) -> AwrReportData | None:
        """Get AWR report via DB Management API."""
        import base64

        # Use the auto-detect tool that finds snapshots automatically
        result = await self._call_mcp_tool(
            "oci_dbmgmt_get_awr_report_auto",
            {
                "managed_database_id": managed_database_id,
                "hours_back": hours_back,
                "profile": profile,
                "report_format": "HTML",
            },
        )

        if not result.get("success"):
            self._logger.warning(
                "DB Management AWR failed",
                error=result.get("error"),
            )
            return None

        # Decode base64 HTML content
        html_base64 = result.get("report_html", "")
        if html_base64:
            try:
                html_content = base64.b64decode(html_base64).decode("utf-8")
            except Exception:
                html_content = html_base64  # Already decoded

            return AwrReportData(
                html_content=html_content,
                database_name=result.get("database_name", "unknown"),
                begin_snapshot_id=result.get("begin_snapshot_id", 0),
                end_snapshot_id=result.get("end_snapshot_id", 0),
                source="dbmgmt",
                generated_at=result.get("generated_at"),
            )

        return None

    def _format_response(self, analysis: DbAnalysisResult) -> str | dict:
        """
        Format analysis result using structured response.

        Returns markdown or Slack Block Kit depending on output_format config.
        """
        from src.formatting import (
            ListItem,
            MetricValue,
            ResponseFooter,
            StatusIndicator,
        )
        from src.formatting import (
            Severity as FmtSeverity,
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

        # Add AWR report as file attachment if present
        if analysis.awr_report:
            from src.formatting import FileAttachment

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            db_name = analysis.awr_report.database_name.replace(" ", "_")
            filename = f"awr_{db_name}_{timestamp}.html"

            response.add_attachment(
                FileAttachment(
                    content=analysis.awr_report.html_content,
                    filename=filename,
                    content_type="text/html",
                    title=f"AWR Report - {analysis.awr_report.database_name}",
                    comment=(
                        f"AWR report from {analysis.awr_report.source} "
                        f"(snapshots {analysis.awr_report.begin_snapshot_id}-"
                        f"{analysis.awr_report.end_snapshot_id})"
                    ),
                )
            )

        return self.format_response(response)

    async def generate_awr_report(
        self,
        database_name: str | None = None,
        database_id: str | None = None,
        managed_database_id: str | None = None,
        hours_back: int = 1,
        source: str = "auto",
        profile: str = "EMDEMO",
    ) -> dict[str, Any]:
        """
        Generate an AWR report and return it ready for Slack attachment.

        This is a high-level method for direct AWR report requests.
        Returns a response dict with the message and attachments list.

        Args:
            database_name: Database display name
            database_id: OPSI database OCID
            managed_database_id: DB Management database OCID
            hours_back: Hours of data for report (default 1)
            source: Force source (auto, sqlcl, dbmgmt)
            profile: OCI profile (default EMDEMO)

        Returns:
            Dict with 'message', 'success', and 'attachments' keys
        """
        with self._tracer.start_as_current_span("generate_awr_report"):
            awr_data = await self.get_awr_report(
                database_name=database_name,
                database_id=database_id,
                managed_database_id=managed_database_id,
                hours_back=hours_back,
                source=source,
                profile=profile,
            )

            if not awr_data:
                return {
                    "success": False,
                    "message": (
                        f"Could not generate AWR report for "
                        f"{database_name or database_id or 'unknown database'}. "
                        f"Ensure the database is registered in DB Management or "
                        f"has a SQLcl connection."
                    ),
                    "attachments": [],
                }

            # Build response with attachment
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            db_name = awr_data.database_name.replace(" ", "_")
            filename = f"awr_{db_name}_{timestamp}.html"

            return {
                "success": True,
                "message": (
                    f"AWR report generated for **{awr_data.database_name}**\n"
                    f"- Source: {awr_data.source.upper()}\n"
                    f"- Snapshots: {awr_data.begin_snapshot_id} - {awr_data.end_snapshot_id}\n"
                    f"- Generated: {awr_data.generated_at or 'now'}\n\n"
                    f"The HTML report is attached below."
                ),
                "attachments": [
                    {
                        "content": awr_data.html_content,
                        "filename": filename,
                        "content_type": "text/html",
                        "title": f"AWR Report - {awr_data.database_name}",
                        "comment": (
                            f"AWR report from {awr_data.source} "
                            f"(snapshots {awr_data.begin_snapshot_id}-"
                            f"{awr_data.end_snapshot_id})"
                        ),
                    }
                ],
            }
