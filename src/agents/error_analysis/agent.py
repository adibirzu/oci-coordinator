"""
Error Analysis Agent for OCI Log Error Detection.

Analyzes OCI logs to identify errors, detect patterns, generate recommendations,
and update an admin todo list for action items.

Capabilities:
- error-identification: Find errors in OCI logs
- error-analysis: Analyze error patterns with LLM
- recommendation-generation: Generate actionable fixes
- admin-todo-management: Update JSON todo list

MCP Tools Used:
- oci_logging_search_logs
- oci_logan_execute_query
- oci_logan_detect_anomalies
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
from src.agents.error_analysis.todo_manager import (
    AdminTodoManager,
    TodoSeverity,
)
from src.agents.self_healing import SelfHealingMixin

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("oci-error-analysis-agent")


# ═══════════════════════════════════════════════════════════════════════════════
# Error Pattern Definitions
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ErrorPattern:
    """A detected error pattern in logs."""

    pattern: str
    count: int
    severity: TodoSeverity
    first_seen: str
    last_seen: str
    sample_messages: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


# Common OCI error patterns
ERROR_PATTERNS = {
    # Database errors
    r"ORA-\d{5}": {
        "name": "Oracle Database Error",
        "severity": TodoSeverity.HIGH,
        "category": "database",
    },
    r"ORA-00060": {
        "name": "Deadlock Detected",
        "severity": TodoSeverity.CRITICAL,
        "category": "database",
    },
    r"ORA-04031": {
        "name": "Shared Pool Memory Error",
        "severity": TodoSeverity.CRITICAL,
        "category": "database",
    },
    # Compute/Network errors
    r"Connection (timed out|refused|reset)": {
        "name": "Connection Error",
        "severity": TodoSeverity.HIGH,
        "category": "network",
    },
    r"OutOfMemory|OOM|memory exhausted": {
        "name": "Out of Memory",
        "severity": TodoSeverity.CRITICAL,
        "category": "compute",
    },
    # Security errors
    r"(Authentication|Authorization) (failed|error|denied)": {
        "name": "Auth Failure",
        "severity": TodoSeverity.HIGH,
        "category": "security",
    },
    r"Invalid (credentials|token|API key)": {
        "name": "Invalid Credentials",
        "severity": TodoSeverity.MEDIUM,
        "category": "security",
    },
    # API errors
    r"HTTP [45]\d{2}": {
        "name": "HTTP Error",
        "severity": TodoSeverity.MEDIUM,
        "category": "api",
    },
    r"Rate limit(ed)? exceeded": {
        "name": "Rate Limited",
        "severity": TodoSeverity.MEDIUM,
        "category": "api",
    },
    # Infrastructure errors
    r"(Disk|Storage) (full|exhausted|quota exceeded)": {
        "name": "Storage Issue",
        "severity": TodoSeverity.CRITICAL,
        "category": "infrastructure",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Agent State
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ErrorAnalysisState:
    """State for error analysis workflow."""

    query: str
    time_range_hours: int = 1
    compartment_id: str | None = None

    # Analysis results
    raw_logs: list[dict[str, Any]] = field(default_factory=list)
    detected_patterns: list[ErrorPattern] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    todos_created: list[str] = field(default_factory=list)

    # Metadata
    error: str | None = None
    final_response: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Error Analysis Agent
# ═══════════════════════════════════════════════════════════════════════════════


class ErrorAnalysisAgent(BaseAgent, SelfHealingMixin):
    """
    Agent specialized in analyzing OCI logs for errors and anomalies.

    Creates admin todos for significant issues requiring attention.

    Self-Healing Features:
    - Automatic retry on log search timeouts
    - Query correction for Logan syntax errors
    - LLM-powered error pattern analysis recovery
    """

    def __init__(
        self,
        memory_manager: SharedMemoryManager | None = None,
        tool_catalog: ToolCatalog | None = None,
        config: dict[str, Any] | None = None,
        llm: BaseChatModel | None = None,
        todo_manager: AdminTodoManager | None = None,
    ):
        """
        Initialize the Error Analysis Agent with self-healing.

        Args:
            memory_manager: Shared memory manager
            tool_catalog: Tool catalog for MCP tools
            config: Agent configuration
            llm: LangChain LLM for analysis
            todo_manager: Admin todo manager (created if not provided)
        """
        super().__init__(memory_manager, tool_catalog, config)
        self.llm = llm
        self.todo_manager = todo_manager or AdminTodoManager()
        self._graph = self._build_graph()

        # Initialize self-healing capabilities
        if llm:
            self.init_self_healing(
                llm=llm,
                max_retries=3,
                enable_validation=True,
                enable_correction=True,
            )

    @classmethod
    def get_definition(cls) -> AgentDefinition:
        """Get the agent definition for catalog registration."""
        return AgentDefinition(
            agent_id="error-analysis-agent",
            role="error-analysis-agent",
            capabilities=[
                "error-identification",
                "error-analysis",
                "pattern-detection",
                "recommendation-generation",
                "admin-todo-management",
            ],
            skills=[
                "error_scan_workflow",
                "pattern_analysis",
                "todo_management",
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.error-analysis-agent"],
                produce=["results.error-analysis-agent"],
            ),
            health_endpoint="http://localhost:8020/health",
            metadata=AgentMetadata(
                version="1.0.0",
                namespace="oci-coordinator",
                max_iterations=15,
                timeout_seconds=300,
            ),
            description="Analyzes OCI logs for errors, detects patterns, and creates admin todos",
            mcp_servers=["oci-unified", "database-observatory"],
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for error analysis."""
        graph = StateGraph(ErrorAnalysisState)

        # Add nodes
        graph.add_node("search_logs", self._search_logs)
        graph.add_node("detect_patterns", self._detect_patterns)
        graph.add_node("analyze_with_llm", self._analyze_with_llm)
        graph.add_node("create_todos", self._create_todos)
        graph.add_node("generate_response", self._generate_response)

        # Define edges
        graph.set_entry_point("search_logs")
        graph.add_edge("search_logs", "detect_patterns")
        graph.add_edge("detect_patterns", "analyze_with_llm")
        graph.add_edge("analyze_with_llm", "create_todos")
        graph.add_edge("create_todos", "generate_response")
        graph.add_edge("generate_response", END)

        return graph.compile()

    async def _search_logs(self, state: ErrorAnalysisState) -> ErrorAnalysisState:
        """Search OCI logs for errors with self-healing."""
        with tracer.start_as_current_span("search_logs") as span:
            try:
                if not self.tools:
                    state.error = "Tool catalog not available"
                    return state

                # Try OCI Logging first
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=state.time_range_hours)

                try:
                    # Use self-healing for automatic retry on log search issues
                    if self._self_healing_enabled:
                        result = await self.healing_call_tool(
                            "oci_logging_search_logs",
                            {
                                "search_query": 'data.level = "ERROR" OR data.level = "FATAL"',
                                "time_start": start_time.isoformat(),
                                "time_end": end_time.isoformat(),
                            },
                            user_intent=state.query,
                            validate=True,
                            correct_on_failure=True,
                        )
                    else:
                        result = await self.call_tool(
                            "oci_logging_search_logs",
                            {
                                "search_query": 'data.level = "ERROR" OR data.level = "FATAL"',
                                "time_start": start_time.isoformat(),
                                "time_end": end_time.isoformat(),
                            },
                        )

                    if result and isinstance(result, list):
                        state.raw_logs.extend(result)
                        span.set_attribute("log_count", len(result))
                except Exception as e:
                    logger.warning("OCI Logging search failed", error=str(e))

                # Also try Log Analytics with self-healing
                try:
                    if self._self_healing_enabled:
                        logan_result = await self.healing_call_tool(
                            "oci_logan_execute_query",
                            {
                                "query": "'Log Source' = * and Level in (ERROR, FATAL, CRITICAL)",
                                "time_start": start_time.isoformat(),
                                "time_end": end_time.isoformat(),
                            },
                            user_intent=state.query,
                            validate=True,
                            correct_on_failure=True,
                        )
                    else:
                        logan_result = await self.call_tool(
                            "oci_logan_execute_query",
                            {
                                "query": "'Log Source' = * and Level in (ERROR, FATAL, CRITICAL)",
                                "time_start": start_time.isoformat(),
                                "time_end": end_time.isoformat(),
                            },
                        )

                    if logan_result and isinstance(logan_result, list):
                        state.raw_logs.extend(logan_result)
                except Exception as e:
                    logger.warning("Logan query failed", error=str(e))

                logger.info(
                    "Log search complete",
                    total_logs=len(state.raw_logs),
                )

            except Exception as e:
                logger.error("Log search failed", error=str(e))
                state.error = str(e)

            return state

    async def _detect_patterns(self, state: ErrorAnalysisState) -> ErrorAnalysisState:
        """Detect error patterns in logs."""
        with tracer.start_as_current_span("detect_patterns"):
            pattern_counts: dict[str, ErrorPattern] = {}

            for log in state.raw_logs:
                message = log.get("message", "") or log.get("data", {}).get("message", "")
                timestamp = log.get("timestamp", datetime.utcnow().isoformat())
                source = log.get("source", "unknown")

                for pattern_regex, pattern_info in ERROR_PATTERNS.items():
                    if re.search(pattern_regex, message, re.IGNORECASE):
                        pattern_key = pattern_regex

                        if pattern_key not in pattern_counts:
                            pattern_counts[pattern_key] = ErrorPattern(
                                pattern=pattern_info["name"],
                                count=0,
                                severity=pattern_info["severity"],
                                first_seen=timestamp,
                                last_seen=timestamp,
                                sample_messages=[],
                                sources=[],
                            )

                        ep = pattern_counts[pattern_key]
                        ep.count += 1
                        ep.last_seen = timestamp
                        if source not in ep.sources:
                            ep.sources.append(source)
                        if len(ep.sample_messages) < 3:
                            ep.sample_messages.append(message[:200])

            # Sort by severity and count
            severity_order = {
                TodoSeverity.CRITICAL: 0,
                TodoSeverity.HIGH: 1,
                TodoSeverity.MEDIUM: 2,
                TodoSeverity.LOW: 3,
            }
            state.detected_patterns = sorted(
                pattern_counts.values(),
                key=lambda p: (severity_order[p.severity], -p.count),
            )

            logger.info(
                "Pattern detection complete",
                patterns_found=len(state.detected_patterns),
            )

            return state

    async def _analyze_with_llm(self, state: ErrorAnalysisState) -> ErrorAnalysisState:
        """Use LLM to analyze patterns and generate recommendations."""
        with tracer.start_as_current_span("analyze_with_llm"):
            if not self.llm or not state.detected_patterns:
                return state

            # Build context for LLM
            pattern_summary = []
            for p in state.detected_patterns[:10]:  # Top 10 patterns
                pattern_summary.append(
                    f"- {p.pattern}: {p.count} occurrences ({p.severity.value} severity)"
                    f"\n  Sources: {', '.join(p.sources[:3])}"
                    f"\n  Sample: {p.sample_messages[0] if p.sample_messages else 'N/A'}"
                )

            prompt = f"""Analyze the following error patterns detected in OCI logs and provide specific recommendations:

{chr(10).join(pattern_summary)}

For each significant pattern:
1. Explain the likely root cause
2. Provide specific remediation steps
3. Suggest monitoring improvements

Format each recommendation as a clear action item."""

            try:
                response = await self.llm.ainvoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)

                # Extract recommendations from response
                recommendations = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                        recommendations.append(line.lstrip("-*0123456789. "))

                state.recommendations = recommendations[:20]  # Limit recommendations

            except Exception as e:
                logger.warning("LLM analysis failed", error=str(e))

            return state

    async def _create_todos(self, state: ErrorAnalysisState) -> ErrorAnalysisState:
        """Create admin todos for significant error patterns."""
        with tracer.start_as_current_span("create_todos"):
            # Only create todos for high-severity patterns with significant count
            for pattern in state.detected_patterns:
                if pattern.severity in (TodoSeverity.CRITICAL, TodoSeverity.HIGH):
                    if pattern.count >= 5 or pattern.severity == TodoSeverity.CRITICAL:
                        todo = self.todo_manager.add_todo(
                            title=f"{pattern.pattern} ({pattern.count} occurrences)",
                            description=(
                                f"Detected {pattern.count} instances of {pattern.pattern} "
                                f"from sources: {', '.join(pattern.sources[:3])}.\n\n"
                                f"Sample: {pattern.sample_messages[0] if pattern.sample_messages else 'N/A'}"
                            ),
                            severity=pattern.severity,
                            error_pattern=pattern.pattern,
                            source="error-analysis-agent",
                            metadata={
                                "first_seen": pattern.first_seen,
                                "last_seen": pattern.last_seen,
                                "sources": pattern.sources,
                            },
                        )
                        state.todos_created.append(todo.id)

            logger.info(
                "Todos created",
                count=len(state.todos_created),
            )

            return state

    async def _generate_response(self, state: ErrorAnalysisState) -> ErrorAnalysisState:
        """Generate the final response."""
        with tracer.start_as_current_span("generate_response"):
            if state.error:
                state.final_response = f"Error during analysis: {state.error}"
                return state

            lines = [f"## Error Analysis Results ({state.time_range_hours}h window)\n"]

            # Summary
            lines.append(f"**Logs analyzed:** {len(state.raw_logs)}")
            lines.append(f"**Patterns detected:** {len(state.detected_patterns)}")
            lines.append(f"**Admin todos created:** {len(state.todos_created)}\n")

            # Top patterns
            if state.detected_patterns:
                lines.append("### Top Error Patterns\n")
                for i, p in enumerate(state.detected_patterns[:5], 1):
                    lines.append(
                        f"{i}. **{p.pattern}** - {p.count} occurrences "
                        f"({p.severity.value})"
                    )
                lines.append("")

            # Recommendations
            if state.recommendations:
                lines.append("### Recommendations\n")
                for rec in state.recommendations[:5]:
                    lines.append(f"- {rec}")
                lines.append("")

            # Admin todo summary
            todo_summary = self.todo_manager.get_summary()
            lines.append("### Admin Todo Summary\n")
            lines.append(f"```\n{todo_summary}\n```")

            state.final_response = "\n".join(lines)
            return state

    async def invoke(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Invoke the error analysis agent."""
        state = ErrorAnalysisState(
            query=query,
            time_range_hours=kwargs.get("time_range_hours", 1),
            compartment_id=kwargs.get("compartment_id"),
        )

        try:
            final_state = await self._graph.ainvoke(state)
            return {
                "response": final_state.final_response or "Analysis complete.",
                "patterns_found": len(final_state.detected_patterns),
                "todos_created": final_state.todos_created,
                "success": not final_state.error,
            }
        except Exception as e:
            logger.error("Error analysis failed", error=str(e))
            return {
                "response": f"Error analysis failed: {e!s}",
                "success": False,
            }

    async def get_pending_todos(self) -> list[dict[str, Any]]:
        """Get all pending admin todos."""
        todos = self.todo_manager.get_todos(status="pending")
        return [t.to_dict() for t in todos]

    async def resolve_todo(self, todo_id: str, resolution: str) -> dict[str, Any]:
        """Mark a todo as resolved."""
        todo = self.todo_manager.update_status(todo_id, "resolved", resolution)
        if todo:
            return {"success": True, "todo": todo.to_dict()}
        return {"success": False, "error": "Todo not found"}
