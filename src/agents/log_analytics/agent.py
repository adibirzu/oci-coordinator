"""
Log Analytics Agent.

Specialized agent for OCI Log Analytics operations including
log search, pattern detection, and correlation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from langgraph.graph import END, StateGraph

from src.agents.base import (
    AgentDefinition,
    AgentMetadata,
    BaseAgent,
    KafkaTopics,
)
from src.agents.self_healing import SelfHealingMixin

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


@dataclass
class LogAnalyticsState:
    """State for log analytics workflow."""

    query: str = ""
    compartment_id: str | None = None
    log_group_id: str | None = None
    time_range: str = "1h"  # 1h, 6h, 24h, 7d, 30d
    search_pattern: str | None = None

    # Results
    log_entries: list[dict] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    correlations: list[dict] = field(default_factory=list)

    # State
    phase: str = "parse_query"
    iteration: int = 0
    error: str | None = None
    result: str | None = None


class LogAnalyticsAgent(BaseAgent, SelfHealingMixin):
    """
    Log Analytics Agent with Self-Healing Capabilities.

    Specializes in OCI Log Analytics operations:
    - Log search and query construction
    - Error pattern detection
    - Service log correlation
    - Audit log analysis
    - Anomaly detection

    Self-Healing Features:
    - Automatic retry on log search timeouts
    - Query correction for Logan syntax errors
    - LLM-powered anomaly detection recovery
    """

    # MCP tools from oci-unified and database-observatory servers
    MCP_TOOLS = [
        # OCI Unified observability tools
        "oci_logging_list_log_groups",
        "oci_logging_search_logs",
        "oci_logging_get_log",
        "oci_observability_query_logs",
        # Database Observatory Logan tools
        "oci_logan_execute_query",
        "oci_logan_list_sources",
        "oci_logan_list_entities",
        "oci_logan_list_parsers",
        "oci_logan_list_labels",
        "oci_logan_list_groups",
        "oci_logan_run_security_query",
        "oci_logan_detect_anomalies",
        "oci_logan_get_summary",
        "oci_logan_suggest_query",
        "oci_logan_list_active_sources",
        "oci_logan_get_entity_logs",
        "oci_logan_list_skills",
    ]

    def __init__(
        self,
        memory_manager: "SharedMemoryManager | None" = None,
        tool_catalog: "ToolCatalog | None" = None,
        config: dict[str, Any] | None = None,
        llm: Any = None,
    ):
        """
        Initialize Log Analytics Agent with self-healing.

        Args:
            memory_manager: Shared memory manager
            tool_catalog: Tool catalog for MCP tools
            config: Agent configuration
            llm: LangChain LLM for analysis
        """
        super().__init__(memory_manager, tool_catalog, config)
        self.llm = llm
        self._graph: StateGraph | None = None

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
        """Return agent definition for catalog registration."""
        return AgentDefinition(
            agent_id="log-analytics-agent",
            role="log-analytics-agent",
            capabilities=[
                "log-search",
                "pattern-detection",
                "log-correlation",
                "audit-analysis",
                "anomaly-detection",
            ],
            skills=[
                "log_search_workflow",
                "pattern_analysis",
                "correlation_analysis",
                "audit_review",
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.log-analytics-agent"],
                produce=["results.log-analytics-agent"],
            ),
            health_endpoint="http://localhost:8011/health",
            metadata=AgentMetadata(
                version="1.0.0",
                namespace="oci-coordinator",
                max_iterations=10,
                timeout_seconds=45,
            ),
            description=(
                "Log Analytics Expert Agent for OCI log search, pattern detection, "
                "and correlation analysis across services."
            ),
            mcp_tools=cls.MCP_TOOLS,
            mcp_servers=["oci-unified", "database-observatory"],
        )

    def build_graph(self) -> StateGraph:
        """Build the log analytics workflow graph."""
        graph = StateGraph(LogAnalyticsState)

        graph.add_node("parse_query", self._parse_query_node)
        graph.add_node("search_logs", self._search_logs_node)
        graph.add_node("detect_patterns", self._detect_patterns_node)
        graph.add_node("correlate", self._correlate_node)
        graph.add_node("output", self._output_node)

        graph.set_entry_point("parse_query")
        graph.add_edge("parse_query", "search_logs")
        graph.add_edge("search_logs", "detect_patterns")
        graph.add_edge("detect_patterns", "correlate")
        graph.add_edge("correlate", "output")
        graph.add_edge("output", END)

        return graph.compile()

    async def _parse_query_node(self, state: LogAnalyticsState) -> dict[str, Any]:
        """Parse user query to extract search parameters."""
        self._logger.info("Parsing log query", query=state.query[:100])

        # Extract time range from query
        time_keywords = {
            "last hour": "1h",
            "last 6 hours": "6h",
            "last day": "24h",
            "yesterday": "24h",
            "last week": "7d",
            "last month": "30d",
        }
        time_range = "1h"
        for keyword, value in time_keywords.items():
            if keyword in state.query.lower():
                time_range = value
                break

        return {"time_range": time_range, "phase": "search_logs"}

    async def _search_logs_node(self, state: LogAnalyticsState) -> dict[str, Any]:
        """Search logs based on query parameters with self-healing."""
        self._logger.info("Searching logs", time_range=state.time_range)

        log_entries = []
        if self.tools:
            try:
                # Use self-healing for automatic retry on log search issues
                if self._self_healing_enabled:
                    result = await self.healing_call_tool(
                        "oci_observability_query_logs",
                        {
                            "compartment_id": state.compartment_id or "default",
                            "query": state.search_pattern or "error",
                            "time_range": state.time_range,
                        },
                        user_intent=state.query,
                        validate=True,
                        correct_on_failure=True,
                    )
                else:
                    result = await self.call_tool(
                        "oci_observability_query_logs",
                        {
                            "compartment_id": state.compartment_id or "default",
                            "query": state.search_pattern or "error",
                            "time_range": state.time_range,
                        },
                    )
                if isinstance(result, list):
                    log_entries = result
            except Exception as e:
                self._logger.warning("Log search failed", error=str(e))

        return {"log_entries": log_entries, "phase": "detect_patterns"}

    async def _detect_patterns_node(self, state: LogAnalyticsState) -> dict[str, Any]:
        """Detect patterns in log entries using frequency analysis."""
        self._logger.info("Detecting patterns", entry_count=len(state.log_entries))

        patterns = []
        anomalies = []

        # Common error patterns to detect
        error_patterns = {
            "ORA-": {"type": "oracle_error", "severity": "error"},
            "ERROR": {"type": "application_error", "severity": "error"},
            "WARN": {"type": "warning", "severity": "warning"},
            "timeout": {"type": "timeout", "severity": "error"},
            "connection refused": {"type": "connectivity", "severity": "error"},
            "authentication failed": {"type": "auth_failure", "severity": "high"},
            "denied": {"type": "permission_denied", "severity": "warning"},
            "exception": {"type": "exception", "severity": "error"},
            "failed": {"type": "failure", "severity": "warning"},
            "OutOfMemory": {"type": "resource_exhaustion", "severity": "critical"},
            "disk full": {"type": "resource_exhaustion", "severity": "critical"},
        }

        # Count pattern occurrences
        pattern_counts: dict[str, dict] = {}

        for entry in state.log_entries:
            message = entry.get("message", "").lower()
            timestamp = entry.get("timestamp", "")
            source = entry.get("source", "unknown")

            for pattern, info in error_patterns.items():
                if pattern.lower() in message:
                    key = f"{info['type']}:{pattern}"
                    if key not in pattern_counts:
                        pattern_counts[key] = {
                            "pattern": pattern,
                            "type": info["type"],
                            "severity": info["severity"],
                            "count": 0,
                            "sources": set(),
                            "first_seen": timestamp,
                            "last_seen": timestamp,
                            "samples": [],
                        }
                    pattern_counts[key]["count"] += 1
                    pattern_counts[key]["sources"].add(source)
                    pattern_counts[key]["last_seen"] = timestamp
                    if len(pattern_counts[key]["samples"]) < 3:
                        pattern_counts[key]["samples"].append(message[:200])

        # Convert to list and sort by count
        for key, data in pattern_counts.items():
            data["sources"] = list(data["sources"])  # Convert set to list
            patterns.append(data)

        patterns.sort(key=lambda x: x["count"], reverse=True)

        # Detect anomalies (high-frequency patterns in short time)
        for pattern in patterns:
            if pattern["count"] > 10:
                anomalies.append({
                    "type": "high_frequency",
                    "description": f"High frequency of '{pattern['pattern']}' errors ({pattern['count']} occurrences)",
                    "pattern": pattern["pattern"],
                    "count": pattern["count"],
                    "severity": "high" if pattern["count"] > 50 else "medium",
                    "first_seen": pattern["first_seen"],
                    "sources": pattern["sources"][:3],
                })

        # Detect error bursts (many errors from same source)
        source_errors: dict[str, int] = {}
        for entry in state.log_entries:
            if "error" in entry.get("level", "").lower():
                source = entry.get("source", "unknown")
                source_errors[source] = source_errors.get(source, 0) + 1

        for source, count in source_errors.items():
            if count > 5:
                anomalies.append({
                    "type": "error_burst",
                    "description": f"Error burst from {source}: {count} errors in time range",
                    "source": source,
                    "count": count,
                    "severity": "high" if count > 20 else "medium",
                })

        return {"patterns": patterns[:10], "anomalies": anomalies[:5], "phase": "correlate"}

    async def _correlate_node(self, state: LogAnalyticsState) -> dict[str, Any]:
        """Correlate logs across services using trace IDs and timestamps."""
        self._logger.info("Correlating logs")

        correlations = []

        # Group entries by trace_id if available
        trace_groups: dict[str, list[dict]] = {}
        for entry in state.log_entries:
            trace_id = entry.get("trace_id") or entry.get("traceId") or entry.get("correlationId")
            if trace_id:
                if trace_id not in trace_groups:
                    trace_groups[trace_id] = []
                trace_groups[trace_id].append(entry)

        # Find multi-service traces
        for trace_id, entries in trace_groups.items():
            services = set(e.get("source", "unknown") for e in entries)
            if len(services) > 1:
                # Multi-service trace - find the flow
                sorted_entries = sorted(entries, key=lambda x: x.get("timestamp", ""))
                has_error = any("error" in e.get("level", "").lower() for e in entries)

                correlations.append({
                    "type": "trace_correlation",
                    "trace_id": trace_id,
                    "services": list(services),
                    "event_count": len(entries),
                    "has_error": has_error,
                    "description": f"Request flowed through {len(services)} services" + (" (contains errors)" if has_error else ""),
                    "severity": "high" if has_error else "info",
                })

        # Temporal correlation - find patterns of errors occurring close together
        if state.anomalies:
            # If multiple services show errors in same time window
            error_sources = set()
            for anomaly in state.anomalies:
                if anomaly.get("type") == "error_burst":
                    error_sources.add(anomaly.get("source", "unknown"))

            if len(error_sources) > 1:
                correlations.append({
                    "type": "temporal_correlation",
                    "services": list(error_sources),
                    "description": f"Concurrent errors across {len(error_sources)} services suggest cascading failure",
                    "severity": "high",
                    "recommendation": "Investigate service dependencies",
                })

        # Pattern-based correlation (similar errors across services)
        pattern_services: dict[str, set] = {}
        for pattern in state.patterns:
            pattern_type = pattern.get("type", "unknown")
            if pattern_type not in pattern_services:
                pattern_services[pattern_type] = set()
            for source in pattern.get("sources", []):
                pattern_services[pattern_type].add(source)

        for pattern_type, services in pattern_services.items():
            if len(services) > 1:
                correlations.append({
                    "type": "pattern_correlation",
                    "pattern_type": pattern_type,
                    "services": list(services),
                    "description": f"Same error type '{pattern_type}' occurring in multiple services",
                    "severity": "medium",
                })

        return {"correlations": correlations[:5], "phase": "output"}

    async def _output_node(self, state: LogAnalyticsState) -> dict[str, Any]:
        """Prepare output with structured formatting."""
        from src.formatting import (
            ListItem,
            MetricValue,
            ResponseFooter,
            Severity,
            TableData,
            TableRow,
        )

        # Determine severity based on anomalies and patterns
        severity = "info"
        if state.anomalies:
            severity = "high"
        elif state.patterns:
            severity = "medium"

        # Create structured response
        response = self.create_response(
            title="Log Analytics Results",
            subtitle=f"Analysis for {state.time_range}",
            severity=severity,
            icon="📊",
        )

        # Add summary metrics
        response.add_metrics(
            "Summary",
            [
                MetricValue(
                    label="Time Range",
                    value=state.time_range,
                    severity=Severity.INFO,
                ),
                MetricValue(
                    label="Entries Found",
                    value=len(state.log_entries),
                    severity=Severity.INFO,
                ),
                MetricValue(
                    label="Patterns Detected",
                    value=len(state.patterns),
                    severity=Severity.MEDIUM if state.patterns else Severity.SUCCESS,
                ),
            ],
            divider_after=True,
        )

        # Add patterns as table
        if state.patterns:
            pattern_table = TableData(
                headers=["Pattern", "Count", "Severity"],
                rows=[
                    TableRow(
                        cells=[
                            p.get("pattern", "Unknown"),
                            str(p.get("count", 0)),
                            p.get("severity", "info").upper(),
                        ],
                        severity=Severity.HIGH if p.get("severity") == "error" else Severity.MEDIUM,
                    )
                    for p in state.patterns[:5]
                ],
            )
            response.add_table("Detected Patterns", pattern_table, divider_after=True)

        # Add anomalies
        if state.anomalies:
            anomaly_items = [
                ListItem(
                    text=a.get("description", "Unknown anomaly"),
                    details=a.get("timestamp", ""),
                    severity=Severity.HIGH,
                )
                for a in state.anomalies[:5]
            ]
            response.add_section(
                title="Anomalies Detected",
                list_items=anomaly_items,
                divider_after=True,
            )

        # Add correlations
        if state.correlations:
            correlation_items = [
                ListItem(
                    text=c.get("description", "Correlation"),
                    details=f"Services: {', '.join(c.get('services', []))}",
                    severity=Severity.MEDIUM,
                )
                for c in state.correlations[:3]
            ]
            response.add_section(
                title="Cross-Service Correlations",
                list_items=correlation_items,
            )

        # Add footer
        response.footer = ResponseFooter(
            next_steps=[
                "Investigate high-frequency patterns",
                "Review anomalous log entries",
            ],
            help_text="Use `/oci logs search <query>` for targeted search",
        )

        return {"result": self.format_response(response)}

    async def invoke(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Execute log analytics workflow."""
        context = context or {}

        graph = self.build_graph()
        initial_state = LogAnalyticsState(
            query=query,
            compartment_id=context.get("compartment_id"),
            log_group_id=context.get("log_group_id"),
            search_pattern=context.get("search_pattern"),
        )

        result = await graph.ainvoke(initial_state)
        return result.get("result", "No results found.")
