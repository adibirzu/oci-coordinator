"""
FinOps Agent.

Specialized agent for cost analysis, budget tracking,
and optimization recommendations in OCI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from src.agents.base import (
    AgentDefinition,
    AgentMetadata,
    BaseAgent,
    KafkaTopics,
)

logger = structlog.get_logger(__name__)


@dataclass
class FinOpsState:
    """State for FinOps analysis workflow."""

    query: str = ""
    compartment_id: str | None = None
    time_range: str = "30d"
    service_filter: str | None = None

    # Analysis results
    cost_summary: dict[str, Any] = field(default_factory=dict)
    cost_by_service: list[dict] = field(default_factory=list)
    cost_anomalies: list[dict] = field(default_factory=list)
    optimization_recommendations: list[dict] = field(default_factory=list)

    # State
    phase: str = "analyze_query"
    total_cost: float = 0.0
    cost_trend: str = "stable"  # increasing, decreasing, stable
    error: str | None = None
    result: str | None = None


class FinOpsAgent(BaseAgent):
    """
    FinOps Agent.

    Specializes in OCI cost management:
    - Cost analysis and breakdown
    - Spending anomaly detection
    - Budget tracking
    - Rightsizing recommendations
    - Usage trend analysis
    """

    @classmethod
    def get_definition(cls) -> AgentDefinition:
        """Return agent definition for catalog registration."""
        return AgentDefinition(
            agent_id="finops-agent",
            role="finops-agent",
            capabilities=[
                "cost-analysis",
                "budget-tracking",
                "optimization",
                "anomaly-detection",
                "usage-forecasting",
            ],
            skills=[
                "cost_breakdown_workflow",
                "anomaly_detection",
                "rightsizing_analysis",
                "budget_check",
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.finops-agent"],
                produce=["results.finops-agent"],
            ),
            health_endpoint="http://localhost:8013/health",
            metadata=AgentMetadata(
                version="1.0.0",
                namespace="oci-coordinator",
                max_iterations=10,
                timeout_seconds=30,
            ),
            description=(
                "FinOps Expert Agent for cost analysis, budget tracking, "
                "and optimization recommendations in OCI."
            ),
            mcp_tools=[
                "oci_cost_get_summary",  # Primary cost tool with 30s timeout
            ],
            mcp_servers=["oci-unified"],
        )

    def build_graph(self) -> StateGraph:
        """Build the FinOps analysis workflow graph."""
        graph = StateGraph(FinOpsState)

        graph.add_node("analyze_query", self._analyze_query_node)
        graph.add_node("get_costs", self._get_costs_node)
        graph.add_node("detect_anomalies", self._detect_anomalies_node)
        graph.add_node("generate_recommendations", self._generate_recommendations_node)
        graph.add_node("output", self._output_node)

        graph.set_entry_point("analyze_query")
        graph.add_edge("analyze_query", "get_costs")
        graph.add_edge("get_costs", "detect_anomalies")
        graph.add_edge("detect_anomalies", "generate_recommendations")
        graph.add_edge("generate_recommendations", "output")
        graph.add_edge("output", END)

        return graph.compile()

    async def _analyze_query_node(self, state: FinOpsState) -> dict[str, Any]:
        """Analyze cost query."""
        self._logger.info("Analyzing FinOps query", query=state.query[:100])

        # Extract time range
        time_keywords = {
            "today": "1d",
            "yesterday": "1d",
            "this week": "7d",
            "last week": "7d",
            "this month": "30d",
            "last month": "30d",
        }
        time_range = "30d"
        for keyword, value in time_keywords.items():
            if keyword in state.query.lower():
                time_range = value
                break

        return {"time_range": time_range, "phase": "get_costs"}

    async def _get_costs_node(self, state: FinOpsState) -> dict[str, Any]:
        """Get cost data from OCI."""
        import json

        self._logger.info("Getting cost data", time_range=state.time_range)

        cost_summary = {}
        cost_by_service = []
        total_cost = 0.0

        if self.tools:
            # Parse days from time_range (e.g., "30d" -> 30)
            days = 30
            if state.time_range:
                try:
                    days = int(state.time_range.replace("d", ""))
                except ValueError:
                    pass

            # Get cost summary - compartment_id=None uses tenancy from OCI config
            try:
                summary_result = await self.call_tool(
                    "oci_cost_get_summary",
                    {
                        "compartment_id": state.compartment_id,  # None is valid, tool handles it
                        "days": days,
                    },
                )

                # Parse JSON response
                if isinstance(summary_result, str):
                    try:
                        summary_data = json.loads(summary_result)
                    except json.JSONDecodeError:
                        summary_data = {}
                elif isinstance(summary_result, dict):
                    summary_data = summary_result
                else:
                    summary_data = {}

                # Check for error in response
                if summary_data.get("error"):
                    self._logger.warning("Cost API returned error", error=summary_data["error"])
                else:
                    cost_summary = summary_data.get("summary", {})

                    # Parse total from the formatted string (e.g., "1,234.56 USD")
                    total_str = cost_summary.get("total", "0")
                    try:
                        total_cost = float(total_str.replace(",", "").split()[0])
                    except (ValueError, IndexError):
                        total_cost = 0.0

                    # Extract service breakdown
                    services = summary_data.get("services", [])
                    for svc in services:
                        cost_str = svc.get("cost", "0")
                        try:
                            cost_val = float(cost_str.replace(",", "").split()[0])
                        except (ValueError, IndexError):
                            cost_val = 0.0
                        cost_by_service.append({
                            "name": svc.get("service", "Unknown"),
                            "cost": cost_val,
                            "percent": svc.get("percent", "0%"),
                        })

                    self._logger.info(
                        "Cost data retrieved",
                        total=total_cost,
                        services_count=len(cost_by_service),
                    )

            except ValueError as e:
                # Tool not found - this is expected if only built-in tools available
                self._logger.warning("Cost tool not available", error=str(e))
            except Exception as e:
                self._logger.warning("Cost retrieval failed", error=str(e))

        return {
            "cost_summary": cost_summary,
            "cost_by_service": cost_by_service,
            "total_cost": total_cost,
            "phase": "detect_anomalies",
        }

    async def _detect_anomalies_node(self, state: FinOpsState) -> dict[str, Any]:
        """Detect cost anomalies - analyzes service data from cost summary."""
        self._logger.info("Detecting cost anomalies")

        anomalies = []
        cost_trend = "stable"

        # Analyze cost by service for anomalies
        if state.cost_by_service:
            # Flag services consuming > 40% of budget
            for service in state.cost_by_service:
                cost = service.get("cost", 0)
                if state.total_cost > 0 and cost / state.total_cost > 0.4:
                    anomalies.append({
                        "type": "high_concentration",
                        "description": f"{service.get('name', 'Unknown')} accounts for {cost/state.total_cost*100:.1f}% of total cost",
                        "service": service.get("name"),
                        "amount": cost,
                        "severity": "medium",
                    })

        return {
            "cost_anomalies": anomalies,
            "cost_trend": cost_trend,
            "phase": "generate_recommendations",
        }

    async def _generate_recommendations_node(
        self, state: FinOpsState
    ) -> dict[str, Any]:
        """Generate optimization recommendations based on cost analysis."""
        self._logger.info("Generating optimization recommendations")

        recommendations = []

        # Analyze cost by service for optimization opportunities
        for service in state.cost_by_service:
            cost = service.get("cost", 0)
            name = service.get("name", "Unknown")

            # High cost concentration recommendation
            if state.total_cost > 0 and cost / state.total_cost > 0.4:
                recommendations.append({
                    "type": "review",
                    "service": name,
                    "potential_savings": cost * 0.1,  # Estimate 10% savings from review
                    "action": f"Review {name} usage - accounts for {cost/state.total_cost*100:.0f}% of spend",
                })

            # Compute-specific recommendations
            if "compute" in name.lower() and cost > 100:
                recommendations.append({
                    "type": "rightsizing",
                    "service": name,
                    "potential_savings": cost * 0.2,
                    "action": "Consider rightsizing compute instances or using preemptible shapes",
                })

            # Storage optimization
            if "storage" in name.lower() or "block" in name.lower():
                if cost > 50:
                    recommendations.append({
                        "type": "optimization",
                        "service": name,
                        "potential_savings": cost * 0.15,
                        "action": "Review storage tiers - consider Standard vs. Archive for cold data",
                    })

        # Add recommendations based on anomalies
        for anomaly in state.cost_anomalies[:3]:
            if anomaly.get("type") == "high_concentration":
                recommendations.append({
                    "type": "investigate",
                    "service": anomaly.get("service", "Unknown"),
                    "potential_savings": anomaly.get("amount", 0) * 0.1,
                    "action": f"Investigate high cost concentration in {anomaly.get('service', 'Unknown')}",
                })

        # Limit to top 5 recommendations
        recommendations = recommendations[:5]

        return {"optimization_recommendations": recommendations, "phase": "output"}

    async def _output_node(self, state: FinOpsState) -> dict[str, Any]:
        """Prepare FinOps report with structured formatting."""
        from src.formatting import (
            ListItem,
            MetricValue,
            ResponseFooter,
            Severity,
            TableData,
            TableRow,
            TrendDirection,
        )

        # Map trend
        trend_map = {
            "increasing": TrendDirection.UP,
            "decreasing": TrendDirection.DOWN,
            "stable": TrendDirection.STABLE,
        }

        # Determine severity based on trend and anomalies
        severity = "info"
        if state.cost_trend == "increasing" and state.total_cost > 1000:
            severity = "medium"
        if state.cost_anomalies:
            severity = "high"

        # Create structured response
        response = self.create_response(
            title="FinOps Analysis Results",
            subtitle=f"Cost analysis for {state.time_range}",
            severity=severity,
        )

        # Add cost metrics
        response.add_metrics(
            "Cost Summary",
            [
                MetricValue(
                    label="Total Cost",
                    value=f"${state.total_cost:,.2f}",
                    trend=trend_map.get(state.cost_trend),
                    severity=Severity.MEDIUM if state.cost_trend == "increasing" else Severity.SUCCESS,
                ),
                MetricValue(
                    label="Time Range",
                    value=state.time_range,
                    severity=Severity.INFO,
                ),
            ],
            divider_after=True,
        )

        # Add cost by service as table
        if state.cost_by_service:
            sorted_services = sorted(
                state.cost_by_service, key=lambda x: x.get("cost", 0), reverse=True
            )[:5]

            table = TableData(
                title="Top Services by Cost",
                headers=["Service", "Cost"],
                rows=[
                    TableRow(
                        cells=[svc.get("name", "Unknown"), f"${svc.get('cost', 0):,.2f}"],
                        severity=Severity.HIGH if svc.get("cost", 0) > state.total_cost * 0.3 else None,
                    )
                    for svc in sorted_services
                ],
            )
            response.add_table("Cost Breakdown", table, divider_after=True)

        # Add anomalies if any
        if state.cost_anomalies:
            anomaly_items = [
                ListItem(
                    text=a.get("description", "Unknown anomaly"),
                    severity=Severity.HIGH,
                )
                for a in state.cost_anomalies[:3]
            ]
            response.add_section(
                title="Anomalies Detected",
                list_items=anomaly_items,
                divider_after=True,
            )

        # Add recommendations
        if state.optimization_recommendations:
            rec_items = [
                ListItem(
                    text=r.get("action", ""),
                    details=f"Potential savings: ${r.get('potential_savings', 0):,.2f}",
                    severity=Severity.SUCCESS,
                )
                for r in state.optimization_recommendations[:3]
            ]
            response.add_recommendations(rec_items)

        # Add footer
        response.footer = ResponseFooter(
            help_text="Use `/oci cost <compartment>` for detailed breakdown",
        )

        return {"result": self.format_response(response)}

    async def invoke(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Execute FinOps analysis workflow."""
        context = context or {}

        graph = self.build_graph()
        initial_state = FinOpsState(
            query=query,
            compartment_id=context.get("compartment_id"),
            service_filter=context.get("service_filter"),
        )

        result = await graph.ainvoke(initial_state)
        return result.get("result", "No cost data available.")
