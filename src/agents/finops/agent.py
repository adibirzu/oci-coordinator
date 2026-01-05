"""
FinOps Agent.

Specialized agent for cost analysis, budget tracking,
and optimization recommendations in OCI.
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

    # LLM reasoning chain
    reasoning_chain: list[str] = field(default_factory=list)
    llm_analysis: str = ""

    # Temporal context
    start_date: str | None = None
    end_date: str | None = None
    comparison_period: str | None = None  # For trend comparisons

    # State
    phase: str = "analyze_query"
    total_cost: float = 0.0
    cost_trend: str = "stable"  # increasing, decreasing, stable
    error: str | None = None
    result: str | None = None


class FinOpsAgent(BaseAgent, SelfHealingMixin):
    """
    FinOps Agent with Self-Healing Capabilities.

    Specializes in OCI cost management:
    - Cost analysis and breakdown
    - Spending anomaly detection
    - Budget tracking
    - Rightsizing recommendations
    - Usage trend analysis

    Self-Healing Features:
    - Automatic retry with parameter correction on API failures
    - Pre-validation of tool calls before execution
    - LLM-powered error analysis for complex issues
    """

    def __init__(
        self,
        memory_manager: "SharedMemoryManager | None" = None,
        tool_catalog: "ToolCatalog | None" = None,
        config: dict[str, Any] | None = None,
        llm: Any = None,
    ):
        """
        Initialize FinOps Agent with self-healing.

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
                timeout_seconds=180,  # OCI Usage API (10-30s) + LLM reasoning + US-EMEA latency (~15-20s/call)
            ),
            description=(
                "FinOps Expert Agent for cost analysis, budget tracking, "
                "and optimization recommendations in OCI."
            ),
            mcp_servers=["oci-unified"],
        )

    def build_graph(self) -> StateGraph:
        """Build the FinOps analysis workflow graph with LLM reasoning."""
        graph = StateGraph(FinOpsState)

        # Add nodes including LLM reasoning
        graph.add_node("analyze_query", self._analyze_query_node)
        graph.add_node("get_costs", self._get_costs_node)
        graph.add_node("llm_reason", self._llm_reasoning_node)  # NEW: LLM analysis
        graph.add_node("detect_anomalies", self._detect_anomalies_node)
        graph.add_node("generate_recommendations", self._generate_recommendations_node)
        graph.add_node("output", self._output_node)

        # Flow: analyze → get_costs → llm_reason → detect_anomalies → recommendations → output
        graph.set_entry_point("analyze_query")
        graph.add_edge("analyze_query", "get_costs")
        graph.add_edge("get_costs", "llm_reason")  # Route through LLM
        graph.add_edge("llm_reason", "detect_anomalies")
        graph.add_edge("detect_anomalies", "generate_recommendations")
        graph.add_edge("generate_recommendations", "output")
        graph.add_edge("output", END)

        return graph.compile()

    async def _analyze_query_node(self, state: FinOpsState) -> dict[str, Any]:
        """Analyze cost query with temporal reasoning."""
        from datetime import datetime, timedelta
        import calendar

        self._logger.info("Analyzing FinOps query", query=state.query[:100])

        query_lower = state.query.lower()
        reasoning_chain = []
        today = datetime.now()

        # Start reasoning chain
        reasoning_chain.append(f"📝 **Analyzing query:** \"{state.query}\"")

        # Extract temporal context with reasoning
        time_range = "30d"
        start_date = None
        end_date = None
        comparison_period = None

        # Check for specific month names (November, December, etc.)
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

        detected_months = []
        for month_name, month_num in month_names.items():
            if month_name in query_lower:
                detected_months.append((month_name, month_num))

        if detected_months:
            # User asked for specific month(s)
            primary_month = detected_months[0]
            month_name, month_num = primary_month

            # Determine the year (assume current year, or previous year if month is in the future)
            year = today.year
            if month_num > today.month:
                year = today.year - 1

            # Calculate date range for that month
            _, last_day = calendar.monthrange(year, month_num)
            start_date = f"{year}-{month_num:02d}-01"
            end_date = f"{year}-{month_num:02d}-{last_day:02d}"

            # Calculate days from start of month to end
            start_dt = datetime(year, month_num, 1)
            end_dt = datetime(year, month_num, last_day)
            days_in_range = (end_dt - start_dt).days + 1
            time_range = f"{days_in_range}d"

            reasoning_chain.append(
                f"🗓️ **Temporal context:** User asked about {month_name.capitalize()} {year}"
            )
            reasoning_chain.append(
                f"📅 **Date range:** {start_date} to {end_date} ({days_in_range} days)"
            )

            # Check for comparison (e.g., "November vs December")
            if len(detected_months) > 1:
                comparison_period = detected_months[1][0].capitalize()
                reasoning_chain.append(
                    f"📊 **Comparison requested:** {month_name.capitalize()} vs {comparison_period}"
                )

        else:
            # Use relative time keywords with actual date calculation
            if "last month" in query_lower:
                # Calculate previous month's date range
                if today.month == 1:
                    year = today.year - 1
                    month = 12
                else:
                    year = today.year
                    month = today.month - 1

                _, last_day = calendar.monthrange(year, month)
                start_date = f"{year}-{month:02d}-01"
                end_date = f"{year}-{month:02d}-{last_day:02d}"
                time_range = f"{last_day}d"
                reasoning_chain.append(f"🗓️ **Temporal context:** Previous month ({start_date} to {end_date})")

            elif "this month" in query_lower:
                # Calculate current month's date range
                year = today.year
                month = today.month
                _, last_day = calendar.monthrange(year, month)
                start_date = f"{year}-{month:02d}-01"
                end_date = f"{year}-{month:02d}-{last_day:02d}"
                time_range = f"{last_day}d"
                reasoning_chain.append(f"🗓️ **Temporal context:** Current month ({start_date} to {end_date})")

            elif "last week" in query_lower:
                time_range = "7d"
                reasoning_chain.append("🗓️ **Temporal context:** Previous week")

            elif "this week" in query_lower:
                time_range = "7d"
                reasoning_chain.append("🗓️ **Temporal context:** Current week (7 days)")

            elif "last quarter" in query_lower:
                time_range = "90d"
                reasoning_chain.append("🗓️ **Temporal context:** Previous quarter")

            elif "this quarter" in query_lower:
                time_range = "90d"
                reasoning_chain.append("🗓️ **Temporal context:** Current quarter (90 days)")

            elif any(kw in query_lower for kw in ["year to date", "ytd"]):
                time_range = "365d"
                reasoning_chain.append("🗓️ **Temporal context:** Year to date")

            else:
                # Default to 30 days
                reasoning_chain.append("🗓️ **Temporal context:** Using default 30-day window (no specific time mentioned)")

        # Check for analysis type
        if any(kw in query_lower for kw in ["trend", "compare", "growth", "change"]):
            reasoning_chain.append("📈 **Analysis type:** Trend/comparison analysis requested")
        elif any(kw in query_lower for kw in ["optimize", "save", "reduce", "recommend"]):
            reasoning_chain.append("💡 **Analysis type:** Optimization recommendations requested")
        elif any(kw in query_lower for kw in ["anomaly", "unusual", "spike"]):
            reasoning_chain.append("⚠️ **Analysis type:** Anomaly detection requested")
        else:
            reasoning_chain.append("📊 **Analysis type:** General cost breakdown")

        self._logger.info(
            "Query analyzed",
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            reasoning_steps=len(reasoning_chain),
        )

        return {
            "time_range": time_range,
            "start_date": start_date,
            "end_date": end_date,
            "comparison_period": comparison_period,
            "reasoning_chain": reasoning_chain,
            "phase": "get_costs",
        }

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
            # Build parameters - include start/end dates if specified for historical queries
            tool_params = {
                "compartment_id": state.compartment_id,  # None is valid, tool handles it
                "days": days,
            }

            # Pass explicit dates for historical queries (e.g., "show costs for November")
            if state.start_date and state.end_date:
                tool_params["start_date"] = state.start_date
                tool_params["end_date"] = state.end_date
                self._logger.info(
                    "Using explicit date range",
                    start_date=state.start_date,
                    end_date=state.end_date,
                )

            # Use self-healing tool call for automatic retry and parameter correction
            try:
                if self._self_healing_enabled:
                    summary_result = await self.healing_call_tool(
                        "oci_cost_get_summary",
                        tool_params,
                        user_intent=state.query,
                        validate=True,
                        correct_on_failure=True,
                    )
                else:
                    summary_result = await self.call_tool(
                        "oci_cost_get_summary",
                        tool_params,
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

        # Add to reasoning chain
        reasoning_update = list(state.reasoning_chain)
        reasoning_update.append(f"💰 **Data retrieved:** Total cost ${total_cost:,.2f} across {len(cost_by_service)} services")

        return {
            "cost_summary": cost_summary,
            "cost_by_service": cost_by_service,
            "total_cost": total_cost,
            "reasoning_chain": reasoning_update,
            "phase": "llm_reason",
        }

    async def _llm_reasoning_node(self, state: FinOpsState) -> dict[str, Any]:
        """
        LLM-powered reasoning about cost data.

        This node uses the LLM to:
        1. Analyze the cost data in context of the user's query
        2. Provide insights beyond simple calculations
        3. Generate a "thought process" visible to the user
        """
        from langchain_core.messages import HumanMessage

        self._logger.info("Performing LLM reasoning on cost data")

        reasoning_update = list(state.reasoning_chain)
        llm_analysis = ""

        if self.llm and state.cost_by_service:
            try:
                # Build context for LLM
                services_text = "\n".join([
                    f"- {svc.get('name', 'Unknown')}: ${svc.get('cost', 0):,.2f} ({svc.get('percent', '0%')})"
                    for svc in sorted(state.cost_by_service, key=lambda x: x.get('cost', 0), reverse=True)[:10]
                ])

                prompt = f"""Analyze the following OCI cost data and provide insights.

**User Query:** {state.query}

**Time Period:** {state.time_range}
{f"**Start Date:** {state.start_date}" if state.start_date else ""}
{f"**End Date:** {state.end_date}" if state.end_date else ""}
{f"**Comparison Period:** {state.comparison_period}" if state.comparison_period else ""}

**Total Cost:** ${state.total_cost:,.2f}

**Top Services by Cost:**
{services_text}

Please provide:
1. A brief analysis of the cost distribution (1-2 sentences)
2. Key observations relevant to the user's query
3. Any concerns or areas worth investigating

Keep your response concise (3-4 bullet points max). Focus on actionable insights."""

                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                llm_analysis = response.content if hasattr(response, 'content') else str(response)

                reasoning_update.append("🤖 **LLM Analysis:**")
                # Add each line of the analysis to reasoning
                for line in llm_analysis.strip().split('\n'):
                    if line.strip():
                        reasoning_update.append(f"   {line.strip()}")

                self._logger.info(
                    "LLM reasoning completed",
                    analysis_length=len(llm_analysis),
                )

            except Exception as e:
                self._logger.warning("LLM reasoning failed", error=str(e))
                reasoning_update.append(f"⚠️ **Note:** LLM analysis unavailable ({str(e)[:50]})")
        else:
            reasoning_update.append("ℹ️ **Note:** Using rule-based analysis (LLM not available)")

        return {
            "reasoning_chain": reasoning_update,
            "llm_analysis": llm_analysis,
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
        """Prepare FinOps report with structured formatting and reasoning chain."""
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

        # Build subtitle with date context
        if state.start_date and state.end_date:
            subtitle = f"Cost analysis for {state.start_date} to {state.end_date}"
        else:
            subtitle = f"Cost analysis for {state.time_range}"

        # Create structured response
        response = self.create_response(
            title="FinOps Analysis Results",
            subtitle=subtitle,
            severity=severity,
        )

        # Add reasoning chain section (shows thought process)
        if state.reasoning_chain:
            reasoning_items = [
                ListItem(text=step, severity=Severity.INFO)
                for step in state.reasoning_chain
            ]
            response.add_section(
                title="🧠 Reasoning Process",
                list_items=reasoning_items,
                divider_after=True,
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
