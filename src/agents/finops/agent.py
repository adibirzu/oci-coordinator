"""
FinOps Agent.

Specialized agent for cost analysis, budget tracking,
and optimization recommendations in OCI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
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
from src.mcp.client import ToolCallResult
from src.oci.profile_manager import ProfileManager

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
    analysis_type: str = "summary"  # summary, trend, anomaly, optimization

    # Analysis results
    cost_summary: dict[str, Any] = field(default_factory=dict)
    cost_by_service: list[dict] = field(default_factory=list)
    cost_anomalies: list[dict] = field(default_factory=list)
    optimization_recommendations: list[dict] = field(default_factory=list)
    service_comparison: dict[str, Any] = field(default_factory=dict)  # New for Q7

    # LLM reasoning chain
    reasoning_chain: list[str] = field(default_factory=list)
    llm_analysis: str = ""

    # Temporal context
    start_date: str | None = None
    end_date: str | None = None
    comparison_period: str | None = None  # For trend comparisons
    oci_profile: str | None = None

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
        memory_manager: SharedMemoryManager | None = None,
        tool_catalog: ToolCatalog | None = None,
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

    def _extract_tool_payload(
        self, result: ToolCallResult | Any
    ) -> tuple[bool, Any, str | None]:
        """Normalize tool results and parse JSON payloads when possible."""
        if isinstance(result, ToolCallResult):
            if not result.success:
                return False, None, result.error or "Tool execution failed"
            payload = result.result
        else:
            payload = result

        if isinstance(payload, dict) and payload.get("error"):
            return False, payload, str(payload.get("error"))

        if isinstance(payload, str):
            cleaned = payload.strip()
            if cleaned:
                try:
                    payload = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        return True, payload, None

    @staticmethod
    def _schema_properties(schema: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        params_schema = props.get("params") if isinstance(props.get("params"), dict) else None
        if params_schema and isinstance(params_schema.get("properties"), dict):
            return params_schema["properties"]
        return props

    @staticmethod
    def _parse_amount(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace(",", "").split()[0])
            except (ValueError, IndexError):
                return 0.0
        return 0.0

    async def _resolve_tenancy_ocid(self, profile_name: str | None) -> str | None:
        manager = ProfileManager.get_instance()
        await manager.initialize()
        profile = manager.get_profile(profile_name or "DEFAULT")
        return profile.tenancy_ocid if profile else None

    @staticmethod
    def _resolve_date_range(state: FinOpsState, days: int) -> tuple[str, str]:
        if state.start_date and state.end_date:
            start_date = state.start_date
            end_date = state.end_date
        else:
            end = date.today()
            start = end - timedelta(days=days)
            start_date = start.isoformat()
            end_date = end.isoformat()

        today = date.today().isoformat()
        end_date = min(end_date, today)
        return start_date, end_date

    async def _build_cost_summary_params(
        self, state: FinOpsState, days: int
    ) -> tuple[str, dict[str, Any], str, str]:
        tool_name = "get_cost_summary"
        tool_def = None
        if self.tools:
            # Try multiple tool name variants - finopsai uses get_cost_summary
            # while oci-unified uses oci_cost_get_summary
            for candidate in (
                "get_cost_summary",  # finopsai primary tool
                "finops_cost_summary",  # finopsai multicloud summary
                "finopsai:get_cost_summary",  # Explicit server prefix
                "oci_cost_get_summary",  # oci-unified variant
                "oci-unified:oci_cost_get_summary",  # Explicit server prefix
                "oci-infrastructure:oci_cost_get_summary",  # Fallback
            ):
                tool_def = self.tools.get_tool(candidate)
                if tool_def:
                    tool_name = tool_def.name  # Use actual tool name from definition
                    self._logger.debug(
                        "Found cost tool",
                        candidate=candidate,
                        actual_name=tool_name,
                        server_id=tool_def.server_id,
                    )
                    break

            if not tool_def:
                self._logger.warning(
                    "No cost summary tool found",
                    tried_candidates=[
                        "get_cost_summary", "finops_cost_summary",
                        "oci_cost_get_summary"
                    ],
                )

        schema_props = self._schema_properties(tool_def.input_schema if tool_def else None)
        server_id = tool_def.server_id if tool_def else ""

        start_date, end_date = self._resolve_date_range(state, days)
        params: dict[str, Any] = {}

        if server_id in ("oci-infrastructure", "finopsai") or "time_start" in schema_props or "time_end" in schema_props:
            # finopsai expects YYYY-MM-DD format (no time component)
            # oci-infrastructure accepts full ISO datetime
            if server_id == "finopsai":
                params["time_start"] = start_date  # YYYY-MM-DD only
                params["time_end"] = end_date
            else:
                params["time_start"] = f"{start_date}T00:00:00Z"
                params["time_end"] = f"{end_date}T23:59:59Z"
            tenancy = await self._resolve_tenancy_ocid(state.oci_profile)
            if tenancy:
                params["tenancy_ocid"] = tenancy
            if "granularity" in schema_props:
                params["granularity"] = "DAILY"
            if "response_format" in schema_props:
                params["response_format"] = "json"
        else:
            params["compartment_id"] = state.compartment_id
            if state.start_date and state.end_date:
                params["start_date"] = start_date
                params["end_date"] = end_date
            else:
                params["days"] = days
            if state.oci_profile:
                params["profile"] = state.oci_profile
            if "response_format" in schema_props:
                params["response_format"] = "json"
            if "format" in schema_props:
                params["format"] = "json"

        return tool_name, params, start_date, end_date

    async def _enrich_with_finops_tools(
        self,
        state: FinOpsState,
        summary_data: dict[str, Any],
        cost_by_service: list[dict],
        reasoning_update: list[str],
    ) -> None:
        """
        Enrich cost analysis using additional finopsai MCP tools.

        Based on analysis_type, calls specialized tools:
        - anomaly: finops_detect_anomalies
        - optimization: finops_rightsizing
        - compare_services: oci_cost_service_drilldown
        - database costs: oci_cost_database_drilldown
        """
        if not self.tools:
            return

        try:
            tenancy = await self._resolve_tenancy_ocid(state.oci_profile)

            # Anomaly detection
            if state.analysis_type == "anomaly":
                anomaly_result = await self.call_tool(
                    "finops_detect_anomalies",
                    {
                        "days_back": 30,
                        "method": "z_score",
                        "threshold": 2,
                        "response_format": "json",
                    },
                )
                success, payload, _ = self._extract_tool_payload(anomaly_result)
                if success and payload:
                    anomalies = payload.get("anomalies", [])
                    if anomalies:
                        reasoning_update.append(f"🔍 **Anomaly scan:** Found {len(anomalies)} cost anomalies")
                        summary_data["anomalies"] = anomalies

            # Rightsizing / Optimization
            elif state.analysis_type == "optimization":
                rightsizing_result = await self.call_tool(
                    "finops_rightsizing",
                    {"resource_type": "compute", "min_savings": 10, "response_format": "json"},
                )
                success, payload, _ = self._extract_tool_payload(rightsizing_result)
                if success and payload:
                    recommendations = payload.get("recommendations", [])
                    if recommendations:
                        reasoning_update.append(f"💡 **Rightsizing:** Found {len(recommendations)} optimization opportunities")
                        summary_data["rightsizing"] = recommendations

            # Service drilldown
            elif state.analysis_type == "compare_services" and tenancy:
                drilldown_result = await self.call_tool(
                    "oci_cost_service_drilldown",
                    {
                        "tenancy_ocid": tenancy,
                        "time_start": state.start_date or (date.today() - timedelta(days=30)).isoformat(),
                        "time_end": state.end_date or date.today().isoformat(),
                        "top_n": 10,
                        "response_format": "json",
                    },
                )
                success, payload, _ = self._extract_tool_payload(drilldown_result)
                if success and payload:
                    services = payload.get("services", [])
                    if services:
                        reasoning_update.append(f"📊 **Service drilldown:** Analyzed top {len(services)} services")
                        # Merge into cost_by_service if we have better data
                        if len(services) > len(cost_by_service):
                            cost_by_service.clear()
                            for svc in services:
                                cost_by_service.append({
                                    "name": svc.get("service", "Unknown"),
                                    "cost": self._parse_amount(svc.get("cost", 0)),
                                    "percent": svc.get("percent", "0%"),
                                })

            # Database-specific cost analysis
            query_lower = state.query.lower()
            if any(kw in query_lower for kw in ["database", "db", "autonomous", "atp", "adw"]) and tenancy:
                db_result = await self.call_tool(
                    "oci_cost_database_drilldown",
                    {
                        "tenancy_ocid": tenancy,
                        "time_start": state.start_date or (date.today() - timedelta(days=30)).isoformat(),
                        "time_end": state.end_date or date.today().isoformat(),
                        "drilldown_level": "summary",
                        "response_format": "json",
                    },
                )
                success, payload, _ = self._extract_tool_payload(db_result)
                if success and payload:
                    db_cost = payload.get("total_cost", 0)
                    if db_cost:
                        reasoning_update.append(f"🗄️ **Database costs:** ${db_cost:,.2f} total")
                        summary_data["database_costs"] = payload

        except Exception as e:
            self._logger.debug("Enrichment with finops tools failed", error=str(e))
            # Don't fail the whole analysis, just skip enrichment

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
                timeout_seconds=300,  # Increased for slow OCI Usage API trend analysis (3+ months)
            ),
            description=(
                "FinOps Expert Agent for cost analysis, budget tracking, "
                "and optimization recommendations in OCI."
            ),
            mcp_servers=["finopsai", "oci-unified"],  # finopsai primary, oci-unified fallback
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
        # graph.add_edge("generate_recommendations", "output")
        # Update flow to support direct output for specific types if needed, but linear is fine
        graph.add_edge("generate_recommendations", "output")
        graph.add_edge("output", END)

        return graph.compile()

    async def _analyze_query_node(self, state: FinOpsState) -> dict[str, Any]:
        """Analyze cost query with temporal reasoning."""
        import calendar
        from datetime import datetime

        self._logger.info("Analyzing FinOps query", query=state.query[:100])

        query_lower = state.query.lower()
        reasoning_chain = []
        today = datetime.now()

        # Start reasoning chain
        reasoning_chain.append(f"📝 **Analyzing query:** \"{state.query}\"")

        # Extract temporal context with state persistence (Drill-down)
        # Default to existing state if available, else "30d"
        time_range = state.time_range or "30d"
        start_date = state.start_date
        end_date = state.end_date
        comparison_period = state.comparison_period

        # Check for compartment override in query
        # Example: "switch to compartment dev", "use production compartment"
        import re
        compartment_match = re.search(r"compartment\s+([a-zA-Z0-9_\-]+)", query_lower)
        if compartment_match:
            # We found a compartment name.
            # In a real app we'd resolve this name to OCID using Identity tools.
            # For now we assume the name/ID is passed or we set a marker for resolution.
            # Since resolving is an async tool call, we can't do it easily in this sync analysis block
            # without adding a resolve node.
            # We'll just note it in reasoning for now, or if it looks like an OCID, set it.
            potential_cmp = compartment_match.group(1)
            reasoning_chain.append(f"🆔 **Compartment detected:** '{potential_cmp}' (Will need resolution)")
            # For this MVP step, we won't implement full resolution logic here to avoid huge complexity shifts.
            # But we WILL persist the ID if it was passed in context (state provided).

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

        # Use relative time keywords with actual date calculation
        elif "last month" in query_lower:
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

        if any(kw in query_lower for kw in ["compare service", "database vs", "compute vs", "service comparison"]):
            analysis_type = "compare_services"
            reasoning_chain.append("⚖️ **Analysis type:** Service comparison requested")
        elif any(kw in query_lower for kw in ["forecast", "future", "predict", "next month"]):
            analysis_type = "trend"  # Forecast is part of trend tool
            reasoning_chain.append("🔮 **Analysis type:** Forecasting requested")
        elif any(kw in query_lower for kw in ["optimize", "save", "reduce", "recommend"]):
            analysis_type = "optimization"
            reasoning_chain.append("💡 **Analysis type:** Optimization recommendations requested")
        elif any(kw in query_lower for kw in ["anomaly", "unusual", "spike", "outlier"]):
            analysis_type = "anomaly"
            reasoning_chain.append("⚠️ **Analysis type:** Anomaly detection requested")
        elif any(kw in query_lower for kw in ["trend", "compare", "growth", "change", "month over month", "vs"]):
            analysis_type = "trend"
            reasoning_chain.append("📈 **Analysis type:** Trend/comparison analysis requested")
        else:
            analysis_type = "summary"
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
            "analysis_type": analysis_type,
        }

    async def _get_costs_node(self, state: FinOpsState) -> dict[str, Any]:
        """Get cost data from OCI using finopsai MCP tools."""

        self._logger.info("Getting cost data", time_range=state.time_range, analysis_type=state.analysis_type)

        cost_summary = {}
        cost_by_service = []
        total_cost = 0.0
        summary_data: dict[str, Any] = {}
        reasoning_update = list(state.reasoning_chain)  # Initialize early to avoid undefined reference

        if self.tools:
            # Parse days from time_range (e.g., "30d" -> 30)
            days = 30
            if state.time_range:
                try:
                    days = int(state.time_range.replace("d", ""))
                except ValueError:
                    pass

            tool_name, tool_params, start_date, end_date = await self._build_cost_summary_params(
                state, days
            )
            self._logger.info(
                "Using cost query range",
                start_date=start_date,
                end_date=end_date,
            )

            # Use self-healing tool call for automatic retry and parameter correction
            try:
                # BRANCH: Trend Analysis
                if state.analysis_type == "trend":
                    # Calculate months_back from days or default to 3
                    months_back = max(int(days / 30), 2)  # At least 2 months for comparison
                    if "last month" in state.query.lower():
                        months_back = 2  # Current + Previous

                    # Resolve tenancy OCID (required by finopsai MCP tool)
                    tenancy = await self._resolve_tenancy_ocid(state.oci_profile)

                    trend_params: dict[str, Any] = {
                        "months_back": months_back,
                        "include_forecast": True,
                    }
                    if tenancy:
                        trend_params["tenancy_ocid"] = tenancy
                    # Note: finopsai uses tenancy_ocid, not profile
                    summary_result = await self.healing_call_tool(
                        "oci_cost_monthly_trend",
                        trend_params,
                        user_intent=state.query,
                        validate=False,  # Skip validation to prevent LLM from adding invalid placeholders
                        correct_on_failure=False,  # Skip correction to avoid bad overrides
                    )

                # BRANCH: Summary / Default
                elif self._self_healing_enabled:
                    summary_result = await self.healing_call_tool(
                        tool_name,
                        tool_params,
                        user_intent=state.query,
                        validate=False,
                        correct_on_failure=False,
                    )
                else:
                    summary_result = await self.call_tool(
                        tool_name,
                        tool_params,
                    )

                success, payload, error = self._extract_tool_payload(summary_result)
                if not success:
                    self._logger.warning("Cost API returned error", error=error)
                    summary_data = payload if isinstance(payload, dict) else {"error": error}
                else:
                    summary_data = payload if isinstance(payload, dict) else {}

                # Check for error in response
                if summary_data.get("error"):
                    self._logger.warning("Cost API returned error", error=summary_data["error"])
                else:
                    if isinstance(summary_data.get("data"), dict):
                        summary_payload = summary_data["data"]
                    else:
                        summary_payload = summary_data

                    if "summary" in summary_payload:
                        cost_summary = summary_payload.get("summary", {})
                        total_cost = self._parse_amount(cost_summary.get("total", "0"))
                        services = summary_payload.get("services", [])
                        for svc in services:
                            cost_val = self._parse_amount(svc.get("cost", "0"))
                            cost_by_service.append({
                                "name": svc.get("service", "Unknown"),
                                "cost": cost_val,
                                "percent": svc.get("percent", "0%"),
                            })
                    elif "total_cost" in summary_payload:
                        currency = summary_payload.get("currency") or ""
                        total_cost = self._parse_amount(summary_payload.get("total_cost", 0))
                        cost_summary = {
                            "total": f"{total_cost:,.2f} {currency}".strip(),
                            "period": f"{start_date} \u2192 {end_date}",
                            "days": days,
                        }
                        services = summary_payload.get("by_service") or summary_payload.get("services") or []
                        for svc in services:
                            cost_val = self._parse_amount(svc.get("cost", 0))
                            pct = svc.get("percentage")
                            if isinstance(pct, (int, float)):
                                pct_str = f"{pct:.1f}%"
                            else:
                                pct_str = svc.get("percent", "0%")
                            cost_by_service.append({
                                "name": svc.get("service", "Unknown"),
                                "cost": cost_val,
                                "percent": pct_str,
                            })

                    self._logger.info(
                        "Cost data retrieved",
                        total=total_cost,
                        services_count=len(cost_by_service),
                    )

            except ValueError as e:
                # Tool not found - this is expected if only built-in tools available
                self._logger.warning("Cost tool not available", error=str(e))
                reasoning_update.append(f"⚠️ **Tool unavailable:** {e}")
            except Exception as e:
                self._logger.warning("Cost retrieval failed", error=str(e))
                reasoning_update.append(f"⚠️ **Cost retrieval error:** {str(e)[:100]}")

            # Try additional finopsai tools based on analysis type
            await self._enrich_with_finops_tools(state, summary_data, cost_by_service, reasoning_update)

        # Handle forecast specifically if requested
        if "forecast" in state.query.lower() and state.analysis_type == "trend":
            reasoning_update.append("🔮 **Forecast:** Generating future spend predictions based on historical trends.")

        # Add final reasoning update
        if summary_data.get("error"):
            reasoning_update.append(f"⚠️ **Cost API error:** {summary_data['error']}")
        elif total_cost > 0 or cost_by_service:
            reasoning_update.append(
                f"💰 **Data retrieved:** Total cost ${total_cost:,.2f} across {len(cost_by_service)} services"
            )
        else:
            reasoning_update.append("ℹ️ **Note:** No cost data returned. Check tenancy/compartment configuration.")

        return {
            "cost_summary": cost_summary,
            "cost_by_service": cost_by_service,
            "total_cost": total_cost,
            "error": summary_data.get("error"),
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

        if self.llm:
            try:
                # Build context for LLM
                services_text = "No services found with cost data."
                if state.cost_by_service:
                    services_text = "\n".join([
                        f"- {svc.get('name', 'Unknown')}: ${svc.get('cost', 0):,.2f} ({svc.get('percent', '0%')})"
                        for svc in sorted(state.cost_by_service, key=lambda x: x.get('cost', 0), reverse=True)[:10]
                    ])

                prompt = f"""Analyze the following OCI cost data and provide insights.

**User Query:** {state.query}
**Analysis Type:** {state.analysis_type.upper()}

**Time Period:** {state.time_range}
{f"**Start Date:** {state.start_date}" if state.start_date else ""}
{f"**End Date:** {state.end_date}" if state.end_date else ""}
{f"**Comparison Period:** {state.comparison_period}" if state.comparison_period else ""}

**Total Cost:** ${state.total_cost:,.2f}

**Top Services by Cost:**
{services_text}

Please provide:
1. A brief analysis or explanation of the cost.
2. Key observations relevant to the user's query.
3. Any concerns or areas worth investigating.

**Specific Instructions:**
- If **FORECAST** / **TREND**: Mention the direction of spend (increasing/decreasing) and any predicted values if available.
- If **COMPARE_SERVICES**: Explicitly compare the top services (e.g. Database vs Compute) and their relative contribution.
- If **TOTAL IS $0.00**: Explain potential reasons (empty compartment, free tier, wrong date) and suggest checks.

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
            reasoning_update.append("ℹ️ **Note:** Using rule-based analysis (LLM not configured)")

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

        # Output guidelines:
        # - Keep Slack/Teams summaries compact; avoid raw JSON blocks.
        # - Prefer top-N tables and short lists; include unit hints when meters are ambiguous.
        # - Streamlit output should stay markdown-first with raw_data for richer UI rendering.

        output_format = self.get_output_format().lower()

        # Create structured response
        response = self.create_response(
            title="FinOps Analysis Results",
            subtitle=subtitle,
            severity=severity,
        )

        # Add reasoning chain section (shows thought process)
        # Add reasoning chain section ONLY if cost is zero (shows troubleshooting)
        if state.reasoning_chain and state.total_cost == 0:
            reasoning_items = [
                ListItem(text=step, severity=Severity.INFO)
                for step in state.reasoning_chain
            ]
            response.add_section(
                title="🧠 Troubleshooting Zero Cost",
                list_items=reasoning_items,
                divider_after=True,
            )

        # Build metrics list
        metrics = [
            MetricValue(
                label="Total Cost",
                value=f"${state.total_cost:,.2f}",
                trend=trend_map.get(state.cost_trend),
                severity=Severity.MEDIUM if state.cost_trend == "increasing" else Severity.SUCCESS,
            )
        ]

        # Use precise dates if available, otherwise relative range
        if state.start_date and state.end_date:
            metrics.append(MetricValue(
                label="Period",
                value=f"{state.start_date} to {state.end_date}",
                severity=Severity.INFO
            ))
        else:
            metrics.append(MetricValue(
                label="Time Range",
                value=state.time_range,
                severity=Severity.INFO
            ))

        # Add cost metrics
        response.add_metrics(
            "Cost Summary",
            metrics,
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

            # Add concise unit explanations for chat channels
            if output_format in ("slack", "teams", "streamlit"):
                unit_hints = [
                    ("Compute", ("compute", "instance", "vm", "ocpu"), "OCPU-hours / instance-hours"),
                    ("Block Storage", ("block", "volume", "storage"), "GB-months + performance units"),
                    ("Object Storage", ("object", "archive"), "GB-months + requests/egress"),
                    ("Load Balancer", ("load balancer", "lb"), "capacity-hours + data/requests"),
                    ("Database", ("database", "autonomous", "mysql", "postgres", "exadata"), "OCPU-hours + storage GB-hours"),
                    ("Logging", ("logging", "log analytics"), "GB ingested/processed"),
                ]

                unit_items = []
                seen = set()
                for svc in sorted_services:
                    service_name = (svc.get("name") or "").lower()
                    for label, keywords, note in unit_hints:
                        if note in seen:
                            continue
                        if any(keyword in service_name for keyword in keywords):
                            unit_items.append(ListItem(text=f"{label}: {note}", severity=Severity.INFO))
                            seen.add(note)
                    if len(unit_items) >= 4:
                        break

                if unit_items:
                    response.add_section(
                        title="Units Explained",
                        list_items=unit_items,
                    )

        # Add Service Comparison Table (New for Q7/Optimization)
        if state.analysis_type == "compare_services" and state.cost_by_service:
             # Just show the top breakdown which is already added above, maybe emphasize?
             # Or add a specific comparison section
             pass

        # Add Forecast Section (New for Q6)
        # Check if trend data has forecast
        if state.analysis_type == "trend" and isinstance(state.cost_summary, dict) and "forecast" in state.cost_summary:
            forecast = state.cost_summary["forecast"]
            est = forecast.get("estimate", 0)
            trend_dir = forecast.get("trend", "stable")

            f_items = [
                ListItem(
                    text="Estimated Spend Next Month",
                    details=f"${est:,.2f} ({trend_dir})",
                    severity=Severity.INFO
                )
            ]
            response.add_section(title="🔮 Forecast", list_items=f_items, divider_after=True)


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
        metadata = context.get("metadata", {}) if isinstance(context, dict) else {}

        graph = self.build_graph()
        initial_state = FinOpsState(
            query=query,
            compartment_id=context.get("compartment_id"),
            service_filter=context.get("service_filter"),
            oci_profile=metadata.get("oci_profile"),
        )

        result = await graph.ainvoke(initial_state)
        return result.get("result", "No cost data available.")
