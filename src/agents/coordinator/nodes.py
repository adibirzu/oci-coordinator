"""
LangGraph nodes for the Coordinator.

Implements the graph nodes:
- input_node: Process initial query
- classifier_node: Classify intent
- router_node: Determine routing (workflow vs agent vs direct)
- workflow_node: Execute deterministic workflow
- agent_node: Delegate to specialized agent
- action_node: Execute tool calls
- output_node: Prepare final response
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import os

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentelemetry import trace

from src.agents.coordinator.state import (
    AgentContext,
    CoordinatorState,
    IntentCategory,
    IntentClassification,
    RoutingType,
    ToolCall,
    ToolResult,
    determine_routing,
)
from src.agents.coordinator.transparency import (
    AgentCandidate,
    ThinkingPhase,
    ThinkingTrace,
    create_thinking_trace,
)

# Get tracer for coordinator nodes
_tracer = trace.get_tracer("oci-coordinator")

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.agents.catalog import AgentCatalog
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


class CoordinatorNodes:
    """
    Collection of LangGraph nodes for the Coordinator.

    Implements workflow-first routing:
    - 70%+ requests → deterministic workflows
    - Multi-domain queries → parallel multi-agent execution
    - Remaining → agentic LLM reasoning
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tool_catalog: ToolCatalog,
        agent_catalog: AgentCatalog,
        memory: SharedMemoryManager,
        workflow_registry: dict[str, Any] | None = None,
        orchestrator: Any | None = None,
    ):
        """
        Initialize coordinator nodes.

        Args:
            llm: LangChain chat model for reasoning
            tool_catalog: Catalog of MCP tools
            agent_catalog: Catalog of specialized agents
            memory: Shared memory manager
            workflow_registry: Map of workflow names to workflow functions
            orchestrator: Parallel orchestrator for multi-agent execution
        """
        self.llm = llm
        self.tool_catalog = tool_catalog
        self.agent_catalog = agent_catalog
        self.memory = memory
        self.workflow_registry = workflow_registry or {}
        self.orchestrator = orchestrator
        self._logger = logger.bind(component="CoordinatorNodes")

    # ─────────────────────────────────────────────────────────────────────────
    # Input Node
    # ─────────────────────────────────────────────────────────────────────────

    async def input_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Process initial input.

        Extracts query from messages and prepares for classification.
        Initializes the thinking trace for transparency.

        Args:
            state: Current coordinator state

        Returns:
            State updates
        """
        with _tracer.start_as_current_span("coordinator.input") as span:
            span.set_attribute("message_count", len(state.messages))
            self._logger.debug("Processing input", message_count=len(state.messages))

            # Extract query from last human message
            query = ""
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break

            if not query:
                span.set_attribute("error", "no_query")
                return {"error": "No query found in messages"}

            span.set_attribute("query_length", len(query))
            span.set_attribute("query_preview", query[:100])

            # Initialize thinking trace for transparency
            thinking_trace = create_thinking_trace()
            thinking_trace.start(query)

            return {
                "query": query,
                "iteration": 0,
                "thinking_trace": thinking_trace,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Classifier Node
    # ─────────────────────────────────────────────────────────────────────────

    async def classifier_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Classify user intent.

        Uses keyword pre-classification followed by LLM to understand:
        - What the user wants
        - Which domain(s) are involved
        - Confidence level
        - Workflow/agent suggestions

        Args:
            state: Current coordinator state

        Returns:
            State updates with intent classification
        """
        with _tracer.start_as_current_span("coordinator.classifier") as span:
            span.set_attribute("query_preview", state.query[:100] if state.query else "")
            self._logger.info("Classifying intent", query=state.query[:100])

            # Add thinking step for classification start
            thinking_trace = state.thinking_trace
            if thinking_trace:
                thinking_trace.add_step(
                    ThinkingPhase.CLASSIFYING,
                    "Analyzing query to understand intent...",
                    {"query_preview": state.query[:100]},
                )

            # Pre-classification: Check for domain-specific queries
            # This is more reliable than LLM for specific patterns

            # 1. Check database listing queries (highest priority for common requests)
            pre_classified = self._pre_classify_database_query(state.query)
            if pre_classified:
                span.set_attribute("pre_classification", "database_listing")
                self._logger.info(
                    "Pre-classified database listing query",
                    intent=pre_classified.intent,
                    workflow=pre_classified.suggested_workflow,
                )
                # Add thinking step for successful pre-classification
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.CLASSIFIED,
                        f"Detected database query → Intent: {pre_classified.intent}",
                        {
                            "method": "pre-classification",
                            "intent": pre_classified.intent,
                            "confidence": pre_classified.confidence,
                            "domains": pre_classified.domains,
                        },
                    )
                return {"intent": pre_classified}

            # 2. Check resource-cost mapping queries (e.g., "what's costing me the most")
            pre_classified = self._pre_classify_resource_cost_query(state.query)
            if pre_classified:
                span.set_attribute("pre_classification", "resource_cost_mapping")
                self._logger.info(
                    "Pre-classified resource-cost mapping query",
                    intent=pre_classified.intent,
                    workflow=pre_classified.suggested_workflow,
                )
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.CLASSIFIED,
                        f"Detected cost analysis query → Intent: {pre_classified.intent}",
                        {
                            "method": "pre-classification",
                            "intent": pre_classified.intent,
                            "confidence": pre_classified.confidence,
                            "domains": pre_classified.domains,
                        },
                    )
                return {"intent": pre_classified}

            # 3. Check domain-specific cost queries
            pre_classified = self._pre_classify_cost_query(state.query)
            if pre_classified:
                span.set_attribute("pre_classification", "cost_domain_specific")
                self._logger.info(
                    "Pre-classified domain-specific cost query",
                    intent=pre_classified.intent,
                    domains=pre_classified.domains,
                )
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.CLASSIFIED,
                        f"Detected cost query → Intent: {pre_classified.intent}",
                        {
                            "method": "pre-classification",
                            "intent": pre_classified.intent,
                            "confidence": pre_classified.confidence,
                            "domains": pre_classified.domains,
                        },
                    )
                return {"intent": pre_classified}

            # 4. Check DB Management queries (fleet health, AWR, SQL performance)
            pre_classified = self._pre_classify_dbmgmt_query(state.query)
            if pre_classified:
                span.set_attribute("pre_classification", "dbmgmt")
                self._logger.info(
                    "Pre-classified DB Management query",
                    intent=pre_classified.intent,
                    workflow=pre_classified.suggested_workflow,
                )
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.CLASSIFIED,
                        f"Detected DB Management query → Intent: {pre_classified.intent}",
                        {
                            "method": "pre-classification",
                            "intent": pre_classified.intent,
                            "confidence": pre_classified.confidence,
                            "domains": pre_classified.domains,
                        },
                    )
                return {"intent": pre_classified}

            # 5. Check OPSI queries (ADDM, capacity, insights)
            pre_classified = self._pre_classify_opsi_query(state.query)
            if pre_classified:
                span.set_attribute("pre_classification", "opsi")
                self._logger.info(
                    "Pre-classified OPSI query",
                    intent=pre_classified.intent,
                    workflow=pre_classified.suggested_workflow,
                )
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.CLASSIFIED,
                        f"Detected OPSI query → Intent: {pre_classified.intent}",
                        {
                            "method": "pre-classification",
                            "intent": pre_classified.intent,
                            "confidence": pre_classified.confidence,
                            "domains": pre_classified.domains,
                        },
                    )
                return {"intent": pre_classified}

            # 6. Check identity queries (compartments, tenancy, regions)
            pre_classified = self._pre_classify_identity_query(state.query)
            if pre_classified:
                span.set_attribute("pre_classification", "identity")
                self._logger.info(
                    "Pre-classified identity query",
                    intent=pre_classified.intent,
                    workflow=pre_classified.suggested_workflow,
                )
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.CLASSIFIED,
                        f"Detected identity query → Intent: {pre_classified.intent}",
                        {
                            "method": "pre-classification",
                            "intent": pre_classified.intent,
                            "confidence": pre_classified.confidence,
                            "domains": pre_classified.domains,
                        },
                    )
                return {"intent": pre_classified}

            # No pre-classification match - use LLM
            if thinking_trace:
                thinking_trace.add_step(
                    ThinkingPhase.CLASSIFYING,
                    "Using LLM for detailed intent analysis...",
                    {"method": "llm"},
                )

            # Build classification prompt
            classification_prompt = self._build_classification_prompt(state.query)

            try:
                response = await self.llm.ainvoke([HumanMessage(content=classification_prompt)])

                # Parse classification from response
                intent = self._parse_classification(response.content, state.query)

                span.set_attribute("intent.name", intent.intent)
                span.set_attribute("intent.category", intent.category.value)
                span.set_attribute("intent.confidence", intent.confidence)
                span.set_attribute("intent.domains", ",".join(intent.domains))

                self._logger.info(
                    "Intent classified",
                    intent=intent.intent,
                    category=intent.category.value,
                    confidence=intent.confidence,
                    domains=intent.domains,
                )

                # Add thinking step for LLM classification result
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.CLASSIFIED,
                        f"LLM classified → Intent: {intent.intent} ({intent.confidence:.0%} confidence)",
                        {
                            "method": "llm",
                            "intent": intent.intent,
                            "category": intent.category.value,
                            "confidence": intent.confidence,
                            "domains": intent.domains,
                        },
                    )

                return {"intent": intent}

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e)[:200])
                self._logger.error("Classification failed", error=str(e))
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.ERROR,
                        f"Classification failed: {str(e)[:100]}",
                        {"error": str(e)},
                    )
                # Return low-confidence fallback
                return {
                    "intent": IntentClassification(
                        intent="unknown",
                        category=IntentCategory.UNKNOWN,
                        confidence=0.3,
                        domains=[],
                    )
                }

    def _extract_date_range(self, query: str) -> dict[str, str]:
        """
        Extract date range from natural language queries.

        Handles patterns like:
        - "for November" → 2025-11-01 to 2025-11-30
        - "last month" → previous month's date range
        - "in October" → 2025-10-01 to 2025-10-31

        Returns:
            Dict with 'start_date' and 'end_date' in YYYY-MM-DD format, or empty dict
        """
        import re
        from datetime import datetime
        from calendar import monthrange

        query_lower = query.lower()
        today = datetime.now()

        # Month names to numbers
        month_map = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }

        # Check for explicit month names (e.g., "for November", "in October")
        for month_name, month_num in month_map.items():
            if month_name in query_lower:
                # Determine year - assume current year unless month is in the future
                year = today.year
                if month_num > today.month:
                    year -= 1  # Previous year if month hasn't happened yet

                # Get last day of the month
                _, last_day = monthrange(year, month_num)

                return {
                    "start_date": f"{year}-{month_num:02d}-01",
                    "end_date": f"{year}-{month_num:02d}-{last_day:02d}",
                }

        # Check for "last month" pattern
        if "last month" in query_lower:
            if today.month == 1:
                year = today.year - 1
                month = 12
            else:
                year = today.year
                month = today.month - 1

            _, last_day = monthrange(year, month)
            return {
                "start_date": f"{year}-{month:02d}-01",
                "end_date": f"{year}-{month:02d}-{last_day:02d}",
            }

        # Check for "this month" pattern
        if "this month" in query_lower:
            year = today.year
            month = today.month
            _, last_day = monthrange(year, month)
            return {
                "start_date": f"{year}-{month:02d}-01",
                "end_date": f"{year}-{month:02d}-{last_day:02d}",
            }

        return {}

    def _pre_classify_cost_query(self, query: str) -> IntentClassification | None:
        """
        Pre-classify domain-specific cost queries using keyword matching.

        UPDATED: Now checks for complexity indicators to route to LLM agent
        when queries require reasoning (temporal, analytical, optimization).
        Also extracts date ranges for historical queries.

        Simple queries like "show costs" → workflow
        Complex queries like "November costs compared to December" → agent with LLM
        """
        from src.agents.coordinator.state import has_complexity_indicators

        query_lower = query.lower()

        # Check if this is a cost-related query
        cost_keywords = ["cost", "spend", "spending", "budget", "expensive", "price", "billing"]
        is_cost_query = any(kw in query_lower for kw in cost_keywords)

        if not is_cost_query:
            return None

        # Extract date range from query (e.g., "November", "last month")
        date_entities = self._extract_date_range(query)

        # Check if query requires LLM reasoning (temporal, analytical, etc.)
        needs_reasoning = has_complexity_indicators(query)

        # Check for domain-specific keywords
        domain_patterns = {
            "database": {
                "keywords": ["database", "db", "autonomous", "atp", "adw", "exadata", "mysql", "nosql"],
                "intent": "database_costs",
                "workflow": "database_costs",
            },
            "compute": {
                "keywords": ["compute", "instance", "vm", "virtual machine", "server"],
                "intent": "compute_costs",
                "workflow": "compute_costs",
            },
            "storage": {
                "keywords": ["storage", "block", "object", "file storage", "archive"],
                "intent": "storage_costs",
                "workflow": "storage_costs",
            },
            "network": {
                "keywords": ["network", "vcn", "load balancer", "fastconnect", "bandwidth"],
                "intent": "network_costs",
                "workflow": "network_costs",
            },
        }

        for domain, config in domain_patterns.items():
            if any(kw in query_lower for kw in config["keywords"]):
                workflow = config["workflow"]
                if workflow not in self.workflow_registry:
                    workflow = "cost_summary"

                # If query needs reasoning, lower confidence so router sends to agent
                confidence = 0.70 if needs_reasoning else 0.95
                category = IntentCategory.ANALYSIS if needs_reasoning else IntentCategory.QUERY

                # Include extracted date entities
                entities = dict(date_entities)  # Copy date entities if any

                return IntentClassification(
                    intent=config["intent"],
                    category=category,
                    confidence=confidence,
                    domains=[domain, "cost"],
                    entities=entities,
                    suggested_workflow=workflow,
                    suggested_agent="finops-agent",  # Always include for fallback
                )

        # General cost query (not domain-specific)
        # If it needs reasoning, route to agent; otherwise to workflow
        if needs_reasoning:
            return IntentClassification(
                intent="cost_analysis",
                category=IntentCategory.ANALYSIS,
                confidence=0.70,  # Lower confidence triggers agent routing
                domains=["cost"],
                entities=dict(date_entities),
                suggested_workflow="cost_summary",
                suggested_agent="finops-agent",
            )

        # Simple cost query with date range (e.g., "costs for November")
        if date_entities:
            self._logger.info(
                "Pre-classified cost query with date range",
                start_date=date_entities.get("start_date"),
                end_date=date_entities.get("end_date"),
            )
            return IntentClassification(
                intent="cost_summary",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["cost"],
                entities=date_entities,
                suggested_workflow="cost_summary",
                suggested_agent=None,
            )

        # Simple general cost query (e.g., "how much am I spending")
        # This is the fallback for cost queries without domain, complexity, or dates
        return IntentClassification(
            intent="cost_summary",
            category=IntentCategory.QUERY,
            confidence=0.90,
            domains=["cost"],
            entities={},
            suggested_workflow="cost_summary",
            suggested_agent=None,
        )

    def _get_agent_domain_mapping(self) -> str:
        """Build a string describing which agent handles which domain/tools.

        This helps the LLM make better routing decisions by understanding
        agent specializations and tool capabilities.
        """
        # Define agent capabilities with their tool domains
        agent_capabilities = {
            "db-troubleshoot-agent": {
                "domains": ["database", "opsi"],
                "tools": ["oci_database_*", "oci_opsi_*", "execute_sql"],
                "description": "Database troubleshooting, performance analysis, SQL execution",
            },
            "log-analytics-agent": {
                "domains": ["observability", "logan", "logs"],
                "tools": ["oci_logan_*", "oci_observability_*"],
                "description": "Log search, pattern detection, cross-service correlation",
            },
            "security-threat-agent": {
                "domains": ["security", "iam", "cloudguard"],
                "tools": ["oci_security_*", "list_cloud_guard_problems"],
                "description": "Security threats, IAM analysis, MITRE ATT&CK mapping",
            },
            "finops-agent": {
                "domains": ["cost", "budget", "spending"],
                "tools": ["oci_cost_*"],
                "description": "Cost analysis, anomaly detection, optimization recommendations",
            },
            "infrastructure-agent": {
                "domains": ["compute", "network", "storage", "vcn"],
                "tools": ["oci_compute_*", "oci_network_*"],
                "description": "Compute instances, VCNs, subnets, lifecycle management",
            },
        }

        # Format for prompt
        lines = []
        for agent_role, caps in agent_capabilities.items():
            domains = ", ".join(caps["domains"])
            lines.append(f"- {agent_role}: {caps['description']} (domains: {domains})")

        return "\n".join(lines)

    def _pre_classify_database_query(self, query: str) -> IntentClassification | None:
        """
        Pre-classify database listing/info queries using keyword matching.

        This catches queries like:
        - "show me the database names" → list_databases
        - "list all databases" → list_databases
        - "what databases do I have" → list_databases
        - "database inventory" → list_databases

        Returns None if not a database listing query.
        """
        query_lower = query.lower()

        # Database listing patterns - these should route to list_databases workflow
        listing_keywords = [
            "list", "show", "get", "what", "which", "display",
            "inventory", "names", "all"
        ]
        database_keywords = [
            "database", "databases", "db", "dbs",
            "autonomous", "atp", "adw", "exadata"
        ]

        # Check for database listing intent
        has_listing_keyword = any(kw in query_lower for kw in listing_keywords)
        has_database_keyword = any(kw in query_lower for kw in database_keywords)

        # Must have both a listing action and database reference
        if has_listing_keyword and has_database_keyword:
            # Make sure this isn't a cost query (handled separately)
            cost_keywords = ["cost", "spend", "spending", "price", "billing", "expensive"]
            if any(kw in query_lower for kw in cost_keywords):
                return None  # Let cost pre-classification handle it

            # Make sure this isn't a performance/troubleshooting query
            perf_keywords = ["performance", "slow", "error", "problem", "issue", "troubleshoot"]
            if any(kw in query_lower for kw in perf_keywords):
                return None  # Let LLM route to appropriate agent

            return IntentClassification(
                intent="list_databases",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["database"],
                entities={},
                suggested_workflow="list_databases",
                suggested_agent=None,  # Workflow handles this
            )

        return None

    def _pre_classify_resource_cost_query(self, query: str) -> IntentClassification | None:
        """
        Pre-classify resource-cost mapping queries.

        These are queries that want to see costs along with the resources:
        - "What's costing me the most and what resources are behind it?"
        - "Show me compute resources and their costs"
        - "Cost breakdown with resource details"
        - "Which compartments are spending the most?"
        - "How can I optimize my costs?"

        Returns None if not a resource-cost query.
        """
        query_lower = query.lower()

        # Cost keywords
        cost_keywords = [
            "cost", "spend", "spending", "expensive", "price", "billing",
            "budget", "money", "paying"
        ]
        has_cost = any(kw in query_lower for kw in cost_keywords)

        # Resource/mapping keywords that indicate wanting both cost + resources
        resource_mapping_keywords = [
            "resource", "resources", "what", "which", "behind", "causing",
            "breakdown", "detail", "with", "and", "along", "overview",
            "full", "complete"
        ]
        has_mapping = any(kw in query_lower for kw in resource_mapping_keywords)

        # Optimization keywords
        optimize_keywords = [
            "optimize", "optimization", "reduce", "save", "saving",
            "underutilized", "unused", "waste", "wasting"
        ]
        has_optimize = any(kw in query_lower for kw in optimize_keywords)

        # Compartment-specific patterns
        compartment_keywords = ["compartment", "compartments", "by compartment"]
        has_compartment = any(kw in query_lower for kw in compartment_keywords)

        # Pattern 1: Compartment cost breakdown
        if has_cost and has_compartment:
            return IntentClassification(
                intent="compartment_cost_breakdown",
                category=IntentCategory.QUERY,
                confidence=0.90,
                domains=["cost", "identity"],
                entities={},
                suggested_workflow="compartment_cost_breakdown",
                suggested_agent=None,
            )

        # Pattern 2: Cost optimization / underutilized resources
        if has_optimize or (has_cost and "utiliz" in query_lower):
            return IntentClassification(
                intent="resource_utilization",
                category=IntentCategory.ANALYSIS,
                confidence=0.90,
                domains=["cost", "infrastructure"],
                entities={},
                suggested_workflow="resource_utilization",
                suggested_agent=None,
            )

        # Pattern 3: Full cost overview with resources
        # Matches: "what's costing me", "cost overview", "full breakdown"
        full_overview_patterns = [
            "what's costing", "whats costing", "what is costing",
            "costing me the most", "full breakdown", "cost overview",
            "spending overview", "full cost", "complete cost"
        ]
        if any(pattern in query_lower for pattern in full_overview_patterns):
            return IntentClassification(
                intent="resource_cost_overview",
                category=IntentCategory.QUERY,
                confidence=0.92,
                domains=["cost", "infrastructure"],
                entities={},
                suggested_workflow="resource_cost_overview",
                suggested_agent=None,
            )

        # Pattern 4: Resource type with costs (compute, database)
        if has_cost and has_mapping:
            # Check for compute-specific
            if any(kw in query_lower for kw in ["compute", "instance", "vm", "server"]):
                return IntentClassification(
                    intent="compute_with_costs",
                    category=IntentCategory.QUERY,
                    confidence=0.90,
                    domains=["compute", "cost"],
                    entities={},
                    suggested_workflow="compute_with_costs",
                    suggested_agent=None,
                )

            # Check for database-specific
            if any(kw in query_lower for kw in ["database", "db", "autonomous"]):
                return IntentClassification(
                    intent="database_with_costs",
                    category=IntentCategory.QUERY,
                    confidence=0.90,
                    domains=["database", "cost"],
                    entities={},
                    suggested_workflow="database_with_costs",
                    suggested_agent=None,
                )

        return None

    def _pre_classify_dbmgmt_query(self, query: str) -> IntentClassification | None:
        """
        Pre-classify Database Management queries using keyword matching.

        Catches queries like:
        - "show fleet health" → db_fleet_health
        - "generate AWR report" → awr_report
        - "top SQL by CPU" → top_sql
        - "show wait events" → wait_events
        - "list SQL plan baselines" → sql_plan_baselines
        - "list managed databases" → managed_databases
        """
        query_lower = query.lower()

        # Fleet health patterns
        fleet_keywords = ["fleet health", "fleet status", "database fleet", "managed database health", "all db health"]
        if any(kw in query_lower for kw in fleet_keywords):
            return IntentClassification(
                intent="db_fleet_health",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["dbmgmt", "database"],
                entities={},
                suggested_workflow="db_fleet_health",
                suggested_agent=None,
            )

        # AWR/ASH report patterns
        awr_keywords = ["awr report", "awr", "ash report", "performance report", "workload report"]
        if any(kw in query_lower for kw in awr_keywords):
            return IntentClassification(
                intent="awr_report",
                category=IntentCategory.ANALYSIS,
                confidence=0.95,
                domains=["dbmgmt", "database"],
                entities={},
                suggested_workflow="awr_report",
                suggested_agent=None,
            )

        # Top SQL / SQL performance patterns
        sql_perf_keywords = ["top sql", "expensive queries", "sql cpu", "high cpu sql", "slow queries", "sql performance"]
        if any(kw in query_lower for kw in sql_perf_keywords):
            return IntentClassification(
                intent="top_sql",
                category=IntentCategory.ANALYSIS,
                confidence=0.95,
                domains=["dbmgmt", "database"],
                entities={},
                suggested_workflow="top_sql",
                suggested_agent=None,
            )

        # Wait events patterns
        wait_keywords = ["wait event", "wait events", "database waits", "performance bottleneck"]
        if any(kw in query_lower for kw in wait_keywords):
            return IntentClassification(
                intent="wait_events",
                category=IntentCategory.ANALYSIS,
                confidence=0.95,
                domains=["dbmgmt", "database"],
                entities={},
                suggested_workflow="wait_events",
                suggested_agent=None,
            )

        # SQL Plan Baselines patterns
        baseline_keywords = ["sql plan baseline", "plan baseline", "execution plan", "plan stability"]
        if any(kw in query_lower for kw in baseline_keywords):
            return IntentClassification(
                intent="sql_plan_baselines",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["dbmgmt", "database"],
                entities={},
                suggested_workflow="sql_plan_baselines",
                suggested_agent=None,
            )

        # Managed databases listing
        managed_db_keywords = ["managed database", "dbmgmt database", "db management"]
        if any(kw in query_lower for kw in managed_db_keywords):
            return IntentClassification(
                intent="managed_databases",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["dbmgmt", "database"],
                entities={},
                suggested_workflow="managed_databases",
                suggested_agent=None,
            )

        # DB performance overview (combined analysis)
        perf_overview_keywords = ["database performance", "db performance", "comprehensive db", "full db analysis"]
        if any(kw in query_lower for kw in perf_overview_keywords):
            return IntentClassification(
                intent="db_performance_overview",
                category=IntentCategory.ANALYSIS,
                confidence=0.95,
                domains=["dbmgmt", "opsi", "database"],
                entities={},
                suggested_workflow="db_performance_overview",
                suggested_agent=None,
            )

        return None

    def _pre_classify_opsi_query(self, query: str) -> IntentClassification | None:
        """
        Pre-classify Operations Insights (OPSI) queries using keyword matching.

        Catches queries like:
        - "show ADDM findings" → addm_findings
        - "ADDM recommendations" → addm_recommendations
        - "capacity forecast" → capacity_forecast
        - "database insights" → database_insights
        - "SQL statistics" → sql_statistics
        """
        query_lower = query.lower()

        # ADDM findings patterns
        addm_findings_keywords = ["addm finding", "addm issue", "database diagnostic", "db issue", "performance finding"]
        if any(kw in query_lower for kw in addm_findings_keywords):
            return IntentClassification(
                intent="addm_findings",
                category=IntentCategory.ANALYSIS,
                confidence=0.95,
                domains=["opsi", "database"],
                entities={},
                suggested_workflow="addm_findings",
                suggested_agent=None,
            )

        # ADDM recommendations patterns
        addm_rec_keywords = ["addm recommend", "optimization suggest", "performance recommend", "db recommend"]
        if any(kw in query_lower for kw in addm_rec_keywords):
            return IntentClassification(
                intent="addm_recommendations",
                category=IntentCategory.ANALYSIS,
                confidence=0.95,
                domains=["opsi", "database"],
                entities={},
                suggested_workflow="addm_recommendations",
                suggested_agent=None,
            )

        # Capacity forecast patterns
        forecast_keywords = ["capacity forecast", "growth forecast", "usage projection", "resource forecast"]
        if any(kw in query_lower for kw in forecast_keywords):
            return IntentClassification(
                intent="capacity_forecast",
                category=IntentCategory.ANALYSIS,
                confidence=0.95,
                domains=["opsi", "database"],
                entities={},
                suggested_workflow="capacity_forecast",
                suggested_agent=None,
            )

        # Capacity trend patterns
        trend_keywords = ["capacity trend", "usage history", "utilization trend", "resource trend"]
        if any(kw in query_lower for kw in trend_keywords):
            return IntentClassification(
                intent="capacity_trend",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["opsi", "database"],
                entities={},
                suggested_workflow="capacity_trend",
                suggested_agent=None,
            )

        # Database insights patterns
        insights_keywords = ["database insight", "opsi database", "monitored database", "operations insight"]
        if any(kw in query_lower for kw in insights_keywords):
            return IntentClassification(
                intent="database_insights",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["opsi", "database"],
                entities={},
                suggested_workflow="database_insights",
                suggested_agent=None,
            )

        # SQL insights/statistics patterns
        sql_insights_keywords = ["sql insight", "sql statistic", "sql metric", "query statistic"]
        if any(kw in query_lower for kw in sql_insights_keywords):
            return IntentClassification(
                intent="sql_insights",
                category=IntentCategory.ANALYSIS,
                confidence=0.95,
                domains=["opsi", "database"],
                entities={},
                suggested_workflow="sql_insights",
                suggested_agent=None,
            )

        # Resource utilization via OPSI
        opsi_util_keywords = ["opsi utilization", "database utilization", "db resource usage", "db usage metric"]
        if any(kw in query_lower for kw in opsi_util_keywords):
            return IntentClassification(
                intent="opsi_utilization",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["opsi", "database"],
                entities={},
                suggested_workflow="opsi_utilization",
                suggested_agent=None,
            )

        return None

    def _pre_classify_identity_query(self, query: str) -> IntentClassification | None:
        """
        Pre-classify identity/IAM queries using keyword matching.

        Catches queries like:
        - "list compartments" → list_compartments
        - "show compartments" → list_compartments
        - "tenancy info" → get_tenancy
        - "list regions" → list_regions

        This avoids LLM classification for common identity queries,
        reducing response time by 1-3 seconds.
        """
        query_lower = query.lower()

        # Compartment listing patterns
        listing_keywords = ["list", "show", "get", "what", "which", "display", "all"]
        compartment_keywords = ["compartment", "compartments"]

        has_listing = any(kw in query_lower for kw in listing_keywords)
        has_compartment = any(kw in query_lower for kw in compartment_keywords)

        if has_listing and has_compartment:
            return IntentClassification(
                intent="list_compartments",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["identity"],
                entities={},
                suggested_workflow="list_compartments",
                suggested_agent=None,
            )

        # Tenancy info patterns
        tenancy_keywords = ["tenancy", "tenant"]
        info_keywords = ["info", "information", "details", "show", "get", "what"]

        has_tenancy = any(kw in query_lower for kw in tenancy_keywords)
        has_info = any(kw in query_lower for kw in info_keywords)

        if has_tenancy and has_info:
            return IntentClassification(
                intent="get_tenancy",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["identity"],
                entities={},
                suggested_workflow="get_tenancy",
                suggested_agent=None,
            )

        # Region listing patterns
        region_keywords = ["region", "regions"]
        if has_listing and any(kw in query_lower for kw in region_keywords):
            return IntentClassification(
                intent="list_regions",
                category=IntentCategory.QUERY,
                confidence=0.95,
                domains=["identity"],
                entities={},
                suggested_workflow="list_regions",
                suggested_agent=None,
            )

        return None

    def _build_classification_prompt(self, query: str) -> str:
        """Build the intent classification prompt."""
        # Get available workflows and agents for context
        available_workflows = list(self.workflow_registry.keys())
        available_agents = [
            agent.role for agent in self.agent_catalog.list_all()
        ]

        # Build agent-to-domain mapping for better routing decisions
        agent_domains = self._get_agent_domain_mapping()

        return f"""Classify the following user query for an OCI (Oracle Cloud Infrastructure) management system.

Query: "{query}"

Available Workflows: {available_workflows}

Agent Capabilities (use suggested_agent based on query domain):
{agent_domains}

IMPORTANT: For cost-related queries:
- If query mentions specific services (database, DB, compute, storage, network), use domain-specific workflows:
  - "database_costs" or "db_costs" for database/DB cost queries
  - "compute_costs" for compute/instance cost queries
  - "storage_costs" for storage cost queries
- Only use "cost_summary" or "show_spending" for general tenancy-wide cost queries
- Extract the domain (database, compute, storage, network) into the domains array

Examples:
- "show me DB costs" → intent: "database_costs", domains: ["database", "cost"]
- "how much am I spending on databases" → intent: "database_costs", domains: ["database", "cost"]
- "what's my total spending" → intent: "cost_summary", domains: ["cost"]
- "tenancy costs" → intent: "cost_summary", domains: ["cost"]

Respond with a JSON object:
{{
    "intent": "<specific intent like list_instances, database_costs, compute_costs>",
    "category": "<query|action|analysis|troubleshoot|unknown>",
    "confidence": <0.0 to 1.0>,
    "domains": ["<domain1>", "<domain2>"],
    "entities": {{"entity_name": "value"}},
    "suggested_workflow": "<workflow name or null>",
    "suggested_agent": "<agent role or null>"
}}

Categories:
- query: Information retrieval (list, get, describe, show costs)
- action: Perform operation (start, stop, create, delete)
- analysis: Complex analysis (deep cost analysis, performance review)
- troubleshoot: Diagnose issues (why is X slow, fix Y)
- unknown: Cannot determine

Domains: compute, network, database, dbmgmt, opsi, security, cost, observability, storage

IMPORTANT: For database management/performance queries:
- "fleet health" → intent: "db_fleet_health", domains: ["dbmgmt", "database"]
- "AWR report" → intent: "awr_report", domains: ["dbmgmt", "database"]
- "top SQL" → intent: "top_sql", domains: ["dbmgmt", "database"]
- "wait events" → intent: "wait_events", domains: ["dbmgmt", "database"]
- "ADDM findings" → intent: "addm_findings", domains: ["opsi", "database"]
- "capacity forecast" → intent: "capacity_forecast", domains: ["opsi", "database"]

Return only the JSON object, no other text."""

    def _parse_classification(
        self, response: str, query: str
    ) -> IntentClassification:
        """Parse LLM response into IntentClassification."""
        import json

        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(
                    line for line in lines if not line.startswith("```")
                )

            data = json.loads(response)

            return IntentClassification(
                intent=data.get("intent", "unknown"),
                category=IntentCategory(data.get("category", "unknown")),
                confidence=float(data.get("confidence", 0.5)),
                domains=data.get("domains", []),
                entities=data.get("entities", {}),
                suggested_workflow=data.get("suggested_workflow"),
                suggested_agent=data.get("suggested_agent"),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._logger.warning(
                "Failed to parse classification",
                error=str(e),
                response=response[:200],
            )
            # Fallback classification
            return IntentClassification(
                intent="unknown",
                category=IntentCategory.UNKNOWN,
                confidence=0.4,
                domains=[],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Router Node
    # ─────────────────────────────────────────────────────────────────────────

    async def router_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Determine routing based on intent classification.

        UPDATED Routing Strategy:
        - Complex queries (temporal, analytical) → AGENT (with LLM reasoning)
        - Simple queries + very high confidence → WORKFLOW
        - Multi-domain → PARALLEL
        - Medium confidence + agent match → AGENT
        - Low confidence → ESCALATE
        - Otherwise → DIRECT

        Now includes agent discovery for transparency.

        Args:
            state: Current coordinator state

        Returns:
            State updates with routing decision and agent candidates
        """
        with _tracer.start_as_current_span("coordinator.router") as span:
            thinking_trace = state.thinking_trace

            if not state.intent:
                span.set_attribute("error", "no_intent")
                self._logger.warning("No intent for routing")
                return {
                    "error": "No intent classification available",
                }

            # Add thinking step for routing start
            if thinking_trace:
                thinking_trace.add_step(
                    ThinkingPhase.ROUTING,
                    "Determining best route for this request...",
                    {"domains": state.intent.domains},
                )

            # Discover matching agents for transparency
            agent_candidates = self._discover_agents(state.intent, state.query)

            if thinking_trace and agent_candidates:
                thinking_trace.add_step(
                    ThinkingPhase.DISCOVERED,
                    f"Found {len(agent_candidates)} potential agents",
                    {
                        "candidates": [
                            {"role": c.agent_role, "confidence": c.confidence}
                            for c in agent_candidates[:3]
                        ]
                    },
                )

            # Pass original query for complexity detection
            routing = determine_routing(state.intent, original_query=state.query)

            span.set_attribute("routing.type", routing.routing_type.value)
            span.set_attribute("routing.target", routing.target or "none")
            span.set_attribute("routing.confidence", routing.confidence)

            self._logger.info(
                "Routing decision",
                routing_type=routing.routing_type.value,
                target=routing.target,
                confidence=routing.confidence,
                reasoning=routing.reasoning,
            )

            # Mark selected agent in candidates
            if routing.target:
                for candidate in agent_candidates:
                    if candidate.agent_role == routing.target:
                        candidate.selected = True
                if thinking_trace:
                    thinking_trace.select_agent(routing.target)

            # Add thinking step for routing decision
            if thinking_trace:
                route_msg = self._format_routing_message(routing)
                thinking_trace.add_step(
                    ThinkingPhase.ROUTED,
                    route_msg,
                    {
                        "routing_type": routing.routing_type.value,
                        "target": routing.target,
                        "confidence": routing.confidence,
                        "reasoning": routing.reasoning,
                    },
                )

            # Prepare agent context if routing to agent
            agent_context = None
            if routing.routing_type == RoutingType.AGENT:
                agent_context = AgentContext(
                    query=state.query,
                    intent=state.intent,
                    previous_results=state.tool_results,
                )

            return {
                "routing": routing,
                "agent_context": agent_context,
                "current_agent": routing.target if routing.routing_type == RoutingType.AGENT else None,
                "workflow_name": routing.target if routing.routing_type == RoutingType.WORKFLOW else None,
                "agent_candidates": agent_candidates,
            }

    def _discover_agents(
        self,
        intent: IntentClassification,
        query: str,
    ) -> list[AgentCandidate]:
        """
        Discover and rank agents that could handle this request.

        Args:
            intent: Classified intent
            query: Original user query

        Returns:
            List of AgentCandidate sorted by confidence
        """
        candidates = []

        # Get all agents from catalog
        all_agents = self.agent_catalog.list_all()

        for agent_def in all_agents:
            match_reasons = []
            base_confidence = 0.0

            # Check domain match
            agent_domains = self._get_agent_domains(agent_def.role)
            domain_overlap = set(intent.domains) & set(agent_domains)
            if domain_overlap:
                match_reasons.append(f"Domain match: {', '.join(domain_overlap)}")
                base_confidence += 0.4 * (len(domain_overlap) / max(len(intent.domains), 1))

            # Check capability match
            for capability in agent_def.capabilities:
                cap_lower = capability.lower()
                if intent.intent and intent.intent.lower() in cap_lower:
                    match_reasons.append(f"Capability: {capability}")
                    base_confidence += 0.3
                    break

            # Check if this is the suggested agent
            if agent_def.role == intent.suggested_agent:
                match_reasons.append("Suggested by classifier")
                base_confidence += 0.3

            # Only include if there's some match
            if match_reasons and base_confidence > 0:
                candidates.append(
                    AgentCandidate(
                        agent_id=agent_def.agent_id,
                        agent_role=agent_def.role,
                        confidence=min(base_confidence, 1.0),
                        capabilities=agent_def.capabilities[:3],
                        match_reasons=match_reasons,
                        selected=False,
                        mcp_servers=agent_def.mcp_servers[:3] if agent_def.mcp_servers else [],
                    )
                )

        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        return candidates[:5]  # Top 5 candidates

    def _get_agent_domains(self, agent_role: str) -> list[str]:
        """Get domains associated with an agent role."""
        domain_map = {
            "db-troubleshoot-agent": ["database", "opsi", "sql"],
            "log-analytics-agent": ["observability", "logs", "logan"],
            "security-threat-agent": ["security", "iam", "cloudguard"],
            "finops-agent": ["cost", "budget", "spending", "finops"],
            "infrastructure-agent": ["compute", "network", "storage", "vcn"],
            "error-analysis-agent": ["logs", "errors", "observability"],
        }
        return domain_map.get(agent_role, [])

    def _format_routing_message(self, routing) -> str:
        """Format a human-readable routing message."""
        if routing.routing_type == RoutingType.WORKFLOW:
            workflow_name = routing.target.replace("_", " ").title() if routing.target else "Unknown"
            return f"Using workflow: {workflow_name}"
        elif routing.routing_type == RoutingType.AGENT:
            agent_name = routing.target.replace("-agent", "").replace("-", " ").title() if routing.target else "Unknown"
            return f"Delegating to agent: {agent_name}"
        elif routing.routing_type == RoutingType.PARALLEL:
            return "Using parallel multi-agent execution"
        elif routing.routing_type == RoutingType.ESCALATE:
            return "Escalating to human"
        else:
            return "Direct LLM response"

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow Node
    # ─────────────────────────────────────────────────────────────────────────

    async def workflow_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Execute deterministic workflow.

        Workflows are pre-defined sequences of operations that don't
        require LLM reasoning - they execute deterministically.

        Args:
            state: Current coordinator state

        Returns:
            State updates with workflow result
        """
        with _tracer.start_as_current_span("coordinator.workflow") as span:
            workflow_name = state.workflow_name
            span.set_attribute("workflow.name", workflow_name or "none")
            thinking_trace = state.thinking_trace

            if not workflow_name:
                span.set_attribute("error", "no_workflow")
                return {"error": "No workflow specified"}

            workflow = self.workflow_registry.get(workflow_name)
            if not workflow:
                span.set_attribute("error", "workflow_not_found")
                self._logger.warning("Workflow not found", workflow=workflow_name)
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.ERROR,
                        f"Workflow '{workflow_name}' not found, trying fallback...",
                    )
                # Fallback to agentic
                return {
                    "routing": replace(state.routing, routing_type=RoutingType.AGENT)
                    if state.routing
                    else None,
                    "error": f"Workflow '{workflow_name}' not found",
                }

            self._logger.info("Executing workflow", workflow=workflow_name)

            # Add thinking step for workflow execution
            if thinking_trace:
                thinking_trace.add_step(
                    ThinkingPhase.EXECUTING,
                    f"Executing workflow: {workflow_name.replace('_', ' ').title()}",
                    {"workflow": workflow_name},
                )

            try:
                start_time = time.time()
                result = await workflow(
                    query=state.query,
                    entities=state.intent.entities if state.intent else {},
                    tool_catalog=self.tool_catalog,
                    memory=self.memory,
                )
                duration_ms = int((time.time() - start_time) * 1000)

                span.set_attribute("workflow.duration_ms", duration_ms)
                span.set_attribute("workflow.success", True)

                self._logger.info(
                    "Workflow completed",
                    workflow=workflow_name,
                    duration_ms=duration_ms,
                )

                # Add thinking step for completion
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.COMPLETE,
                        f"Workflow completed in {duration_ms}ms",
                        {"duration_ms": duration_ms, "success": True},
                        duration_ms=duration_ms,
                    )

                return {
                    "final_response": result,
                    "workflow_state": {"completed": True, "duration_ms": duration_ms},
                }

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e)[:200])
                self._logger.error(
                    "Workflow failed",
                    workflow=workflow_name,
                    error=str(e),
                )
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.ERROR,
                        f"Workflow failed: {str(e)[:100]}",
                        {"error": str(e)},
                    )
                # Try fallback if available
                if state.routing and state.routing.fallback:
                    return {
                        "routing": state.routing.fallback,
                        "workflow_name": None,
                        "current_agent": state.routing.fallback.target,
                    }
                return {"error": f"Workflow failed: {e}"}

    # ─────────────────────────────────────────────────────────────────────────
    # Parallel Node
    # ─────────────────────────────────────────────────────────────────────────

    async def parallel_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Execute multi-domain query using parallel orchestrator.

        Routes complex cross-domain requests (2+ domains) to the parallel
        orchestrator which decomposes the task and runs multiple agents
        concurrently for optimal performance.

        Args:
            state: Current coordinator state

        Returns:
            State updates with synthesized response from multiple agents
        """
        with _tracer.start_as_current_span("coordinator.parallel") as span:
            span.set_attribute("query_preview", state.query[:100] if state.query else "")
            domains = state.intent.domains if state.intent else []
            span.set_attribute("domains", ",".join(domains))

            if not self.orchestrator:
                span.set_attribute("error", "orchestrator_not_initialized")
                self._logger.warning("Parallel orchestrator not available")
                # Fallback to single agent execution
                return {
                    "routing": state.routing._replace(routing_type=RoutingType.AGENT)
                    if state.routing and hasattr(state.routing, "_replace")
                    else state.routing,
                    "error": "Parallel orchestrator not available, falling back to agent",
                }

            self._logger.info(
                "Executing parallel orchestration",
                domains=domains,
                category=state.intent.category.value if state.intent else "unknown",
            )

            start_time = time.time()
            try:
                # Build context for orchestrator
                context = {
                    "intent": state.intent.to_dict() if state.intent else {},
                    "domains": domains,
                    "previous_results": [r.to_dict() for r in state.tool_results],
                    "output_format": state.output_format,
                    "channel_type": state.channel_type,
                }

                # Execute parallel orchestration
                result = await self.orchestrator.execute(
                    query=state.query,
                    context=context,
                )

                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("parallel.duration_ms", duration_ms)
                span.set_attribute("parallel.success", result.success)
                span.set_attribute("parallel.agents_used", ",".join(result.agents_used))
                span.set_attribute("parallel.total_tool_calls", result.total_tool_calls)

                self._logger.info(
                    "Parallel orchestration completed",
                    success=result.success,
                    agents_used=result.agents_used,
                    duration_ms=duration_ms,
                    total_tool_calls=result.total_tool_calls,
                )

                if result.success:
                    return {
                        "final_response": result.response,
                        "agent_response": result.response,
                        "workflow_state": {
                            "parallel": True,
                            "agents_used": result.agents_used,
                            "duration_ms": duration_ms,
                            "total_tool_calls": result.total_tool_calls,
                        },
                    }
                else:
                    # Parallel execution failed, try fallback
                    if state.routing and state.routing.fallback:
                        self._logger.info(
                            "Parallel failed, using fallback",
                            fallback=state.routing.fallback.routing_type.value,
                        )
                        return {
                            "routing": state.routing.fallback,
                            "current_agent": state.routing.fallback.target,
                            "error": result.error,
                        }
                    return {
                        "error": result.error or "Parallel orchestration failed",
                        "final_response": f"Error: {result.error or 'Parallel execution failed'}",
                    }

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e)[:200])

                self._logger.error(
                    "Parallel orchestration failed",
                    error=str(e),
                    duration_ms=duration_ms,
                )

                # Try fallback if available
                if state.routing and state.routing.fallback:
                    return {
                        "routing": state.routing.fallback,
                        "current_agent": state.routing.fallback.target,
                    }
                return {"error": f"Parallel execution failed: {e}"}

    # ─────────────────────────────────────────────────────────────────────────
    # Agent Node
    # ─────────────────────────────────────────────────────────────────────────

    async def agent_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Invoke LLM for reasoning or delegate to specialized agent.

        Args:
            state: Current coordinator state

        Returns:
            State updates with agent response
        """
        with _tracer.start_as_current_span("coordinator.agent") as span:
            span.set_attribute("iteration", state.iteration)
            span.set_attribute("current_agent", state.current_agent or "coordinator-llm")
            thinking_trace = state.thinking_trace

            self._logger.debug(
                "Agent node",
                iteration=state.iteration,
                current_agent=state.current_agent,
            )

            # If delegating to specialized agent
            if state.current_agent:
                span.set_attribute("delegation", "specialized_agent")
                agent_name = state.current_agent.replace("-agent", "").replace("-", " ").title()
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.DELEGATING,
                        f"Delegating to {agent_name} agent...",
                        {"agent": state.current_agent},
                    )
                return await self._invoke_specialized_agent(state)

            # Otherwise, use coordinator LLM
            span.set_attribute("delegation", "coordinator_llm")
            if thinking_trace:
                thinking_trace.add_step(
                    ThinkingPhase.EXECUTING,
                    "Using coordinator LLM for response...",
                    {"method": "coordinator_llm"},
                )
            return await self._invoke_coordinator_llm(state)

    async def _invoke_specialized_agent(
        self, state: CoordinatorState
    ) -> dict[str, Any]:
        """Delegate to a specialized agent."""
        agent_role = state.current_agent

        if not agent_role:
            return {"error": "No agent specified"}

        with _tracer.start_as_current_span(f"agent.invoke.{agent_role}") as span:
            span.set_attribute("agent.role", agent_role)

            # Get agent from catalog with output format config
            agent_config = {
                "output_format": state.output_format,
                "channel_type": state.channel_type,
            }

            agent_instance = self.agent_catalog.instantiate(
                role=agent_role,
                memory_manager=self.memory,
                tool_catalog=self.tool_catalog,
                config=agent_config,
                llm=self.llm,
            )

            if not agent_instance:
                span.set_attribute("error", "agent_not_found")
                self._logger.warning("Agent not found", role=agent_role)
                return {"error": f"Agent '{agent_role}' not available"}

            # Get agent timeout from definition (default 120s, max 300s)
            # Higher cap needed for cross-region LLM providers (e.g., US-EMEA OCA latency)
            agent_def = agent_instance.get_definition()
            timeout_seconds = min(
                agent_def.metadata.timeout_seconds if agent_def.metadata else 120,
                300  # Cap allows for high-latency LLM providers (OCA cross-region ~15-20s/call)
            )
            span.set_attribute("agent.timeout_seconds", timeout_seconds)

            self._logger.info(
                "Delegating to agent",
                agent_role=agent_role,
                output_format=state.output_format,
                timeout_seconds=timeout_seconds,
            )

            start_time = time.time()
            try:
                context = state.agent_context.to_dict() if state.agent_context else {}
                context["output_format"] = state.output_format
                context["channel_type"] = state.channel_type

                # Wrap agent invocation with timeout
                result = await asyncio.wait_for(
                    agent_instance.invoke(state.query, context),
                    timeout=timeout_seconds,
                )
                duration_ms = int((time.time() - start_time) * 1000)

                span.set_attribute("agent.duration_ms", duration_ms)
                span.set_attribute("agent.success", True)

                # Record metrics
                self.agent_catalog.record_invocation(
                    role=agent_role,
                    duration_ms=duration_ms,
                    success=True,
                )

                self._logger.info(
                    "Agent completed",
                    agent_role=agent_role,
                    duration_ms=duration_ms,
                )

                # Add thinking step for agent completion
                thinking_trace = state.thinking_trace
                if thinking_trace:
                    agent_name = agent_role.replace("-agent", "").replace("-", " ").title()
                    thinking_trace.add_step(
                        ThinkingPhase.COMPLETE,
                        f"{agent_name} agent completed in {duration_ms}ms",
                        {"agent": agent_role, "duration_ms": duration_ms, "success": True},
                        duration_ms=duration_ms,
                    )

                return {
                    "agent_response": result,
                    "final_response": result,
                }

            except asyncio.TimeoutError:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("error", "timeout")
                span.set_attribute("agent.duration_ms", duration_ms)

                # Record timeout as failed invocation
                self.agent_catalog.record_invocation(
                    role=agent_role,
                    duration_ms=duration_ms,
                    success=False,
                )

                self._logger.error(
                    "Agent timed out",
                    agent_role=agent_role,
                    timeout_seconds=timeout_seconds,
                    duration_ms=duration_ms,
                )

                # Add thinking step for timeout
                thinking_trace = state.thinking_trace
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.ERROR,
                        f"Agent timed out after {timeout_seconds}s",
                        {"agent": agent_role, "timeout_seconds": timeout_seconds},
                    )

                # Provide agent-specific timeout guidance
                agent_name = agent_role.replace("-agent", "").replace("-", " ").title()
                timeout_guidance = {
                    "finops": "Try asking about a shorter time period (e.g., 'last 7 days' instead of 'last month').",
                    "db-troubleshoot": "Try asking about a specific database or a simpler query.",
                    "log-analytics": "Try narrowing the time range or specifying a specific log source.",
                    "infrastructure": "Try asking about specific resources instead of listing all.",
                }
                guidance = timeout_guidance.get(agent_role.replace("-agent", ""), "Try a more specific query.")

                return {
                    "error": f"Agent '{agent_role}' timed out after {timeout_seconds}s",
                    "final_response": (
                        f"The {agent_name} Agent took longer than expected (>{timeout_seconds}s). "
                        f"This usually happens with complex queries or when OCI APIs are slow.\n\n"
                        f"**Suggestion:** {guidance}\n\n"
                        f"_You can also try again - sometimes API response times vary._"
                    ),
                }

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e)[:200])
                span.set_attribute("agent.duration_ms", duration_ms)

                # Record failed invocation
                self.agent_catalog.record_invocation(
                    role=agent_role,
                    duration_ms=duration_ms,
                    success=False,
                )

                self._logger.error(
                    "Agent failed",
                    agent_role=agent_role,
                    error=str(e),
                )

                # Add thinking step for error
                thinking_trace = state.thinking_trace
                if thinking_trace:
                    thinking_trace.add_step(
                        ThinkingPhase.ERROR,
                        f"Agent failed: {str(e)[:100]}",
                        {"agent": agent_role, "error": str(e)},
                    )

                return {"error": f"Agent failed: {e}"}

    async def _invoke_coordinator_llm(
        self, state: CoordinatorState
    ) -> dict[str, Any]:
        """Invoke the coordinator's LLM for reasoning."""
        # Build messages for LLM
        messages = list(state.messages)

        # RAG context injection - DISABLED
        # To re-enable: set RAG_ENABLED=true in environment
        # if os.getenv("RAG_ENABLED", "false").lower() == "true" and state.query:
        #     try:
        #         from src.rag import get_retriever
        #
        #         namespace = os.getenv("RAG_NAMESPACE", "oci-docs")
        #         retriever = await get_retriever(namespace)
        #         result = await retriever.retrieve(state.query)
        #         if result.context:
        #             messages.insert(
        #                 0,
        #                 SystemMessage(
        #                     content=(
        #                         "Relevant documentation context:\n"
        #                         f"{result.context}"
        #                     )
        #                 ),
        #             )
        #     except Exception as e:
        #         self._logger.warning("RAG retrieval failed", error=str(e))

        # Runtime feedback injection
        try:
            feedback_text = await self.memory.get_feedback_text()
            if feedback_text:
                messages.insert(
                    0,
                    SystemMessage(
                        content=f"Runtime feedback directives:\n{feedback_text}"
                    ),
                )
        except Exception as e:
            self._logger.warning("Feedback retrieval failed", error=str(e))

        # Add tool results as messages
        for result in state.tool_results:
            messages.append(
                ToolMessage(
                    content=str(result.result),
                    tool_call_id=result.tool_call_id,
                )
            )

        try:
            response = await self.llm.ainvoke(messages)

            # Log response details for debugging
            response_content = response.content if hasattr(response, "content") else ""
            self._logger.info(
                "LLM response received",
                content_length=len(response_content) if response_content else 0,
                content_preview=response_content[:100] if response_content else "(empty)",
                has_tool_calls=bool(
                    hasattr(response, "tool_calls") and response.tool_calls
                ),
            )

            # Check for empty response
            if not response_content or not response_content.strip():
                self._logger.warning(
                    "LLM returned empty response",
                    llm_type=type(self.llm).__name__,
                    query_preview=state.query[:100] if state.query else "",
                )

            # Extract tool calls if any
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", f"call_{i}"),
                        name=tc.get("name", ""),
                        arguments=tc.get("args", {}),
                    )
                    for i, tc in enumerate(response.tool_calls)
                ]

            return {
                "messages": [response],
                "tool_calls": tool_calls,
                "tool_results": [],  # Clear previous results
                "iteration": state.iteration + 1,
            }

        except Exception as e:
            self._logger.error("LLM invocation failed", error=str(e))
            return {
                "error": str(e),
                "messages": [AIMessage(content=f"Error: {e}")],
                "tool_calls": [],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Action Node
    # ─────────────────────────────────────────────────────────────────────────

    async def action_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Execute pending tool calls.

        Args:
            state: Current coordinator state

        Returns:
            State updates with tool results
        """
        if not state.tool_calls:
            return {"tool_results": []}

        self._logger.info(
            "Executing tools",
            tool_count=len(state.tool_calls),
        )

        results = []
        for tool_call in state.tool_calls:
            start_time = time.time()

            try:
                result = await self._execute_tool(tool_call)
                duration_ms = int((time.time() - start_time) * 1000)

                results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=result,
                        success=True,
                        duration_ms=duration_ms,
                    )
                )

                self._logger.debug(
                    "Tool executed",
                    tool=tool_call.name,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                self._logger.error(
                    "Tool execution failed",
                    tool=tool_call.name,
                    error=str(e),
                )

                results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=None,
                        success=False,
                        duration_ms=duration_ms,
                        error=str(e),
                    )
                )

        return {
            "tool_results": results,
            "tool_calls": [],  # Clear after processing
        }

    async def _execute_tool(self, tool_call: ToolCall) -> Any:
        """Execute a single tool via the tool catalog."""
        if not self.tool_catalog:
            raise RuntimeError("Tool catalog not initialized")

        tool_def = self.tool_catalog.get_tool(tool_call.name)
        if not tool_def:
            raise ValueError(f"Tool not found: {tool_call.name}")

        # Execute through catalog
        return await self.tool_catalog.execute(tool_call.name, tool_call.arguments)

    # ─────────────────────────────────────────────────────────────────────────
    # Output Node
    # ─────────────────────────────────────────────────────────────────────────

    async def output_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Prepare final output with agent/workflow attribution.

        Extracts the final response from state and prepends the selected
        agent or workflow information for transparency.

        Args:
            state: Current coordinator state

        Returns:
            State updates with final response
        """
        self._logger.debug(
            "Output node",
            iterations=state.iteration,
            has_error=bool(state.error),
        )

        # Build agent/workflow attribution header
        attribution = self._build_attribution_header(state)

        # If we already have a final response (and it's not empty), add attribution
        if state.final_response and state.final_response.strip():
            if attribution:
                return {"final_response": f"{attribution}\n\n{state.final_response}"}
            return {}

        # Extract from last AI message
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                content = msg.content if msg.content else ""
                # Check for non-empty, meaningful response
                if content.strip():
                    if attribution:
                        return {"final_response": f"{attribution}\n\n{content}"}
                    return {"final_response": content}
                else:
                    self._logger.warning(
                        "AI message has empty content",
                        message_type=type(msg).__name__,
                    )

        # Fallback for errors
        if state.error:
            error_response = f"Error: {state.error}"
            if attribution:
                return {"final_response": f"{attribution}\n\n{error_response}"}
            return {"final_response": error_response}

        # Enhanced fallback with context about what was attempted
        routing_info = ""
        if state.routing:
            routing_info = f" (routing: {state.routing.routing_type.value})"

        self._logger.warning(
            "No meaningful response from AI",
            iterations=state.iteration,
            routing=routing_info,
            query_preview=state.query[:100] if state.query else "",
        )

        fallback_response = (
            "I processed your request but couldn't generate a complete response. "
            "Please try rephrasing your question or check the system logs for details."
        )
        if attribution:
            return {"final_response": f"{attribution}\n\n{fallback_response}"}
        return {"final_response": fallback_response}

    def _build_attribution_header(self, state: CoordinatorState) -> str:
        """
        Build attribution header showing which agent/workflow was selected.

        Args:
            state: Current coordinator state

        Returns:
            Attribution string like "🤖 Agent: db-troubleshoot-agent" or empty string
        """
        if not state.routing:
            return ""

        routing_type = state.routing.routing_type

        # Workflow attribution
        if routing_type == RoutingType.WORKFLOW and state.workflow_name:
            workflow_display = state.workflow_name.replace("_", " ").title()
            return f"📋 **Workflow:** {workflow_display}"

        # Agent attribution
        if routing_type == RoutingType.AGENT and state.current_agent:
            agent_display = state.current_agent.replace("-agent", "").replace("-", " ").title()
            return f"🤖 **Agent:** {agent_display}"

        # Parallel execution attribution
        if routing_type == RoutingType.PARALLEL:
            if state.workflow_state and "agents_used" in state.workflow_state:
                agents = state.workflow_state["agents_used"]
                if agents:
                    agents_display = ", ".join(
                        a.replace("-agent", "").replace("-", " ").title()
                        for a in agents
                    )
                    return f"🔄 **Parallel Agents:** {agents_display}"
            return "🔄 **Parallel Execution**"

        # Direct LLM or escalate
        if routing_type == RoutingType.DIRECT:
            return "💬 **Coordinator LLM**"

        if routing_type == RoutingType.ESCALATE:
            return "⚠️ **Escalated to Human**"

        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Routing Functions for Conditional Edges
# ─────────────────────────────────────────────────────────────────────────────


def should_continue_after_router(state: CoordinatorState) -> str:
    """
    Determine next node after routing decision.

    Returns:
        Node name: "workflow", "parallel", "agent", or "output"
    """
    if state.error:
        return "output"

    if not state.routing:
        return "agent"  # Default to agentic

    routing_type = state.routing.routing_type

    if routing_type == RoutingType.WORKFLOW:
        return "workflow"
    elif routing_type == RoutingType.PARALLEL:
        return "parallel"  # Multi-domain parallel execution
    elif routing_type == RoutingType.AGENT:
        return "agent"
    elif routing_type == RoutingType.ESCALATE:
        return "output"  # Escalate goes directly to output
    else:
        return "agent"  # DIRECT uses agent node


def should_continue_after_agent(state: CoordinatorState) -> str:
    """
    Determine if we should continue to action or end.

    Returns:
        "action" if tool calls pending, "output" otherwise
    """
    if state.error:
        return "output"

    if state.final_response:
        return "output"

    if state.iteration >= state.max_iterations:
        return "output"

    if state.has_pending_tools():
        return "action"

    return "output"


def should_loop_from_action(state: CoordinatorState) -> str:
    """
    Determine if we should loop back to agent after action.

    Returns:
        "agent" to continue, "output" to end
    """
    if state.error:
        return "output"

    if state.iteration >= state.max_iterations:
        return "output"

    return "agent"
