"""
Infrastructure Agent with OCI MCP Server Integration.

Specialized agent for OCI infrastructure management including
compute, network, and security operations using the mcp-oci MCP server.

Key Features:
- Full OCI compute management (instances, shapes, metrics)
- Network topology analysis (VCNs, subnets, security lists)
- Security operations (IAM, Cloud Guard, policies)
- OpenTelemetry tracing integration
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
from src.agents.self_healing import SelfHealingMixin
from src.observability import get_trace_id

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


class ActionType(str, Enum):
    """Types of infrastructure actions."""

    LIST = "list"
    DESCRIBE = "describe"
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    ANALYZE = "analyze"


@dataclass
class ResourceSummary:
    """Summary of OCI resources."""

    instances: int = 0
    running_instances: int = 0
    vcns: int = 0
    subnets: int = 0
    security_lists: int = 0


@dataclass
class InfrastructureState:
    """State for infrastructure management workflow."""

    query: str = ""
    compartment_id: str | None = None
    resource_id: str | None = None
    action_type: ActionType = ActionType.LIST

    # Resource data from MCP tools
    instances: list[dict] = field(default_factory=list)
    vcns: list[dict] = field(default_factory=list)
    subnets: list[dict] = field(default_factory=list)
    security_lists: list[dict] = field(default_factory=list)
    security_analysis: dict[str, Any] = field(default_factory=dict)

    # Workflow state
    phase: str = "analyze_request"
    iteration: int = 0
    max_iterations: int = 10

    # Results
    action_result: dict[str, Any] = field(default_factory=dict)
    summary: ResourceSummary = field(default_factory=ResourceSummary)
    error: str | None = None
    result: str | None = None


# System prompt for the Infrastructure Agent
INFRASTRUCTURE_SYSTEM_PROMPT = """You are the OCI Infrastructure Agent, a specialized AI expert in Oracle Cloud Infrastructure management.

Your expertise includes:
- Compute instance management (start, stop, restart, metrics)
- Network topology (VCNs, subnets, route tables, security lists)
- Security analysis (IAM, Cloud Guard, policies, compliance)
- Capacity planning and resource optimization

## Available MCP Tools (oci-infrastructure server)

Tool naming convention: `oci_{domain}_{action}_{resource}`

### Discovery Tools (Tier 1 - Instant, <100ms)
- `oci_ping`: Server health check
- `oci_list_domains`: List available capability domains
- `oci_search_tools`: Search for specific tools by keyword
- `oci_get_capabilities`: Get server capability summary

### Compute Tools (Tier 2-4 - API)
- `oci_compute_list_instances`: List compute instances with filtering
- `oci_compute_get_instance`: Get detailed instance information
- `oci_observability_get_instance_metrics`: Get instance CPU, memory, network metrics
- `oci_compute_start_instance`: Start a stopped instance (requires ALLOW_MUTATIONS=true)
- `oci_compute_stop_instance`: Stop a running instance (requires ALLOW_MUTATIONS=true)
- `oci_compute_restart_instance`: Restart an instance (requires ALLOW_MUTATIONS=true)

### Network Tools (Tier 2 - API)
- `oci_network_list_vcns`: List Virtual Cloud Networks
- `oci_network_get_vcn`: Get VCN details with subnet info
- `oci_network_list_subnets`: List subnets in a VCN
- `oci_network_list_security_lists`: List security lists with rules
- `oci_network_analyze_security`: Analyze security configuration (Tier 3)

### Security Tools (Tier 2-3 - API)
- `oci_security_list_users`: List IAM users
- `oci_security_get_user`: Get user details and group memberships
- `oci_security_list_groups`: List IAM groups
- `oci_security_list_policies`: List IAM policies with statements
- `oci_security_list_cloud_guard_problems`: List Cloud Guard problems
- `oci_security_audit`: Full security audit (Tier 3)

## Response Formats

All tools support two response formats:
- `response_format: "markdown"` (default): Human-readable, context-efficient
- `response_format: "json"`: Machine-readable, complete data

## Workflow Guidelines

1. **Discovery First**: Use `oci_search_tools` to find relevant tools
2. **List Before Action**: Always list resources before taking actions
3. **Mutations Disabled**: By default, write operations are disabled (ALLOW_MUTATIONS=true to enable)
4. **Pagination**: Use limit/offset for large result sets
5. **Format Selection**: Use markdown for exploration, JSON for processing
"""


class InfrastructureAgent(BaseAgent, SelfHealingMixin):
    """
    Infrastructure Agent with OCI MCP Server Integration and Self-Healing.

    Specializes in OCI infrastructure operations using the mcp-oci MCP server:
    - Instance management (start, stop, restart, metrics)
    - Network topology analysis
    - Security configuration
    - Resource inventory and health checks

    Self-Healing Features:
    - Automatic retry on compute API failures
    - Parameter correction for network tool calls
    - LLM-powered infrastructure troubleshooting

    Workflow:
    1. Analyze request to determine action type
    2. Discover available resources
    3. Execute requested action or gather inventory
    4. Format and return results
    """

    # MCP tools from oci-infrastructure server
    # Tool naming convention: oci_{domain}_{action}_{resource}
    MCP_TOOLS = [
        # Discovery (Tier 1 - Instant)
        "oci_ping",
        "oci_list_domains",
        "oci_search_tools",
        "oci_get_capabilities",
        "search_capabilities",
        # Identity (Tier 2 - API)
        "oci_list_compartments",
        "oci_search_compartments",
        "oci_get_compartment",
        "oci_get_tenancy",
        "oci_list_regions",
        # Compute (Tier 2-4)
        "oci_compute_list_instances",
        "oci_compute_get_instance",
        "oci_compute_start_instance",
        "oci_compute_stop_instance",
        "oci_compute_restart_instance",
        "oci_observability_get_instance_metrics",
        # Network (Tier 2)
        "oci_network_list_vcns",
        "oci_network_get_vcn",
        "oci_network_list_subnets",
        "oci_network_list_security_lists",
        "oci_network_analyze_security",
        # Security (Tier 2-3)
        "oci_security_list_users",
        "oci_security_get_user",
        "oci_security_list_groups",
        "oci_security_list_policies",
        "oci_security_list_cloud_guard_problems",
        "oci_security_audit",
    ]

    @classmethod
    def get_definition(cls) -> AgentDefinition:
        """Return agent definition for catalog registration."""
        return AgentDefinition(
            agent_id="infrastructure-agent",
            role="infrastructure-agent",
            capabilities=[
                "compute-management",
                "network-analysis",
                "security-operations",
                "capacity-planning",
                "resource-inventory",
            ],
            skills=[
                "infra_inventory_workflow",
                "infra_instance_management_workflow",
                "infra_network_analysis_workflow",
                "infra_security_audit_workflow",
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.infrastructure-agent"],
                produce=["results.infrastructure-agent"],
            ),
            health_endpoint="http://localhost:8014/health",
            metadata=AgentMetadata(
                version="2.0.0",  # Updated for MCP integration
                namespace="oci-coordinator",
                max_iterations=15,
                timeout_seconds=60,
            ),
            description=(
                "Infrastructure Expert Agent for OCI compute, network, "
                "and security management using the mcp-oci MCP server."
            ),
            mcp_tools=cls.MCP_TOOLS,
            mcp_servers=["oci-unified", "oci-infrastructure"],
        )

    def __init__(
        self,
        memory_manager: SharedMemoryManager | None = None,
        tool_catalog: ToolCatalog | None = None,
        config: dict[str, Any] | None = None,
        llm: Any = None,
    ):
        """
        Initialize Infrastructure Agent with self-healing.

        Args:
            memory_manager: Shared memory manager
            tool_catalog: Tool catalog for MCP tools
            config: Agent configuration
            llm: LangChain LLM for analysis
        """
        super().__init__(memory_manager, tool_catalog, config)
        self.llm = llm
        self._graph: StateGraph | None = None
        self._tracer = trace.get_tracer("oci-infrastructure-agent")

        # Initialize self-healing capabilities
        if llm:
            self.init_self_healing(
                llm=llm,
                max_retries=3,
                enable_validation=True,
                enable_correction=True,
            )

    def build_graph(self) -> StateGraph:
        """
        Build the infrastructure management workflow graph.

        Graph structure:
        analyze_request → gather_inventory → [execute_action] →
        security_check (optional) → output
        """
        graph = StateGraph(InfrastructureState)

        # Add nodes
        graph.add_node("analyze_request", self._analyze_request_node)
        graph.add_node("gather_inventory", self._gather_inventory_node)
        graph.add_node("execute_action", self._execute_action_node)
        graph.add_node("security_check", self._security_check_node)
        graph.add_node("output", self._output_node)

        # Set entry point
        graph.set_entry_point("analyze_request")

        # Add conditional edges
        graph.add_conditional_edges(
            "analyze_request",
            self._route_after_analysis,
            {
                "inventory": "gather_inventory",
                "action": "execute_action",
                "security": "security_check",
            },
        )
        graph.add_edge("gather_inventory", "output")
        graph.add_edge("execute_action", "output")
        graph.add_edge("security_check", "output")
        graph.add_edge("output", END)

        self._graph = graph.compile()
        return self._graph

    def _route_after_analysis(self, state: InfrastructureState) -> str:
        """Determine next step based on action type."""
        if state.action_type in (ActionType.START, ActionType.STOP, ActionType.RESTART):
            return "action"
        elif state.action_type == ActionType.ANALYZE:
            return "security"
        return "inventory"

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_intent: str | None = None,
        use_self_healing: bool = True,
    ) -> dict[str, Any]:
        """
        Call an MCP tool with tracing, error handling, and self-healing.

        Args:
            tool_name: Name of the MCP tool
            arguments: Tool arguments
            user_intent: Original user query for context in self-healing
            use_self_healing: Whether to use self-healing capabilities

        Returns:
            Tool result or error dict
        """
        with self._tracer.start_as_current_span(f"mcp.tool.{tool_name}") as span:
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.tool.args", str(arguments)[:200])
            span.set_attribute("mcp.self_healing", use_self_healing and self._self_healing_enabled)

            start_time = time.time()
            try:
                if not self.tools:
                    return {"success": False, "error": "Tool catalog not initialized"}

                # Use self-healing tool call if enabled
                if use_self_healing and self._self_healing_enabled:
                    result = await self.healing_call_tool(
                        tool_name,
                        arguments,
                        user_intent=user_intent,
                        validate=True,
                        correct_on_failure=True,
                    )
                else:
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

    async def _analyze_request_node(self, state: InfrastructureState) -> dict[str, Any]:
        """Analyze infrastructure request to determine action type."""
        with self._tracer.start_as_current_span("analyze_request") as span:
            self._logger.info(
                "Analyzing infrastructure request",
                query=state.query[:100],
                trace_id=get_trace_id(),
            )

            # Determine action type from query
            query_lower = state.query.lower()

            action_keywords = {
                ActionType.START: ["start", "boot", "power on"],
                ActionType.STOP: ["stop", "shutdown", "power off", "halt"],
                ActionType.RESTART: ["restart", "reboot", "cycle"],
                ActionType.ANALYZE: ["analyze", "audit", "security", "check security"],
                ActionType.DESCRIBE: ["describe", "details", "info", "status"],
                ActionType.LIST: ["list", "show", "get", "inventory"],
            }

            action_type = ActionType.LIST
            for action, keywords in action_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    action_type = action
                    break

            # Extract resource ID if present (OCID pattern)
            resource_id = state.resource_id
            if not resource_id and "ocid1." in state.query:
                # Extract OCID from query
                import re
                ocid_match = re.search(r"ocid1\.[a-z]+\.[a-z0-9]+\.[a-z0-9-]+\.[a-z0-9]+", state.query)
                if ocid_match:
                    resource_id = ocid_match.group(0)

            # Resolve compartment name to OCID if not already set
            compartment_id = state.compartment_id
            if not compartment_id:
                compartment_id = await self._resolve_compartment_from_query(state.query)

            span.set_attribute("action_type", action_type.value)
            span.set_attribute("has_resource_id", resource_id is not None)
            span.set_attribute("compartment_resolved", compartment_id is not None)

            return {
                "action_type": action_type,
                "resource_id": resource_id,
                "compartment_id": compartment_id,
                "phase": "gather_inventory",
                "iteration": state.iteration + 1,
            }

    async def _resolve_compartment_from_query(self, query: str) -> str | None:
        """
        Extract and resolve compartment name from user query.

        Handles patterns like:
        - "in Adrian_birzu compartment"
        - "compartment Adrian_birzu"
        - "in the dev compartment"
        - Just "Adrian_birzu" as a potential compartment name

        Returns:
            Compartment OCID if found, None otherwise
        """
        import re
        from src.oci.tenancy_manager import TenancyManager

        query_lower = query.lower()

        # Try to extract compartment name from common patterns
        # \w includes word chars and underscore; also allow hyphens with [\w-]+
        patterns = [
            r"(?:in|from|for)\s+(?:the\s+)?([\w-]+)\s+compartment",  # "in Adrian_birzu compartment"
            r"compartment\s+([\w-]+)",  # "compartment Adrian_birzu"
            r"(?:in|from|for)\s+compartment\s+([\w-]+)",  # "in compartment dev"
        ]

        compartment_name = None
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                compartment_name = match.group(1)
                break

        if not compartment_name:
            # No pattern matched, try to find any word that might be a compartment
            # by checking against known compartments
            pass

        if compartment_name:
            try:
                manager = TenancyManager.get_instance()
                if not manager._initialized:
                    await manager.initialize()

                # Try exact match first
                ocid = await manager.get_compartment_ocid(compartment_name)
                if ocid:
                    self._logger.info(
                        "Resolved compartment name to OCID",
                        name=compartment_name,
                        ocid=ocid[:40] + "...",
                    )
                    return ocid

                # Try search for partial match
                matches = await manager.search_compartments(compartment_name)
                if matches:
                    # Return first match
                    self._logger.info(
                        "Found compartment via search",
                        query=compartment_name,
                        match=matches[0].name,
                        ocid=matches[0].id[:40] + "...",
                    )
                    return matches[0].id

            except Exception as e:
                self._logger.warning(
                    "Compartment resolution failed",
                    name=compartment_name,
                    error=str(e),
                )

        return None

    async def _gather_inventory_node(self, state: InfrastructureState) -> dict[str, Any]:
        """Gather infrastructure inventory using MCP tools."""
        with self._tracer.start_as_current_span("gather_inventory") as span:
            self._logger.info(
                "Gathering infrastructure inventory",
                compartment_id=state.compartment_id,
                trace_id=get_trace_id(),
            )

            instances = []
            vcns = []
            subnets = []
            security_lists = []

            # Determine what to fetch based on query
            query_lower = state.query.lower()
            fetch_instances = any(kw in query_lower for kw in ["instance", "compute", "vm", "all", "inventory"])
            fetch_network = any(kw in query_lower for kw in ["network", "vcn", "subnet", "all", "inventory"])
            fetch_security = any(kw in query_lower for kw in ["security", "firewall", "all"])

            # Default to instances if nothing specific mentioned
            if not fetch_instances and not fetch_network and not fetch_security:
                fetch_instances = True

            # Get instances
            if fetch_instances:
                inst_result = await self._call_mcp_tool(
                    "oci_compute_list_instances",
                    {
                        "compartment_id": state.compartment_id,
                        "limit": 50,
                        "response_format": "json",
                    },
                )
                if inst_result.get("success", True) and not inst_result.get("error"):
                    if isinstance(inst_result, dict) and "instances" in inst_result:
                        instances = inst_result.get("instances", [])
                    elif isinstance(inst_result, list):
                        instances = inst_result
                    elif isinstance(inst_result, str):
                        # Parse markdown/text response
                        self._logger.debug("Got text response for instances")

            # Get VCNs
            if fetch_network:
                vcn_result = await self._call_mcp_tool(
                    "oci_network_list_vcns",
                    {
                        "compartment_id": state.compartment_id,
                        "limit": 20,
                        "response_format": "json",
                    },
                )
                if vcn_result.get("success", True) and not vcn_result.get("error"):
                    if isinstance(vcn_result, dict) and "vcns" in vcn_result:
                        vcns = vcn_result.get("vcns", [])
                    elif isinstance(vcn_result, list):
                        vcns = vcn_result

                # Get subnets
                subnet_result = await self._call_mcp_tool(
                    "oci_network_list_subnets",
                    {
                        "compartment_id": state.compartment_id,
                        "limit": 50,
                        "response_format": "json",
                    },
                )
                if subnet_result.get("success", True) and not subnet_result.get("error"):
                    if isinstance(subnet_result, dict) and "subnets" in subnet_result:
                        subnets = subnet_result.get("subnets", [])

            # Get security lists
            if fetch_security:
                sec_result = await self._call_mcp_tool(
                    "oci_network_list_security_lists",
                    {
                        "compartment_id": state.compartment_id,
                        "limit": 20,
                        "response_format": "json",
                    },
                )
                if sec_result.get("success", True) and not sec_result.get("error"):
                    if isinstance(sec_result, dict) and "security_lists" in sec_result:
                        security_lists = sec_result.get("security_lists", [])

            # Calculate summary
            running_count = sum(
                1 for i in instances
                if isinstance(i, dict) and i.get("lifecycle_state") == "RUNNING"
            )

            summary = ResourceSummary(
                instances=len(instances),
                running_instances=running_count,
                vcns=len(vcns),
                subnets=len(subnets),
                security_lists=len(security_lists),
            )

            span.set_attribute("instances_count", len(instances))
            span.set_attribute("vcns_count", len(vcns))

            return {
                "instances": instances,
                "vcns": vcns,
                "subnets": subnets,
                "security_lists": security_lists,
                "summary": summary,
                "phase": "output",
            }

    async def _execute_action_node(self, state: InfrastructureState) -> dict[str, Any]:
        """Execute infrastructure action (start, stop, restart)."""
        with self._tracer.start_as_current_span("execute_action") as span:
            self._logger.info(
                "Executing infrastructure action",
                action=state.action_type.value,
                resource_id=state.resource_id,
                trace_id=get_trace_id(),
            )

            if not state.resource_id:
                return {
                    "error": "No resource ID specified for action",
                    "phase": "output",
                }

            action_result = {}
            tool_map = {
                ActionType.START: "oci_compute_start_instance",
                ActionType.STOP: "oci_compute_stop_instance",
                ActionType.RESTART: "oci_compute_restart_instance",
            }

            tool_name = tool_map.get(state.action_type)
            if tool_name:
                result = await self._call_mcp_tool(
                    tool_name,
                    {"instance_id": state.resource_id},
                )

                if result.get("error"):
                    # Check if mutations are disabled
                    if "ALLOW_MUTATIONS" in str(result.get("error", "")):
                        return {
                            "error": "Write operations are disabled. Set ALLOW_MUTATIONS=true to enable.",
                            "phase": "output",
                        }
                    return {
                        "error": result.get("error"),
                        "phase": "output",
                    }

                action_result = {
                    "action": state.action_type.value,
                    "resource_id": state.resource_id,
                    "status": "success",
                    "result": result,
                }

            span.set_attribute("action_success", bool(action_result))

            return {
                "action_result": action_result,
                "phase": "output",
            }

    async def _security_check_node(self, state: InfrastructureState) -> dict[str, Any]:
        """Perform security analysis using MCP tools."""
        with self._tracer.start_as_current_span("security_check") as span:
            self._logger.info(
                "Performing security analysis",
                compartment_id=state.compartment_id,
                trace_id=get_trace_id(),
            )

            security_analysis = {}

            # Get Cloud Guard problems
            problems_result = await self._call_mcp_tool(
                "oci_security_list_cloud_guard_problems",
                {
                    "compartment_id": state.compartment_id,
                    "status": "OPEN",
                    "limit": 20,
                },
            )

            if problems_result.get("success", True) and not problems_result.get("error"):
                security_analysis["cloud_guard_problems"] = problems_result.get("problems", [])

            # Get security analysis
            analysis_result = await self._call_mcp_tool(
                "oci_network_analyze_security",
                {"compartment_id": state.compartment_id},
            )

            if analysis_result.get("success", True) and not analysis_result.get("error"):
                security_analysis["network_security"] = analysis_result

            # Get policies
            policies_result = await self._call_mcp_tool(
                "oci_security_list_policies",
                {
                    "compartment_id": state.compartment_id,
                    "limit": 20,
                },
            )

            if policies_result.get("success", True) and not policies_result.get("error"):
                security_analysis["policies"] = policies_result.get("policies", [])

            span.set_attribute(
                "problems_count",
                len(security_analysis.get("cloud_guard_problems", []))
            )

            return {
                "security_analysis": security_analysis,
                "phase": "output",
            }

    async def _output_node(self, state: InfrastructureState) -> dict[str, Any]:
        """Prepare infrastructure report with structured formatting."""
        from src.formatting import (
            ListItem,
            MetricValue,
            ResponseFooter,
            Severity,
            StatusIndicator,
            TableData,
            TableRow,
        )

        # Handle errors
        if state.error:
            return {"result": self.format_error_response(state.error, "Infrastructure Error")}

        # Determine overall severity
        summary = state.summary
        if summary.instances > 0:
            health_ratio = summary.running_instances / summary.instances
            severity = "success" if health_ratio >= 0.9 else "medium" if health_ratio >= 0.5 else "high"
        else:
            severity = "info"

        # Create structured response
        response = self.create_response(
            title="Infrastructure Analysis",
            subtitle=f"Action: {state.action_type.value}",
            severity=severity,
            icon="🏗️",
        )

        # Add action result if any
        if state.action_result:
            response.add_section(
                title="Action Result",
                fields=[
                    StatusIndicator(
                        label="Action",
                        value=state.action_result.get("action", "unknown"),
                        severity=Severity.INFO,
                    ),
                    StatusIndicator(
                        label="Status",
                        value=state.action_result.get("status", "unknown"),
                        severity=Severity.SUCCESS if state.action_result.get("status") == "success" else Severity.HIGH,
                    ),
                    StatusIndicator(
                        label="Resource",
                        value=state.action_result.get("resource_id", "N/A")[:40],
                        severity=Severity.INFO,
                    ),
                ],
                divider_after=True,
            )

        # Add summary metrics
        response.add_metrics(
            "Resource Summary",
            [
                MetricValue(
                    label="Instances",
                    value=summary.instances,
                    severity=Severity.INFO,
                ),
                MetricValue(
                    label="Running",
                    value=summary.running_instances,
                    severity=Severity.SUCCESS if summary.running_instances == summary.instances else Severity.MEDIUM,
                ),
                MetricValue(
                    label="VCNs",
                    value=summary.vcns,
                    severity=Severity.INFO,
                ),
                MetricValue(
                    label="Subnets",
                    value=summary.subnets,
                    severity=Severity.INFO,
                ),
            ],
            divider_after=True,
        )

        # Add instances as table
        if state.instances:
            instance_rows = []
            for inst in state.instances[:10]:
                if isinstance(inst, dict):
                    lifecycle = inst.get("lifecycle_state", "Unknown")
                    instance_rows.append(
                        TableRow(
                            cells=[
                                inst.get("display_name", "Unknown"),
                                inst.get("shape", "N/A"),
                                lifecycle,
                            ],
                            severity=Severity.SUCCESS if lifecycle == "RUNNING" else Severity.HIGH,
                        )
                    )

            if instance_rows:
                instance_table = TableData(
                    headers=["Name", "Shape", "State"],
                    rows=instance_rows,
                    footer=f"Showing {min(10, len(state.instances))} of {len(state.instances)} instances",
                )
                response.add_table("Compute Instances", instance_table, divider_after=True)

        # Add VCNs
        if state.vcns:
            vcn_items = [
                ListItem(
                    text=vcn.get("display_name", "Unknown") if isinstance(vcn, dict) else str(vcn),
                    details=f"CIDR: {vcn.get('cidr_block', 'N/A')}" if isinstance(vcn, dict) else "",
                    severity=Severity.INFO,
                )
                for vcn in state.vcns[:5]
            ]
            response.add_section(
                title="Virtual Cloud Networks",
                list_items=vcn_items,
            )

        # Add security analysis
        if state.security_analysis:
            problems = state.security_analysis.get("cloud_guard_problems", [])
            if problems:
                response.add_section(
                    title=f"Security Issues ({len(problems)} found)",
                    list_items=[
                        ListItem(
                            text=p.get("problem_name", "Unknown") if isinstance(p, dict) else str(p),
                            details=p.get("risk_level", "MEDIUM") if isinstance(p, dict) else "",
                            severity=Severity.HIGH,
                        )
                        for p in problems[:5]
                    ],
                )

        # Add footer
        if not state.instances and not state.vcns and not state.action_result and not state.security_analysis:
            response.add_section(content="No infrastructure resources found.")

        response.footer = ResponseFooter(
            help_text="Use `/oci infra start <instance_id>` to manage instances",
            next_steps=[
                "Run security audit for compliance check",
                "Review instance utilization metrics",
            ],
            trace_id=get_trace_id(),
        )

        return {"result": self.format_response(response)}

    async def invoke(self, query: str, context: dict[str, Any] | None = None) -> str:
        """
        Execute infrastructure management workflow.

        Args:
            query: User query describing the infrastructure request
            context: Additional context (compartment_id, resource_id, etc.)

        Returns:
            Formatted infrastructure analysis result
        """
        context = context or {}

        with self._tracer.start_as_current_span("infrastructure_invoke") as span:
            span.set_attribute("query", query[:100])
            span.set_attribute("compartment_id", context.get("compartment_id", "none"))

            # Build graph if not already built
            if not self._graph:
                self.build_graph()

            # Create initial state
            initial_state = InfrastructureState(
                query=query,
                compartment_id=context.get("compartment_id"),
                resource_id=context.get("resource_id"),
            )

            self._logger.info(
                "Starting infrastructure analysis",
                query=query[:100],
                compartment_id=initial_state.compartment_id,
                trace_id=get_trace_id(),
            )

            try:
                # Run the workflow
                result = await self._graph.ainvoke(initial_state)

                # Return formatted result
                return result.get("result", "No infrastructure data available.")

            except Exception as e:
                self._logger.error(
                    "Infrastructure analysis failed",
                    error=str(e),
                    trace_id=get_trace_id(),
                )
                span.set_attribute("error", True)
                return f"Infrastructure analysis failed: {e}"
