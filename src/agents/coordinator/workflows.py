"""
Pre-built Deterministic Workflows for the Coordinator.

These workflows execute without LLM reasoning, providing fast,
predictable responses for common OCI operations.

Workflow signature:
    async def workflow(
        query: str,
        entities: dict[str, Any],
        tool_catalog: ToolCatalog,
        memory: SharedMemoryManager,
    ) -> str

Usage:
    from src.agents.coordinator.workflows import WORKFLOW_REGISTRY

    coordinator = LangGraphCoordinator(
        llm=llm,
        workflow_registry=WORKFLOW_REGISTRY,
        ...
    )
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Identity Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def list_compartments_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List compartments in the tenancy.

    Fast workflow that directly calls the MCP tool.
    Matches intents: list_compartments, show_compartments, get_compartments
    """
    try:
        # Extract parameters from entities
        compartment_id = entities.get("compartment_id")
        include_subtree = entities.get("include_subtree", True)

        result = await tool_catalog.execute(
            "oci_list_compartments",
            {
                "compartment_id": compartment_id,
                "include_subtree": include_subtree,
                "limit": 100,
                "format": "markdown",
            },
        )
        return result

    except Exception as e:
        logger.error("list_compartments workflow failed", error=str(e))
        return f"Error listing compartments: {e}"


async def get_tenancy_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get tenancy information.

    Matches intents: get_tenancy, tenancy_info, show_tenancy
    """
    try:
        result = await tool_catalog.execute(
            "oci_get_tenancy",
            {"format": "markdown"},
        )
        return result

    except Exception as e:
        logger.error("get_tenancy workflow failed", error=str(e))
        return f"Error getting tenancy: {e}"


async def list_regions_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List subscribed regions.

    Matches intents: list_regions, show_regions, available_regions
    """
    try:
        result = await tool_catalog.execute("oci_list_regions", {})
        return result

    except Exception as e:
        logger.error("list_regions workflow failed", error=str(e))
        return f"Error listing regions: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Compute Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def list_instances_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List compute instances.

    Matches intents: list_instances, show_instances, get_instances
    """
    try:
        compartment_id = entities.get("compartment_id")
        lifecycle_state = entities.get("lifecycle_state", "RUNNING")

        result = await tool_catalog.execute(
            "oci_compute_list_instances",
            {
                "compartment_id": compartment_id,
                "lifecycle_state": lifecycle_state,
                "limit": 50,
                "format": "markdown",
            },
        )
        return result

    except Exception as e:
        logger.error("list_instances workflow failed", error=str(e))
        return f"Error listing instances: {e}"


async def get_instance_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get details of a specific instance.

    Matches intents: get_instance, instance_details, describe_instance
    """
    try:
        instance_id = entities.get("instance_id")
        if not instance_id:
            return "Error: instance_id is required. Please provide the OCID of the instance."

        result = await tool_catalog.execute(
            "oci_compute_get_instance",
            {"instance_id": instance_id, "format": "markdown"},
        )
        return result

    except Exception as e:
        logger.error("get_instance workflow failed", error=str(e))
        return f"Error getting instance: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Network Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def list_vcns_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List Virtual Cloud Networks.

    Matches intents: list_vcns, show_networks, get_vcns
    """
    try:
        compartment_id = entities.get("compartment_id")

        result = await tool_catalog.execute(
            "oci_network_list_vcns",
            {
                "compartment_id": compartment_id,
                "format": "markdown",
            },
        )
        return result

    except Exception as e:
        logger.error("list_vcns workflow failed", error=str(e))
        return f"Error listing VCNs: {e}"


async def list_subnets_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List subnets in a compartment or VCN.

    Matches intents: list_subnets, show_subnets, get_subnets
    """
    try:
        compartment_id = entities.get("compartment_id")
        vcn_id = entities.get("vcn_id")

        result = await tool_catalog.execute(
            "oci_network_list_subnets",
            {
                "compartment_id": compartment_id,
                "vcn_id": vcn_id,
                "format": "markdown",
            },
        )
        return result

    except Exception as e:
        logger.error("list_subnets workflow failed", error=str(e))
        return f"Error listing subnets: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Cost Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def cost_summary_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get cost summary for the tenancy.

    Fast deterministic workflow that directly calls the cost tool.
    Matches intents: cost_summary, get_costs, show_spending, tenancy_costs,
                     how_much_spending, current_spend, monthly_costs
    """
    try:
        compartment_id = entities.get("compartment_id")

        # Parse days from time_range entity if present
        time_range = entities.get("time_range", "30d")
        days = 30
        if isinstance(time_range, str):
            if time_range.endswith("d"):
                try:
                    days = int(time_range.replace("d", ""))
                except ValueError:
                    pass
            elif "7" in time_range or "week" in time_range.lower():
                days = 7
            elif "90" in time_range or "quarter" in time_range.lower():
                days = 90

        result = await tool_catalog.execute(
            "oci_cost_get_summary",
            {
                "compartment_id": compartment_id,  # None defaults to tenancy in tool
                "days": days,
            },
        )

        # Return raw JSON for channel-specific formatting
        # The Slack handler has built-in cost_summary detection that:
        # 1. Parses the JSON
        # 2. Extracts services array
        # 3. Formats as native Slack table block
        # Other channels (API, CLI) also benefit from structured data
        return result

    except Exception as e:
        logger.error("cost_summary workflow failed", error=str(e))
        return f"Error getting cost summary: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Discovery Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def discovery_summary_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get resource discovery summary from cache.

    Matches intents: resource_summary, show_resources, discovery_summary
    """
    try:
        result = await tool_catalog.execute("oci_discovery_summary", {})
        return result

    except Exception as e:
        logger.error("discovery_summary workflow failed", error=str(e))
        return f"Error getting discovery summary: {e}"


async def search_resources_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Search for resources by type or name.

    Matches intents: search_resources, find_resource, resource_search
    """
    try:
        resource_type = entities.get("resource_type")
        name_pattern = entities.get("name_pattern") or entities.get("name")

        result = await tool_catalog.execute(
            "oci_discovery_search",
            {
                "resource_type": resource_type,
                "name_pattern": name_pattern,
                "limit": 50,
            },
        )
        return result

    except Exception as e:
        logger.error("search_resources workflow failed", error=str(e))
        return f"Error searching resources: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Capability Search Workflow
# ─────────────────────────────────────────────────────────────────────────────


async def search_capabilities_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Search for available capabilities/tools.

    Matches intents: help, what_can_you_do, capabilities, available_tools
    """
    try:
        search_term = entities.get("search_term", "")
        domain = entities.get("domain")

        # Use the query as search term if not specified
        if not search_term and query:
            # Extract keywords from query
            stop_words = {"what", "can", "you", "do", "how", "help", "me", "with", "the", "a", "an"}
            words = query.lower().split()
            keywords = [w for w in words if w not in stop_words]
            search_term = " ".join(keywords[:3]) if keywords else ""

        result = await tool_catalog.execute(
            "search_capabilities",
            {
                "query": search_term or "all",
                "domain": domain,
            },
        )
        return result

    except Exception as e:
        logger.error("search_capabilities workflow failed", error=str(e))
        return f"Error searching capabilities: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Registry
# ─────────────────────────────────────────────────────────────────────────────


# Map workflow names to functions
# These names should match what the classifier suggests
WORKFLOW_REGISTRY: dict[str, Any] = {
    # Identity
    "list_compartments": list_compartments_workflow,
    "get_tenancy": get_tenancy_workflow,
    "list_regions": list_regions_workflow,

    # Compute
    "list_instances": list_instances_workflow,
    "get_instance": get_instance_workflow,

    # Network
    "list_vcns": list_vcns_workflow,
    "list_subnets": list_subnets_workflow,

    # Cost - many aliases for common phrasings
    "cost_summary": cost_summary_workflow,
    "get_cost_summary": cost_summary_workflow,
    "get_costs": cost_summary_workflow,
    "show_costs": cost_summary_workflow,
    "get_tenancy_costs": cost_summary_workflow,  # Common classifier output
    "tenancy_costs": cost_summary_workflow,
    "spending": cost_summary_workflow,
    "show_spending": cost_summary_workflow,
    "monthly_cost": cost_summary_workflow,
    "how_much_spent": cost_summary_workflow,

    # Discovery
    "discovery_summary": discovery_summary_workflow,
    "resource_summary": discovery_summary_workflow,  # Alias
    "search_resources": search_resources_workflow,

    # Help/Capabilities
    "search_capabilities": search_capabilities_workflow,
    "help": search_capabilities_workflow,
    "capabilities": search_capabilities_workflow,
}


def get_workflow_registry() -> dict[str, Any]:
    """Get the workflow registry for use with the coordinator."""
    return WORKFLOW_REGISTRY.copy()


def list_workflows() -> list[str]:
    """List all available workflow names."""
    return list(WORKFLOW_REGISTRY.keys())


def get_workflow_descriptions() -> dict[str, str]:
    """Get workflow names with their descriptions."""
    return {
        name: (func.__doc__ or "").split("\n")[1].strip()
        for name, func in WORKFLOW_REGISTRY.items()
        if func.__doc__
    }
