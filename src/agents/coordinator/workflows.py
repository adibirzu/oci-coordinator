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

from src.mcp.client import ToolCallResult

logger = structlog.get_logger(__name__)


def _get_root_compartment() -> str | None:
    """
    Get the root compartment (tenancy) ID from OCI config.

    Returns the tenancy OCID which can be used as default compartment
    when no specific compartment is provided.
    """
    try:
        import oci
        config = oci.config.from_file()
        return config.get("tenancy")
    except Exception as e:
        logger.debug("Could not get tenancy from OCI config", error=str(e))
        return None


def _extract_result(result: ToolCallResult | str | Any) -> str:
    """Extract string result from ToolCallResult or return string directly."""
    if isinstance(result, ToolCallResult):
        if result.success:
            return str(result.result) if result.result is not None else ""
        else:
            return f"Error: {result.error}"
    return str(result) if result is not None else ""


async def _resolve_compartment(
    name_or_id: str | None,
    tool_catalog: ToolCatalog,
    query: str | None = None,
) -> str | None:
    """
    Resolve a compartment name or ID to an OCID.

    Args:
        name_or_id: Compartment name, partial name, or OCID
        tool_catalog: Tool catalog for executing MCP tools
        query: Original query to extract compartment name from if name_or_id is None

    Returns:
        Compartment OCID if found, root compartment as fallback, or None
    """
    # If already an OCID, return as-is
    if name_or_id and name_or_id.startswith("ocid1."):
        return name_or_id

    # Try to extract compartment name from query if not provided
    search_name = name_or_id
    if not search_name and query:
        # Look for common patterns like "in X compartment" or "X compartment"
        import re
        patterns = [
            r"(?:in|from|for)\s+(\w+(?:[-_]\w+)*)\s+compartment",
            r"(\w+(?:[-_]\w+)*)\s+compartment",
            r"compartment\s+(\w+(?:[-_]\w+)*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                search_name = match.group(1)
                logger.debug("Extracted compartment name from query", name=search_name)
                break

    # If we have a name to search, try to find it
    if search_name:
        try:
            # Search for compartments matching the name
            result = await tool_catalog.execute(
                "oci_search_compartments",
                {"query": search_name, "limit": 5},
            )
            result_str = _extract_result(result)

            # Parse the result to find matching compartment
            if result_str and "ocid1.compartment" in result_str:
                # Try to extract OCID from various formats
                import re
                # Match OCID pattern
                ocid_match = re.search(r"(ocid1\.compartment\.[^\s,\]\"']+)", result_str)
                if ocid_match:
                    compartment_id = ocid_match.group(1)
                    logger.info("Resolved compartment name to OCID",
                               name=search_name, ocid=compartment_id[:50])
                    return compartment_id

            # Try JSON parsing if it's structured
            try:
                data = json.loads(result_str)
                if isinstance(data, list) and data:
                    # Return first matching compartment
                    return data[0].get("id") or data[0].get("ocid")
                elif isinstance(data, dict):
                    return data.get("id") or data.get("ocid")
            except (json.JSONDecodeError, TypeError):
                pass

        except Exception as e:
            logger.debug("Compartment search failed", name=search_name, error=str(e))

    # Fall back to root compartment
    root = _get_root_compartment()
    if root:
        logger.debug("Using root compartment as fallback", ocid=root[:50])
    return root


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
        # Resolve compartment - default to root for listing
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        compartment_id = await _resolve_compartment(compartment_input, tool_catalog, query)

        # Use root compartment if nothing resolved
        if not compartment_id:
            compartment_id = _get_root_compartment()

        include_subtree = entities.get("include_subtree", True)

        result = await tool_catalog.execute(
            "oci_list_compartments",
            {
                "compartment_id": compartment_id,
                "include_subtree": include_subtree,
                "limit": 100,
                "format": "json",
            },
        )
        return _extract_result(result)

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
            {"format": "json"},
        )
        return _extract_result(result)

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
        return _extract_result(result)

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
    Supports compartment name resolution (e.g., "list instances in Adrian_birzu compartment")
    """
    try:
        # Resolve compartment name to OCID
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        logger.debug(
            "list_instances workflow starting",
            query=query,
            compartment_input=compartment_input,
            entities=entities,
        )
        compartment_id = await _resolve_compartment(compartment_input, tool_catalog, query)

        # Fall back to root compartment if not resolved
        if not compartment_id:
            compartment_id = _get_root_compartment()
            logger.debug("Using root compartment as fallback", compartment_id=compartment_id)
            if not compartment_id:
                return "Error: Could not determine compartment. Please specify a compartment name or configure OCI CLI."

        lifecycle_state = entities.get("lifecycle_state", "RUNNING")

        logger.info(
            "Executing oci_compute_list_instances",
            compartment_id=compartment_id,
            compartment_id_type=type(compartment_id).__name__,
            compartment_id_len=len(compartment_id) if compartment_id else 0,
            lifecycle_state=lifecycle_state,
        )

        result = await tool_catalog.execute(
            "oci_compute_list_instances",
            {
                "compartment_id": compartment_id,
                "lifecycle_state": lifecycle_state,
                "limit": 50,
                "format": "json",  # JSON format for Slack table rendering
            },
        )

        logger.debug(
            "Tool execution completed",
            success=result.success if hasattr(result, 'success') else 'unknown',
            error=result.error if hasattr(result, 'error') else None,
        )
        return _extract_result(result)

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
            {"instance_id": instance_id, "format": "json"},
        )
        return _extract_result(result)

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
    Supports compartment name resolution.
    """
    try:
        # Resolve compartment name to OCID
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        compartment_id = await _resolve_compartment(compartment_input, tool_catalog, query)

        # Fall back to root compartment if not resolved
        if not compartment_id:
            compartment_id = _get_root_compartment()
            if not compartment_id:
                return "Error: Could not determine compartment. Please specify a compartment name or configure OCI CLI."

        result = await tool_catalog.execute(
            "oci_network_list_vcns",
            {
                "compartment_id": compartment_id,
                "format": "json",
            },
        )
        return _extract_result(result)

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
    Supports compartment name resolution.
    """
    try:
        # Resolve compartment name to OCID
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        compartment_id = await _resolve_compartment(compartment_input, tool_catalog, query)

        # Fall back to root compartment if not resolved
        if not compartment_id:
            compartment_id = _get_root_compartment()
            if not compartment_id:
                return "Error: Could not determine compartment. Please specify a compartment name or configure OCI CLI."

        vcn_id = entities.get("vcn_id")

        result = await tool_catalog.execute(
            "oci_network_list_subnets",
            {
                "compartment_id": compartment_id,
                "vcn_id": vcn_id,
                "format": "json",
            },
        )
        return _extract_result(result)

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
        return _extract_result(result)

    except Exception as e:
        logger.error("cost_summary workflow failed", error=str(e))
        return f"Error getting cost summary: {e}"


async def database_costs_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get database-specific cost summary.

    Filters cost data to show only database-related services
    (Autonomous Database, ATP, ADW, Exadata, MySQL, NoSQL, etc.)

    Matches intents: database_costs, db_costs, database_spending,
                     db_spending, autonomous_costs, atp_costs
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
                "compartment_id": compartment_id,
                "days": days,
                "service_filter": "database",  # Filter for database services only
            },
        )

        return _extract_result(result)

    except Exception as e:
        logger.error("database_costs workflow failed", error=str(e))
        return f"Error getting database costs: {e}"


async def compute_costs_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get compute-specific cost summary.

    Filters cost data to show only compute-related services.

    Matches intents: compute_costs, instance_costs, vm_costs
    """
    try:
        compartment_id = entities.get("compartment_id")
        time_range = entities.get("time_range", "30d")
        days = 30
        if isinstance(time_range, str) and time_range.endswith("d"):
            try:
                days = int(time_range.replace("d", ""))
            except ValueError:
                pass

        result = await tool_catalog.execute(
            "oci_cost_get_summary",
            {
                "compartment_id": compartment_id,
                "days": days,
                "service_filter": "compute",
            },
        )

        return _extract_result(result)

    except Exception as e:
        logger.error("compute_costs workflow failed", error=str(e))
        return f"Error getting compute costs: {e}"


async def storage_costs_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get storage-specific cost summary.

    Filters cost data to show only storage-related services.

    Matches intents: storage_costs, block_storage_costs, object_storage_costs
    """
    try:
        compartment_id = entities.get("compartment_id")
        time_range = entities.get("time_range", "30d")
        days = 30
        if isinstance(time_range, str) and time_range.endswith("d"):
            try:
                days = int(time_range.replace("d", ""))
            except ValueError:
                pass

        result = await tool_catalog.execute(
            "oci_cost_get_summary",
            {
                "compartment_id": compartment_id,
                "days": days,
                "service_filter": "storage",
            },
        )

        return _extract_result(result)

    except Exception as e:
        logger.error("storage_costs workflow failed", error=str(e))
        return f"Error getting storage costs: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Database Listing Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def list_databases_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List databases in the tenancy.

    Lists Autonomous Databases and DB Systems accessible to the user.
    Uses fast data sources first (OPSI cache) with fallback to slower APIs.

    Matches intents: list_databases, show_databases, database_names,
                     list_db, show_db, get_databases
    """
    import asyncio

    # Timeout for each tool call (30 seconds max per source)
    TOOL_TIMEOUT = 30

    async def safe_execute(tool_name: str, params: dict) -> str | None:
        """Execute tool with timeout and error handling."""
        try:
            result = await asyncio.wait_for(
                tool_catalog.execute(tool_name, params),
                timeout=TOOL_TIMEOUT
            )
            result_str = _extract_result(result)
            # Filter out error messages and validation errors
            if result_str and not any(err in result_str.lower() for err in [
                "error", "timeout", "validation error", "missing required"
            ]):
                return result_str
        except asyncio.TimeoutError:
            logger.warning(f"Tool {tool_name} timed out after {TOOL_TIMEOUT}s")
        except Exception as e:
            logger.debug(f"Tool {tool_name} failed", error=str(e))
        return None

    try:
        # Resolve compartment name to OCID
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        compartment_id = await _resolve_compartment(compartment_input, tool_catalog, query)

        # Fall back to root compartment if nothing resolved
        if not compartment_id:
            compartment_id = _get_root_compartment()
            if compartment_id:
                logger.info("list_databases: using root compartment as default")

        results = []
        errors = []

        # Priority 1: Try OPSI search_databases (fastest - uses cache)
        # OPSI can work without compartment_id (searches all accessible)
        logger.info("list_databases: trying OPSI search_databases")
        opsi_params = {"limit": 50}
        if compartment_id:
            opsi_params["compartment_id"] = compartment_id
        opsi_result = await safe_execute("oci_opsi_search_databases", opsi_params)
        if opsi_result:
            results.append("## Databases (from OPSI)\n" + opsi_result)
        else:
            errors.append("OPSI search unavailable")

        # Priority 2: Try autonomous databases (slower API)
        # This tool requires compartment_id
        if not results and compartment_id:
            logger.info("list_databases: trying oci_database_list_autonomous")
            adb_result = await safe_execute(
                "oci_database_list_autonomous",
                {"compartment_id": compartment_id}
            )
            if adb_result:
                results.append("## Autonomous Databases\n" + adb_result)
            else:
                errors.append("Autonomous DB API unavailable")
        elif not results:
            errors.append("Autonomous DB API skipped (no compartment)")

        # Priority 3: Try database connections (database-observatory)
        if not results:
            logger.info("list_databases: trying oci_database_list_connections")
            conn_result = await safe_execute(
                "oci_database_list_connections",
                {}
            )
            if conn_result:
                results.append("## Database Connections\n" + conn_result)
            else:
                errors.append("Database connections unavailable")

        if results:
            return "\n\n".join(results)
        else:
            # Provide helpful error message
            return (
                "No databases found. Possible reasons:\n"
                "- No databases exist in the accessible compartments\n"
                "- OPSI is not configured for database monitoring\n"
                "- Insufficient permissions to list databases\n\n"
                f"Data sources tried: {', '.join(errors)}"
            )

    except Exception as e:
        logger.error("list_databases workflow failed", error=str(e))
        return f"Error listing databases: {e}"


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
        return _extract_result(result)

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
        return _extract_result(result)

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
        return _extract_result(result)

    except Exception as e:
        logger.error("search_capabilities workflow failed", error=str(e))
        return f"Error searching capabilities: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Cost-to-Resource Mapping Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def resource_cost_overview_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get comprehensive cost overview with associated resources.

    Combines cost summary with resource discovery to show what's costing
    money and what resources are behind those costs.

    Matches intents: resource_cost_overview, cost_with_resources, full_cost_breakdown
    """
    import asyncio

    try:
        days = entities.get("days", 30)
        compartment_id = entities.get("compartment_id")

        results = []
        results.append(f"# Cost & Resource Overview (Last {days} days)\n")

        # Fetch cost summary and resource discovery in parallel
        async def get_costs():
            try:
                result = await tool_catalog.execute(
                    "oci_cost_get_summary",
                    {"days": days, "compartment_id": compartment_id},
                )
                return _extract_result(result)
            except Exception as e:
                logger.warning("Cost fetch failed", error=str(e))
                return None

        async def get_resources():
            try:
                result = await tool_catalog.execute(
                    "oci_discovery_summary",
                    {"compartment_id": compartment_id},
                )
                return _extract_result(result)
            except Exception as e:
                logger.warning("Resource discovery failed", error=str(e))
                return None

        cost_data, resource_data = await asyncio.gather(get_costs(), get_resources())

        # Parse and format cost data
        if cost_data:
            try:
                cost_json = json.loads(cost_data)
                if cost_json.get("type") == "cost_summary":
                    summary = cost_json.get("summary", {})
                    services = cost_json.get("services", [])

                    results.append("## Cost Summary")
                    results.append(f"**Total Spend:** {summary.get('total', 'N/A')}")
                    results.append(f"**Period:** {summary.get('period', 'N/A')}\n")

                    if services:
                        results.append("### Top Services by Cost")
                        for svc in services[:10]:
                            results.append(
                                f"- **{svc['service']}**: {svc['cost']} ({svc['percent']})"
                            )
                        results.append("")
                else:
                    results.append("## Cost Data\n" + cost_data)
            except json.JSONDecodeError:
                results.append("## Cost Data\n" + cost_data)

        # Add resource discovery data
        if resource_data:
            results.append("## Resource Inventory")
            results.append(resource_data)

        if not cost_data and not resource_data:
            return "Unable to fetch cost or resource data. Please check your permissions."

        return "\n".join(results)

    except Exception as e:
        logger.error("resource_cost_overview workflow failed", error=str(e))
        return f"Error getting cost/resource overview: {e}"


async def compute_with_costs_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List compute instances with associated cost data.

    Shows running instances and their contribution to compute costs.

    Matches intents: compute_with_costs, instance_costs_detail, vm_cost_breakdown
    """
    import asyncio

    try:
        compartment_id = entities.get("compartment_id")
        days = entities.get("days", 30)

        results = []
        results.append(f"# Compute Resources & Costs (Last {days} days)\n")

        # Fetch both in parallel
        async def get_instances():
            try:
                result = await tool_catalog.execute(
                    "oci_compute_list_instances",
                    {"compartment_id": compartment_id},
                )
                return _extract_result(result)
            except Exception as e:
                logger.warning("Instance list failed", error=str(e))
                return None

        async def get_compute_costs():
            try:
                result = await tool_catalog.execute(
                    "oci_cost_get_summary",
                    {
                        "days": days,
                        "compartment_id": compartment_id,
                        "service_filter": "compute",
                    },
                )
                return _extract_result(result)
            except Exception as e:
                logger.warning("Compute cost fetch failed", error=str(e))
                return None

        instances_data, cost_data = await asyncio.gather(get_instances(), get_compute_costs())

        # Add cost summary
        if cost_data:
            results.append("## Compute Costs")
            try:
                cost_json = json.loads(cost_data)
                if cost_json.get("type") == "cost_summary":
                    summary = cost_json.get("summary", {})
                    services = cost_json.get("services", [])
                    results.append(f"**Total Compute Spend:** {summary.get('total', 'N/A')}")
                    if services:
                        for svc in services[:5]:
                            results.append(f"- {svc['service']}: {svc['cost']}")
                    results.append("")
                else:
                    results.append(cost_data + "\n")
            except json.JSONDecodeError:
                results.append(cost_data + "\n")

        # Add instance list
        if instances_data:
            results.append("## Running Instances")
            results.append(instances_data)
        else:
            results.append("## Running Instances\nNo compute instances found.\n")

        return "\n".join(results)

    except Exception as e:
        logger.error("compute_with_costs workflow failed", error=str(e))
        return f"Error getting compute resources and costs: {e}"


async def database_with_costs_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List databases with associated cost data.

    Shows database resources and their contribution to database costs.

    Matches intents: database_with_costs, db_cost_breakdown, database_cost_detail
    """
    import asyncio

    try:
        # Get compartment_id from entities, or fall back to root compartment (tenancy)
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()
            if compartment_id:
                logger.info("database_with_costs: using root compartment as default")

        days = entities.get("days", 30)

        results = []
        results.append(f"# Database Resources & Costs (Last {days} days)\n")

        # Fetch both in parallel
        async def get_databases():
            # Try OPSI first (faster, uses cache, optional compartment_id)
            try:
                opsi_params = {"limit": 50}
                if compartment_id:
                    opsi_params["compartment_id"] = compartment_id
                result = await tool_catalog.execute(
                    "oci_opsi_search_databases",
                    opsi_params,
                )
                result_str = _extract_result(result)
                # Only return if it's valid data (not an error message)
                if result_str and "Error" not in result_str and "validation error" not in result_str.lower():
                    return result_str
            except Exception as e:
                logger.debug("OPSI search failed", error=str(e))

            # Fall back to OCI Database API (requires compartment_id)
            if compartment_id:
                try:
                    result = await tool_catalog.execute(
                        "oci_database_list_autonomous",
                        {"compartment_id": compartment_id},
                    )
                    result_str = _extract_result(result)
                    # Only return if it's valid data
                    if result_str and "Error" not in result_str and "validation error" not in result_str.lower():
                        return result_str
                except Exception as e:
                    logger.debug("Autonomous DB list failed", error=str(e))

            logger.warning("All database listing methods failed")
            return None

        async def get_database_costs():
            try:
                cost_params = {
                    "days": days,
                    "service_filter": "database",
                }
                if compartment_id:
                    cost_params["compartment_id"] = compartment_id
                result = await tool_catalog.execute(
                    "oci_cost_get_summary",
                    cost_params,
                )
                return _extract_result(result)
            except Exception as e:
                logger.warning("Database cost fetch failed", error=str(e))
                return None

        databases_data, cost_data = await asyncio.gather(get_databases(), get_database_costs())

        # Add cost summary
        if cost_data:
            results.append("## Database Costs")
            try:
                cost_json = json.loads(cost_data)
                if cost_json.get("type") == "cost_summary":
                    summary = cost_json.get("summary", {})
                    services = cost_json.get("services", [])
                    results.append(f"**Total Database Spend:** {summary.get('total', 'N/A')}")
                    if services:
                        for svc in services[:5]:
                            results.append(f"- {svc['service']}: {svc['cost']}")
                    results.append("")
                else:
                    results.append(cost_data + "\n")
            except json.JSONDecodeError:
                results.append(cost_data + "\n")

        # Add database list
        if databases_data:
            results.append("## Database Resources")
            results.append(databases_data)
        else:
            results.append("## Database Resources\nNo databases found.\n")

        return "\n".join(results)

    except Exception as e:
        logger.error("database_with_costs workflow failed", error=str(e))
        return f"Error getting database resources and costs: {e}"


async def compartment_cost_breakdown_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get costs grouped by compartment with resource summary.

    Shows which compartments are spending the most and what resources they contain.

    Matches intents: compartment_costs, cost_by_compartment, compartment_breakdown
    """
    import asyncio

    try:
        days = entities.get("days", 30)

        results = []
        results.append(f"# Cost by Compartment (Last {days} days)\n")

        # Fetch both in parallel
        async def get_compartments():
            try:
                result = await tool_catalog.execute(
                    "oci_list_compartments",
                    {},
                )
                return _extract_result(result)
            except Exception as e:
                logger.warning("Compartment list failed", error=str(e))
                return None

        async def get_total_costs():
            try:
                result = await tool_catalog.execute(
                    "oci_cost_get_summary",
                    {"days": days},
                )
                return _extract_result(result)
            except Exception as e:
                logger.warning("Cost fetch failed", error=str(e))
                return None

        compartments_data, cost_data = await asyncio.gather(
            get_compartments(), get_total_costs()
        )

        # Add overall cost summary
        if cost_data:
            results.append("## Overall Cost Summary")
            try:
                cost_json = json.loads(cost_data)
                if cost_json.get("type") == "cost_summary":
                    summary = cost_json.get("summary", {})
                    services = cost_json.get("services", [])
                    results.append(f"**Total Tenancy Spend:** {summary.get('total', 'N/A')}")
                    results.append(f"**Period:** {summary.get('period', 'N/A')}\n")

                    if services:
                        results.append("### Top Services")
                        for svc in services[:5]:
                            results.append(f"- {svc['service']}: {svc['cost']} ({svc['percent']})")
                        results.append("")
                else:
                    results.append(cost_data + "\n")
            except json.JSONDecodeError:
                results.append(cost_data + "\n")

        # Add compartment structure
        if compartments_data:
            results.append("## Compartment Structure")
            results.append(compartments_data)
            results.append("\n*Note: To see costs for a specific compartment, ask about costs for that compartment by name.*")
        else:
            results.append("## Compartments\nUnable to list compartments.\n")

        return "\n".join(results)

    except Exception as e:
        logger.error("compartment_cost_breakdown workflow failed", error=str(e))
        return f"Error getting compartment cost breakdown: {e}"


async def resource_utilization_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get resource utilization summary to identify optimization opportunities.

    Shows resource usage metrics to help identify underutilized resources.

    Matches intents: resource_utilization, usage_metrics, optimization_opportunities
    """
    try:
        compartment_id = entities.get("compartment_id")

        results = []
        results.append("# Resource Utilization & Optimization\n")

        # Try to get OPSI fleet summary for database utilization
        try:
            fleet_result = await tool_catalog.execute(
                "oci_opsi_get_fleet_summary",
                {"compartment_id": compartment_id},
            )
            fleet_data = _extract_result(fleet_result)
            if fleet_data and "Error" not in fleet_data:
                results.append("## Database Fleet Utilization")
                results.append(fleet_data)
                results.append("")
        except Exception as e:
            logger.debug("Fleet summary not available", error=str(e))

        # Get resource discovery summary
        try:
            discovery_result = await tool_catalog.execute(
                "oci_discovery_summary",
                {"compartment_id": compartment_id},
            )
            discovery_data = _extract_result(discovery_result)
            if discovery_data and "Error" not in discovery_data:
                results.append("## Resource Inventory")
                results.append(discovery_data)
        except Exception as e:
            logger.debug("Discovery summary not available", error=str(e))

        if len(results) == 1:
            results.append("No utilization data available. Check compartment access or OPSI configuration.")

        results.append("\n### Optimization Tips")
        results.append("- Review databases with low CPU utilization for potential downsizing")
        results.append("- Check for stopped instances that are still incurring storage costs")
        results.append("- Consider reserved capacity for predictable workloads")

        return "\n".join(results)

    except Exception as e:
        logger.error("resource_utilization workflow failed", error=str(e))
        return f"Error getting resource utilization: {e}"


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

    # Cost - many aliases for common phrasings (generic tenancy-wide)
    "cost_summary": cost_summary_workflow,
    "get_cost_summary": cost_summary_workflow,
    "get_costs": cost_summary_workflow,
    "show_costs": cost_summary_workflow,
    "get_tenancy_costs": cost_summary_workflow,
    "tenancy_costs": cost_summary_workflow,
    "spending": cost_summary_workflow,
    "show_spending": cost_summary_workflow,
    "monthly_cost": cost_summary_workflow,
    "how_much_spent": cost_summary_workflow,

    # Database-specific costs (higher priority for DB-related queries)
    "database_costs": database_costs_workflow,
    "db_costs": database_costs_workflow,
    "database_spending": database_costs_workflow,
    "db_spending": database_costs_workflow,
    "autonomous_costs": database_costs_workflow,
    "atp_costs": database_costs_workflow,
    "adw_costs": database_costs_workflow,
    "show_database_costs": database_costs_workflow,
    "get_database_costs": database_costs_workflow,

    # Compute-specific costs
    "compute_costs": compute_costs_workflow,
    "instance_costs": compute_costs_workflow,
    "vm_costs": compute_costs_workflow,
    "show_compute_costs": compute_costs_workflow,

    # Storage-specific costs
    "storage_costs": storage_costs_workflow,
    "block_storage_costs": storage_costs_workflow,
    "object_storage_costs": storage_costs_workflow,
    "show_storage_costs": storage_costs_workflow,

    # Database listing (names, inventory)
    "list_databases": list_databases_workflow,
    "show_databases": list_databases_workflow,
    "database_names": list_databases_workflow,
    "get_databases": list_databases_workflow,
    "list_db": list_databases_workflow,
    "show_db": list_databases_workflow,
    "database_list": list_databases_workflow,
    "list_autonomous": list_databases_workflow,
    "autonomous_databases": list_databases_workflow,
    "show_autonomous": list_databases_workflow,

    # Discovery
    "discovery_summary": discovery_summary_workflow,
    "resource_summary": discovery_summary_workflow,  # Alias
    "search_resources": search_resources_workflow,

    # Cost-to-Resource Mapping (combines costs with resource discovery)
    "resource_cost_overview": resource_cost_overview_workflow,
    "cost_with_resources": resource_cost_overview_workflow,
    "full_cost_breakdown": resource_cost_overview_workflow,
    "cost_overview": resource_cost_overview_workflow,
    "what_costs_most": resource_cost_overview_workflow,
    "spending_breakdown": resource_cost_overview_workflow,

    # Compute with costs
    "compute_with_costs": compute_with_costs_workflow,
    "instance_costs_detail": compute_with_costs_workflow,
    "vm_cost_breakdown": compute_with_costs_workflow,
    "show_instances_with_costs": compute_with_costs_workflow,
    "compute_spending": compute_with_costs_workflow,

    # Database with costs
    "database_with_costs": database_with_costs_workflow,
    "db_cost_breakdown": database_with_costs_workflow,
    "database_cost_detail": database_with_costs_workflow,
    "show_databases_with_costs": database_with_costs_workflow,

    # Compartment cost breakdown
    "compartment_cost_breakdown": compartment_cost_breakdown_workflow,
    "compartment_costs": compartment_cost_breakdown_workflow,
    "cost_by_compartment": compartment_cost_breakdown_workflow,
    "spending_by_compartment": compartment_cost_breakdown_workflow,

    # Resource utilization / optimization
    "resource_utilization": resource_utilization_workflow,
    "usage_metrics": resource_utilization_workflow,
    "optimization_opportunities": resource_utilization_workflow,
    "optimize_costs": resource_utilization_workflow,
    "underutilized_resources": resource_utilization_workflow,

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
