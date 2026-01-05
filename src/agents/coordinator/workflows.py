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

import asyncio
import json
import subprocess
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

from src.mcp.client import ToolCallResult

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OCI CLI Fallback Support
# ─────────────────────────────────────────────────────────────────────────────


async def _run_oci_cli(
    command: list[str],
    timeout: int = 60,
) -> tuple[bool, str]:
    """
    Run an OCI CLI command as fallback when REST API fails.

    Args:
        command: OCI CLI command as list (e.g., ["oci", "compute", "instance", "list"])
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        # Run in thread pool to not block event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            ),
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            error_msg = result.stderr or f"OCI CLI exited with code {result.returncode}"
            logger.warning("OCI CLI command failed", command=command[0:4], error=error_msg)
            return False, error_msg

    except subprocess.TimeoutExpired:
        logger.warning("OCI CLI command timed out", command=command[0:4], timeout=timeout)
        return False, f"OCI CLI command timed out after {timeout}s"
    except FileNotFoundError:
        return False, "OCI CLI not installed or not in PATH"
    except Exception as e:
        logger.error("OCI CLI execution error", command=command[0:4], error=str(e))
        return False, str(e)


async def _execute_with_cli_fallback(
    tool_catalog: "ToolCatalog",
    tool_name: str,
    tool_params: dict[str, Any],
    cli_command: list[str],
    cli_timeout: int = 60,
) -> str:
    """
    Execute a tool with OCI CLI fallback on failure.

    First tries the MCP tool. If it fails, falls back to OCI CLI command.

    Args:
        tool_catalog: Tool catalog for MCP execution
        tool_name: MCP tool name
        tool_params: MCP tool parameters
        cli_command: OCI CLI command as fallback
        cli_timeout: Timeout for CLI command

    Returns:
        Result string from tool or CLI
    """
    # Try MCP tool first
    try:
        result = await tool_catalog.execute(tool_name, tool_params)
        result_str = _extract_result(result)

        # Check if it's an error response
        if result_str and not result_str.startswith("Error"):
            return result_str

        logger.warning(
            "MCP tool returned error, trying OCI CLI fallback",
            tool=tool_name,
            error=result_str[:100] if result_str else "empty",
        )
    except Exception as e:
        logger.warning(
            "MCP tool failed, trying OCI CLI fallback",
            tool=tool_name,
            error=str(e),
        )

    # Try OCI CLI fallback
    success, cli_result = await _run_oci_cli(cli_command, timeout=cli_timeout)
    if success:
        logger.info("OCI CLI fallback succeeded", tool=tool_name)
        return cli_result

    # Both failed
    return f"Error: Both MCP tool and OCI CLI failed. MCP tool: {tool_name}, CLI error: {cli_result}"


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


def _format_database_connections(raw_result: str) -> str:
    """
    Format database connection results as structured JSON.

    Returns JSON with type "database_connections" which the ResponseParser
    will convert to TableData, and SlackFormatter will render as native
    Slack table blocks.

    Args:
        raw_result: Raw result from oci_database_list_connections

    Returns:
        JSON string with type "database_connections" and connections list
    """

    def _detect_connection_type(conn_name: str) -> str:
        """Detect connection type from name pattern."""
        conn_upper = conn_name.upper()
        if "ATP" in conn_upper or "ADB" in conn_upper:
            return "Autonomous Transaction Processing"
        elif "ADW" in conn_upper:
            return "Autonomous Data Warehouse"
        elif "_HIGH" in conn_upper:
            return "High Priority Service"
        elif "_MEDIUM" in conn_upper:
            return "Medium Priority Service"
        elif "_LOW" in conn_upper:
            return "Low Priority Service"
        elif "_TP" in conn_upper:
            return "Transaction Processing"
        else:
            return "Database"

    try:
        # Try to parse as JSON
        data = json.loads(raw_result)

        if isinstance(data, dict):
            connections_str = data.get("connections", "")
            success = data.get("success", True)

            if not success:
                return json.dumps({
                    "type": "database_connections",
                    "error": "Failed to list connections",
                    "count": 0,
                    "connections": [],
                })

            # Parse comma-separated connection names
            if isinstance(connections_str, str) and connections_str:
                connection_names = [c.strip() for c in connections_str.split(",") if c.strip()]
            elif isinstance(connections_str, list):
                connection_names = connections_str
            else:
                connection_names = []

            if not connection_names:
                return json.dumps({
                    "type": "database_connections",
                    "count": 0,
                    "connections": [],
                    "message": "No database connections configured",
                })

            # Build structured connection list
            connections = []
            for name in connection_names:
                connections.append({
                    "name": name,
                    "connection_type": _detect_connection_type(name),
                })

            return json.dumps({
                "type": "database_connections",
                "count": len(connections),
                "connections": connections,
            })

        # If not a dict, return error
        return json.dumps({
            "type": "database_connections",
            "error": "Unexpected response format",
            "raw": str(raw_result)[:200],
        })

    except json.JSONDecodeError:
        # Not JSON, return error
        return json.dumps({
            "type": "database_connections",
            "error": "Invalid JSON response",
            "raw": str(raw_result)[:200],
        })
    except Exception as e:
        logger.warning("Failed to format database connections", error=str(e))
        return json.dumps({
            "type": "database_connections",
            "error": str(e),
        })


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
                # Match OCID pattern (exclude backticks, quotes, whitespace, brackets)
                # The result may be markdown-formatted with backticks like `ocid1.xxx`
                ocid_match = re.search(r"(ocid1\.compartment\.[^\s,\]\[\"'`|]+)", result_str)
                if ocid_match:
                    compartment_id = ocid_match.group(1)
                    # Clean any trailing backticks or special chars that might have snuck in
                    compartment_id = compartment_id.rstrip("`|")
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

    Fast workflow that directly calls the MCP tool with OCI CLI fallback.
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

        params = {
            "compartment_id": compartment_id,
            "include_subtree": include_subtree,
            "limit": 100,
            "format": "json",
        }

        # Build OCI CLI fallback command
        cli_command = ["oci", "iam", "compartment", "list", "--output", "json"]
        if compartment_id:
            cli_command.extend(["--compartment-id", compartment_id])
        if include_subtree:
            cli_command.append("--compartment-id-in-subtree=true")

        return await _execute_with_cli_fallback(
            tool_catalog=tool_catalog,
            tool_name="oci_list_compartments",
            tool_params=params,
            cli_command=cli_command,
            cli_timeout=60,
        )

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

        # Check if user asked for specific state, otherwise list all
        lifecycle_state = entities.get("lifecycle_state")

        # Detect state from query if not in entities
        if not lifecycle_state and query:
            query_lower = query.lower()
            if "running" in query_lower:
                lifecycle_state = "RUNNING"
            elif "stopped" in query_lower:
                lifecycle_state = "STOPPED"
            elif "terminated" in query_lower:
                lifecycle_state = "TERMINATED"
            # If no state mentioned, list all instances

        logger.info(
            "Executing oci_compute_list_instances",
            compartment_id=compartment_id,
            compartment_id_type=type(compartment_id).__name__,
            compartment_id_len=len(compartment_id) if compartment_id else 0,
            lifecycle_state=lifecycle_state,
        )

        params = {
            "compartment_id": compartment_id,
            "limit": 50,
            "format": "json",  # JSON format for Slack table rendering
        }
        if lifecycle_state:
            params["lifecycle_state"] = lifecycle_state

        # Build OCI CLI fallback command
        cli_command = [
            "oci", "compute", "instance", "list",
            "--compartment-id", compartment_id,
            "--output", "json",
        ]
        if lifecycle_state:
            cli_command.extend(["--lifecycle-state", lifecycle_state])

        # Execute with CLI fallback
        result_str = await _execute_with_cli_fallback(
            tool_catalog=tool_catalog,
            tool_name="oci_compute_list_instances",
            tool_params=params,
            cli_command=cli_command,
            cli_timeout=60,
        )

        logger.debug("Tool execution completed", result_preview=result_str[:100] if result_str else None)

        # If result is empty or [], provide helpful message
        if result_str in ("[]", "", "null"):
            state_msg = f" with state {lifecycle_state}" if lifecycle_state else ""
            return f"No instances found{state_msg} in the specified compartment."

        return result_str

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

        # Build tool parameters
        tool_params = {
            "compartment_id": compartment_id,  # None defaults to tenancy in tool
            "days": days,
        }

        # Support explicit date ranges for historical queries (e.g., "November costs")
        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        if start_date and end_date:
            tool_params["start_date"] = start_date
            tool_params["end_date"] = end_date
            logger.info(
                "cost_summary_workflow using explicit dates",
                start_date=start_date,
                end_date=end_date,
            )

        result = await tool_catalog.execute("oci_cost_get_summary", tool_params)

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

        # Build tool parameters
        tool_params = {
            "compartment_id": compartment_id,
            "days": days,
            "service_filter": "database",  # Filter for database services only
        }

        # Support explicit date ranges
        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        if start_date and end_date:
            tool_params["start_date"] = start_date
            tool_params["end_date"] = end_date

        result = await tool_catalog.execute("oci_cost_get_summary", tool_params)

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

        # Build tool parameters
        tool_params = {
            "compartment_id": compartment_id,
            "days": days,
            "service_filter": "compute",
        }

        # Support explicit date ranges
        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        if start_date and end_date:
            tool_params["start_date"] = start_date
            tool_params["end_date"] = end_date

        result = await tool_catalog.execute("oci_cost_get_summary", tool_params)

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

        # Build tool parameters
        tool_params = {
            "compartment_id": compartment_id,
            "days": days,
            "service_filter": "storage",
        }

        # Support explicit date ranges
        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        if start_date and end_date:
            tool_params["start_date"] = start_date
            tool_params["end_date"] = end_date

        result = await tool_catalog.execute("oci_cost_get_summary", tool_params)

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
    Runs data sources in PARALLEL for fast response, prioritizing results.

    Features:
    - 5-minute cache to avoid repeated OPSI/API calls
    - Parallel execution of multiple data sources
    - Graceful fallback when some sources fail

    Matches intents: list_databases, show_databases, database_names,
                     list_db, show_db, get_databases
    """
    import asyncio
    from datetime import timedelta

    # Cache configuration
    CACHE_TTL = timedelta(minutes=5)
    CACHE_PREFIX = "workflow:list_databases"

    # Timeout for each tool call (reduced from 30s to 15s for fail-fast)
    TOOL_TIMEOUT = 15

    async def safe_execute(tool_name: str, params: dict) -> tuple[str, str | None]:
        """Execute tool with timeout and error handling. Returns (source_name, result)."""
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
                return (tool_name, result_str)
        except asyncio.TimeoutError:
            logger.warning(f"Tool {tool_name} timed out after {TOOL_TIMEOUT}s")
        except Exception as e:
            logger.debug(f"Tool {tool_name} failed", error=str(e))
        return (tool_name, None)

    try:
        # Resolve compartment name to OCID
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        compartment_id = await _resolve_compartment(compartment_input, tool_catalog, query)

        # Fall back to root compartment if nothing resolved
        if not compartment_id:
            compartment_id = _get_root_compartment()
            if compartment_id:
                logger.info("list_databases: using root compartment as default")

        # ── Cache Check ──────────────────────────────────────────────────────
        # Build cache key from compartment (or "root" if none)
        cache_key = f"{CACHE_PREFIX}:{compartment_id or 'root'}"

        # Try to get cached result
        try:
            cached_result = await memory.cache.get(cache_key)
            if cached_result:
                logger.info(
                    "list_databases: returning cached result",
                    cache_key=cache_key,
                    ttl_minutes=5,
                )
                return cached_result
        except Exception as cache_err:
            # Cache failure should not block the workflow
            logger.debug("Cache read failed", error=str(cache_err))

        # ── Execute Database Queries ─────────────────────────────────────────
        # Prepare all tool calls
        tasks = []

        # OPSI search_databases (fastest - uses cache)
        opsi_params = {"limit": 50}
        if compartment_id:
            opsi_params["compartment_id"] = compartment_id
        tasks.append(safe_execute("oci_opsi_search_databases", opsi_params))

        # Autonomous databases (requires compartment_id)
        if compartment_id:
            tasks.append(safe_execute(
                "oci_database_list_autonomous",
                {"compartment_id": compartment_id}
            ))

        # Database connections (database-observatory)
        tasks.append(safe_execute("oci_database_list_connections", {}))

        # Run all tools in PARALLEL - significantly faster than sequential
        logger.info(f"list_databases: running {len(tasks)} tools in parallel")
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results with priority ordering
        results = []
        errors = []
        source_priority = {
            "oci_opsi_search_databases": ("Databases (from OPSI)", 1),
            "oci_database_list_autonomous": ("Autonomous Databases", 2),
            "oci_database_list_connections": ("Database Connections", 3),
        }

        for item in results_raw:
            if isinstance(item, Exception):
                errors.append(str(item))
                continue
            tool_name, result_str = item
            if result_str:
                source_name, priority = source_priority.get(tool_name, (tool_name, 99))
                if tool_name == "oci_database_list_connections":
                    # Returns JSON - don't add markdown header, parser handles formatting
                    result_str = _format_database_connections(result_str)
                    results.append((priority, result_str, True))  # True = is_json
                else:
                    results.append((priority, f"## {source_name}\n{result_str}", False))
            else:
                source_name, _ = source_priority.get(tool_name, (tool_name, 99))
                errors.append(f"{source_name} unavailable")

        if results:
            # Sort by priority and combine results
            results.sort(key=lambda x: x[0])

            # If only one result and it's JSON, return it directly for parser
            # This enables proper Slack table rendering via ResponseParser
            if len(results) == 1 and len(results[0]) == 3 and results[0][2]:
                final_result = results[0][1]  # Return JSON directly
            else:
                # Multiple results - combine as markdown
                final_result = "\n\n".join(r[1] for r in results)

            # ── Cache Storage ────────────────────────────────────────────────
            # Cache the successful result for 5 minutes
            try:
                await memory.cache.set(cache_key, final_result, ttl=CACHE_TTL)
                logger.debug(
                    "list_databases: cached result",
                    cache_key=cache_key,
                    ttl_minutes=5,
                    result_length=len(final_result),
                )
            except Exception as cache_err:
                # Cache failure should not block returning results
                logger.debug("Cache write failed", error=str(cache_err))

            return final_result
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
        # Extract search query from entities or use the original query
        search_query = (
            entities.get("name_pattern")
            or entities.get("name")
            or entities.get("query")
            or query  # Fallback to original user query
        )
        compartment_name = entities.get("compartment_name") or entities.get("compartment")

        # Build params - only include non-None values
        params: dict[str, Any] = {"query": search_query}
        if resource_type:
            params["resource_type"] = resource_type
        if compartment_name:
            params["compartment_name"] = compartment_name

        result = await tool_catalog.execute("oci_discovery_search", params)
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
                        results.append("### Top Services by Spend")
                        # Markdown table format
                        results.append("| Service | Cost | % |")
                        results.append("|---------|------|---|")
                        for svc in services[:10]:  # Show top 10
                            results.append(f"| {svc['service']} | {svc['cost']} | {svc['percent']} |")
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
# DB Management Workflows (AWR, SQL Performance)
# ─────────────────────────────────────────────────────────────────────────────


async def db_fleet_health_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get fleet-wide database health summary.

    Shows health status across all managed databases in the tenancy.

    Matches intents: db_fleet_health, fleet_health, database_fleet_status,
                     all_db_health, managed_database_health
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        result = await tool_catalog.execute(
            "oci_dbmgmt_get_fleet_health",
            {"compartment_id": compartment_id, "include_subtree": True},
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("db_fleet_health workflow failed", error=str(e))
        return f"Error getting database fleet health: {e}"


async def db_top_sql_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get top SQL statements by CPU activity for a database.

    Shows the most resource-intensive SQL statements currently running.

    Matches intents: top_sql, db_top_sql, high_cpu_sql, expensive_queries,
                     sql_performance, sql_cpu_usage
    """
    try:
        managed_database_id = entities.get("managed_database_id") or entities.get("database_id")
        if not managed_database_id:
            return "Error: Please provide a managed_database_id or database_id to analyze SQL performance."

        hours_back = entities.get("hours_back", 1)
        limit = entities.get("limit", 10)

        result = await tool_catalog.execute(
            "oci_dbmgmt_get_top_sql",
            {
                "managed_database_id": managed_database_id,
                "hours_back": hours_back,
                "limit": limit,
            },
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("db_top_sql workflow failed", error=str(e))
        return f"Error getting top SQL: {e}"


async def db_wait_events_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get top wait events from AWR data for a database.

    Shows what the database is waiting on most - useful for performance tuning.

    Matches intents: wait_events, db_wait_events, awr_wait_events,
                     database_waits, performance_bottlenecks
    """
    try:
        managed_database_id = entities.get("managed_database_id") or entities.get("database_id")
        if not managed_database_id:
            return "Error: Please provide a managed_database_id or database_id to analyze wait events."

        hours_back = entities.get("hours_back", 1)
        top_n = entities.get("top_n", 10)

        result = await tool_catalog.execute(
            "oci_dbmgmt_get_wait_events",
            {
                "managed_database_id": managed_database_id,
                "hours_back": hours_back,
                "top_n": top_n,
            },
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("db_wait_events workflow failed", error=str(e))
        return f"Error getting wait events: {e}"


async def db_awr_report_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Generate AWR or ASH report for a managed database.

    Creates a detailed performance report for the specified time period.

    Matches intents: awr_report, generate_awr, ash_report, performance_report,
                     db_performance_report, database_report
    """
    try:
        managed_database_id = entities.get("managed_database_id") or entities.get("database_id")
        if not managed_database_id:
            return "Error: Please provide a managed_database_id or database_id to generate an AWR report."

        hours_back = entities.get("hours_back", 24)
        report_type = entities.get("report_type", "AWR").upper()
        report_format = entities.get("report_format", "TEXT").upper()

        if report_type not in ("AWR", "ASH"):
            report_type = "AWR"
        if report_format not in ("HTML", "TEXT"):
            report_format = "TEXT"

        result = await tool_catalog.execute(
            "oci_dbmgmt_get_awr_report",
            {
                "managed_database_id": managed_database_id,
                "hours_back": hours_back,
                "report_type": report_type,
                "report_format": report_format,
            },
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("db_awr_report workflow failed", error=str(e))
        return f"Error generating AWR report: {e}"


async def db_sql_plan_baselines_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List SQL Plan Baselines for a database.

    Shows SQL statements with captured execution plans for plan stability.

    Matches intents: sql_plan_baselines, db_baselines, execution_plans,
                     plan_stability, sql_plans
    """
    try:
        managed_database_id = entities.get("managed_database_id") or entities.get("database_id")
        if not managed_database_id:
            return "Error: Please provide a managed_database_id or database_id to list SQL Plan Baselines."

        limit = entities.get("limit", 50)

        result = await tool_catalog.execute(
            "oci_dbmgmt_list_sql_plan_baselines",
            {
                "managed_database_id": managed_database_id,
                "limit": limit,
            },
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("db_sql_plan_baselines workflow failed", error=str(e))
        return f"Error listing SQL Plan Baselines: {e}"


async def db_managed_databases_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List managed databases in Database Management service.

    Shows all databases registered with the DB Management service.

    Matches intents: managed_databases, list_managed_databases, dbmgmt_databases,
                     db_management_list, registered_databases
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        database_type = entities.get("database_type")  # EXTERNAL_SIDB, EXTERNAL_RAC, etc.
        deployment_type = entities.get("deployment_type")  # ONPREMISE, BM, etc.

        params = {
            "compartment_id": compartment_id,
            "include_subtree": True,
        }
        if database_type:
            params["database_type"] = database_type
        if deployment_type:
            params["deployment_type"] = deployment_type

        result = await tool_catalog.execute("oci_dbmgmt_list_databases", params)
        return _extract_result(result)

    except Exception as e:
        logger.error("db_managed_databases workflow failed", error=str(e))
        return f"Error listing managed databases: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Operations Insights (OPSI) Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def opsi_database_insights_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List database insights from Operations Insights.

    Shows all databases registered with OPSI for monitoring.

    Matches intents: database_insights, opsi_databases, opsi_list,
                     operations_insights_databases, monitored_databases
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        result = await tool_catalog.execute(
            "oci_opsi_list_database_insights",
            {"compartment_id": compartment_id, "include_subtree": True},
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("opsi_database_insights workflow failed", error=str(e))
        return f"Error listing database insights: {e}"


async def opsi_resource_utilization_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get database resource utilization statistics from OPSI.

    Shows CPU, memory, and storage utilization across databases.

    Matches intents: opsi_utilization, db_resource_usage, database_utilization,
                     opsi_resource_stats, db_usage_metrics
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        days = entities.get("days", 7)
        resource_metric = entities.get("resource_metric", "CPU")  # CPU, STORAGE, IO, MEMORY

        result = await tool_catalog.execute(
            "oci_opsi_summarize_resource_stats",
            {
                "compartment_id": compartment_id,
                "resource_metric": resource_metric,
                "days": days,
            },
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("opsi_resource_utilization workflow failed", error=str(e))
        return f"Error getting resource utilization: {e}"


async def opsi_sql_insights_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get SQL performance insights from OPSI.

    Shows SQL performance patterns, degradations, and anomalies.

    Matches intents: sql_insights, opsi_sql, sql_performance_insights,
                     sql_analysis, sql_patterns
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        database_id = entities.get("database_id")
        days = entities.get("days", 7)

        params = {
            "compartment_id": compartment_id,
            "days": days,
        }
        if database_id:
            params["database_id"] = database_id

        result = await tool_catalog.execute("oci_opsi_summarize_sql_insights", params)
        return _extract_result(result)

    except Exception as e:
        logger.error("opsi_sql_insights workflow failed", error=str(e))
        return f"Error getting SQL insights: {e}"


async def opsi_addm_findings_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get ADDM findings from Operations Insights.

    Shows Automatic Database Diagnostic Monitor findings for performance issues.

    Matches intents: addm_findings, opsi_addm, database_diagnostics,
                     performance_findings, db_issues, addm_analysis
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        database_id = entities.get("database_id")
        days = entities.get("days", 7)
        finding_type = entities.get("finding_type")  # PERFORMANCE, CONFIGURATION, etc.

        params = {
            "compartment_id": compartment_id,
            "days": days,
        }
        if database_id:
            params["database_id"] = database_id
        if finding_type:
            params["finding_type"] = finding_type

        result = await tool_catalog.execute("oci_opsi_get_addm_findings", params)
        return _extract_result(result)

    except Exception as e:
        logger.error("opsi_addm_findings workflow failed", error=str(e))
        return f"Error getting ADDM findings: {e}"


async def opsi_addm_recommendations_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get ADDM recommendations from Operations Insights.

    Shows actionable recommendations to improve database performance.

    Matches intents: addm_recommendations, opsi_recommendations, db_recommendations,
                     performance_recommendations, optimization_suggestions
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        database_id = entities.get("database_id")
        days = entities.get("days", 7)

        params = {
            "compartment_id": compartment_id,
            "days": days,
        }
        if database_id:
            params["database_id"] = database_id

        result = await tool_catalog.execute("oci_opsi_get_addm_recommendations", params)
        return _extract_result(result)

    except Exception as e:
        logger.error("opsi_addm_recommendations workflow failed", error=str(e))
        return f"Error getting ADDM recommendations: {e}"


async def opsi_capacity_forecast_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get capacity forecast from Operations Insights.

    Projects future resource usage based on historical trends.

    Matches intents: capacity_forecast, opsi_forecast, resource_forecast,
                     db_capacity_planning, usage_projection, growth_forecast
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        database_id = entities.get("database_id")
        resource_metric = entities.get("resource_metric", "CPU")  # CPU, STORAGE, IO, MEMORY
        forecast_days = entities.get("forecast_days", 30)
        analysis_days = entities.get("analysis_days", 30)

        params = {
            "compartment_id": compartment_id,
            "resource_metric": resource_metric,
            "forecast_days": forecast_days,
            "analysis_days": analysis_days,
        }
        if database_id:
            params["database_id"] = database_id

        result = await tool_catalog.execute("oci_opsi_get_capacity_forecast", params)
        return _extract_result(result)

    except Exception as e:
        logger.error("opsi_capacity_forecast workflow failed", error=str(e))
        return f"Error getting capacity forecast: {e}"


async def opsi_capacity_trend_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get capacity utilization trend from Operations Insights.

    Shows historical resource usage patterns over time.

    Matches intents: capacity_trend, opsi_trend, utilization_trend,
                     resource_trend, usage_history, capacity_history
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        database_id = entities.get("database_id")
        resource_metric = entities.get("resource_metric", "CPU")
        days = entities.get("days", 30)

        params = {
            "compartment_id": compartment_id,
            "resource_metric": resource_metric,
            "days": days,
        }
        if database_id:
            params["database_id"] = database_id

        result = await tool_catalog.execute("oci_opsi_get_capacity_trend", params)
        return _extract_result(result)

    except Exception as e:
        logger.error("opsi_capacity_trend workflow failed", error=str(e))
        return f"Error getting capacity trend: {e}"


async def opsi_sql_statistics_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get detailed SQL statistics from Operations Insights.

    Shows comprehensive SQL execution metrics across databases.

    Matches intents: sql_statistics, opsi_sql_stats, sql_metrics,
                     query_statistics, sql_execution_stats
    """
    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        database_id = entities.get("database_id")
        sql_identifier = entities.get("sql_id")
        days = entities.get("days", 7)

        params = {
            "compartment_id": compartment_id,
            "days": days,
        }
        if database_id:
            params["database_id"] = database_id
        if sql_identifier:
            params["sql_identifier"] = sql_identifier

        result = await tool_catalog.execute("oci_opsi_summarize_sql_statistics", params)
        return _extract_result(result)

    except Exception as e:
        logger.error("opsi_sql_statistics workflow failed", error=str(e))
        return f"Error getting SQL statistics: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Combined Database Performance Workflow
# ─────────────────────────────────────────────────────────────────────────────


async def db_performance_overview_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get comprehensive database performance overview.

    Combines multiple data sources for a complete performance picture:
    - Fleet health summary
    - Top SQL by CPU
    - ADDM findings and recommendations
    - Capacity trends

    Matches intents: db_performance_overview, database_performance, db_health_check,
                     comprehensive_db_status, full_db_analysis
    """
    import asyncio

    try:
        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            compartment_id = _get_root_compartment()

        database_id = entities.get("database_id") or entities.get("managed_database_id")

        results = []
        results.append("# Database Performance Overview\n")

        async def get_fleet_health():
            try:
                result = await tool_catalog.execute(
                    "oci_dbmgmt_get_fleet_health",
                    {"compartment_id": compartment_id, "include_subtree": True},
                )
                return _extract_result(result)
            except Exception as e:
                logger.debug("Fleet health failed", error=str(e))
                return None

        async def get_addm_findings():
            try:
                params = {"compartment_id": compartment_id, "days": 7}
                if database_id:
                    params["database_id"] = database_id
                result = await tool_catalog.execute("oci_opsi_get_addm_findings", params)
                return _extract_result(result)
            except Exception as e:
                logger.debug("ADDM findings failed", error=str(e))
                return None

        async def get_recommendations():
            try:
                params = {"compartment_id": compartment_id, "days": 7}
                if database_id:
                    params["database_id"] = database_id
                result = await tool_catalog.execute("oci_opsi_get_addm_recommendations", params)
                return _extract_result(result)
            except Exception as e:
                logger.debug("ADDM recommendations failed", error=str(e))
                return None

        async def get_resource_stats():
            try:
                result = await tool_catalog.execute(
                    "oci_opsi_summarize_resource_stats",
                    {"compartment_id": compartment_id, "resource_metric": "CPU", "days": 7},
                )
                return _extract_result(result)
            except Exception as e:
                logger.debug("Resource stats failed", error=str(e))
                return None

        # Execute all in parallel
        fleet_health, addm_findings, recommendations, resource_stats = await asyncio.gather(
            get_fleet_health(),
            get_addm_findings(),
            get_recommendations(),
            get_resource_stats(),
        )

        # Add fleet health
        if fleet_health and "Error" not in fleet_health:
            results.append("## Fleet Health Status")
            results.append(fleet_health)
            results.append("")

        # Add resource utilization
        if resource_stats and "Error" not in resource_stats:
            results.append("## Resource Utilization (Last 7 Days)")
            results.append(resource_stats)
            results.append("")

        # Add ADDM findings
        if addm_findings and "Error" not in addm_findings:
            results.append("## ADDM Performance Findings")
            results.append(addm_findings)
            results.append("")

        # Add recommendations
        if recommendations and "Error" not in recommendations:
            results.append("## Optimization Recommendations")
            results.append(recommendations)
            results.append("")

        if len(results) == 1:  # Only header
            results.append(
                "No performance data available. Ensure databases are registered with "
                "DB Management and Operations Insights services."
            )

        return "\n".join(results)

    except Exception as e:
        logger.error("db_performance_overview workflow failed", error=str(e))
        return f"Error getting database performance overview: {e}"


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

    # DB Management - Fleet and Database Health
    "db_fleet_health": db_fleet_health_workflow,
    "fleet_health": db_fleet_health_workflow,
    "database_fleet_status": db_fleet_health_workflow,
    "all_db_health": db_fleet_health_workflow,
    "managed_database_health": db_fleet_health_workflow,
    "managed_databases": db_managed_databases_workflow,
    "list_managed_databases": db_managed_databases_workflow,
    "dbmgmt_databases": db_managed_databases_workflow,
    "db_management_list": db_managed_databases_workflow,

    # DB Management - SQL Performance
    "top_sql": db_top_sql_workflow,
    "db_top_sql": db_top_sql_workflow,
    "high_cpu_sql": db_top_sql_workflow,
    "expensive_queries": db_top_sql_workflow,
    "sql_performance": db_top_sql_workflow,
    "sql_cpu_usage": db_top_sql_workflow,

    # DB Management - Wait Events
    "wait_events": db_wait_events_workflow,
    "db_wait_events": db_wait_events_workflow,
    "awr_wait_events": db_wait_events_workflow,
    "database_waits": db_wait_events_workflow,
    "performance_bottlenecks": db_wait_events_workflow,

    # DB Management - AWR Reports
    "awr_report": db_awr_report_workflow,
    "generate_awr": db_awr_report_workflow,
    "ash_report": db_awr_report_workflow,
    "performance_report": db_awr_report_workflow,
    "db_performance_report": db_awr_report_workflow,

    # DB Management - SQL Plan Baselines
    "sql_plan_baselines": db_sql_plan_baselines_workflow,
    "db_baselines": db_sql_plan_baselines_workflow,
    "execution_plans": db_sql_plan_baselines_workflow,
    "plan_stability": db_sql_plan_baselines_workflow,

    # OPSI - Database Insights
    "database_insights": opsi_database_insights_workflow,
    "opsi_databases": opsi_database_insights_workflow,
    "opsi_list": opsi_database_insights_workflow,
    "operations_insights_databases": opsi_database_insights_workflow,
    "monitored_databases": opsi_database_insights_workflow,

    # OPSI - Resource Utilization
    "opsi_utilization": opsi_resource_utilization_workflow,
    "db_resource_usage": opsi_resource_utilization_workflow,
    "database_utilization": opsi_resource_utilization_workflow,
    "opsi_resource_stats": opsi_resource_utilization_workflow,
    "db_usage_metrics": opsi_resource_utilization_workflow,

    # OPSI - SQL Insights
    "sql_insights": opsi_sql_insights_workflow,
    "opsi_sql": opsi_sql_insights_workflow,
    "sql_performance_insights": opsi_sql_insights_workflow,
    "sql_analysis": opsi_sql_insights_workflow,
    "sql_patterns": opsi_sql_insights_workflow,

    # OPSI - SQL Statistics
    "sql_statistics": opsi_sql_statistics_workflow,
    "opsi_sql_stats": opsi_sql_statistics_workflow,
    "sql_metrics": opsi_sql_statistics_workflow,
    "query_statistics": opsi_sql_statistics_workflow,
    "sql_execution_stats": opsi_sql_statistics_workflow,

    # OPSI - ADDM Findings
    "addm_findings": opsi_addm_findings_workflow,
    "opsi_addm": opsi_addm_findings_workflow,
    "database_diagnostics": opsi_addm_findings_workflow,
    "performance_findings": opsi_addm_findings_workflow,
    "db_issues": opsi_addm_findings_workflow,
    "addm_analysis": opsi_addm_findings_workflow,

    # OPSI - ADDM Recommendations
    "addm_recommendations": opsi_addm_recommendations_workflow,
    "opsi_recommendations": opsi_addm_recommendations_workflow,
    "db_recommendations": opsi_addm_recommendations_workflow,
    "performance_recommendations": opsi_addm_recommendations_workflow,
    "optimization_suggestions": opsi_addm_recommendations_workflow,

    # OPSI - Capacity Planning
    "capacity_forecast": opsi_capacity_forecast_workflow,
    "opsi_forecast": opsi_capacity_forecast_workflow,
    "resource_forecast": opsi_capacity_forecast_workflow,
    "db_capacity_planning": opsi_capacity_forecast_workflow,
    "usage_projection": opsi_capacity_forecast_workflow,
    "growth_forecast": opsi_capacity_forecast_workflow,

    # OPSI - Capacity Trends
    "capacity_trend": opsi_capacity_trend_workflow,
    "opsi_trend": opsi_capacity_trend_workflow,
    "utilization_trend": opsi_capacity_trend_workflow,
    "resource_trend": opsi_capacity_trend_workflow,
    "usage_history": opsi_capacity_trend_workflow,
    "capacity_history": opsi_capacity_trend_workflow,

    # Combined Database Performance Overview
    "db_performance_overview": db_performance_overview_workflow,
    "database_performance": db_performance_overview_workflow,
    "db_health_check": db_performance_overview_workflow,
    "comprehensive_db_status": db_performance_overview_workflow,
    "full_db_analysis": db_performance_overview_workflow,
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
