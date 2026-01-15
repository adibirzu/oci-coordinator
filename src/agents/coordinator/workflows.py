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
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def _safe_str_value(value: Any, default: str = "") -> str:
    """
    Safely extract a string value from potentially nested dict structures.

    OCI APIs sometimes return nested dict structures for what should be simple
    string fields. This function handles these cases to prevent AttributeError
    when calling string methods like .upper() on dict objects.

    Args:
        value: The value to convert (string, dict, or None)
        default: Default value if extraction fails

    Returns:
        String value suitable for string operations

    Examples:
        >>> _safe_str_value("FINANCE")
        'FINANCE'
        >>> _safe_str_value({"name": "FINANCE"})
        'FINANCE'
        >>> _safe_str_value(None)
        ''
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Try common nested field names
        for key in ("name", "value", "display_name", "id"):
            nested = value.get(key)
            if nested and isinstance(nested, str):
                return nested
        # Fallback to string representation of dict
        return str(value)
    return str(value)


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
                check=False, capture_output=True,
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
    tool_catalog: ToolCatalog,
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


def _get_root_compartment(profile: str | None = None) -> str | None:
    """
    Get the root compartment (tenancy) ID from OCI config.

    Args:
        profile: OCI profile name (e.g., "DEFAULT", "EMDEMO"). If None, uses DEFAULT.

    Returns the tenancy OCID which can be used as default compartment
    when no specific compartment is provided.
    """
    try:
        import oci
        profile_name = (profile or "DEFAULT").upper()

        # Handle profile name variations
        if profile_name == "EMDEMO":
            profile_name = "emdemo"  # OCI config uses lowercase

        config = oci.config.from_file(profile_name=profile_name)
        return config.get("tenancy")
    except Exception as e:
        logger.debug("Could not get tenancy from OCI config", profile=profile, error=str(e))
        # Fall back to DEFAULT profile
        if profile:
            try:
                import oci
                config = oci.config.from_file()
                return config.get("tenancy")
            except Exception:
                pass
        return None


def _get_profile_compartment(
    profile: str | None,
    service: str = "default",
) -> tuple[str | None, str | None]:
    """
    Get profile-specific compartment and region for OCI services.

    Different profiles may have different compartments for different services:
    - EMDEMO: DB Management in us-ashburn-1, OPSI in uk-london-1, etc.
    - DEFAULT: Uses root tenancy by default

    Args:
        profile: Profile name (e.g., "EMDEMO", "DEFAULT")
        service: Service type - "dbmgmt", "opsi", "logan", "root"

    Returns:
        Tuple of (compartment_id, region) - either may be None to use defaults
    """
    import os

    profile_upper = (profile or "DEFAULT").upper()
    service_lower = service.lower()

    # Environment variable prefixes for known profiles
    env_prefix_map = {
        "EMDEMO": "OCI_EMDEMO",
        "DEFAULT": "OCI_DEFAULT",
        "PROD": "OCI_PROD",
        "DEV": "OCI_DEV",
    }

    prefix = env_prefix_map.get(profile_upper)
    if not prefix:
        return None, None

    # Service-specific compartment and region mapping
    service_map = {
        "dbmgmt": ("DBMGMT_COMPARTMENT_ID", "DBMGMT_REGION"),
        "opsi": ("OPSI_COMPARTMENT_ID", "OPSI_REGION"),
        "logan": ("LOGAN_COMPARTMENT_ID", "LOGAN_REGION"),
        "exadata": ("EXADATA_COMPARTMENT_ID", "EXADATA_REGION"),
        "oandm": ("OANDM_DEMO_COMPARTMENT_ID", None),  # OandM-Demo parent
        "root": ("TENANCY_ID", "HOME_REGION"),
    }

    compartment_suffix, region_suffix = service_map.get(service_lower, (None, None))

    compartment_id = None
    region = None

    if compartment_suffix:
        env_var = f"{prefix}_{compartment_suffix}"
        compartment_id = os.getenv(env_var)
        if compartment_id:
            logger.debug(
                "Using profile-specific compartment",
                profile=profile_upper,
                service=service_lower,
                env_var=env_var,
                compartment_id=compartment_id[:50] + "..." if len(compartment_id) > 50 else compartment_id,
            )

    if region_suffix:
        region = os.getenv(f"{prefix}_{region_suffix}")

    return compartment_id, region


def _auto_select_db_profile(profile: str | None) -> str | None:
    """
    Auto-select the best profile for database operations.

    If no profile is specified, checks which profiles have DB Management
    compartments configured and returns the best one.

    Priority:
    1. If profile is explicitly provided, use it
    2. If EMDEMO has DBMGMT config, use EMDEMO (typically has managed databases)
    3. If DEFAULT has DBMGMT config, use None (DEFAULT)
    4. Otherwise return None

    Args:
        profile: Explicitly provided profile name, or None

    Returns:
        Profile name to use (None means DEFAULT)
    """
    import os

    if profile:
        return profile

    default_dbmgmt = os.getenv("OCI_DEFAULT_DBMGMT_COMPARTMENT_ID")
    emdemo_dbmgmt = os.getenv("OCI_EMDEMO_DBMGMT_COMPARTMENT_ID")

    if emdemo_dbmgmt:
        # EMDEMO typically has more managed databases
        logger.debug("Auto-selecting EMDEMO profile for DB operations")
        return "EMDEMO"
    elif default_dbmgmt:
        logger.debug("Using DEFAULT profile for DB operations")
        return None

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
            # Check if this is the new rich format from MCP tool
            connections_raw = data.get("connections", [])
            default_connection = data.get("default_connection")
            sqlcl_available = data.get("sqlcl_available", False)
            tns_admin = data.get("tns_admin")

            # Handle both list of dicts (new format) and comma-separated string (legacy)
            if isinstance(connections_raw, list) and connections_raw:
                # Check if it's list of dicts (rich format) or list of strings
                if isinstance(connections_raw[0], dict):
                    # Rich format from MCP tool - use all available data
                    connections = []
                    for conn in connections_raw:
                        name = conn.get("name", "unknown")
                        conn_type = conn.get("type", "")
                        status = conn.get("status", "unknown")
                        description = conn.get("description", "")
                        is_default = conn.get("is_default", False)
                        is_fallback = conn.get("is_fallback", False)
                        user = conn.get("user", "")
                        tns_alias = conn.get("tns_alias", name)
                        database_name = conn.get("database_name", "")

                        # Determine display connection type
                        if conn_type == "sqlcl_tns":
                            connection_type = _detect_connection_type(name)
                        elif conn_type == "oracledb_wallet":
                            connection_type = "Autonomous Transaction Process"
                        elif conn_type == "sqlcl_cli":
                            connection_type = "SQLcl CLI"
                        else:
                            connection_type = _detect_connection_type(name)

                        # Build connection entry with all useful info
                        entry = {
                            "name": name,
                            "connection_type": connection_type,
                            "status": status,
                        }

                        # Add database name if available (from V$DATABASE query)
                        if database_name:
                            entry["database_name"] = database_name

                        # Add optional fields if present
                        if user:
                            entry["user"] = user
                        if is_default:
                            entry["is_default"] = True
                        if is_fallback:
                            entry["is_fallback"] = True

                        connections.append(entry)

                    result = {
                        "type": "database_connections",
                        "count": len([c for c in connections if c.get("status") in ("available", "configured")]),
                        "connections": connections,
                    }

                    if default_connection:
                        result["default_connection"] = default_connection
                    if tns_admin:
                        result["tns_admin"] = tns_admin

                    return json.dumps(result)

                else:
                    # List of strings - convert to dicts
                    connection_names = [str(c) for c in connections_raw if c]
            elif isinstance(connections_raw, str) and connections_raw:
                # Legacy comma-separated string
                connection_names = [c.strip() for c in connections_raw.split(",") if c.strip()]
            else:
                connection_names = []

            # Handle legacy/simple format
            if not connection_names:
                return json.dumps({
                    "type": "database_connections",
                    "count": 0,
                    "connections": [],
                    "message": "No database connections configured",
                })

            # Build structured connection list from names only
            connections = []
            for name in connection_names:
                name_str = _safe_str_value(name) if isinstance(name, dict) else str(name)
                connections.append({
                    "name": name_str,
                    "connection_type": _detect_connection_type(name_str),
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


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Context Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Thread-local storage for last used compartment (simple in-memory fallback)
_last_compartment_cache: dict[str, tuple[str, str]] = {}  # thread_id -> (name, ocid)


async def _store_last_compartment(
    memory: SharedMemoryManager,
    thread_id: str,
    compartment_name: str | None,
    compartment_id: str,
) -> None:
    """Store the last used compartment in conversation context."""
    try:
        # Store in memory cache for quick retrieval
        _last_compartment_cache[thread_id] = (compartment_name or "unknown", compartment_id)

        # Also try to store in the shared memory manager if available
        if memory and hasattr(memory, "set_session_state"):
            await memory.set_session_state(
                thread_id,
                {
                    "last_compartment_name": compartment_name,
                    "last_compartment_id": compartment_id,
                },
            )
        logger.debug("Stored last compartment in context",
                    thread_id=thread_id,
                    compartment_name=compartment_name,
                    compartment_id=compartment_id[:30] if compartment_id else None)
    except Exception as e:
        logger.debug("Failed to store compartment context", error=str(e))


async def _get_last_compartment(
    memory: SharedMemoryManager,
    thread_id: str | None,
) -> tuple[str | None, str | None]:
    """Get the last used compartment from conversation context.

    Returns:
        Tuple of (compartment_name, compartment_id) or (None, None) if not found
    """
    # If no thread_id, can't look up context
    if not thread_id:
        logger.debug("No thread_id provided for context lookup")
        return None, None

    logger.debug("Looking up compartment context",
                thread_id=thread_id,
                cache_keys=list(_last_compartment_cache.keys())[:5],
                cache_size=len(_last_compartment_cache))

    try:
        # Try local cache first
        if thread_id in _last_compartment_cache:
            name, ocid = _last_compartment_cache[thread_id]
            logger.debug("Retrieved last compartment from cache", thread_id=thread_id, name=name)
            return name, ocid

        # Try shared memory manager
        if memory and hasattr(memory, "get_session_state"):
            state = await memory.get_session_state(thread_id)
            if state:
                name = state.get("last_compartment_name")
                ocid = state.get("last_compartment_id")
                if ocid:
                    logger.debug("Retrieved last compartment from memory", thread_id=thread_id, name=name)
                    return name, ocid

    except Exception as e:
        logger.debug("Failed to get compartment context", error=str(e))

    return None, None


async def _resolve_compartment(
    name_or_id: str | None,
    tool_catalog: ToolCatalog,
    query: str | None = None,
    skip_root_fallback: bool = False,
) -> str | None:
    """
    Resolve a compartment name or ID to an OCID.

    Args:
        name_or_id: Compartment name, partial name, or OCID
        tool_catalog: Tool catalog for executing MCP tools
        query: Original query to extract compartment name from if name_or_id is None
        skip_root_fallback: If True, return None instead of falling back to root compartment.
                           Use this when the caller wants to try conversation context first.

    Returns:
        Compartment OCID if found, root compartment as fallback (unless skip_root_fallback), or None
    """
    # If already an OCID, return as-is
    if name_or_id and name_or_id.startswith("ocid1."):
        return name_or_id

    # Try to extract compartment name from query if not provided
    search_name = name_or_id
    if not search_name and query:
        # Look for common patterns - support multiple variations:
        # - "in/from compartment NAME"
        # - "NAME compartment"
        # - "compartment NAME"
        # - "from NAME with status"
        import re
        patterns = [
            r"(?:in|from|for)\s+compartment\s+(\w+(?:[-_]\w+)*)",  # "in/from compartment NAME"
            r"(\w+(?:[-_]\w+)*)\s+compartment",  # "NAME compartment"
            r"compartment\s+(\w+(?:[-_]\w+)*)",  # "compartment NAME"
            r"(?:in|from)\s+(\w+(?:[-_]\w+)*)\s+(?:compartment|with)",  # "from NAME with..."
            r"(?:in|from)\s+(\w+(?:[-_]\w+)*)(?:\s|$)",  # "from NAME" at end
        ]
        skip_words = {
            # Articles and pronouns
            "the", "a", "an", "all", "my", "our", "this", "that",
            # Common query verbs (these are NOT compartment names)
            "list", "show", "get", "display", "find", "search", "check", "describe",
            "what", "which", "where", "how", "view", "fetch",
            # Status words
            "current", "status", "instances", "running", "stopped", "available",
            # Common nouns that might appear before "compartment"
            "root", "parent", "child", "default", "main", "home",
        }
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidate = match.group(1)
                if candidate.lower() not in skip_words:
                    search_name = candidate
                    logger.debug("Extracted compartment name from query", name=search_name, pattern=pattern)
                    break

    # If we have a name to search, try to find it
    if search_name:
        try:
            logger.debug("Searching for compartment by name", search_name=search_name)
            # Search for compartments matching the name
            result = await tool_catalog.execute(
                "oci_search_compartments",
                {"query": search_name, "limit": 5},
            )
            result_str = _extract_result(result)
            logger.debug("Compartment search result preview",
                        search_name=search_name,
                        result_preview=result_str[:200] if result_str else None,
                        has_ocid="ocid1.compartment" in result_str if result_str else False)

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
                else:
                    logger.warning("Compartment OCID found in result but regex failed to extract",
                                  search_name=search_name)
            else:
                logger.warning("Compartment search returned no OCID matches",
                              search_name=search_name,
                              result_preview=result_str[:100] if result_str else "empty")

            # Try JSON parsing if it's structured
            try:
                data = json.loads(result_str)
                if isinstance(data, list) and data:
                    # Return first matching compartment
                    comp_id = data[0].get("id") or data[0].get("ocid")
                    if comp_id:
                        logger.info("Resolved compartment from JSON list", name=search_name, ocid=comp_id[:50])
                    return comp_id
                elif isinstance(data, dict):
                    comp_id = data.get("id") or data.get("ocid")
                    if comp_id:
                        logger.info("Resolved compartment from JSON dict", name=search_name, ocid=comp_id[:50])
                    return comp_id
            except (json.JSONDecodeError, TypeError):
                pass

        except Exception as e:
            logger.warning("Compartment search failed", name=search_name, error=str(e))

    # Skip root fallback if caller wants to try conversation context first
    if skip_root_fallback:
        logger.debug("Skipping root compartment fallback (caller will try context)")
        return None

    # Fall back to root compartment
    root = _get_root_compartment()
    if root:
        logger.debug("Using root compartment as fallback", ocid=root[:50])
    return root


async def _resolve_managed_database(
    name_or_id: str | None,
    tool_catalog: ToolCatalog,
    compartment_id: str | None = None,
    profile: str | None = None,
) -> str | None:
    """
    Resolve a database name to a managed_database_id.

    Args:
        name_or_id: Database name or managed_database_id (OCID)
        tool_catalog: Tool catalog for executing MCP tools
        compartment_id: Compartment to search in (defaults to profile's dbmgmt compartment)
        profile: OCI profile to use (auto-selected if not provided)

    Returns:
        Managed database OCID if found, or None
    """
    if not name_or_id:
        return None

    # If already an OCID, return as-is
    if name_or_id.startswith("ocid1."):
        return name_or_id

    # Auto-select profile for DB operations
    profile = _auto_select_db_profile(profile)

    # Get profile-specific compartment and region for DB Management
    dbmgmt_compartment, dbmgmt_region = _get_profile_compartment(profile, "dbmgmt")

    # Use profile-specific compartment, or fall back to provided, then root
    search_compartment = dbmgmt_compartment or compartment_id or _get_root_compartment(profile)
    if not search_compartment:
        logger.warning("No compartment for database search")
        return None

    name_upper = name_or_id.upper()

    # Helper function to search databases list for a name match
    def _search_databases(databases: list[dict], source: str) -> str | None:
        for db in databases:
            # Safely extract database name - handle nested dicts and None values
            raw_name = db.get("name") or db.get("database_name") or db.get("display_name") or ""
            # Handle case where value is a dict (nested structure from API)
            if isinstance(raw_name, dict):
                raw_name = raw_name.get("name") or raw_name.get("value") or str(raw_name)
            db_name = str(raw_name).upper() if raw_name else ""
            if db_name == name_upper or name_upper in db_name:
                db_id = db.get("id") or db.get("managed_database_id")
                if db_id:
                    logger.info(f"Resolved database name to ID ({source})",
                               name=name_or_id, db_id=db_id[:50])
                    return db_id
        return None

    # ── First: Search Managed Databases ────────────────────────────────────
    try:
        search_params: dict[str, Any] = {
            "compartment_id": search_compartment,
            "include_subtree": True,
        }
        if profile:
            search_params["profile"] = profile
        if dbmgmt_region:
            search_params["region"] = dbmgmt_region

        result = await tool_catalog.execute(
            "oci_dbmgmt_list_databases",
            search_params,
        )
        result_str = _extract_result(result)

        if result_str:
            try:
                data = json.loads(result_str)
                databases = []
                if isinstance(data, dict):
                    databases = data.get("databases", data.get("items", []))
                elif isinstance(data, list):
                    databases = data

                db_id = _search_databases(databases, "managed")
                if db_id:
                    return db_id
            except (json.JSONDecodeError, TypeError):
                # Try regex extraction from markdown/text output
                import re
                pattern = rf"{re.escape(name_or_id)}.*?(ocid1\.manageddatabase\.[^\s,\]\[\"'`|]+)"
                match = re.search(pattern, result_str, re.IGNORECASE | re.DOTALL)
                if match:
                    db_id = match.group(1).rstrip("`|")
                    logger.info("Resolved database name to ID (regex)",
                               name=name_or_id, db_id=db_id[:50])
                    return db_id

    except Exception as e:
        logger.debug("Managed database search failed", name=name_or_id, error=str(e))

    # ── Second: Search Autonomous Databases (fallback) ─────────────────────
    # If not found in managed databases, try autonomous databases
    try:
        result = await tool_catalog.execute(
            "oci_db_list_autonomous",
            {"compartment_id": search_compartment},
        )
        result_str = _extract_result(result)

        if result_str:
            try:
                data = json.loads(result_str)
                databases = []
                if isinstance(data, dict):
                    databases = data.get("databases", data.get("items", []))
                elif isinstance(data, list):
                    databases = data

                db_id = _search_databases(databases, "autonomous")
                if db_id:
                    return db_id
            except (json.JSONDecodeError, TypeError):
                # Try regex extraction for autonomous database OCIDs
                import re
                pattern = rf"{re.escape(name_or_id)}.*?(ocid1\.autonomousdatabase\.[^\s,\]\[\"'`|]+)"
                match = re.search(pattern, result_str, re.IGNORECASE | re.DOTALL)
                if match:
                    db_id = match.group(1).rstrip("`|")
                    logger.info("Resolved database name to ID (autonomous regex)",
                               name=name_or_id, db_id=db_id[:50])
                    return db_id

    except Exception as e:
        logger.debug("Autonomous database search failed", name=name_or_id, error=str(e))

    logger.warning("Database name resolution failed - not found", name=name_or_id)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Identity Workflows
# ─────────────────────────────────────────────────────────────────────────────


def _extract_compartment_filter(query: str) -> dict[str, Any]:
    """
    Extract compartment filtering criteria from query.

    Supports:
    - "starting with B" / "start with B" / "that start with B"
    - "containing finance" / "with finance in name"
    - "named production" / "called prod"
    - "in state ACTIVE" / "with state DELETED"

    Returns dict with filter type and value.
    """
    import re
    query_lower = query.lower()
    filters: dict[str, Any] = {}

    # Pattern: starting with / start with / that start with
    prefix_patterns = [
        r"(?:that\s+)?start(?:s|ing)?\s+with\s+(?:letter\s+)?['\"]?([a-zA-Z])['\"]?",
        r"beginning\s+with\s+(?:letter\s+)?['\"]?([a-zA-Z])['\"]?",
        r"with\s+(?:letter|prefix)\s+['\"]?([a-zA-Z])['\"]?",
    ]
    for pattern in prefix_patterns:
        match = re.search(pattern, query_lower)
        if match:
            filters["starts_with"] = match.group(1).upper()
            break

    # Pattern: containing / with ... in name
    contains_patterns = [
        r"contain(?:s|ing)?\s+['\"]?([a-zA-Z0-9_-]+)['\"]?",
        r"with\s+['\"]?([a-zA-Z0-9_-]+)['\"]?\s+in\s+(?:the\s+)?name",
        r"that\s+have\s+['\"]?([a-zA-Z0-9_-]+)['\"]?\s+in",
    ]
    for pattern in contains_patterns:
        match = re.search(pattern, query_lower)
        if match and "starts_with" not in filters:
            filters["contains"] = match.group(1).upper()
            break

    # Pattern: named / called
    name_patterns = [
        r"named\s+['\"]?([a-zA-Z0-9_-]+)['\"]?",
        r"called\s+['\"]?([a-zA-Z0-9_-]+)['\"]?",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, query_lower)
        if match:
            filters["exact_name"] = match.group(1).upper()
            break

    # Pattern: state filter
    state_pattern = r"(?:in\s+)?state\s+['\"]?(ACTIVE|DELETED|CREATING|DELETING)['\"]?"
    match = re.search(state_pattern, query, re.IGNORECASE)
    if match:
        filters["state"] = match.group(1).upper()

    return filters


def _fuzzy_match_compartment(name: str, compartments: list[dict], threshold: float = 0.6) -> list[dict]:
    """
    Find compartments with similar names using fuzzy matching.

    Returns list of similar compartments with match scores.
    """
    import difflib
    name_upper = name.upper()
    matches = []

    for comp in compartments:
        # Safely extract name - OCI API can return nested dicts
        comp_name_raw = _safe_str_value(comp.get("name"), "")
        comp_name = comp_name_raw.upper()
        if not comp_name:
            continue

        # Calculate similarity ratio
        ratio = difflib.SequenceMatcher(None, name_upper, comp_name).ratio()

        # Also check for partial matches
        if name_upper in comp_name or comp_name in name_upper:
            ratio = max(ratio, 0.8)

        if ratio >= threshold:
            matches.append({
                "name": comp_name_raw,  # Use the extracted string value
                "id": _safe_str_value(comp.get("id"), ""),
                "state": _safe_str_value(
                    comp.get("lifecycle-state", comp.get("lifecycle_state")), "ACTIVE"
                ),
                "score": ratio,
            })

    # Sort by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:5]  # Return top 5 matches


async def list_compartments_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List compartments in the tenancy with filtering and OCID mapping.

    Features:
    - Filter by name prefix (e.g., "starting with B")
    - Filter by name contains (e.g., "containing finance")
    - Filter by exact name
    - Filter by lifecycle state
    - Fuzzy name matching for suggestions
    - OCID to name mapping in output
    - Paginated output for large results (25 per page)
    - Cache with daily refresh support

    Matches intents: list_compartments, show_compartments, get_compartments
    """
    try:
        import json

        # Extract filter criteria from query
        filters = _extract_compartment_filter(query)

        # Check for force refresh flag
        force_refresh = entities.get("force_refresh", False) or "latest" in query.lower() or "refresh" in query.lower()

        # Resolve compartment - default to root for listing
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        compartment_id = await _resolve_compartment(compartment_input, tool_catalog, query)

        # Use root compartment if nothing resolved
        if not compartment_id:
            compartment_id = _get_root_compartment()

        include_subtree = entities.get("include_subtree", True)

        # Build cache key
        cache_key = f"compartments:{compartment_id or 'root'}:{json.dumps(filters, sort_keys=True)}"

        # Check cache unless force refresh
        if not force_refresh:
            try:
                cached = await memory.cache.get(cache_key)
                if cached:
                    logger.debug("Returning cached compartments", cache_key=cache_key)
                    return cached
            except Exception:
                pass

        params = {
            "compartment_id": compartment_id,
            "include_subtree": include_subtree,
            "limit": 500,  # Increased to get all compartments for filtering
            "format": "json",
        }

        # Build OCI CLI fallback command
        cli_command = ["oci", "iam", "compartment", "list", "--output", "json", "--all"]
        if compartment_id:
            cli_command.extend(["--compartment-id", compartment_id])
        if include_subtree:
            cli_command.append("--compartment-id-in-subtree=true")

        result_str = await _execute_with_cli_fallback(
            tool_catalog=tool_catalog,
            tool_name="oci_list_compartments",
            tool_params=params,
            cli_command=cli_command,
            cli_timeout=60,
        )

        # Parse and filter results
        try:
            data = json.loads(result_str)
            # Support multiple result formats: list, {"data": [...]}, {"items": [...]}, {"compartments": [...]}
            compartments = data if isinstance(data, list) else data.get("data", data.get("items", data.get("compartments", [])))
        except (json.JSONDecodeError, TypeError):
            # Return raw result if not JSON
            return result_str

        # Apply filters
        filtered = []
        for comp in compartments:
            # Safely extract values - OCI API can return nested dicts
            comp_name = _safe_str_value(comp.get("name"), "")
            comp_state = _safe_str_value(
                comp.get("lifecycle-state", comp.get("lifecycle_state")), "ACTIVE"
            )

            # Skip if state filter doesn't match
            if filters.get("state") and comp_state.upper() != filters["state"]:
                continue

            # Skip if starts_with filter doesn't match
            if filters.get("starts_with") and not comp_name.upper().startswith(filters["starts_with"]):
                continue

            # Skip if contains filter doesn't match
            if filters.get("contains") and filters["contains"] not in comp_name.upper():
                continue

            # Skip if exact_name filter doesn't match
            if filters.get("exact_name") and comp_name.upper() != filters["exact_name"]:
                filtered_matches = _fuzzy_match_compartment(filters["exact_name"], compartments)
                if filtered_matches and not any(
                    _safe_str_value(m.get("name")).upper() == comp_name.upper()
                    for m in filtered_matches
                ):
                    continue

            filtered.append(comp)

        # If no results and we had filters, suggest similar names
        if not filtered and filters.get("starts_with"):
            # Find compartments with similar starting letters
            similar = [c for c in compartments if _safe_str_value(c.get("name")).upper().startswith(filters["starts_with"])]
            if not similar:
                # Suggest compartments that might match
                all_first_letters = sorted(set(_safe_str_value(c.get("name"), "X")[0].upper() for c in compartments if c.get("name")))
                suggestions = [l for l in all_first_letters if abs(ord(l) - ord(filters["starts_with"])) <= 1]
                suggestion_text = ""
                if suggestions:
                    suggestion_text = f"\n\nDid you mean compartments starting with: {', '.join(suggestions)}?"
                return f"No compartments found starting with '{filters['starts_with']}'.{suggestion_text}\n\nAvailable starting letters: {', '.join(all_first_letters)}"

        if not filtered and filters.get("exact_name"):
            # Find similar names for suggestions
            similar = _fuzzy_match_compartment(filters["exact_name"], compartments)
            if similar:
                suggestions = "\n".join([f"  - {m['name']} (OCID: {m['id'][:40]}...)" for m in similar])
                return f"No compartment found with exact name '{filters['exact_name']}'.\n\nDid you mean one of these?\n{suggestions}"

        # Build output with OCID mapping and pagination
        total = len(filtered)
        page_size = 25
        page = entities.get("page", 1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)

        # Build formatted output
        output_data = {
            "type": "compartments",
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "filters_applied": filters,
            "compartments": []
        }

        for comp in filtered[start_idx:end_idx]:
            output_data["compartments"].append({
                "name": comp.get("name"),
                "id": comp.get("id"),
                "state": comp.get("lifecycle-state", comp.get("lifecycle_state", "ACTIVE")),
                "description": comp.get("description", ""),
                "parent_id": comp.get("compartment-id", comp.get("compartment_id", "")),
            })

        result = json.dumps(output_data, indent=2)

        # Cache result (4 hour TTL for compartments)
        try:
            from datetime import timedelta
            await memory.cache.set(cache_key, result, ttl=timedelta(hours=4))
        except Exception:
            pass

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
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    List compute instances.

    Matches intents: list_instances, show_instances, get_instances
    Supports compartment name resolution (e.g., "list instances in Adrian_birzu compartment")
    Stores resolved compartment in conversation context for subsequent queries.
    """
    try:
        # Get thread_id for conversation context
        thread_id = metadata.get("thread_id") if metadata else None

        # Resolve compartment name to OCID
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        logger.info(
            "list_instances workflow starting",
            query=query,
            compartment_input=compartment_input,
            entities=entities,
            thread_id=thread_id,
        )
        compartment_id = await _resolve_compartment(compartment_input, tool_catalog, query, skip_root_fallback=True)

        # Track which compartment we're using
        used_fallback = False
        used_context = False
        context_compartment_name = None  # Track context name for reporting

        # If not resolved, check conversation context for previously used compartment
        if not compartment_id and thread_id:
            context_name, context_id = await _get_last_compartment(memory, thread_id)
            if context_id:
                compartment_id = context_id
                used_context = True
                context_compartment_name = context_name  # Save for reporting
                logger.info("Using compartment from conversation context",
                           thread_id=thread_id,
                           context_name=context_name,
                           context_id=context_id[:50] if context_id else None)

        # Fall back to root compartment if still not resolved
        if not compartment_id:
            compartment_id = _get_root_compartment()
            used_fallback = True
            logger.debug("Using root compartment as fallback",
                        ocid=compartment_id[:50] if compartment_id else None)
            if not compartment_id:
                return "Error: Could not determine compartment. Please specify a compartment name or configure OCI CLI."
        elif not used_context:
            logger.info("Compartment resolved successfully",
                       requested=compartment_input,
                       resolved_id=compartment_id[:50] if compartment_id else None)
            # Store compartment in conversation context for subsequent queries
            if thread_id:
                await _store_last_compartment(memory, thread_id, compartment_input, compartment_id)

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
            comp_info = f" (compartment: {compartment_input or 'root'})" if compartment_input else ""
            if used_fallback and compartment_input:
                return f"No instances found{state_msg}. Note: Compartment '{compartment_input}' was not found, searched root compartment instead."
            return f"No instances found{state_msg}{comp_info}."

        # Add compartment context to successful results for debugging
        try:
            result_data = json.loads(result_str)
            if isinstance(result_data, dict) and result_data.get("type") == "compute_instances":
                # Determine compartment name: explicit input > context > root
                if compartment_input:
                    searched_name = compartment_input
                elif context_compartment_name:
                    searched_name = f"{context_compartment_name} (from context)"
                else:
                    searched_name = "root"
                result_data["searched_compartment"] = searched_name
                result_data["used_fallback"] = used_fallback
                result_data["used_context"] = used_context
                return json.dumps(result_data)
        except (json.JSONDecodeError, TypeError):
            pass

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


async def instance_metrics_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get metrics for a compute instance.

    Matches intents: instance_metrics, show_metrics, get_instance_metrics
    Can look up instance by name. Asks for instance name if not provided.
    Returns CPU, memory, network, disk metrics with health assessment.
    """
    try:
        # Get thread_id for conversation context
        thread_id = metadata.get("thread_id") if metadata else None

        # Get instance identifier from entities
        instance_id = entities.get("instance_id")
        instance_name = entities.get("instance_name") or entities.get("name")

        # Extract instance name from query if not in entities
        if not instance_name and not instance_id:
            instance_name = _extract_instance_name_for_metrics(query)

        # If still no instance specified, ask the user
        if not instance_name and not instance_id:
            return json.dumps({
                "type": "clarification_needed",
                "message": "Please provide the instance name. Which instance would you like to see metrics for?",
                "examples": [
                    "show metrics for my-web-server",
                    "get instance metrics for app-server-01",
                    "CPU and memory usage of database-vm"
                ],
                "suggestion": "Use 'list instances' to see available instances"
            })

        # Get compartment from context or entities
        compartment_id = entities.get("compartment_id")
        if not compartment_id and thread_id and memory:
            # Try to get compartment from conversation context
            context = await memory.get_conversation_context(thread_id)
            compartment_id = context.get("compartment_id")

        # Get hours_back from entities (default 1 hour)
        hours_back = entities.get("hours_back", 1)

        # Extract hours from query if specified
        query_lower = query.lower()
        if "last" in query_lower:
            import re
            match = re.search(r"last\s+(\d+)\s*(?:hour|hr)", query_lower)
            if match:
                hours_back = int(match.group(1))

        logger.info(
            "instance_metrics workflow starting",
            instance_id=instance_id,
            instance_name=instance_name,
            compartment_id=compartment_id,
            hours_back=hours_back,
        )

        # Call the metrics tool
        params = {"hours_back": hours_back}
        if instance_id:
            params["instance_id"] = instance_id
        if instance_name:
            params["instance_name"] = instance_name
        if compartment_id:
            params["compartment_id"] = compartment_id

        result = await tool_catalog.execute(
            "oci_observability_get_instance_metrics",
            params,
        )

        return _extract_result(result)

    except Exception as e:
        logger.error("instance_metrics workflow failed", error=str(e))
        return f"Error getting instance metrics: {e}"


def _extract_instance_name_for_metrics(query: str) -> str | None:
    """
    Extract instance name from metrics-related queries.

    Handles patterns like:
    - "show metrics for my-instance"
    - "get CPU usage of web-server-01"
    - "instance metrics for app-vm"
    - "memory usage on database-server"
    """
    import re

    query_lower = query.lower()

    # Pattern 1: "for <name>" or "of <name>"
    match = re.search(r"(?:for|of)\s+['\"]?([a-zA-Z0-9_-]+)['\"]?", query_lower)
    if match:
        name = match.group(1)
        if name not in ("the", "a", "an", "my", "instance", "vm", "server", "this"):
            return name

    # Pattern 2: "on <name>"
    match = re.search(r"on\s+['\"]?([a-zA-Z0-9_-]+)['\"]?", query_lower)
    if match:
        name = match.group(1)
        if name not in ("the", "a", "an", "my", "instance", "vm"):
            return name

    # Pattern 3: "instance <name>" or "instance '<name>'"
    match = re.search(r"instance\s+(?:named\s+)?['\"]?([a-zA-Z0-9_-]+)['\"]?", query_lower)
    if match:
        name = match.group(1)
        if name not in ("metrics", "usage", "health"):
            return name

    # Pattern 4: "<name> metrics" or "<name> usage"
    match = re.search(r"([a-zA-Z0-9_-]+)\s+(?:metrics|usage|health)", query_lower)
    if match:
        name = match.group(1)
        if name not in ("instance", "vm", "compute", "cpu", "memory", "show", "get"):
            return name

    return None


def _extract_instance_name(query: str) -> str | None:
    """
    Extract instance name from natural language query.

    Handles patterns like:
    - "start instance my-instance"
    - "stop my-instance"
    - "restart the instance 'my-app-server'"
    - "start vm named my-server"
    """
    import re

    query_lower = query.lower()

    # Pattern 1: "instance <name>" or "instance '<name>'" or "instance named <name>"
    match = re.search(r"instance\s+(?:named\s+)?['\"]?([a-zA-Z0-9_-]+)['\"]?", query_lower)
    if match:
        return match.group(1)

    # Pattern 2: "vm <name>" or "vm named <name>"
    match = re.search(r"vm\s+(?:named\s+)?['\"]?([a-zA-Z0-9_-]+)['\"]?", query_lower)
    if match:
        return match.group(1)

    # Pattern 3: "start/stop/restart <name>" (action followed directly by name)
    match = re.search(r"(?:start|stop|restart|reboot)\s+['\"]?([a-zA-Z0-9_-]+)['\"]?", query_lower)
    if match:
        name = match.group(1)
        # Filter out common non-name words
        if name not in ("the", "a", "an", "my", "instance", "vm", "server", "machine"):
            return name

    # Pattern 4: "server <name>" or "server named <name>"
    match = re.search(r"server\s+(?:named\s+)?['\"]?([a-zA-Z0-9_-]+)['\"]?", query_lower)
    if match:
        return match.group(1)

    return None


async def start_instance_by_name_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Start a compute instance by name.

    Matches intents: start_instance, start_vm, power_on
    Supports instance name extraction from query and compartment resolution.
    Uses conversation context to infer compartment from previous queries.
    """
    try:
        # Get thread_id for conversation context
        thread_id = metadata.get("thread_id") if metadata else None

        # Get instance name from entities or extract from query
        instance_name = entities.get("instance_name") or entities.get("name")
        if not instance_name:
            instance_name = _extract_instance_name(query)

        if not instance_name:
            return (
                "Error: Instance name is required. Please specify which instance to start.\n"
                "Example: 'start instance my-app-server' or 'start my-web-server'"
            )

        # Resolve compartment - try entities first, then conversation context
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        # Use skip_root_fallback=True to allow conversation context to be checked
        compartment_id = await _resolve_compartment(
            compartment_input, tool_catalog, query, skip_root_fallback=True
        )
        used_context = False

        if not compartment_id:
            # Try to get compartment from conversation context
            last_comp_name, last_comp_id = await _get_last_compartment(memory, thread_id)
            if last_comp_id:
                compartment_id = last_comp_id
                compartment_input = last_comp_name
                used_context = True
                logger.info("Using compartment from conversation context",
                           compartment_name=last_comp_name,
                           compartment_id=last_comp_id[:30] if last_comp_id else None)

        if not compartment_id:
            compartment_id = _get_root_compartment()
            if not compartment_id:
                return (
                    "Error: Could not determine compartment. Please specify a compartment name or "
                    "list instances in a compartment first.\n"
                    "Example: 'start arkime in adrian_birzu compartment'"
                )

        logger.info(
            "Starting instance by name",
            instance_name=instance_name,
            compartment_id=compartment_id[:30] + "..." if compartment_id else None,
            used_context=used_context,
        )

        result = await tool_catalog.execute(
            "oci_compute_start_by_name",
            {
                "instance_name": instance_name,
                "compartment_id": compartment_id,
            },
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("start_instance workflow failed", error=str(e))
        return f"Error starting instance: {e}"


async def stop_instance_by_name_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Stop a compute instance by name (graceful shutdown).

    Matches intents: stop_instance, stop_vm, power_off, shutdown
    Supports instance name extraction from query and compartment resolution.
    Uses conversation context to infer compartment from previous queries.
    """
    try:
        # Get thread_id for conversation context
        thread_id = metadata.get("thread_id") if metadata else None
        logger.info("stop_instance_by_name workflow starting",
                   query=query,
                   thread_id=thread_id,
                   metadata_keys=list(metadata.keys()) if metadata else None)

        # Get instance name from entities or extract from query
        instance_name = entities.get("instance_name") or entities.get("name")
        if not instance_name:
            instance_name = _extract_instance_name(query)

        if not instance_name:
            return (
                "Error: Instance name is required. Please specify which instance to stop.\n"
                "Example: 'stop instance my-app-server' or 'stop my-web-server'"
            )

        # Resolve compartment - try entities first, then conversation context
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        # Use skip_root_fallback=True to allow conversation context to be checked
        compartment_id = await _resolve_compartment(
            compartment_input, tool_catalog, query, skip_root_fallback=True
        )
        used_context = False

        if not compartment_id:
            # Try to get compartment from conversation context
            last_comp_name, last_comp_id = await _get_last_compartment(memory, thread_id)
            if last_comp_id:
                compartment_id = last_comp_id
                compartment_input = last_comp_name
                used_context = True
                logger.info("Using compartment from conversation context",
                           compartment_name=last_comp_name,
                           compartment_id=last_comp_id[:30] if last_comp_id else None)

        if not compartment_id:
            compartment_id = _get_root_compartment()
            if not compartment_id:
                return (
                    "Error: Could not determine compartment. Please specify a compartment name or "
                    "list instances in a compartment first.\n"
                    "Example: 'stop arkime in adrian_birzu compartment'"
                )

        logger.info(
            "Stopping instance by name",
            instance_name=instance_name,
            compartment_id=compartment_id[:30] + "..." if compartment_id else None,
            used_context=used_context,
        )

        result = await tool_catalog.execute(
            "oci_compute_stop_by_name",
            {
                "instance_name": instance_name,
                "compartment_id": compartment_id,
            },
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("stop_instance workflow failed", error=str(e))
        return f"Error stopping instance: {e}"


async def restart_instance_by_name_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Restart a compute instance by name (graceful reboot).

    Matches intents: restart_instance, reboot_vm, reboot_instance
    Supports instance name extraction from query and compartment resolution.
    Uses conversation context to infer compartment from previous queries.
    """
    try:
        # Get thread_id for conversation context
        thread_id = metadata.get("thread_id") if metadata else None

        # Get instance name from entities or extract from query
        instance_name = entities.get("instance_name") or entities.get("name")
        if not instance_name:
            instance_name = _extract_instance_name(query)

        if not instance_name:
            return (
                "Error: Instance name is required. Please specify which instance to restart.\n"
                "Example: 'restart instance my-app-server' or 'reboot my-web-server'"
            )

        # Resolve compartment - try entities first, then conversation context
        compartment_input = entities.get("compartment_id") or entities.get("compartment_name")
        # Use skip_root_fallback=True to allow conversation context to be checked
        compartment_id = await _resolve_compartment(
            compartment_input, tool_catalog, query, skip_root_fallback=True
        )
        used_context = False

        if not compartment_id:
            # Try to get compartment from conversation context
            last_comp_name, last_comp_id = await _get_last_compartment(memory, thread_id)
            if last_comp_id:
                compartment_id = last_comp_id
                compartment_input = last_comp_name
                used_context = True
                logger.info("Using compartment from conversation context",
                           compartment_name=last_comp_name,
                           compartment_id=last_comp_id[:30] if last_comp_id else None)

        if not compartment_id:
            compartment_id = _get_root_compartment()
            if not compartment_id:
                return (
                    "Error: Could not determine compartment. Please specify a compartment name or "
                    "list instances in a compartment first.\n"
                    "Example: 'restart arkime in adrian_birzu compartment'"
                )

        logger.info(
            "Restarting instance by name",
            instance_name=instance_name,
            compartment_id=compartment_id[:30] + "..." if compartment_id else None,
            used_context=used_context,
        )

        result = await tool_catalog.execute(
            "oci_compute_restart_by_name",
            {
                "instance_name": instance_name,
                "compartment_id": compartment_id,
            },
        )
        return _extract_result(result)

    except Exception as e:
        logger.error("restart_instance workflow failed", error=str(e))
        return f"Error restarting instance: {e}"


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
    metadata: dict[str, Any] | None = None,
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

        # Get profile from entities first (extracted from query), then metadata (user's active profile)
        oci_profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        if oci_profile:
            tool_params["profile"] = oci_profile
            logger.info(
                "cost_summary_workflow using profile",
                profile=oci_profile,
                source="entities" if entities.get("oci_profile") else "metadata",
            )

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
    metadata: dict[str, Any] | None = None,
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

        # Get profile from entities first (extracted from query), then metadata (user's active profile)
        oci_profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        if oci_profile:
            tool_params["profile"] = oci_profile

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
    metadata: dict[str, Any] | None = None,
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

        # Get profile from entities first (extracted from query), then metadata (user's active profile)
        oci_profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        if oci_profile:
            tool_params["profile"] = oci_profile

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
    metadata: dict[str, Any] | None = None,
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

        # Get profile from entities first (extracted from query), then metadata (user's active profile)
        oci_profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        if oci_profile:
            tool_params["profile"] = oci_profile

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


async def cost_by_service_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get detailed cost breakdown by service with SKU-level detail.

    Provides deeper service analysis showing top resources within each service.

    Matches intents: cost_by_service, service_drilldown, service_breakdown,
                     show_costs_by_service, detailed_costs
    """
    try:
        time_range = entities.get("time_range", "30d")
        days = 30
        if isinstance(time_range, str) and time_range.endswith("d"):
            try:
                days = int(time_range.replace("d", ""))
            except ValueError:
                pass

        tool_params = {"days": days}

        # Get profile from entities first, then metadata
        oci_profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        if oci_profile:
            tool_params["profile"] = oci_profile

        # Support explicit date ranges
        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        if start_date and end_date:
            tool_params["start_date"] = start_date
            tool_params["end_date"] = end_date

        result = await tool_catalog.execute("oci_cost_service_drilldown", tool_params)

        return _extract_result(result)

    except Exception as e:
        logger.error("cost_by_service workflow failed", error=str(e))
        return f"Error getting service cost breakdown: {e}"


async def monthly_cost_trend_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get month-over-month cost trend.

    Shows how costs have changed over the specified number of months.

    Matches intents: monthly_trend, cost_trend, monthly_costs,
                     spending_trend, month_over_month
    """
    try:
        # Parse months from query
        months = 6  # default
        time_range = entities.get("time_range", "")
        if isinstance(time_range, str):
            if "3" in time_range:
                months = 3
            elif "6" in time_range:
                months = 6
            elif "12" in time_range or "year" in time_range.lower():
                months = 12

        tool_params = {"months": months}

        # Get profile
        oci_profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        if oci_profile:
            tool_params["profile"] = oci_profile

        result = await tool_catalog.execute("oci_cost_monthly_trend", tool_params)

        return _extract_result(result)

    except Exception as e:
        logger.error("monthly_cost_trend workflow failed", error=str(e))
        return f"Error getting monthly cost trend: {e}"


async def cost_comparison_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Compare costs between specific months.

    Takes month names from the query like "August vs October vs November"
    or "compare October and November".

    Matches intents: cost_comparison, compare_costs, compare_months,
                     month_comparison
    """
    try:
        # Extract month references from query
        # Common patterns: "August vs October", "compare Oct and Nov"
        import re

        # Find month names in the query
        month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b'
        matches = re.findall(month_pattern, query.lower())

        if len(matches) < 2:
            return json.dumps({
                "type": "cost_comparison",
                "error": "Please specify at least 2 months to compare (e.g., 'compare costs August vs October')"
            })

        # Build the months string
        months_str = " vs ".join(matches)

        tool_params = {"months": months_str}

        # Get profile
        oci_profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        if oci_profile:
            tool_params["profile"] = oci_profile

        result = await tool_catalog.execute("oci_cost_usage_comparison", tool_params)

        return _extract_result(result)

    except Exception as e:
        logger.error("cost_comparison workflow failed", error=str(e))
        return f"Error comparing costs: {e}"


async def cost_by_compartment_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get cost breakdown by compartment.

    Shows which compartments are consuming the most resources.

    Matches intents: cost_by_compartment, compartment_spending,
                     show_costs_by_compartment
    """
    try:
        time_range = entities.get("time_range", "30d")
        days = 30
        if isinstance(time_range, str) and time_range.endswith("d"):
            try:
                days = int(time_range.replace("d", ""))
            except ValueError:
                pass

        tool_params = {"days": days}

        # Get profile
        oci_profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        if oci_profile:
            tool_params["profile"] = oci_profile

        # Support explicit date ranges
        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        if start_date and end_date:
            tool_params["start_date"] = start_date
            tool_params["end_date"] = end_date

        result = await tool_catalog.execute("oci_cost_by_compartment", tool_params)

        return _extract_result(result)

    except Exception as e:
        logger.error("cost_by_compartment workflow failed", error=str(e))
        return f"Error getting compartment cost breakdown: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Database Listing Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def list_databases_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    List databases in the tenancy.

    Lists Managed Databases (DB Management), Autonomous Databases, and DB Systems.
    Runs data sources in PARALLEL for fast response, prioritizing results.

    Features:
    - 5-minute cache to avoid repeated OPSI/API calls
    - Parallel execution of multiple data sources
    - Graceful fallback when some sources fail
    - Profile support for multi-tenancy (e.g., EMDEMO, Default)

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
        except TimeoutError:
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

        # ── Extract Profiles ──────────────────────────────────────────────────
        # Extract profiles from entities only (profiles explicitly mentioned in query text)
        # For "list databases", we want comprehensive results across ALL profiles by default
        # Only limit to specific profile if user explicitly mentions it in query
        # Priority: entities.profiles > entities.oci_profile > query ALL configured profiles
        profiles_to_query: list[str] = []

        # Check entities first (extracted from query text - user explicitly mentioned profile)
        if entities.get("profiles"):
            profiles_to_query = entities["profiles"]
            logger.info(f"list_databases: found profiles {profiles_to_query} from query")
        elif entities.get("oci_profile"):
            profiles_to_query = [entities["oci_profile"]]
            logger.info(f"list_databases: using profile '{profiles_to_query[0]}' from entities")
        # NOTE: We intentionally don't use metadata.oci_profile here because for listing
        # databases, users want to see ALL their databases, not just the current profile's.
        # The metadata profile is used for other workflows where profile context matters.

        # If no profiles explicitly requested, query ALL configured profiles for comprehensive results
        # This ensures multi-tenancy users see all their databases
        if not profiles_to_query:
            import os
            # Check which profiles have DB Management compartments configured
            default_dbmgmt = os.getenv("OCI_DEFAULT_DBMGMT_COMPARTMENT_ID")
            emdemo_dbmgmt = os.getenv("OCI_EMDEMO_DBMGMT_COMPARTMENT_ID")

            if default_dbmgmt and emdemo_dbmgmt:
                # Both profiles configured - query both for comprehensive results
                profiles_to_query = [None, "EMDEMO"]  # type: ignore[list-item]
                logger.info("list_databases: querying both DEFAULT and EMDEMO profiles")
            elif emdemo_dbmgmt:
                # Only EMDEMO has DB Management configured
                profiles_to_query = ["EMDEMO"]
                logger.info("list_databases: using EMDEMO profile (has DBMGMT config)")
            elif default_dbmgmt:
                # Only DEFAULT has DB Management configured
                profiles_to_query = [None]  # type: ignore[list-item]
                logger.info("list_databases: using DEFAULT profile")
            else:
                # No DB Management configured - use DEFAULT and hope for the best
                profiles_to_query = [None]  # type: ignore[list-item]
                logger.info("list_databases: no DBMGMT compartment configured, using default")

        # ── Cache Check ──────────────────────────────────────────────────────
        # Build cache key from compartment and profiles
        profiles_key = "+".join(sorted(p or "default" for p in profiles_to_query))
        cache_key = f"{CACHE_PREFIX}:{compartment_id or 'root'}:{profiles_key}"

        # Try to get cached result
        try:
            cached_result = await memory.cache.get(cache_key)
            if cached_result:
                logger.info(
                    "list_databases: returning cached result",
                    cache_key=cache_key,
                    profiles=profiles_to_query,
                    ttl_minutes=5,
                )
                return cached_result
        except Exception as cache_err:
            # Cache failure should not block the workflow
            logger.debug("Cache read failed", error=str(cache_err))

        # ── Build tasks for ALL profiles in parallel ──────────────────────────
        # Each profile gets its own set of tool calls, all executed concurrently
        tasks: list[Any] = []
        task_metadata: list[tuple[str, str | None]] = []  # (tool_name, profile)

        for profile in profiles_to_query:
            profile_label = profile or "DEFAULT"

            # ── DB Management - Managed Databases ────────────────────────────
            # Use profile-specific compartment (e.g., EMDEMO uses OCI_EMDEMO_DBMGMT_COMPARTMENT_ID)
            dbmgmt_compartment, dbmgmt_region = _get_profile_compartment(profile, "dbmgmt")
            dbmgmt_params: dict[str, Any] = {"limit": 50, "include_subtree": True}

            # Use profile-specific compartment, or fall back to query compartment, then root
            if dbmgmt_compartment:
                dbmgmt_params["compartment_id"] = dbmgmt_compartment
            elif compartment_id:
                dbmgmt_params["compartment_id"] = compartment_id
            else:
                # Get tenancy root for this profile
                profile_root = _get_root_compartment(profile)
                if profile_root:
                    dbmgmt_params["compartment_id"] = profile_root

            if profile:
                dbmgmt_params["profile"] = profile
            if dbmgmt_region:
                dbmgmt_params["region"] = dbmgmt_region

            tasks.append(safe_execute("oci_dbmgmt_list_databases", dbmgmt_params))
            task_metadata.append(("oci_dbmgmt_list_databases", profile_label))

            # ── OPSI - Operations Insights Databases ─────────────────────────
            # OPSI may be in a different region (e.g., EMDEMO OPSI is in uk-london-1)
            opsi_compartment, opsi_region = _get_profile_compartment(profile, "opsi")
            opsi_params: dict[str, Any] = {"limit": 50}

            if opsi_compartment:
                opsi_params["compartment_id"] = opsi_compartment
            elif compartment_id:
                opsi_params["compartment_id"] = compartment_id

            if profile:
                opsi_params["profile"] = profile
            if opsi_region:
                opsi_params["region"] = opsi_region

            tasks.append(safe_execute("oci_opsi_list_database_insights", opsi_params))
            task_metadata.append(("oci_opsi_list_database_insights", profile_label))

            # ── Autonomous Databases ─────────────────────────────────────────
            adb_compartment = dbmgmt_compartment or compartment_id or _get_root_compartment(profile)
            if adb_compartment:
                adb_params: dict[str, Any] = {"compartment_id": adb_compartment}
                if profile:
                    adb_params["profile"] = profile
                tasks.append(safe_execute("oci_db_list_autonomous", adb_params))
                task_metadata.append(("oci_db_list_autonomous", profile_label))

        # Database connections (profile-agnostic - SQLcl connections)
        tasks.append(safe_execute("oci_database_list_connections", {}))
        task_metadata.append(("oci_database_list_connections", None))

        # Run all tools in PARALLEL across all profiles
        logger.info(
            f"list_databases: running {len(tasks)} tools in parallel",
            profiles=profiles_to_query,
            task_count=len(tasks),
        )
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results with priority ordering and profile grouping
        results: list[tuple[int, str, bool, str | None]] = []  # (priority, content, is_json, profile)
        errors: list[str] = []
        source_priority = {
            "oci_dbmgmt_list_databases": ("Managed Databases", 1),
            "oci_opsi_list_database_insights": ("OPSI Databases", 2),
            "oci_db_list_autonomous": ("Autonomous Databases", 3),
            "oci_database_list_connections": ("Database Connections", 4),
        }

        for idx, item in enumerate(results_raw):
            tool_name, profile_label = task_metadata[idx]
            if isinstance(item, Exception):
                errors.append(f"{tool_name} ({profile_label}): {item}")
                continue
            _, result_str = item
            if result_str:
                source_name, priority = source_priority.get(tool_name, (tool_name, 99))
                if tool_name == "oci_database_list_connections":
                    # Returns JSON - don't add markdown header, parser handles formatting
                    result_str = _format_database_connections(result_str)
                    results.append((priority, result_str, True, None))  # True = is_json
                elif '"type":' in result_str:
                    # Typed JSON response - pass directly to parser for Slack table formatting
                    # Parser handles: managed_databases, database_connections, etc.
                    results.append((priority, result_str, True, profile_label))
                else:
                    # Non-JSON result - add markdown header
                    if len(profiles_to_query) > 1 and profile_label:
                        header = f"## {source_name} ({profile_label})"
                    else:
                        header = f"## {source_name}"
                    results.append((priority, f"{header}\n{result_str}", False, profile_label))
            else:
                source_name, _ = source_priority.get(tool_name, (tool_name, 99))
                errors.append(f"{source_name} ({profile_label}) unavailable")

        if results:
            # Sort by priority, then by profile for consistent ordering
            results.sort(key=lambda x: (x[0], x[3] or ""))

            # If only one result and it's JSON, return it directly for parser
            # This enables proper Slack table rendering via ResponseParser
            if len(results) == 1 and results[0][2]:  # Check is_json flag
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
    metadata: dict[str, Any] | None = None,
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
                tool_params = {"days": days, "compartment_id": compartment_id}
                if metadata and metadata.get("oci_profile"):
                    tool_params["profile"] = metadata["oci_profile"]
                result = await tool_catalog.execute(
                    "oci_cost_get_summary",
                    tool_params,
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
    metadata: dict[str, Any] | None = None,
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
                tool_params = {
                    "days": days,
                    "compartment_id": compartment_id,
                    "service_filter": "compute",
                }
                if metadata and metadata.get("oci_profile"):
                    tool_params["profile"] = metadata["oci_profile"]
                result = await tool_catalog.execute(
                    "oci_cost_get_summary",
                    tool_params,
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
    metadata: dict[str, Any] | None = None,
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
                    "oci_opsi_list_database_insights",
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
                        "oci_db_list_autonomous",
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
                if metadata and metadata.get("oci_profile"):
                    cost_params["profile"] = metadata["oci_profile"]
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
    metadata: dict[str, Any] | None = None,
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
                tool_params = {"days": days}
                if metadata and metadata.get("oci_profile"):
                    tool_params["profile"] = metadata["oci_profile"]
                result = await tool_catalog.execute(
                    "oci_cost_get_summary",
                    tool_params,
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
                    results.append(f"*Total Tenancy Spend:* {summary.get('total', 'N/A')}")
                    results.append(f"*Period:* {summary.get('period', 'N/A')}\n")

                    if services:
                        results.append("*Top Services by Spend:*")
                        # Format as Slack-friendly list instead of markdown table
                        for i, svc in enumerate(services[:10], 1):
                            # Use mrkdwn formatting that renders well in Slack
                            results.append(f"  {i}. `{svc['service']}` — {svc['cost']} ({svc['percent']})")
                        results.append("")
                else:
                    results.append(cost_data + "\n")
            except json.JSONDecodeError:
                results.append(cost_data + "\n")

        # Add compartment structure - clean format for Slack
        if compartments_data:
            results.append("## Compartment Structure")
            # Parse compartments and format cleanly
            try:
                compartments_json = json.loads(compartments_data)
                if isinstance(compartments_json, dict) and "compartments" in compartments_json:
                    compartment_list = compartments_json["compartments"]
                elif isinstance(compartments_json, list):
                    compartment_list = compartments_json
                else:
                    compartment_list = []

                if compartment_list:
                    # Count and show summary
                    active_count = sum(1 for c in compartment_list if c.get("state") == "ACTIVE" or c.get("lifecycle_state") == "ACTIVE")
                    results.append(f"Found *{len(compartment_list)}* compartments ({active_count} active)\n")

                    # Show top compartments (by name, alphabetically) - limit to 20 for Slack
                    sorted_compartments = sorted(
                        compartment_list,
                        key=lambda x: x.get("name", x.get("display_name", "")).lower()
                    )[:20]

                    results.append("*Active Compartments:*")
                    for comp in sorted_compartments:
                        name = comp.get("name") or comp.get("display_name", "Unknown")
                        state = comp.get("state") or comp.get("lifecycle_state", "UNKNOWN")
                        if state == "ACTIVE":
                            results.append(f"  • `{name}`")

                    if len(compartment_list) > 20:
                        results.append(f"\n_... and {len(compartment_list) - 20} more compartments_")
                else:
                    results.append(compartments_data)
            except json.JSONDecodeError:
                # Fallback: try to clean up raw output
                results.append(compartments_data)

            results.append("\n_Tip: Ask about costs for a specific compartment by name._")
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
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get fleet-wide database health summary.

    Shows health status across all managed databases in the tenancy.

    Matches intents: db_fleet_health, fleet_health, database_fleet_status,
                     all_db_health, managed_database_health
    """
    try:
        compartment_id = entities.get("compartment_id")
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)

        if not compartment_id:
            # Get profile-specific compartment for DB Management
            dbmgmt_compartment, _ = _get_profile_compartment(profile, "dbmgmt")
            compartment_id = dbmgmt_compartment or _get_root_compartment()

        result = await tool_catalog.execute(
            "oci_dbmgmt_get_fleet_health",
            {"compartment_id": compartment_id, "include_subtree": True, "profile": profile},
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
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get top SQL statements by CPU activity for a database.

    Shows the most resource-intensive SQL statements currently running.
    Supports resolution from database name (e.g., "top SQL for FINANCE").

    Matches intents: top_sql, db_top_sql, high_cpu_sql, expensive_queries,
                     sql_performance, sql_cpu_usage
    """
    try:
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)

        # First try direct ID, then resolve from name
        managed_database_id = entities.get("managed_database_id") or entities.get("database_id")

        if not managed_database_id:
            db_name = entities.get("database_name")
            if db_name:
                logger.info("Resolving database name for top SQL", name=db_name)
                managed_database_id = await _resolve_managed_database(
                    db_name, tool_catalog, entities.get("compartment_id"), profile=profile
                )

        if not managed_database_id:
            db_name = entities.get("database_name", "")
            return (
                f"Error: Could not find database '{db_name}'. "
                "Please provide a valid database name or managed_database_id."
            )

        hours_back = entities.get("hours_back", 1)
        limit = entities.get("limit", 10)

        result = await tool_catalog.execute(
            "oci_dbmgmt_get_top_sql",
            {
                "managed_database_id": managed_database_id,
                "hours_back": hours_back,
                "limit": limit,
                "profile": profile,
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
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get top wait events from AWR data for a database.

    Shows what the database is waiting on most - useful for performance tuning.
    Supports resolution from database name (e.g., "wait events for FINANCE").

    Matches intents: wait_events, db_wait_events, awr_wait_events,
                     database_waits, performance_bottlenecks
    """
    try:
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)

        # First try direct ID, then resolve from name
        managed_database_id = entities.get("managed_database_id") or entities.get("database_id")

        if not managed_database_id:
            db_name = entities.get("database_name")
            if db_name:
                logger.info("Resolving database name for wait events", name=db_name)
                managed_database_id = await _resolve_managed_database(
                    db_name, tool_catalog, entities.get("compartment_id"), profile=profile
                )

        if not managed_database_id:
            db_name = entities.get("database_name", "")
            return (
                f"Error: Could not find database '{db_name}'. "
                "Please provide a valid database name or managed_database_id."
            )

        hours_back = entities.get("hours_back", 1)
        top_n = entities.get("top_n", 10)

        result = await tool_catalog.execute(
            "oci_dbmgmt_get_wait_events",
            {
                "managed_database_id": managed_database_id,
                "hours_back": hours_back,
                "top_n": top_n,
                "profile": profile,
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
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Generate AWR or ASH report for a managed database.

    Creates a detailed performance report for the specified time period.
    Supports resolution from database name (e.g., "AWR report for FINANCE").

    Matches intents: awr_report, generate_awr, ash_report, performance_report,
                     db_performance_report, database_report
    """
    try:
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)

        # First try direct ID, then resolve from name
        managed_database_id = entities.get("managed_database_id") or entities.get("database_id")

        if not managed_database_id:
            # Try to resolve from database_name
            db_name = entities.get("database_name")
            if db_name:
                logger.info("Resolving database name to ID", name=db_name)
                managed_database_id = await _resolve_managed_database(
                    db_name, tool_catalog, entities.get("compartment_id"), profile=profile
                )

        if not managed_database_id:
            # Provide helpful error with available databases
            db_name = entities.get("database_name", "")
            return (
                f"Error: Could not find database '{db_name}'. "
                "Please provide a valid database name or managed_database_id. "
                "Use 'list databases' or 'show managed databases' to see available databases."
            )

        hours_back = entities.get("hours_back", 24)
        report_type = entities.get("report_type", "AWR").upper()
        report_format = entities.get("report_format", "TEXT").upper()

        if report_type not in ("AWR", "ASH"):
            report_type = "AWR"
        if report_format not in ("HTML", "TEXT"):
            report_format = "TEXT"

        logger.info("Generating AWR report",
                   database_id=managed_database_id[:50] if managed_database_id else None,
                   hours_back=hours_back, report_type=report_type)

        result = await tool_catalog.execute(
            "oci_dbmgmt_get_awr_report",
            {
                "managed_database_id": managed_database_id,
                "hours_back": hours_back,
                "report_type": report_type,
                "report_format": report_format,
                "profile": profile,
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
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    List SQL Plan Baselines for a database.

    Shows SQL statements with captured execution plans for plan stability.

    Matches intents: sql_plan_baselines, db_baselines, execution_plans,
                     plan_stability, sql_plans
    """
    try:
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)

        managed_database_id = entities.get("managed_database_id") or entities.get("database_id")

        # Try to resolve from database name if ID not provided
        if not managed_database_id:
            db_name = entities.get("database_name")
            if db_name:
                managed_database_id = await _resolve_managed_database(
                    db_name, tool_catalog, entities.get("compartment_id"), profile=profile
                )

        if not managed_database_id:
            return "Error: Please provide a managed_database_id, database_id, or database_name to list SQL Plan Baselines."

        limit = entities.get("limit", 50)

        result = await tool_catalog.execute(
            "oci_dbmgmt_list_sql_plan_baselines",
            {
                "managed_database_id": managed_database_id,
                "limit": limit,
                "profile": profile,
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
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    List managed databases in Database Management service.

    Shows all databases registered with the DB Management service.

    Matches intents: managed_databases, list_managed_databases, dbmgmt_databases,
                     db_management_list, registered_databases
    """
    try:
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)

        compartment_id = entities.get("compartment_id")
        if not compartment_id:
            # Get profile-specific compartment for DB Management
            dbmgmt_compartment, _ = _get_profile_compartment(profile, "dbmgmt")
            compartment_id = dbmgmt_compartment or _get_root_compartment()

        database_type = entities.get("database_type")  # EXTERNAL_SIDB, EXTERNAL_RAC, etc.
        deployment_type = entities.get("deployment_type")  # ONPREMISE, BM, etc.

        params = {
            "compartment_id": compartment_id,
            "include_subtree": True,
            "profile": profile,
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
# SQLcl-Based Database Performance Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def _resolve_sqlcl_connection(
    database_name: str | None,
    tool_catalog: ToolCatalog,
) -> tuple[str | None, str | None]:
    """
    Resolve database name to an active SQLcl connection.

    Returns:
        tuple: (connection_name, error_message)
        - If successful: (connection_name, None)
        - If failed: (None, error_message)
    """
    import json

    if not database_name:
        return None, "No database name provided. Please specify a database name."

    try:
        # Get available connections - returns ToolCallResult object
        result = await tool_catalog.execute("oci_database_list_connections", {})

        # Handle ToolCallResult object (has .success, .result, .error attributes)
        if isinstance(result, ToolCallResult):
            if not result.success:
                return None, (
                    f"Could not list SQLcl connections. "
                    f"Error: {result.error or 'Unknown error'}"
                )
            raw_result = result.result
        else:
            # Fallback for dict-like results
            raw_result = result

        # Parse result - might be JSON string or dict
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
            except json.JSONDecodeError:
                # Result is plain string, try to extract connection names
                # Format might be "Connections: conn1, conn2" or similar
                if ":" in raw_result:
                    parts = raw_result.split(":", 1)
                    if len(parts) > 1:
                        conn_list = [c.strip() for c in parts[1].split(",")]
                        parsed = {"connections": conn_list}
                    else:
                        parsed = {"connections": []}
                else:
                    parsed = {"connections": []}
        elif isinstance(raw_result, dict):
            parsed = raw_result
        else:
            parsed = {"connections": []}

        # Extract connections list
        connections = parsed.get("connections", [])
        if not connections:
            return None, (
                "No SQLcl connections available. "
                "Please configure a database connection first using SQLcl."
            )

        # Try to find a matching connection (case-insensitive partial match)
        db_lower = database_name.lower()
        for conn in connections:
            conn_name = conn.get("name", "") if isinstance(conn, dict) else str(conn)
            if db_lower in conn_name.lower():
                return conn_name, None

        # List available connections in error message
        available = [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in connections]
        return None, (
            f"No SQLcl connection found matching '{database_name}'. "
            f"Available connections: {', '.join(available)}"
        )

    except Exception as e:
        logger.error("Failed to resolve SQLcl connection", error=str(e))
        return None, f"Error resolving database connection: {e}"


async def db_sql_monitoring_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get SQL Monitoring Report for active or recent SQL statements.

    Uses DB Management API (oci_dbmgmt_get_top_sql) as primary source.
    Falls back to OPSI SQL statistics if specific DB not found.

    Matches intents: sql_monitoring, sql_monitor, active_sql, running_queries,
                     execution_monitoring, real_time_sql
    """
    try:
        database_name = entities.get("connection_name") or entities.get("database_name")
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)

        # First, try to find the managed database OCID
        managed_db_id = None
        if database_name:
            # Search for the database in DB Management
            search_result = await tool_catalog.execute(
                "oci_dbmgmt_search_databases",
                {"name": database_name, "profile": profile},
            )
            search_str = _extract_result(search_result)

            # Try to extract OCID from search results (match both regular and autonomous databases)
            if search_str and ("ocid1.database" in search_str.lower() or "ocid1.autonomousdatabase" in search_str.lower()):
                import re
                ocid_match = re.search(r'(ocid1\.(?:autonomous)?database\.[^"\s,]+)', search_str, re.IGNORECASE)
                if ocid_match:
                    managed_db_id = ocid_match.group(1)
                    logger.info("Found managed database OCID", database_id=managed_db_id)

        # If we found a managed database, get top SQL from DB Management
        if managed_db_id:
            logger.info("Using DB Management API for SQL monitoring", database_id=managed_db_id)
            result = await tool_catalog.execute(
                "oci_dbmgmt_get_top_sql",
                {
                    "managed_database_id": managed_db_id,
                    "hours_back": 1,  # Last hour of SQL activity
                    "limit": 10,      # Top 10 SQL statements
                    "profile": profile,
                },
            )
            extracted = _extract_result(result)
            if extracted and not extracted.startswith("Error"):
                return f"**Top SQL Activity for {database_name or 'database'}** (Last Hour)\n\n{extracted}"

        # Fallback to OPSI SQL statistics for broader view
        logger.info("Using OPSI SQL statistics for SQL monitoring")
        result = await tool_catalog.execute(
            "oci_opsi_summarize_sql_statistics",
            {
                "profile": profile,
                "sort_by": "databaseTimeInSec",  # Sort by total database time
                "limit": 15,  # Top 15 SQL statements
            },
        )
        extracted = _extract_result(result)
        if extracted and not extracted.startswith("Error"):
            return f"**SQL Activity Summary{' for ' + database_name if database_name else ''}**\n\n{extracted}"

        # If all else fails, try OPSI SQL insights
        result = await tool_catalog.execute(
            "oci_opsi_summarize_sql_insights",
            {"profile": profile},
        )
        return f"**SQL Insights{' for ' + database_name if database_name else ''}**\n\n{_extract_result(result)}"

    except Exception as e:
        logger.error("db_sql_monitoring workflow failed", error=str(e))
        return f"Error getting SQL monitoring data: {e}"


async def db_long_running_ops_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Check long-running operations from v$session_longops.

    Shows operations like table scans, index builds, and backup/recovery
    that are still in progress or recently completed.

    Matches intents: long_running_ops, longops, long_operations,
                     running_operations, batch_progress, operation_status
    """
    try:
        # Resolve database name to SQLcl connection
        database_name = entities.get("connection_name") or entities.get("database_name")
        connection_name, error = await _resolve_sqlcl_connection(database_name, tool_catalog)
        if error:
            return error

        limit = entities.get("limit", 20)

        sql = f"""
            SELECT sid, serial#, opname, target,
                   sofar, totalwork,
                   CASE WHEN totalwork > 0
                        THEN ROUND(sofar/totalwork*100, 2)
                        ELSE 0 END as pct_complete,
                   ROUND(elapsed_seconds/60, 2) as elapsed_min,
                   ROUND(time_remaining/60, 2) as remaining_min,
                   message
            FROM v$session_longops
            WHERE sofar < totalwork OR time_remaining > 0
            ORDER BY elapsed_seconds DESC
            FETCH FIRST {limit} ROWS ONLY
        """

        result = await tool_catalog.execute(
            "oci_database_execute_sql",
            {"sql": sql.strip(), "connection_name": connection_name},
        )

        extracted = _extract_result(result)
        # Provide helpful message if no long-running ops found
        if "No rows" in extracted or (isinstance(extracted, str) and "[]" in extracted):
            return "✅ No long-running operations in progress. All batch operations have completed."
        return extracted

    except Exception as e:
        logger.error("db_long_running_ops workflow failed", error=str(e))
        return f"Error getting long-running operations: {e}"


async def db_parallelism_stats_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Compare requested vs actual parallelism degree for SQL statements.

    Uses OPSI SQL statistics to analyze parallel query execution.
    Falls back to DB Management top SQL if OPSI unavailable.

    Matches intents: parallelism_stats, parallel_query, px_stats,
                     degree_comparison, parallel_execution, px_downgrade
    """
    try:
        database_name = entities.get("connection_name") or entities.get("database_name")
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)

        # Try OPSI SQL statistics for parallel query analysis
        logger.info("Using OPSI for parallelism analysis", database=database_name)
        result = await tool_catalog.execute(
            "oci_opsi_summarize_sql_statistics",
            {
                "profile": profile,
                "sort_by": "cpuTimeInSec",  # CPU-intensive queries often use parallelism
                "limit": 15,
            },
        )
        extracted = _extract_result(result)

        if extracted and not extracted.startswith("Error"):
            # Format response with parallelism context
            response_parts = [
                f"**Parallel Query Analysis{' for ' + database_name if database_name else ''}**",
                "",
                "SQL statements analyzed for parallel execution patterns:",
                "",
                extracted,
                "",
                "---",
                "**Interpretation Guide:**",
                "- High CPU Time with low Buffer Gets may indicate parallelism working well",
                "- High Execution Count with long elapsed time may indicate PX downgrade",
                "- Use AWR report for detailed parallel execution statistics",
            ]
            return "\n".join(response_parts)

        # Fallback to DB Management top SQL
        if database_name:
            search_result = await tool_catalog.execute(
                "oci_dbmgmt_search_databases",
                {"name": database_name, "profile": profile},
            )
            search_str = _extract_result(search_result)

            # Match both regular and autonomous database OCIDs
            if search_str and ("ocid1.database" in search_str.lower() or "ocid1.autonomousdatabase" in search_str.lower()):
                import re
                ocid_match = re.search(r'(ocid1\.(?:autonomous)?database\.[^"\s,]+)', search_str, re.IGNORECASE)
                if ocid_match:
                    managed_db_id = ocid_match.group(1)
                    logger.info("Found managed database OCID for top SQL", database_id=managed_db_id)
                    result = await tool_catalog.execute(
                        "oci_dbmgmt_get_top_sql",
                        {"managed_database_id": managed_db_id, "profile": profile},
                    )
                    return f"**Top SQL Analysis for {database_name}**\n\n{_extract_result(result)}"

        # If no specific DB, get ADDM findings which include parallelism recommendations
        result = await tool_catalog.execute(
            "oci_opsi_get_addm_findings",
            {"profile": profile},
        )
        extracted = _extract_result(result)
        if extracted and not extracted.startswith("Error"):
            return f"**ADDM Findings (includes parallel execution analysis)**\n\n{extracted}"

        return "No parallelism statistics available. Ensure database is enrolled in OPSI or DB Management."

    except Exception as e:
        logger.error("db_parallelism_stats workflow failed", error=str(e))
        return f"Error getting parallelism statistics: {e}"


async def db_full_table_scan_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Detect full table scans on large tables.

    Shows SQL statements performing TABLE ACCESS FULL operations,
    which can indicate missing indexes or suboptimal execution plans.

    Matches intents: full_table_scan, table_scan, fts_detection,
                     large_table_scan, missing_index, scan_analysis
    """
    try:
        # Resolve database name to SQLcl connection
        database_name = entities.get("connection_name") or entities.get("database_name")
        connection_name, error = await _resolve_sqlcl_connection(database_name, tool_catalog)
        if error:
            return error

        sql_id = entities.get("sql_id")
        min_size_gb = entities.get("min_size_gb", 1)  # Default 1 GB threshold
        limit = entities.get("limit", 20)

        if sql_id:
            sql = f"""
                SELECT DISTINCT p.sql_id, p.object_owner, p.object_name,
                       s.elapsed_time/1000000 as elapsed_sec,
                       s.buffer_gets, s.disk_reads,
                       ROUND(seg.bytes/1024/1024/1024, 2) as table_size_gb
                FROM v$sql_plan p
                JOIN v$sql s ON p.sql_id = s.sql_id AND p.child_number = s.child_number
                LEFT JOIN dba_segments seg ON p.object_owner = seg.owner
                                          AND p.object_name = seg.segment_name
                WHERE p.sql_id = '{sql_id}'
                  AND p.operation = 'TABLE ACCESS'
                  AND p.options = 'FULL'
                ORDER BY s.elapsed_time DESC
            """
        else:
            sql = f"""
                SELECT DISTINCT p.sql_id, p.object_owner, p.object_name,
                       s.elapsed_time/1000000 as elapsed_sec,
                       s.buffer_gets, s.disk_reads,
                       ROUND(seg.bytes/1024/1024/1024, 2) as table_size_gb
                FROM v$sql_plan p
                JOIN v$sql s ON p.sql_id = s.sql_id AND p.child_number = s.child_number
                LEFT JOIN dba_segments seg ON p.object_owner = seg.owner
                                          AND p.object_name = seg.segment_name
                WHERE p.operation = 'TABLE ACCESS'
                  AND p.options = 'FULL'
                  AND (seg.bytes IS NULL OR seg.bytes > {min_size_gb} * 1024 * 1024 * 1024)
                ORDER BY s.elapsed_time DESC
                FETCH FIRST {limit} ROWS ONLY
            """

        result = await tool_catalog.execute(
            "oci_database_execute_sql",
            {"sql": sql.strip(), "connection_name": connection_name},
        )

        extracted = _extract_result(result)
        # Provide helpful message if no full table scans found
        if "No rows" in extracted or (isinstance(extracted, str) and "[]" in extracted):
            return f"✅ No full table scans detected on tables larger than {min_size_gb} GB."
        return extracted

    except Exception as e:
        logger.error("db_full_table_scan workflow failed", error=str(e))
        return f"Error detecting full table scans: {e}"


async def db_blocking_sessions_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Check for blocking sessions / lock contention in the database.

    Uses DB Management API (oci_dbmgmt_get_wait_events) to identify lock-related
    wait events, and OPSI ADDM findings for blocking analysis.

    Matches intents: blocking_sessions, blocked_sessions, lock_contention,
                     session_blocking, lock_analysis, database_locks
    """
    try:
        database_name = entities.get("connection_name") or entities.get("database_name")
        # Extract profile from entities or fall back to metadata (Slack passes profile via metadata)
        profile = _auto_select_db_profile(
            entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
        )

        # First, try to find the managed database OCID
        managed_db_id = None
        if database_name:
            search_result = await tool_catalog.execute(
                "oci_dbmgmt_search_databases",
                {"name": database_name, "profile": profile},
            )
            search_str = _extract_result(search_result)
            # Match both ocid1.database and ocid1.autonomousdatabase OCIDs
            if search_str and ("ocid1.database" in search_str.lower() or "ocid1.autonomousdatabase" in search_str.lower()):
                import re
                # Match both regular and autonomous database OCIDs
                ocid_match = re.search(r'(ocid1\.(?:autonomous)?database\.[^"\s,]+)', search_str, re.IGNORECASE)
                if ocid_match:
                    managed_db_id = ocid_match.group(1)
                    logger.info("Found managed database OCID", database_id=managed_db_id)

        # If we found a managed database, get wait events (includes lock waits)
        if managed_db_id:
            logger.info("Using DB Management API for blocking/lock analysis", database_id=managed_db_id)
            result = await tool_catalog.execute(
                "oci_dbmgmt_get_wait_events",
                {
                    "managed_database_id": managed_db_id,
                    "hours_back": 1,
                    "top_n": 15,  # Get more events to catch lock-related ones
                    "profile": profile,
                },
            )
            extracted = _extract_result(result)
            if extracted and not extracted.startswith("Error"):
                # Check for lock-related wait events
                lock_keywords = ["lock", "enq:", "tx -", "row lock", "library cache", "latch"]
                has_lock_events = any(kw.lower() in extracted.lower() for kw in lock_keywords)

                header = f"**Wait Events Analysis for {database_name or 'database'}** (Last Hour)\n\n"
                if has_lock_events:
                    header += "⚠️ **Lock-related wait events detected!** Review the events below:\n\n"
                else:
                    header += "✅ **No significant lock contention detected.** Current wait events:\n\n"
                return header + extracted

        # Fallback: Try to get OPSI database insight ID if we have a database name
        opsi_db_id = None
        if database_name:
            # Search OPSI database insights for the database
            logger.info("Searching OPSI for database", database_name=database_name)
            opsi_result = await tool_catalog.execute(
                "oci_opsi_list_database_insights",
                {"profile": profile, "limit": 100},
            )
            opsi_str = _extract_result(opsi_result)
            if opsi_str and database_name.lower() in opsi_str.lower():
                import re
                # Look for OPSI database insight ID
                opsi_match = re.search(r'"id"\s*:\s*"(ocid1\.databaseinsight\.[^"]+)"', opsi_str)
                if opsi_match:
                    opsi_db_id = opsi_match.group(1)
                    logger.info("Found OPSI database insight", opsi_db_id=opsi_db_id)

        # Try OPSI ADDM findings with database context if available
        db_id_for_addm = opsi_db_id or managed_db_id
        if db_id_for_addm:
            logger.info("Using OPSI ADDM for blocking analysis", db_id=db_id_for_addm)
            result = await tool_catalog.execute(
                "oci_opsi_get_addm_findings",
                {"database_id": db_id_for_addm, "profile": profile},
            )
            extracted = _extract_result(result)
            if extracted and not extracted.startswith("Error") and '"error"' not in extracted.lower():
                return f"**ADDM Blocking/Performance Findings{' for ' + database_name if database_name else ''}**\n\n{extracted}"
        else:
            logger.debug("Skipping ADDM - no database ID available", opsi_db_id=opsi_db_id, managed_db_id=managed_db_id)

        # Fallback: Use SQLcl to directly query blocking sessions if we have a connection
        if database_name:
            logger.info("Using SQLcl for direct blocking session query", connection=database_name)
            blocking_sql = """
SELECT s.sid || ',' || s.serial# as blocked_session,
       bs.sid || ',' || bs.serial# as blocking_session,
       s.username as blocked_user,
       bs.username as blocking_user,
       s.event as wait_event,
       s.seconds_in_wait as wait_seconds,
       s.sql_id as blocked_sql_id,
       bs.sql_id as blocking_sql_id
FROM v$session s
JOIN v$session bs ON s.blocking_session = bs.sid
WHERE s.blocking_session IS NOT NULL
ORDER BY s.seconds_in_wait DESC"""
            result = await tool_catalog.execute(
                "oci_database_execute_sql",
                {"connection_name": database_name, "sql": blocking_sql},
            )
            extracted = _extract_result(result)
            if extracted and not extracted.startswith("Error"):
                if "No rows" in extracted or "0 rows" in extracted:
                    return f"✅ **No blocking sessions detected on {database_name}**\n\nNo sessions are currently blocked by other sessions."
                return f"**Blocking Sessions on {database_name}**\n\n{extracted}"

        # Final fallback - check wait events across all databases
        result = await tool_catalog.execute(
            "oci_opsi_summarize_sql_insights",
            {"profile": profile},
        )
        extracted = _extract_result(result)
        return f"**SQL Insights (may include lock waits){' for ' + database_name if database_name else ''}**\n\n{extracted}"

    except Exception as e:
        logger.error("db_blocking_sessions workflow failed", error=str(e))
        return f"Error checking blocking/lock contention: {e}"


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
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Get comprehensive database performance overview.

    Combines multiple data sources for a complete performance picture:
    - Fleet health summary (DB Management in dbmgmt region)
    - Top SQL by CPU
    - ADDM findings and recommendations (OPSI in opsi region)
    - Capacity trends

    Uses profile-specific compartments/regions:
    - EMDEMO: DB Management in us-ashburn-1, OPSI in uk-london-1
    - DEFAULT: Uses root tenancy compartment

    Matches intents: db_performance_overview, database_performance, db_health_check,
                     comprehensive_db_status, full_db_analysis
    """
    import asyncio

    try:
        # Extract profile - supports multi-tenancy (EMDEMO, DEFAULT, etc.)
        # Priority: entities.profile > entities.oci_profile > metadata.oci_profile
        profile = (
            entities.get("profile")
            or entities.get("oci_profile")
            or (metadata.get("oci_profile") if metadata else None)
        )
        compartment_id = entities.get("compartment_id")

        # Auto-select best profile for DB operations if none specified
        profile = _auto_select_db_profile(profile)

        # Get profile-specific compartments and regions for different services
        dbmgmt_compartment, dbmgmt_region = _get_profile_compartment(profile, "dbmgmt")
        opsi_compartment, opsi_region = _get_profile_compartment(profile, "opsi")

        # Use profile-specific compartment, or fall back to provided, then root
        if not dbmgmt_compartment:
            dbmgmt_compartment = compartment_id or _get_root_compartment(profile)
        if not opsi_compartment:
            opsi_compartment = compartment_id or _get_root_compartment(profile)

        database_id = entities.get("database_id") or entities.get("managed_database_id")

        results = []
        profile_label = profile.upper() if profile else "DEFAULT"
        results.append(f"# Database Performance Overview ({profile_label})\n")

        async def get_fleet_health():
            try:
                params = {"compartment_id": dbmgmt_compartment, "include_subtree": True}
                if profile:
                    params["profile"] = profile
                if dbmgmt_region:
                    params["region"] = dbmgmt_region
                result = await tool_catalog.execute("oci_dbmgmt_get_fleet_health", params)
                return _extract_result(result)
            except Exception as e:
                logger.debug("Fleet health failed", error=str(e))
                return None

        async def get_addm_findings():
            try:
                params = {"compartment_id": opsi_compartment, "days": 7}
                if database_id:
                    params["database_id"] = database_id
                if profile:
                    params["profile"] = profile
                if opsi_region:
                    params["region"] = opsi_region
                result = await tool_catalog.execute("oci_opsi_get_addm_findings", params)
                return _extract_result(result)
            except Exception as e:
                logger.debug("ADDM findings failed", error=str(e))
                return None

        async def get_recommendations():
            try:
                params = {"compartment_id": opsi_compartment, "days": 7}
                if database_id:
                    params["database_id"] = database_id
                if profile:
                    params["profile"] = profile
                if opsi_region:
                    params["region"] = opsi_region
                result = await tool_catalog.execute("oci_opsi_get_addm_recommendations", params)
                return _extract_result(result)
            except Exception as e:
                logger.debug("ADDM recommendations failed", error=str(e))
                return None

        async def get_resource_stats():
            try:
                params = {"compartment_id": opsi_compartment, "resource_metric": "CPU", "days": 7}
                if profile:
                    params["profile"] = profile
                if opsi_region:
                    params["region"] = opsi_region
                result = await tool_catalog.execute("oci_opsi_summarize_resource_stats", params)
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
            results.append("## Status: No Database Performance Data Available\n")
            results.append("The performance analysis could not retrieve any database metrics.\n")
            results.append("**Possible causes:**")
            results.append("1. **No databases registered with DB Management Service** - Enable Database Management for your databases")
            results.append("2. **No databases enrolled in Operations Insights (OPSI)** - Enroll databases to get ADDM findings and SQL insights")
            results.append("3. **Database performance monitoring not enabled** - Check database monitoring settings in OCI Console\n")
            results.append("**Recommended actions:**")
            results.append("- Run `list databases` to see available databases in the tenancy")
            results.append("- Enable DB Management for Autonomous Databases via OCI Console > Autonomous Databases > Database Details > Enable DB Management")
            results.append("- Enable Operations Insights via OCI Console > Database Management > Operations Insights\n")
            results.append("For more information, see the [OCI Database Management documentation](https://docs.oracle.com/en-us/iaas/database-management/home.htm)")

        return "\n".join(results)

    except Exception as e:
        logger.error("db_performance_overview workflow failed", error=str(e))
        return f"Error getting database performance overview: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Observability Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def list_alarms_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List active alarms from OCI Monitoring service.

    Returns alarms with severity, state, and namespace information.

    Matches intents: list_alarms, show_alarms, active_alarms, monitoring_alarms,
                     alarm_status, alert_status
    """
    try:
        compartment_id = entities.get("compartment_id")
        profile = entities.get("oci_profile")
        lifecycle_state = entities.get("lifecycle_state", "ACTIVE")

        params: dict[str, Any] = {"limit": 50}
        if compartment_id:
            params["compartment_id"] = compartment_id
        if profile:
            params["profile"] = profile
        if lifecycle_state:
            params["lifecycle_state"] = lifecycle_state

        result = await tool_catalog.execute(
            "oci_observability_list_alarms",
            params,
        )

        extracted = _extract_result(result)
        if not extracted or extracted.startswith("Error"):
            return "Error listing alarms: Unable to retrieve alarm data."

        return extracted

    except Exception as e:
        logger.error("list_alarms workflow failed", error=str(e))
        return f"Error listing alarms: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Log Analytics Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def log_summary_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get Log Analytics summary.

    Returns storage usage, source counts, and log group information.

    Matches intents: log_summary, show_log_summary, logging_summary,
                     log_analytics_status
    """
    try:
        compartment_id = entities.get("compartment_id")
        profile = entities.get("oci_profile")
        hours_back = entities.get("hours_back", 24)

        params: dict[str, Any] = {"hours_back": hours_back}
        if compartment_id:
            params["compartment_id"] = compartment_id
        if profile:
            params["profile"] = profile

        result = await tool_catalog.execute(
            "oci_logan_get_summary",
            params,
        )

        extracted = _extract_result(result)
        if not extracted or extracted.startswith("Error"):
            return "Error getting log summary: Unable to retrieve Log Analytics data."

        return extracted

    except Exception as e:
        logger.error("log_summary workflow failed", error=str(e))
        return f"Error getting log summary: {e}"


async def log_search_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Search logs for errors or specific text.

    Executes a Log Analytics query to find matching log entries.

    Matches intents: log_search, search_logs, find_logs, error_logs,
                     search_for_errors
    """
    try:
        compartment_id = entities.get("compartment_id")
        profile = entities.get("oci_profile")
        hours_back = entities.get("hours_back", 24)

        # Extract search term from query
        search_text = "error"  # Default to searching for errors
        query_lower = query.lower()
        if "warning" in query_lower:
            search_text = "warning"
        elif "exception" in query_lower:
            search_text = "exception"
        elif "fail" in query_lower:
            search_text = "fail"

        params: dict[str, Any] = {
            "search_text": search_text,
            "hours_back": hours_back,
            "limit": 100,
        }
        if compartment_id:
            params["compartment_id"] = compartment_id
        if profile:
            params["profile"] = profile

        result = await tool_catalog.execute(
            "oci_logan_search_logs",
            params,
        )

        extracted = _extract_result(result)
        if not extracted or extracted.startswith("Error"):
            return "Error searching logs: Unable to retrieve log data."

        return extracted

    except Exception as e:
        logger.error("log_search workflow failed", error=str(e))
        return f"Error searching logs: {e}"


async def error_analysis_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Analyze error messages and provide explanations.

    This workflow handles queries where users paste error messages, stack traces,
    or exceptions and want help understanding what went wrong.

    Uses LLM analysis to explain the error and suggest fixes.

    Matches intents: error_analysis, explain_error, what_is_this_error,
                     help_with_error, understand_exception
    """
    try:
        # The error message is the query itself or extracted from entities
        error_message = entities.get("error_message") or query

        # Use the LLM to analyze the error
        from src.memory.context import SharedMemoryManager
        from src.llm.provider import get_llm_provider

        llm = get_llm_provider()

        analysis_prompt = f"""You are an expert software engineer. Analyze the following error message or exception and provide:

1. **What went wrong**: A clear explanation of what the error means
2. **Likely cause**: The most probable reason(s) this error occurred
3. **How to fix it**: Concrete steps to resolve the issue
4. **Prevention tips**: How to avoid this error in the future

Error to analyze:
```
{error_message}
```

Provide a helpful, educational response that a developer would find useful."""

        # Call LLM for analysis
        response = await llm.ainvoke(analysis_prompt)

        # Extract content from response
        if hasattr(response, "content"):
            analysis = response.content
        else:
            analysis = str(response)

        return f"# Error Analysis\n\n{analysis}"

    except Exception as e:
        logger.error("error_analysis workflow failed", error=str(e))
        return f"Error analyzing the error message: {e}\n\nPlease share more details about the error you're encountering."


# ─────────────────────────────────────────────────────────────────────────────
# Security Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def security_overview_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get comprehensive security posture overview.

    Combines Cloud Guard problems, security score, and audit activity
    to provide a holistic security view with recommendations.

    Matches intents: security_overview, show_security, security_status,
                     security_posture, cloud_guard_summary
    """
    try:
        compartment_id = entities.get("compartment_id")
        profile = entities.get("oci_profile")

        params: dict[str, Any] = {}
        if compartment_id:
            params["compartment_id"] = compartment_id
        if profile:
            params["profile"] = profile

        result = await tool_catalog.execute(
            "oci_security_overview",
            params,
        )

        extracted = _extract_result(result)
        if not extracted or extracted.startswith("Error"):
            return "Error getting security overview: Unable to retrieve security data."

        return extracted

    except Exception as e:
        logger.error("security_overview workflow failed", error=str(e))
        return f"Error getting security overview: {e}"


async def cloud_guard_problems_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List Cloud Guard security problems.

    Returns problems with severity summary and risk level breakdown.

    Matches intents: cloud_guard_problems, list_security_problems,
                     security_issues, cloud_guard_findings
    """
    try:
        compartment_id = entities.get("compartment_id")
        profile = entities.get("oci_profile")
        risk_level = entities.get("risk_level")  # CRITICAL, HIGH, MEDIUM, LOW

        params: dict[str, Any] = {"limit": 50}
        if compartment_id:
            params["compartment_id"] = compartment_id
        if profile:
            params["profile"] = profile
        if risk_level:
            params["risk_level"] = risk_level.upper()

        result = await tool_catalog.execute(
            "oci_security_cloudguard_list_problems",
            params,
        )

        extracted = _extract_result(result)
        if not extracted or extracted.startswith("Error"):
            return "Error listing Cloud Guard problems: Unable to retrieve security problems."

        return extracted

    except Exception as e:
        logger.error("cloud_guard_problems workflow failed", error=str(e))
        return f"Error listing Cloud Guard problems: {e}"


async def security_score_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Get Cloud Guard security score.

    Returns security score, grade (A-F), and problem counts by severity.

    Matches intents: security_score, get_security_score, show_security_score,
                     security_grade, security_rating
    """
    try:
        compartment_id = entities.get("compartment_id")
        profile = entities.get("oci_profile")

        params: dict[str, Any] = {}
        if compartment_id:
            params["compartment_id"] = compartment_id
        if profile:
            params["profile"] = profile

        result = await tool_catalog.execute(
            "oci_security_cloudguard_get_security_score",
            params,
        )

        extracted = _extract_result(result)
        if not extracted or extracted.startswith("Error"):
            return "Error getting security score: Unable to retrieve score data."

        return extracted

    except Exception as e:
        logger.error("security_score workflow failed", error=str(e))
        return f"Error getting security score: {e}"


async def audit_events_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List recent audit events.

    Returns audit events with event type summary and time range.

    Matches intents: audit_events, list_audit_events, show_audit,
                     recent_audit, audit_log
    """
    try:
        compartment_id = entities.get("compartment_id")
        profile = entities.get("oci_profile")
        hours_back = entities.get("hours_back", 24)

        params: dict[str, Any] = {"hours_back": hours_back, "limit": 100}
        if compartment_id:
            params["compartment_id"] = compartment_id
        if profile:
            params["profile"] = profile

        result = await tool_catalog.execute(
            "oci_security_audit_list_events",
            params,
        )

        extracted = _extract_result(result)
        if not extracted or extracted.startswith("Error"):
            return "Error listing audit events: Unable to retrieve audit data."

        return extracted

    except Exception as e:
        logger.error("audit_events workflow failed", error=str(e))
        return f"Error listing audit events: {e}"


async def security_threats_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Analyze security threats and potential indicators.

    Combines Cloud Guard threat intelligence with MITRE ATT&CK mapping
    to provide actionable threat analysis and recommendations.

    Matches intents: security_threats, threat_analysis, threat_detection,
                     show_threats, list_threats
    """
    import asyncio
    import json

    try:
        compartment_id = entities.get("compartment_id")
        profile = entities.get("oci_profile")

        # Gather threat data from multiple sources in parallel
        async def get_cloud_guard_problems() -> dict[str, Any]:
            params: dict[str, Any] = {"limit": 50}
            if compartment_id:
                params["compartment_id"] = compartment_id
            if profile:
                params["profile"] = profile
            # Focus on high-severity problems that may indicate threats
            params["risk_level"] = "CRITICAL"

            try:
                result = await tool_catalog.execute(
                    "oci_security_cloudguard_list_problems",
                    params,
                )
                return {"source": "cloud_guard_critical", "data": _extract_result(result)}
            except Exception as e:
                return {"source": "cloud_guard_critical", "error": str(e)}

        async def get_high_problems() -> dict[str, Any]:
            params: dict[str, Any] = {"limit": 50}
            if compartment_id:
                params["compartment_id"] = compartment_id
            if profile:
                params["profile"] = profile
            params["risk_level"] = "HIGH"

            try:
                result = await tool_catalog.execute(
                    "oci_security_cloudguard_list_problems",
                    params,
                )
                return {"source": "cloud_guard_high", "data": _extract_result(result)}
            except Exception as e:
                return {"source": "cloud_guard_high", "error": str(e)}

        async def get_security_score() -> dict[str, Any]:
            params: dict[str, Any] = {}
            if compartment_id:
                params["compartment_id"] = compartment_id
            if profile:
                params["profile"] = profile

            try:
                result = await tool_catalog.execute(
                    "oci_security_cloudguard_get_security_score",
                    params,
                )
                return {"source": "security_score", "data": _extract_result(result)}
            except Exception as e:
                return {"source": "security_score", "error": str(e)}

        async def get_recent_audit() -> dict[str, Any]:
            params: dict[str, Any] = {"hours_back": 24, "limit": 50}
            if compartment_id:
                params["compartment_id"] = compartment_id
            if profile:
                params["profile"] = profile

            try:
                result = await tool_catalog.execute(
                    "oci_security_audit_list_events",
                    params,
                )
                return {"source": "audit_events", "data": _extract_result(result)}
            except Exception as e:
                return {"source": "audit_events", "error": str(e)}

        # Run all queries in parallel
        results = await asyncio.gather(
            get_cloud_guard_problems(),
            get_high_problems(),
            get_security_score(),
            get_recent_audit(),
            return_exceptions=True,
        )

        # Build threat analysis report
        threat_report: dict[str, Any] = {
            "type": "security_threats",
            "generated_at": datetime.now(UTC).isoformat(),
            "compartment_id": compartment_id or "root",
            "critical_threats": [],
            "high_risk_issues": [],
            "security_score": None,
            "recent_suspicious_events": [],
            "mitre_mappings": [],
            "recommendations": [],
        }

        # MITRE ATT&CK mapping for common Cloud Guard detectors
        mitre_mapping = {
            "PRIVILEGED_ACCESS": {"technique": "T1078", "tactic": "Privilege Escalation"},
            "SUSPICIOUS_ACTIVITY": {"technique": "T1071", "tactic": "Command and Control"},
            "DATA_EXFILTRATION": {"technique": "T1041", "tactic": "Exfiltration"},
            "ANOMALY": {"technique": "T1078.004", "tactic": "Initial Access"},
            "BRUTE_FORCE": {"technique": "T1110", "tactic": "Credential Access"},
            "CONFIGURATION": {"technique": "T1562", "tactic": "Defense Evasion"},
            "NETWORK": {"technique": "T1090", "tactic": "Command and Control"},
            "IAM": {"technique": "T1098", "tactic": "Persistence"},
        }

        for result in results:
            if isinstance(result, Exception):
                continue
            if result.get("error"):
                continue

            source = result.get("source", "")
            data = result.get("data", "")

            if source == "cloud_guard_critical" and data:
                # Parse critical problems
                try:
                    problems = json.loads(data) if isinstance(data, str) else data
                    if isinstance(problems, list):
                        threat_report["critical_threats"] = problems[:10]
                        # Add MITRE mappings
                        for problem in problems[:10]:
                            detector = problem.get("detector_id", "").upper()
                            for key, mapping in mitre_mapping.items():
                                if key in detector:
                                    threat_report["mitre_mappings"].append({
                                        "problem": problem.get("name", "Unknown"),
                                        "technique_id": mapping["technique"],
                                        "tactic": mapping["tactic"],
                                    })
                                    break
                except (json.JSONDecodeError, TypeError):
                    if "critical" in data.lower() or "threat" in data.lower():
                        threat_report["critical_threats"].append({"raw": data[:500]})

            elif source == "cloud_guard_high" and data:
                try:
                    problems = json.loads(data) if isinstance(data, str) else data
                    if isinstance(problems, list):
                        threat_report["high_risk_issues"] = problems[:10]
                except (json.JSONDecodeError, TypeError):
                    pass

            elif source == "security_score" and data:
                try:
                    score_data = json.loads(data) if isinstance(data, str) else data
                    if isinstance(score_data, dict):
                        threat_report["security_score"] = score_data
                except (json.JSONDecodeError, TypeError):
                    pass

            elif source == "audit_events" and data:
                try:
                    events = json.loads(data) if isinstance(data, str) else data
                    if isinstance(events, list):
                        # Filter for suspicious events
                        suspicious = [
                            e for e in events
                            if any(keyword in str(e).lower() for keyword in [
                                "delete", "create user", "update policy",
                                "modify security", "change password", "failed login"
                            ])
                        ]
                        threat_report["recent_suspicious_events"] = suspicious[:10]
                except (json.JSONDecodeError, TypeError):
                    pass

        # Generate recommendations based on findings
        if threat_report["critical_threats"]:
            threat_report["recommendations"].append({
                "priority": "CRITICAL",
                "action": "Immediately investigate and remediate critical Cloud Guard findings",
                "count": len(threat_report["critical_threats"]),
            })

        if threat_report["high_risk_issues"]:
            threat_report["recommendations"].append({
                "priority": "HIGH",
                "action": "Review and address high-risk security issues within 24 hours",
                "count": len(threat_report["high_risk_issues"]),
            })

        score = threat_report.get("security_score")
        if score and isinstance(score, dict):
            score_value = score.get("score", score.get("security_score", 0))
            if isinstance(score_value, (int, float)) and score_value < 50:
                threat_report["recommendations"].append({
                    "priority": "HIGH",
                    "action": f"Security score is {score_value}/100. Review Cloud Guard recommendations to improve posture.",
                })

        if threat_report["mitre_mappings"]:
            threat_report["recommendations"].append({
                "priority": "MEDIUM",
                "action": "Review MITRE ATT&CK mapped threats and implement defensive controls",
                "techniques": list(set(m["technique_id"] for m in threat_report["mitre_mappings"])),
            })

        return json.dumps(threat_report, indent=2, default=str)

    except Exception as e:
        logger.error("security_threats workflow failed", error=str(e))
        return f"Error analyzing security threats: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure Provisioning Workflows
# ─────────────────────────────────────────────────────────────────────────────


# Predefined instance size configurations
INSTANCE_SIZE_OPTIONS = {
    "1": {
        "name": "Small (Dev/Test)",
        "shape": "VM.Standard.E4.Flex",
        "ocpus": 1,
        "memory_gb": 8,
        "boot_volume_size_gb": 50,
        "description": "1 OCPU, 8 GB RAM, 50 GB boot volume",
    },
    "2": {
        "name": "Medium (Standard)",
        "shape": "VM.Standard.E4.Flex",
        "ocpus": 2,
        "memory_gb": 16,
        "boot_volume_size_gb": 100,
        "description": "2 OCPUs, 16 GB RAM, 100 GB boot volume",
    },
    "3": {
        "name": "Large (Production)",
        "shape": "VM.Standard.E4.Flex",
        "ocpus": 4,
        "memory_gb": 32,
        "boot_volume_size_gb": 200,
        "description": "4 OCPUs, 32 GB RAM, 200 GB boot volume",
    },
    "4": {
        "name": "XLarge (High Performance)",
        "shape": "VM.Standard.E4.Flex",
        "ocpus": 8,
        "memory_gb": 64,
        "boot_volume_size_gb": 500,
        "description": "8 OCPUs, 64 GB RAM, 500 GB boot volume",
    },
}


def _get_provisioning_defaults() -> dict[str, str | None]:
    """Get provisioning defaults from environment variables."""
    import os

    return {
        "compartment_id": os.getenv("DEFAULT_COMPARTMENT_ID") or _get_root_compartment(),
        "availability_domain": os.getenv("DEFAULT_AVAILABILITY_DOMAIN"),
        "subnet_id": os.getenv("DEFAULT_SUBNET_ID"),
        "ssh_public_key": os.getenv("SSH_PUBLIC_KEY"),
    }


async def provision_instance_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Interactive instance provisioning workflow with predefined size options.

    Shows users available instance sizes (Small/Medium/Large/XLarge) and allows
    them to select by number. Supports custom configurations via entities.

    Requires ALLOW_MUTATIONS=true to actually launch instances.
    """
    import os

    try:
        # Check if this is a follow-up with a size selection
        size_selection = entities.get("size_selection") or entities.get("option")
        instance_name = entities.get("instance_name") or entities.get("name")

        # If no selection yet, show options
        if not size_selection:
            options_text = [
                "# 🖥️ Instance Provisioning",
                "",
                "Select an instance size (reply with number or 'cancel'):",
                "",
            ]
            for num, config in INSTANCE_SIZE_OPTIONS.items():
                options_text.append(f"**{num}. {config['name']}**")
                options_text.append(f"   {config['description']}")
                options_text.append("")

            options_text.extend([
                "---",
                "*Reply with the number (1-4) to proceed, or 'cancel' to abort.*",
                "",
                "**Note**: You'll be asked to confirm the instance name before creation.",
            ])

            return "\n".join(options_text)

        # User selected a size
        if size_selection.lower() in ("cancel", "abort", "no"):
            return "Instance provisioning cancelled."

        if size_selection not in INSTANCE_SIZE_OPTIONS:
            return f"Invalid selection '{size_selection}'. Please reply with 1, 2, 3, 4, or 'cancel'."

        config = INSTANCE_SIZE_OPTIONS[size_selection]

        # Get defaults from environment
        defaults = _get_provisioning_defaults()

        # Check required configuration
        missing = []
        if not defaults["availability_domain"]:
            missing.append("DEFAULT_AVAILABILITY_DOMAIN")
        if not defaults["subnet_id"]:
            missing.append("DEFAULT_SUBNET_ID")

        if missing:
            return (
                f"**Configuration Required**\n\n"
                f"Missing environment variables: {', '.join(missing)}\n\n"
                f"Please set these in your `.env.local` file:\n"
                f"```\n"
                f"DEFAULT_AVAILABILITY_DOMAIN=<your-AD>  # e.g., Uocm:EU-FRANKFURT-1-AD-1\n"
                f"DEFAULT_SUBNET_ID=<your-subnet-ocid>\n"
                f"DEFAULT_COMPARTMENT_ID=<your-compartment-ocid>\n"
                f"SSH_PUBLIC_KEY=<your-ssh-public-key>  # Optional\n"
                f"ALLOW_MUTATIONS=true  # Required to launch instances\n"
                f"```"
            )

        # Check if mutations are allowed
        allow_mutations = os.getenv("ALLOW_MUTATIONS", "").lower() in {"1", "true", "yes"}

        # If no instance name yet, show confirmation
        if not instance_name:
            confirmation = [
                f"# Confirm Instance Creation",
                "",
                f"**Size**: {config['name']}",
                f"**Configuration**: {config['description']}",
                f"**Shape**: {config['shape']}",
                f"**Compartment**: {defaults['compartment_id'][:40]}...",
                f"**Availability Domain**: {defaults['availability_domain']}",
                "",
            ]

            if not allow_mutations:
                confirmation.extend([
                    "⚠️ **Dry Run Mode**: ALLOW_MUTATIONS is not enabled.",
                    "Set `ALLOW_MUTATIONS=true` in your environment to actually create instances.",
                    "",
                ])

            confirmation.extend([
                "**Please provide an instance name** to proceed.",
                "",
                "*Example: `create instance my-dev-server`*",
            ])

            return "\n".join(confirmation)

        # Ready to launch - call the MCP tool
        launch_params = {
            "display_name": instance_name,
            "compartment_id": defaults["compartment_id"],
            "availability_domain": defaults["availability_domain"],
            "subnet_id": defaults["subnet_id"],
            "shape": config["shape"],
            "ocpus": config["ocpus"],
            "memory_gb": config["memory_gb"],
            "boot_volume_size_gb": config["boot_volume_size_gb"],
            "assign_public_ip": True,
        }

        if defaults["ssh_public_key"]:
            launch_params["ssh_public_key"] = defaults["ssh_public_key"]

        result = await tool_catalog.execute("oci_compute_launch_instance", launch_params)
        result_str = _extract_result(result)

        # Parse result and format response
        try:
            result_data = json.loads(result_str)
            if result_data.get("error"):
                return (
                    f"**Instance Creation Failed**\n\n"
                    f"Error: {result_data.get('message', result_data['error'])}\n\n"
                    f"Suggestion: {result_data.get('suggestion', 'Check your configuration.')}"
                )

            instance = result_data.get("instance", {})
            return (
                f"# ✅ Instance Launched Successfully\n\n"
                f"**Name**: {instance.get('name', instance_name)}\n"
                f"**ID**: `{instance.get('id', 'pending')}`\n"
                f"**State**: {instance.get('state', 'PROVISIONING')}\n"
                f"**Shape**: {instance.get('shape', config['shape'])}\n"
                f"**Configuration**: {config['description']}\n\n"
                f"---\n"
                f"*Note: Instance provisioning takes 2-5 minutes. "
                f"Use `get instance {instance_name}` to check status.*"
            )
        except json.JSONDecodeError:
            return f"Instance launch initiated. Response:\n\n{result_str}"

    except Exception as e:
        logger.error("provision_instance workflow failed", error=str(e))
        return f"Error in instance provisioning: {e}"


async def list_instance_shapes_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List available compute shapes for instance provisioning.
    """
    try:
        compartment_id = entities.get("compartment_id") or _get_root_compartment()
        if not compartment_id:
            return "Error: No compartment ID found. Set OCI_COMPARTMENT_ID or DEFAULT_COMPARTMENT_ID."

        result = await tool_catalog.execute(
            "oci_compute_list_shapes",
            {"compartment_id": compartment_id},
        )
        result_str = _extract_result(result)

        try:
            shapes = json.loads(result_str)
            if isinstance(shapes, list):
                # Format as markdown
                lines = ["# Available Compute Shapes\n"]

                # Group by type
                flex_shapes = [s for s in shapes if s.get("is_flexible")]
                fixed_shapes = [s for s in shapes if not s.get("is_flexible")]

                if flex_shapes:
                    lines.append("## Flex Shapes (Configurable)")
                    for s in flex_shapes[:10]:
                        lines.append(f"- **{s['name']}**: Up to {s.get('max_ocpus', '?')} OCPUs, "
                                   f"{s.get('max_memory_gb', '?')} GB RAM")
                    lines.append("")

                if fixed_shapes:
                    lines.append("## Standard Shapes")
                    for s in fixed_shapes[:10]:
                        ocpus = s.get('ocpus', '?')
                        mem = s.get('memory_gb', '?')
                        lines.append(f"- **{s['name']}**: {ocpus} OCPUs, {mem} GB RAM")

                return "\n".join(lines)
            return result_str
        except json.JSONDecodeError:
            return result_str

    except Exception as e:
        logger.error("list_instance_shapes workflow failed", error=str(e))
        return f"Error listing shapes: {e}"


async def list_instance_images_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List available compute images for instance provisioning.
    """
    try:
        compartment_id = entities.get("compartment_id") or _get_root_compartment()
        if not compartment_id:
            return "Error: No compartment ID found."

        os_filter = entities.get("operating_system", "Oracle Linux")

        result = await tool_catalog.execute(
            "oci_compute_list_images",
            {"compartment_id": compartment_id, "operating_system": os_filter, "limit": 10},
        )
        result_str = _extract_result(result)

        try:
            images = json.loads(result_str)
            if isinstance(images, list):
                lines = [f"# Available {os_filter} Images\n"]
                for img in images[:10]:
                    lines.append(f"- **{img.get('name', 'Unknown')}**")
                    lines.append(f"  OS: {img.get('operating_system', '?')} {img.get('operating_system_version', '')}")
                    if img.get('size_gb'):
                        lines.append(f"  Size: {img['size_gb']} GB")
                    lines.append("")
                return "\n".join(lines)
            return result_str
        except json.JSONDecodeError:
            return result_str

    except Exception as e:
        logger.error("list_instance_images workflow failed", error=str(e))
        return f"Error listing images: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# SelectAI Workflows
# ─────────────────────────────────────────────────────────────────────────────


async def selectai_list_tables_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List tables associated with a SelectAI profile.

    Uses oci_selectai_get_profile_tables to retrieve the tables that
    a profile has been configured to access for natural language queries.

    Matches intents: list_tables, show_tables, selectai_tables,
                     profile_tables, database_tables
    """
    try:
        # Extract profile name from entities or use default
        profile_name = (
            entities.get("profile_name")
            or entities.get("database_name")
            or entities.get("connection_name")
        )

        if not profile_name:
            # Try to extract from query
            import re
            match = re.search(r'(?:on|for|profile|selectai)\s+([a-zA-Z][a-zA-Z0-9_]*)', query, re.IGNORECASE)
            if match:
                profile_name = match.group(1)

        if not profile_name:
            return "Please specify a SelectAI profile name (e.g., 'show tables on MY_PROFILE')"

        logger.info("SelectAI list tables workflow", profile_name=profile_name)

        result = await tool_catalog.execute(
            "oci_selectai_get_profile_tables",
            {"profile_name": profile_name},
        )

        extracted = _extract_result(result)
        if extracted and not extracted.startswith("Error"):
            # Parse JSON response for better formatting
            try:
                data = json.loads(extracted)
                if data.get("type") == "selectai_profile_tables":
                    tables = data.get("tables", [])
                    if tables:
                        lines = [f"**Tables for SelectAI Profile '{profile_name}'** ({len(tables)} tables)\n"]
                        for i, table in enumerate(tables, 1):
                            lines.append(f"{i}. `{table}`")
                        return "\n".join(lines)
                    else:
                        return f"No tables configured for SelectAI profile '{profile_name}'"
            except json.JSONDecodeError:
                pass
            return f"**Tables for SelectAI Profile '{profile_name}'**\n\n{extracted}"

        return f"Error getting tables for profile '{profile_name}': {extracted}"

    except Exception as e:
        logger.error("selectai_list_tables workflow failed", error=str(e))
        return f"Error listing SelectAI tables: {e}"


async def selectai_generate_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Execute a SelectAI natural language query.

    Translates natural language queries into SQL and executes them
    using Oracle Autonomous Database's SelectAI feature.

    Matches intents: selectai_query, nl_query, natural_language_sql,
                     ask_database, ask_selectai
    """
    try:
        # Extract profile name and query
        profile_name = entities.get("profile_name") or entities.get("database_name")
        user_query = entities.get("user_query") or query
        action = entities.get("action", "runsql")

        logger.info("SelectAI generate workflow", profile=profile_name, action=action)

        params = {
            "prompt": user_query,
            "action": action,
        }
        if profile_name:
            params["profile_name"] = profile_name

        result = await tool_catalog.execute("oci_selectai_generate", params)

        extracted = _extract_result(result)
        return f"**SelectAI Results**\n\n{extracted}"

    except Exception as e:
        logger.error("selectai_generate workflow failed", error=str(e))
        return f"Error executing SelectAI query: {e}"


async def db_schema_tables_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    List tables from a database schema via SQLcl connection.

    Uses oci_database_get_schema (from database-observatory) to retrieve
    tables, views, and other objects from a database schema accessible
    via a SQLcl connection name.

    This is different from SelectAI profile tables - this queries actual
    database schema metadata using the Oracle data dictionary.

    Matches intents: db_schema_tables, schema_tables, connection_tables,
                     list_schema_objects, show_database_tables

    Examples:
        - "show me the tables from ATPAdi_high"
        - "list tables on th_high"
        - "what tables are in the ADMIN schema on ATPAdi"
    """
    try:
        # Extract connection name and schema from entities or query
        connection_name = (
            entities.get("connection_name")
            or entities.get("database_name")
            or entities.get("profile_name")  # May be confused with SelectAI profile
        )
        schema_name = entities.get("schema_name", "")

        if not connection_name:
            # Try to extract connection name from query patterns
            # Patterns: "from X", "on X", "in X connection", "connection X"
            import re

            # Look for known connection patterns (e.g., ATPAdi_high, th_high, SelectAI)
            conn_patterns = [
                r'(?:from|on|in|connection)\s+([a-zA-Z][a-zA-Z0-9_]*(?:_(?:high|medium|low|tp))?)',
                r'([a-zA-Z][a-zA-Z0-9_]*(?:_(?:high|medium|low|tp)))\s+(?:database|connection|db)',
            ]

            for pattern in conn_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    connection_name = match.group(1)
                    break

        if not connection_name:
            # List available connections for user guidance
            try:
                connections_result = await tool_catalog.execute(
                    "oci_database_list_connections",
                    {},
                )
                conn_extracted = _extract_result(connections_result)

                # Parse connections for helpful message
                available_conns = []
                try:
                    conn_data = json.loads(conn_extracted)
                    if isinstance(conn_data, dict):
                        conns = conn_data.get("connections", [])
                        if isinstance(conns, list):
                            available_conns = [
                                c.get("name", str(c)) if isinstance(c, dict) else str(c)
                                for c in conns[:10]
                            ]
                except (json.JSONDecodeError, TypeError):
                    pass

                if available_conns:
                    conn_list = ", ".join(f"`{c}`" for c in available_conns)
                    return (
                        f"Please specify a connection name. Available connections: {conn_list}\n\n"
                        f"Example: 'show tables from ATPAdi_high'"
                    )
            except Exception:
                pass

            return (
                "Please specify a database connection name.\n\n"
                "Example: 'show tables from ATPAdi_high' or 'list tables on th_high'"
            )

        # Extract schema name if not provided
        if not schema_name:
            # Try to extract from query (e.g., "in ADMIN schema", "schema HR")
            import re
            schema_match = re.search(
                r'(?:schema|in)\s+([A-Z][A-Z0-9_$#]*)',
                query, re.IGNORECASE
            )
            if schema_match:
                schema_name = schema_match.group(1).upper()
            else:
                # Default to common schema or use connected user's schema
                # Most autonomous databases use ADMIN
                schema_name = "ADMIN"

        logger.info(
            "db_schema_tables workflow",
            connection=connection_name,
            schema=schema_name
        )

        # Use oci_database_get_schema to get tables
        result = await tool_catalog.execute(
            "oci_database_get_schema",
            {
                "schema": schema_name,
                "connection": connection_name,
                "object_types": "TABLES,VIEWS",
                "include_columns": True,
                "include_comments": True,
            },
        )

        extracted = _extract_result(result)

        if extracted and not extracted.startswith("Error"):
            # Try to parse and format nicely
            try:
                data = json.loads(extracted)

                # Handle the schema info response format
                if isinstance(data, dict):
                    tables = data.get("tables", [])
                    views = data.get("views", [])
                    error = data.get("error")

                    if error:
                        return f"Error querying schema on `{connection_name}`: {error}"

                    lines = [
                        f"**Database Schema: `{schema_name}` on `{connection_name}`**\n"
                    ]

                    if tables:
                        lines.append(f"\n### Tables ({len(tables)})\n")
                        for i, table in enumerate(tables, 1):
                            if isinstance(table, dict):
                                name = table.get("name", table.get("table_name", "unknown"))
                                comment = table.get("comment", "")
                                cols = table.get("columns", [])
                                col_count = len(cols) if cols else ""
                                col_info = f" ({col_count} columns)" if col_count else ""

                                lines.append(f"{i}. `{name}`{col_info}")
                                if comment:
                                    lines.append(f"   - {comment}")
                            else:
                                lines.append(f"{i}. `{table}`")

                    if views:
                        lines.append(f"\n### Views ({len(views)})\n")
                        for i, view in enumerate(views, 1):
                            if isinstance(view, dict):
                                name = view.get("name", view.get("view_name", "unknown"))
                                lines.append(f"{i}. `{name}`")
                            else:
                                lines.append(f"{i}. `{view}`")

                    if not tables and not views:
                        lines.append(f"\nNo tables or views found in schema `{schema_name}`.")
                        lines.append(
                            "\nTip: Try a different schema name or check if "
                            "the connection has access to this schema."
                        )

                    return "\n".join(lines)

            except json.JSONDecodeError:
                # Not JSON, return raw result formatted
                pass

            return (
                f"**Tables in `{schema_name}` on `{connection_name}`**\n\n{extracted}"
            )

        return f"Error getting schema for `{connection_name}`: {extracted}"

    except Exception as e:
        logger.error("db_schema_tables workflow failed", error=str(e))
        return f"Error listing database tables: {e}"


async def db_execute_sql_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
) -> str:
    """
    Execute a SQL query against a database via SQLcl connection.

    Uses oci_database_execute_sql (from database-observatory) to run
    arbitrary SQL queries against an Oracle database.

    This enables direct SQL access similar to what the Cline MCP SQLcl
    server provides.

    Matches intents: execute_sql, run_sql, sql_query, database_query

    Examples:
        - "run SELECT * FROM employees on ATPAdi_high"
        - "execute SQL: SELECT COUNT(*) FROM orders"
    """
    try:
        # Extract SQL and connection from entities
        sql_query = entities.get("sql") or entities.get("sql_query", "")
        connection_name = (
            entities.get("connection_name")
            or entities.get("database_name")
        )

        if not sql_query:
            # Try to extract SQL from the query itself
            import re
            # Look for SQL patterns after keywords like "run", "execute"
            sql_match = re.search(
                r'(?:run|execute|sql:?)\s*(SELECT\s+.+?)(?:\s+on\s+|\s*$)',
                query, re.IGNORECASE | re.DOTALL
            )
            if sql_match:
                sql_query = sql_match.group(1).strip()

        if not sql_query:
            return (
                "Please provide a SQL query to execute.\n\n"
                "Example: 'run SELECT * FROM employees FETCH FIRST 10 ROWS ONLY on ATPAdi_high'"
            )

        # Extract connection if not in entities
        if not connection_name:
            import re
            conn_match = re.search(
                r'(?:on|using|connection)\s+([a-zA-Z][a-zA-Z0-9_]*(?:_(?:high|medium|low|tp))?)',
                query, re.IGNORECASE
            )
            if conn_match:
                connection_name = conn_match.group(1)

        logger.info(
            "db_execute_sql workflow",
            connection=connection_name,
            sql_length=len(sql_query)
        )

        # Build parameters
        params = {"sql": sql_query}
        if connection_name:
            params["connection"] = connection_name

        result = await tool_catalog.execute(
            "oci_database_execute_sql",
            params,
        )

        extracted = _extract_result(result)

        if extracted and not extracted.startswith("Error"):
            conn_info = f" (on `{connection_name}`)" if connection_name else ""
            return f"**SQL Results{conn_info}**\n\n{extracted}"

        return f"Error executing SQL: {extracted}"

    except Exception as e:
        logger.error("db_execute_sql workflow failed", error=str(e))
        return f"Error executing SQL query: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection Workflows (Logs/Metrics Correlation)
# ─────────────────────────────────────────────────────────────────────────────


async def anomaly_detection_workflow(
    query: str,
    entities: dict[str, Any],
    tool_catalog: ToolCatalog,
    memory: SharedMemoryManager,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Detect anomalies by correlating metrics and logs.

    This workflow:
    1. Gets instance metrics (CPU, memory, network, disk)
    2. Queries logs for the same time period
    3. Analyzes patterns to identify anomalies
    4. Correlates metric spikes with log events

    Matches intents: anomaly_detection, detect_anomalies, find_anomalies,
                     correlate_logs_metrics, analyze_anomalies
    """
    try:
        # Get instance identifier from entities
        instance_id = entities.get("instance_id")
        instance_name = entities.get("instance_name") or entities.get("name")

        # Extract instance name from query if not in entities
        if not instance_name and not instance_id:
            instance_name = _extract_instance_name_for_metrics(query)

        # If still no instance specified, ask the user
        if not instance_name and not instance_id:
            return json.dumps({
                "type": "clarification_needed",
                "message": "Please provide the instance name for anomaly detection. Which instance should I analyze?",
                "examples": [
                    "detect anomalies on my-web-server",
                    "find anomalies for app-server-01",
                    "analyze health of database-vm"
                ],
                "suggestion": "Use 'list instances' to see available instances"
            })

        # Get compartment from context or entities
        compartment_id = entities.get("compartment_id")
        thread_id = metadata.get("thread_id") if metadata else None
        if not compartment_id and thread_id and memory:
            context = await memory.get_conversation_context(thread_id)
            compartment_id = context.get("compartment_id")

        # Get time range (default 4 hours for anomaly detection)
        hours_back = entities.get("hours_back", 4)

        # Extract hours from query if specified
        query_lower = query.lower()
        import re
        if "last" in query_lower:
            match = re.search(r"last\s+(\d+)\s*(?:hour|hr)", query_lower)
            if match:
                hours_back = int(match.group(1))

        logger.info(
            "anomaly_detection workflow starting",
            instance_id=instance_id,
            instance_name=instance_name,
            hours_back=hours_back,
        )

        # Step 1: Get instance metrics
        metrics_params = {"hours_back": hours_back}
        if instance_id:
            metrics_params["instance_id"] = instance_id
        if instance_name:
            metrics_params["instance_name"] = instance_name
        if compartment_id:
            metrics_params["compartment_id"] = compartment_id

        metrics_result = await tool_catalog.execute(
            "oci_observability_get_instance_metrics",
            metrics_params,
        )
        metrics_data = _extract_result(metrics_result)

        # Step 2: Get logs for the same time period (if logan tools available)
        logs_data = None
        try:
            profile = entities.get("oci_profile") or (metadata.get("oci_profile") if metadata else None)
            log_params = {
                "hours_back": hours_back,
                "limit": 100,
            }
            if profile:
                log_params["profile"] = profile

            # Try to get log summary first
            log_result = await tool_catalog.execute(
                "oci_logan_get_summary",
                log_params,
            )
            logs_data = _extract_result(log_result)
        except Exception as log_err:
            logger.warning("Could not fetch logs for correlation", error=str(log_err))
            logs_data = None

        # Step 3: Build analysis response
        analysis = {
            "type": "anomaly_analysis",
            "instance": instance_name or instance_id,
            "time_range_hours": hours_back,
            "metrics": None,
            "logs": None,
            "anomalies": [],
            "correlations": [],
            "recommendations": [],
        }

        # Parse metrics data
        try:
            if metrics_data and not metrics_data.startswith("Error"):
                metrics_json = json.loads(metrics_data) if isinstance(metrics_data, str) else metrics_data
                analysis["metrics"] = metrics_json

                # Check for metric anomalies
                if isinstance(metrics_json, dict):
                    health = metrics_json.get("health_assessment", {})
                    if health.get("issues"):
                        for issue in health["issues"]:
                            analysis["anomalies"].append({
                                "type": "metric_issue",
                                "description": issue,
                                "source": "metrics"
                            })

                    # Check for high utilization
                    summary = metrics_json.get("summary", {})
                    if summary.get("avg_cpu_percent", 0) > 80:
                        analysis["anomalies"].append({
                            "type": "high_cpu",
                            "value": summary.get("avg_cpu_percent"),
                            "threshold": 80,
                            "description": f"High CPU utilization: {summary.get('avg_cpu_percent'):.1f}%"
                        })
                    if summary.get("avg_memory_percent", 0) > 85:
                        analysis["anomalies"].append({
                            "type": "high_memory",
                            "value": summary.get("avg_memory_percent"),
                            "threshold": 85,
                            "description": f"High memory utilization: {summary.get('avg_memory_percent'):.1f}%"
                        })
        except (json.JSONDecodeError, TypeError):
            analysis["metrics"] = {"raw": metrics_data}

        # Parse logs data
        try:
            if logs_data and not logs_data.startswith("Error"):
                logs_json = json.loads(logs_data) if isinstance(logs_data, str) else logs_data
                analysis["logs"] = logs_json

                # Check for log anomalies
                if isinstance(logs_json, dict):
                    if logs_json.get("error"):
                        analysis["logs"] = {"status": "unavailable", "reason": logs_json.get("error")}
        except (json.JSONDecodeError, TypeError):
            analysis["logs"] = {"raw": logs_data} if logs_data else None

        # Generate recommendations based on anomalies
        if analysis["anomalies"]:
            for anomaly in analysis["anomalies"]:
                if anomaly["type"] == "high_cpu":
                    analysis["recommendations"].append(
                        "Consider scaling up the instance or investigating CPU-intensive processes"
                    )
                elif anomaly["type"] == "high_memory":
                    analysis["recommendations"].append(
                        "Check for memory leaks or consider increasing instance memory"
                    )
                elif anomaly["type"] == "metric_issue":
                    analysis["recommendations"].append(
                        f"Investigate: {anomaly['description']}"
                    )

            # Add correlation suggestion if logs available
            if analysis["logs"] and not isinstance(analysis["logs"], dict) or not analysis["logs"].get("status") == "unavailable":
                analysis["correlations"].append({
                    "suggestion": "Review logs for error messages that correlate with metric spikes",
                    "time_range": f"Last {hours_back} hours"
                })
        else:
            analysis["summary"] = "No anomalies detected. System appears to be operating normally."

        return json.dumps(analysis, indent=2, default=str)

    except Exception as e:
        logger.error("anomaly_detection workflow failed", error=str(e))
        return json.dumps({
            "type": "error",
            "error": f"Error during anomaly detection: {e}",
            "suggestion": "Try specifying a valid instance name or check OCI permissions"
        })


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

    # Instance Metrics
    "instance_metrics": instance_metrics_workflow,
    "show_instance_metrics": instance_metrics_workflow,
    "get_instance_metrics": instance_metrics_workflow,
    "show_metrics": instance_metrics_workflow,
    "cpu_usage": instance_metrics_workflow,
    "memory_usage": instance_metrics_workflow,
    "instance_health": instance_metrics_workflow,
    "compute_metrics": instance_metrics_workflow,
    "vm_metrics": instance_metrics_workflow,

    # Infrastructure Provisioning
    "provision_instance": provision_instance_workflow,
    "create_instance": provision_instance_workflow,
    "launch_instance": provision_instance_workflow,
    "new_instance": provision_instance_workflow,
    "create_vm": provision_instance_workflow,
    "launch_vm": provision_instance_workflow,
    "provision_vm": provision_instance_workflow,
    "list_shapes": list_instance_shapes_workflow,
    "list_compute_shapes": list_instance_shapes_workflow,
    "available_shapes": list_instance_shapes_workflow,
    "list_images": list_instance_images_workflow,
    "list_compute_images": list_instance_images_workflow,
    "available_images": list_instance_images_workflow,

    # Instance Lifecycle (start/stop/restart by name)
    "start_instance": start_instance_by_name_workflow,
    "start_instance_by_name": start_instance_by_name_workflow,
    "start_vm": start_instance_by_name_workflow,
    "power_on": start_instance_by_name_workflow,
    "power_on_instance": start_instance_by_name_workflow,
    "stop_instance": stop_instance_by_name_workflow,
    "stop_instance_by_name": stop_instance_by_name_workflow,
    "stop_vm": stop_instance_by_name_workflow,
    "power_off": stop_instance_by_name_workflow,
    "shutdown_instance": stop_instance_by_name_workflow,
    "restart_instance": restart_instance_by_name_workflow,
    "restart_instance_by_name": restart_instance_by_name_workflow,
    "reboot_instance": restart_instance_by_name_workflow,
    "reboot_vm": restart_instance_by_name_workflow,
    "restart_vm": restart_instance_by_name_workflow,

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

    # Cost by service - detailed SKU-level breakdown
    "cost_by_service": cost_by_service_workflow,
    "service_drilldown": cost_by_service_workflow,
    "service_breakdown": cost_by_service_workflow,
    "show_costs_by_service": cost_by_service_workflow,
    "detailed_costs": cost_by_service_workflow,
    "service_cost_breakdown": cost_by_service_workflow,

    # Monthly cost trend
    "monthly_trend": monthly_cost_trend_workflow,
    "cost_trend": monthly_cost_trend_workflow,
    "show_cost_trend": monthly_cost_trend_workflow,
    "spending_trend": monthly_cost_trend_workflow,
    "month_over_month": monthly_cost_trend_workflow,
    "monthly_spending": monthly_cost_trend_workflow,

    # Cost comparison between months
    "cost_comparison": cost_comparison_workflow,
    "compare_costs": cost_comparison_workflow,
    "compare_months": cost_comparison_workflow,
    "month_comparison": cost_comparison_workflow,
    "compare_spending": cost_comparison_workflow,

    # Cost by compartment - direct tool call
    "cost_by_compartment_direct": cost_by_compartment_workflow,
    "compartment_spending": cost_by_compartment_workflow,
    "show_costs_by_compartment": cost_by_compartment_workflow,

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

    # SQLcl - SQL Monitoring (Real-time)
    "sql_monitoring": db_sql_monitoring_workflow,
    "sql_monitor": db_sql_monitoring_workflow,
    "active_sql": db_sql_monitoring_workflow,
    "running_queries": db_sql_monitoring_workflow,
    "execution_monitoring": db_sql_monitoring_workflow,
    "real_time_sql": db_sql_monitoring_workflow,
    "sql_monitor_report": db_sql_monitoring_workflow,

    # SQLcl - Long Running Operations
    "long_running_ops": db_long_running_ops_workflow,
    "longops": db_long_running_ops_workflow,
    "long_operations": db_long_running_ops_workflow,
    "running_operations": db_long_running_ops_workflow,
    "batch_progress": db_long_running_ops_workflow,
    "operation_status": db_long_running_ops_workflow,
    "session_longops": db_long_running_ops_workflow,

    # SQLcl - Parallelism Statistics
    "parallelism_stats": db_parallelism_stats_workflow,
    "parallel_query": db_parallelism_stats_workflow,
    "px_stats": db_parallelism_stats_workflow,
    "degree_comparison": db_parallelism_stats_workflow,
    "parallel_execution": db_parallelism_stats_workflow,
    "px_downgrade": db_parallelism_stats_workflow,
    "req_degree": db_parallelism_stats_workflow,
    "actual_degree": db_parallelism_stats_workflow,

    # SQLcl - Full Table Scan Detection
    "full_table_scan": db_full_table_scan_workflow,
    "table_scan": db_full_table_scan_workflow,
    "fts_detection": db_full_table_scan_workflow,
    "large_table_scan": db_full_table_scan_workflow,
    "missing_index": db_full_table_scan_workflow,
    "scan_analysis": db_full_table_scan_workflow,
    "full_scan": db_full_table_scan_workflow,

    # SQLcl - Blocking Sessions
    "blocking_sessions": db_blocking_sessions_workflow,
    "blocked_sessions": db_blocking_sessions_workflow,
    "lock_contention": db_blocking_sessions_workflow,
    "session_blocking": db_blocking_sessions_workflow,
    "lock_analysis": db_blocking_sessions_workflow,
    "database_locks": db_blocking_sessions_workflow,
    "check_blocking": db_blocking_sessions_workflow,

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

    # =========================================================================
    # SelectAI - DISABLED (using DB Observatory + SQLcl instead)
    # Uncomment these lines to re-enable SelectAI profile-based workflows
    # =========================================================================
    # "list_tables": selectai_list_tables_workflow,
    # "show_tables": selectai_list_tables_workflow,
    # "selectai_tables": selectai_list_tables_workflow,
    # "profile_tables": selectai_list_tables_workflow,
    # "database_tables": selectai_list_tables_workflow,
    # "get_tables": selectai_list_tables_workflow,
    # "tables_on": selectai_list_tables_workflow,
    # "selectai_query": selectai_generate_workflow,
    # "nl_query": selectai_generate_workflow,
    # "natural_language_sql": selectai_generate_workflow,
    # "ask_database": selectai_generate_workflow,
    # "ask_selectai": selectai_generate_workflow,
    # "selectai_generate": selectai_generate_workflow,
    # =========================================================================

    # SQLcl Direct - Schema Tables (via database-observatory)
    # Different from SelectAI profile tables - queries actual Oracle data dictionary
    "db_schema_tables": db_schema_tables_workflow,
    "schema_tables": db_schema_tables_workflow,
    "connection_tables": db_schema_tables_workflow,
    "list_schema_objects": db_schema_tables_workflow,
    "show_database_tables": db_schema_tables_workflow,
    "sqlcl_tables": db_schema_tables_workflow,
    "show_schema": db_schema_tables_workflow,

    # SQLcl Direct - Execute SQL (via database-observatory)
    "execute_sql": db_execute_sql_workflow,
    "run_sql": db_execute_sql_workflow,
    "sql_query": db_execute_sql_workflow,
    "database_query": db_execute_sql_workflow,
    "run_query": db_execute_sql_workflow,

    # Observability - Alarms and Monitoring
    "list_alarms": list_alarms_workflow,
    "show_alarms": list_alarms_workflow,
    "active_alarms": list_alarms_workflow,
    "monitoring_alarms": list_alarms_workflow,
    "get_alarms": list_alarms_workflow,
    "alarm_status": list_alarms_workflow,
    "check_alarms": list_alarms_workflow,

    # Log Analytics - Summary
    "log_summary": log_summary_workflow,
    "show_log_summary": log_summary_workflow,
    "logging_summary": log_summary_workflow,
    "log_analytics_status": log_summary_workflow,
    "logs_overview": log_summary_workflow,

    # Log Analytics - Search
    "log_search": log_search_workflow,
    "search_logs": log_search_workflow,
    "find_logs": log_search_workflow,
    "error_logs": log_search_workflow,
    "search_for_errors": log_search_workflow,
    "log_query": log_search_workflow,

    # Error Analysis - explain error messages, exceptions, stack traces
    "error_analysis": error_analysis_workflow,
    "explain_error": error_analysis_workflow,
    "what_is_this_error": error_analysis_workflow,
    "help_with_error": error_analysis_workflow,
    "understand_exception": error_analysis_workflow,
    "analyze_error": error_analysis_workflow,
    "error_explanation": error_analysis_workflow,

    # Security - Overview
    "security_overview": security_overview_workflow,
    "show_security": security_overview_workflow,
    "security_status": security_overview_workflow,
    "security_posture": security_overview_workflow,
    "cloud_guard_summary": security_overview_workflow,

    # Security - Cloud Guard Problems
    "cloud_guard_problems": cloud_guard_problems_workflow,
    "list_cloud_guard_problems": cloud_guard_problems_workflow,
    "security_problems": cloud_guard_problems_workflow,
    "security_issues": cloud_guard_problems_workflow,
    "cloud_guard_findings": cloud_guard_problems_workflow,

    # Security - Security Score
    "security_score": security_score_workflow,
    "get_security_score": security_score_workflow,
    "show_security_score": security_score_workflow,
    "security_grade": security_score_workflow,
    "security_rating": security_score_workflow,

    # Security - Audit Events
    "audit_events": audit_events_workflow,
    "list_audit_events": audit_events_workflow,
    "show_audit": audit_events_workflow,
    "recent_audit": audit_events_workflow,
    "audit_log": audit_events_workflow,

    # Security - Threats Analysis
    "security_threats": security_threats_workflow,
    "threat_analysis": security_threats_workflow,
    "threat_detection": security_threats_workflow,
    "show_threats": security_threats_workflow,
    "list_threats": security_threats_workflow,
    "mitre_analysis": security_threats_workflow,
    "threat_intelligence": security_threats_workflow,

    # Anomaly Detection - Logs/Metrics Correlation
    "anomaly_detection": anomaly_detection_workflow,
    "detect_anomalies": anomaly_detection_workflow,
    "find_anomalies": anomaly_detection_workflow,
    "correlate_logs_metrics": anomaly_detection_workflow,
    "analyze_anomalies": anomaly_detection_workflow,
    "log_metric_correlation": anomaly_detection_workflow,
    "performance_analysis": anomaly_detection_workflow,
    "health_analysis": anomaly_detection_workflow,
    "troubleshoot_instance": anomaly_detection_workflow,
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
