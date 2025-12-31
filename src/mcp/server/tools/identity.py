"""
OCI Identity MCP Tools.

Provides tools for IAM operations including compartment listing,
tenancy information, and user/group management.
"""

import json
from typing import Any

from src.mcp.server.auth import get_identity_client, get_oci_config


async def _list_compartments_logic(
    compartment_id: str | None = None,
    include_subtree: bool = True,
    lifecycle_state: str = "ACTIVE",
    limit: int = 100,
    format: str = "markdown",
) -> str:
    """Internal logic for listing compartments."""
    config = get_oci_config()
    client = get_identity_client()

    try:
        # Use tenancy as root if no compartment specified
        root_compartment = compartment_id or config.get("tenancy")

        if not root_compartment:
            return "Error: No compartment_id provided and tenancy not found in config"

        response = client.list_compartments(
            compartment_id=root_compartment,
            compartment_id_in_subtree=include_subtree,
            access_level="ACCESSIBLE",
            lifecycle_state=lifecycle_state,
            limit=limit,
        )

        compartments = response.data

        if format == "json":
            return json.dumps(
                [
                    {
                        "name": c.name,
                        "id": c.id,
                        "description": c.description,
                        "lifecycle_state": c.lifecycle_state,
                        "parent_id": c.compartment_id,
                    }
                    for c in compartments
                ],
                indent=2,
            )

        # Markdown table for LLM efficiency
        lines = [
            f"Found {len(compartments)} compartments:\n",
            "| Name | State | OCID |",
            "| --- | --- | --- |",
        ]
        for c in compartments:
            # Truncate OCID for display
            short_ocid = f"{c.id[:30]}..." if len(c.id) > 30 else c.id
            lines.append(f"| {c.name} | {c.lifecycle_state} | `{short_ocid}` |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing compartments: {e}"


async def _get_compartment_logic(compartment_id: str, format: str = "markdown") -> str:
    """Get details of a specific compartment."""
    client = get_identity_client()

    try:
        response = client.get_compartment(compartment_id=compartment_id)
        c = response.data

        if format == "json":
            return json.dumps(
                {
                    "name": c.name,
                    "id": c.id,
                    "description": c.description,
                    "lifecycle_state": c.lifecycle_state,
                    "parent_id": c.compartment_id,
                    "time_created": str(c.time_created),
                },
                indent=2,
            )

        return f"""## Compartment: {c.name}

- **OCID**: `{c.id}`
- **State**: {c.lifecycle_state}
- **Description**: {c.description or 'N/A'}
- **Parent ID**: `{c.compartment_id}`
- **Created**: {c.time_created}
"""

    except Exception as e:
        return f"Error getting compartment: {e}"


async def _search_compartments_logic(query: str, limit: int = 20) -> str:
    """Search compartments by name pattern."""
    config = get_oci_config()
    client = get_identity_client()

    try:
        tenancy_id = config.get("tenancy")
        if not tenancy_id:
            return "Error: Tenancy not found in config"

        # List all compartments and filter by name
        response = client.list_compartments(
            compartment_id=tenancy_id,
            compartment_id_in_subtree=True,
            access_level="ACCESSIBLE",
            lifecycle_state="ACTIVE",
            limit=500,  # Get more to search through
        )

        query_lower = query.lower()
        matches = [c for c in response.data if query_lower in c.name.lower()][:limit]

        if not matches:
            return f"No compartments found matching '{query}'"

        lines = [
            f"Found {len(matches)} compartments matching '{query}':\n",
            "| Name | OCID |",
            "| --- | --- |",
        ]
        for c in matches:
            lines.append(f"| {c.name} | `{c.id}` |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error searching compartments: {e}"


async def _get_tenancy_logic(format: str = "markdown") -> str:
    """Get current tenancy information."""
    config = get_oci_config()
    client = get_identity_client()

    try:
        tenancy_id = config.get("tenancy")
        if not tenancy_id:
            return "Error: Tenancy not found in config"

        response = client.get_tenancy(tenancy_id=tenancy_id)
        t = response.data

        if format == "json":
            return json.dumps(
                {
                    "name": t.name,
                    "id": t.id,
                    "description": t.description,
                    "home_region_key": t.home_region_key,
                },
                indent=2,
            )

        return f"""## Tenancy: {t.name}

- **OCID**: `{t.id}`
- **Home Region**: {t.home_region_key}
- **Description**: {t.description or 'N/A'}
"""

    except Exception as e:
        return f"Error getting tenancy: {e}"


async def _list_regions_logic() -> str:
    """List available OCI regions for the tenancy."""
    client = get_identity_client()
    config = get_oci_config()

    try:
        tenancy_id = config.get("tenancy")
        if not tenancy_id:
            return "Error: Tenancy not found in config"

        response = client.list_region_subscriptions(tenancy_id=tenancy_id)
        regions = response.data

        lines = [
            f"Found {len(regions)} subscribed regions:\n",
            "| Region | Key | Status |",
            "| --- | --- | --- |",
        ]
        for r in regions:
            home = " (Home)" if r.is_home_region else ""
            lines.append(f"| {r.region_name}{home} | {r.region_key} | {r.status} |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing regions: {e}"


def register_identity_tools(mcp: Any) -> None:
    """Register identity/IAM tools with the MCP server."""

    @mcp.tool()
    async def oci_list_compartments(
        compartment_id: str | None = None,
        include_subtree: bool = True,
        lifecycle_state: str = "ACTIVE",
        limit: int = 100,
        format: str = "markdown",
    ) -> str:
        """List OCI compartments in the tenancy.

        Args:
            compartment_id: Parent compartment OCID (defaults to tenancy root)
            include_subtree: Include nested compartments (default True)
            lifecycle_state: Filter by state (ACTIVE, CREATING, etc)
            limit: Maximum compartments to return (default 100)
            format: Output format ('json' or 'markdown')

        Returns:
            List of compartments with names and OCIDs
        """
        return await _list_compartments_logic(
            compartment_id, include_subtree, lifecycle_state, limit, format
        )

    @mcp.tool()
    async def oci_get_compartment(compartment_id: str, format: str = "markdown") -> str:
        """Get details of a specific compartment.

        Args:
            compartment_id: OCID of the compartment
            format: Output format ('json' or 'markdown')

        Returns:
            Compartment details including name, state, and parent
        """
        return await _get_compartment_logic(compartment_id, format)

    @mcp.tool()
    async def oci_search_compartments(query: str, limit: int = 20) -> str:
        """Search compartments by name pattern.

        Use this when you need to find a compartment by partial name match.

        Args:
            query: Search string (case-insensitive, partial match)
            limit: Maximum results to return (default 20)

        Returns:
            Matching compartments with names and OCIDs
        """
        return await _search_compartments_logic(query, limit)

    @mcp.tool()
    async def oci_get_tenancy(format: str = "markdown") -> str:
        """Get current tenancy information.

        Returns:
            Tenancy details including name, OCID, and home region
        """
        return await _get_tenancy_logic(format)

    @mcp.tool()
    async def oci_list_regions() -> str:
        """List OCI regions subscribed by the tenancy.

        Returns:
            List of regions with status and home region indicator
        """
        return await _list_regions_logic()
