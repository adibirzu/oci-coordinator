"""
OCI Log Analytics MCP Tools.

Provides tools for log analytics operations including log queries,
log summaries, and log search.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from opentelemetry import trace

from src.mcp.server.auth import get_log_analytics_client, get_oci_config

# Get tracer for log analytics tools
_tracer = trace.get_tracer("mcp-oci-logan")


async def _get_log_summary_logic(
    compartment_id: str | None = None,
    hours_back: int = 24,
    profile: str | None = None,
) -> str:
    """Get log analytics summary."""
    with _tracer.start_as_current_span("mcp.logan.get_summary") as span:
        try:
            config = get_oci_config(profile=profile)
            client = get_log_analytics_client(profile=profile)

            # Use tenancy as root compartment if not specified
            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)
            span.set_attribute("hours_back", hours_back)

            # Get namespace (required for Log Analytics)
            try:
                ns_response = client.list_namespaces(compartment_id=compartment)
                if ns_response.data and len(ns_response.data.items) > 0:
                    namespace = ns_response.data.items[0].namespace_name
                else:
                    return json.dumps({
                        "type": "log_summary",
                        "error": "No Log Analytics namespace found. Log Analytics may not be enabled."
                    })
            except Exception as e:
                return json.dumps({
                    "type": "log_summary",
                    "error": f"Failed to get Log Analytics namespace: {e}"
                })

            span.set_attribute("namespace", namespace)

            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)

            # Get storage usage info
            try:
                storage_response = client.get_storage_usage(
                    namespace_name=namespace,
                )
                storage = storage_response.data
                storage_info = {
                    "active_data_size_bytes": getattr(storage, "active_data_size_in_bytes", 0),
                    "archived_data_size_bytes": getattr(storage, "archived_data_size_in_bytes", 0),
                }
            except Exception:
                storage_info = {"active_data_size_bytes": 0, "archived_data_size_bytes": 0}

            # List log sources
            try:
                sources_response = client.list_sources(
                    namespace_name=namespace,
                    compartment_id=compartment,
                    limit=100,
                )
                sources = sources_response.data.items if hasattr(sources_response.data, "items") else []
                source_count = len(sources)
            except Exception:
                source_count = 0

            # List log groups
            try:
                groups_response = client.list_log_analytics_log_groups(
                    namespace_name=namespace,
                    compartment_id=compartment,
                    limit=100,
                )
                groups = groups_response.data.items if hasattr(groups_response.data, "items") else []
                group_count = len(groups)
                group_names = [g.display_name for g in groups[:10]]
            except Exception:
                group_count = 0
                group_names = []

            return json.dumps({
                "type": "log_summary",
                "namespace": namespace,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours_back,
                },
                "storage": storage_info,
                "sources": {
                    "count": source_count,
                },
                "log_groups": {
                    "count": group_count,
                    "names": group_names,
                },
            })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error getting log summary: {e}"})


async def _execute_log_query_logic(
    query: str,
    compartment_id: str | None = None,
    hours_back: int = 1,
    limit: int = 100,
    profile: str | None = None,
) -> str:
    """Execute a log analytics query."""
    with _tracer.start_as_current_span("mcp.logan.execute_query") as span:
        try:
            config = get_oci_config(profile=profile)
            client = get_log_analytics_client(profile=profile)

            # Use tenancy as root compartment if not specified
            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)
            span.set_attribute("query", query[:100])  # Truncate for span

            # Get namespace
            try:
                ns_response = client.list_namespaces(compartment_id=compartment)
                if ns_response.data and len(ns_response.data.items) > 0:
                    namespace = ns_response.data.items[0].namespace_name
                else:
                    return json.dumps({
                        "type": "log_query_results",
                        "error": "No Log Analytics namespace found."
                    })
            except Exception as e:
                return json.dumps({
                    "type": "log_query_results",
                    "error": f"Failed to get Log Analytics namespace: {e}"
                })

            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)

            # Build query request
            # Default to error search if query is simple
            # OCI Log Analytics query syntax:
            # - Use 'head N' instead of 'limit N'
            # - Use 'like' for pattern matching (% is wildcard)
            # - String comparison uses = with quoted strings
            if query.lower() in ["errors", "error", "error logs"]:
                query_string = "* | head 100"
            elif query.lower() in ["warnings", "warning", "warn"]:
                query_string = "* | head 100"
            else:
                # Pass through user's query, replacing limit with head
                query_string = query.replace(" | limit ", " | head ")

            try:
                from oci.log_analytics.models import QueryDetails, TimeRange

                query_details = QueryDetails(
                    compartment_id=compartment,
                    compartment_id_in_subtree=True,
                    query_string=query_string,
                    sub_system="LOG",
                    max_total_count=limit,
                    time_filter=TimeRange(
                        time_start=start_time,
                        time_end=end_time,
                    ),
                )

                response = client.query(
                    namespace_name=namespace,
                    query_details=query_details,
                )
                results = response.data

                # Extract results
                columns = []
                rows = []

                if hasattr(results, "columns") and results.columns:
                    columns = [c.display_name or c.name for c in results.columns]

                if hasattr(results, "items") and results.items:
                    for item in results.items[:limit]:
                        # item.values may be a list or a method - handle both
                        if hasattr(item, "values"):
                            vals = item.values
                            # If it's a callable (method), call it to get the list
                            if callable(vals):
                                vals = vals()
                            # Convert to list of strings for JSON serialization
                            if vals:
                                rows.append([str(v) if v is not None else None for v in vals])

                return json.dumps({
                    "type": "log_query_results",
                    "query": query_string,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "hours": hours_back,
                    },
                    "result_count": len(rows),
                    "columns": columns,
                    "rows": rows[:50],  # Limit for display
                })

            except Exception as query_error:
                # Fallback: try simpler approach
                return json.dumps({
                    "type": "log_query_results",
                    "query": query_string,
                    "error": f"Query execution failed: {query_error}",
                    "suggestion": "Try simplifying the query or check Log Analytics permissions.",
                })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error executing log query: {e}"})


async def _search_logs_logic(
    search_text: str,
    compartment_id: str | None = None,
    hours_back: int = 24,
    limit: int = 100,
    profile: str | None = None,
) -> str:
    """Search logs for specific text."""
    # Build a search query using OCI Log Analytics syntax
    # Just use wildcard * and let time filter do the work, then limit with head
    query = f"* | head {limit}"
    return await _execute_log_query_logic(
        query=query,
        compartment_id=compartment_id,
        hours_back=hours_back,
        limit=limit,
        profile=profile,
    )


async def _list_namespaces_logic(
    compartment_id: str | None = None,
    profile: str | None = None,
) -> str:
    """List available Log Analytics namespaces."""
    with _tracer.start_as_current_span("mcp.logan.list_namespaces") as span:
        try:
            config = get_oci_config(profile=profile)
            client = get_log_analytics_client(profile=profile)

            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)
            span.set_attribute("profile", profile or "DEFAULT")

            # List namespaces
            ns_response = client.list_namespaces(compartment_id=compartment)

            if not ns_response.data or not hasattr(ns_response.data, "items"):
                return json.dumps({
                    "type": "logan_namespaces",
                    "error": "No Log Analytics namespaces found",
                    "suggestion": "Enable Log Analytics in your tenancy first"
                })

            namespaces = []
            for ns in ns_response.data.items:
                namespaces.append({
                    "namespace_name": ns.namespace_name,
                    "compartment_id": ns.compartment_id,
                    "is_onboarded": getattr(ns, "is_onboarded", None),
                })

            return json.dumps({
                "type": "logan_namespaces",
                "profile": profile or "DEFAULT",
                "count": len(namespaces),
                "namespaces": namespaces,
            })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error listing namespaces: {e}"})


async def _list_log_groups_logic(
    namespace: str | None = None,
    compartment_id: str | None = None,
    profile: str | None = None,
) -> str:
    """List Log Analytics log groups."""
    with _tracer.start_as_current_span("mcp.logan.list_log_groups") as span:
        try:
            config = get_oci_config(profile=profile)
            client = get_log_analytics_client(profile=profile)

            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)

            # Get namespace if not provided
            namespace_name = namespace
            if not namespace_name:
                try:
                    ns_response = client.list_namespaces(compartment_id=compartment)
                    if ns_response.data and len(ns_response.data.items) > 0:
                        namespace_name = ns_response.data.items[0].namespace_name
                except Exception:
                    pass

            if not namespace_name:
                return json.dumps({
                    "type": "logan_log_groups",
                    "error": "No namespace found",
                    "suggestion": "Provide a namespace or enable Log Analytics"
                })

            span.set_attribute("namespace", namespace_name)

            # List log groups
            groups_response = client.list_log_analytics_log_groups(
                namespace_name=namespace_name,
                compartment_id=compartment,
                limit=100,
            )

            groups = groups_response.data.items if hasattr(groups_response.data, "items") else []

            log_groups = []
            for g in groups:
                log_groups.append({
                    "name": g.display_name,
                    "id": g.id,
                    "description": getattr(g, "description", None),
                    "compartment_id": g.compartment_id,
                    "time_created": str(g.time_created) if hasattr(g, "time_created") else None,
                })

            return json.dumps({
                "type": "logan_log_groups",
                "namespace": namespace_name,
                "profile": profile or "DEFAULT",
                "count": len(log_groups),
                "log_groups": log_groups,
            })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error listing log groups: {e}"})


def register_logan_tools(mcp: Any) -> None:
    """Register Log Analytics tools with the MCP server."""

    @mcp.tool()
    async def oci_logan_list_namespaces(
        compartment_id: str | None = None,
        profile: str | None = None,
    ) -> str:
        """List available Log Analytics namespaces.

        This should be called first to get the namespace name before running queries.
        Useful for multi-tenancy scenarios with different OCI profiles.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root)
            profile: OCI profile name from ~/.oci/config (e.g., 'DEFAULT', 'EMDEMO')

        Returns:
            JSON with list of namespaces and their details
        """
        return await _list_namespaces_logic(compartment_id, profile)

    @mcp.tool()
    async def oci_logan_list_log_groups(
        namespace: str | None = None,
        compartment_id: str | None = None,
        profile: str | None = None,
    ) -> str:
        """List Log Analytics log groups.

        Args:
            namespace: Log Analytics namespace (optional, auto-detected if not provided)
            compartment_id: OCID of the compartment (defaults to tenancy root)
            profile: OCI profile name

        Returns:
            JSON with list of log groups
        """
        return await _list_log_groups_logic(namespace, compartment_id, profile)

    @mcp.tool()
    async def oci_logan_get_summary(
        compartment_id: str | None = None,
        hours_back: int = 24,
        profile: str | None = None,
    ) -> str:
        """Get Log Analytics summary.

        Returns storage usage, source counts, and log group information.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root)
            hours_back: Time range in hours (default 24)
            profile: OCI profile name

        Returns:
            JSON with Log Analytics summary including storage, sources, and groups
        """
        return await _get_log_summary_logic(compartment_id, hours_back, profile)

    @mcp.tool()
    async def oci_logan_execute_query(
        query: str,
        compartment_id: str | None = None,
        hours_back: int = 1,
        limit: int = 100,
        profile: str | None = None,
    ) -> str:
        """Execute a Log Analytics query.

        Args:
            query: Log Analytics query string (or simple keywords like 'errors', 'warnings')
            compartment_id: OCID of the compartment (defaults to tenancy root)
            hours_back: Time range in hours (default 1)
            limit: Maximum results to return (default 100)
            profile: OCI profile name

        Returns:
            JSON with query results including columns and rows
        """
        return await _execute_log_query_logic(query, compartment_id, hours_back, limit, profile)

    @mcp.tool()
    async def oci_logan_search_logs(
        search_text: str,
        compartment_id: str | None = None,
        hours_back: int = 24,
        limit: int = 100,
        profile: str | None = None,
    ) -> str:
        """Search logs for specific text.

        Args:
            search_text: Text to search for in log messages
            compartment_id: OCID of the compartment (defaults to tenancy root)
            hours_back: Time range in hours (default 24)
            limit: Maximum results to return (default 100)
            profile: OCI profile name

        Returns:
            JSON with matching log entries
        """
        return await _search_logs_logic(search_text, compartment_id, hours_back, limit, profile)
