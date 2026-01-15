"""
Response Parser for OCI Coordinator Agent Outputs.

Converts raw agent JSON responses (cost_summary, fleet_health, managed_databases, etc.)
into StructuredResponse objects that can be rendered by any channel formatter.

This centralizes response parsing logic that was previously scattered in channel adapters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import structlog

from src.formatting.base import (
    MetricValue,
    ResponseHeader,
    Section,
    Severity,
    StructuredResponse,
    TableData,
    TableRow,
)

logger = structlog.get_logger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a raw response."""

    response: StructuredResponse
    raw_text: str | None = None
    table_data: list[dict] | None = None


class ResponseParser:
    """
    Parses raw agent JSON responses into StructuredResponse objects.

    Handles:
    - React agent format: {"thought": "...", "final_answer": "..."}
    - Typed responses: {"type": "cost_summary", ...}
    - Plain JSON arrays from tools
    - Plain text with embedded JSON

    Usage:
        parser = ResponseParser()
        result = parser.parse(agent_output)
        slack_blocks = SlackFormatter().format_response(result.response)
    """

    # Response type handlers registry
    _handlers: dict[str, callable] = {}

    def __init__(self):
        self._logger = logger.bind(component="ResponseParser")

    @classmethod
    def register_handler(cls, response_type: str):
        """Decorator to register a response type handler."""
        def decorator(func):
            cls._handlers[response_type] = func
            return func
        return decorator

    def parse(self, message: str, agent_name: str | None = None) -> ParseResult:
        """
        Parse a raw agent response into a StructuredResponse.

        Args:
            message: Raw response string from agent
            agent_name: Optional agent name for header context

        Returns:
            ParseResult with StructuredResponse and optional raw data
        """
        if not message:
            return self._empty_response(agent_name)

        # Extract content from React agent format
        extracted, raw_json = self._extract_from_react_format(message)
        if extracted:
            message = extracted

        # Try parsing as typed JSON response
        typed_result = self._parse_typed_response(message, agent_name)
        if typed_result:
            return typed_result

        # Try parsing as JSON array (table data)
        table_data = self._extract_json_array(message)
        if table_data:
            return self._table_response(table_data, agent_name)

        # Fall back to plain text response
        cleaned = self._clean_text(message)
        return ParseResult(
            response=StructuredResponse(
                header=ResponseHeader(
                    title="Response",
                    agent_name=agent_name,
                    severity=Severity.INFO,
                ),
                sections=[Section(content=cleaned)] if cleaned else [],
            ),
            raw_text=cleaned,
        )

    def _empty_response(self, agent_name: str | None) -> ParseResult:
        """Create an empty response."""
        return ParseResult(
            response=StructuredResponse(
                header=ResponseHeader(
                    title="No Response",
                    agent_name=agent_name,
                    severity=Severity.INFO,
                ),
            )
        )

    def _extract_from_react_format(self, message: str) -> tuple[str | None, dict | None]:
        """
        Extract final_answer from React agent JSON format.

        Handles multiple concatenated JSON objects from iterations,
        returning the last final_answer (most complete).
        """
        if '"thought"' not in message and '"final_answer"' not in message:
            return None, None

        extracted = None
        raw_json = None

        # Try regex to find all final_answer values
        final_answers = re.findall(
            r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"|"final_answer"\s*:\s*"([^"]*)"',
            message,
            re.DOTALL
        )
        if final_answers:
            for match in final_answers:
                answer = match[0] or match[1]
                if answer:
                    extracted = answer.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')

        # Try response field if no final_answer
        if not extracted:
            response_matches = re.findall(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', message, re.DOTALL)
            if response_matches:
                extracted = response_matches[-1].replace('\\"', '"').replace('\\n', '\n')

        # Parse JSON objects if still nothing
        if not extracted:
            json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', message)
            for json_str in reversed(json_objects):
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        if "final_answer" in parsed:
                            extracted = parsed["final_answer"]
                            raw_json = parsed
                            break
                        elif "response" in parsed:
                            extracted = parsed["response"]
                            raw_json = parsed
                            break
                except json.JSONDecodeError:
                    continue

        return extracted, raw_json

    def _parse_typed_response(self, message: str, agent_name: str | None) -> ParseResult | None:
        """
        Parse typed JSON response(s) (with "type" field).

        Handles both single and multiple JSON objects in the message.
        Multiple JSON objects are combined into a single StructuredResponse.

        Returns None if not a typed response.
        """
        if "{" not in message or '"type"' not in message:
            return None

        # Try parsing as single JSON first
        try:
            start = message.find("{")
            end = message.rfind("}") + 1
            json_part = message[start:end]
            parsed = json.loads(json_part)

            if isinstance(parsed, dict) and "type" in parsed:
                return self._handle_single_typed_response(parsed, agent_name)

        except json.JSONDecodeError:
            # Multiple JSON objects - try parsing each separately
            pass

        # Try parsing multiple JSON objects (separated by whitespace/newlines)
        json_objects = self._extract_multiple_json_objects(message)
        if json_objects:
            return self._combine_typed_responses(json_objects, agent_name)

        return None

    def _extract_multiple_json_objects(self, message: str) -> list[dict]:
        """Extract multiple JSON objects from a message."""
        objects = []
        # Find all balanced JSON objects
        i = 0
        while i < len(message):
            if message[i] == '{':
                # Find matching closing brace
                depth = 0
                start = i
                for j in range(i, len(message)):
                    if message[j] == '{':
                        depth += 1
                    elif message[j] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(message[start:j+1])
                                if isinstance(obj, dict) and "type" in obj:
                                    objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
                else:
                    break
            else:
                i += 1
        return objects

    def _handle_single_typed_response(
        self, parsed: dict, agent_name: str | None
    ) -> ParseResult | None:
        """Handle a single typed JSON response."""
        if not isinstance(parsed, dict) or "type" not in parsed:
            return None

        data_type = parsed.get("type")

        # Handle error responses
        if parsed.get("error"):
            return self._error_response(
                parsed["error"],
                parsed.get("suggestion"),
                agent_name,
            )

        # Route to type-specific handler
        handler = getattr(self, f"_parse_{data_type}", None)
        if handler:
            return handler(parsed, agent_name)

        # Unknown type - return as raw JSON
        return ParseResult(
            response=StructuredResponse(
                header=ResponseHeader(
                    title=data_type.replace("_", " ").title(),
                    agent_name=agent_name,
                    severity=Severity.INFO,
                ),
                raw_data=parsed,
            ),
            raw_text=json.dumps(parsed, indent=2),
        )

    def _combine_typed_responses(
        self, json_objects: list[dict], agent_name: str | None
    ) -> ParseResult | None:
        """Combine multiple typed JSON responses into a single StructuredResponse."""
        if not json_objects:
            return None

        # Parse each object individually
        parsed_results = []
        for obj in json_objects:
            result = self._handle_single_typed_response(obj, agent_name)
            if result:
                parsed_results.append(result)

        if not parsed_results:
            return None

        if len(parsed_results) == 1:
            return parsed_results[0]

        # Combine multiple results into a single response
        combined = StructuredResponse(
            header=ResponseHeader(
                title="Results",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add sections from each result
        for result in parsed_results:
            resp = result.response
            # Add header as a section title (using add_section properly)
            if resp.header.title and resp.header.title != "Results":
                icon = resp.header.icon or ""
                title = f"{icon} {resp.header.title}".strip()
                combined.add_section(title=title)

            # Copy sections (sections may contain tables)
            for section in resp.sections:
                combined.add_section(
                    title=section.title,
                    content=section.content,
                    fields=section.fields,
                    list_items=section.list_items,
                    table=section.table,
                    code_block=section.code_block,
                    actions=section.actions,
                    divider_after=section.divider_after,
                )

        return ParseResult(
            response=combined,
            raw_text="\n\n".join(json.dumps(obj, indent=2) for obj in json_objects),
        )

    def _error_response(
        self, error: str, suggestion: str | None, agent_name: str | None
    ) -> ParseResult:
        """Create an error response."""
        content = error
        if suggestion:
            content += f"\n\n*Suggestion:* {suggestion}"

        return ParseResult(
            response=StructuredResponse(
                header=ResponseHeader(
                    title="Error",
                    icon="⚠️",
                    agent_name=agent_name,
                    severity=Severity.HIGH,
                ),
                sections=[Section(content=content)],
                success=False,
                error=error,
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Type-Specific Parsers
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_cost_summary(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse cost_summary response."""
        summary = data.get("summary", {})
        services = data.get("services", [])

        response = StructuredResponse(
            header=ResponseHeader(
                title="Tenancy Cost Summary",
                icon="💰",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add summary metrics
        metrics = [
            MetricValue(
                label="Period",
                value=summary.get("period", "N/A"),
                unit=f"({summary.get('days', 30)} days)",
            ),
            MetricValue(
                label="Total Spend",
                value=summary.get("total", "N/A"),
                severity=Severity.INFO,
            ),
        ]
        response.add_metrics("Summary", metrics)

        # Add services table if present
        if services:
            table = self._list_to_table(services, title="Cost by Service")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=services)

    def _parse_cost_by_compartment(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse cost_by_compartment response."""
        summary = data.get("summary", {})
        compartments = data.get("compartments", [])

        response = StructuredResponse(
            header=ResponseHeader(
                title="Cost by Compartment",
                icon="📁",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add summary metrics
        metrics = [
            MetricValue(
                label="Period",
                value=summary.get("period", "N/A"),
                unit=f"({summary.get('days', 30)} days)",
            ),
            MetricValue(
                label="Total Spend",
                value=summary.get("total", "N/A"),
                severity=Severity.INFO,
            ),
        ]
        response.add_metrics("Summary", metrics)

        # Add compartments table if present
        if compartments:
            table = self._list_to_table(compartments, title="Cost by Compartment")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=compartments)

    def _parse_cost_service_drilldown(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse cost_service_drilldown response with SKU-level detail."""
        summary = data.get("summary", {})
        services = data.get("services", [])

        response = StructuredResponse(
            header=ResponseHeader(
                title="Cost Service Drilldown",
                icon="🔍",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add summary metrics
        metrics = [
            MetricValue(
                label="Period",
                value=summary.get("period", "N/A"),
                unit=f"({summary.get('days', 30)} days)",
            ),
            MetricValue(
                label="Total Spend",
                value=summary.get("total", "N/A"),
                severity=Severity.INFO,
            ),
        ]
        response.add_metrics("Summary", metrics)

        # Add services with their SKU breakdown
        for svc in services[:5]:  # Top 5 services with detail
            svc_name = svc.get("service", "Unknown")
            svc_cost = svc.get("cost", "N/A")
            svc_pct = svc.get("percent", "")

            response.add_section(
                title=f"📊 {svc_name}",
                content=f"*Cost:* {svc_cost} ({svc_pct})",
            )

            # Add top resources as a sub-table
            resources = svc.get("top_resources", [])
            if resources:
                table = self._list_to_table(resources, title="Top Resources")
                response.add_section(table=table)

        return ParseResult(response=response, table_data=services)

    def _parse_cost_monthly_trend(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse cost_monthly_trend response."""
        summary = data.get("summary", {})
        trend = data.get("trend", [])

        response = StructuredResponse(
            header=ResponseHeader(
                title="Monthly Cost Trend",
                icon="📈",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add summary metrics
        metrics = [
            MetricValue(
                label="Period",
                value=summary.get("period", "N/A"),
            ),
            MetricValue(
                label="Total",
                value=summary.get("total", "N/A"),
            ),
            MetricValue(
                label="Average",
                value=summary.get("average", "N/A"),
            ),
        ]
        response.add_metrics("Summary", metrics)

        # Add trend table
        if trend:
            table = self._list_to_table(trend, title="Monthly Trend")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=trend)

    def _parse_cost_comparison(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse cost_comparison response."""
        summary = data.get("summary", {})
        comparison = data.get("comparison", [])

        response = StructuredResponse(
            header=ResponseHeader(
                title="Cost Comparison",
                icon="⚖️",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add summary metrics
        metrics = [
            MetricValue(
                label="Months",
                value=str(summary.get("months_compared", "N/A")),
            ),
            MetricValue(
                label="Total",
                value=summary.get("total", "N/A"),
            ),
            MetricValue(
                label="Highest",
                value=summary.get("highest", "N/A"),
                severity=Severity.HIGH,
            ),
            MetricValue(
                label="Lowest",
                value=summary.get("lowest", "N/A"),
                severity=Severity.SUCCESS,
            ),
        ]
        response.add_metrics("Comparison Summary", metrics)

        # Build comparison table
        if comparison:
            table_items = []
            for item in comparison:
                row = {
                    "Month": item.get("month", "N/A"),
                    "Cost": item.get("formatted", "N/A"),
                }
                if "change_from_prev" in item:
                    row["Change"] = item["change_from_prev"]
                table_items.append(row)
            table = self._list_to_table(table_items, title="Month Comparison")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=comparison)

    def _parse_fleet_health(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse fleet_health response."""
        summary = data.get("summary", {}) or {}
        statistics = data.get("statistics", [])

        # Helper to safely get count values (None -> 0)
        def safe_count(key: str) -> int | str:
            val = summary.get(key)
            return val if val is not None else 0

        response = StructuredResponse(
            header=ResponseHeader(
                title="Database Fleet Health",
                icon="💓",
                agent_name=agent_name,
                severity=self._health_severity(summary),
            ),
        )

        # Check if we have any databases
        db_count = safe_count("database_count")
        has_databases = db_count and db_count > 0

        # Add health metrics
        metrics = [
            MetricValue(label="Databases", value=db_count),
            MetricValue(
                label="Healthy",
                value=safe_count("healthy_count"),
                severity=Severity.SUCCESS,
            ),
            MetricValue(
                label="Warning",
                value=safe_count("warning_count"),
                severity=Severity.MEDIUM,
            ),
            MetricValue(
                label="Critical",
                value=safe_count("critical_count"),
                severity=Severity.CRITICAL,
            ),
            MetricValue(
                label="Unavailable",
                value=safe_count("unavailable_count"),
                severity=Severity.HIGH,
            ),
        ]
        response.add_metrics("Health Status", metrics, divider_after=True)

        # Add guidance if no databases found
        if not has_databases:
            response.add_section(
                title="No Databases Found",
                content=(
                    "No databases are registered with the OCI Database Management Service.\n\n"
                    "*To enable database monitoring:*\n"
                    "1. Enable Database Management on your databases in OCI Console\n"
                    "2. Ensure databases have the management agent installed\n"
                    "3. Verify the compartment ID and region are correct"
                ),
            )

        # Add statistics table
        if statistics:
            table = self._list_to_table(statistics, title="Fleet Statistics")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=statistics)

    def _parse_managed_databases(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse managed_databases response."""
        return self._parse_database_list(data, "Managed Databases", agent_name)

    def _parse_managed_database_search(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse managed_database_search response."""
        response = self._parse_database_list(data, "Database Search Results", agent_name)

        # Add search query to header
        query = data.get("query")
        if query and response.response.header:
            response.response.header.subtitle = f"Query: {query}"

        return response

    def _parse_database_connections(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse database_connections response from oci_database_list_connections.

        Converts connection data to a table for proper Slack rendering.
        """
        connections = data.get("connections", [])
        count = data.get("count", len(connections))
        message = data.get("message")
        default_connection = data.get("default_connection")
        tns_admin = data.get("tns_admin")

        # Build subtitle with default connection info
        subtitle_parts = []
        if count:
            subtitle_parts.append(f"{count} connection(s) available")
        else:
            subtitle_parts.append(message or "No connections")
        if default_connection:
            subtitle_parts.append(f"Default: {default_connection}")

        response = StructuredResponse(
            header=ResponseHeader(
                title="Database Connections",
                icon="🔗",
                subtitle=" | ".join(subtitle_parts),
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        if connections:
            # Check if any connections have database names
            has_db_names = any(conn.get("database_name") for conn in connections)

            if has_db_names:
                # Create table with database name column
                table = TableData(
                    title="Connections",
                    headers=["Name", "Database", "Connection Type", "Status", "Default"],
                    rows=[
                        TableRow(cells=[
                            conn.get("name", "N/A"),
                            conn.get("database_name", "-"),
                            conn.get("connection_type", "N/A"),
                            conn.get("status", "N/A"),
                            "✓" if conn.get("is_default") else ("fallback" if conn.get("is_fallback") else "-"),
                        ])
                        for conn in connections
                    ],
                )
            else:
                # Create table without database name column
                table = TableData(
                    title="Connections",
                    headers=["Name", "Connection Type", "Status", "User", "Default"],
                    rows=[
                        TableRow(cells=[
                            conn.get("name", "N/A"),
                            conn.get("connection_type", "N/A"),
                            conn.get("status", "N/A"),
                            conn.get("user", "-"),
                            "✓" if conn.get("is_default") else ("fallback" if conn.get("is_fallback") else "-"),
                        ])
                        for conn in connections
                    ],
                )
            response.add_section(table=table)

            # Add TNS admin path if available
            if tns_admin:
                response.add_section(content=f"_TNS Admin: {tns_admin}_")
        elif message:
            response.add_section(content=message)

        return ParseResult(response=response, table_data=connections)

    def _parse_compute_instances(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse compute_instances response from oci_compute_list_instances.

        Converts instance data to a table for proper Slack rendering.
        """
        instances = data.get("instances", [])
        count = data.get("count", len(instances))
        lifecycle_state = data.get("lifecycle_state")

        # Build subtitle with filter context
        if lifecycle_state:
            subtitle = f"{count} {lifecycle_state.lower()} instance(s)"
        else:
            subtitle = f"{count} instance(s) found"

        response = StructuredResponse(
            header=ResponseHeader(
                title="Compute Instances",
                icon="💻",
                subtitle=subtitle,
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        if instances:
            # Create table with relevant columns
            table = TableData(
                title="Instances",
                headers=["Name", "State", "Shape", "Availability Domain"],
                rows=[
                    TableRow(cells=[
                        inst.get("name", "N/A"),
                        inst.get("state", "N/A"),
                        inst.get("shape", "N/A"),
                        inst.get("availability_domain", "N/A"),
                    ])
                    for inst in instances
                ],
            )
            response.add_section(table=table)
        else:
            response.add_section(content="No instances found in the specified compartment.")

        return ParseResult(response=response, table_data=instances)

    def _parse_compartments(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse compartments response from oci_tenancy_list_compartments.

        Converts compartment data to a table for proper Slack rendering.
        """
        compartments = data.get("compartments", [])
        count = data.get("count", len(compartments))

        response = StructuredResponse(
            header=ResponseHeader(
                title="Compartments",
                icon="📁",
                subtitle=f"{count} compartment(s) found",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        if compartments:
            # Create table with relevant columns
            table = TableData(
                title="Compartments",
                headers=["Name", "State", "Description"],
                rows=[
                    TableRow(cells=[
                        comp.get("name", "N/A"),
                        comp.get("lifecycle_state", "N/A"),
                        (comp.get("description") or "")[:50],  # Truncate description
                    ])
                    for comp in compartments
                ],
            )
            response.add_section(table=table)
        else:
            response.add_section(content="No compartments found.")

        return ParseResult(response=response, table_data=compartments)

    def _parse_alarms(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse alarms response from oci_observability_list_alarms.

        Converts alarm data to a table for proper Slack rendering.
        """
        alarms = data.get("alarms", [])
        count = data.get("count", len(alarms))

        # Count by severity
        critical = sum(1 for a in alarms if a.get("severity") == "CRITICAL")
        warning = sum(1 for a in alarms if a.get("severity") == "WARNING")

        subtitle_parts = [f"{count} alarm(s)"]
        if critical > 0:
            subtitle_parts.append(f"🔴 {critical} critical")
        if warning > 0:
            subtitle_parts.append(f"🟡 {warning} warning")

        response = StructuredResponse(
            header=ResponseHeader(
                title="Active Alarms",
                icon="🔔",
                subtitle=" | ".join(subtitle_parts),
                agent_name=agent_name,
                severity=Severity.WARNING if critical > 0 or warning > 0 else Severity.INFO,
            ),
        )

        if alarms:
            # Create table with relevant columns
            table = TableData(
                title="Alarms",
                headers=["Name", "Severity", "State", "Namespace", "Enabled"],
                rows=[
                    TableRow(cells=[
                        alarm.get("name", "N/A"),
                        alarm.get("severity", "INFO"),
                        alarm.get("state", "N/A"),
                        alarm.get("namespace", "N/A"),
                        "✓" if alarm.get("is_enabled") else "✗",
                    ])
                    for alarm in alarms
                ],
            )
            response.add_section(table=table)
        else:
            response.add_section(content="✅ No active alarms.")

        return ParseResult(response=response, table_data=alarms)

    def _parse_cloud_guard_problems(
        self, data: dict, agent_name: str | None
    ) -> ParseResult:
        """Parse cloud_guard_problems response."""
        problems = data.get("problems", [])
        count = data.get("count", len(problems))
        severity_summary = data.get("severity_summary", {})

        # Determine overall severity
        critical = severity_summary.get("CRITICAL", 0)
        high = severity_summary.get("HIGH", 0)
        if critical > 0:
            severity = Severity.CRITICAL
        elif high > 0:
            severity = Severity.HIGH
        elif count > 0:
            severity = Severity.MEDIUM
        else:
            severity = Severity.SUCCESS

        response = StructuredResponse(
            header=ResponseHeader(
                title="Cloud Guard Problems",
                icon="🛡️",
                subtitle=f"{count} security problems found",
                agent_name=agent_name,
                severity=severity,
            ),
        )

        # Add severity summary
        if severity_summary:
            summary_parts = []
            for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                if severity_summary.get(level, 0) > 0:
                    summary_parts.append(f"{level}: {severity_summary[level]}")
            if summary_parts:
                response.add_section(content="*Severity breakdown:* " + " | ".join(summary_parts))

        # Add problems table
        if problems:
            table = TableData(
                headers=["Name", "Risk Level", "Resource Type", "State"],
                rows=[
                    TableRow(
                        cells=[
                            p.get("name", "Unknown"),
                            p.get("risk_level", "N/A"),
                            p.get("resource_type", "N/A"),
                            p.get("lifecycle_state", "N/A"),
                        ])
                    for p in problems[:20]  # Limit display
                ],
            )
            response.add_section(table=table)
        else:
            response.add_section(content="✅ No security problems detected.")

        return ParseResult(response=response, table_data=problems)

    def _parse_security_score(
        self, data: dict, agent_name: str | None
    ) -> ParseResult:
        """Parse security_score response."""
        score = data.get("score", 0)
        grade = data.get("grade", "N/A")
        problem_counts = data.get("problem_counts", {})

        # Determine severity from score
        if score >= 80:
            severity = Severity.SUCCESS
        elif score >= 60:
            severity = Severity.MEDIUM
        else:
            severity = Severity.HIGH

        response = StructuredResponse(
            header=ResponseHeader(
                title="Security Score",
                icon="📊",
                subtitle=f"Grade: {grade} ({score}/100)",
                agent_name=agent_name,
                severity=severity,
            ),
        )

        # Add score details
        response.add_section(
            content=f"**Security Score:** {score}/100 (Grade: {grade})"
        )

        # Add problem counts
        if problem_counts:
            counts_parts = []
            for level in ["critical", "high", "medium", "low"]:
                if problem_counts.get(level, 0) > 0:
                    counts_parts.append(f"{level.title()}: {problem_counts[level]}")
            if counts_parts:
                response.add_section(
                    content="*Problem counts:* " + " | ".join(counts_parts)
                )

        return ParseResult(response=response, table_data=data)

    def _parse_audit_events(
        self, data: dict, agent_name: str | None
    ) -> ParseResult:
        """Parse audit_events response."""
        events = data.get("events", [])
        count = data.get("count", len(events))
        time_range = data.get("time_range", {})
        event_summary = data.get("event_type_summary", {})

        response = StructuredResponse(
            header=ResponseHeader(
                title="Audit Events",
                icon="📋",
                subtitle=f"{count} events in last {time_range.get('hours', 24)}h",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add event type summary
        if event_summary:
            top_types = sorted(event_summary.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_types:
                summary_text = ", ".join([f"{k}: {v}" for k, v in top_types])
                response.add_section(content=f"*Top event types:* {summary_text}")

        # Add events table
        if events:
            table = TableData(
                headers=["Event Type", "Time", "Source", "Resource"],
                rows=[
                    TableRow(
                        cells=[
                            e.get("event_type", "N/A")[:40],  # Truncate long types
                            e.get("event_time", "N/A")[:19] if e.get("event_time") else "N/A",
                            e.get("source", "N/A")[:20],
                            e.get("resource_name", "N/A")[:30],
                        ])
                    for e in events[:15]  # Limit display
                ],
            )
            response.add_section(table=table)
        else:
            response.add_section(content="No audit events found in the specified time range.")

        return ParseResult(response=response, table_data=events)

    def _parse_security_overview(
        self, data: dict, agent_name: str | None
    ) -> ParseResult:
        """Parse security_overview response."""
        posture = data.get("posture", {})
        critical_issues = data.get("critical_issues", {})
        audit_activity = data.get("audit_activity", {})
        recommendations = data.get("recommendations", [])

        score = posture.get("score", 0)
        grade = posture.get("grade", "N/A")

        # Determine severity
        if score >= 80 and critical_issues.get("count", 0) == 0:
            severity = Severity.SUCCESS
        elif critical_issues.get("count", 0) > 0:
            severity = Severity.CRITICAL
        elif score < 60:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

        response = StructuredResponse(
            header=ResponseHeader(
                title="Security Overview",
                icon="🔐",
                subtitle=f"Score: {score}/100 (Grade: {grade})",
                agent_name=agent_name,
                severity=severity,
            ),
        )

        # Add posture summary
        response.add_section(
            content=f"**Security Posture:** {score}/100 ({grade})\n"
                    f"*Total problems:* {posture.get('total_problems', 0)}"
        )

        # Add critical issues
        if critical_issues.get("count", 0) > 0:
            response.add_section(
                content=f"⚠️ **Critical Issues:** {critical_issues['count']} require immediate attention"
            )

        # Add audit activity
        response.add_section(
            content=f"*Audit activity (24h):* {audit_activity.get('events_24h', 0)} events, "
                    f"{audit_activity.get('event_types', 0)} event types"
        )

        # Add recommendations
        if recommendations:
            rec_items = []
            for rec in recommendations[:3]:
                priority = rec.get("priority", "medium").upper()
                action = rec.get("action", "Review security posture")
                rec_items.append(f"• [{priority}] {action}")
            if rec_items:
                response.add_section(
                    content="**Recommendations:**\n" + "\n".join(rec_items)
                )

        return ParseResult(response=response, table_data=data)

    def _parse_log_summary(
        self, data: dict, agent_name: str | None
    ) -> ParseResult:
        """Parse log_summary response."""
        namespace = data.get("namespace", "Unknown")
        time_range = data.get("time_range", {})
        storage = data.get("storage", {})
        sources = data.get("sources", {})
        log_groups = data.get("log_groups", {})

        response = StructuredResponse(
            header=ResponseHeader(
                title="Log Analytics Summary",
                icon="📊",
                subtitle=f"Namespace: {namespace}",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add time range
        hours = time_range.get("hours", 24)
        response.add_section(content=f"*Time range:* Last {hours} hours")

        # Add storage info
        active_bytes = storage.get("active_data_size_bytes", 0)
        archived_bytes = storage.get("archived_data_size_bytes", 0)
        active_gb = round(active_bytes / (1024**3), 2) if active_bytes else 0
        archived_gb = round(archived_bytes / (1024**3), 2) if archived_bytes else 0
        response.add_section(
            content=f"**Storage:** Active: {active_gb} GB | Archived: {archived_gb} GB"
        )

        # Add sources and groups
        response.add_section(
            content=f"**Log Sources:** {sources.get('count', 0)} | **Log Groups:** {log_groups.get('count', 0)}"
        )

        # List log groups if available
        group_names = log_groups.get("names", [])
        if group_names:
            response.add_section(
                content="*Log Groups:* " + ", ".join(group_names[:5])
            )

        return ParseResult(response=response, table_data=data)

    def _parse_log_query_results(
        self, data: dict, agent_name: str | None
    ) -> ParseResult:
        """Parse log_query_results response."""
        query = data.get("query", "Unknown query")
        result_count = data.get("result_count", 0)
        time_range = data.get("time_range", {})
        columns = data.get("columns", [])
        rows = data.get("rows", [])
        error = data.get("error")

        if error:
            response = StructuredResponse(
                header=ResponseHeader(
                    title="Log Query",
                    icon="🔍",
                    subtitle="Query failed",
                    agent_name=agent_name,
                    severity=Severity.HIGH,
                ),
            )
            response.add_section(content=f"**Error:** {error}")
            if data.get("suggestion"):
                response.add_section(content=f"*Suggestion:* {data['suggestion']}")
            return ParseResult(response=response, table_data=data)

        response = StructuredResponse(
            header=ResponseHeader(
                title="Log Query Results",
                icon="🔍",
                subtitle=f"{result_count} results found",
                agent_name=agent_name,
                severity=Severity.INFO if result_count > 0 else Severity.SUCCESS,
            ),
        )

        # Add query info
        hours = time_range.get("hours", 1)
        response.add_section(content=f"*Query:* `{query[:80]}...`" if len(query) > 80 else f"*Query:* `{query}`")
        response.add_section(content=f"*Time range:* Last {hours} hour(s)")

        # Add results table
        if rows and columns:
            # Take first 4 columns max for display
            display_cols = columns[:4]
            table = TableData(
                headers=display_cols,
                rows=[
                    TableRow(cells=[str(row[i])[:50] if i < len(row) else "" for i in range(len(display_cols))])
                    for row in rows[:20]
                ],
            )
            response.add_section(table=table)
        elif result_count == 0:
            response.add_section(content="No matching log entries found.")

        return ParseResult(response=response, table_data=rows)

    def _parse_database_list(
        self, data: dict, title: str, agent_name: str | None
    ) -> ParseResult:
        """Parse a database list response."""
        databases = data.get("databases", [])
        count = data.get("count", len(databases))

        response = StructuredResponse(
            header=ResponseHeader(
                title=title,
                icon="🗄️",
                subtitle=f"{count} databases found",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add compartment info if present
        compartments = data.get("compartments_searched")
        if compartments is not None:
            response.add_section(
                content=f"*Compartments searched:* {compartments}",
            )

        # Add databases table
        if databases:
            table = self._list_to_table(databases, title="Databases")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=databases)

    def _parse_managed_database(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse managed_database (single database details) response."""
        name = data.get("name", "Unknown")
        db_type = data.get("database_type")
        db_subtype = data.get("database_sub_type")
        status = data.get("database_status")
        version = data.get("database_version")

        response = StructuredResponse(
            header=ResponseHeader(
                title=f"Database: {name}",
                icon="🗄️",
                agent_name=agent_name,
                severity=self._status_severity(status),
            ),
        )

        # Add details as metrics
        fields = []
        if db_type or db_subtype:
            type_str = " / ".join([t for t in (db_type, db_subtype) if t])
            fields.append(MetricValue(label="Type", value=type_str))
        if status:
            fields.append(MetricValue(
                label="Status",
                value=status,
                severity=self._status_severity(status),
            ))
        if version:
            fields.append(MetricValue(label="Version", value=version))

        if fields:
            response.add_section(title="Details", fields=fields)

        return ParseResult(response=response, raw_text=name)

    def _parse_top_sql_cpu_activity(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse top_sql_cpu_activity response."""
        activities = data.get("activities", [])
        sql_count = data.get("sql_count", len(activities))

        response = StructuredResponse(
            header=ResponseHeader(
                title="Top SQL by CPU",
                icon="⏱️",
                subtitle=f"{sql_count} SQL statements",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        if activities:
            table = self._list_to_table(activities, title="SQL Activity")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=activities)

    def _parse_top_wait_events(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse top_wait_events response."""
        wait_events = data.get("wait_events", [])
        snapshot_range = data.get("snapshot_range", "N/A")

        response = StructuredResponse(
            header=ResponseHeader(
                title="Top Wait Events",
                icon="⏳",
                subtitle=f"Snapshot: {snapshot_range}",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        response.add_section(
            content=f"*Events:* {data.get('wait_event_count', len(wait_events))}"
        )

        if wait_events:
            table = self._list_to_table(wait_events, title="Wait Events")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=wait_events)

    def _parse_sql_plan_baselines(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse sql_plan_baselines response."""
        baselines = data.get("baselines", [])
        count = data.get("baseline_count", len(baselines))

        response = StructuredResponse(
            header=ResponseHeader(
                title="SQL Plan Baselines",
                icon="🔖",
                subtitle=f"{count} baselines",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        if baselines:
            table = self._list_to_table(baselines, title="Baselines")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=baselines)

    def _parse_awr_report(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse awr_report response."""
        return self._parse_awr(data, "AWR Report", agent_name)

    def _parse_awr_sql_report(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse awr_sql_report response."""
        return self._parse_awr(data, "AWR SQL Report", agent_name)

    def _parse_awr(self, data: dict, title: str, agent_name: str | None) -> ParseResult:
        """Parse AWR/ASH report response."""
        content = data.get("content", "")
        truncated = data.get("truncated", False)

        response = StructuredResponse(
            header=ResponseHeader(
                title=title,
                icon="📄",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add metadata
        fields = []
        if data.get("db_name"):
            fields.append(MetricValue(label="Database", value=data["db_name"]))
        if data.get("sql_id"):
            fields.append(MetricValue(label="SQL ID", value=data["sql_id"]))
        if data.get("snapshot_range"):
            fields.append(MetricValue(label="Snapshot Range", value=data["snapshot_range"]))
        if data.get("hours_covered") is not None:
            fields.append(MetricValue(label="Hours Covered", value=data["hours_covered"]))

        if fields:
            response.add_section(title="Report Info", fields=fields, divider_after=True)

        # Add content or file attachment
        if content:
            # Check if it's HTML content (likely full report)
            if content.strip().startswith("<") or len(content) > 5000:
                # Add as file attachment for large reports
                filename = f"{title.lower().replace(' ', '_')}.html"
                response.add_file(
                    content=content,
                    filename=filename,
                    content_type="text/html",
                    title=title,
                    comment="Full AWR report attached" + (" (truncated)" if truncated else ""),
                )
            else:
                # Add as code block for small text content
                suffix = " (truncated)" if truncated else ""
                from src.formatting.base import CodeBlock
                response.add_section(
                    title=f"Report Content{suffix}",
                    code_block=CodeBlock(code=content, language="sql"),
                )

        return ParseResult(response=response, raw_text=content[:500] if content else None)

    def _parse_database_metrics(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse database_metrics response."""
        metrics = data.get("metrics", [])
        time_range = data.get("time_range", "N/A")

        response = StructuredResponse(
            header=ResponseHeader(
                title="Database Metrics",
                icon="📈",
                subtitle=f"Time Range: {time_range}",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        response.add_section(
            content=f"*Metrics:* {data.get('metric_count', len(metrics))}"
        )

        if metrics:
            table = self._list_to_table(metrics, title="Metrics")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # OPSI Type Parsers (Operations Insights)
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_sql_statistics(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse sql_statistics response from OPSI."""
        distilled = data.get("distilled_summary", {})
        items = data.get("items", [])
        sql_count = data.get("sql_count", len(items))

        response = StructuredResponse(
            header=ResponseHeader(
                title="SQL Statistics",
                icon="📊",
                subtitle=f"{sql_count} SQL statements",
                agent_name=agent_name,
                severity=self._sql_stats_severity(distilled),
            ),
        )

        # Add distilled summary as content
        summary_text = distilled.get("summary", "")
        if summary_text:
            response.add_section(content=summary_text)

        # Add totals as metrics
        totals = distilled.get("totals", {})
        if totals:
            metrics = [
                MetricValue(label="DB Time", value=f"{totals.get('database_time_sec', 0):.1f}s"),
                MetricValue(label="CPU Time", value=f"{totals.get('cpu_time_sec', 0):.1f}s"),
                MetricValue(label="IO Time", value=f"{totals.get('io_time_sec', 0):.1f}s"),
                MetricValue(label="Executions", value=str(totals.get("executions", 0))),
            ]
            response.add_metrics("Totals", metrics)

        # Add problem counts if any issues
        problem_counts = distilled.get("problem_counts", {})
        degrading = problem_counts.get("degrading", 0)
        inefficient = problem_counts.get("inefficient", 0)
        if degrading > 0 or inefficient > 0:
            problem_metrics = []
            if degrading > 0:
                problem_metrics.append(MetricValue(
                    label="Degrading",
                    value=str(degrading),
                    severity=Severity.HIGH,
                ))
            if inefficient > 0:
                problem_metrics.append(MetricValue(
                    label="Inefficient",
                    value=str(inefficient),
                    severity=Severity.MEDIUM,
                ))
            response.add_metrics("Issues Detected", problem_metrics)

        # Add top offenders metrics
        top_offenders = distilled.get("top_offenders", [])
        if top_offenders:
            top_metrics = []
            for sql in top_offenders[:3]:
                label = (sql.get("sql_id") or "SQL")[:15]
                db_time = sql.get("db_time_sec", 0)
                top_metrics.append(
                    MetricValue(
                        label=label,
                        value=f"{db_time:.1f}s",
                        unit="db time",
                    )
                )
            if top_metrics:
                response.add_metrics("Top SQL by DB Time", top_metrics)

        # Add next action
        next_action = distilled.get("next_action", "")
        if next_action:
            response.add_section(content=f"*Next Action:* {next_action}")

        # Add items table if present
        if items:
            # Create a simplified table with key columns
            table_items = []
            for item in items[:10]:  # Limit to 10 rows
                table_items.append({
                    "SQL": (item.get("sql_identifier") or "")[:15],
                    "Database": (item.get("database_display_name") or "")[:15],
                    "DB Time (s)": f"{item.get('database_time_in_sec', 0):.1f}",
                    "CPU (s)": f"{item.get('cpu_time_in_sec', 0):.1f}",
                    "Executions": item.get("executions_count", 0),
                })
            table = self._list_to_table(table_items, title="SQL Details")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=items)

    def _parse_addm_findings(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse addm_findings response from OPSI."""
        distilled = data.get("distilled_summary", {})
        items = data.get("items", [])
        finding_count = data.get("finding_count", len(items))

        severity = Severity.SUCCESS
        if distilled.get("severity") == "critical":
            severity = Severity.CRITICAL
        elif distilled.get("severity") == "high":
            severity = Severity.HIGH
        elif distilled.get("severity") == "medium":
            severity = Severity.MEDIUM

        response = StructuredResponse(
            header=ResponseHeader(
                title="ADDM Findings",
                icon="🔍",
                subtitle=f"{finding_count} findings",
                agent_name=agent_name,
                severity=severity,
            ),
        )

        # Add distilled summary
        summary_text = distilled.get("summary", "")
        if summary_text:
            response.add_section(content=summary_text)

        # Add severity counts as metrics
        metrics = []
        for sev_name, sev_level in [("critical_count", Severity.CRITICAL),
                                     ("high_count", Severity.HIGH),
                                     ("medium_count", Severity.MEDIUM)]:
            count = distilled.get(sev_name, 0)
            if count > 0:
                metrics.append(MetricValue(
                    label=sev_name.replace("_count", "").title(),
                    value=str(count),
                    severity=sev_level,
                ))
        if metrics:
            response.add_metrics("Finding Severity", metrics)

        # Add next action
        next_action = distilled.get("next_action", "")
        if next_action:
            response.add_section(content=f"*Next Action:* {next_action}")

        # Add findings table
        if items:
            table_items = []
            for item in items[:10]:
                table_items.append({
                    "Category": (item.get("category_name") or "")[:20],
                    "Database": (item.get("database_display_name") or "")[:15],
                    "Impact %": f"{item.get('impact_overall_percent', 0):.1f}",
                    "Details": (item.get("finding_details") or "")[:40],
                })
            table = self._list_to_table(table_items, title="Finding Details")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=items)

    def _parse_addm_recommendations(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse addm_recommendations response from OPSI."""
        distilled = data.get("distilled_summary", {})
        items = data.get("items", [])
        rec_count = data.get("recommendation_count", len(items))

        response = StructuredResponse(
            header=ResponseHeader(
                title="ADDM Recommendations",
                icon="💡",
                subtitle=f"{rec_count} recommendations",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        # Add summary
        summary_text = distilled.get("summary", "")
        if summary_text:
            response.add_section(content=summary_text)

        # Add key metrics
        total_benefit = distilled.get("total_potential_benefit_percent", 0)
        restart_count = distilled.get("requires_restart_count", 0)
        if total_benefit > 0 or restart_count > 0:
            metrics = [
                MetricValue(
                    label="Potential Benefit",
                    value=f"{total_benefit:.1f}%",
                    severity=Severity.SUCCESS if total_benefit > 10 else Severity.INFO,
                ),
            ]
            if restart_count > 0:
                metrics.append(MetricValue(
                    label="Require Restart",
                    value=str(restart_count),
                    severity=Severity.MEDIUM,
                ))
            response.add_metrics("Summary", metrics)

        # Add next action
        next_action = distilled.get("next_action", "")
        if next_action:
            response.add_section(content=f"*Next Action:* {next_action}")

        # Add recommendations table
        if items:
            table_items = []
            for item in items[:10]:
                table_items.append({
                    "Type": (item.get("type") or "")[:15],
                    "Benefit %": f"{item.get('overall_benefit_percent', 0):.1f}",
                    "Restart": "Yes" if item.get("require_restart") else "No",
                    "Message": (item.get("message") or "")[:40],
                })
            table = self._list_to_table(table_items, title="Recommendation Details")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=items)

    def _parse_sql_insights_summary(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse sql_insights_summary response from OPSI."""
        insights = data.get("insights", [])

        response = StructuredResponse(
            header=ResponseHeader(
                title="SQL Insights",
                icon="🔎",
                subtitle=f"{len(insights)} insights",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        if insights:
            # Add insights as list items
            insight_texts = []
            for insight in insights[:10]:
                category = insight.get("category", "")
                text = insight.get("text") or insight.get("description", "")
                if text:
                    insight_texts.append(f"*{category}*: {text[:100]}")
            if insight_texts:
                response.add_section(list_items=insight_texts)

            # Also add as table for structured view
            table = self._list_to_table(insights[:10], title="Insight Details")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=insights)

    def _parse_database_insights(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse database_insights response from OPSI."""
        items = data.get("items", [])
        db_count = data.get("database_count", len(items))

        response = StructuredResponse(
            header=ResponseHeader(
                title="Database Insights",
                icon="📈",
                subtitle=f"{db_count} databases",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        if items:
            table_items = []
            for item in items[:10]:
                table_items.append({
                    "Database": (item.get("database_display_name") or "")[:20],
                    "Type": (item.get("database_type") or "")[:15],
                    "Status": item.get("status", ""),
                    "Host": (item.get("host_name") or "")[:20],
                })
            table = self._list_to_table(table_items, title="Database List")
            response.add_section(table=table)

        return ParseResult(response=response, table_data=items)

    def _sql_stats_severity(self, distilled: dict) -> Severity:
        """Determine severity from SQL statistics summary."""
        severity_str = distilled.get("severity", "low")
        if severity_str == "critical":
            return Severity.CRITICAL
        if severity_str == "warning":
            return Severity.MEDIUM

        # Check problem counts
        problem_counts = distilled.get("problem_counts", {})
        if problem_counts.get("degrading", 0) > 0:
            return Severity.HIGH
        if problem_counts.get("inefficient", 0) > 0:
            return Severity.MEDIUM
        return Severity.INFO

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_json_array(self, message: str) -> list[dict] | None:
        """Extract a JSON array from the message if present."""
        if "[" not in message or "]" not in message:
            return None

        try:
            start = message.find("[")
            end = message.rfind("]") + 1
            json_part = message[start:end]
            parsed = json.loads(json_part)

            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                return parsed
        except (json.JSONDecodeError, IndexError):
            pass

        return None

    def _table_response(
        self, data: list[dict], agent_name: str | None
    ) -> ParseResult:
        """Create a response from table data."""
        title = self._infer_table_title(data)

        response = StructuredResponse(
            header=ResponseHeader(
                title=title,
                icon="📋",
                subtitle=f"{len(data)} items",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        table = self._list_to_table(data, title=title)
        response.add_section(table=table)

        return ParseResult(response=response, table_data=data)

    def _list_to_table(
        self,
        data: list[dict],
        title: str | None = None,
        max_columns: int = 6,
    ) -> TableData:
        """Convert a list of dicts to TableData."""
        if not data:
            return TableData(title=title)

        # Get headers from first item (limit columns)
        all_keys = list(data[0].keys())
        headers = all_keys[:max_columns]

        # Convert rows
        rows = []
        for item in data:
            cells = [str(item.get(key, "")) for key in headers]
            rows.append(TableRow(cells=cells))

        return TableData(
            title=title,
            headers=[self._format_header(h) for h in headers],
            rows=rows,
        )

    def _format_header(self, key: str) -> str:
        """Format a dict key as a table header."""
        return key.replace("_", " ").title()

    def _infer_table_title(self, data: list[dict]) -> str:
        """Infer a table title from the data structure."""
        if not data:
            return "Results"

        first_item = data[0]
        keys = set(first_item.keys())

        # Cost data
        if "service" in keys and ("cost" in keys or "percent" in keys):
            return "Cost by Service"

        # OCI resources
        if "compartment_id" in keys or "compartment_ocid" in keys:
            return "OCI Resources"

        if "display_name" in keys and "lifecycle_state" in keys:
            if "cidr_block" in keys:
                return "VCNs"
            if "shape" in keys:
                return "Compute Instances"
            if "db_name" in keys:
                return "Databases"
            return "Resources"

        if "name" in keys and "id" in keys:
            first_id = str(first_item.get("id", ""))
            if "tenancy" in first_id:
                return "Compartments"
            return "Items"

        return "Results"

    def _clean_text(self, message: str) -> str:
        """Clean up message text by removing JSON artifacts."""
        # Remove JSON blocks
        if '"thought"' in message or message.strip().startswith("{"):
            message = re.sub(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', message)

        # Clean up markdown
        message = message.replace('```json', '').replace('```', '')
        message = re.sub(r'\s+', ' ', message)

        return message.strip()

    def _health_severity(self, summary: dict) -> Severity:
        """Determine severity from health summary."""
        critical = summary.get("critical_count", 0) or 0
        warning = summary.get("warning_count", 0) or 0

        if critical > 0:
            return Severity.CRITICAL
        if warning > 0:
            return Severity.MEDIUM
        return Severity.SUCCESS

    def _status_severity(self, status: str | None) -> Severity:
        """Convert database status to severity."""
        if not status:
            return Severity.INFO

        status_lower = status.lower()
        if status_lower in ("available", "running", "active", "up"):
            return Severity.SUCCESS
        if status_lower in ("warning", "degraded"):
            return Severity.MEDIUM
        if status_lower in ("critical", "down", "failed", "terminated"):
            return Severity.CRITICAL
        if status_lower in ("stopped", "stopping", "unavailable"):
            return Severity.HIGH

        return Severity.INFO
