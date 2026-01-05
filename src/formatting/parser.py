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
from typing import Any

import structlog

from src.formatting.base import (
    FileAttachment,
    ListItem,
    MetricValue,
    ResponseFooter,
    ResponseHeader,
    Section,
    Severity,
    StructuredResponse,
    TableData,
    TableRow,
    TrendDirection,
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
        Parse a typed JSON response (with "type" field).

        Returns None if not a typed response.
        """
        if "{" not in message or '"type"' not in message:
            return None

        try:
            start = message.find("{")
            end = message.rfind("}") + 1
            json_part = message[start:end]
            parsed = json.loads(json_part)

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

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            self._logger.debug("Failed to parse typed response", error=str(e))
            return None

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

    def _parse_fleet_health(self, data: dict, agent_name: str | None) -> ParseResult:
        """Parse fleet_health response."""
        summary = data.get("summary", {}) or {}
        statistics = data.get("statistics", [])

        response = StructuredResponse(
            header=ResponseHeader(
                title="Database Fleet Health",
                icon="💓",
                agent_name=agent_name,
                severity=self._health_severity(summary),
            ),
        )

        # Add health metrics
        metrics = [
            MetricValue(label="Databases", value=summary.get("database_count", "N/A")),
            MetricValue(
                label="Healthy",
                value=summary.get("healthy_count", "N/A"),
                severity=Severity.SUCCESS,
            ),
            MetricValue(
                label="Warning",
                value=summary.get("warning_count", "N/A"),
                severity=Severity.MEDIUM,
            ),
            MetricValue(
                label="Critical",
                value=summary.get("critical_count", "N/A"),
                severity=Severity.CRITICAL,
            ),
            MetricValue(
                label="Unavailable",
                value=summary.get("unavailable_count", "N/A"),
                severity=Severity.HIGH,
            ),
        ]
        response.add_metrics("Health Status", metrics, divider_after=True)

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

        response = StructuredResponse(
            header=ResponseHeader(
                title="Database Connections",
                icon="🔗",
                subtitle=f"{count} connection(s) found" if count else message or "No connections",
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
        )

        if connections:
            table = self._list_to_table(connections, title="Connections")
            response.add_section(table=table)
        elif message:
            response.add_section(content=message)

        return ParseResult(response=response, table_data=connections)

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
