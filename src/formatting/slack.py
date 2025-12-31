"""
Slack Block Kit formatter for agent responses.

Converts StructuredResponse to Slack Block Kit format for rich,
interactive messages in Slack.

Reference: https://api.slack.com/block-kit
"""

from __future__ import annotations

from typing import Any

from src.formatting.base import (
    ActionButton,
    BaseFormatter,
    CodeBlock,
    FormatterRegistry,
    ListItem,
    MetricValue,
    OutputFormat,
    Section,
    Severity,
    StatusIndicator,
    StructuredResponse,
    TableData,
    TrendDirection,
)


class SlackFormatter(BaseFormatter):
    """
    Slack Block Kit formatter.

    Converts structured responses to Slack's Block Kit format,
    supporting sections, fields, buttons, and rich formatting.
    """

    # Slack-specific emoji mappings
    SEVERITY_EMOJI = {
        Severity.CRITICAL: ":red_circle:",
        Severity.HIGH: ":large_orange_circle:",
        Severity.MEDIUM: ":large_yellow_circle:",
        Severity.LOW: ":large_blue_circle:",
        Severity.INFO: ":information_source:",
        Severity.SUCCESS: ":white_check_mark:",
    }

    TREND_EMOJI = {
        TrendDirection.UP: ":chart_with_upwards_trend:",
        TrendDirection.DOWN: ":chart_with_downwards_trend:",
        TrendDirection.STABLE: ":arrow_right:",
    }

    @property
    def format_type(self) -> OutputFormat:
        return OutputFormat.SLACK

    def format_response(self, response: StructuredResponse) -> dict[str, Any]:
        """
        Format a structured response as Slack Block Kit.

        Args:
            response: Structured response to format

        Returns:
            Slack Block Kit message payload
        """
        blocks: list[dict[str, Any]] = []

        # Header
        blocks.append(self._format_header(response))

        # Error handling
        if response.error:
            blocks.append(self._format_error_block(response.error))
            return {"blocks": blocks}

        # Sections
        for section in response.sections:
            section_blocks = self._format_section(section)
            blocks.extend(section_blocks)

        # Footer
        if response.footer:
            blocks.append(self._format_footer(response.footer))

        return {"blocks": blocks}

    def format_error(
        self, error: str, title: str = "Error"
    ) -> dict[str, Any]:
        """Format an error message as Slack Block Kit."""
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f":x: {title}",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"```{error}```",
                    },
                },
            ]
        }

    def _format_header(self, response: StructuredResponse) -> dict[str, Any]:
        """Format response header as Slack block."""
        header = response.header
        icon = self._get_icon(header.icon, header.severity)
        title_text = f"{icon} {header.title}" if icon else header.title

        block: dict[str, Any] = {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title_text[:150],  # Slack header limit
                "emoji": True,
            },
        }

        # Add context with subtitle if present
        if header.subtitle or header.agent_name or header.timestamp:
            context_elements = []
            if header.subtitle:
                context_elements.append({
                    "type": "mrkdwn",
                    "text": header.subtitle,
                })
            if header.agent_name:
                context_elements.append({
                    "type": "mrkdwn",
                    "text": f"_Agent: {header.agent_name}_",
                })
            if header.timestamp:
                context_elements.append({
                    "type": "mrkdwn",
                    "text": f"_Generated: {header.timestamp}_",
                })

            return [
                block,
                {
                    "type": "context",
                    "elements": context_elements[:10],  # Slack limit
                },
            ]

        return block

    def _format_section(self, section: Section) -> list[dict[str, Any]]:
        """Format a section as Slack blocks."""
        blocks: list[dict[str, Any]] = []

        # Section title
        if section.title:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{section.title}*",
                },
            })

        # Content text
        if section.content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": section.content[:3000],  # Slack limit
                },
            })

        # Fields (metrics or status indicators)
        if section.fields:
            field_blocks = self._format_fields(section.fields)
            blocks.extend(field_blocks)

        # List items
        if section.list_items:
            list_block = self._format_list_items(section.list_items)
            blocks.append(list_block)

        # Table
        if section.table:
            table_blocks = self._format_table(section.table)
            blocks.extend(table_blocks)

        # Code block
        if section.code_block:
            code_block = self._format_code_block(section.code_block)
            blocks.append(code_block)

        # Actions (buttons)
        if section.actions:
            actions_block = self._format_actions(section.actions)
            blocks.append(actions_block)

        # Divider
        if section.divider_after:
            blocks.append({"type": "divider"})

        return blocks

    def _format_fields(
        self, fields: list[StatusIndicator | MetricValue]
    ) -> list[dict[str, Any]]:
        """Format fields as Slack section with fields."""
        slack_fields: list[dict[str, Any]] = []

        for f in fields:
            if isinstance(f, StatusIndicator):
                emoji = self.SEVERITY_EMOJI.get(f.severity, "")
                trend = self.TREND_EMOJI.get(f.trend, "") if f.trend else ""
                text = f"{emoji} *{f.label}*\n{f.value} {trend}"
                if f.description:
                    text += f"\n_{f.description}_"
            else:  # MetricValue
                emoji = self.SEVERITY_EMOJI.get(f.severity, "")
                trend = self.TREND_EMOJI.get(f.trend, "") if f.trend else ""
                value_str = f"{f.value}"
                if f.unit:
                    value_str += f" {f.unit}"
                text = f"{emoji} *{f.label}*\n{value_str} {trend}"
                if f.threshold:
                    text += f"\n_Threshold: {f.threshold}{' ' + f.unit if f.unit else ''}_"
                if f.change_percent is not None:
                    sign = "+" if f.change_percent > 0 else ""
                    text += f" ({sign}{f.change_percent:.1f}%)"

            slack_fields.append({
                "type": "mrkdwn",
                "text": text[:2000],  # Slack field limit
            })

        # Slack allows max 10 fields per section, 2 columns
        result: list[dict[str, Any]] = []
        for i in range(0, len(slack_fields), 10):
            chunk = slack_fields[i:i + 10]
            result.append({
                "type": "section",
                "fields": chunk,
            })

        return result

    def _format_list_items(self, items: list[ListItem]) -> dict[str, Any]:
        """Format list items as mrkdwn text."""
        lines = []
        for item in items:
            icon = self.SEVERITY_EMOJI.get(item.severity, "•") if item.severity else "•"
            lines.append(f"{icon} {item.text}")
            if item.details:
                lines.append(f"    _{item.details}_")
            if item.sublist:
                for sub in item.sublist:
                    sub_icon = self.SEVERITY_EMOJI.get(sub.severity, "◦") if sub.severity else "◦"
                    lines.append(f"    {sub_icon} {sub.text}")

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(lines)[:3000],
            },
        }

    def _format_table(self, table: TableData) -> list[dict[str, Any]]:
        """
        Format table using native Slack table blocks.

        Uses the new Slack table block type (2024) which provides:
        - Native table rendering with header row
        - Column alignment and wrapping options
        - Up to 100 rows and 20 columns

        Note: Only ONE table block is allowed per message.

        Reference: https://docs.slack.dev/reference/block-kit/blocks/table-block
        """
        blocks: list[dict[str, Any]] = []

        # Add title as section before table
        if table.title:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{table.title}*",
                },
            })

        # Build native table block rows
        # First row is the header row
        table_rows = []

        # Header row with column names
        header_cells = []
        for h in table.headers[:20]:  # Max 20 columns
            header_cells.append({
                "type": "raw_text",
                "text": str(h)[:100],  # Limit cell text length
            })
        table_rows.append(header_cells)

        # Data rows (max 100 rows including header)
        for row in table.rows[:99]:  # Leave room for header
            row_cells = []
            for i, cell in enumerate(row.cells[:20]):  # Max 20 columns
                cell_text = str(cell) if cell is not None else ""
                # Add severity emoji prefix if present
                if i == 0 and row.severity:
                    emoji = self.SEVERITY_EMOJI.get(row.severity, "")
                    if emoji:
                        cell_text = f"{emoji} {cell_text}"
                row_cells.append({
                    "type": "raw_text",
                    "text": cell_text[:100],  # Limit cell text length
                })

            # Pad row if fewer cells than headers
            while len(row_cells) < len(header_cells):
                row_cells.append({"type": "raw_text", "text": ""})

            table_rows.append(row_cells)

        # Build column settings (alignment)
        column_settings = []
        for i, h in enumerate(table.headers[:20]):
            # First column left-aligned, others based on content type
            h_lower = h.lower()
            if any(x in h_lower for x in ["id", "ocid", "name"]):
                align = "left"
            elif any(x in h_lower for x in ["count", "number", "size", "cost", "price"]):
                align = "right"
            else:
                align = "left"

            column_settings.append({
                "align": align,
                "is_wrapped": True,  # Enable wrapping for readability
            })

        # Create native table block
        table_block = {
            "type": "table",
            "rows": table_rows,
            "column_settings": column_settings,
        }
        blocks.append(table_block)

        # Add footer context
        if table.footer:
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": f"_{table.footer}_",
                }],
            })

        return blocks

    def format_table_from_list(
        self,
        items: list[dict[str, Any]],
        columns: list[str] | None = None,
        title: str | None = None,
        footer: str | None = None,
    ) -> dict[str, Any]:
        """
        Create Slack table block from a list of dictionaries.

        This is a convenience method for directly formatting list data
        without going through StructuredResponse.

        Args:
            items: List of dicts to display as table rows
            columns: Column keys to include (defaults to all keys from first item)
            title: Optional table title
            footer: Optional footer text

        Returns:
            Slack Block Kit payload with table block

        Example:
            >>> formatter.format_table_from_list(
            ...     items=[{"name": "prod", "state": "ACTIVE", "vcn_id": "ocid1..."}],
            ...     columns=["name", "state"],
            ...     title="Compartments"
            ... )
        """
        if not items:
            return {
                "blocks": [{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "_No data to display_"},
                }]
            }

        # Determine columns from first item if not specified
        if columns is None:
            columns = list(items[0].keys())

        blocks: list[dict[str, Any]] = []

        # Title
        if title:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{title}*"},
            })

        # Build table rows
        table_rows = []

        # Header row
        header_cells = [
            {"type": "raw_text", "text": self._format_column_header(col)}
            for col in columns[:20]
        ]
        table_rows.append(header_cells)

        # Data rows
        for item in items[:99]:
            row_cells = []
            for col in columns[:20]:
                value = item.get(col, "")
                # Format special values
                cell_text = self._format_cell_value(value)
                row_cells.append({"type": "raw_text", "text": cell_text[:100]})
            table_rows.append(row_cells)

        # Column settings
        column_settings = [
            {"align": "left", "is_wrapped": True}
            for _ in columns[:20]
        ]

        # Table block
        blocks.append({
            "type": "table",
            "rows": table_rows,
            "column_settings": column_settings,
        })

        # Footer
        if footer:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"_{footer}_"}],
            })

        return {"blocks": blocks}

    def _format_column_header(self, col: str) -> str:
        """Format column name as readable header."""
        # Convert snake_case to Title Case
        return col.replace("_", " ").title()

    def _format_cell_value(self, value: Any) -> str:
        """Format cell value for display."""
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value[:3])
        if isinstance(value, dict):
            return str(value)[:50]
        return str(value)[:100]

    def _format_code_block(self, code_block: CodeBlock) -> dict[str, Any]:
        """Format code block as Slack section."""
        text = ""
        if code_block.title:
            text = f"*{code_block.title}*\n"
        text += f"```{code_block.code}```"

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text[:3000],
            },
        }

    def _format_actions(self, actions: list[ActionButton]) -> dict[str, Any]:
        """Format action buttons as Slack actions block."""
        buttons: list[dict[str, Any]] = []

        for action in actions[:5]:  # Slack limit: 5 buttons per block
            button: dict[str, Any] = {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": action.label[:75],  # Slack limit
                    "emoji": True,
                },
                "action_id": action.action_id,
            }

            if action.value:
                button["value"] = action.value

            if action.style in ("primary", "danger"):
                button["style"] = action.style

            if action.confirm_title and action.confirm_text:
                button["confirm"] = {
                    "title": {
                        "type": "plain_text",
                        "text": action.confirm_title,
                    },
                    "text": {
                        "type": "mrkdwn",
                        "text": action.confirm_text,
                    },
                    "confirm": {
                        "type": "plain_text",
                        "text": "Confirm",
                    },
                    "deny": {
                        "type": "plain_text",
                        "text": "Cancel",
                    },
                }

            buttons.append(button)

        return {
            "type": "actions",
            "elements": buttons,
        }

    def _format_footer(self, footer: ResponseFooter) -> dict[str, Any]:
        """Format footer as Slack context block."""
        elements: list[dict[str, Any]] = []

        if footer.text:
            elements.append({
                "type": "mrkdwn",
                "text": footer.text,
            })

        if footer.duration_ms is not None:
            elements.append({
                "type": "mrkdwn",
                "text": f":stopwatch: {footer.duration_ms}ms",
            })

        if footer.timestamp:
            elements.append({
                "type": "mrkdwn",
                "text": f"_{footer.timestamp}_",
            })

        if footer.next_steps:
            steps_text = "*Next Steps:* " + " → ".join(footer.next_steps[:3])
            elements.append({
                "type": "mrkdwn",
                "text": steps_text[:2000],
            })

        if footer.help_text:
            elements.append({
                "type": "mrkdwn",
                "text": f":question: {footer.help_text}",
            })

        return {
            "type": "context",
            "elements": elements[:10],  # Slack limit
        }

    def _format_error_block(self, error: str) -> dict[str, Any]:
        """Format error message block."""
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":x: *Error*\n```{error}```",
            },
        }

    def _get_icon(self, icon: str | None, severity: Severity | None) -> str:
        """Get icon emoji."""
        if icon:
            # If it's already an emoji or starts with :, return as-is
            if icon.startswith(":") or not icon.isascii():
                return icon
            # Otherwise wrap in colons for Slack
            return f":{icon}:"

        if severity:
            return self.SEVERITY_EMOJI.get(severity, "")

        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Register the Slack formatter
# ─────────────────────────────────────────────────────────────────────────────

# Auto-register on import
FormatterRegistry.register(SlackFormatter())
