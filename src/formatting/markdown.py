"""
Markdown formatter for agent responses.

Converts StructuredResponse to Markdown format for CLI output,
documentation, and other markdown-supporting environments.
"""

from __future__ import annotations

from src.formatting.base import (
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


class MarkdownFormatter(BaseFormatter):
    """
    Markdown formatter.

    Converts structured responses to clean, readable Markdown.
    """

    # Unicode icons for markdown
    SEVERITY_ICONS = {
        Severity.CRITICAL: "🔴",
        Severity.HIGH: "🟠",
        Severity.MEDIUM: "🟡",
        Severity.LOW: "🔵",
        Severity.INFO: "ℹ️",
        Severity.SUCCESS: "✅",
    }

    TREND_ICONS = {
        TrendDirection.UP: "📈",
        TrendDirection.DOWN: "📉",
        TrendDirection.STABLE: "➡️",
    }

    @property
    def format_type(self) -> OutputFormat:
        return OutputFormat.MARKDOWN

    def format_response(self, response: StructuredResponse) -> str:
        """
        Format a structured response as Markdown.

        Args:
            response: Structured response to format

        Returns:
            Markdown formatted string
        """
        lines: list[str] = []

        # Header
        lines.extend(self._format_header(response))

        # Error handling
        if response.error:
            lines.append("")
            lines.append(f"> ⚠️ **Error**: {response.error}")
            return "\n".join(lines)

        # Sections
        for section in response.sections:
            section_lines = self._format_section(section)
            lines.append("")
            lines.extend(section_lines)

        # Footer
        if response.footer:
            lines.append("")
            lines.extend(self._format_footer(response.footer))

        return "\n".join(lines)

    def format_error(self, error: str, title: str = "Error") -> str:
        """Format an error message as Markdown."""
        return f"## ❌ {title}\n\n```\n{error}\n```"

    def _format_header(self, response: StructuredResponse) -> list[str]:
        """Format response header."""
        header = response.header
        lines = []

        # Icon
        icon = ""
        if header.icon:
            icon = f"{header.icon} "
        elif header.severity:
            icon = f"{self.SEVERITY_ICONS.get(header.severity, '')} "

        # Slack doesn't support headers (#), use Bold
        lines.append(f"*{icon}{header.title}*")

        if header.subtitle:
            lines.append("")
            lines.append(f"_{header.subtitle}_")

        # Context line
        context_parts = []
        if header.agent_name:
            context_parts.append(f"Agent: {header.agent_name}")
        if header.timestamp:
            context_parts.append(f"Generated: {header.timestamp}")

        if context_parts:
            lines.append("")
            lines.append(f"_{' | '.join(context_parts)}_")

        return lines

    def _format_section(self, section: Section) -> list[str]:
        """Format a section."""
        lines: list[str] = []

        # Section title - Slack doesn't support ###, use Bold
        if section.title:
            lines.append(f"*{section.title}*")
            lines.append("")

        # Content text
        if section.content:
            lines.append(section.content)
            lines.append("")

        # Fields (metrics or status indicators)
        if section.fields:
            field_lines = self._format_fields(section.fields)
            lines.extend(field_lines)
            lines.append("")

        # List items
        if section.list_items:
            list_lines = self._format_list_items(section.list_items)
            lines.extend(list_lines)
            lines.append("")

        # Table
        if section.table:
            table_lines = self._format_table(section.table)
            lines.extend(table_lines)
            lines.append("")

        # Code block
        if section.code_block:
            code_lines = self._format_code_block(section.code_block)
            lines.extend(code_lines)
            lines.append("")

        # Divider
        if section.divider_after:
            lines.append("---")

        return lines

    def _format_fields(
        self, fields: list[StatusIndicator | MetricValue]
    ) -> list[str]:
        """Format fields as markdown list."""
        lines = []

        for f in fields:
            if isinstance(f, StatusIndicator):
                icon = self.SEVERITY_ICONS.get(f.severity, "•")
                trend = self.TREND_ICONS.get(f.trend, "") if f.trend else ""
                line = f"- {icon} **{f.label}**: {f.value} {trend}"
                if f.description:
                    line += f"\n  - _{f.description}_"
            else:  # MetricValue
                icon = self.SEVERITY_ICONS.get(f.severity, "")
                trend = self.TREND_ICONS.get(f.trend, "") if f.trend else ""
                value_str = f"{f.value}"
                if f.unit:
                    value_str += f" {f.unit}"
                line = f"- {icon} **{f.label}**: {value_str} {trend}"
                if f.threshold:
                    threshold_str = f"{f.threshold}"
                    if f.unit:
                        threshold_str += f" {f.unit}"
                    line += f"\n  - Threshold: {threshold_str}"
                if f.change_percent is not None:
                    sign = "+" if f.change_percent > 0 else ""
                    line += f" ({sign}{f.change_percent:.1f}%)"

            lines.append(line.strip())

        return lines

    def _format_list_items(self, items: list[ListItem], indent: int = 0) -> list[str]:
        """Format list items."""
        lines = []
        prefix = "  " * indent

        for item in items:
            icon = self.SEVERITY_ICONS.get(item.severity, "") if item.severity else ""
            bullet = "•" if not icon else icon
            lines.append(f"{prefix}- {bullet} {item.text}")

            if item.details:
                lines.append(f"{prefix}  - _{item.details}_")

            if item.sublist:
                sub_lines = self._format_list_items(item.sublist, indent + 1)
                lines.extend(sub_lines)

        return lines

    def _format_table(self, table: TableData) -> list[str]:
        """Format table as markdown table."""
        lines = []

        if table.title:
            lines.append(f"**{table.title}**")
            lines.append("")

        if not table.headers:
            return lines

        # Calculate column widths
        widths = [len(h) for h in table.headers]
        for row in table.rows:
            for i, cell in enumerate(row.cells):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Header
        header_cells = [h.ljust(widths[i]) for i, h in enumerate(table.headers)]
        lines.append("| " + " | ".join(header_cells) + " |")

        # Separator
        separator_cells = ["-" * w for w in widths]
        lines.append("| " + " | ".join(separator_cells) + " |")

        # Rows
        for row in table.rows:
            icon = self.SEVERITY_ICONS.get(row.severity, "") if row.severity else ""
            cells = []
            for i, cell in enumerate(row.cells):
                cell_str = str(cell)
                if i == 0 and icon:
                    cell_str = f"{icon} {cell_str}"
                if i < len(widths):
                    cell_str = cell_str.ljust(widths[i])
                cells.append(cell_str)
            lines.append("| " + " | ".join(cells) + " |")

        if table.footer:
            lines.append("")
            lines.append(f"_{table.footer}_")

        # Wrap in code block to ensure alignment in Slack/Text interfaces
        return ["```text"] + lines + ["```"]

    def _format_code_block(self, code_block: CodeBlock) -> list[str]:
        """Format code block."""
        lines = []

        if code_block.title:
            lines.append(f"**{code_block.title}**")
            lines.append("")

        lang = code_block.language or ""
        lines.append(f"```{lang}")
        lines.append(code_block.code)
        lines.append("```")

        return lines

    def _format_footer(self, footer: ResponseFooter) -> list[str]:
        """Format footer."""
        lines = []

        if footer.next_steps:
            lines.append("### Next Steps")
            for step in footer.next_steps:
                lines.append(f"- {step}")
            lines.append("")

        context_parts = []
        if footer.text:
            context_parts.append(footer.text)
        if footer.duration_ms is not None:
            context_parts.append(f"⏱️ {footer.duration_ms}ms")
        if footer.timestamp:
            context_parts.append(footer.timestamp)

        if context_parts:
            lines.append("---")
            lines.append(f"_{' | '.join(context_parts)}_")

        if footer.help_text:
            lines.append("")
            lines.append(f"> 💡 {footer.help_text}")

        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Register the Markdown formatter
# ─────────────────────────────────────────────────────────────────────────────

FormatterRegistry.register(MarkdownFormatter())
