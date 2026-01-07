"""
Teams formatter for agent responses.

Produces compact Markdown suitable for Microsoft Teams and adaptive-card-like
renderers that accept limited Markdown.
"""

from __future__ import annotations

from src.formatting.base import FormatterRegistry, OutputFormat, TableData
from src.formatting.markdown import MarkdownFormatter


class TeamsFormatter(MarkdownFormatter):
    """Teams-friendly formatter with compact tables."""

    @property
    def format_type(self) -> OutputFormat:
        return OutputFormat.TEAMS

    def _format_table(self, table: TableData) -> list[str]:
        """Render tables as bullet lists for Teams compatibility."""
        lines: list[str] = []
        if table.title:
            lines.append(f"**{table.title}**")

        max_rows = 5
        headers = table.headers or []
        for row in table.rows[:max_rows]:
            parts = []
            for idx, cell in enumerate(row.cells):
                label = headers[idx] if idx < len(headers) else f"Col {idx + 1}"
                parts.append(f"{label}: {cell}")
            lines.append(f"- {' | '.join(parts)}")

        remaining = len(table.rows) - max_rows
        if remaining > 0:
            lines.append(f"... and {remaining} more")

        if table.footer:
            lines.append(table.footer)

        return lines


# Register Teams formatter
FormatterRegistry.register(TeamsFormatter())
