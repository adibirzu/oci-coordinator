"""
Base formatting infrastructure for agent responses.

This module provides:
- Structured response types for consistent agent output
- Abstract formatter interface
- Common formatting utilities
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputFormat(str, Enum):
    """Supported output formats."""

    MARKDOWN = "markdown"
    SLACK = "slack"
    TEAMS = "teams"
    HTML = "html"
    PLAIN = "plain"


class Severity(str, Enum):
    """Severity levels for status indicators."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    SUCCESS = "success"


class TrendDirection(str, Enum):
    """Trend direction indicators."""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"


# ─────────────────────────────────────────────────────────────────────────────
# Structured Response Components
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StatusIndicator:
    """A status indicator with severity."""

    label: str
    value: str
    severity: Severity
    description: str | None = None
    trend: TrendDirection | None = None


@dataclass
class MetricValue:
    """A metric with value, unit, and optional comparison."""

    label: str
    value: float | int | str
    unit: str | None = None
    threshold: float | int | str | None = None
    severity: Severity = Severity.INFO
    trend: TrendDirection | None = None
    change_percent: float | None = None


@dataclass
class TableRow:
    """A table row with cell values."""

    cells: list[str | int | float]
    severity: Severity | None = None


@dataclass
class TableData:
    """Table data with headers and rows."""

    title: str | None = None
    headers: list[str] = field(default_factory=list)
    rows: list[TableRow] = field(default_factory=list)
    footer: str | None = None


@dataclass
class ListItem:
    """A list item with optional icon/severity."""

    text: str
    severity: Severity | None = None
    details: str | None = None
    sublist: list[ListItem] | None = None


@dataclass
class CodeBlock:
    """A code block with optional language."""

    code: str
    language: str = ""
    title: str | None = None


@dataclass
class ActionButton:
    """An interactive action button (for Slack/Teams)."""

    label: str
    action_id: str
    style: str = "primary"  # primary, danger, default
    value: str | None = None
    confirm_title: str | None = None
    confirm_text: str | None = None


@dataclass
class FileAttachment:
    """A file attachment for the response (e.g., AWR HTML report).

    Used for sending files to Slack/Teams alongside the formatted message.
    """

    content: bytes | str
    filename: str
    content_type: str = "text/html"
    title: str | None = None
    comment: str | None = None

    def get_content_bytes(self) -> bytes:
        """Get content as bytes."""
        if isinstance(self.content, bytes):
            return self.content
        return self.content.encode("utf-8")


@dataclass
class Section:
    """A section of the response."""

    title: str | None = None
    content: str | None = None
    fields: list[StatusIndicator | MetricValue] | None = None
    list_items: list[ListItem] | None = None
    table: TableData | None = None
    code_block: CodeBlock | None = None
    actions: list[ActionButton] | None = None
    divider_after: bool = False

    @property
    def metrics(self) -> list[StatusIndicator | MetricValue] | None:
        """Alias for fields for backward compatibility."""
        return self.fields

    @metrics.setter
    def metrics(self, value: list[StatusIndicator | MetricValue] | None):
        """Alias for fields for backward compatibility."""
        self.fields = value


@dataclass
class ResponseHeader:
    """Response header with title and context."""

    title: str
    subtitle: str | None = None
    icon: str | None = None  # emoji or icon name
    severity: Severity | None = None
    timestamp: str | None = None
    agent_name: str | None = None


@dataclass
class ResponseFooter:
    """Response footer with metadata."""

    text: str | None = None
    timestamp: str | None = None
    duration_ms: int | None = None
    next_steps: list[str] | None = None
    help_text: str | None = None
    trace_id: str | None = None  # For APM correlation


@dataclass
class StructuredResponse:
    """
    Complete structured response from an agent.

    This is the canonical format that all agents should produce.
    Formatters convert this to channel-specific formats (Slack, Teams, etc.).
    """

    header: ResponseHeader
    sections: list[Section] = field(default_factory=list)
    footer: ResponseFooter | None = None
    raw_data: dict[str, Any] | None = None
    error: str | None = None
    success: bool = True
    attachments: list[FileAttachment] = field(default_factory=list)

    def add_section(
        self,
        title: str | None = None,
        content: str | None = None,
        fields: list[StatusIndicator | MetricValue] | None = None,
        list_items: list[ListItem] | None = None,
        table: TableData | None = None,
        code_block: CodeBlock | None = None,
        actions: list[ActionButton] | None = None,
        divider_after: bool = False,
    ) -> StructuredResponse:
        """Add a section to the response."""
        self.sections.append(
            Section(
                title=title,
                content=content,
                fields=fields,
                list_items=list_items,
                table=table,
                code_block=code_block,
                actions=actions,
                divider_after=divider_after,
            )
        )
        return self

    def add_metrics(
        self, title: str, metrics: list[MetricValue], divider_after: bool = False
    ) -> StructuredResponse:
        """Add a metrics section."""
        return self.add_section(title=title, fields=metrics, divider_after=divider_after)

    def add_status_list(
        self, title: str, items: list[StatusIndicator], divider_after: bool = False
    ) -> StructuredResponse:
        """Add a status list section."""
        return self.add_section(title=title, fields=items, divider_after=divider_after)

    def add_table(
        self, title: str, table: TableData, divider_after: bool = False
    ) -> StructuredResponse:
        """Add a table section."""
        return self.add_section(title=title, table=table, divider_after=divider_after)

    def add_recommendations(
        self, items: list[ListItem], divider_after: bool = False
    ) -> StructuredResponse:
        """Add a recommendations section."""
        return self.add_section(
            title="Recommendations", list_items=items, divider_after=divider_after
        )

    def set_error(self, error: str) -> StructuredResponse:
        """Mark response as error."""
        self.error = error
        self.success = False
        return self

    def add_attachment(self, attachment: FileAttachment) -> StructuredResponse:
        """Add a file attachment to the response.

        Args:
            attachment: FileAttachment to add (e.g., AWR HTML report)

        Returns:
            Self for chaining
        """
        self.attachments.append(attachment)
        return self

    def add_file(
        self,
        content: bytes | str,
        filename: str,
        content_type: str = "text/html",
        title: str | None = None,
        comment: str | None = None,
    ) -> StructuredResponse:
        """Add a file attachment with content.

        Convenience method for adding files without creating FileAttachment.

        Args:
            content: File content (bytes or string)
            filename: Filename for the attachment
            content_type: MIME type
            title: Display title
            comment: Comment/description

        Returns:
            Self for chaining
        """
        self.attachments.append(
            FileAttachment(
                content=content,
                filename=filename,
                content_type=content_type,
                title=title,
                comment=comment,
            )
        )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Formatter
# ─────────────────────────────────────────────────────────────────────────────


class BaseFormatter(ABC):
    """
    Abstract base class for response formatters.

    Subclasses implement format_response() to convert StructuredResponse
    to channel-specific formats (Slack Block Kit, Teams Adaptive Cards, etc.).
    """

    @property
    @abstractmethod
    def format_type(self) -> OutputFormat:
        """Return the output format type."""
        pass

    @abstractmethod
    def format_response(self, response: StructuredResponse) -> str | dict[str, Any]:
        """
        Format a structured response.

        Args:
            response: Structured response to format

        Returns:
            Formatted response (string for text formats, dict for JSON-based formats)
        """
        pass

    @abstractmethod
    def format_error(self, error: str, title: str = "Error") -> str | dict[str, Any]:
        """
        Format an error message.

        Args:
            error: Error message
            title: Error title

        Returns:
            Formatted error response
        """
        pass

    def get_severity_icon(self, severity: Severity) -> str:
        """Get icon for severity level."""
        icons = {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🔵",
            Severity.INFO: "ℹ️",
            Severity.SUCCESS: "✅",
        }
        return icons.get(severity, "•")

    def get_trend_icon(self, trend: TrendDirection | None) -> str:
        """Get icon for trend direction."""
        if trend == TrendDirection.UP:
            return "📈"
        elif trend == TrendDirection.DOWN:
            return "📉"
        elif trend == TrendDirection.STABLE:
            return "➡️"
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Formatter Registry
# ─────────────────────────────────────────────────────────────────────────────


class FormatterRegistry:
    """Registry for response formatters."""

    _formatters: dict[OutputFormat, BaseFormatter] = {}

    @classmethod
    def register(cls, formatter: BaseFormatter) -> None:
        """Register a formatter."""
        cls._formatters[formatter.format_type] = formatter

    @classmethod
    def get(cls, format_type: OutputFormat) -> BaseFormatter | None:
        """Get a formatter by type."""
        return cls._formatters.get(format_type)

    @classmethod
    def format(
        cls, response: StructuredResponse, format_type: OutputFormat
    ) -> str | dict[str, Any]:
        """Format a response using the appropriate formatter."""
        formatter = cls.get(format_type)
        if not formatter:
            raise ValueError(f"No formatter registered for {format_type}")
        return formatter.format_response(response)
