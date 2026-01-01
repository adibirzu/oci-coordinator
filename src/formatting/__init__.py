"""
Response formatting module for OCI Coordinator.

Provides:
- Structured response types for consistent agent output
- Formatters for different output channels (Slack, Markdown, Teams, etc.)
- Formatter registry for dynamic format selection

Usage:
    from src.formatting import (
        StructuredResponse,
        ResponseHeader,
        Section,
        MetricValue,
        Severity,
        OutputFormat,
        FormatterRegistry,
    )

    # Create structured response
    response = StructuredResponse(
        header=ResponseHeader(title="Analysis Complete", severity=Severity.SUCCESS),
    )
    response.add_metrics("Performance", [
        MetricValue(label="CPU", value=45.2, unit="%", threshold=80),
        MetricValue(label="Memory", value=72.1, unit="%", severity=Severity.MEDIUM),
    ])

    # Format for Slack
    slack_message = FormatterRegistry.format(response, OutputFormat.SLACK)

    # Format for Markdown
    markdown_text = FormatterRegistry.format(response, OutputFormat.MARKDOWN)
"""

from src.formatting.base import (
    ActionButton,
    BaseFormatter,
    CodeBlock,
    FileAttachment,
    FormatterRegistry,
    ListItem,
    MetricValue,
    OutputFormat,
    ResponseFooter,
    ResponseHeader,
    Section,
    Severity,
    StatusIndicator,
    StructuredResponse,
    TableData,
    TableRow,
    TrendDirection,
)

# Import formatters to register them
from src.formatting.markdown import MarkdownFormatter
from src.formatting.slack import SlackFormatter

__all__ = [
    # Core types
    "OutputFormat",
    "Severity",
    "TrendDirection",
    # Response components
    "StatusIndicator",
    "MetricValue",
    "TableRow",
    "TableData",
    "ListItem",
    "CodeBlock",
    "ActionButton",
    "FileAttachment",
    "Section",
    "ResponseHeader",
    "ResponseFooter",
    "StructuredResponse",
    # Formatters
    "BaseFormatter",
    "FormatterRegistry",
    "MarkdownFormatter",
    "SlackFormatter",
]
