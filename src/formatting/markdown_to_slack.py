"""
Markdown to Slack converter for agent responses.

Converts GitHub-flavored markdown to Slack mrkdwn format and Block Kit structures.

Key differences:
- Markdown: **bold**, _italic_, `code`, ```code blocks```
- Slack: *bold*, _italic_, `code`, ```code blocks```
- Markdown tables: | col | col | -> Slack native table blocks

This module enables LLM-generated markdown responses to render properly in Slack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.formatting.base import (
    CodeBlock,
    ListItem,
    ResponseHeader,
    Section,
    Severity,
    StructuredResponse,
    TableData,
    TableRow,
)


@dataclass
class ParsedMarkdown:
    """Result of parsing markdown content."""

    sections: list[Section] = field(default_factory=list)
    title: str | None = None
    has_tables: bool = False
    has_code_blocks: bool = False


class MarkdownToSlackConverter:
    """
    Converts GitHub-flavored markdown to Slack-compatible format.

    Handles:
    - Bold: **text** -> *text*
    - Italic: _text_ stays _text_ (same in Slack)
    - Strikethrough: ~~text~~ -> ~text~
    - Code: `code` stays `code` (same in Slack)
    - Code blocks: ```lang\ncode``` -> preserved
    - Tables: | col | col | -> Slack native table blocks
    - Headers: # H1, ## H2, etc. -> *bold text*
    - Lists: - item, * item, 1. item -> proper mrkdwn lists
    - Links: [text](url) -> <url|text>
    """

    # Emoji mappings for common patterns in agent responses
    SEVERITY_PATTERNS = {
        r"\[CRITICAL\]": ":red_circle: [CRITICAL]",
        r"\[HIGH\]": ":large_orange_circle: [HIGH]",
        r"\[MEDIUM\]": ":large_yellow_circle: [MEDIUM]",
        r"\[LOW\]": ":large_blue_circle: [LOW]",
        r"\[INFO\]": ":information_source: [INFO]",
        r"\[SUCCESS\]": ":white_check_mark: [SUCCESS]",
        r"✅": ":white_check_mark:",
        r"❌": ":x:",
        r"⚠️": ":warning:",
        r"💡": ":bulb:",
        r"📊": ":bar_chart:",
        r"💰": ":moneybag:",
        r"🔍": ":mag:",
    }

    def __init__(self):
        self._table_pattern = re.compile(
            r"(?:^|\n)"  # Start of string or newline
            r"(\|[^\n]+\|)\s*\n"  # Header row
            r"(\|[-:\s|]+\|)\s*\n"  # Separator row (dashes, colons, pipes)
            r"((?:\|[^\n]+\|\s*\n?)+)",  # Data rows
            re.MULTILINE,
        )

    def convert(self, markdown: str, agent_name: str | None = None) -> StructuredResponse:
        """
        Convert markdown text to a StructuredResponse.

        Args:
            markdown: Raw markdown text from agent/LLM
            agent_name: Optional agent name for header context

        Returns:
            StructuredResponse ready for SlackFormatter
        """
        if not markdown:
            return StructuredResponse(
                header=ResponseHeader(title="Response", agent_name=agent_name),
            )

        # Parse markdown into sections
        parsed = self._parse_markdown(markdown)

        # Determine title
        title = parsed.title or "Response"

        response = StructuredResponse(
            header=ResponseHeader(
                title=title,
                agent_name=agent_name,
                severity=Severity.INFO,
            ),
            sections=parsed.sections,
        )

        return response

    def convert_text(self, markdown: str) -> str:
        """
        Convert markdown text to Slack mrkdwn format (text only, no blocks).

        Use this for simple text conversion when you don't need full
        StructuredResponse parsing.

        Args:
            markdown: Markdown text

        Returns:
            Slack mrkdwn formatted text
        """
        return self._convert_inline_formatting(markdown)

    def _parse_markdown(self, markdown: str) -> ParsedMarkdown:
        """Parse markdown into structured sections."""
        result = ParsedMarkdown()

        # Extract title from first H1/H2 header if present
        title_match = re.match(r"^#{1,2}\s+(.+?)(?:\n|$)", markdown)
        if title_match:
            result.title = title_match.group(1).strip()
            markdown = markdown[title_match.end() :].strip()

        # Split into chunks by tables and code blocks
        chunks = self._split_by_special_blocks(markdown)

        for chunk in chunks:
            if chunk["type"] == "table":
                table = self._parse_table(chunk["content"])
                if table:
                    result.sections.append(Section(table=table))
                    result.has_tables = True
            elif chunk["type"] == "code_block":
                code_block = self._parse_code_block(chunk["content"])
                if code_block:
                    result.sections.append(Section(code_block=code_block))
                    result.has_code_blocks = True
            else:
                # Regular text - convert formatting and add as section
                text = self._convert_inline_formatting(chunk["content"])
                if text.strip():
                    # Check if it starts with a header
                    header_match = re.match(r"^(#{1,6})\s+(.+?)(?:\n|$)", text)
                    if header_match:
                        section_title = header_match.group(2).strip()
                        rest = text[header_match.end() :].strip()
                        result.sections.append(
                            Section(title=section_title, content=rest if rest else None)
                        )
                    else:
                        result.sections.append(Section(content=text))

        return result

    def _split_by_special_blocks(self, markdown: str) -> list[dict[str, Any]]:
        """Split markdown into chunks of text, tables, and code blocks."""
        chunks: list[dict[str, Any]] = []
        remaining = markdown

        while remaining:
            # Find next table
            table_match = self._table_pattern.search(remaining)

            # Find next code block
            code_match = re.search(r"```(\w*)\n(.*?)```", remaining, re.DOTALL)

            # Determine which comes first
            table_pos = table_match.start() if table_match else len(remaining)
            code_pos = code_match.start() if code_match else len(remaining)

            if table_pos < code_pos and table_match:
                # Table comes first
                if table_match.start() > 0:
                    chunks.append(
                        {"type": "text", "content": remaining[: table_match.start()]}
                    )
                chunks.append({"type": "table", "content": table_match.group(0)})
                remaining = remaining[table_match.end() :]
            elif code_pos < table_pos and code_match:
                # Code block comes first
                if code_match.start() > 0:
                    chunks.append(
                        {"type": "text", "content": remaining[: code_match.start()]}
                    )
                chunks.append({"type": "code_block", "content": code_match.group(0)})
                remaining = remaining[code_match.end() :]
            else:
                # No more special blocks
                if remaining.strip():
                    chunks.append({"type": "text", "content": remaining})
                break

        return chunks

    def _parse_table(self, table_text: str) -> TableData | None:
        """Parse a markdown table into TableData."""
        lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]

        if len(lines) < 2:
            return None

        # Parse header row
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split("|") if cell.strip()]

        if not headers:
            return None

        # Skip separator row (line with dashes)
        # Find data rows (skip header and separator)
        data_start = 1
        for i, line in enumerate(lines[1:], 1):
            if re.match(r"^[\s|:-]+$", line):
                data_start = i + 1
                break

        # Parse data rows
        rows: list[TableRow] = []
        for line in lines[data_start:]:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                # Detect severity from first cell content
                severity = self._detect_row_severity(cells[0] if cells else "")
                rows.append(TableRow(cells=cells, severity=severity))

        if not rows:
            return None

        return TableData(headers=headers, rows=rows)

    def _detect_row_severity(self, text: str) -> Severity | None:
        """Detect severity from cell content."""
        text_lower = text.lower()
        if "critical" in text_lower or "error" in text_lower:
            return Severity.CRITICAL
        if "high" in text_lower or "warning" in text_lower:
            return Severity.HIGH
        if "medium" in text_lower:
            return Severity.MEDIUM
        if "low" in text_lower:
            return Severity.LOW
        if "success" in text_lower or "ok" in text_lower:
            return Severity.SUCCESS
        return None

    def _parse_code_block(self, code_text: str) -> CodeBlock | None:
        """Parse a fenced code block."""
        match = re.match(r"```(\w*)\n?(.*?)```", code_text, re.DOTALL)
        if match:
            language = match.group(1) or ""
            code = match.group(2).strip()
            return CodeBlock(code=code, language=language)
        return None

    def _convert_inline_formatting(self, text: str) -> str:
        """Convert markdown inline formatting to Slack mrkdwn."""
        result = text

        # Convert bold: **text** -> *text* (Slack uses single asterisk)
        result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)

        # Convert strikethrough: ~~text~~ -> ~text~
        result = re.sub(r"~~(.+?)~~", r"~\1~", result)

        # Convert links: [text](url) -> <url|text>
        result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", result)

        # Convert headers to bold (within text sections)
        # ### Header -> *Header*
        result = re.sub(r"^#{1,6}\s+(.+?)$", r"*\1*", result, flags=re.MULTILINE)

        # Convert horizontal rules: --- or *** or ___ -> ───────────
        result = re.sub(r"^[-*_]{3,}$", "───────────────────────────", result, flags=re.MULTILINE)

        # Add severity emojis where appropriate
        for pattern, replacement in self.SEVERITY_PATTERNS.items():
            result = re.sub(pattern, replacement, result)

        return result.strip()


def convert_markdown_to_slack(markdown: str, agent_name: str | None = None) -> StructuredResponse:
    """
    Convenience function to convert markdown to StructuredResponse.

    Args:
        markdown: Raw markdown text
        agent_name: Optional agent name

    Returns:
        StructuredResponse ready for SlackFormatter
    """
    converter = MarkdownToSlackConverter()
    return converter.convert(markdown, agent_name)


def convert_markdown_text(markdown: str) -> str:
    """
    Convenience function for simple text conversion.

    Args:
        markdown: Markdown text

    Returns:
        Slack mrkdwn text
    """
    converter = MarkdownToSlackConverter()
    return converter.convert_text(markdown)
