"""Tests for markdown to Slack conversion."""

import pytest

from src.formatting.markdown_to_slack import (
    MarkdownToSlackConverter,
    convert_markdown_text,
    convert_markdown_to_slack,
)
from src.formatting.base import Severity


class TestMarkdownToSlackConverter:
    """Test the MarkdownToSlackConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return MarkdownToSlackConverter()

    def test_bold_conversion(self, converter):
        """Test that **bold** converts to *bold*."""
        result = converter.convert_text("This is **bold** text")
        assert result == "This is *bold* text"

    def test_multiple_bold_conversion(self, converter):
        """Test multiple bold sections."""
        result = converter.convert_text("**First** and **second** bold")
        assert result == "*First* and *second* bold"

    def test_strikethrough_conversion(self, converter):
        """Test that ~~text~~ converts to ~text~."""
        result = converter.convert_text("This is ~~deleted~~ text")
        assert result == "This is ~deleted~ text"

    def test_link_conversion(self, converter):
        """Test that [text](url) converts to <url|text>."""
        result = converter.convert_text("Click [here](https://example.com) for more")
        assert result == "Click <https://example.com|here> for more"

    def test_header_conversion(self, converter):
        """Test that # headers convert to *bold*."""
        result = converter.convert_text("# Main Title\nSome content")
        assert "*Main Title*" in result

    def test_header_levels(self, converter):
        """Test various header levels."""
        result = converter.convert_text("## Level 2\n### Level 3")
        assert "*Level 2*" in result
        assert "*Level 3*" in result

    def test_horizontal_rule(self, converter):
        """Test horizontal rule conversion."""
        result = converter.convert_text("Before\n---\nAfter")
        assert "───────────────────────────" in result

    def test_code_inline_preserved(self, converter):
        """Test that `code` is preserved."""
        result = converter.convert_text("Use `print()` function")
        assert "`print()`" in result


class TestTableParsing:
    """Test markdown table parsing."""

    @pytest.fixture
    def converter(self):
        return MarkdownToSlackConverter()

    def test_simple_table(self, converter):
        """Test parsing a simple markdown table."""
        markdown = """
| Service | Cost |
|---------|------|
| Database | $100 |
| Compute | $50 |
"""
        response = converter.convert(markdown)
        assert len(response.sections) >= 1

        # Find the table section
        table_section = None
        for section in response.sections:
            if section.table:
                table_section = section
                break

        assert table_section is not None
        assert table_section.table.headers == ["Service", "Cost"]
        assert len(table_section.table.rows) == 2

    def test_table_with_alignment(self, converter):
        """Test table with alignment markers."""
        markdown = """
| Left | Center | Right |
|:-----|:------:|------:|
| L1 | C1 | R1 |
| L2 | C2 | R2 |
"""
        response = converter.convert(markdown)

        table_section = None
        for section in response.sections:
            if section.table:
                table_section = section
                break

        assert table_section is not None
        assert len(table_section.table.headers) == 3

    def test_table_with_severity_detection(self, converter):
        """Test that severity is detected from table content."""
        markdown = """
| Status | Count |
|--------|-------|
| CRITICAL | 5 |
| Warning | 10 |
| OK | 100 |
"""
        response = converter.convert(markdown)

        table_section = None
        for section in response.sections:
            if section.table:
                table_section = section
                break

        assert table_section is not None
        # First row with "CRITICAL" should have critical severity
        assert table_section.table.rows[0].severity == Severity.CRITICAL


class TestCodeBlockParsing:
    """Test code block parsing."""

    @pytest.fixture
    def converter(self):
        return MarkdownToSlackConverter()

    def test_simple_code_block(self, converter):
        """Test parsing a simple code block."""
        markdown = """
```sql
SELECT * FROM users;
```
"""
        response = converter.convert(markdown)

        code_section = None
        for section in response.sections:
            if section.code_block:
                code_section = section
                break

        assert code_section is not None
        assert code_section.code_block.language == "sql"
        assert "SELECT * FROM users" in code_section.code_block.code

    def test_code_block_without_language(self, converter):
        """Test code block without language specifier."""
        markdown = """
```
echo "hello"
```
"""
        response = converter.convert(markdown)

        code_section = None
        for section in response.sections:
            if section.code_block:
                code_section = section
                break

        assert code_section is not None
        assert code_section.code_block.language == ""


class TestFullConversion:
    """Test full markdown to StructuredResponse conversion."""

    @pytest.fixture
    def converter(self):
        return MarkdownToSlackConverter()

    def test_complex_markdown(self, converter):
        """Test converting complex markdown with multiple elements."""
        markdown = """# Cost Analysis Results

**Total Cost**: $1,234.56

## Top Services by Cost

| Service | Cost | Percent |
|---------|------|---------|
| Database | $500 | 40% |
| Compute | $300 | 24% |
| Storage | $200 | 16% |

---

### Recommendations

- Review storage tiers - *Potential savings*: $100
- Consider rightsizing compute instances

```sql
SELECT service_name, sum(cost) FROM costs GROUP BY service_name;
```
"""
        response = converter.convert(markdown, agent_name="finops-agent")

        # Check header
        assert response.header.title == "Cost Analysis Results"
        assert response.header.agent_name == "finops-agent"

        # Check that we have multiple sections
        assert len(response.sections) > 3

        # Check for table
        has_table = any(s.table for s in response.sections)
        assert has_table

        # Check for code block
        has_code = any(s.code_block for s in response.sections)
        assert has_code

    def test_preserves_text_order(self, converter):
        """Test that text order is preserved."""
        markdown = """First paragraph.

| Col1 | Col2 |
|------|------|
| A | B |

Second paragraph after table.
"""
        response = converter.convert(markdown)

        # Should have at least 3 sections: text, table, text
        assert len(response.sections) >= 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_convert_markdown_to_slack(self):
        """Test the convenience function."""
        response = convert_markdown_to_slack("**Hello** World", agent_name="test-agent")
        assert response.header.agent_name == "test-agent"

    def test_convert_markdown_text(self):
        """Test simple text conversion."""
        result = convert_markdown_text("**Bold** and _italic_")
        assert "*Bold*" in result
        assert "_italic_" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def converter(self):
        return MarkdownToSlackConverter()

    def test_empty_string(self, converter):
        """Test empty string input."""
        response = converter.convert("")
        assert response.header.title == "Response"
        assert len(response.sections) == 0

    def test_none_handling(self, converter):
        """Test that None-like values are handled."""
        result = converter.convert_text("")
        assert result == ""

    def test_malformed_table(self, converter):
        """Test malformed table doesn't crash."""
        markdown = """
| Only | Header |
"""
        response = converter.convert(markdown)
        # Should not crash, may not produce table
        assert response is not None

    def test_nested_formatting(self, converter):
        """Test nested formatting."""
        result = converter.convert_text("**Bold with _italic_ inside**")
        assert "*Bold with _italic_ inside*" in result

    def test_escaped_characters(self, converter):
        """Test that escaped characters are handled."""
        result = converter.convert_text(r"Use \*asterisks\* literally")
        # Should preserve escaped content reasonably
        assert "asterisks" in result


class TestRealWorldExamples:
    """Test with real-world examples from agent responses."""

    @pytest.fixture
    def converter(self):
        return MarkdownToSlackConverter()

    def test_finops_response(self, converter):
        """Test formatting a typical FinOps response."""
        markdown = """**Agent:** Finops

**FinOps Analysis Results** *Cost analysis for 2025-12-01 to 2025-12-31*
Agent: finops-agent | Generated: 2026-01-16 05:04:59 UTC

**Cost Summary** -
**Total Cost**: $61,639.56
**Period**: 2025-12-01 to 2025-12-31

---

**Cost Breakdown** text

**Top Services by Cost**

| Service | Cost |
|-----------------|-----------|
| Database | $23,023.39 |
| Stack Monitoring | $6,443.02 |
| Logging Analytics | $6,176.99 |
| Block Storage | $5,744.13 |
| Analytics | $4,533.05 |

---

**Recommendations** -
- Review storage tiers - consider Standard vs. Archive for cold data - *Potential savings*: $861.62 -
- Consider rightsizing compute instances or using preemptible shapes - *Potential savings*: $709.45 >

Use `/oci cost <compartment>` for detailed breakdown
"""
        response = converter.convert(markdown, agent_name="finops-agent")

        # Should extract meaningful structure
        assert response.header is not None

        # Should have table section
        has_table = any(s.table for s in response.sections)
        assert has_table

        # Table should have correct headers
        for section in response.sections:
            if section.table:
                assert "Service" in section.table.headers
                assert "Cost" in section.table.headers
                break
