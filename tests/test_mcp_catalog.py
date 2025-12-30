
import pytest
from src.mcp.catalog import ToolCatalog

def test_tool_catalog_initialization():
    """Verify ToolCatalog initialization."""
    catalog = ToolCatalog()
    assert catalog is not None
    assert catalog.tools == {}

@pytest.mark.asyncio
async def test_tool_catalog_refresh():
    """Verify catalog refresh logic."""
    catalog = ToolCatalog()
    # Mocking would be needed here
    await catalog.refresh()
    # For now just checking it doesn't crash on empty
