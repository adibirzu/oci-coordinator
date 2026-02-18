import json

import pytest

pytest.importorskip("langchain_core")

from src.mcp.catalog import TOOL_ALIASES, TOOL_TIERS, ToolCatalog
from src.mcp.client import ToolDefinition
from src.mcp.validation import validate_server_manifests


class FakeConfig:
    def __init__(self) -> None:
        self.tool_timeouts: dict[str, int] = {}


class FakeClient:
    def __init__(self, raw_manifest: str | None, has_manifest: bool = True) -> None:
        self.connected = True
        self.resources = {"server://manifest": {"uri": "server://manifest"}} if has_manifest else {}
        self._raw_manifest = raw_manifest
        self.config = FakeConfig()

    async def read_resource(self, uri: str) -> str | None:
        return self._raw_manifest


class FakeRegistry:
    def __init__(
        self,
        tools: dict[str, ToolDefinition],
        clients: dict[str, FakeClient],
    ) -> None:
        self._tools = tools
        self._clients = clients

    def get_all_tools(self) -> dict[str, ToolDefinition]:
        return self._tools

    def list_connected(self) -> list[str]:
        return list(self._clients.keys())

    def get_client(self, server_id: str) -> FakeClient | None:
        return self._clients.get(server_id)


class FakeRegistryForValidation:
    def __init__(self, clients: dict[str, FakeClient]) -> None:
        self._clients = clients

    def list_servers(self) -> list[str]:
        return list(self._clients.keys())

    def get_client(self, server_id: str) -> FakeClient | None:
        return self._clients.get(server_id)


def _build_manifest(tool_name: str, alias: str) -> dict:
    return {
        "schema_version": "1.0",
        "server_id": "test-server",
        "server_name": "Test MCP Server",
        "server_version": "0.1.0",
        "generated_at": "2024-01-01T00:00:00Z",
        "domains": [{"name": "test", "tool_count": 1, "skill_count": 0}],
        "tools": [
            {
                "name": tool_name,
                "description": "List widgets",
                "domain": "test",
                "tier": 2,
                "risk": "low",
                "read_only": True,
                "idempotent": True,
                "mutates": False,
                "requires_confirmation": False,
                "latency_ms": 500,
                "cache_ttl_seconds": 120,
                "aliases": [alias],
                "timeouts": {"default_seconds": 45},
            }
        ],
        "skills": [],
        "policies": {},
    }


@pytest.mark.asyncio
async def test_manifest_overrides_apply_metadata_and_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_USE_MANIFEST_OVERRIDES", "true")
    original_aliases = dict(TOOL_ALIASES)
    original_tiers = dict(TOOL_TIERS)
    tool_name = "oci_test_list_widgets"
    alias = "legacy_test_widgets"

    try:
        tools = {
            tool_name: ToolDefinition(
                name=tool_name,
                description="List widgets",
                input_schema={},
                server_id="test-server",
            )
        }
        manifest = _build_manifest(tool_name, alias)
        client = FakeClient(json.dumps(manifest))
        registry = FakeRegistry(tools, {"test-server": client})

        catalog = ToolCatalog(registry)
        await catalog.refresh()

        metadata = catalog.get_tool_metadata(tool_name)
        assert metadata is not None
        assert metadata.cache_ttl_seconds == 120
        assert metadata.read_only is True
        assert TOOL_ALIASES.get(alias) == tool_name
        assert TOOL_TIERS[tool_name].tier == 2
        assert client.config.tool_timeouts[tool_name] == 45
    finally:
        TOOL_ALIASES.clear()
        TOOL_ALIASES.update(original_aliases)
        TOOL_TIERS.clear()
        TOOL_TIERS.update(original_tiers)


@pytest.mark.asyncio
async def test_manifest_overrides_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_USE_MANIFEST_OVERRIDES", "false")
    original_aliases = dict(TOOL_ALIASES)
    original_tiers = dict(TOOL_TIERS)
    tool_name = "oci_test_list_widgets_disabled"
    alias = "legacy_test_widgets_disabled"

    try:
        tools = {
            tool_name: ToolDefinition(
                name=tool_name,
                description="List widgets",
                input_schema={},
                server_id="test-server",
            )
        }
        manifest = _build_manifest(tool_name, alias)
        client = FakeClient(json.dumps(manifest))
        registry = FakeRegistry(tools, {"test-server": client})

        catalog = ToolCatalog(registry)
        await catalog.refresh()

        assert catalog.get_tool_metadata(tool_name) is None
        assert TOOL_ALIASES.get(alias) is None
        assert TOOL_TIERS.get(tool_name) == original_tiers.get(tool_name)
        assert client.config.tool_timeouts == {}
    finally:
        TOOL_ALIASES.clear()
        TOOL_ALIASES.update(original_aliases)
        TOOL_TIERS.clear()
        TOOL_TIERS.update(original_tiers)


@pytest.mark.asyncio
async def test_validate_server_manifests_reports_issues() -> None:
    valid_manifest = _build_manifest("oci_test_list_widgets", "legacy_test_widgets")
    clients = {
        "valid-server": FakeClient(json.dumps(valid_manifest)),
        "missing-server": FakeClient(json.dumps(valid_manifest), has_manifest=False),
        "invalid-server": FakeClient("not-json"),
    }
    registry = FakeRegistryForValidation(clients)

    results = await validate_server_manifests(registry)

    assert "missing-server" in results["missing_manifest"]
    assert any(
        entry["server_id"] == "invalid-server" for entry in results["invalid_manifest"]
    )
    assert "valid-server" not in results["missing_manifest"]
    assert results["checked"] == 3


@pytest.mark.asyncio
async def test_gateway_namespaced_tools_register_canonical_aliases() -> None:
    original_aliases = dict(TOOL_ALIASES)
    try:
        tool_name = "oci_oci_compute_list_instances"
        tools = {
            tool_name: ToolDefinition(
                name=tool_name,
                description="List instances via gateway backend namespace",
                input_schema={},
                server_id="oci-gateway",
            )
        }
        registry = FakeRegistry(tools, {})
        catalog = ToolCatalog(registry)
        await catalog.refresh()

        resolved = catalog.get_tool("oci_compute_list_instances")
        assert resolved is not None
        assert resolved.name == tool_name
    finally:
        TOOL_ALIASES.clear()
        TOOL_ALIASES.update(original_aliases)
