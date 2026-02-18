from __future__ import annotations

from pathlib import Path

from src.agents.catalog import AgentCatalog


def _write_mcp_config(path: Path, include_all: bool = True) -> Path:
    if include_all:
        content = """
servers:
  oci-gateway:
    transport: http
    url: http://gateway.local/mcp
    enabled: true
    domains: [database, security, finops, observability, infrastructure, selectai]
  oci-unified:
    transport: stdio
    command: python
    args: ["-m", "src.mcp.server.main"]
    enabled: true
    domains: [database, security, finops, observability, infrastructure]
  database-observatory:
    transport: stdio
    command: python
    args: ["-m", "src.mcp_server"]
    enabled: true
    domains: [database, observability]
  finopsai:
    transport: stdio
    command: python
    args: ["-m", "finopsai_mcp.server"]
    enabled: true
    domains: [finops, cost]
  oci-mcp-security:
    transport: stdio
    command: python
    args: ["-m", "oci_mcp_security.server"]
    enabled: true
    domains: [security]
  oci-infrastructure:
    transport: stdio
    command: python
    args: ["-m", "mcp_server_oci.server"]
    enabled: true
    domains: [infrastructure]

groups:
  database: [oci-gateway, oci-unified, database-observatory]
  security: [oci-gateway, oci-mcp-security, oci-unified]
  finops: [oci-gateway, finopsai, oci-unified]
  observability: [oci-gateway, database-observatory, oci-unified]
  infrastructure: [oci-gateway, oci-unified, oci-infrastructure]
  selectai: [oci-gateway, database-observatory]
"""
    else:
        content = """
servers:
  oci-gateway:
    transport: http
    url: http://gateway.local/mcp
    enabled: true
    domains: [database, security, finops, observability, infrastructure, selectai]
  oci-unified:
    transport: stdio
    command: python
    args: ["-m", "src.mcp.server.main"]
    enabled: true
    domains: [database, security, finops, observability, infrastructure]
"""
    path.write_text(content)
    return path


def test_capability_alias_lookup(monkeypatch, tmp_path):
    config_file = _write_mcp_config(tmp_path / "mcp_servers_runtime.yaml", include_all=True)
    monkeypatch.setenv("MCP_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("OCI_DEPLOYMENT_MODE", "oci-demo")

    AgentCatalog.reset_instance()
    catalog = AgentCatalog.get_instance()
    catalog.auto_discover()

    roles_for_cost = {a.role for a in catalog.get_by_capability("cost")}
    roles_for_security = {a.role for a in catalog.get_by_capability("security audit")}
    roles_for_nl2sql = {a.role for a in catalog.get_by_capability("text-to-sql")}

    assert "finops-agent" in roles_for_cost
    assert "security-threat-agent" in roles_for_security
    assert "selectai-agent" in roles_for_nl2sql


def test_runtime_profile_oke_enforces_gateway_first(monkeypatch, tmp_path):
    config_file = _write_mcp_config(tmp_path / "mcp_servers_oke.yaml", include_all=False)
    monkeypatch.setenv("MCP_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("OCI_DEPLOYMENT_MODE", "oke")

    AgentCatalog.reset_instance()
    catalog = AgentCatalog.get_instance()
    catalog.auto_discover()

    for agent in catalog.list_all():
        assert agent.mcp_servers[0] == "oci-gateway"


def test_runtime_profile_oci_demo_prefers_specialized_servers(monkeypatch, tmp_path):
    config_file = _write_mcp_config(tmp_path / "mcp_servers_demo.yaml", include_all=True)
    monkeypatch.setenv("MCP_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("OCI_DEPLOYMENT_MODE", "oci-demo")

    AgentCatalog.reset_instance()
    catalog = AgentCatalog.get_instance()
    catalog.auto_discover()

    role_map = {a.role: a for a in catalog.list_all()}

    finops_servers = role_map["finops-agent"].mcp_servers
    security_servers = role_map["security-threat-agent"].mcp_servers
    infra_servers = role_map["infrastructure-agent"].mcp_servers

    assert finops_servers[0] == "oci-gateway"
    assert "finopsai" in finops_servers

    assert security_servers[0] == "oci-gateway"
    assert "oci-mcp-security" in security_servers

    assert infra_servers[0] == "oci-gateway"
    assert "oci-infrastructure" in infra_servers
