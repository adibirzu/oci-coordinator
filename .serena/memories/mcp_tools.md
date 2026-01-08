# MCP Tools Reference

## Server Overview

| Server | Tools | Purpose |
|--------|-------|---------|
| **oci-unified** | 34 | Identity, compute, network, security, cost, discovery |
| **database-observatory** | 50+ | OPSI, SQLcl, Logan for database observability |
| **oci-infrastructure** | 44 | Full OCI SDK wrapper |
| **finopsai** | 33 | Multicloud cost, anomaly detection, rightsizing |

**Tool Timeouts:** Cost: 30s, Discovery: 60s, Standard: 120s

## Domain Prefixes (`src/mcp/catalog.py`)

```python
DOMAIN_PREFIXES = {
    "database": ["oci_database_", "oci_opsi_", "oci_dbmgmt_"],
    "infrastructure": ["oci_compute_", "oci_network_"],
    "finops": ["oci_cost_", "finops_"],
    "security": ["oci_security_"],
}
```

## Database Observatory Tiers (OPSI)

| Tier | Response | Tools |
|------|----------|-------|
| **1 (Cache)** | <100ms | `get_fleet_summary`, `search_databases` |
| **2 (OPSI API)** | 1-5s | `analyze_cpu_usage`, `get_performance_summary` |
| **3 (SQL)** | 5-30s | `execute_sql`, `get_schema_info` |

## Key Tools by Domain

**Identity**
- `oci_list_compartments` - List compartments

**Compute**
- `oci_compute_list_instances` - List instances
- `oci_compute_list_shapes` - List available shapes
- `oci_compute_list_images` - List available images
- `oci_compute_launch_instance` - Launch instance (requires ALLOW_MUTATIONS=true)

**Cost**
- `oci_cost_get_summary` - Get cost summary

**Database**
- `oci_database_execute_sql` - Execute SQL
- `oci_dbmgmt_get_awr_report` - Generate AWR report
- `oci_dbmgmt_list_databases` - List managed databases
- `oci_opsi_get_addm_findings` - Get ADDM findings

## Adding MCP Tools

1. Add tool function in `src/mcp/server/tools/{domain}.py`
2. Register via `register_{domain}_tools(mcp)` in `server/main.py`
3. Assign tool tier in `src/mcp/catalog.py` TOOL_TIERS
