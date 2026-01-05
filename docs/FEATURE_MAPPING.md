# Feature Mapping Matrix

## Overview

This document provides a comprehensive mapping of all features, MCP servers, tools, agents, and their interconnections in the OCI AI Agent Coordinator system.

**Last Updated**: 2026-01-03
**Test Status**: 21/41 tools tested (new DB Mgmt/OPSI tools pending)

---

## MCP Server Inventory

### Server Summary

| Server | Tools | Status | Transport | Primary Use |
|--------|-------|--------|-----------|-------------|
| **oci-unified** | 51 | Enabled | stdio | Core OCI operations + DB Mgmt + OPSI |
| **database-observatory** | 50+ | Enabled | stdio | Database/SQLcl/Logan |
| **oci-infrastructure** | 44 | Enabled | stdio | Full OCI SDK wrapper |
| **finopsai-mcp** | 33 | Enabled | stdio | Multicloud FinOps |

**Total Available Tools**: 178+

### New in v1.3 (2026-01-03)
- **DB Management Tools (10)**: AWR reports, Top SQL, Wait Events, SQL Plan Baselines, Fleet Health
- **OPSI Tools (10)**: Database Insights, ADDM, Capacity Planning, SQL Statistics
- **Workflows (20+)**: Pre-built workflows for all new capabilities

---

## Tool-to-Feature Mapping

### Identity Domain

| Tool | MCP Server | Agent | Workflow | Test Status |
|------|------------|-------|----------|-------------|
| `oci_list_compartments` | oci-unified | Coordinator | `list_compartments` | ✅ Passed |
| `oci_get_compartment` | oci-unified | Coordinator | - | ✅ Passed |
| `oci_search_compartments` | oci-unified | Coordinator | - | ✅ Passed |
| `oci_get_tenancy` | oci-unified | Coordinator | `get_tenancy_info` | ✅ Passed |
| `oci_list_regions` | oci-unified | Coordinator | `list_regions` | ✅ Passed |

### Compute Domain

| Tool | MCP Server | Agent | Workflow | Test Status |
|------|------------|-------|----------|-------------|
| `oci_compute_list_instances` | oci-unified | Infrastructure | `list_instances` | ✅ Passed |
| `oci_compute_get_instance` | oci-unified | Infrastructure | `get_instance` | - |
| `oci_compute_find_instance` | oci-unified | Infrastructure | - | ✅ Passed |
| `oci_compute_start_instance` | oci-unified | Infrastructure | - | - |
| `oci_compute_stop_instance` | oci-unified | Infrastructure | - | - |
| `oci_compute_restart_instance` | oci-unified | Infrastructure | - | - |
| `oci_compute_start_by_name` | oci-unified | Infrastructure | - | - |
| `oci_compute_stop_by_name` | oci-unified | Infrastructure | - | - |
| `oci_compute_restart_by_name` | oci-unified | Infrastructure | - | - |

### Network Domain

| Tool | MCP Server | Agent | Workflow | Test Status |
|------|------------|-------|----------|-------------|
| `oci_network_list_vcns` | oci-unified | Infrastructure | `list_vcns` | ✅ Passed |
| `oci_network_list_subnets` | oci-unified | Infrastructure | `list_subnets` | ✅ Passed |
| `oci_network_list_security_lists` | oci-unified | Infrastructure | - | ✅ Passed |

### Cost Domain

| Tool | MCP Server | Agent | Workflow | Test Status |
|------|------------|-------|----------|-------------|
| `oci_cost_get_summary` | oci-unified | FinOps | `cost_summary` | ✅ Passed |

**Note**: Cost tool has 30-second timeout due to OCI Usage API latency.

### Security Domain

| Tool | MCP Server | Agent | Workflow | Test Status |
|------|------------|-------|----------|-------------|
| `oci_security_list_users` | oci-unified | Security | - | ✅ Passed |

### DB Management Domain

| Tool | MCP Server | Agent | Workflow | Test Status |
|------|------------|-------|----------|-------------|
| `oci_dbmgmt_list_databases` | oci-unified | DB Troubleshoot | `managed_databases` | - |
| `oci_dbmgmt_search_databases` | oci-unified | DB Troubleshoot | - | - |
| `oci_dbmgmt_get_database` | oci-unified | DB Troubleshoot | - | - |
| `oci_dbmgmt_get_awr_report` | oci-unified | DB Troubleshoot | `awr_report` | - |
| `oci_dbmgmt_get_metrics` | oci-unified | DB Troubleshoot | - | - |
| `oci_dbmgmt_get_top_sql` | oci-unified | DB Troubleshoot | `top_sql` | - |
| `oci_dbmgmt_get_wait_events` | oci-unified | DB Troubleshoot | `wait_events` | - |
| `oci_dbmgmt_list_sql_plan_baselines` | oci-unified | DB Troubleshoot | `sql_plan_baselines` | - |
| `oci_dbmgmt_get_fleet_health` | oci-unified | DB Troubleshoot | `db_fleet_health` | - |
| `oci_dbmgmt_get_sql_report` | oci-unified | DB Troubleshoot | - | - |

### Operations Insights (OPSI) Domain

| Tool | MCP Server | Agent | Workflow | Test Status |
|------|------------|-------|----------|-------------|
| `oci_opsi_list_database_insights` | oci-unified | DB Troubleshoot | `database_insights` | - |
| `oci_opsi_get_database_insight` | oci-unified | DB Troubleshoot | - | - |
| `oci_opsi_summarize_resource_stats` | oci-unified | DB Troubleshoot | `opsi_utilization` | - |
| `oci_opsi_summarize_sql_insights` | oci-unified | DB Troubleshoot | `sql_insights` | - |
| `oci_opsi_summarize_sql_statistics` | oci-unified | DB Troubleshoot | `sql_statistics` | - |
| `oci_opsi_get_addm_findings` | oci-unified | DB Troubleshoot | `addm_findings` | - |
| `oci_opsi_get_addm_recommendations` | oci-unified | DB Troubleshoot | `addm_recommendations` | - |
| `oci_opsi_get_capacity_trend` | oci-unified | DB Troubleshoot | `capacity_trend` | - |
| `oci_opsi_get_capacity_forecast` | oci-unified | DB Troubleshoot | `capacity_forecast` | - |
| `oci_opsi_list_awr_hubs` | oci-unified | DB Troubleshoot | - | - |

### Discovery Domain

| Tool | MCP Server | Agent | Workflow | Test Status |
|------|------------|-------|----------|-------------|
| `oci_discovery_cache_status` | oci-unified | Coordinator | - | ✅ Passed |
| `oci_discovery_summary` | oci-unified | Coordinator | `discovery_summary` | ✅ Passed |
| `oci_discovery_search` | oci-unified | Coordinator | `search_resources` | ✅ Passed |
| `oci_discovery_run` | oci-unified | Coordinator | - | - |
| `oci_discovery_refresh` | oci-unified | Coordinator | - | - |

### Meta Tools

| Tool | MCP Server | Purpose | Test Status |
|------|------------|---------|-------------|
| `search_capabilities` | oci-unified | Domain/tool discovery | ✅ Passed (3x) |

---

## Agent-to-Tool Mapping

### Coordinator Agent
**Role**: Master orchestrator for intent classification and routing

**Direct Tools**:
- `oci_list_compartments`
- `oci_get_compartment`
- `oci_search_compartments`
- `oci_get_tenancy`
- `oci_list_regions`
- `search_capabilities`
- `oci_discovery_*`

**Workflows Owned**: 35+ pre-built workflows (100+ intent aliases)

### Database Troubleshoot Agent
**Role**: Multi-database observability and troubleshooting

**Primary Tools (DB Management)**:
| Tool | MCP Server | Tier |
|------|------------|------|
| `oci_dbmgmt_list_databases` | oci-unified | 1 |
| `oci_dbmgmt_search_databases` | oci-unified | 1 |
| `oci_dbmgmt_get_database` | oci-unified | 1 |
| `oci_dbmgmt_get_awr_report` | oci-unified | 2 |
| `oci_dbmgmt_get_top_sql` | oci-unified | 2 |
| `oci_dbmgmt_get_wait_events` | oci-unified | 2 |
| `oci_dbmgmt_list_sql_plan_baselines` | oci-unified | 2 |
| `oci_dbmgmt_get_fleet_health` | oci-unified | 2 |
| `oci_dbmgmt_get_sql_report` | oci-unified | 2 |

**Primary Tools (Operations Insights)**:
| Tool | MCP Server | Tier |
|------|------------|------|
| `oci_opsi_list_database_insights` | oci-unified | 1 |
| `oci_opsi_get_database_insight` | oci-unified | 1 |
| `oci_opsi_summarize_resource_stats` | oci-unified | 2 |
| `oci_opsi_summarize_sql_insights` | oci-unified | 2 |
| `oci_opsi_summarize_sql_statistics` | oci-unified | 2 |
| `oci_opsi_get_addm_findings` | oci-unified | 2 |
| `oci_opsi_get_addm_recommendations` | oci-unified | 2 |
| `oci_opsi_get_capacity_trend` | oci-unified | 2 |
| `oci_opsi_get_capacity_forecast` | oci-unified | 2 |

**Legacy Tools (database-observatory)**:
| Tool | MCP Server | Tier |
|------|------------|------|
| `oci_opsi_get_fleet_summary` | database-observatory | 1 |
| `oci_opsi_analyze_cpu` | database-observatory | 2 |
| `oci_opsi_analyze_memory` | database-observatory | 2 |
| `oci_database_execute_sql` | database-observatory | 3 |

**Skills**:
- `db_rca_workflow` (7 steps)
- `db_health_check_workflow` (3 steps)
- `db_sql_analysis_workflow` (5 steps)

**Workflows**:
- `db_fleet_health` - Fleet-wide health summary
- `top_sql` - Top SQL by CPU
- `wait_events` - AWR wait events
- `awr_report` - Generate AWR/ASH report
- `sql_plan_baselines` - SQL Plan Baselines
- `database_insights` - OPSI database list
- `addm_findings` - ADDM diagnostic findings
- `addm_recommendations` - Optimization suggestions
- `capacity_forecast` - Usage projection
- `db_performance_overview` - Comprehensive health check

### Log Analytics Agent
**Role**: Log search, pattern detection, cross-service correlation

**Primary Tools**:
| Tool | MCP Server | Tier |
|------|------------|------|
| `oci_logan_search` | database-observatory | 2 |
| `oci_observability_query_logs` | oci-unified | 3 |
| `execute_logan_query` | database-observatory | 2 |

### Security Threat Agent
**Role**: Threat hunting, MITRE ATT&CK mapping

**Primary Tools**:
| Tool | MCP Server | Tier |
|------|------------|------|
| `oci_security_list_users` | oci-unified | 2 |
| `oci_security_list_policies` | oci-unified | 2 |
| `list_cloud_guard_problems` | oci-infrastructure | 2 |

### FinOps Agent
**Role**: Cost analysis, anomaly detection, optimization

**Primary Tools**:
| Tool | MCP Server | Tier | Timeout |
|------|------------|------|---------|
| `oci_cost_get_summary` | oci-unified | 2 | 30s |

**Features**:
- Cost breakdown by service
- High-concentration anomaly detection
- Rightsizing recommendations (heuristics-based)

### Infrastructure Agent
**Role**: Compute, network, storage lifecycle management

**Primary Tools**:
| Tool | MCP Server | Tier |
|------|------------|------|
| `oci_compute_list_instances` | oci-unified | 2 |
| `oci_compute_get_instance` | oci-unified | 2 |
| `oci_compute_find_instance` | oci-unified | 2 |
| `oci_compute_start_instance` | oci-unified | 4 |
| `oci_compute_stop_instance` | oci-unified | 4 |
| `oci_network_list_vcns` | oci-unified | 2 |
| `oci_network_list_subnets` | oci-unified | 2 |

---

## Workflow-to-Tool Mapping

### Deterministic Workflows (16 total)

| Workflow | Primary Tool | Intent Aliases |
|----------|--------------|----------------|
| `list_compartments` | `oci_list_compartments` | show_compartments, get_compartments |
| `get_tenancy_info` | `oci_get_tenancy` | tenancy_info, whoami |
| `list_regions` | `oci_list_regions` | show_regions, available_regions |
| `list_instances` | `oci_compute_list_instances` | show_instances, get_vms |
| `get_instance` | `oci_compute_get_instance` | describe_instance, instance_details |
| `list_vcns` | `oci_network_list_vcns` | show_networks, get_vcns |
| `list_subnets` | `oci_network_list_subnets` | get_subnets, show_subnets |
| `cost_summary` | `oci_cost_get_summary` | get_costs, tenancy_costs, how_much_spent, monthly_cost, spending, show_spending |
| `discovery_summary` | `oci_discovery_summary` | resource_summary, what_resources |
| `search_resources` | `oci_discovery_search` | find_resource, search_oci |
| `search_capabilities` | `search_capabilities` | capabilities, what_can_you_do |
| `help` | N/A | help_me, how_to |
| `db_health_check` | `oci_opsi_get_fleet_summary` | database_status, db_health |
| `fleet_summary` | `oci_opsi_get_fleet_summary` | opsi_summary, db_fleet |
| `security_overview` | `oci_security_list_users` | security_status, threats |
| `recent_errors` | `oci_observability_query_logs` | show_errors, log_errors |

---

## Tool Tier Classification

### Tier 1: Instant (<100ms, no risk)
- `search_capabilities`
- `oci_discovery_cache_status`
- `oci_opsi_get_fleet_summary` (cached)

### Tier 2: Fast Reads (<1s, no risk)
- `oci_list_compartments`
- `oci_get_compartment`
- `oci_search_compartments`
- `oci_get_tenancy`
- `oci_list_regions`
- `oci_compute_list_instances`
- `oci_compute_get_instance`
- `oci_compute_find_instance`
- `oci_network_list_vcns`
- `oci_network_list_subnets`
- `oci_network_list_security_lists`
- `oci_security_list_users`
- `oci_discovery_summary`
- `oci_discovery_search`

### Tier 3: Moderate Operations (<30s, low risk)
- `oci_cost_get_summary` (30s timeout)
- `oci_database_execute_sql`
- `oci_observability_query_logs`
- `oci_discovery_run`

### Tier 4: Mutation (variable, confirmation required)
- `oci_compute_start_instance`
- `oci_compute_stop_instance`
- `oci_compute_restart_instance`
- `oci_compute_start_by_name`
- `oci_compute_stop_by_name`
- `oci_compute_restart_by_name`

---

## Domain Prefix Mapping

For dynamic tool discovery by agents:

```python
DOMAIN_PREFIXES = {
    "database": ["oci_database_", "oci_opsi_"],
    "infrastructure": ["oci_compute_", "oci_network_", "oci_list_"],
    "finops": ["oci_cost_"],
    "security": ["oci_security_"],
    "observability": ["oci_observability_", "oci_logan_"],
    "identity": ["oci_list_compartments", "oci_get_tenancy", "oci_list_regions"],
    "discovery": ["oci_discovery_"],
}
```

---

## Tool Aliases (Backward Compatibility)

Legacy tool names automatically resolve to standard names:

```python
TOOL_ALIASES = {
    # database-observatory legacy names
    "execute_sql": "oci_database_execute_sql",
    "get_fleet_summary": "oci_opsi_get_fleet_summary",
    "analyze_cpu_usage": "oci_opsi_analyze_cpu",
    "analyze_memory_usage": "oci_opsi_analyze_memory",
    "get_blocking_sessions": "oci_opsi_get_blocking_sessions",
    "list_autonomous_databases": "oci_database_list_autonomous",
}
```

---

## Channel-to-Feature Mapping

### Slack Integration

| Feature | Implementation | Status |
|---------|----------------|--------|
| 3-second ack | `slack.py` | ✅ |
| Thread-based memory | `conversation.py` | ✅ |
| Troubleshooting catalog | `slack_catalog.py` | ✅ |
| Table block formatting | `SlackFormatter` | ✅ |
| Follow-up suggestions | `slack_catalog.py` | ✅ |
| OAuth login button | `oca_callback_server.py` | ✅ |

### API Server

| Endpoint | Feature | Status |
|----------|---------|--------|
| `/chat` | LangGraph coordinator | ✅ |
| `/tools` | Tool listing/discovery | ✅ |
| `/tools/execute` | Direct tool execution | ✅ |
| `/agents` | Agent status | ✅ |
| `/mcp/servers` | MCP server health | ✅ |
| `/health` | System health | ✅ |

---

## Test Compartment Reference

**Test Compartment**: `adrian_birzu`
**OCID**: `ocid1.compartment.oc1..aaaaaaaagy3yddkkampnhj3cqm5ar7w2p7tuq5twbojyycvol6wugfav3ckq`

All 21 tests passed against this compartment covering:
- Identity tools (6 tests)
- Compute tools (2 tests)
- Network tools (3 tests)
- Cost tools (3 tests)
- Security tools (1 test)
- Discovery tools (3 tests)
- Meta tools (3 tests)

---

## Architecture Flow

```
User Request
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                    INPUT CHANNELS                               │
│  Slack │ API │ (Teams - planned)                               │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│                    COORDINATOR (LangGraph)                      │
│  Intent Classification → Routing Decision                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Confidence ≥ 0.80 → Workflow (16 pre-built)              │   │
│  │ Confidence ≥ 0.60 + 2+ domains → Parallel Orchestrator   │   │
│  │ Confidence ≥ 0.60 → Single Agent                         │   │
│  │ Confidence < 0.30 → Escalate/Clarify                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────┬────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ DB Troublesh │ │   FinOps     │ │   Security   │
│    Agent     │ │    Agent     │ │    Agent     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│                    TOOL CATALOG                                 │
│  158+ tools across 4 MCP servers                               │
│  Tier classification │ Aliases │ Domain prefixes               │
└───────────────────────┬────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ oci-unified  │ │ database-obs │ │ oci-infra    │ │ finopsai     │
│  (31 tools)  │ │  (50+ tools) │ │  (44 tools)  │ │  (33 tools)  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │               │               │               │
        └───────────────┼───────────────┴───────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│                    OCI APIs (via OCI SDK)                       │
└────────────────────────────────────────────────────────────────┘
```

---

## Deployment Components

### Production (Cloud Run / Cloud Shell)

| Component | Type | Purpose |
|-----------|------|---------|
| Coordinator | Container | Main orchestration |
| MCP Servers | Sidecar/Process | Tool execution |
| Redis | Service | State/cache |
| OCI APM | Managed | Tracing |
| OCI Logging | Managed | Structured logs |

### GitHub Actions Workflow

```yaml
# .github/workflows/build-deploy.yaml
# Builds and deploys to OCI Cloud Run or Cloud Shell
```

---

*Generated from comprehensive MCP tool testing on 2026-01-02*
