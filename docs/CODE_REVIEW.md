# OCI AI Agent Coordinator - Comprehensive Code Review

**Date**: 2025-12-31
**Reviewer**: Claude Code
**Status**: Phase 3 - Gap Analysis Complete

## Executive Summary

This document provides a complete code review and feature mapping of the OCI AI Agent Coordinator. The system implements a multi-agent architecture for Oracle Cloud Infrastructure operations with MCP (Model Context Protocol) tool integration.

---

## 1. System Architecture Overview

### 1.1 Component Summary

| Component | Count | Status |
|-----------|-------|--------|
| Agents | 5 + Coordinator | Active |
| MCP Servers | 4 | Connected |
| MCP Tools | 86+ | Available |
| Channels | Slack, API (planned) | Active |
| Skills/Workflows | 10+ | Registered |

### 1.2 Request Flow

```
User (Slack/API)
     ↓
Channel Handler (Slack/FastAPI)
     ↓
AsyncRuntime (shared event loop)
     ↓
Agent Routing (keyword-based currently)
     ↓
SpecializedReActAgent
     ↓
MCP Tool Catalog → MCP Servers
     ↓
OCI APIs
     ↓
Response Formatting (Slack Block Kit)
     ↓
User
```

---

## 2. Agent Inventory

### 2.1 Active Agents

| Agent | Role ID | Domain | MCP Servers |
|-------|---------|--------|-------------|
| DB Troubleshoot | `db-troubleshoot-agent` | database | database-observatory |
| Infrastructure | `infrastructure-agent` | compute, network | oci-unified, oci-infrastructure |
| FinOps | `finops-agent` | cost, budget | finopsai, oci-unified |
| Security Threat | `security-threat-agent` | security, IAM | oci-unified |
| Log Analytics | `log-analytics-agent` | observability | oci-unified, database-observatory |

### 2.2 Agent Capabilities Matrix

| Agent | Capabilities |
|-------|--------------|
| DB Troubleshoot | database-analysis, performance-diagnostics, sql-tuning, blocking-analysis |
| Infrastructure | compute-management, network-analysis, security-operations, resource-inventory |
| FinOps | cost-analysis, budget-tracking, optimization, anomaly-detection |
| Security Threat | threat-detection, compliance-monitoring, security-posture, mitre-mapping |
| Log Analytics | log-search, pattern-detection, log-correlation, audit-analysis |

---

## 3. MCP Tool Inventory

### 3.1 Server Distribution

| Server | Tools | Domains | Transport |
|--------|-------|---------|-----------|
| oci-unified | 28 | identity, compute, network, cost, security | stdio |
| database-observatory | 43 | database, opsi, logan | stdio |
| oci-infrastructure | 44 | compute, network, security, database | stdio |
| finopsai | 12+ | cost, finops, anomaly | stdio |

### 3.2 Tool Naming Conventions

**Current State** (INCONSISTENT):
```
oci-unified:         oci_{domain}_{action}     (oci_compute_list_instances)
database-observatory: {action}_{noun}          (execute_sql, get_fleet_summary)
oci-infrastructure:  oci_{domain}_{action}     (oci_compute_list_instances)
finopsai:            oci_cost_{action}         (oci_cost_by_compartment)
```

**Recommended Standard**:
```
All tools:           oci_{domain}_{action}
Examples:
  - oci_compute_list_instances
  - oci_database_execute_sql
  - oci_opsi_get_fleet_summary
  - oci_cost_get_summary
```

---

## 4. Identified Issues

### 4.1 CRITICAL: Naming Inconsistencies

| Issue | Location | Current | Should Be |
|-------|----------|---------|-----------|
| Tool naming | database-observatory | `execute_sql` | `oci_database_execute_sql` |
| Tool naming | database-observatory | `get_fleet_summary` | `oci_opsi_get_fleet_summary` |
| Tool naming | database-observatory | `analyze_cpu_usage` | `oci_opsi_analyze_cpu_usage` |
| Agent reference | react_agent.py:479 | `list_autonomous_databases` | `oci_database_list_autonomous` |
| Agent reference | react_agent.py:482 | `analyze_performance` | `oci_opsi_get_performance_summary` |
| Agent reference | react_agent.py:483 | `get_blocking_sessions` | `oci_opsi_get_blocking_sessions` |

### 4.2 HIGH: Missing Interconnections

| Issue | Description | Impact |
|-------|-------------|--------|
| Slack bypasses Coordinator | Slack handler uses keyword routing, not LangGraph | No workflow-first routing |
| No agent-to-agent calls | Agents can't delegate to each other | Limited multi-agent workflows |
| ReAct not using coordinator | SpecializedReActAgent runs independently | Inconsistent routing |
| Skill handlers not implemented | Skills defined but handlers are stubs | Workflows don't execute |

### 4.3 MEDIUM: Configuration Gaps

| Issue | Description | Fix |
|-------|-------------|-----|
| Hardcoded tool lists | SpecializedReActAgent has outdated tool lists | Use tool catalog dynamically |
| Missing tool descriptions | Some tools lack proper descriptions | Add descriptions |
| Duplicate tool functionality | oci-unified and oci-infrastructure overlap | Use server groups |

### 4.4 LOW: Documentation Gaps

| Issue | Description |
|-------|-------------|
| Agent prompt files incomplete | prompts/ folder has 6 files but content varies |
| Skill definitions not documented | Skills exist but not in CLAUDE.md |
| Tool tiers not documented | Risk levels exist but not surfaced |

---

## 5. Agent-Tool Mapping (Current vs Expected)

### 5.1 Infrastructure Agent

**Current Tools Referenced**:
```python
# In react_agent.py - OUTDATED
- oci_list_compartments ✓
- oci_search_compartments ✓
- oci_get_tenancy ✓
- oci_compute_list_instances ✓
- oci_network_list_vcns ✓
```

**All Available Tools** (oci-unified + oci-infrastructure):
```
# Identity
- oci_list_compartments
- oci_get_compartment
- oci_search_compartments
- oci_get_tenancy
- oci_list_regions

# Compute
- oci_compute_list_instances
- oci_compute_get_instance
- oci_compute_find_instance
- oci_compute_start_instance (Tier 4)
- oci_compute_stop_instance (Tier 4)
- oci_compute_restart_instance (Tier 4)
- oci_compute_start_by_name (Tier 4)
- oci_compute_stop_by_name (Tier 4)
- oci_compute_restart_by_name (Tier 4)

# Network
- oci_network_list_vcns
- oci_network_list_subnets
- oci_network_list_security_lists
```

### 5.2 Database Agent

**Current Tools Referenced**:
```python
# In react_agent.py - OUTDATED NAMES
- list_autonomous_databases → oci_database_list_autonomous
- database_status → oci_database_status (doesn't exist)
- analyze_performance → ?
- get_blocking_sessions → ?
```

**All Available Tools** (database-observatory):
```
# SQLcl
- execute_sql
- get_schema_info
- list_connections
- database_status

# OPSI Discovery
- get_fleet_summary
- search_databases
- list_database_insights

# OPSI Analytics
- analyze_cpu_usage
- analyze_memory_usage
- analyze_io_usage
- get_sql_statistics
- analyze_wait_events
- get_blocking_sessions
- compare_awr_periods

# Database System
- list_tablespaces
- list_users
- get_sql_plan
- list_awr_snapshots
```

### 5.3 FinOps Agent

**Current Tools Referenced**:
```python
# In react_agent.py
- oci_cost_get_summary ✓
- oci_cost_by_service ✓
- oci_cost_by_compartment ✓
- oci_cost_detect_anomalies ✗ (doesn't exist)
```

**All Available Tools** (finopsai + oci-unified):
```
# From finopsai
- oci_cost_ping
- oci_cost_templates
- oci_cost_by_compartment
- oci_cost_service_drilldown
- oci_cost_by_tag
- oci_cost_monthly_trend
- oci_cost_budget_status
- oci_cost_object_storage
- oci_cost_unit_cost
- oci_cost_forecast_credits
- oci_cost_focus_health
- oci_cost_spikes
- oci_cost_schedules

# From oci-unified
- oci_cost_get_summary
```

---

## 6. Recommended Fixes

### 6.1 Phase 1: Naming Standardization

1. **Update database-observatory tools** to use `oci_` prefix
2. **Update agent DOMAIN_PROMPTS** with correct tool names
3. **Add tool name aliases** in catalog for backward compatibility

### 6.2 Phase 2: Enable LangGraph Coordinator

1. **Modify Slack handler** to use LangGraph coordinator
2. **Implement agent delegation** in coordinator nodes
3. **Add workflow execution** handlers

### 6.3 Phase 3: Agent Enhancement

1. **Dynamic tool discovery** - agents query catalog, not hardcoded lists
2. **Cross-agent delegation** - agents can invoke other agents
3. **Skill execution** - implement actual handlers for skills

### 6.4 Phase 4: Documentation

1. **Update CLAUDE.md** with complete tool inventory
2. **Add tool tier documentation** (risk levels)
3. **Document agent-to-agent protocol**

---

## 7. Tool Tier Reference

### Tier 1: Discovery (Instant, <100ms)
- Health checks, templates, capability searches
- Example: `oci_ping`, `search_capabilities`

### Tier 2: Fast Reads (100ms-1s)
- List/get operations, no side effects
- Example: `oci_compute_list_instances`, `oci_cost_get_summary`

### Tier 3: Analysis (1-30s)
- Heavy queries, analytics, diagnostics
- Example: `analyze_cpu_usage`, `oci_observability_query_logs`

### Tier 4: Mutations (10-15s, Requires Confirmation)
- Start/stop/restart, scaling, remediation
- Example: `oci_compute_start_instance`, `oci_compute_stop_by_name`

---

## 8. Action Items

### Immediate (Before Next Release)

- [ ] Fix tool name references in `react_agent.py` DOMAIN_PROMPTS
- [ ] Add missing tools to agent MCP tool lists
- [ ] Update Slack handler to use coordinator (optional)

### Short Term (Next Sprint)

- [ ] Standardize database-observatory tool names
- [ ] Implement skill execution handlers
- [ ] Add agent-to-agent delegation protocol

### Long Term (Roadmap)

- [ ] Full LangGraph coordinator integration
- [ ] Multi-agent orchestration
- [ ] Cross-domain workflow execution

---

## 9. Test Coverage Recommendations

| Area | Current | Target | Tests Needed |
|------|---------|--------|--------------|
| Agent Use Cases | 50 tests | 50+ | ✓ Complete |
| MCP Tools | ~30 tests | 80+ | Tool execution tests |
| E2E Workflows | ~10 tests | 50+ | Full workflow tests |
| Channel Integration | ~5 tests | 20+ | Slack/API tests |

---

## Appendix A: Complete Tool List by Server

### oci-unified (28 tools)

```
# Identity
oci_list_compartments
oci_get_compartment
oci_search_compartments
oci_get_tenancy
oci_list_regions

# Compute
oci_compute_list_instances
oci_compute_get_instance
oci_compute_find_instance
oci_compute_start_instance
oci_compute_stop_instance
oci_compute_restart_instance
oci_compute_start_by_name
oci_compute_stop_by_name
oci_compute_restart_by_name

# Network
oci_network_list_vcns
oci_network_list_subnets
oci_network_list_security_lists

# Cost
oci_cost_get_summary

# Security
oci_security_list_users

# Observability
oci_observability_get_metrics

# Discovery (ShowOCI)
oci_discovery_run
oci_discovery_get_cached
oci_discovery_refresh
oci_discovery_summary
oci_discovery_search
oci_discovery_cache_status

# Feedback
set_feedback
append_feedback
get_feedback
search_capabilities
```

### database-observatory (43 tools)

```
# SQLcl
execute_sql
get_schema_info
list_connections
database_status

# OPSI Discovery
get_fleet_summary
search_databases
list_database_insights

# OPSI Analytics
analyze_cpu_usage
analyze_memory_usage
analyze_io_usage
query_warehouse_standard

# Database System
list_tablespaces
list_users
get_sql_plan
list_awr_snapshots

# OPSI Diagnostics
get_sql_statistics
analyze_wait_events
get_blocking_sessions
compare_awr_periods

# SQLWatch
sqlwatch_get_plan_history
sqlwatch_analyze_regression
```

### finopsai (12+ tools)

```
oci_cost_ping
oci_cost_templates
oci_cost_by_compartment
oci_cost_service_drilldown
oci_cost_by_tag
oci_cost_monthly_trend
oci_cost_budget_status
oci_cost_object_storage
oci_cost_unit_cost
oci_cost_forecast_credits
oci_cost_focus_health
oci_cost_spikes
oci_cost_schedules
```

---

## Appendix B: Agent-Domain-Tool Matrix

| Agent | Primary Domain | MCP Server | Key Tools |
|-------|---------------|------------|-----------|
| db-troubleshoot | database | database-observatory | execute_sql, get_fleet_summary, analyze_* |
| infrastructure | compute | oci-unified | oci_compute_*, oci_network_*, oci_list_* |
| finops | cost | finopsai | oci_cost_*, oci_cost_spikes |
| security-threat | security | oci-unified | oci_security_*, search_capabilities |
| log-analytics | observability | oci-unified | oci_observability_*, search_capabilities |

---

*End of Code Review Document*
