# MCP Server Compliance Report

## Overview

This document assesses all OCI MCP servers against the BUILD_MCP_SERVER guidelines from the mcp-oci reference implementation.

**Assessment Date:** 2025-12-31
**Reference Implementation:** mcp-oci v2.3.0

---

## Compliance Matrix

| Server | Naming | Annotations | Pydantic | Response Formats | Caching | Tier Docs | Overall |
|--------|--------|-------------|----------|-----------------|---------|-----------|---------|
| mcp-oci | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **A** |
| oci-mcp-security | ⚠️ | ✅ | ✅ | ❌ | ✅ | ✅ | **B+** |
| mcp-oci-database-observatory | ❌ | ❌ | ❌ | ❌ | ✅ | ⚠️ | **C** |
| finopsai-mcp | ❌ | ❌ | ⚠️ | ✅ | ❌ | ⚠️ | **C** |
| opsi | ❌ | ❌ | ❌ | ❌ | ✅ | ⚠️ | **C-** |

Legend: ✅ Compliant | ⚠️ Partial | ❌ Non-compliant

---

## Detailed Assessments

### 1. mcp-oci (Reference Implementation)

**Grade: A**

**Strengths:**
- Tool naming follows `oci_{domain}_{action}_{resource}` convention
- All tools have FastMCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`)
- Uses Pydantic `BaseModel` for input validation
- Supports both markdown and JSON response formats
- TTL-based tiered caching implemented
- Tool tiers well-documented (1=instant, 2=fast, 3=moderate, 4=mutations)
- Backward-compatible tool aliases
- Skills framework with `SkillExecutor`

**Tools Count:** 44 tools across 8 domains

---

### 2. oci-mcp-security

**Grade: B+**

**Strengths:**
- Excellent FastMCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`)
- Modular architecture using `mcp.mount()` for service separation
- Uses `Annotated[type, Field(...)]` pattern for validation
- Tool tiers well-documented in manifest (tier1-tier4)
- Comprehensive service coverage (CloudGuard, VSS, Bastion, DataSafe, WAF, Audit, KMS)
- OAuth support for HTTP/SSE transports
- MCP Resources support (`cloudguard://problems/{id}`)
- MCP Prompts support

**Issues:**
- Tool naming uses `{service}_{action}` instead of `oci_security_{service}_{action}`
  - Current: `cloudguard_list_problems`, `vss_list_host_scans`
  - Recommended: `oci_security_cloudguard_list_problems`
- No markdown response format support

**Recommendations:**
1. Add `oci_security_` prefix to all tools for consistency
2. Implement `response_format` parameter for markdown output

---

### 3. mcp-oci-database-observatory

**Grade: C**

**Type:** Multi-agent coordinator with embedded MCP server

**Strengths:**
- Comprehensive agent orchestration (LangGraph)
- Multi-tenancy support via OCI profiles
- File-based cache for OPSI data
- ACL error detection and guidance
- Multi-provider LLM support (OCA, Anthropic, OpenAI, Gemini)

**Issues:**
- Tool naming uses simple names without `oci_` prefix
  - Current: `execute_sql`, `get_schema_info`, `list_connections`
  - Recommended: `oci_database_execute_sql`, `oci_database_get_schema`
- No FastMCP tool annotations
- Inline parameters instead of Pydantic `BaseModel` classes
- Only JSON output, no markdown format

**Recommendations:**
1. Rename tools with `oci_database_` prefix for SQLcl tools
2. Rename OPSI tools with `oci_opsi_` prefix
3. Add FastMCP annotations for tool safety hints
4. Add `response_format` parameter

---

### 4. finopsai-mcp

**Grade: C**

**Strengths:**
- Response envelope with both `summary` (markdown) and `data` (JSON)
- OCI APM/OTEL tracing configured
- Input validation helpers (`_validate_date()`, `_validate_positive_int()`)
- Pydantic output schemas defined

**Issues:**
- Tool naming without `oci_` prefix
  - Current: `cost_by_compartment_daily`, `service_cost_drilldown`
  - Recommended: `oci_cost_by_compartment_daily`, `oci_finops_service_drilldown`
- No FastMCP tool annotations
- No caching implementation
- Inline parameters instead of Pydantic input models

**Recommendations:**
1. Add `oci_cost_` or `oci_finops_` prefix to all tools
2. Add FastMCP annotations
3. Implement TTL-based caching for expensive queries
4. Convert inline parameters to Pydantic `BaseModel` classes

---

### 5. opsi (mcp-oci-opsi)

**Grade: C-**

**Strengths:**
- Modular sub-servers (cache_server, opsi_server, dbm_server, admin_server)
- Large tool collection (22+ tool files)
- Skills framework (tools_skills.py, tools_skills_v2.py)
- File-based caching

**Issues:**
- Plain function names without `oci_` prefix
  - Current: `list_database_insights`, `query_warehouse_standard`
  - Recommended: `oci_opsi_list_database_insights`
- Functions not decorated with `@mcp.tool()`
- Only JSON output
- No FastMCP annotations

**Recommendations:**
1. Add `oci_opsi_` prefix to all tools
2. Convert functions to `@mcp.tool()` decorated async functions
3. Add FastMCP annotations
4. Add `response_format` parameter for markdown output

---

## Priority Actions

### High Priority (P0)

1. **Standardize naming convention across all servers**
   - All tools should follow: `oci_{domain}_{action}_{resource}`
   - This enables unified tool discovery and routing

2. **Add FastMCP annotations to all servers**
   - `readOnlyHint: True` for read operations
   - `destructiveHint: True` for mutations
   - `idempotentHint: True` where applicable

### Medium Priority (P1)

3. **Implement response format parameter**
   - Add `response_format: Literal["markdown", "json"] = "markdown"`
   - Markdown for LLM context efficiency
   - JSON for programmatic processing

4. **Add caching to finopsai-mcp**
   - Cost queries are expensive and relatively stable
   - Implement TTL-based caching

### Low Priority (P2)

5. **Convert inline parameters to Pydantic models**
   - Improves validation and documentation
   - Enables auto-generated JSON schemas

6. **Standardize error handling**
   - Use structured `OCIError` pattern from mcp-oci
   - Include category, message, and suggestion

---

## Tool Inventory by Server

### mcp-oci (44 tools)
- Discovery: 4 (`oci_ping`, `oci_list_domains`, `oci_search_tools`, `oci_get_cache_stats`)
- Compute: 5 (`oci_compute_list_instances`, `oci_compute_get_instance`, etc.)
- Network: 5 (`oci_network_list_vcns`, `oci_network_get_vcn`, etc.)
- Database: 5 (`oci_database_list_autonomous`, etc.)
- Security: 6 (`oci_security_list_users`, `oci_security_audit`, etc.)
- Cost: 5 (`oci_cost_get_summary`, `oci_cost_detect_anomalies`, etc.)
- Observability: 6 (`oci_observability_get_instance_metrics`, etc.)
- Skills: 1 (`oci_skill_troubleshoot_instance`)

### oci-mcp-security (~35 tools)
- CloudGuard: 7 (`cloudguard_list_problems`, `cloudguard_remediate_problem`, etc.)
- VSS: 4 (`vss_list_host_scans`, `vss_list_vulnerabilities`, etc.)
- Security Zones: 3 (`securityzones_list_zones`, etc.)
- Bastion: 4 (`bastion_list`, `bastion_list_sessions`, etc.)
- DataSafe: 4 (`datasafe_list_targets`, `datasafe_list_findings`, etc.)
- WAF: 4 (`waf_list_firewalls`, `waf_list_policies`, etc.)
- Audit: 2 (`audit_list_events`, `audit_get_configuration`)
- Access Governance: 2 (`accessgov_list_instances`, etc.)
- KMS: 4 (`kms_list_vaults`, `kms_list_keys`, etc.)
- Skills: 3 (`skill_security_posture_summary`, etc.)

### finopsai-mcp (12 tools)
- `templates`
- `cost_by_compartment_daily`
- `service_cost_drilldown`
- `cost_by_tag_key_value`
- `monthly_trend_forecast`
- `focus_etl_healthcheck`
- `budget_status_and_actions`
- `schedule_report_create_or_list`
- `object_storage_costs_and_tiering`
- `top_cost_spikes_explain`
- `per_compartment_unit_cost`
- `forecast_vs_universal_credits`

### mcp-oci-database-observatory (~15 tools)
- SQLcl: `execute_sql`, `get_schema_info`, `list_connections`, `database_status`
- OPSI: `get_fleet_summary`, `search_databases`, `analyze_cpu_usage`, etc.
- Network: `check_network_access`, `get_my_ip`
- Health: `health_check`

### opsi (50+ tools across 22 files)
- Database Discovery: registration, discovery
- DB Management: AWR metrics, monitoring, SQL plans, tablespaces, users
- OPSI: diagnostics, extended, resource stats, SQL insights
- SQLWatch: bulk operations
- Skills: v1 and v2 workflows

---

## Coordination Catalog Updates Required

The following tools need to be added to `oci-coordinator/src/mcp/catalog.py` TOOL_TIERS:

```python
# oci-mcp-security tools (with prefix recommendation)
"oci_security_cloudguard_list_problems": ToolTier(2, 600, "none"),
"oci_security_cloudguard_get_problem": ToolTier(2, 400, "none"),
"oci_security_cloudguard_remediate_problem": ToolTier(4, 5000, "medium", True),
"oci_security_vss_list_host_scans": ToolTier(2, 600, "none"),
"oci_security_bastion_list_sessions": ToolTier(2, 500, "none"),
"oci_security_audit_list_events": ToolTier(3, 2000, "none"),

# finopsai-mcp tools (with prefix recommendation)
"oci_cost_by_compartment_daily": ToolTier(2, 800, "none"),
"oci_cost_service_drilldown": ToolTier(2, 1000, "none"),
"oci_cost_monthly_trend": ToolTier(2, 800, "none"),
"oci_cost_top_spikes": ToolTier(3, 2000, "none"),
"oci_cost_forecast_credits": ToolTier(2, 800, "none"),

# mcp-oci-database-observatory tools
"oci_database_execute_sql": ToolTier(3, 5000, "medium"),
"oci_database_get_schema": ToolTier(2, 1000, "none"),
"oci_opsi_analyze_cpu": ToolTier(2, 2000, "none"),
"oci_opsi_analyze_memory": ToolTier(2, 2000, "none"),
```
