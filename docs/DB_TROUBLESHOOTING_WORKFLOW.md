# Database Troubleshooting Workflow Mapping

**Version**: 1.0
**Last Updated**: 2026-01-08

## Overview

This document maps the end-to-end database troubleshooting workflow to available MCP tools, agents, and workflows in the OCI AI Agent Coordinator system.

---

## Workflow Phases

### Phase 1: Ingestion & Triage (Teams/Slack <-> GenAI)

| Step | Description | Status | Tool/Workflow | Notes |
|------|-------------|--------|---------------|-------|
| 1.1 | **Intent Recognition** | ✅ Implemented | LangGraph Router (`classify_intent_node`) | Classifies to `db_performance_overview`, `check_blocking`, etc. |
| 1.2 | **Entity Extraction** | ✅ Implemented | LangGraph Router (`extract_entities`) | Extracts `DB_NAME`, `database_id` from query |
| 1.3 | **Triage (Interactive)** | ⚠️ Partial | `DBTriageState` in `troubleshoot_database.py` | Priority/severity classification exists but interactive prompting needs enhancement |

**Relevant Intents**:
- `db_performance_overview` - Comprehensive performance check
- `check_blocking` - Blocking session detection
- `long_running_ops` - Long operation monitoring
- `slow_queries` - SQL performance analysis

---

### Phase 2: Diagnostic Execution ("Runbook" Loop)

#### Step 1: Check Blocking Sessions

| Aspect | Value |
|--------|-------|
| **Excel Reference** | Row 2 |
| **Status** | ✅ Fully Implemented |
| **Primary Tool** | `oci_database_execute_sql` (via SQLcl) |
| **Workflow** | `db_blocking_sessions_workflow` |
| **Intent Aliases** | `blocking_sessions`, `check_blocking`, `lock_contention`, `session_blocking` |
| **SQL View Used** | `v$session`, `v$lock` |
| **Interpretation Logic** | IF `blocking_session IS NOT NULL` THEN flag as "Blocking Issue" |

**Tool Call Example**:
```json
{
  "tool": "oci_database_execute_sql",
  "connection_name": "ATPAdi",
  "sql": "SELECT blocking_session, sid, serial#, username, event, wait_class, seconds_in_wait FROM v$session WHERE blocking_session IS NOT NULL"
}
```

---

#### Step 2: Analyze CPU Usage & Wait Events

| Aspect | Value |
|--------|-------|
| **Excel Reference** | Row 3 |
| **Status** | ✅ Fully Implemented |
| **Primary Tools** | `oci_opsi_summarize_resource_stats`, `oci_dbmgmt_get_wait_events` |
| **Alternative Tools** | `oci_opsi_analyze_cpu`, `oci_dbmgmt_get_metrics` |
| **Workflows** | `db_wait_events_workflow`, `opsi_utilization` |
| **Intent Aliases** | `wait_events`, `cpu_usage`, `db_utilization` |

**Interpretation Logic**:
- IF `CPU_Usage > 64%` AND `Duration > 30mins` THEN flag as "CPU Capacity Issue"
- Calculate `CPU_Allocated = 16 * allocation_percentage`

**Tool Call Examples**:
```json
// Via OPSI
{
  "tool": "oci_opsi_summarize_resource_stats",
  "resource_type": "CPU",
  "analysis_time_interval": "PT30M"
}

// Via DB Management
{
  "tool": "oci_dbmgmt_get_wait_events",
  "managed_database_id": "ocid1.database...",
  "top_n": 10
}
```

---

#### Step 3: SQL Monitoring Report

| Aspect | Value |
|--------|-------|
| **Excel Reference** | Row 4 |
| **Status** | ✅ Fully Implemented |
| **Primary Tool** | `oci_database_execute_sql` (via SQLcl) |
| **Alternative Tools** | `oci_dbmgmt_get_top_sql`, `oci_opsi_summarize_sql_statistics` |
| **Workflow** | `db_sql_monitoring_workflow` |
| **Intent Aliases** | `sql_monitoring`, `sql_monitor`, `active_sql`, `running_queries` |
| **SQL View Used** | `v$sql_monitor` |

**Tool Call Example**:
```json
{
  "tool": "oci_database_execute_sql",
  "connection_name": "ATPAdi",
  "sql": "SELECT sql_id, status, cpu_time/1e6 cpu_secs, elapsed_time/1e6 elapsed_secs, sql_text FROM v$sql_monitor WHERE status = 'EXECUTING' ORDER BY elapsed_time DESC FETCH FIRST 10 ROWS ONLY"
}
```

---

#### Step 4: Check Long Running Operations

| Aspect | Value |
|--------|-------|
| **Excel Reference** | Row 5 |
| **Status** | ✅ Fully Implemented |
| **Primary Tool** | `oci_database_execute_sql` (via SQLcl) |
| **Workflow** | `db_long_running_ops_workflow` |
| **Intent Aliases** | `long_running_ops`, `longops`, `batch_progress`, `session_longops` |
| **SQL View Used** | `gv$session_longops` |

**Interpretation Logic**:
- Return "Time Remaining" estimation: `time_remaining = (sofar / totalwork) * elapsed_seconds`

**Tool Call Example**:
```json
{
  "tool": "oci_database_execute_sql",
  "connection_name": "ATPAdi",
  "sql": "SELECT sid, serial#, opname, target, sofar, totalwork, units, elapsed_seconds, time_remaining FROM gv$session_longops WHERE time_remaining > 0 ORDER BY time_remaining DESC"
}
```

---

#### Step 5: Parallelism Check

| Aspect | Value |
|--------|-------|
| **Excel Reference** | Row 6 |
| **Status** | ✅ Fully Implemented |
| **Primary Tool** | `oci_database_execute_sql` (via SQLcl) |
| **Workflow** | `db_parallelism_stats_workflow` |
| **Intent Aliases** | `parallelism_stats`, `px_stats`, `req_degree`, `px_downgrade` |
| **SQL View Used** | `v$sql` |

**Interpretation Logic**:
- IF `actual_degree < requested_degree` THEN flag as "Parallelism Downgrade"

**Tool Call Example**:
```json
{
  "tool": "oci_database_execute_sql",
  "connection_name": "ATPAdi",
  "sql": "SELECT sql_id, px_servers_requested, px_servers_allocated, cpu_time/1e6 cpu_secs FROM v$sql WHERE px_servers_requested > 0 AND px_servers_allocated < px_servers_requested ORDER BY cpu_time DESC FETCH FIRST 10 ROWS ONLY"
}
```

---

#### Step 6: Table Scans & AWR Collection

| Aspect | Value |
|--------|-------|
| **Excel Reference** | Row 7 & 8 |
| **Status** | ✅ Fully Implemented |
| **Primary Tools** | `oci_database_execute_sql`, `oci_dbmgmt_get_awr_report` |
| **Workflows** | `db_full_table_scan_workflow`, `db_awr_report_workflow` |
| **Intent Aliases** | `full_table_scan`, `table_scan`, `awr_report`, `awr_analysis` |

**Table Scan Detection**:
```json
{
  "tool": "oci_database_execute_sql",
  "connection_name": "ATPAdi",
  "sql": "SELECT p.sql_id, p.object_name, t.bytes/1024/1024/1024 size_gb, p.operation, p.options FROM v$sql_plan p JOIN dba_tables t ON p.object_name = t.table_name WHERE p.operation = 'TABLE ACCESS' AND p.options = 'FULL' AND t.bytes > 1073741824 ORDER BY t.bytes DESC"
}
```

**AWR Report Generation**:
```json
{
  "tool": "oci_dbmgmt_get_awr_report",
  "managed_database_id": "ocid1.database...",
  "begin_snapshot_id": 100,
  "end_snapshot_id": 110
}
```

---

### Phase 3: Synthesis & Reporting

| Step | Description | Status | Implementation |
|------|-------------|--------|----------------|
| 3.1 | **Aggregate Findings** | ✅ Implemented | `DBTroubleshootSkill._generate_report_node()` |
| 3.2 | **Draft Response** | ✅ Implemented | LLM synthesis via `invoke()` method |
| 3.3 | **Send to Teams/Slack** | ✅ Implemented | `SlackMessageHandler` with mrkdwn formatting |

---

## Capability Matrix

### Full Workflow Support

| Step | Tool Available | Workflow Exists | Intent Mapped | Test Status |
|------|---------------|-----------------|---------------|-------------|
| Blocking Sessions | ✅ | ✅ | ✅ | ⚠️ Pending |
| CPU/Wait Analysis | ✅ | ✅ | ✅ | ⚠️ Pending |
| SQL Monitoring | ✅ | ✅ | ✅ | ⚠️ Pending |
| Long Running Ops | ✅ | ✅ | ✅ | ⚠️ Pending |
| Parallelism Check | ✅ | ✅ | ✅ | ⚠️ Pending |
| Full Table Scan | ✅ | ✅ | ✅ | ⚠️ Pending |
| AWR Report | ✅ | ✅ | ✅ | ⚠️ Pending |
| Report Generation | ✅ | ✅ | ✅ | ✅ Passed |

---

## MCP Server Tool Mapping

### oci-unified Server (DB Management & OPSI)

| Tool | Capability | Workflow |
|------|------------|----------|
| `oci_dbmgmt_get_wait_events` | Wait event analysis | `db_wait_events_workflow` |
| `oci_dbmgmt_get_top_sql` | Top SQL by resource | `db_top_sql_workflow` |
| `oci_dbmgmt_get_awr_report` | AWR snapshot report | `db_awr_report_workflow` |
| `oci_dbmgmt_get_fleet_health` | Fleet health summary | `db_fleet_health_workflow` |
| `oci_opsi_summarize_resource_stats` | CPU/Memory utilization | `opsi_utilization` |
| `oci_opsi_get_addm_findings` | ADDM recommendations | `addm_findings` |

### database-observatory Server (SQLcl Real-time)

| Tool | Capability | Workflow |
|------|------------|----------|
| `oci_database_execute_sql` | Direct v$ queries | All SQLcl workflows |
| `oci_opsi_search_databases` | Database discovery | `list_databases` |
| `oci_opsi_get_fleet_summary` | Fleet overview | `fleet_summary` |

---

## Gap Analysis

### Identified Gaps

| Gap | Priority | Recommendation |
|-----|----------|----------------|
| **Interactive Triage** | High | Add follow-up question prompting for severity/urgency |
| **Automated Root Cause Detection** | Medium | Enhance `DBTroubleshootSkill` with ML-based anomaly detection |
| **Historical vs Real-time Context** | Medium | Add time range selection in triage phase |
| **Multi-database Comparison** | Low | Add cross-database correlation analysis |

### Enhancement Opportunities

1. **Composite Workflow**: Create `db_full_rca_workflow` that chains all 6 diagnostic steps automatically
2. **Threshold Configuration**: Make CPU/wait thresholds configurable via env vars
3. **Recommendation Engine**: Auto-generate tuning recommendations based on findings

---

## Usage Examples

### Slack/Teams Commands

```
# Full performance check
@Oracle OCI Ops Agent check database performance for ATPAdi

# Specific checks
@Oracle OCI Ops Agent check blocking sessions on ATPAdi
@Oracle OCI Ops Agent show long running operations on ATPAdi
@Oracle OCI Ops Agent analyze parallelism for ATPAdi
@Oracle OCI Ops Agent find full table scans on ATPAdi
@Oracle OCI Ops Agent generate AWR report for ATPAdi

# Fleet-wide
@Oracle OCI Ops Agent show database fleet health
```

### API Endpoints

```bash
# Full RCA
curl -X POST http://localhost:3001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Check database performance for ATPAdi", "context": {"priority": "P1"}}'

# Specific workflow
curl -X POST http://localhost:3001/workflow \
  -H "Content-Type: application/json" \
  -d '{"workflow": "db_blocking_sessions", "params": {"connection_name": "ATPAdi"}}'
```

---

## References

- **FEATURE_MAPPING.md**: Complete tool-to-workflow mapping
- **OCI_AGENT_REFERENCE.md**: Agent configurations and schemas
- **RUNBOOK.md**: Operational procedures
- **troubleshoot_database.py**: Main skill implementation
