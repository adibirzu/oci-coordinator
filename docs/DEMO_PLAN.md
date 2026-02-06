# OCI AI Agent Coordinator - Demo Plan

**Version**: 1.0
**Last Updated**: 2026-01-08
**Purpose**: 30 working Slack commands demonstrating end-to-end capabilities

---

## Overview

This demo plan showcases 30 production-ready commands that demonstrate:
- Database troubleshooting and performance analysis
- Instance lifecycle management (create, start, stop)
- Cost analysis and comparison
- Security posture assessment
- Log analytics and observability
- Infrastructure discovery

All commands work via **Slack** and **REST API**.

---

## Quick Reference

| Category | Commands | Primary Use |
|----------|----------|-------------|
| Database Troubleshooting | 1-10 | Performance, blocking, wait events |
| Instance Management | 11-15 | List, start, stop, restart by name |
| Cost & Usage | 16-22 | Costs, comparisons, trends |
| Security | 23-26 | Posture, Cloud Guard, audit |
| Logs & Observability | 27-29 | Log queries, metrics |
| Discovery | 30 | Compartments, fleet overview |

---

## Commands by Category

### Database Troubleshooting (Commands 1-10)

#### Command 1: Database Fleet Overview
**Slack**:
```
@OCI Agent list databases
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list databases"}'
```
**Workflow**: `list_databases` → `oci_opsi_search_databases`

---

#### Command 2: Database Health Check
**Slack**:
```
@OCI Agent check database health for ATPAdi
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "check database health for ATPAdi"}'
```
**Workflow**: `db_performance_overview` → Multiple tools (SQLcl + OPSI)

---

#### Command 3: Check Blocking Sessions
**Slack**:
```
@OCI Agent check blocking sessions on ATPAdi
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "check blocking sessions on ATPAdi"}'
```
**Workflow**: `db_blocking_sessions_workflow` → `oci_database_execute_sql`
**SQL**: Queries `v$session` for `blocking_session IS NOT NULL`

---

#### Command 4: Wait Event Analysis
**Slack**:
```
@OCI Agent show wait events for ATPAdi
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show wait events for ATPAdi"}'
```
**Workflow**: `db_wait_events_workflow` → `oci_dbmgmt_get_wait_events`

---

#### Command 5: SQL Monitoring
**Slack**:
```
@OCI Agent show running SQL on ATPAdi
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show running SQL on ATPAdi"}'
```
**Workflow**: `db_sql_monitoring_workflow` → `oci_database_execute_sql`
**SQL**: Queries `v$sql_monitor` for `status = 'EXECUTING'`

---

#### Command 6: Long Running Operations
**Slack**:
```
@OCI Agent show long running operations on ATPAdi
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show long running operations on ATPAdi"}'
```
**Workflow**: `db_long_running_ops_workflow` → `oci_database_execute_sql`
**SQL**: Queries `gv$session_longops` for `time_remaining > 0`

---

#### Command 7: Parallelism Check
**Slack**:
```
@OCI Agent check parallelism for ATPAdi
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "check parallelism for ATPAdi"}'
```
**Workflow**: `db_parallelism_stats_workflow` → `oci_database_execute_sql`
**Detects**: PX downgrade (allocated < requested)

---

#### Command 8: Full Table Scan Detection
**Slack**:
```
@OCI Agent find full table scans on ATPAdi
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "find full table scans on ATPAdi"}'
```
**Workflow**: `db_full_table_scan_workflow` → `oci_database_execute_sql`
**Flags**: Tables > 1GB with full scans

---

#### Command 9: AWR Report Generation
**Slack**:
```
@OCI Agent generate AWR report for ATPAdi last hour
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "generate AWR report for ATPAdi last hour"}'
```
**Workflow**: `db_awr_report_workflow` → `oci_dbmgmt_get_awr_report`

---

#### Command 10: Top SQL by CPU
**Slack**:
```
@OCI Agent show top SQL by CPU on ATPAdi
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show top SQL by CPU on ATPAdi"}'
```
**Workflow**: `db_top_sql_workflow` → `oci_opsi_get_sql_statistics`

---

### Instance Management (Commands 11-15)

#### Command 11: List Running Instances
**Slack**:
```
@OCI Agent list running instances
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list running instances"}'
```
**Workflow**: `list_instances` → `oci_compute_list_instances`

---

#### Command 12: List Stopped Instances
**Slack**:
```
@OCI Agent list stopped instances
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list stopped instances"}'
```
**Workflow**: `list_instances` → `oci_compute_list_instances` (state=STOPPED)

---

#### Command 13: Start Instance by Name
**Slack**:
```
@OCI Agent start instance remove-dev-server
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "start instance remove-dev-server"}'
```
**Workflow**: `start_instance_by_name` → `oci_compute_start_by_name`
**Note**: No OCID required - uses name-based lookup

---

#### Command 14: Stop Instance by Name
**Slack**:
```
@OCI Agent stop instance remove-dev-server
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "stop instance remove-dev-server"}'
```
**Workflow**: `stop_instance_by_name` → `oci_compute_stop_by_name`
**Note**: Requires confirmation before execution

---

#### Command 15: Restart Instance by Name
**Slack**:
```
@OCI Agent restart instance remove-dev-server
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "restart instance remove-dev-server"}'
```
**Workflow**: `restart_instance_by_name` → `oci_compute_restart_by_name`

---

### Cost & Usage Analysis (Commands 16-22)

#### Command 16: Cost Summary (Current Month)
**Slack**:
```
@OCI Agent show cost summary
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show cost summary"}'
```
**Workflow**: `cost_summary` → `oci_cost_get_summary`

---

#### Command 17: Cost Summary for Specific Month
**Slack**:
```
@OCI Agent show cost summary for October
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show cost summary for October"}'
```
**Workflow**: `cost_summary` → `oci_cost_get_summary` (with date parsing)

---

#### Command 18: Cost by Service
**Slack**:
```
@OCI Agent show costs by service
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show costs by service"}'
```
**Workflow**: `cost_by_service` → `oci_cost_service_drilldown`

---

#### Command 19: Cost by Compartment
**Slack**:
```
@OCI Agent show costs by compartment
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show costs by compartment"}'
```
**Workflow**: `cost_by_compartment` → `oci_cost_by_compartment`

---

#### Command 20: Database Cost Drilldown
**Slack**:
```
@OCI Agent show database costs
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show database costs"}'
```
**Workflow**: `database_costs` → `oci_cost_database_drilldown`

---

#### Command 21: Compare Monthly Costs
**Slack**:
```
@OCI Agent compare costs August vs October vs November
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "compare costs August vs October vs November"}'
```
**Workflow**: `cost_comparison` → `oci_cost_usage_comparison`

---

#### Command 22: Monthly Cost Trend
**Slack**:
```
@OCI Agent show cost trend last 6 months
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show cost trend last 6 months"}'
```
**Workflow**: `monthly_trend` → `oci_cost_monthly_trend`

---

### Security (Commands 23-26)

#### Command 23: Security Overview
**Slack**:
```
@OCI Agent show security overview
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show security overview"}'
```
**Workflow**: `security_overview` → `oci_security_skill_skill_security_posture_summary`

---

#### Command 24: Cloud Guard Problems
**Slack**:
```
@OCI Agent list Cloud Guard problems
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list Cloud Guard problems"}'
```
**Workflow**: `cloud_guard_problems` → `oci_security_cloudguard_list_problems`

---

#### Command 25: Security Score
**Slack**:
```
@OCI Agent show security score
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show security score"}'
```
**Workflow**: `security_score` → `oci_security_cloudguard_cloudguard_get_security_score`

---

#### Command 26: Audit Events
**Slack**:
```
@OCI Agent show recent audit events
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show recent audit events"}'
```
**Workflow**: `audit_events` → `oci_security_audit_audit_list_events`

---

### Logs & Observability (Commands 27-29)

#### Command 27: Log Summary
**Slack**:
```
@OCI Agent show log summary last 24 hours
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show log summary last 24 hours"}'
```
**Workflow**: `log_summary` → `oci_logan_get_summary`

---

#### Command 28: Error Log Search
**Slack**:
```
@OCI Agent search logs for errors
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "search logs for errors"}'
```
**Workflow**: `log_search` → `oci_logan_execute_query`

---

#### Command 29: Active Alarms
**Slack**:
```
@OCI Agent list active alarms
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list active alarms"}'
```
**Workflow**: `list_alarms` → `oci_observability_list_alarms`

---

### Discovery (Command 30)

#### Command 30: List Compartments
**Slack**:
```
@OCI Agent list compartments
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list compartments"}'
```
**Workflow**: `list_compartments` → `oci_tenancy_list_compartments`

---

## Demo Scenarios

### Scenario A: Database Performance Investigation

```
1. @OCI Agent list databases
2. @OCI Agent check database health for ATPAdi
3. @OCI Agent check blocking sessions on ATPAdi
4. @OCI Agent show wait events for ATPAdi
5. @OCI Agent show running SQL on ATPAdi
6. @OCI Agent generate AWR report for ATPAdi last hour
```

### Scenario B: Cost Analysis Deep Dive

```
1. @OCI Agent show cost summary
2. @OCI Agent show costs by service
3. @OCI Agent show database costs
4. @OCI Agent compare costs October vs November
5. @OCI Agent show cost trend last 6 months
```

### Scenario C: Instance Lifecycle

```
1. @OCI Agent list stopped instances
2. @OCI Agent start instance remove-dev-server
3. @OCI Agent list running instances
4. @OCI Agent stop instance remove-dev-server
```

### Scenario D: Security Audit

```
1. @OCI Agent show security overview
2. @OCI Agent list Cloud Guard problems
3. @OCI Agent show recent audit events
4. @OCI Agent show security score
```

---

## Advanced Use Cases (New)

### Instance Metrics Analysis (Commands 31-33)

#### Command 31: Instance Metrics by Name
**Slack**:
```
@OCI Agent show metrics for remove-dev-server
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show metrics for remove-dev-server"}'
```
**Workflow**: `instance_metrics` → `oci_observability_get_instance_metrics`
**Note**: Looks up instance by name, returns CPU, memory, disk, network metrics with health assessment

---

#### Command 32: Instance Health Check
**Slack**:
```
@OCI Agent show CPU and memory usage of database-vm last 2 hours
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show CPU and memory usage of database-vm last 2 hours"}'
```
**Workflow**: `instance_metrics` → `oci_observability_get_instance_metrics` (hours_back=2)
**Returns**: CPU/memory avg, min, max with recommendations (scale up/down suggestions)

---

#### Command 33: Instance Metrics (ask for name)
**Slack**:
```
@OCI Agent show instance metrics
```
**Response**: "Please provide the instance name. Which instance would you like to see metrics for?"
**Note**: If no instance name provided, system asks for clarification with examples

---

### Log Analytics with Namespace Selection (Commands 34-37)

#### Command 34: List OCI Profiles
**Slack**:
```
@OCI Agent list OCI profiles
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list OCI profiles"}'
```
**Tool**: `oci_list_profiles`
**Returns**: Available profiles from ~/.oci/config with their regions

---

#### Command 35: List Log Analytics Namespaces
**Slack**:
```
@OCI Agent list log analytics namespaces using profile EMDEMO
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list log analytics namespaces using profile EMDEMO"}'
```
**Tool**: `oci_logan_list_namespaces`
**Returns**: Available Log Analytics namespaces for the specified tenancy

---

#### Command 36: List Log Groups
**Slack**:
```
@OCI Agent list log groups
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list log groups"}'
```
**Tool**: `oci_logan_list_log_groups`
**Returns**: Log groups with names, IDs, and descriptions

---

#### Command 37: Execute Log Query with Profile
**Slack**:
```
@OCI Agent search logs for errors using profile EMDEMO
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "search logs for errors using profile EMDEMO"}'
```
**Tool**: `oci_logan_execute_query`
**Note**: Uses specified OCI profile for multi-tenancy support

---

### Anomaly Detection & Correlation (Commands 38-40)

#### Command 38: Analyze Instance Anomalies
**Slack**:
```
@OCI Agent analyze performance anomalies for remove-dev-server
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "analyze performance anomalies for remove-dev-server"}'
```
**Workflow**: Combines metrics retrieval with LLM analysis
**Returns**: Anomaly detection based on CPU/memory patterns, deviations from baseline

---

#### Command 39: Correlate Logs with Metrics
**Slack**:
```
@OCI Agent correlate logs and metrics for database-vm last 1 hour
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "correlate logs and metrics for database-vm last 1 hour"}'
```
**Workflow**: Multi-step - gets metrics, queries logs, uses LLM to find correlations
**Returns**: Unified view showing metric spikes aligned with log events

---

#### Command 40: Performance Investigation
**Slack**:
```
@OCI Agent investigate slow response times on app-server-01
```
**API**:
```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "investigate slow response times on app-server-01"}'
```
**Workflow**: Deep investigation combining:
- Instance metrics (CPU, memory, network)
- Log analysis for errors/warnings
- LLM-powered root cause analysis
**Returns**: Probable causes with recommendations

---

## Demo Scenarios (Extended)

### Scenario E: Instance Metrics Investigation

```
1. @OCI Agent list running instances
2. @OCI Agent show metrics for remove-dev-server
3. @OCI Agent what's the CPU usage trend for the last 4 hours
4. @OCI Agent any anomalies in the metrics?
```

### Scenario F: Multi-Tenancy Log Analytics

```
1. @OCI Agent list OCI profiles
2. @OCI Agent list log analytics namespaces using profile EMDEMO
3. @OCI Agent list log groups using profile EMDEMO
4. @OCI Agent search logs for errors using profile EMDEMO
5. @OCI Agent show log summary last 24 hours
```

### Scenario G: Correlation Analysis

```
1. @OCI Agent show metrics for app-server-01 last 2 hours
2. @OCI Agent search logs for errors on app-server-01
3. @OCI Agent correlate the logs and metrics
4. @OCI Agent what could be causing the CPU spike at 2pm?
```

---

## API Streaming Support

For long-running commands, use the streaming endpoint:

```bash
curl -X POST http://localhost:3001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "generate AWR report for ATPAdi last hour"}' \
  --no-buffer
```

---

## Quick Actions Menu

In Slack, type `@OCI Agent` and use the interactive quick actions:

| Category | Actions |
|----------|---------|
| Database | Fleet Overview, Health Check, Blocking, Wait Events |
| Cost | Summary, By Service, By Compartment, Trends |
| Security | Overview, Cloud Guard, Audit Events |
| Infrastructure | Instances, VCNs, Compartments |
| Logs | Summary, Errors, Active Alarms |
| Discovery | Compartments, Fleet Overview |

---

## Testing Commands

To verify a command is working:

```bash
# Health check
curl http://localhost:3001/health

# System status
curl http://localhost:3001/status

# Test chat endpoint
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list compartments"}'
```

---

## References

- **FEATURE_MAPPING.md** - Complete tool-to-workflow mapping
- **DB_TROUBLESHOOTING_WORKFLOW.md** - Database diagnostic runbook
- **OCI_AGENT_REFERENCE.md** - Agent configurations
- **ARCHITECTURE.md** - System architecture details
