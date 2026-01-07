# OCI AI Agent Coordinator - Comprehensive Code Review

**Date**: 2026-01-01
**Reviewer**: Claude Code (Opus 4.5)
**Status**: Phase 4 - Complete Feature Mapping

---

## Executive Summary

This document provides a complete code review and feature mapping of the OCI AI Agent Coordinator. The system implements a **workflow-first multi-agent architecture** using LangGraph for Oracle Cloud Infrastructure operations with MCP (Model Context Protocol) tool integration across 4 external MCP servers providing **150+ tools**.

### Key Metrics

| Component | Count | Status |
|-----------|-------|--------|
| Agents | 5 specialized + Coordinator | Active |
| MCP Servers | 4 external + 1 built-in | Connected |
| MCP Tools | 150+ | Available |
| Channels | Slack (active), API (active) | Production |
| Skills/Workflows | 16 deterministic + 14 skills | Registered |
| Test Coverage | 212 tests | 80%+ target |

---

## 1. System Architecture Overview

### 1.1 Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT CHANNELS                                    │
├──────────────────┬──────────────────┬────────────────────────────────────────┤
│  Slack Bot       │  FastAPI Server  │           (Teams, Web - planned)       │
│  Socket Mode     │  REST + SSE      │                                        │
└────────┬─────────┴────────┬─────────┴───────────────────┬────────────────────┘
         └──────────────────┴───────────┬─────────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │   ASYNC RUNTIME (Shared Loop) │
                        │   Persistent MCP connections   │
                        └───────────────┬───────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │   LANGGRAPH COORDINATOR        │
                        │   8 Nodes, Intent → Route      │
                        └───────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
     ┌────────────────┐     ┌──────────────────┐     ┌──────────────────┐
     │   WORKFLOW     │     │   PARALLEL       │     │   AGENT          │
     │   (≥0.80 conf) │     │   ORCHESTRATOR   │     │   DELEGATION     │
     │   16 workflows │     │   3-5 agents     │     │   (≥0.60 conf)   │
     └────────┬───────┘     └────────┬─────────┘     └────────┬─────────┘
              │                      │                         │
              └──────────────────────┼─────────────────────────┘
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          │                      │                               │
          ▼                      ▼                               ▼
     ┌─────────┐          ┌─────────┐          ┌─────────────────────────┐
     │   DB    │          │ FinOps  │          │   Infrastructure        │
     │Troublesh│          │  Agent  │          │   Security, Log         │
     └────┬────┘          └────┬────┘          └────────────┬────────────┘
          └────────────────────┼────────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │         TOOL CATALOG            │
              │   Domain routing, aliases, tiers │
              └────────────────┬────────────────┘
                               │
     ┌─────────────────────────┼─────────────────────────┐
     │              │          │          │              │
     ▼              ▼          ▼          ▼              ▼
┌─────────┐  ┌──────────┐  ┌───────┐  ┌────────┐  ┌──────────┐
│oci-     │  │database- │  │finops │  │oci-mcp │  │oci-      │
│unified  │  │observatory│  │ai-mcp │  │security│  │infra     │
└─────────┘  └──────────┘  └───────┘  └────────┘  └──────────┘
     │              │          │          │              │
     └──────────────┴──────────┴──────────┴──────────────┘
                               │
              ┌────────────────▼────────────────┐
              │   ORACLE CLOUD INFRASTRUCTURE   │
              │   + Multicloud (AWS, Azure, GCP) │
              └─────────────────────────────────┘
```

### 1.2 Coordinator Node Flow

```
input → classifier → router → [workflow|parallel|agent] → action* → output
                        │
                        ├─→ workflow (≥0.80) → output
                        ├─→ parallel (2+ domains) → output
                        ├─→ agent (≥0.60) → action ←→ agent → output
                        └─→ escalate (<0.30) → output
```

---

## 2. Complete Agent Inventory

### 2.1 Active Agents (5 + Coordinator)

| Agent | Role ID | Domain | MCP Servers | Capabilities |
|-------|---------|--------|-------------|--------------|
| **Coordinator** | `coordinator` | routing | all | Intent classification, workflow routing, parallel orchestration |
| **DB Troubleshoot** | `db-troubleshoot-agent` | database | database-observatory | 11 capabilities, 4 skills, 30+ tools |
| **Infrastructure** | `infrastructure-agent` | compute, network | oci-unified, oci-infra | 9 capabilities, 4 skills, 20+ tools |
| **FinOps** | `finops-agent` | cost, budget | finopsai-mcp, oci-unified | 11 capabilities, 33+ multicloud tools |
| **Security Threat** | `security-threat-agent` | security, IAM | oci-mcp-security | 13 capabilities, MITRE mapping, 41+ tools |
| **Log Analytics** | `log-analytics-agent` | observability | oci-unified, database-observatory | 9 capabilities, pattern detection |

### 2.2 Agent Capabilities Matrix

| Agent | Primary Capabilities |
|-------|---------------------|
| DB Troubleshoot | database-analysis, performance-diagnostics, sql-tuning, blocking-analysis, wait-event-analysis, awr-analysis, ash-analysis, opsi-diagnostics, autonomous-db-management, db-system-management, mysql-management |
| Infrastructure | compute-management, network-management, storage-management, resource-scaling, vcn-analysis, instance-troubleshooting, security-list-analysis, subnet-management, instance-lifecycle |
| FinOps | cost-analysis, budget-tracking, optimization, anomaly-detection, usage-forecasting, commitment-tracking, rightsizing, tag-coverage, sustainability, k8s-cost-allocation |
| Security Threat | threat-detection, compliance-monitoring, security-posture, mitre-mapping, cloud-guard-analysis, vulnerability-scanning, bastion-management, waf-analysis, kms-management, data-safe-analysis, iam-analysis, policy-analysis, security-audit |
| Log Analytics | log-search, pattern-detection, log-correlation, audit-analysis, anomaly-detection, trace-correlation, error-pattern-analysis, temporal-analysis |

### 2.3 Skills/Workflows per Agent

| Agent | Skills | Description |
|-------|--------|-------------|
| DB Troubleshoot | `db_rca_workflow` (7 steps, 180s), `db_health_check_workflow` (3 steps, 60s), `db_sql_analysis_workflow` (5 steps, 300s), `db_awr_report_workflow` (4 steps, 180s) | Database RCA, health checks, SQL analysis, AWR reports |
| Infrastructure | `infra_inventory_workflow` (120s), `infra_instance_management_workflow` (150s), `infra_network_analysis_workflow` (180s), `infra_security_audit_workflow` (300s) | Inventory, lifecycle, network, security |
| FinOps | `cost_analysis_workflow` (90s), `cost_summary_workflow` | Spending analysis, optimization |
| Security Threat | `threat_hunting_workflow`, `compliance_check`, `security_assessment`, `incident_analysis` | Threat hunting, compliance, assessment |
| Log Analytics | `log_search_workflow`, `pattern_analysis`, `correlation_analysis`, `audit_review` | Log search, patterns, correlation |

---

## 3. Complete MCP Tool Inventory

### 3.1 MCP Server Summary

| Server | Location | Tools | Domains | Transport | Timeout |
|--------|----------|-------|---------|-----------|---------|
| **oci-unified** | `src/mcp/server/` | 28 | identity, compute, network, cost, security, observability, discovery | stdio | 60-180s |
| **database-observatory** | `/dev/MCP/mcp-oci-database-observatory` | 30+ | database, opsi, logan | stdio | 60s |
| **finopsai-mcp** | `/dev/MCP/finopsai-mcp` | 33 | cost, finops, anomaly, k8s, sustainability | stdio | 120s |
| **oci-mcp-security** | `/dev/MCP/oci-mcp-security` | 41+ | security, cloudguard, vss, bastion, waf, kms, datasafe | stdio | 30-180s |
| **oci-infrastructure** | `/dev/MCP/mcp-oci` | 44 | compute, network, database, security (fallback) | stdio | 60s |

**Total: 150+ unique tools**

### 3.2 Tool Naming Convention (STANDARDIZED)

```
oci_{domain}_{action}[_{resource}]

Examples:
- oci_compute_list_instances
- oci_database_execute_sql
- oci_opsi_get_fleet_summary
- oci_cost_get_summary
- oci_security_cloudguard_list_problems
- finops_detect_anomalies (multicloud)
```

### 3.3 Tool Aliases (Backward Compatibility)

Located in `src/mcp/catalog.py`:

| Legacy Name | Standard Name |
|-------------|---------------|
| `execute_sql` | `oci_database_execute_sql` |
| `get_fleet_summary` | `oci_opsi_get_fleet_summary` |
| `analyze_cpu_usage` | `oci_opsi_analyze_cpu` |
| `list_instances` | `oci_compute_list_instances` |
| `list_autonomous_databases` | `oci_database_list_autonomous` |

### 3.4 Domain Prefix Mapping

```python
DOMAIN_PREFIXES = {
    "database": ["oci_database_", "oci_opsi_", "execute_sql"],
    "infrastructure": ["oci_compute_", "oci_network_", "oci_list_"],
    "finops": ["oci_cost_", "finops_"],
    "security": ["oci_security_"],
    "observability": ["oci_observability_", "oci_logan_"],
    "identity": ["oci_list_compartments", "oci_get_compartment", "oci_get_tenancy"],
}
```

### 3.5 Tool Tier Classification

| Tier | Latency | Risk | Examples | Confirmation |
|------|---------|------|----------|--------------|
| **1** | <100ms | None | `oci_ping`, `search_capabilities`, `oci_discovery_cache_status` | No |
| **2** | 100ms-1s | Low | `oci_compute_list_instances`, `oci_cost_get_summary`, `oci_list_compartments` | No |
| **3** | 1-30s | Medium | `oci_opsi_analyze_cpu`, `oci_observability_query_logs`, `finops_detect_anomalies` | No |
| **4** | 5-15s+ | High | `oci_compute_stop_instance`, `oci_compute_restart_instance` | **Yes** |

---

## 4. Complete Tool Lists by Server

### 4.1 oci-unified (28 tools)

**Identity:**
- `oci_list_compartments` - List compartments in tenancy
- `oci_get_compartment` - Get specific compartment details
- `oci_search_compartments` - Search compartments by name pattern
- `oci_get_tenancy` - Get current tenancy info
- `oci_list_regions` - List subscribed regions

**Compute (Tier 2-4):**
- `oci_compute_list_instances` - List instances in compartment
- `oci_compute_get_instance` - Get instance details
- `oci_compute_find_instance` - Find instance by name
- `oci_compute_start_instance` (Tier 4) - Start stopped instance
- `oci_compute_stop_instance` (Tier 4) - Stop instance
- `oci_compute_restart_instance` (Tier 4) - Restart instance
- `oci_compute_start_by_name` (Tier 4)
- `oci_compute_stop_by_name` (Tier 4)
- `oci_compute_restart_by_name` (Tier 4)

**Network:**
- `oci_network_list_vcns` - List VCNs
- `oci_network_list_subnets` - List subnets
- `oci_network_list_security_lists` - List security lists

**Cost (30s timeout):**
- `oci_cost_get_summary` - Monthly cost breakdown

**Security:**
- `oci_security_list_users` - List IAM users

**Discovery:**
- `oci_discovery_run` - Run ShowOCI discovery
- `oci_discovery_get_cached` - Get cached resources
- `oci_discovery_refresh` - Refresh cache
- `oci_discovery_summary` - Resource summary
- `oci_discovery_search` - Search cached resources
- `oci_discovery_cache_status` - Cache health

**Meta:**
- `search_capabilities` - Tool discovery
- `set_feedback`, `append_feedback`, `get_feedback` - Feedback directives

### 4.2 database-observatory (30+ tools)

**SQLcl (Tier 3):**
- `oci_database_execute_sql` / `execute_sql` - Execute SQL
- `oci_database_get_schema` / `get_schema_info` - Schema metadata
- `oci_database_list_connections` / `list_connections` - List connections
- `oci_database_get_status` / `database_status` - DB status

**OPSI Discovery (Tier 1-2):**
- `oci_opsi_get_fleet_summary` / `get_fleet_summary` - Fleet overview
- `oci_opsi_search_databases` / `search_databases` - Find databases
- `oci_opsi_list_insights` / `list_database_insights` - List insights

**OPSI Analytics (Tier 2-3):**
- `oci_opsi_analyze_cpu` / `analyze_cpu_usage` - CPU analysis
- `oci_opsi_analyze_memory` / `analyze_memory_usage` - Memory analysis
- `oci_opsi_analyze_io` / `analyze_io_usage` - I/O analysis
- `oci_opsi_get_performance_summary` - Combined metrics
- `oci_opsi_get_sql_statistics` / `get_sql_statistics` - SQL stats
- `oci_opsi_analyze_wait_events` / `analyze_wait_events` - Wait events
- `oci_opsi_get_blocking_sessions` / `get_blocking_sessions` - Blocking
- `oci_opsi_compare_awr` / `compare_awr_periods` - AWR comparison
- `query_warehouse_standard` - OPSI warehouse queries

**Database System:**
- `list_tablespaces`, `list_users`, `get_sql_plan`, `list_awr_snapshots`

**SQLWatch:**
- `sqlwatch_get_plan_history` - Plan regression history
- `sqlwatch_analyze_regression` - Regression analysis

### 4.3 finopsai-mcp (33 tools - Multicloud)

**Legacy OCI (13 tools):**
- `oci_cost_ping` - Connectivity check
- `oci_cost_templates` - Template catalog
- `oci_cost_by_compartment` - Daily cost by compartment
- `oci_cost_service_drilldown` - Top N services
- `oci_cost_by_tag` - Cost by defined tag
- `oci_cost_monthly_trend` - Month-over-month trend
- `oci_cost_budget_status` - Budget monitoring
- `oci_cost_object_storage` - Storage costs
- `oci_cost_unit_cost` - Per-unit analysis
- `oci_cost_forecast_credits` - Credit forecast
- `oci_cost_focus_health` - FOCUS ETL validation
- `oci_cost_spikes` - Cost spike detection
- `oci_cost_schedules` - Cost schedules

**Multicloud Core (20 tools):**
- `finops_list_providers` - List enabled providers (OCI, AWS, Azure, GCP)
- `finops_cost_summary` - Unified cross-cloud summary
- `finops_discover_capabilities` - MCP tool discovery
- `finops_detect_anomalies` - Anomaly detection (z-score, IQR, rolling, isolation forest)
- `finops_anomaly_details` - Root cause analysis
- `finops_list_commitments` - RI/SP/Credits tracking
- `finops_commitment_alerts` - Expiring/underutilized alerts
- `finops_rightsizing` - Compute rightsizing
- `finops_database_rightsizing` - Database rightsizing
- `finops_cost_by_tags` - Tag-based allocation
- `finops_tag_coverage` - Tagging compliance
- `finops_shared_costs` - Shared cost distribution
- `finops_carbon_footprint` - CO2e emissions
- `finops_sustainability_recommendations` - Green recommendations
- `finops_k8s_cluster_costs` - Kubernetes cluster costs
- `finops_k8s_namespace_costs` - Namespace allocation
- `finops_k8s_workload_efficiency` - Pod efficiency
- `finops_coordinator_status` - Coordinator integration
- `finops_register_with_coordinator` - Auto-registration
- `finops_session_context` - Cross-tool state

### 4.4 oci-mcp-security (41+ tools)

**Cloud Guard (7 tools):**
- `oci_security_cloudguard_list_problems` - List security problems
- `oci_security_cloudguard_get_problem` - Problem details
- `oci_security_cloudguard_remediate_problem` - Execute remediation
- `oci_security_cloudguard_list_detectors` - List detectors
- `oci_security_cloudguard_list_responders` - List responders
- `oci_security_cloudguard_get_security_score` - Security score (0-100)
- `oci_security_cloudguard_list_recommendations` - Recommendations

**VSS Vulnerability Scanning (4 tools):**
- `oci_security_vss_list_host_scans` - Host vulnerability scans
- `oci_security_vss_get_host_scan` - Scan results
- `oci_security_vss_list_container_scans` - Container scans
- `oci_security_vss_list_vulnerabilities` - CVE list by severity

**Data Safe (4 tools):**
- `oci_security_datasafe_list_targets` - Database targets
- `oci_security_datasafe_list_assessments` - Security assessments
- `oci_security_datasafe_get_assessment` - Assessment details
- `oci_security_datasafe_list_findings` - Security findings

**WAF (4 tools):**
- `oci_security_waf_list_firewalls` - List WAF instances
- `oci_security_waf_get_firewall` - WAF config
- `oci_security_waf_list_policies` - WAF policies
- `oci_security_waf_get_policy` - Policy rules

**Security Zones (3 tools):**
- `oci_security_zones_list` - List zones
- `oci_security_zones_get` - Zone config
- `oci_security_zones_list_policies` - Zone policies

**Bastion (4 tools):**
- `oci_security_bastion_list` - List bastions
- `oci_security_bastion_get` - Bastion config
- `oci_security_bastion_list_sessions` - Active sessions
- `oci_security_bastion_terminate_session` - Terminate session

**KMS (4 tools):**
- `oci_security_kms_list_vaults` - List vaults
- `oci_security_kms_get_vault` - Vault details
- `oci_security_kms_list_keys` - List keys
- `oci_security_kms_get_key` - Key metadata

**IAM & Access (6 tools):**
- `oci_security_list_users` - IAM users
- `oci_security_get_user` - User details
- `oci_security_list_groups` - IAM groups
- `oci_security_list_policies` - IAM policies
- `oci_security_get_security_assessment` - Security posture
- `oci_security_audit` - Full audit (Tier 3)

**Audit (2 tools):**
- `oci_security_audit_list_events` - Audit events
- `oci_security_audit_get_configuration` - Audit config

**High-Level Skills (3 tools):**
- `oci_security_skill_posture_summary` - Executive summary
- `oci_security_skill_vulnerability_overview` - Risk assessment
- `oci_security_skill_audit_digest` - Audit digest

---

## 5. Coordinator Workflows (16 Deterministic)

### 5.1 Workflow Registry

| Workflow | Intent Aliases | Tool | Response Time |
|----------|----------------|------|---------------|
| `list_compartments` | show_compartments, get_compartments | `oci_list_compartments` | <1s |
| `get_tenancy` | tenancy_info, show_tenancy | `oci_get_tenancy` | <1s |
| `list_regions` | show_regions, available_regions | `oci_list_regions` | <1s |
| `list_instances` | show_instances, get_instances | `oci_compute_list_instances` | 2-3s |
| `get_instance` | instance_details, describe_instance | `oci_compute_get_instance` | 1-2s |
| `list_vcns` | show_networks, get_vcns | `oci_network_list_vcns` | 1-2s |
| `list_subnets` | show_subnets, get_subnets | `oci_network_list_subnets` | 1-2s |
| `cost_summary` | **10 aliases**: get_costs, show_spending, tenancy_costs, how_much_spent, monthly_cost, get_tenancy_costs, spending, show_costs, monthly_costs, get_cost_summary | `oci_cost_get_summary` | 5-30s |
| `discovery_summary` | resource_summary | `oci_discovery_summary` | 5-10s |
| `search_resources` | find_resource, resource_search | `oci_discovery_search` | 5-10s |
| `search_capabilities` | capabilities, what_can_you_do, help | `search_capabilities` | <1s |

### 5.2 Routing Thresholds

```python
WORKFLOW_CONFIDENCE_THRESHOLD = 0.80   # High → deterministic workflow
AGENT_CONFIDENCE_THRESHOLD = 0.60      # Medium → agent delegation
PARALLEL_DOMAIN_THRESHOLD = 2          # Multi-domain → parallel execution
ESCALATION_CONFIDENCE_THRESHOLD = 0.30 # Low → escalate/clarify
```

### 5.3 Routing Decision Logic

```
IF confidence >= 0.80 AND workflow_match:
    → Execute deterministic workflow (no LLM, fast)
ELIF domains >= 2 AND category IN [ANALYSIS, TROUBLESHOOT]:
    → Parallel orchestrator (3-5 agents concurrent)
ELIF confidence >= 0.60 AND agent_match:
    → Delegate to specialized agent
ELIF confidence < 0.30:
    → Escalate (ask clarifying question)
ELSE:
    → Direct LLM response with tools
```

---

## 6. Memory & Caching Architecture

### 6.1 Tiered Memory Layer

| Layer | Backend | Purpose | TTL |
|-------|---------|---------|-----|
| **Hot Cache** | Redis | Session state, tool results | 1-4 hours |
| **Checkpoints** | LangGraph MemorySaver | Graph state snapshots | Session |

### 6.2 OCI Resource Cache Features

- **Tag-Based Invalidation**: Atomic group invalidation via Redis sets
- **Stale-While-Revalidate**: Return stale data, refresh in background
- **Pub/Sub Invalidation**: Multi-instance cache coordination
- **Discovery Integration**: ShowOCI-based resource caching

### 6.3 Context Compression

For long conversations (>150k tokens):
- Sliding window with LLM-based summarization
- Keep 10 most recent messages uncompressed
- Heuristic fallback (entity/decision extraction)

---

## 7. Observability Integration

### 7.1 OCI APM (Tracing)

- OpenTelemetry spans for all operations
- Custom Zipkin exporter for OCI APM
- Per-agent service names: `oci-{agent}-agent`
- Trace attributes: tool calls, durations, errors

### 7.2 OCI Logging (Structured)

- Per-agent log streams (`OCI_LOG_ID_{AGENT}`)
- Trace ID correlation: `trace_id`, `span_id`
- Batch flushing (50 entries / 5 seconds)
- JSON structured format

### 7.3 LLM-as-a-Judge Evaluation

Evaluation criteria:
- **Correctness** (40%): Intent, routing, domains, tools
- **Quality** (25%): Relevance, completeness, accuracy
- **Safety** (25%): Hallucinations, harmful actions, privacy
- **Efficiency** (10%): Latency, minimal tool calls

Targets:
- 85%+ task success rate
- 70%+ workflow routing ratio

---

## 8. Identified Issues & Recommendations

### 8.1 RESOLVED: Naming Standardization

**Status**: ✅ Complete (2026-01-01)

- Tool aliases implemented in `catalog.py`
- Domain prefixes for dynamic discovery
- Backward compatibility maintained

### 8.2 RESOLVED: LangGraph Integration

**Status**: ✅ Complete (2025-12-31)

- Slack handler uses LangGraph coordinator (`USE_LANGGRAPH_COORDINATOR=true`)
- Fallback to keyword routing if coordinator fails
- ToolConverter for MCP → LangChain conversion

### 8.3 IN PROGRESS: External MCP Server Tool Names

**Status**: 🔄 Partial

Some tools in external servers still use legacy names:
- database-observatory: `execute_sql` (should be `oci_database_execute_sql`)
- Aliases provide backward compatibility, but native names should be updated

**Recommendation**: Update external MCP servers to use standardized naming

### 8.4 ENHANCEMENT: Multicloud Cost Integration

**Status**: 📋 Planned

The finopsai-mcp provides multicloud support (AWS, Azure, GCP), but only OCI is fully tested.

**Recommendation**: Enable and test AWS/Azure/GCP providers

### 8.5 ENHANCEMENT: Security Agent Coverage

**Status**: 📋 Planned

41+ security tools available, but agent only uses ~10 actively.

**Recommendation**: Expand Security Threat Agent skills to cover full tool inventory

---

## 9. Action Items

### Immediate (Complete)
- [x] Tool alias resolution in catalog
- [x] Domain-based tool discovery
- [x] Slack → LangGraph coordinator integration
- [x] Parallel orchestrator implementation
- [x] Context compression
- [x] Cost tool 30-second timeout

### Short Term (In Progress)
- [ ] Update database-observatory tool names to `oci_` prefix
- [ ] Expand Security Agent skills coverage
- [ ] Add remaining deterministic workflows
- [ ] Microsoft Teams integration

### Long Term (Roadmap)
- [ ] Enable multicloud cost providers
- [ ] Web UI dashboard
- [ ] OKE deployment manifests
- [ ] Full RAG implementation with OCI docs

---

## Appendix A: MITRE ATT&CK Mapping

| MITRE ID | Technique | Cloud Guard Problem | Tactic |
|----------|-----------|---------------------|--------|
| T1078 | Valid Accounts | SUSPICIOUS_LOGIN | Initial Access |
| T1098 | Account Manipulation | IAM_POLICY_CHANGE | Persistence |
| T1110 | Brute Force | BRUTEFORCE_ATTEMPTS | Credential Access |
| T1190 | Exploit Public-Facing | PUBLIC_EXPOSURE | Initial Access |
| T1485 | Data Destruction | RESOURCE_DELETION | Impact |
| T1496 | Resource Hijacking | CRYPTO_MINING | Impact |
| T1530 | Data from Cloud Storage | UNAUTHORIZED_ACCESS | Collection |
| T1537 | Transfer to Cloud | DATA_EXFILTRATION | Exfiltration |
| T1562 | Impair Defenses | SECURITY_GROUP_CHANGE | Defense Evasion |
| T1567 | Exfiltration Over Web | DATA_EXFILTRATION | Exfiltration |

---

## Appendix B: Environment Variables Reference

```bash
# Core
OCI_CONFIG_FILE=~/.oci/config
OCI_CLI_PROFILE=DEFAULT
LLM_PROVIDER=oracle_code_assist  # or anthropic, openai, lm_studio

# Slack
SLACK_BOT_TOKEN=xoxb-xxx
SLACK_APP_TOKEN=xapp-xxx

# Coordinator
USE_LANGGRAPH_COORDINATOR=true

# Caching
REDIS_URL=redis://localhost:6379
SHOWOCI_CACHE_ENABLED=true
SHOWOCI_REFRESH_HOURS=4

# Observability
OCI_APM_ENDPOINT=https://xxx.apm-agt.region.oci.oraclecloud.com
OCI_APM_PRIVATE_DATA_KEY=xxx
OCI_LOG_GROUP_ID=ocid1.loggroup.oc1...
OCI_LOGGING_ENABLED=true

# RAG
OCI_COMPARTMENT_ID=ocid1.compartment.oc1...
OCI_GENAI_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com
```

---

*End of Code Review Document*
*Updated: 2026-01-01 by Claude Code (Opus 4.5)*
