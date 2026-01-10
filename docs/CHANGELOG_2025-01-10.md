# OCI Coordinator Agent Enhancement Changelog

## Date: 2025-01-10

This document summarizes the bug fixes and enhancements made to address issues identified in agent operations.

---

## Issues Addressed

| Issue | Status | Description |
|-------|--------|-------------|
| Database listing error | Fixed | `'dict' object has no attribute 'upper'` error in database name resolution |
| Compartment filtering | Fixed | Filtering by name prefix (e.g., "starting with B") not working |
| Missing security_threats workflow | Fixed | Intent detected but no workflow registered |
| Bulkhead timeout errors | Fixed | Cost queries timing out due to 10s acquire timeout |
| Fuzzy compartment matching | Added | Suggestions for misspelled compartment names |
| Paginated output | Added | Large result sets split into 25-item pages |
| Force refresh capability | Added | Bypass cache for "latest data" requests |
| Daily cache refresh | Added | Scheduled background refresh of compartments |

---

## 1. Bug Fixes

### 1.1 Database Listing Error Fix
**File:** `src/agents/coordinator/workflows.py` (lines ~493-502)

**Problem:** The `list_databases_workflow` failed with `'dict' object has no attribute 'upper'` because OCI API sometimes returns nested dict structures for database names.

**Solution:** Added type checking and safe extraction before calling `.upper()`:
```python
# Safely extract database name - handle nested dicts and None values
raw_name = db.get("name") or db.get("database_name") or db.get("display_name") or ""
# Handle case where value is a dict (nested structure from API)
if isinstance(raw_name, dict):
    raw_name = raw_name.get("name") or raw_name.get("value") or str(raw_name)
db_name = str(raw_name).upper() if raw_name else ""
```

### 1.2 Bulkhead Timeout Fix
**Files:**
- `src/resilience/bulkhead.py` - Added partition-specific timeouts
- `src/mcp/catalog.py` - Updated to use dynamic timeouts

**Problem:** Cost comparison queries failing with "Resource pool cost exhausted (bulkhead timeout)" because the hardcoded 10-second acquire timeout was too short for slow operations like Usage API calls (which can take 60+ seconds).

**Solution:** Implemented partition-specific acquire timeouts:
```python
PARTITION_ACQUIRE_TIMEOUTS = {
    BulkheadPartition.DATABASE: 30.0,      # SQL ops can take 30-60s
    BulkheadPartition.INFRASTRUCTURE: 15.0,  # Compute/network are fairly quick
    BulkheadPartition.COST: 45.0,          # Usage API is notoriously slow (60s+)
    BulkheadPartition.SECURITY: 20.0,      # Cloud Guard queries are moderate
    BulkheadPartition.DISCOVERY: 15.0,     # List/search are usually quick
    BulkheadPartition.LLM: 60.0,           # LLM calls can be very slow
    BulkheadPartition.DEFAULT: 10.0,       # Default: fail fast
}
```

---

## 2. New Features

### 2.1 Compartment Filtering with Fuzzy Matching
**File:** `src/agents/coordinator/workflows.py` (lines ~532-796)

Added `_extract_compartment_filter()` function supporting:
- **Prefix filtering:** "compartments starting with B"
- **Contains filtering:** "compartments containing finance"
- **Exact name matching:** "compartment named production"
- **Lifecycle state filtering:** "active compartments"

Added `_fuzzy_match_compartment()` function for misspelled names:
- Uses `difflib.SequenceMatcher` with 0.6 similarity threshold
- Returns top 5 suggestions sorted by similarity score

### 2.2 Paginated Table Output
**File:** `src/agents/coordinator/workflows.py`

Enhanced `list_compartments_workflow` with pagination:
- Default page size: 25 items
- Page navigation info in response
- Total count displayed

### 2.3 Security Threats Workflow
**File:** `src/agents/coordinator/workflows.py` (lines ~4066-4287)

Added comprehensive `security_threats_workflow` including:
- Cloud Guard problems (CRITICAL/HIGH severity)
- Security score assessment
- Audit event analysis
- **MITRE ATT&CK technique mapping**
- Prioritized recommendations

Registered in WORKFLOW_REGISTRY with aliases:
- `security_threats`, `threat_analysis`, `threat_detection`
- `show_threats`, `list_threats`, `mitre_analysis`, `threat_intelligence`

### 2.4 Force Refresh Capability
**Files:**
- `src/agents/coordinator/workflows.py` - Workflows detect "latest"/"refresh" keywords
- `src/cache/oci_resource_cache.py` - `force_refresh()` method

Trigger words: "latest", "refresh", "current", "real-time", "force"

### 2.5 Daily Compartment Cache Refresh
**File:** `src/cache/oci_resource_cache.py` (lines ~1090-1287)

Added scheduled cache refresh system:
```python
# Configuration
COMPARTMENT_TTL = timedelta(hours=24)  # Daily TTL
COMPARTMENT_REFRESH_INTERVAL = timedelta(hours=24)  # Daily refresh
RESOURCE_REFRESH_INTERVAL = timedelta(hours=4)  # Resources every 4 hours

# Methods
await cache.start_scheduled_refresh()  # Start background scheduler
await cache.stop_scheduled_refresh()   # Stop scheduler
await cache.force_refresh()            # Immediate refresh
cache.is_scheduler_running()           # Check scheduler status
```

Features:
- Initial cache warmup on startup
- Automatic stale detection
- Background refresh loop
- Graceful shutdown

---

## 3. Configuration Changes

### 3.1 Updated TTLs
| Setting | Old Value | New Value | Reason |
|---------|-----------|-----------|--------|
| `COMPARTMENT_TTL` | 4 hours | 24 hours | Match daily refresh cycle |
| Bulkhead COST timeout | 10s | 45s | Usage API needs time |
| Bulkhead DATABASE timeout | 10s | 30s | SQL queries can be slow |

---

## 4. Testing Verification

All changes validated with:
```bash
# Bulkhead timeout configuration
poetry run python -c "from src.resilience.bulkhead import ..."

# Cache configuration
poetry run python -c "from src.cache.oci_resource_cache import ..."

# Catalog import
poetry run python -c "from src.mcp.catalog import get_partition_timeout..."
```

---

## 5. Files Modified

| File | Type | Changes |
|------|------|---------|
| `src/agents/coordinator/workflows.py` | Modified | DB name fix, compartment filtering, security_threats workflow |
| `src/resilience/bulkhead.py` | Modified | Added PARTITION_ACQUIRE_TIMEOUTS, get_partition_timeout() |
| `src/mcp/catalog.py` | Modified | Use partition-specific timeouts |
| `src/cache/oci_resource_cache.py` | Modified | Added scheduled refresh, force_refresh(), updated TTLs |

---

## 6. Usage Examples

### Compartment Filtering
```
User: "Show me compartments starting with B"
User: "List compartments containing finance"
User: "Show active compartments"
```

### Security Threats
```
User: "Show security threats"
User: "Analyze threats with MITRE mapping"
User: "What are our security issues?"
```

### Force Refresh
```
User: "Show me the latest compartments"
User: "Refresh and show databases"
User: "Get current cost data"
```

---

## 7. Session 2: MCP Server & Observability Enhancements (2026-01-10)

### 7.1 MCP Servers Re-enabled
All external MCP servers now enabled with OpenTelemetry:
- `database-observatory` - SQLcl/Logan queries
- `oci-infrastructure` - Full OCI SDK wrapper
- `finopsai` - Multicloud FinOps
- `oci-mcp-security` - Cloud Guard, WAF, KMS, Bastion

Configuration via environment variables:
```bash
DB_OBSERVATORY_PATH=/path/to/mcp-oci-database-observatory
OCI_INFRASTRUCTURE_PATH=/path/to/mcp-oci
FINOPSAI_PATH=/path/to/finopsai-mcp
OCI_SECURITY_PATH=/path/to/oci-mcp-security
```

### 7.2 Log Analytics (Logan) Tools Added
5 new tools with multi-tenancy support:
- `oci_logan_list_namespaces` - List Log Analytics namespaces per OCI profile
- `oci_logan_list_log_groups` - List log groups
- `oci_logan_get_summary` - Storage usage, source counts
- `oci_logan_execute_query` - Execute Log Analytics queries
- `oci_logan_search_logs` - Text search in logs

### 7.3 Instance Metrics Workflow
New interactive workflow for instance metrics:
- Prompts for instance name if not provided
- Retrieves CPU, memory, network, disk metrics
- Uses `oci_observability_get_instance_metrics` tool

### 7.4 Anomaly Detection Workflow
New workflow correlating metrics and logs:
- Gets instance metrics (CPU, memory, network, disk)
- Queries logs for the same time period
- Analyzes patterns to identify anomalies
- Correlates metric spikes with log events

### 7.5 LLM Observability with GenAI Semantic Conventions
Enhanced OpenTelemetry integration for LLM calls:
- `OracleCodeAssistInstrumentor` class for OCA tracing
- GenAI semantic conventions (gen_ai.usage.input_tokens, etc.)
- Token usage, latency, and error tracking

### 7.6 Documentation Updates
- **CLAUDE.md**: Updated to 7 agents, 5 MCP servers, added Agents Overview table
- **docs/agents.md**: Created comprehensive agent implementation guide
- **docs/gemini.md**: Created LLM provider configuration guide
- **docs/FEATURE_MAPPING.md**: Updated with v1.5 features

### Files Modified in Session 2
| File | Changes |
|------|---------|
| `config/mcp_servers.yaml` | Re-enabled all MCP servers, OTEL enabled |
| `src/mcp/server/tools/logan.py` | New Logan tools (5 tools) |
| `src/mcp/server/tools/observability.py` | Instance metrics tool |
| `src/agents/coordinator/workflows.py` | anomaly_detection_workflow |
| `src/observability/llm_tracing.py` | OracleCodeAssistInstrumentor, GenAI conventions |
| `src/observability/__init__.py` | Exports for new classes |
| `src/llm/oca.py` | Integrated OracleCodeAssistInstrumentor |
| `CLAUDE.md` | Updated summary, agents, MCP servers |
| `docs/agents.md` | New - Agent implementation guide |
| `docs/gemini.md` | New - LLM provider configuration |
| `docs/FEATURE_MAPPING.md` | Updated with v1.5 features |

---

## 8. Architecture Summary

### Agents (7)
1. DbTroubleshootAgent - Database performance, AWR, blocking
2. LogAnalyticsAgent - Log queries, patterns, anomaly correlation
3. SecurityThreatAgent - Cloud Guard, MITRE mapping
4. FinOpsAgent - Cost analysis, optimization
5. InfrastructureAgent - Compute, network, storage
6. ErrorAnalysisAgent - Error classification, RCA
7. SelectAIAgent - NL2SQL, data chat

### MCP Servers (5)
1. oci-unified (77 tools) - Core OCI + DB Mgmt + OPSI + Logan
2. database-observatory (50+ tools) - SQLcl/Logan
3. oci-infrastructure (44 tools) - Full OCI SDK
4. finopsai (33 tools) - Multicloud FinOps
5. oci-mcp-security (60+ tools) - Comprehensive security

### Naming Conventions
- Tools: `oci_{domain}_{action}` (e.g., `oci_compute_list_instances`)
- Agents: `{domain}-agent` (e.g., `db-troubleshoot-agent`)
- Workflows: `{domain}_{action}_workflow` (e.g., `cost_summary_workflow`)

---

## 9. Future Recommendations

1. **Cost Query Investigation:** If cost queries still return $0, check OCI Usage API permissions and tenancy billing configuration.

2. **Full Test Suite:** Run complete test suite after changes stabilize:
   ```bash
   poetry run pytest --cov=src
   ```

3. **Start Cache Scheduler:** Add to application startup:
   ```python
   cache = OCIResourceCache.get_instance()
   await cache.initialize()
   await cache.start_scheduled_refresh()
   ```

4. **Configure External MCP Servers:** Set environment variables for external MCP server paths.

5. **Verify OCI APM Integration:** Test that OpenTelemetry traces are being sent to OCI APM.
