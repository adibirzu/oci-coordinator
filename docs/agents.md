# OCI AI Agent Implementation Guide

## Overview

This document provides implementation details for all agents in the OCI AI Coordinator system. Each agent follows the `BaseAgent` interface and integrates with the LangGraph-based coordinator for workflow orchestration.

## Agent Architecture

```
Coordinator (LangGraph)
    │
    ├── DbTroubleshootAgent     ─→ Database/OPSI/SQLcl
    ├── LogAnalyticsAgent       ─→ OCI Log Analytics
    ├── SecurityThreatAgent     ─→ Cloud Guard/Security
    ├── FinOpsAgent             ─→ Cost/Budget APIs
    ├── InfrastructureAgent     ─→ Compute/Network/Storage
    ├── ErrorAnalysisAgent      ─→ Error Classification
    └── SelectAIAgent           ─→ NL2SQL/AI Orchestration
```

## Agent Catalog

All agents are auto-discovered from `src/agents/` and registered in the `AgentCatalog` singleton. The catalog provides:
- Domain-based lookup
- Capability-based routing
- Performance metrics tracking
- Health monitoring

```python
from src.agents.catalog import AgentCatalog, initialize_agents

# Initialize all agents
catalog = initialize_agents("src/agents")

# Find best agent for a domain
best = catalog.find_best_agent(domains=["database"])

# Get agent by capability
agents = catalog.get_by_capability("performance-diagnostics")
```

---

## Agent Implementations

### 1. DbTroubleshootAgent

**Location**: `src/agents/database/troubleshoot.py`

**Capabilities**:
- `database-analysis`, `performance-diagnostics`, `sql-tuning`
- `blocking-analysis`, `wait-event-analysis`, `awr-analysis`, `ash-analysis`

**Skills/Workflows**:
- `rca_workflow` - Root Cause Analysis
- `blocking_analysis` - Blocking session detection
- `wait_event_analysis` - Wait event diagnostics
- `awr_analysis` - AWR report generation
- `sql_monitoring` - Active SQL monitoring
- `long_running_ops` - Long operation tracking
- `parallelism_stats` - Parallel query statistics

**MCP Tools Used**:
```python
# DB Management API
oci_dbmgmt_list_databases
oci_dbmgmt_get_awr_report
oci_dbmgmt_get_wait_events
oci_dbmgmt_get_top_sql

# OPSI API
oci_opsi_search_databases
oci_opsi_get_fleet_summary
oci_opsi_analyze_cpu

# SQLcl (via database-observatory)
oci_database_execute_sql
```

**Self-Healing**: Yes (via `SelfHealingMixin`)

---

### 2. LogAnalyticsAgent

**Location**: `src/agents/log_analytics/agent.py`

**Capabilities**:
- `log-analysis`, `pattern-detection`, `metric-analysis`
- `trace-correlation`, `log-query-execution`

**Skills/Workflows**:
- `log_search` - Text search in logs
- `log_query` - Execute Log Analytics queries
- `log_summary` - Get log analytics summary
- `anomaly_detection` - Correlate logs with metrics

**MCP Tools Used**:
```python
oci_logan_list_namespaces
oci_logan_list_log_groups
oci_logan_get_summary
oci_logan_execute_query
oci_logan_search_logs
```

---

### 3. SecurityThreatAgent

**Location**: `src/agents/security/agent.py`

**Capabilities**:
- `threat-detection`, `compliance-monitoring`, `security-posture`
- `mitre-mapping`, `cloud-guard-analysis`

**Skills/Workflows**:
- `security_overview` - Cloud Guard summary
- `threat_analysis` - Active threat detection
- `compliance_check` - Compliance posture assessment

**MCP Tools Used**:
```python
oci_cloudguard_list_problems
oci_cloudguard_get_risk_score
oci_security_list_users
```

---

### 4. FinOpsAgent

**Location**: `src/agents/finops/agent.py`

**Capabilities**:
- `cost-analysis`, `budget-tracking`, `optimization`
- `anomaly-detection`, `usage-forecasting`

**Skills/Workflows**:
- `cost_summary` - Monthly cost overview
- `cost_by_service` - Service-level breakdown
- `cost_anomaly` - Spending anomaly detection
- `optimization_recommendations` - Cost optimization tips

**MCP Tools Used**:
```python
oci_cost_get_summary
oci_cost_get_by_service
oci_cost_get_trend
```

**Self-Healing**: Yes (via `SelfHealingMixin`)

---

### 5. InfrastructureAgent

**Location**: `src/agents/infrastructure/agent.py`

**Capabilities**:
- `compute-management`, `network-management`, `storage-management`
- `vcn-analysis`, `instance-troubleshooting`

**Skills/Workflows**:
- `list_instances` - List compute instances
- `instance_metrics` - Get instance metrics (CPU, memory, network)
- `network_overview` - VCN and subnet listing

**MCP Tools Used**:
```python
oci_compute_list_instances
oci_compute_find_instance
oci_compute_start_instance
oci_network_list_vcns
oci_network_list_subnets
oci_observability_get_instance_metrics
```

---

### 6. ErrorAnalysisAgent

**Location**: `src/agents/error_analysis/agent.py`

**Capabilities**:
- `error-classification`, `root-cause-analysis`
- `error-pattern-detection`

**Skills/Workflows**:
- `error_classify` - Classify error type
- `error_analyze` - Deep error analysis

---

### 7. SelectAIAgent

**Location**: `src/agents/selectai/agent.py`

**Capabilities**:
- `nl2sql`, `data-chat`, `text-summarization`
- `ai-agent-orchestration`, `natural-language-query`

**Skills/Workflows**:
- `nl2sql_query` - Convert natural language to SQL
- `data_chat` - Conversational data exploration
- `report_generation` - Generate reports from data

---

## Creating a New Agent

### Step 1: Create Agent Module

```python
# src/agents/myagent/agent.py
from src.agents.base import AgentDefinition, AgentMetadata, BaseAgent

class MyCustomAgent(BaseAgent):
    """Custom agent for specific domain."""

    @classmethod
    def get_definition(cls) -> AgentDefinition:
        return AgentDefinition(
            agent_id=f"my-agent-{uuid.uuid4().hex[:8]}",
            role="my-agent",
            capabilities=["my-capability", "another-capability"],
            skills=["my_workflow"],
            health_endpoint="http://localhost:8010/health",
            metadata=AgentMetadata(version="1.0.0"),
            description="My custom agent description",
        )

    async def run(self, query: str, **kwargs) -> str:
        """Execute agent logic."""
        # Your implementation here
        pass
```

### Step 2: Add Domain Mapping

Update `src/agents/catalog.py`:

```python
DOMAIN_CAPABILITIES = {
    # ...existing domains...
    "mydomain": [
        "my-capability",
        "another-capability",
    ],
}

DOMAIN_PRIORITY = {
    # ...existing priorities...
    "mydomain": {
        "my-agent": 100,
    },
}
```

### Step 3: Register Workflows

Update `src/agents/coordinator/workflows.py`:

```python
WORKFLOW_REGISTRY = {
    # ...existing workflows...
    "my_workflow": my_workflow_function,
}
```

---

## Agent Routing

The coordinator uses a scoring system to select the best agent:

| Factor | Weight | Description |
|--------|--------|-------------|
| Capability Match | 40% | How well agent capabilities match request |
| Domain Priority | 30% | Pre-configured domain expertise score |
| Health Score | 15% | Agent health status |
| Performance Score | 15% | Historical success rate |

---

## OpenTelemetry Integration

All agents should use the observability module for tracing:

```python
from opentelemetry import trace

_tracer = trace.get_tracer("my-agent")

class MyAgent(BaseAgent):
    async def run(self, query: str, **kwargs) -> str:
        with _tracer.start_as_current_span("my-agent.run") as span:
            span.set_attribute("query", query[:100])
            # ... agent logic ...
```

---

## Self-Healing Agents

Agents with `SelfHealingMixin` can automatically recover from errors:

```python
from src.agents.self_healing import SelfHealingMixin

class MyAgent(BaseAgent, SelfHealingMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_self_healing(
            max_retries=3,
            retry_delay=1.0,
        )
```

Features:
- Automatic retry with exponential backoff
- Parameter correction using LLM
- Fallback to alternative tools
- Error classification and reporting
