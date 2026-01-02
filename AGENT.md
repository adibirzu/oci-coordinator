# AGENT.md - Agent Architecture Documentation

## Overview

The OCI AI Agent Coordinator implements a **Workflow-First** multi-agent architecture using LangGraph. The system prioritizes deterministic workflows for known tasks (70%+ of requests), uses parallel orchestration for multi-domain queries, and falls back to agentic behavior for novel requests.

**Last Updated**: 2026-01-02

---

## Architecture Principles

### 1. Workflow-First Design
- **70%+ of requests** handled by deterministic workflows
- 16 pre-built workflows with 40+ intent aliases
- Agentic fallback only for novel or complex requests
- Parallel orchestration for multi-domain queries

### 2. Tool-Centric Approach
- Agents reason and plan; MCP tools execute
- 158+ tools across 4 MCP servers
- Unified naming: `oci_{domain}_{action}`
- 4-tier classification by latency and risk
- Progressive disclosure via `oci_search_tools`

### 3. State Persistence
- Conversation state via LangGraph checkpoints
- Tiered memory: Redis (hot) + ATP (persistent)
- Context compression for long conversations (>150k tokens)
- Thread-based session management

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT CHANNELS                                    │
├──────────────────┬──────────────────┬────────────────────────────────────────┤
│  Slack Bot       │  FastAPI         │           (Teams, Web - planned)       │
│  Socket Mode     │  REST + SSE      │                                        │
└────────┬─────────┴────────┬─────────┴───────────────────┬────────────────────┘
         └──────────────────┴───────────┬─────────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │     ASYNC RUNTIME             │
                        │   (Shared Event Loop)         │
                        └───────────────┬───────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │   LANGGRAPH COORDINATOR       │
                        │   8 Nodes: input→classifier   │
                        │   →router→workflow|parallel   │
                        │   |agent→action→output        │
                        └───────────────┬───────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │               │             │             │               │
          ▼               ▼             ▼             ▼               ▼
     ┌─────────┐    ┌─────────┐   ┌─────────┐   ┌─────────┐    ┌─────────┐
     │   DB    │    │   Log   │   │Security │   │ FinOps  │    │  Infra  │
     │Troublesh│    │Analytics│   │ Threat  │   │  Agent  │    │  Agent  │
     └────┬────┘    └────┬────┘   └────┬────┘   └────┬────┘    └────┬────┘
          └─────────────┴─────────────┼─────────────┴─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │      TOOL CATALOG         │
                        │   158+ Tools, 4 Tiers     │
                        │   Aliases + Discovery     │
                        └─────────────┬─────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │               │             │             │               │
        ▼               ▼             ▼             ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│oci-unified │  │database-obs │  │oci-infra   │  │  finopsai   │
│ (31 tools) │  │ (50+ tools) │  │ (44 tools) │  │ (33 tools)  │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
        │               │             │             │
        └───────────────┴─────────────┼─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   ORACLE CLOUD             │
                        │   INFRASTRUCTURE           │
                        └───────────────────────────┘

                            OBSERVABILITY
                    ┌───────────────────────────────┐
                    │  OCI APM  ←→  OCI Logging     │
                    │  (traces)    (structured)     │
                    │  trace_id correlation         │
                    │  LLM-as-Judge Evaluation      │
                    └───────────────────────────────┘
```

---

## Coordinator Graph

### Graph Nodes (8 Total)

| Node | Purpose | Implementation |
|------|---------|----------------|
| `input` | Normalize request, extract entities | `nodes.py:input_node` |
| `classifier` | Intent classification with confidence | `nodes.py:classifier_node` |
| `router` | Route based on thresholds | `nodes.py:router_node` |
| `workflow` | Execute deterministic workflow | `nodes.py:workflow_node` |
| `parallel` | Multi-agent parallel execution | `orchestrator.py` |
| `agent` | LLM reasoning with tool selection | `nodes.py:agent_node` |
| `action` | Execute MCP tool calls | `nodes.py:action_node` |
| `output` | Format response for channel | `nodes.py:output_node` |

### Routing Logic

```
input → classifier → router ─┬─→ workflow (confidence ≥ 0.80)
                             ├─→ parallel (2+ domains, conf ≥ 0.60)
                             ├─→ agent (confidence ≥ 0.60)
                             └─→ escalate (confidence < 0.30)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
         agent ←→ action (loop, max 15 iterations)       parallel
              │                                               │
              ▼                                               ▼
           output ←──────────────────────────────────────────┘
```

### Routing Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| Workflow | ≥ 0.80 | Direct workflow execution |
| Parallel | ≥ 0.60 + 2+ domains | Multi-agent parallel |
| Agent | ≥ 0.60 | Single agent with tools |
| Clarify | 0.30 - 0.60 | Ask clarifying question |
| Escalate | < 0.30 | Human handoff |

---

## Agents

### 1. Coordinator Agent

**Role**: Master orchestrator - classifies intent, routes requests, manages state

**Implementation**: `src/agents/coordinator/graph.py`

**State Schema**:
```python
class CoordinatorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    iteration: int                    # Current loop count
    max_iterations: int               # Guard (default: 15)
    oci_context: dict[str, str]       # compartment, region, tenancy
    classification: ClassificationResult  # intent, confidence, domains
    error: str | None
```

### 2. Database Troubleshoot Agent

**Role**: Multi-database observability and troubleshooting

**ID**: `db-troubleshoot-agent`
**Implementation**: `src/agents/database/troubleshoot.py`

**Triggers**: `database`, `slow query`, `AWR`, `wait events`, `blocking`, `performance`, `OPSI`

**Capabilities**:
| Capability | Tools Used |
|------------|------------|
| Fleet Overview | `oci_opsi_get_fleet_summary` |
| Performance Analysis | `oci_opsi_analyze_cpu`, `oci_opsi_analyze_memory` |
| SQL Execution | `oci_database_execute_sql` |
| Wait Event Analysis | `oci_opsi_analyze_wait_events` |
| Blocking Detection | `oci_opsi_get_blocking_sessions` |

**RCA Workflow** (7 steps):
1. Symptom Detection (LLM analysis)
2. Blocking Sessions Check (`oci_opsi_get_blocking_sessions`)
3. CPU & Wait Event Analysis (`oci_opsi_analyze_cpu`, `oci_opsi_analyze_wait_events`)
4. SQL Monitoring (`oci_database_execute_sql`)
5. Long Operations Check (SQL)
6. Parallelism Analysis (SQL)
7. Archive & Report (LLM synthesis)

### 3. Log Analytics Agent

**Role**: Log search, pattern detection, cross-service correlation

**ID**: `log-analytics-agent`
**Implementation**: `src/agents/log_analytics/agent.py`

**Triggers**: `logs`, `error logs`, `log search`, `audit`, `log patterns`, `logan`

**Capabilities**:
| Capability | Description |
|------------|-------------|
| Pattern Detection | Frequency analysis (ORA-, timeout, auth failures) |
| Anomaly Detection | High-frequency bursts, error concentration |
| Trace Correlation | Cross-service via trace_id |
| Temporal Correlation | Concurrent errors across services |

**Tools**: `execute_logan_query`, `search_logs`, `get_log_sources`, `analyze_trends`

### 4. Security & Threat Agent

**Role**: Threat hunting, MITRE ATT&CK mapping, security posture

**ID**: `security-threat-agent`
**Implementation**: `src/agents/security/agent.py`

**Triggers**: `security`, `threat`, `MITRE`, `attack`, `vulnerability`, `compliance`, `Cloud Guard`

**MITRE ATT&CK Mapping**:
| MITRE Technique | Cloud Guard Problem | Tactic |
|-----------------|---------------------|--------|
| T1078 | SUSPICIOUS_LOGIN | Initial Access |
| T1098 | IAM_POLICY_CHANGE | Persistence |
| T1562 | SECURITY_GROUP_CHANGE | Defense Evasion |
| T1567 | DATA_EXFILTRATION | Exfiltration |
| T1496 | CRYPTO_MINING | Impact |
| T1190 | PUBLIC_BUCKET | Initial Access |
| T1530 | OBJECT_STORAGE_ACCESS | Collection |

**Tools**: `list_cloud_guard_problems`, `get_security_events`, `scan_vulnerabilities`, `list_waf_policies`

### 5. FinOps Agent

**Role**: Cost analysis, anomaly detection, optimization recommendations

**ID**: `finops-agent`
**Implementation**: `src/agents/finops/agent.py`

**Triggers**: `cost`, `spending`, `budget`, `forecast`, `optimization`, `FinOps`

**Capabilities**:
| Capability | Description |
|------------|-------------|
| Cost Analysis | Service/compartment breakdown (30s timeout) |
| Anomaly Detection | High-concentration analysis |
| Recommendations | Rightsizing based on spend patterns |
| Multicloud | AWS/GCP cost analysis (via finopsai-mcp) |

**Tools**: `oci_cost_get_summary` (30s timeout), `get_usage_report`, `analyze_cost_anomaly`

### 6. Infrastructure Agent

**Role**: Compute, network, storage lifecycle management

**ID**: `infrastructure-agent`
**Implementation**: `src/agents/infrastructure/agent.py`

**Triggers**: `compute`, `instance`, `VM`, `network`, `VCN`, `storage`, `block volume`

**Capabilities**:
| Capability | Description |
|------------|-------------|
| Instance Lifecycle | Start, stop, terminate, list |
| Network Topology | VCN, subnet, security list analysis |
| Storage Operations | Block volume management |
| Resource Inventory | Cross-compartment discovery |

**Tools**: `oci_compute_list_instances`, `oci_compute_start_instance`, `oci_network_list_vcns`, `oci_network_list_subnets`

---

## Tool Architecture

### Tool Naming Convention

All tools follow the pattern: `oci_{domain}_{action}`

| Domain | Prefix | Examples |
|--------|--------|----------|
| compute | `oci_compute_` | `oci_compute_list_instances`, `oci_compute_get_instance` |
| network | `oci_network_` | `oci_network_list_vcns`, `oci_network_list_subnets` |
| database | `oci_database_` | `oci_database_execute_sql`, `oci_database_list_autonomous` |
| opsi | `oci_opsi_` | `oci_opsi_get_fleet_summary`, `oci_opsi_analyze_cpu` |
| cost | `oci_cost_` | `oci_cost_get_summary` |
| security | `oci_security_` | `oci_security_list_users`, `oci_security_list_policies` |
| identity | `oci_` | `oci_list_compartments`, `oci_search_compartments` |

### Tool Aliases (Backward Compatibility)

Legacy tool names are automatically resolved via `TOOL_ALIASES` in `src/mcp/catalog.py`:

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

### Tool Tiers

| Tier | Name | Latency | Risk | Approval | Examples |
|------|------|---------|------|----------|----------|
| 1 | Instant | <100ms | None | Auto | `oci_opsi_get_fleet_summary`, cache reads |
| 2 | Fast | 100ms-1s | Low | Auto | `oci_compute_list_instances`, `oci_list_compartments` |
| 3 | Moderate | 1-30s | Medium | Auto | `oci_database_execute_sql`, `oci_cost_get_summary` |
| 4 | Mutation | Variable | High | Human | `oci_compute_stop_instance`, `oci_security_update_policy` |

### Domain Prefixes (Dynamic Discovery)

Agents discover tools dynamically using `DOMAIN_PREFIXES`:

```python
DOMAIN_PREFIXES = {
    "database": ["oci_database_", "oci_opsi_"],
    "infrastructure": ["oci_compute_", "oci_network_", "oci_list_"],
    "finops": ["oci_cost_"],
    "security": ["oci_security_"],
    "observability": ["oci_observability_", "oci_logan_"],
}

# Get tools for a domain
tools = catalog.get_tools_for_domain("database")
```

### MCP Servers

| Server | Location | Tools | Domains |
|--------|----------|-------|---------|
| oci-unified | `src/mcp/server/` | 31 | identity, compute, network, cost, security, discovery |
| database-observatory | External | 50+ | database, opsi, logan, observability |
| oci-infrastructure | External | 44 | compute, network, security, cost, database |
| finopsai | External | 33 | cost, finops, anomaly, rightsizing |

> See `docs/FEATURE_MAPPING.md` for comprehensive tool-to-feature mapping.

### Tool Catalog

```python
class ToolCatalog:
    async def ensure_fresh(self) -> None:
        """Refresh tools if stale (>5 min default)."""

    def get_tool(self, name: str) -> ToolDefinition:
        """Get tool by name (resolves aliases)."""

    def get_tools_for_domain(self, domain: str) -> list[ToolDefinition]:
        """Get all tools for a domain via prefix matching."""

    def search(self, query: str) -> list[ToolDefinition]:
        """Semantic search for relevant tools."""

    async def execute(self, tool_name: str, params: dict) -> Any:
        """Execute tool via appropriate MCP server."""
```

### ToolConverter (MCP → LangChain)

```python
from src.mcp.tools.converter import ToolConverter

converter = ToolConverter(tool_catalog)

# Get all tools up to tier 3
tools = converter.to_langchain_tools(max_tier=3)

# Get only database tools
db_tools = converter.get_domain_tools("database")

# Get safe tools only (tier 1-2)
safe_tools = converter.get_safe_tools()
```

---

## Parallel Orchestrator

For queries spanning multiple domains, the parallel orchestrator executes agents concurrently.

### When Parallel Execution Triggers

1. Query involves 2+ domains (e.g., "database" + "cost")
2. Intent category is ANALYSIS or TROUBLESHOOT
3. Confidence score ≥ 0.60

### Implementation

```python
# src/agents/coordinator/orchestrator.py
class ParallelOrchestrator:
    async def execute(
        self,
        query: str,
        agents: list[BaseAgent],
        context: dict,
    ) -> SynthesizedResult:
        """Execute agents in parallel with bounded concurrency."""

        # Max 3-5 concurrent agents
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_agent(agent):
            async with semaphore:
                return await agent.invoke(query, context)

        results = await asyncio.gather(*[run_agent(a) for a in agents])
        return await self.synthesize(results, query)
```

### Example Flow

```
Query: "Analyze database performance and associated costs"

Classification:
  - domains: ["database", "finops"]
  - confidence: 0.85
  - route: "parallel"

Parallel Execution:
  ├─ db-troubleshoot-agent → Performance metrics
  └─ finops-agent → Cost breakdown

Synthesis (LLM):
  Combined analysis with correlations
```

---

## Deterministic Workflows

16 pre-built workflows handle common operations without LLM reasoning.

### Workflow Registry

| Category | Workflow | Intent Aliases |
|----------|----------|----------------|
| Identity | `list_compartments` | show_compartments, get_compartments |
| Identity | `get_tenancy_info` | tenancy_info, whoami |
| Identity | `list_regions` | show_regions, available_regions |
| Compute | `list_instances` | show_instances, get_vms |
| Compute | `get_instance` | describe_instance, instance_details |
| Network | `list_vcns` | show_networks, get_vcns |
| Network | `list_subnets` | get_subnets, show_subnets |
| Cost | `cost_summary` | get_costs, tenancy_costs, how_much_spent, monthly_cost |
| Discovery | `discovery_summary` | resource_summary, what_resources |
| Discovery | `search_resources` | find_resource, search_oci |
| Help | `search_capabilities` | capabilities, what_can_you_do |
| Help | `help` | help_me, how_to |
| Database | `db_health_check` | database_status, db_health |
| Database | `fleet_summary` | opsi_summary, db_fleet |
| Security | `security_overview` | security_status, threats |
| Logs | `recent_errors` | show_errors, log_errors |

### Workflow Execution

```python
# Workflows bypass LLM for fast, predictable responses
async def list_compartments_workflow(
    query: str,
    entities: dict,
    tool_catalog: ToolCatalog,
    memory: MemoryManager,
) -> str:
    """Execute list compartments workflow."""
    result = await tool_catalog.execute("oci_list_compartments", {
        "compartment_id": entities.get("compartment_id", "root"),
    })
    return format_compartment_table(result)
```

---

## Memory Architecture

### Tiered Memory

| Layer | Backend | Purpose | TTL |
|-------|---------|---------|-----|
| Hot Cache | Redis | Session state, tool results | 1 hour |
| Persistent | OCI ATP | Conversation history, audit | Permanent |
| Checkpoints | LangGraph | Graph state, iterations | Session |
| Vector Store | Redis VSS | RAG embeddings | Permanent |

### Context Compression

For long conversations exceeding token limits:

```python
# src/memory/context.py
class ContextCompressor:
    def compress(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Compress context when exceeding threshold."""

        if self.token_count(messages) < 150_000:
            return messages

        # Strategy: Keep recent + summarize old
        recent = messages[-20:]
        old = messages[:-20]

        summary = self.llm.summarize(old)
        return [SystemMessage(content=f"Prior context: {summary}")] + recent
```

### State Schema

```python
@dataclass
class CoordinatorState:
    # Message History
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Tool Execution
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    # Loop Control
    iteration: int
    max_iterations: int  # Default: 15

    # Context
    oci_context: dict[str, str]  # compartment, region, tenancy
    classification: ClassificationResult

    # Parallel Execution
    parallel_results: list[AgentResult] | None

    # Error Handling
    error: str | None
```

---

## Intent Classification

### Classification Categories

| Category | Pattern | Target | Workflow |
|----------|---------|--------|----------|
| `identity.list` | compartments, regions | Workflow | `list_compartments` |
| `identity.tenancy` | tenancy, whoami | Workflow | `get_tenancy_info` |
| `database.troubleshoot` | slow query, db hang | Agent | RCA workflow |
| `database.performance` | AWR, wait events | Agent | Performance analysis |
| `database.status` | db health, database status | Workflow | `db_health_check` |
| `logs.search` | find logs, show errors | Agent | Log search |
| `logs.analyze` | correlate, pattern | Agent | Pattern analysis |
| `security.threat` | attack, breach | Agent | Threat analysis |
| `security.compliance` | compliance, audit | Agent | Compliance check |
| `cost.summary` | spending, how much | Workflow | `cost_summary` |
| `cost.analyze` | cost trends, optimize | Agent | Cost analysis |
| `infra.list` | instances, VMs | Workflow | `list_instances` |
| `infra.manage` | start, stop | Agent | Lifecycle mgmt |

### Classification Flow

```python
async def classify_intent(query: str) -> ClassificationResult:
    """Classify query into intent with confidence."""

    # 1. Check exact workflow match
    for workflow, aliases in WORKFLOW_ALIASES.items():
        if any(alias in query.lower() for alias in aliases):
            return ClassificationResult(
                intent=workflow,
                confidence=0.95,
                route="workflow",
            )

    # 2. LLM classification for complex queries
    result = await llm.classify(query, categories=INTENT_CATEGORIES)

    # 3. Domain detection for parallel routing
    domains = detect_domains(query)
    if len(domains) >= 2 and result.confidence >= 0.60:
        result.route = "parallel"

    return result
```

---

## Observability

### Tracing (OCI APM)

```python
from src.observability import init_observability, get_trace_id

# Initialize on startup
init_observability(agent_name="db-troubleshoot-agent")

# Automatic trace propagation
@tracer.start_as_current_span("agent.invoke")
async def invoke(self, query: str) -> str:
    logger.info("Processing", trace_id=get_trace_id())
    ...
```

### Logging (OCI Logging)

Per-agent dedicated logs with trace correlation:

| Log ID | Agent |
|--------|-------|
| `OCI_LOG_ID_COORDINATOR` | Coordinator |
| `OCI_LOG_ID_DB_TROUBLESHOOT` | Database Troubleshoot |
| `OCI_LOG_ID_LOG_ANALYTICS` | Log Analytics |
| `OCI_LOG_ID_SECURITY_THREAT` | Security Threat |
| `OCI_LOG_ID_FINOPS` | FinOps |
| `OCI_LOG_ID_INFRASTRUCTURE` | Infrastructure |

### LLM-as-Judge Evaluation

```python
# src/evaluation/judge.py
evaluation_criteria = {
    "is_correct": "Did the answer match expected output?",
    "is_safe": "Did it respect guardrails?",
    "is_efficient": "Did it use minimum tool calls?",
    "is_complete": "Did it address all aspects of the query?",
}

# Run evaluation
result = await judge.evaluate(
    query="Check database performance",
    response=agent_response,
    expected="Performance metrics with recommendations",
)
```

---

## Guardrails & Safety

### Confirmation Required (Tier 4 Tools)

These actions require explicit user approval:
- `oci_compute_stop_instance` - Stop compute instances
- `oci_compute_terminate_instance` - Terminate instances
- `oci_database_stop_autonomous` - Stop autonomous databases
- `oci_security_update_policy` - Modify security policies
- `oci_network_delete_*` - Delete network resources

### Loop Prevention

```python
# Max iterations per agent invocation
MAX_ITERATIONS = 15

# In coordinator graph
if state["iteration"] >= state["max_iterations"]:
    return {"error": "Max iterations reached", "route": "output"}
```

### Rate Limiting

```python
RATE_LIMITS = {
    "requests_per_minute": 10,
    "cooldown_threshold": 8,
    "max_concurrent_agents": 5,
    "tool_timeout_default": 120,
    "cost_tool_timeout": 30,  # OCI Usage API is slow
}
```

### Audit Logging

All actions logged with:
- User ID and session
- Request intent classification
- Agent routing decisions
- Tool calls with parameters (redacted)
- Tool results (summarized)
- Errors and exceptions
- trace_id for APM correlation

---

## Resilience Architecture

The system implements production-grade resilience patterns for fault tolerance and self-healing.

### Circuit Breaker Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CIRCUIT BREAKER STATE MACHINE                     │
└─────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │      CLOSED      │
                         │  (Normal Flow)   │
                         └────────┬─────────┘
                                  │
                     3 consecutive failures
                                  │
                                  ▼
                         ┌──────────────────┐
                         │       OPEN       │◄────── 60s timeout
                         │  (Fast Fail)     │        resets here
                         └────────┬─────────┘
                                  │
                      After 60s cooldown
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    HALF-OPEN     │
                         │  (Probe Mode)    │
                         └────────┬─────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
         Success               Failure              Timeout
            │                     │                     │
            ▼                     ▼                     ▼
         CLOSED                 OPEN                  OPEN
```

**Configuration** (`src/mcp/registry.py`):
```python
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 3,      # Failures before open
    "reset_timeout_seconds": 60, # Cooldown before half-open
    "half_open_max_calls": 1,    # Probes before closing
}
```

### Bulkhead Pattern

Resource isolation prevents cascading failures between operation types:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BULKHEAD PARTITIONS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│   │   DATABASE      │   │ INFRASTRUCTURE  │   │      COST       │  │
│   │   Max: 3        │   │   Max: 5        │   │   Max: 2        │  │
│   │   ░░░░░░░░░░░   │   │   ░░░░░░░░░░░   │   │   ░░░░░░░░░░░   │  │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘  │
│                                                                      │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│   │    SECURITY     │   │   DISCOVERY     │   │       LLM       │  │
│   │   Max: 3        │   │   Max: 2        │   │   Max: 5        │  │
│   │   ░░░░░░░░░░░   │   │   ░░░░░░░░░░░   │   │   ░░░░░░░░░░░   │  │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Usage** (`src/resilience/bulkhead.py`):
```python
from src.resilience import Bulkhead

bulkhead = Bulkhead.get_instance()

# Acquire slot before operation
async with bulkhead.acquire("database"):
    result = await catalog.execute("oci_database_execute_sql", params)
```

### Deadletter Queue

Failed operations persist for analysis and retry:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEADLETTER QUEUE FLOW                             │
└─────────────────────────────────────────────────────────────────────┘

  Tool Call          Failure             Enqueue           Retry/Analyze
      │                  │                  │                    │
      ▼                  ▼                  ▼                    ▼
┌──────────┐      ┌──────────┐      ┌──────────────┐      ┌───────────┐
│ Execute  │ ───▶ │  Error   │ ───▶ │  Redis DLQ   │ ───▶ │  Admin    │
│          │      │ Detected │      │  (7 days)    │      │  Review   │
└──────────┘      └──────────┘      └──────────────┘      └───────────┘
                                            │
                                    ┌───────┴───────┐
                                    │               │
                                    ▼               ▼
                            ┌───────────┐   ┌───────────┐
                            │  Retried  │   │ Discarded │
                            │  (auto)   │   │ (manual)  │
                            └───────────┘   └───────────┘
```

**Failure Types**:
| Type | Description | Auto-Retry |
|------|-------------|------------|
| `TIMEOUT` | Operation exceeded timeout | Yes (3x) |
| `CONNECTION` | Server connection failed | Yes (5x) |
| `AUTHENTICATION` | Auth token expired | No |
| `RATE_LIMIT` | API rate limited | Yes (backoff) |
| `VALIDATION` | Invalid parameters | No |
| `SERVER_ERROR` | MCP server error | Yes (2x) |

**Usage** (`src/resilience/deadletter.py`):
```python
from src.resilience import DeadLetterQueue, FailureType

dlq = DeadLetterQueue(redis_url="redis://localhost:6379")

# Enqueue failed operation
await dlq.enqueue(
    failure_type=FailureType.TIMEOUT,
    operation="oci_cost_get_summary",
    error="Tool call timed out after 30s",
    params={"compartment_id": "..."},
    context={"user_id": "user_123", "thread_id": "..."},
)

# Retry failed operations
retried = await dlq.retry_failed(
    failure_types=[FailureType.TIMEOUT],
    max_retries=3,
)
```

### Health Monitor

Component-level health tracking with auto-restart:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HEALTH MONITOR                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Component              Status        Last Check    Auto-Restart  │
│   ─────────              ──────        ──────────    ────────────  │
│   mcp_oci-unified        🟢 HEALTHY    10s ago       Yes           │
│   mcp_database-obs       🟡 DEGRADED   5s ago        Yes           │
│   redis                  🟢 HEALTHY    15s ago       No            │
│   llm_oca                🟢 HEALTHY    8s ago        No            │
│                                                                      │
│   Failure Thresholds:                                               │
│   - Critical components: 3 failures → restart                       │
│   - Non-critical: 5 failures → alert only                          │
│   - Check interval: 60 seconds                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Registration** (`src/main.py`):
```python
from src.resilience import HealthMonitor, HealthCheck

monitor = HealthMonitor.get_instance()

# Register MCP server health check
monitor.register_check(HealthCheck(
    name="mcp_database-observatory",
    check_func=check_database_obs,
    restart_func=restart_database_obs,
    interval_seconds=60.0,
    failure_threshold=3,
    critical=True,  # Auto-restart enabled
))

await monitor.start()
```

---

## Self-Healing Framework

Agents can automatically recover from transient failures using the `SelfHealingMixin`.

### Error Categories

| Category | Pattern | Recovery Action |
|----------|---------|-----------------|
| `TIMEOUT` | Tool call exceeded timeout | Retry with extended timeout |
| `RATE_LIMIT` | API rate limited (429) | Wait and retry with backoff |
| `AUTH_EXPIRED` | Token expired | Refresh token and retry |
| `TRANSIENT` | Temporary server error | Retry after delay |
| `PARAMETER` | Invalid parameters | Correct and retry |
| `RESOURCE_NOT_FOUND` | OCID not found | Search by name and retry |
| `PERMANENT` | Unrecoverable error | Fail with explanation |

### SelfHealingMixin

```python
from src.agents.self_healing import SelfHealingMixin

class MyAgent(BaseAgent, SelfHealingMixin):
    def __init__(self, llm, tool_catalog):
        super().__init__(llm, tool_catalog)
        self.init_self_healing(llm, max_retries=3)

    async def invoke(self, query: str) -> str:
        # Use healing-aware tool calls
        result = await self.healing_call_tool(
            tool_name="oci_database_execute_sql",
            params={"sql": query},
            intent="Execute SQL query",
        )
        return result
```

### Self-Healing Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SELF-HEALING FLOW                                 │
└─────────────────────────────────────────────────────────────────────┘

  Tool Call                                           Success
      │                                                  ▲
      ▼                                                  │
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Execute    │ ─────▶  │   Success?   │ ─ Yes ─▶│   Return     │
│   Tool       │         │              │         │   Result     │
└──────────────┘         └──────┬───────┘         └──────────────┘
                                │
                               No
                                │
                                ▼
                        ┌──────────────┐
                        │    Analyze   │
                        │    Error     │
                        └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
       ┌───────────┐     ┌───────────┐     ┌───────────┐
       │ Transient │     │ Parameter │     │ Permanent │
       │  Error    │     │  Error    │     │  Error    │
       └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
             │                 │                 │
             ▼                 ▼                 │
       ┌───────────┐     ┌───────────┐           │
       │  Retry    │     │  Correct  │           │
       │  Strategy │     │  Params   │           │
       └─────┬─────┘     └─────┬─────┘           │
             │                 │                 │
             └────────┬────────┘                 │
                      │                          │
                      ▼                          ▼
                 ┌───────────┐            ┌───────────┐
                 │  Retry    │            │  Fail     │
                 │  (1-3x)   │            │  + DLQ    │
                 └───────────┘            └───────────┘
```

### Error Analyzer

```python
from src.agents.self_healing import ErrorAnalyzer

analyzer = ErrorAnalyzer(llm)

# Analyze an error
analysis = await analyzer.analyze(
    error="oci.exceptions.ServiceError: NotAuthorizedOrNotFound",
    tool_name="oci_compute_get_instance",
    params={"instance_id": "ocid1..."},
)

print(analysis.category)          # RESOURCE_NOT_FOUND
print(analysis.retry_worthwhile)  # True
print(analysis.suggestion)        # "Search by instance name instead"
print(analysis.wait_seconds)      # 0
```

### Parameter Corrector

```python
from src.agents.self_healing import ParameterCorrector

corrector = ParameterCorrector(llm, tool_catalog)

# Correct parameters based on error
corrected = await corrector.correct(
    params={"compartment_id": "root"},
    error_analysis=analysis,
)

print(corrected)  # {"compartment_id": "ocid1.tenancy.oc1..."}
```

---

## Multi-User Session Flow

The system supports concurrent users through session isolation.

### Session Isolation

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-USER SESSION FLOW                           │
└─────────────────────────────────────────────────────────────────────┘

  User A (Slack)              User B (API)              User C (Teams)
       │                           │                           │
       ▼                           ▼                           ▼
  ┌─────────┐                ┌─────────┐                ┌─────────┐
  │Session A│                │Session B│                │Session C│
  │thread_A │                │thread_B │                │thread_C │
  └────┬────┘                └────┬────┘                └────┬────┘
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      COORDINATOR GRAPH       │
                    │    (Async Event Loop)        │
                    │                              │
                    │  ┌────────────────────────┐ │
                    │  │  Rate Limiter (5 conc) │ │
                    │  └────────────────────────┘ │
                    │              │               │
                    │  ┌───────────┼───────────┐  │
                    │  │ Bulkhead  │ Bulkhead  │  │
                    │  │ (DB: 3)   │ (Cost: 2) │  │
                    │  └───────────┴───────────┘  │
                    │                              │
                    └──────────────┬──────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
  ┌─────────┐                ┌─────────┐                ┌─────────┐
  │Redis Key│                │Redis Key│                │Redis Key│
  │session:A│                │session:B│                │session:C│
  └─────────┘                └─────────┘                └─────────┘
```

### LLM Rate Limiting

```python
from src.llm.rate_limiter import RateLimitedLLM, wrap_with_rate_limiter

# Automatic wrapping via factory (default enabled)
llm = get_llm()  # Returns RateLimitedLLM

# Manual wrapping
base_llm = ChatOCA(...)
rate_limited = wrap_with_rate_limiter(base_llm, max_concurrent=5)

# Metrics
metrics = rate_limited.get_metrics()
print(f"Queue size: {metrics.current_queue_size}")
print(f"Wait time (avg): {metrics.average_wait_time_ms}ms")
```

---

## Evaluation Metrics

### Success Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| Task Success Rate | >85% | User accepts solution without retry |
| Workflow Ratio | >70% | Requests handled by deterministic workflows |
| Tool Accuracy | >90% | Correct tool selection and parameters |
| MTTR Reduction | >50% | vs. manual console operations |
| Human Intervention | <10% | Tier 4 approval triggers |
| P95 Latency | <5s | Workflow response time |
| P95 Latency | <30s | Agent response time |

---

## Extension Points

### Adding a New Agent

1. **Create Agent Class**: `src/agents/{domain}/{function}.py`
   ```python
   class MyAgent(BaseAgent):
       @classmethod
       def get_definition(cls) -> AgentDefinition:
           return AgentDefinition(
               agent_id="my-agent",
               name="My Agent",
               domains=["my_domain"],
               capabilities=["capability_a", "capability_b"],
           )
   ```

2. **Create System Prompt**: `prompts/NN-MY-AGENT.md`

3. **Agent auto-registers** via catalog discovery

4. **Add Tests**: `tests/test_my_agent.py`

### Adding a Workflow

1. **Define Workflow Function**:
   ```python
   async def my_workflow(query, entities, tool_catalog, memory) -> str:
       result = await tool_catalog.execute("my_tool", entities)
       return format_result(result)
   ```

2. **Register in workflows.py**:
   ```python
   WORKFLOWS["my_workflow"] = my_workflow
   WORKFLOW_ALIASES["my_workflow"] = ["alias1", "alias2"]
   ```

3. **Add Evaluation Cases**: Gold-standard test cases

### Adding an MCP Tool

1. **Add Tool Function**: `src/mcp/server/tools/{domain}.py`
   ```python
   @mcp.tool()
   async def oci_domain_action(param: str) -> str:
       """Tool description."""
       ...
   ```

2. **Register**: `register_{domain}_tools(mcp)` in `server/main.py`

3. **Assign Tier**: `TOOL_TIERS` in `src/mcp/catalog.py`

4. **Tool auto-discovers** via progressive disclosure

### Adding an MCP Server

1. **Configure**: `config/mcp_servers.yaml`
   ```yaml
   my-server:
     transport: stdio
     command: python
     args: ["-m", "my_server.main"]
     domains: [my_domain]
   ```

2. **Test**: `poetry run pytest tests/mcp/test_my_server.py`

---

## Quick Reference

### Tool Execution

```python
# Via ToolCatalog (recommended)
result = await tool_catalog.execute("oci_compute_list_instances", {
    "compartment_id": "ocid1.compartment.oc1..xxx"
})

# Via ToolConverter (for LangChain agents)
tools = converter.to_langchain_tools(max_tier=3)
agent = create_react_agent(llm, tools)
```

### Agent Invocation

```python
# Direct agent invocation
result = await db_agent.invoke("Check database performance")

# Via coordinator (recommended)
result = await coordinator.invoke(
    "Check database performance",
    config={"thread_id": "user_123"},
)
```

### Skill Execution

```python
# Execute a skill workflow
result = await agent.execute_skill(
    "db_rca_workflow",
    context={"database_id": "ocid1.autonomousdatabase..."},
)
```

---

*End of Agent Architecture Documentation*
