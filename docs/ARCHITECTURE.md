# Architecture Document

## 1. Executive Summary

The OCI AI Agent Coordinator is a LangGraph-based orchestration system implementing a **Workflow-First** architecture. It prioritizes deterministic workflows for known tasks (target: 70%+ of requests) while providing agentic fallback for complex or novel queries.

**Current Capabilities:**
- **395+ MCP Tools** across 4 servers
- **35+ Pre-built Workflows** with 100+ intent aliases
- **6 Specialized Agents** (DB Troubleshoot, Log Analytics, Security, FinOps, Infrastructure, Error Analysis)
- **281+ Tests** passing (80%+ coverage target)

See `docs/DEMO_PLAN.md` for 30 production-ready Slack commands demonstrating end-to-end capabilities.

## 2. System Context

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SYSTEMS                                   │
├───────────────┬───────────────┬───────────────┬───────────────┬────────────────┤
│    Slack      │    Teams      │  Web Client   │  OCI Events   │  Third-party   │
│   (Bolt SDK)  │ (Bot Builder) │  (REST API)   │  (Functions)  │    APIs        │
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬───────┴────────┬───────┘
        │               │               │               │                │
        └───────────────┴───────────────┼───────────────┴────────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │   OCI AI Agent Coordinator    │
                        │     (This System)             │
                        └───────────────┬───────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────┐               ┌───────────────┐               ┌───────────────┐
│  LLM Provider │               │  MCP Servers  │               │  OCI Services │
│  (OCA/Claude/ │               │  (OPSI/Logan/ │               │  (APM/Vault/  │
│   OpenAI)     │               │   Unified)    │               │   Logging)    │
└───────────────┘               └───────────────┘               └───────────────┘
```

## 3. Component Architecture

### 3.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            OCI AI AGENT COORDINATOR                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         INPUT LAYER                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │ │
│  │  │  Slack      │  │  Teams      │  │  FastAPI    │  │  Event Handler      │ │ │
│  │  │  Handler    │  │  Handler    │  │  Router     │  │  (OCI Functions)    │ │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │ │
│  │         └────────────────┴────────────────┴───────────────────┘             │ │
│  └────────────────────────────────────┬───────────────────────────────────────┘ │
│                                       │                                          │
│  ┌────────────────────────────────────▼───────────────────────────────────────┐ │
│  │                     ORCHESTRATION LAYER (LangGraph)                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      COORDINATOR GRAPH                                │  │ │
│  │  │  ┌────────┐   ┌────────────┐   ┌──────────┐   ┌────────┐            │  │ │
│  │  │  │ Input  │──▶│ Classifier │──▶│ Workflow │──▶│ Output │            │  │ │
│  │  │  └────────┘   └──────┬─────┘   └────┬─────┘   └────────┘            │  │ │
│  │  │                      │              │                                │  │ │
│  │  │                      ▼              │                                │  │ │
│  │  │               ┌──────────┐          │                                │  │ │
│  │  │               │  Agent   │◀─────────┘                                │  │ │
│  │  │               │  (LLM)   │                                           │  │ │
│  │  │               └────┬─────┘                                           │  │ │
│  │  │                    │                                                 │  │ │
│  │  │                    ▼                                                 │  │ │
│  │  │               ┌──────────┐                                           │  │ │
│  │  │               │  Action  │ ←──── Tool Execution Loop                 │  │ │
│  │  │               └──────────┘                                           │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────┬───────────────────────────────────────┘ │
│                                       │                                          │
│  ┌────────────────────────────────────▼───────────────────────────────────────┐ │
│  │                         INTEGRATION LAYER                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                       MCP CLIENT                                      │  │ │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐ │  │ │
│  │  │  │   Server     │  │     Tool     │  │  Tool-to-LangChain          │ │  │ │
│  │  │  │   Registry   │  │   Catalog    │  │  Converter                  │ │  │ │
│  │  │  └──────────────┘  └──────────────┘  └──────────────────────────────┘ │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                             │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                       LLM FACTORY                                     │  │ │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │  │ │
│  │  │  │   OCA   │  │OCI GenAI│  │Anthropic│  │ OpenAI  │  │  Ollama     │ │  │ │
│  │  │  │Adapter  │  │ Adapter │  │ Adapter │  │ Adapter │  │  Adapter    │ │  │ │
│  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────────┘ │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                      OBSERVABILITY LAYER                                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │ │
│  │  │ OTEL Tracing │  │   Metrics    │  │   Logging    │  │ LangSmith      │  │ │
│  │  │ (OCI APM)    │  │  (Prometheus)│  │  (structlog) │  │ (opt-in)       │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Descriptions

#### Input Layer
- **Slack Handler**: Bolt SDK integration for Slack workspace commands
- **Teams Handler**: Bot Framework for Microsoft Teams
- **FastAPI Router**: REST API with OpenAPI documentation
- **Event Handler**: OCI Functions for event-driven triggers

#### Orchestration Layer
- **Coordinator Graph**: Main LangGraph StateGraph
- **Intent Classifier**: Routes requests to workflows or agentic mode
- **Workflow Executor**: Deterministic subgraphs for known tasks
- **Agent Node**: LLM-driven reasoning with tool binding
- **Action Node**: MCP tool execution

#### Integration Layer
- **Server Registry**: Manages MCP server connections
- **Tool Catalog**: Aggregated tools from all servers
- **LLM Factory**: Provider-agnostic LLM instantiation

#### Observability Layer
- **OTEL Tracing**: OpenTelemetry spans to OCI APM
- **Metrics**: Prometheus-compatible metrics
- **Logging**: Structured logging via structlog
- **LangSmith**: Optional LangChain tracing

## 4. Data Flow

### 4.1 Request Processing Flow

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            REQUEST FLOW                                        │
└───────────────────────────────────────────────────────────────────────────────┘

1. INPUT RECEPTION
   ┌─────────┐      ┌─────────────┐
   │ Channel │ ──▶  │ Normalizer  │ ──▶ Unified Request Format
   └─────────┘      └─────────────┘

2. INTENT CLASSIFICATION
   ┌─────────────────┐      ┌────────────────┐
   │ Unified Request │ ──▶  │ LLM Classifier │ ──▶ Intent + Confidence
   └─────────────────┘      └────────────────┘

3. ROUTING DECISION
   ┌──────────────────┐
   │  Confidence ≥ 0.8 │ ──▶ Deterministic Workflow
   ├──────────────────┤
   │ 0.5 ≤ Conf < 0.8 │ ──▶ Agentic Mode
   ├──────────────────┤
   │  Confidence < 0.5 │ ──▶ Clarification Request
   └──────────────────┘

4. EXECUTION
   ┌───────────┐      ┌───────────┐      ┌────────────┐
   │ Workflow/ │ ──▶  │ Tool Exec │ ──▶  │  Response  │
   │ Agent     │      │ (MCP)     │      │  Formatter │
   └───────────┘      └───────────┘      └────────────┘

5. OUTPUT
   ┌─────────────────┐      ┌─────────┐
   │ Formatted Resp  │ ──▶  │ Channel │
   └─────────────────┘      └─────────┘
```

### 4.2 Tool Execution Flow

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          TOOL EXECUTION FLOW                                   │
└───────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Agent decides  │
│  to call tool   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  ToolCatalog    │ ──▶  │  Find Tool      │ ──▶ ToolDefinition
│  lookup         │      │  by name        │
└─────────────────┘      └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Get MCP Client │
                         │  for server     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Execute via    │
                         │  MCP Protocol   │
                         └────────┬────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  stdio (SQLcl)  │      │  HTTP (OPSI)    │      │  SSE (Unified)  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Result to      │
                         │  ToolMessage    │
                         └─────────────────┘
```

## 5. State Management

### 5.1 CoordinatorState Schema

```python
@dataclass
class CoordinatorState:
    # Conversation
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Tool Execution
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    # Control Flow
    iteration: int
    max_iterations: int
    current_workflow: str | None

    # Context
    oci_context: OciContext
    conversation_context: ConversationContext

    # Status
    error: str | None
    requires_approval: bool


@dataclass
class OciContext:
    tenancy_ocid: str | None
    compartment_id: str | None
    region: str | None
    resource_focus: dict[str, str] | None


@dataclass
class ConversationContext:
    session_id: str
    channel: str  # slack, teams, web, api
    user_id: str
    user_role: str
    topic: str | None
```

### 5.2 Persistence Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STATE PERSISTENCE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Development:                                                        │
│  ┌────────────────┐                                                 │
│  │  MemorySaver   │  In-memory checkpoints                          │
│  └────────────────┘                                                 │
│                                                                      │
│  Production:                                                         │
│  ┌────────────────┐      ┌────────────────┐                         │
│  │  RedisSaver    │ ──▶  │  Redis Cluster │                         │
│  └────────────────┘      └────────────────┘                         │
│         │                                                            │
│         ├── Session TTL: 60 minutes                                  │
│         ├── Max history: 20 messages per thread                      │
│         └── Checkpoint on every state change                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 6. End-to-End Communication Flow

### 6.1 Complete Message Flow (Slack → Backend → OCI)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE MESSAGE FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

 User sends                                                          OCI Response
 Slack message                                                       returned
      │                                                                    ▲
      ▼                                                                    │
┌───────────────┐                                                   ┌───────────────┐
│  1. SLACK     │                                                   │  8. SLACK     │
│  Socket Mode  │ ◄─────────────────────────────────────────────────│  Response     │
│  Bolt SDK     │                                                   │  Blocks       │
└───────┬───────┘                                                   └───────┬───────┘
        │                                                                   ▲
        │ @mention or DM                                                    │
        ▼                                                                   │
┌───────────────┐      ┌───────────────┐      ┌───────────────┐           │
│  2. MESSAGE   │ ───▶ │  3. TRACER    │ ───▶ │  4. CLASSIFY  │           │
│  HANDLER      │      │  Start Span   │      │  Intent       │           │
│  (slack.py)   │      │  (OCI APM)    │      │  (LLM)        │           │
└───────────────┘      └───────────────┘      └───────┬───────┘           │
                                                      │                    │
                                                      ▼                    │
┌───────────────────────────────────────────────────────────────────────────┐
│                         5. LANGGRAPH COORDINATOR                           │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│    ┌───────────────────────────────────────────────────────────────────┐ │
│    │  CoordinatorState                                                  │ │
│    │  ├── messages: Sequence[BaseMessage]                              │ │
│    │  ├── tool_calls: list[dict]                                       │ │
│    │  ├── current_agent: str                                           │ │
│    │  └── oci_context: {compartment, region, tenancy}                  │ │
│    └───────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     │
│    │ Input    │ ──▶ │ Agent    │ ──▶ │ Action   │ ──▶ │ Output   │     │
│    │ Node     │     │ Node     │     │ Node     │     │ Node     │     │
│    └──────────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘     │
│                          │                │                 │           │
│                          ▼                ▼                 │           │
│                   ┌────────────────────────────┐            │           │
│                   │  SPECIALIZED AGENTS (6)    │            │           │
│                   │  ┌──────────────────────┐ │            │           │
│                   │  │ • DB Troubleshoot    │ │            │           │
│                   │  │ • Log Analytics      │ │            │           │
│                   │  │ • Security Threat    │ │            │           │
│                   │  │ • FinOps             │ │            │           │
│                   │  │ • Infrastructure     │ │            │           │
│                   │  │ • Error Analysis     │ │            │           │
│                   │  └──────────────────────┘ │            │           │
│                   └──────────────┬────────────┘            │           │
│                                  │                          │           │
└──────────────────────────────────┼──────────────────────────┼───────────┘
                                   │                          │
                                   ▼                          │
┌──────────────────────────────────────────────────────────────────────────┐
│                          6. MCP CLIENT LAYER                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│   │ ServerRegistry│ ──▶ │  ToolCatalog │ ──▶ │  MCP Client  │            │
│   │ (health)     │     │  (search)    │     │  (execute)   │            │
│   └──────────────┘     └──────────────┘     └──────┬───────┘            │
│                                                     │                    │
│   Transport Selection:                              │                    │
│   ├── stdio: Local process (SQLcl, Python)         │                    │
│   ├── HTTP:  REST endpoints                        │                    │
│   └── SSE:   Streaming responses                   │                    │
│                                                     │                    │
│   Timeout & Retry Configuration:                   │                    │
│   ├── Default: 120s (increased for large compartments)                  │
│   ├── Tool-specific: 180s-300s for slow operations                      │
│   ├── Retry: 3 attempts with exponential backoff                        │
│   └── Backoff multiplier: 1.5x per retry                                │
│                                                     │                    │
└─────────────────────────────────────────────────────┼────────────────────┘
                                                      │
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          7. MCP SERVERS                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  oci-unified (Built-in) — 51 tools                                  ││
│   │  └── Identity, Compute, Network, Cost, Security, Observability      ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                           │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  database-observatory (External) — 50+ tools                         ││
│   │  └── OPSI, SQLcl, Logan Analytics, DB Management                    ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                           │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  oci-infrastructure (External) — 44 tools                            ││
│   │  └── Compute, Network, Security (via mcp-oci)                       ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                           │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  finopsai (External) — 33 tools                                      ││
│   │  └── Multicloud Cost, Anomaly Detection, Rightsizing                ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                           │                                               │
└───────────────────────────┼───────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   OCI APIs    │
                    │ (via OCI SDK) │
                    └───────────────┘
```

### 6.2 Message Processing Steps

| Step | Component | Action | Latency |
|------|-----------|--------|---------|
| 1 | Slack (Socket Mode) | Receive @mention or DM | <50ms |
| 2 | Message Handler | Parse, validate, extract context | <10ms |
| 3 | Tracer | Start OpenTelemetry span | <5ms |
| 4 | Classifier | LLM intent classification | 500ms-1s |
| 5 | Coordinator | Route to specialized agent | <50ms |
| 6 | Agent | Execute workflow/reasoning | 1-10s |
| 7 | MCP Client | Call tools via servers | 100ms-30s |
| 8 | Response | Format and send to Slack | <100ms |

## 7. MCP Server Integration

### 7.1 Server Configuration

Configure in `config/mcp_servers.yaml`:

```yaml
servers:
  database-observatory:
    transport: stdio
    command: python
    args: ["-m", "src.mcp_server"]
    working_dir: /path/to/mcp-oci-database-observatory
    enabled: true
    domains: [database, opsi, logan, observability]
    timeout_seconds: 60

defaults:
  timeout_seconds: 120  # Default for most tools
  retry_attempts: 3
  backoff_multiplier: 2
  tool_timeouts:
    oci_cost_get_summary: 30     # OCI Usage API can be slow
    oci_compute_list_instances: 180
    oci_observability_query_logs: 300
    oci_discovery_run: 60        # Full resource discovery
```

### 7.2 GitHub References for MCP Servers

> **Disclaimer**: The external MCP servers listed below are personal projects created to demonstrate OCI AI integration capabilities. These are **NOT official Oracle products** and are not endorsed by Oracle Corporation.

| Server | Description | GitHub Repository |
|--------|-------------|-------------------|
| **oci-unified** | Built-in unified server with ShowOCI-style discovery | `src/mcp/server/` (this project) |
| **database-observatory** | OPSI, SQLcl, Logan Analytics | [adibirzu/mcp-oci-database-observatory](https://github.com/adibirzu/mcp-oci-database-observatory) |
| **oci-infrastructure** | Full OCI management (mcp-oci) | [adibirzu/mcp-oci](https://github.com/adibirzu/mcp-oci) |
| **finopsai-mcp** | Multicloud FinOps with anomaly detection | [adibirzu/finopsai-mcp](https://github.com/adibirzu/finopsai-mcp) |

### 7.3 MCP Server Capabilities Comparison

| Feature | oci-unified | mcp-oci | finopsai-mcp (optional) |
|---------|-------------|---------|-------------------------|
| **Compartment Discovery** | ✅ ShowOCI-style with Redis caching | ✅ Basic | - |
| **Instance by Name** | ✅ find/start/stop/restart | ❌ Requires OCID | - |
| **Resource Caching** | ✅ Redis-backed | ❌ Direct API | - |
| **Compute Operations** | ✅ Basic | ✅ Full | - |
| **Network Operations** | ✅ Basic | ✅ Full | - |
| **Cost Analysis** | ✅ With 30s timeout | ✅ Basic | ✅ Advanced |
| **Cost Anomaly Detection** | ✅ Built-in (high-concentration) | ❌ | ✅ Cost spikes |
| **Rightsizing Recommendations** | ✅ Built-in heuristics | ❌ | ✅ Detailed |
| **Budget Tracking** | ❌ | ❌ | ✅ Budget status |
| **Multicloud** | ❌ | ❌ | ✅ OCI, AWS, Azure, GCP |
| **Troubleshoot Skills** | ✅ | ❌ | ❌ |

**Note:** oci-unified is the primary server for cost analysis. It includes a 30-second timeout on the OCI Usage API to prevent hanging. The FinOps agent uses built-in heuristics for anomaly detection and recommendations when finopsai-mcp is not available.

### 7.4 Server Capabilities

The Database Observatory MCP server provides unified access to:
- **OPSI** - Operations Insights for performance metrics
- **SQLcl** - Direct SQL execution
- **Logan** - Logging Analytics queries

### 7.5 Database Observatory Tool Tiers

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATABASE OBSERVATORY TOOLS                        │
└─────────────────────────────────────────────────────────────────────┘

  Tier 1: Cache (<100ms)
  ├── get_fleet_summary
  ├── search_databases
  └── get_cached_database

  Tier 2: OPSI API (1-5s)
  ├── analyze_cpu_usage
  ├── analyze_memory_usage
  ├── get_performance_summary
  ├── get_wait_events
  └── get_tablespace_usage

  Tier 3: SQL Execution (5-30s)
  ├── execute_sql
  ├── get_schema_info
  ├── database_status
  └── get_blocking_sessions
```

## 8. Tool Tier Classification

### 8.1 Progressive Disclosure Strategy

To prevent overwhelming agents with too many tools, the system implements progressive disclosure via tool tiers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOOL TIER CLASSIFICATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Tier 1: Instant (< 100ms, no risk)                                 │
│  ├── oci_ping                                                        │
│  ├── oci_search_tools                                                │
│  └── oci_get_capabilities                                            │
│                                                                      │
│  Tier 2: Fast Reads (< 1s, no risk)                                 │
│  ├── oci_compute_list_instances                                      │
│  ├── oci_database_list_autonomous                                    │
│  ├── oci_network_list_vcns                                           │
│  └── oci_cost_get_summary                                            │
│                                                                      │
│  Tier 3: Moderate Operations (< 5s, low risk)                       │
│  ├── oci_observability_get_metrics                                   │
│  ├── oci_observability_query_logs                                    │
│  └── oci_database_get_awr_report                                     │
│                                                                      │
│  Tier 4: Slow/Mutating (> 5s, medium-high risk)                     │
│  ├── oci_compute_start_instance      [REQUIRES CONFIRMATION]        │
│  ├── oci_compute_stop_instance       [REQUIRES CONFIRMATION]        │
│  └── oci_database_scale              [REQUIRES CONFIRMATION]        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Tool Search API

Agents discover tools via the `search_tools` meta-tool:

```python
# Search by domain
results = catalog.search_tools(domain="database", max_tier=3)

# Search by capability
results = catalog.search_tools(query="performance", limit=10)

# Full catalog for planning
results = catalog.search_tools(max_tier=2)  # Only fast, safe tools
```

### 8.3 Risk Management

| Risk Level | Confirmation | Use Case |
|------------|--------------|----------|
| `none` | No | Read-only operations |
| `low` | No | Slow queries, reports |
| `medium` | Yes | Start/restart operations |
| `high` | Yes | Stop/delete/modify operations |

### 8.4 Instance Operations by Name

The unified MCP server includes convenience tools for operating on instances by display name:

```python
# Find instances by name (supports partial match)
oci_compute_find_instance(instance_name="prod", compartment_id="ocid1...")

# Start/Stop/Restart by name (auto-resolves to OCID)
oci_compute_start_by_name(instance_name="my-instance", compartment_id="ocid1...")
oci_compute_stop_by_name(instance_name="my-instance", compartment_id="ocid1...")
oci_compute_restart_by_name(instance_name="my-instance", compartment_id="ocid1...")
```

These tools handle:
- Partial name matching (case-insensitive)
- Disambiguation when multiple instances match
- Automatic OCID resolution
- Error handling for non-existent instances

## 9. LLM Integration

### 9.1 Factory Pattern

```python
class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create(provider: str, **kwargs) -> BaseChatModel:
        match provider:
            case "oca":
                return OCALangChainClient(**kwargs)
            case "oci-genai":
                return ChatOCIGenAI(**kwargs)
            case "anthropic":
                return ChatAnthropic(**kwargs)
            case "openai":
                return ChatOpenAI(**kwargs)
            case "ollama":
                return ChatOllama(**kwargs)
            case _:
                raise ValueError(f"Unknown provider: {provider}")
```

### 9.2 Tool Binding

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOOL BINDING FLOW                                 │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  MCP Tools     │ ──▶  │  ToolConverter │ ──▶  │ LangChain      │
│  (JSON Schema) │      │                │      │ StructuredTool │
└────────────────┘      └────────────────┘      └────────┬───────┘
                                                         │
                                                         ▼
                                                ┌────────────────┐
                                                │  llm.bind_tools│
                                                │  (tools)       │
                                                └────────────────┘
```

## 10. Observability

### 10.1 Tracing Architecture (OCI APM)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OTEL TRACING → OCI APM                            │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                          TRACE HIERARCHY                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  coordinator.invoke (root span)                                       │
│    │                                                                  │
│    ├── input_node                                                     │
│    │                                                                  │
│    ├── classifier_node                                                │
│    │     └── llm.invoke (classify intent)                            │
│    │                                                                  │
│    ├── agent_node (iteration 1)                                       │
│    │     └── llm.invoke (with tools)                                 │
│    │                                                                  │
│    ├── action_node                                                    │
│    │     ├── mcp.call_tool (tool_1)                                  │
│    │     └── mcp.call_tool (tool_2)                                  │
│    │                                                                  │
│    ├── agent_node (iteration 2)                                       │
│    │     └── llm.invoke (with tool results)                          │
│    │                                                                  │
│    └── output_node                                                    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.2 Per-Agent OCI Logging

Each agent writes to a dedicated OCI Log with trace_id correlation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OCI LOGGING                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Log Group: oci-coordinator-agents                                  │
│    │                                                                 │
│    ├── Log: coordinator                                              │
│    ├── Log: db-troubleshoot-agent                                    │
│    ├── Log: log-analytics-agent                                      │
│    ├── Log: security-threat-agent                                    │
│    ├── Log: finops-agent                                             │
│    └── Log: infrastructure-agent                                     │
│                                                                      │
│  Each log entry includes:                                           │
│    - trace_id (links to OCI APM)                                    │
│    - span_id                                                         │
│    - timestamp                                                       │
│    - message                                                         │
│    - agent_name                                                      │
│    - level (INFO/WARNING/ERROR)                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.3 Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `coordinator_requests_total` | Counter | channel, intent |
| `coordinator_latency_seconds` | Histogram | channel, intent |
| `tool_calls_total` | Counter | tool_name, server |
| `tool_latency_seconds` | Histogram | tool_name, server |
| `llm_tokens_total` | Counter | provider, direction |
| `workflow_ratio` | Gauge | - |

## 11. Security Architecture

### 11.1 Authentication Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTHENTICATION                                    │
└─────────────────────────────────────────────────────────────────────┘

Channel Authentication:
  Slack  ──▶ HMAC signature verification
  Teams  ──▶ Bot Framework token validation
  API    ──▶ Bearer token / API key

OCI Authentication:
  ┌──────────────────┐      ┌──────────────────┐
  │  ~/.oci/config   │ ──▶  │  OCI SDK Signer  │
  └──────────────────┘      └──────────────────┘
           │
           ├── Config file auth (development)
           ├── Instance Principal (production)
           └── Resource Principal (OCI Functions)

Secrets:
  ┌──────────────────┐      ┌──────────────────┐
  │  .env.local      │ ──▶  │  Development     │
  └──────────────────┘      └──────────────────┘

  ┌──────────────────┐      ┌──────────────────┐
  │  OCI Vault       │ ──▶  │  Production      │
  └──────────────────┘      └──────────────────┘
```

### 11.2 Authorization

```python
# Action authorization matrix
AUTHORIZATION = {
    "admin": ["read", "write", "delete", "approve"],
    "operator": ["read", "write"],
    "viewer": ["read"],
}

# Dangerous actions requiring approval
APPROVAL_REQUIRED = [
    "delete_*",
    "stop_instance",
    "terminate_*",
    "modify_policy",
]
```

## 12. Deployment Architecture

### 12.1 Development

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT                                 │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┐      ┌────────────────┐
│  uvicorn       │      │  MCP Servers   │
│  (hot reload)  │ ──▶  │  (localhost)   │
└────────────────┘      └────────────────┘
         │
         ▼
┌────────────────┐
│  .env.local    │  Configuration
└────────────────┘
```

### 12.2 Production (OKE)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OKE DEPLOYMENT                                    │
└─────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────────────────────┐
                    │         OCI Load Balancer             │
                    └───────────────────┬───────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │              Ingress                   │
                    └───────────────────┬───────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         │                              │                              │
         ▼                              ▼                              ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│  Coordinator    │          │  MCP OPSI       │          │  MCP Logan      │
│  (3 replicas)   │          │  (2 replicas)   │          │  (2 replicas)   │
└─────────────────┘          └─────────────────┘          └─────────────────┘
         │                              │                              │
         └──────────────────────────────┼──────────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │          Redis Cluster                │
                    │         (State Storage)               │
                    └───────────────────────────────────────┘
```

## 13. Implementation Phases

### Phase 1: Foundation ✅
- [x] Project documentation (CLAUDE.md, ARCHITECTURE.md)
- [x] Python project setup (Poetry, pyproject.toml)
- [x] OpenTelemetry tracing module (`src/observability/`)
- [x] Multi-LLM factory (`src/llm/factory.py`)
- [x] Basic LangGraph coordinator (`src/agents/coordinator/`)

### Phase 2: MCP Integration ✅
- [x] MCP client library (`src/mcp/client.py`)
- [x] Server registry (`src/mcp/registry.py`)
- [x] Tool catalog with progressive disclosure (`src/mcp/catalog.py`)
- [x] Tool tier classification for risk management
- [x] LangChain tool conversion
- [x] Retry logic with exponential backoff
- [x] Tool-specific timeouts for large compartments

### Phase 3: Agents ✅
- [x] Agent Catalog with auto-discovery (`src/agents/catalog.py`)
- [x] Database Troubleshoot agent (`src/agents/database/`)
- [x] Log Analytics agent (`src/agents/log_analytics/`)
- [x] Security Threat agent (`src/agents/security/`)
- [x] FinOps agent (`src/agents/finops/`)
- [x] Infrastructure agent (`src/agents/infrastructure/`)
- [x] Error Analysis agent (`src/agents/error_analysis/`)
- [x] Channel-aware formatting (`src/formatting/`)
- [x] Instance operations by name (`oci_compute_*_by_name`)
- [x] DB Troubleshooting workflows (see `docs/DB_TROUBLESHOOTING_WORKFLOW.md`)

### Phase 4: Evaluation & Production 🔄
- [x] FastAPI endpoints (`src/api/`)
- [x] Evaluation framework (`src/evaluation/`)
- [x] Slack integration (`src/channels/slack.py`)
- [x] Resilience infrastructure (bulkhead, circuit breaker, health monitor)
- [ ] Teams integration
- [ ] OKE deployment
- [ ] Production observability

## 14. Resilience Architecture

### 14.1 Production Hardening Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RESILIENCE ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                          REQUEST FLOW                                       │ │
│  │                                                                             │ │
│  │   User Request                                                              │ │
│  │        │                                                                    │ │
│  │        ▼                                                                    │ │
│  │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     │ │
│  │   │  Rate Limiter   │────▶│    Bulkhead     │────▶│ Circuit Breaker │     │ │
│  │   │  (LLM: 5 conc)  │     │  (by partition) │     │  (by server)    │     │ │
│  │   └─────────────────┘     └─────────────────┘     └────────┬────────┘     │ │
│  │                                                             │              │ │
│  │                                                             ▼              │ │
│  │                                                    ┌─────────────────┐     │ │
│  │                                                    │   Tool Catalog  │     │ │
│  │                                                    │   (execute)     │     │ │
│  │                                                    └────────┬────────┘     │ │
│  │                                                             │              │ │
│  │                            ┌────────────────────────────────┤              │ │
│  │                            │                                │              │ │
│  │                            ▼                                ▼              │ │
│  │                   ┌─────────────────┐              ┌─────────────────┐     │ │
│  │                   │    Success      │              │    Failure      │     │ │
│  │                   │                 │              │                 │     │ │
│  │                   └─────────────────┘              └────────┬────────┘     │ │
│  │                                                             │              │ │
│  │                                                             ▼              │ │
│  │                                                    ┌─────────────────┐     │ │
│  │                                                    │ Deadletter Queue│     │ │
│  │                                                    │ (Redis, 7 days) │     │ │
│  │                                                    └─────────────────┘     │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                        BACKGROUND SERVICES                                  │ │
│  │                                                                             │ │
│  │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     │ │
│  │   │ Health Monitor  │     │  Auto-Restart   │     │   DLQ Retry     │     │ │
│  │   │ (60s interval)  │────▶│  (on failure)   │     │  (scheduled)    │     │ │
│  │   └─────────────────┘     └─────────────────┘     └─────────────────┘     │ │
│  │                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 14.2 Circuit Breaker State Machine

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CIRCUIT BREAKER STATE MACHINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                              ┌───────────────────┐                              │
│                              │      CLOSED       │                              │
│                              │   (Normal Mode)   │                              │
│                              │                   │                              │
│                              │ • All calls pass  │                              │
│                              │ • Count failures  │                              │
│                              └─────────┬─────────┘                              │
│                                        │                                         │
│                            failures >= threshold (3)                             │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                              OPEN                                        │   │
│   │                          (Fast Fail Mode)                                │   │
│   │                                                                          │   │
│   │  • All calls fail immediately with "Server temporarily unavailable"     │   │
│   │  • No requests reach the server                                          │   │
│   │  • Timer: 60 seconds                                                     │   │
│   └─────────────────────────────────┬───────────────────────────────────────┘   │
│                                     │                                            │
│                           timeout expires (60s)                                  │
│                                     │                                            │
│                                     ▼                                            │
│                              ┌───────────────────┐                              │
│                              │    HALF-OPEN      │                              │
│                              │   (Probe Mode)    │                              │
│                              │                   │                              │
│                              │ • Allow 1 probe   │                              │
│                              │ • Test if healthy │                              │
│                              └─────────┬─────────┘                              │
│                                        │                                         │
│                    ┌───────────────────┴───────────────────┐                    │
│                    │                                       │                    │
│               probe succeeds                         probe fails                 │
│                    │                                       │                    │
│                    ▼                                       ▼                    │
│             ┌───────────┐                           ┌───────────┐              │
│             │  CLOSED   │                           │   OPEN    │              │
│             │ (healthy) │                           │ (restart) │              │
│             └───────────┘                           └───────────┘              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 14.3 Bulkhead Partitions

Resource isolation prevents one domain from exhausting resources:

| Partition | Max Concurrent | Timeout | Use Case |
|-----------|----------------|---------|----------|
| DATABASE | 3 | 60s | SQL execution, OPSI queries |
| INFRASTRUCTURE | 5 | 30s | Compute/network operations |
| COST | 2 | 30s | OCI Usage API (slow) |
| SECURITY | 3 | 30s | Cloud Guard, IAM |
| DISCOVERY | 2 | 60s | Full resource scans |
| LLM | 5 | 300s | LLM inference |
| DEFAULT | 10 | 120s | Other operations |

### 14.4 Health Monitor Configuration

| Component | Check Interval | Failure Threshold | Auto-Restart | Critical |
|-----------|----------------|-------------------|--------------|----------|
| MCP Servers | 60s | 3 | Yes | Yes |
| Redis | 30s | 3 | No | Yes |
| LLM Provider | 120s | 5 | No | No |

### 14.5 Deadletter Queue Retention

| Failure Type | Retention | Auto-Retry | Max Retries |
|--------------|-----------|------------|-------------|
| TIMEOUT | 7 days | Yes | 3 |
| CONNECTION | 7 days | Yes | 5 |
| RATE_LIMIT | 1 day | Yes | 10 |
| VALIDATION | 7 days | No | 0 |
| AUTHENTICATION | 1 day | No | 0 |
| SERVER_ERROR | 7 days | Yes | 2 |

## 15. Multi-User Concurrency

### 15.1 Session Isolation

Each user gets an isolated session with:
- Thread-specific Redis keys for state
- LangGraph checkpoint isolation via `thread_id`
- Bulkhead slot reservation per operation type

### 15.2 Rate Limiting

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LLM RATE LIMITING                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Concurrent Users                                                              │
│        │                                                                         │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────┐                                   │
│   │          SEMAPHORE (max: 5)              │                                   │
│   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│                                   │
│   │  │slot1│ │slot2│ │slot3│ │slot4│ │slot5││                                   │
│   │  │ ✓   │ │ ✓   │ │ ✓   │ │wait │ │wait ││                                   │
│   │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│                                   │
│   └─────────────────────────────────────────┘                                   │
│        │                                                                         │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────┐                                   │
│   │          LLM PROVIDER                    │                                   │
│   │  (OCA / Claude / OpenAI)                 │                                   │
│   └─────────────────────────────────────────┘                                   │
│                                                                                  │
│   Queue Metrics:                                                                │
│   • current_queue_size: Number waiting for slot                                 │
│   • average_wait_time_ms: Avg wait for slot                                     │
│   • total_requests: Total requests processed                                    │
│   • timeout_count: Requests that timed out                                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 15.3 Initialization Lock

Prevents race conditions during concurrent startup:

```python
# src/main.py
_init_lock: asyncio.Lock | None = None
_initialized: bool = False

async def initialize_coordinator() -> None:
    global _initialized
    async with _get_init_lock():
        if _initialized:
            return  # Already initialized
        # ... initialization code ...
        _initialized = True
```

---

## 16. Related Documentation

| Document | Purpose |
|----------|---------|
| `DEMO_PLAN.md` | 30 production-ready Slack commands with API examples |
| `DB_TROUBLESHOOTING_WORKFLOW.md` | Database diagnostic workflow mapping |
| `FEATURE_MAPPING.md` | Complete tool-to-workflow mapping |
| `OCI_AGENT_REFERENCE.md` | Agent configurations and schemas |
| `SLACK_WORKFLOW_AUDIT.md` | Slack integration patterns |
