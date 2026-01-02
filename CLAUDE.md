# CLAUDE.md - OCI AI Agent Coordinator

## Project Overview

The OCI AI Agent Coordinator is a Python-based LangGraph orchestration system that manages specialized AI agents for Oracle Cloud Infrastructure operations. It acts as a central hub receiving requests from multiple channels (Slack, Teams, Web, API) and routing them to specialized agents via MCP (Model Context Protocol) servers.

**Status**: Phase 4 - Production Readiness (In Progress)

| Phase | Status | Key Components |
|-------|--------|----------------|
| 1. Environment & LLM | ✅ Complete | Multi-LLM factory, OCA OAuth |
| 2. Unified MCP Layer | ✅ Complete | 4 MCP servers, 158+ tools, ToolCatalog |
| 3. LangGraph Coordinator | ✅ Complete | See details below |
| 4. Production Readiness | 🔄 In Progress | Teams integration, OKE deployment |

**Phase 3 Highlights:**
- LangGraph coordinator with intent routing and parallel orchestration
- 6 specialized agents: DB Troubleshoot, Log Analytics, Security, FinOps, Infrastructure, Error Analysis
- 16 pre-built deterministic workflows for common operations
- RAG with OCI GenAI embeddings
- Self-healing framework with error recovery
- Resilience infrastructure (Bulkhead, Circuit Breaker, Dead Letter Queue)
- Slack integration with 3-second ack pattern
- FastAPI REST API with SSE streaming
- OCI APM tracing + per-agent OCI Logging

**Last Updated**: 2026-01-02

## Quick Start

```bash
# Install dependencies
poetry install

# Run tests (212 passing)
poetry run pytest --cov=src

# Start coordinator (Slack + API on port 3001) - DEFAULT
poetry run python -m src.main

# Or with explicit modes
poetry run python -m src.main --mode both      # Slack + API (default)
poetry run python -m src.main --mode slack     # Slack only
poetry run python -m src.main --mode api       # API only
poetry run python -m src.main --port 8080      # Custom API port
```

**On startup, the following services are initialized:**
- Observability (OCI APM + Logging)
- OCA OAuth callback server (port 48801)
- OCI Discovery service
- MCP server connections
- Agent catalog with auto-discovery
- ShowOCI cache (if enabled)

**First-time OCA Login:**
When using Slack and OCA authentication is required, click the "Login with Oracle SSO" button that appears. The callback server handles the OAuth redirect automatically.

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT CHANNELS                                    │
├──────────────────┬──────────────────┬────────────────────────────────────────┤
│  Slack Bot       │  API Server      │           (Teams, Web - planned)       │
│  Socket Mode     │  FastAPI         │                                        │
└────────┬─────────┴────────┬─────────┴───────────────────┬────────────────────┘
         └──────────────────┴───────────┬─────────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │   COORDINATOR (LangGraph)      │
                        │   Intent → Route → Workflow    │
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
                        │   DATABASE OBSERVATORY     │
                        │   MCP Server (stdio)       │
                        │   ┌─────┬─────┬─────┐     │
                        │   │OPSI │SQLcl│Logan│     │
                        │   └─────┴─────┴─────┘     │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   ORACLE CLOUD             │
                        │   INFRASTRUCTURE           │
                        └───────────────────────────┘

                            OBSERVABILITY
                    ┌───────────────────────────┐
                    │  OCI APM  ←→  OCI Logging │
                    │  (traces)    (structured) │
                    │       trace_id correlation│
                    └───────────────────────────┘
```

---

## Naming Conventions

### Agent Naming

| Component | Convention | Example |
|-----------|------------|---------|
| Agent Role | `{domain}-{function}-agent` | `db-troubleshoot-agent` |
| Agent ID | `{role}-{uuid-suffix}` | `db-troubleshoot-agent-c5b6cd64b` |
| Agent Class | `{Domain}{Function}Agent` | `DbTroubleshootAgent` |
| Agent Module | `src/agents/{domain}/{function}.py` | `src/agents/database/troubleshoot.py` |

### Tool Naming

| Component | Convention | Example |
|-----------|------------|---------|
| MCP Tool | `oci_{domain}_{action}` | `oci_compute_list_instances` |
| Skill | `{domain}_{workflow}` | `database_rca_workflow` |
| Discovery Tool | `oci_{action}_{noun}` | `oci_search_tools` |

**MCP Tool Examples by Domain:**
| Domain | Tool | Description |
|--------|------|-------------|
| identity | `oci_list_compartments` | List compartments in tenancy |
| identity | `oci_search_compartments` | Search compartments by name |
| compute | `oci_compute_list_instances` | List compute instances |
| compute | `oci_compute_get_instance` | Get instance details |
| network | `oci_network_list_vcns` | List VCNs |
| network | `oci_network_list_subnets` | List subnets |
| cost | `oci_cost_get_summary` | Get cost summary |
| security | `oci_security_list_users` | List IAM users |
| database | `oci_database_execute_sql` | Execute SQL query |
| opsi | `oci_opsi_get_fleet_summary` | Get database fleet summary |

### Tool Aliases (Backward Compatibility)

Legacy tool names are automatically resolved to standard names via `TOOL_ALIASES` in `src/mcp/catalog.py`:

```python
# Legacy names → Standard names
"execute_sql" → "oci_database_execute_sql"
"get_fleet_summary" → "oci_opsi_get_fleet_summary"
"analyze_cpu_usage" → "oci_opsi_analyze_cpu"
"list_autonomous_databases" → "oci_database_list_autonomous"
```

### Domain Prefixes (Dynamic Tool Discovery)

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

### File Naming

| Type | Convention | Example |
|------|------------|---------|
| Agent module | `src/agents/{domain}/{function}.py` | `database/troubleshoot.py` |
| Tool module | `src/mcp/server/tools/{domain}.py` | `tools/compute.py` |
| Test file | `tests/test_{module}.py` | `test_mcp_server.py` |
| Prompt file | `prompts/{NN}-{AGENT-NAME}.md` | `01-DB-TROUBLESHOOT-AGENT.md` |

---

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/agents/` | LangGraph agent implementations |
| `src/agents/catalog.py` | Agent Catalog with auto-registration |
| `src/agents/skills.py` | Skill execution framework |
| `src/agents/coordinator/` | LangGraph coordinator with orchestrator |
| `src/agents/error_analysis/` | Error Analysis Agent with admin todo management |
| `src/agents/protocol.py` | A2A message protocol for agents |
| `data/admin_todos.json` | Admin todo list for Error Analysis Agent |
| `src/mcp/` | MCP client infrastructure |
| `src/mcp/client.py` | Multi-transport MCP client |
| `src/mcp/registry.py` | Server connection management |
| `src/mcp/catalog.py` | Tool catalog with progressive disclosure |
| `src/mcp/tools/converter.py` | MCP → LangChain ToolConverter |
| `src/mcp/server/` | Unified FastMCP server |
| `src/mcp/server/tools/` | Domain-specific MCP tools |
| `src/llm/` | Multi-LLM factory |
| `src/llm/oca.py` | Oracle Code Assist LLM provider |
| `src/llm/oca_callback_server.py` | OCA OAuth callback server |
| `src/memory/` | Shared memory layer (Redis + ATP) |
| `src/memory/checkpointer.py` | ATP-backed LangGraph checkpointer |
| `src/memory/context.py` | Context compression for long conversations |
| `src/channels/` | Input channel handlers |
| `src/channels/slack.py` | Slack bot integration with catalog and memory |
| `src/channels/slack_catalog.py` | Enterprise troubleshooting catalog |
| `src/channels/conversation.py` | Thread-based conversation memory |
| `src/channels/async_runtime.py` | Shared async event loop for handlers |
| `src/formatting/` | Channel-aware response formatting |
| `src/observability/` | OpenTelemetry tracing |
| `src/evaluation/` | Evaluation framework (judge, metrics) |
| `src/resilience/` | Resilience patterns (Bulkhead, DLQ, HealthMonitor) |
| `src/rag/` | RAG with OCI GenAI embeddings |
| `src/api/` | FastAPI REST API server |
| `src/api/main.py` | Chat, tools, agents, MCP endpoints |
| `tests/` | Pytest test suite (212 tests) |
| `prompts/` | Agent system prompts (7 agents) |
| `scripts/` | Utility scripts (oca_auth.py, etc.) |
| `conductor/` | Project management (plans, workflow) |
| `docs/` | Architecture and reference docs |

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11+ |
| Orchestration | LangGraph, LangChain |
| MCP Server | FastMCP |
| API | FastAPI, Uvicorn |
| LLM Providers | OCA, Anthropic, OpenAI, OCI GenAI |
| Cache | Redis (hot cache) |
| Database | OCI ATP (persistent memory) |
| Observability | OpenTelemetry → OCI APM |
| Testing | Pytest, Coverage (80%+ target) |
| Package Manager | Poetry |
| Linting | Ruff, Black, MyPy |

---

## MCP Server Integration

The system integrates with **4 MCP servers** providing **158+ tools** across all OCI domains.

Configure in `config/mcp_servers.yaml`:

```yaml
servers:
  oci-unified:
    transport: stdio
    command: python
    args: ["-m", "src.mcp.server.main"]
    enabled: true
    domains: [identity, compute, network, cost, security, observability]
    timeout_seconds: 60

  database-observatory:
    transport: stdio
    command: python
    args: ["-m", "src.mcp_server"]
    working_dir: /path/to/mcp-oci-database-observatory
    enabled: true
    domains: [database, opsi, logan, observability]
    timeout_seconds: 60

  oci-infrastructure:
    transport: stdio
    command: uv
    args: ["run", "python", "-m", "mcp_server_oci.server"]
    working_dir: /path/to/mcp-oci
    enabled: true
    domains: [compute, network, security, cost, database, observability]
    timeout_seconds: 60

  finopsai:
    transport: stdio
    command: python
    args: ["-m", "finopsai_mcp.server"]
    working_dir: /path/to/finopsai-mcp
    enabled: true
    domains: [cost, budget, finops, anomaly, forecasting, rightsizing]
    timeout_seconds: 120
```

| Server | Tools | Transport | Purpose |
|--------|-------|-----------|---------|
| **oci-unified** | 31 | stdio | Identity, compute, network, security, cost (60s timeout), discovery |
| **database-observatory** | 50+ | stdio | OPSI, SQLcl, Logan unified for database observability |
| **oci-infrastructure** | 44 | stdio | Full OCI SDK wrapper (fallback for comprehensive coverage) |
| **finopsai** | 33 | stdio | Multicloud cost (OCI/AWS/Azure/GCP), anomaly detection, rightsizing |

**Tool Timeouts:**
- Cost tools: 30 seconds (OCI Usage API can be slow)
- Discovery tools: 60 seconds (full resource discovery)
- Standard tools: 120 seconds (default)

### Dynamic Server Registration

The MCP Server Registry supports dynamic registration at runtime:

```python
from src.mcp.registry import ServerRegistry

registry = ServerRegistry.get_instance()

# Register a new MCP server dynamically
await registry.register_dynamic(
    server_id="custom-server",
    config_dict={
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "custom.server"],
        "working_dir": "/path/to/server",
        "domains": ["custom"],
    },
    auto_connect=True,
)

# Get best server for a domain
server_id = registry.get_best_server_for_domain("database")
```

### Health Checks and Circuit Breaker

The registry includes automatic health monitoring:
- Health check loop runs every 30 seconds
- Failed servers trigger circuit breaker (3 failures = 60s cooldown)
- Automatic reconnection attempts
- Event callbacks for tool updates

```python
# Start health checks
await registry.start_health_checks(interval_seconds=30)

# Register event callback
def on_event(event_type, server_id, data):
    print(f"{event_type}: {server_id}")

registry.on_event(on_event)
```

### Database Observatory Tools

| Tier | Response Time | Tools |
|------|---------------|-------|
| **1 (Cache)** | <100ms | `get_fleet_summary`, `search_databases`, `get_cached_database` |
| **2 (OPSI API)** | 1-5s | `analyze_cpu_usage`, `analyze_memory_usage`, `get_performance_summary` |
| **3 (SQL)** | 5-30s | `execute_sql`, `get_schema_info`, `database_status` |

---

## Database Observatory Workflows

The DB Troubleshoot Agent uses tiered MCP tools for optimal response times:

| Workflow | Steps | Purpose |
|----------|-------|---------|
| `db_rca_workflow` | 7 steps | Full root cause analysis |
| `db_health_check_workflow` | 3 steps | Quick health check via cache |
| `db_sql_analysis_workflow` | 5 steps | Deep SQL-level analysis |

### Workflow Execution

```python
# Execute a skill workflow
result = await agent.execute_skill(
    "db_rca_workflow",
    context={"database_id": "ocid1.autonomousdatabase..."}
)

# Check results
print(f"Health Score: {result.health_score}/100")
print(f"Severity: {result.severity}")
for rec in result.recommendations:
    print(f"- {rec}")
```

---

## Agent Catalog

Agents auto-register when placed in `src/agents/{domain}/`:

```python
# Auto-discovery on startup
from src.agents import initialize_agents
catalog = initialize_agents()

# Get agent by capability
db_agents = catalog.get_by_capability("database-analysis")

# Get agent by skill
rca_agents = catalog.get_by_skill("rca_workflow")
```

See `docs/OCI_AGENT_REFERENCE.md` for complete agent schema.

---

## Shared Memory Layer

| Layer | Backend | Purpose | TTL |
|-------|---------|---------|-----|
| Hot Cache | Redis | Session state, tool results | 1 hour |
| Persistent | OCI ATP | Conversation history, audit | Permanent |
| Checkpoints | LangGraph | Graph state, iterations | Session |

```python
# Usage
from src.memory import SharedMemoryManager
memory = SharedMemoryManager(redis_url, atp_connection)

await memory.set_session_state(session_id, state)
await memory.append_conversation(thread_id, message)
await memory.set_agent_memory(agent_id, "context", value)
```

### Redis Caching with Tag-Based Invalidation

The `OCIResourceCache` provides advanced caching features:

```python
from src.cache import OCIResourceCache

cache = OCIResourceCache(redis_url="redis://localhost:6379")

# Set with tags for group invalidation
await cache.set(
    "compute:instances:prod",
    instances_data,
    ttl=3600,
    tags=["compartment:prod", "service:compute"]
)

# Invalidate all entries with a tag
deleted = await cache.invalidate_by_tag("compartment:prod")

# Stale-while-revalidate pattern
async def fetch_instances():
    return await oci_client.list_instances()

data, was_stale = await cache.get_with_swr(
    "instances",
    refresh_func=fetch_instances,
    ttl=3600,
    stale_ttl=60  # Serve stale for 60s while refreshing
)

# Event subscription for cache updates
async def handle_cache_event(event_type, key, data):
    print(f"Cache {event_type}: {key}")

await cache.subscribe_events(handle_cache_event)
```

**Cache Event Types:**
| Event | Description |
|-------|-------------|
| `cache:set` | Key was set or updated |
| `cache:delete` | Key was explicitly deleted |
| `cache:invalidate` | Key invalidated via tag |
| `cache:expire` | Key expired naturally |

### Per-Interaction Tool Refresh

The Tool Catalog ensures fresh tool availability per interaction:

```python
from src.mcp.catalog import ToolCatalog

catalog = ToolCatalog(registry)

# Before each interaction, ensure tools are fresh
await catalog.ensure_fresh()  # Refreshes if stale (>5 min default)

# Force refresh
await catalog.ensure_fresh(force=True)

# Check staleness
if catalog.is_stale():
    await catalog.refresh()

# Get available tools (auto-refreshes if needed)
tools = await catalog.get_all_tools()
```

---

## Core Components

### 1. Agent Catalog (`src/agents/catalog.py`)
- Auto-discovery from agents directory
- Capability-based agent lookup
- Health monitoring and status tracking

### 2. Skill Framework (`src/agents/skills.py`)
- Reusable workflow definitions (SkillDefinition)
- Step-by-step execution with error handling
- Pre-defined skills: RCA, Cost Analysis, Security Assessment
- Tool validation before execution

### 3. LangGraph Coordinator (`src/agents/coordinator/`)
- **8 Graph Nodes**: input → classifier → router → workflow|parallel|agent → action → output
- Workflow-first routing (70%+ deterministic)
- Intent classification with confidence scoring
- Multi-agent orchestration
- State persistence via checkpoints
- **Slack Integration**: Slack handler uses LangGraph coordinator for routing (enable via `USE_LANGGRAPH_COORDINATOR=true`)
- **Fallback Routing**: If coordinator fails, falls back to keyword-based agent routing

**Routing Thresholds:**
| Threshold | Value | Action |
|-----------|-------|--------|
| Workflow | ≥ 0.80 | Direct workflow execution |
| Parallel | ≥ 0.60 + 2+ domains | Multi-agent parallel |
| Agent | ≥ 0.60 | Single agent with tools |
| Clarify | 0.30 - 0.60 | Ask clarifying question |
| Escalate | < 0.30 | Human handoff |

```python
# Slack → LangGraph flow (in src/channels/slack.py)
async def _invoke_coordinator(self, text, user_id, ...):
    if os.getenv("USE_LANGGRAPH_COORDINATOR", "true") == "true":
        result = await self._invoke_langgraph_coordinator(text, user_id, thread_id)
        if result and result.get("success"):
            return result
    # Fallback to keyword routing
    return await self._route_to_agent(text, catalog, user_id)
```

### 4. Parallel Orchestrator (`src/agents/coordinator/orchestrator.py`)
- Multi-agent parallel execution for complex cross-domain queries
- Automatic task decomposition by domain
- Result synthesis using LLM
- Bounded concurrency (3-5 agents max)
- Loop prevention with max iterations

**When Parallel Execution is Used:**
- Query involves 2+ domains (e.g., "database" + "cost")
- Intent category is ANALYSIS or TROUBLESHOOT
- Confidence score ≥ 0.60

```python
# Parallel execution is automatic via routing
result = await coordinator.invoke(
    "Analyze database performance and associated costs",
    thread_id=thread_id,
)
# Routes to: db-troubleshoot-agent AND finops-agent in parallel
```

### 5. Pre-built Workflows (`src/agents/coordinator/workflows.py`)
- **16 deterministic workflows** for common OCI operations
- Execute without LLM reasoning for fast, predictable responses
- Triggered when classifier confidence ≥ 0.80
- **40+ intent aliases** for reliable routing

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

**Cost Workflow Details:**
The `cost_summary` workflow has 10 intent aliases to reliably capture cost queries:
- `get_tenancy_costs`, `tenancy_costs`, `spending`, `show_spending`
- `monthly_cost`, `how_much_spent`, `get_costs`, `show_costs`

This ensures queries like "How much am I spending?" route to the fast workflow instead of the full FinOps agent.

**Slack Visualization:**
Cost responses in Slack display as native table blocks with:
- Summary header (total spend, period, days)
- Service breakdown table (Service | Cost | %)
- Auto-detected list data also renders as tables

```python
# Workflows are registered automatically via create_coordinator()
from src.agents.coordinator import create_coordinator

coordinator = await create_coordinator(llm=llm)
# 15 workflows auto-registered

# Or manually register a custom workflow
async def my_workflow(query, entities, tool_catalog, memory) -> str:
    result = await tool_catalog.execute("my_tool", entities)
    return result

coordinator.register_workflow("my_workflow", my_workflow)
```

### 6. MCP Client (`src/mcp/`)
- Multi-transport support: stdio, HTTP, SSE
- Dynamic server registry with health tracking
- Unified tool catalog with namespacing
- Progressive disclosure via `search_tools`
- Tool tier classification (1-4) for risk management

### 7. Shared Memory (`src/memory/`)
- Tiered storage (Redis + ATP)
- Cross-agent context sharing
- Conversation history persistence

### 8. Observability (`src/observability/`)
- OpenTelemetry tracing to OCI APM via OTLP
- Per-agent dedicated OCI Logging with trace_id correlation
- Structured logging via structlog
- Agent execution metrics

```python
# Trace-log correlation
from src.observability import init_observability, get_trace_id

# Initialize on startup (enables both APM and Logging)
init_observability(agent_name="db-troubleshoot-agent")

# Logs automatically include trace_id for APM correlation
logger.info("Processing request", trace_id=get_trace_id())
```

### 9. Response Formatting (`src/formatting/`)
- Channel-aware output (Slack, Teams, Markdown)
- Structured responses with metrics, tables, code blocks
- Native Slack table blocks for list data (compartments, instances, etc.)
- Auto-detection and formatting of JSON array responses
- Severity-based styling

**Slack Table Block Example:**
```python
from src.formatting.slack import SlackFormatter

formatter = SlackFormatter()

# Cost data (auto-detected from service/cost/percent keys)
cost_result = formatter.format_table_from_list(
    items=[{"service": "Database", "cost": "22,296.92 ILS", "percent": "37.5%"}],
    columns=["service", "cost", "percent"],
    title=":bar_chart: *Top Services by Spend*",
)

# Generic list data
result = formatter.format_table_from_list(
    items=[{"name": "prod", "state": "ACTIVE", "id": "ocid1..."}],
    columns=["name", "state"],
    title="Compartments",
)
# Returns native Slack table block (max 100 rows, 20 columns)
```

### 10. API Server (`src/api/main.py`)
- FastAPI REST endpoints for programmatic access
- Chat endpoint with LangGraph coordinator
- Tool listing, execution, and discovery
- Agent status and management
- MCP server monitoring

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with component status |
| `/chat` | POST | Process chat message through coordinator (returns `content_type`, `structured_data`) |
| `/tools` | GET | List tools with filtering (query, domain, tier) |
| `/tools/execute` | POST | Execute a specific tool |
| `/agents` | GET | List registered agents |
| `/mcp/servers` | GET | MCP server status |
| `/slack/status` | GET | Slack connection status (bot_id, team, connected) |

### 11. Slack Integration (`src/channels/`)

Enterprise-grade Slack integration with conversation memory and troubleshooting catalog:

**Components:**
| Component | Purpose |
|-----------|---------|
| `slack.py` | Main Slack handler with 3-second ack pattern |
| `slack_catalog.py` | Interactive troubleshooting catalog |
| `conversation.py` | Thread-based conversation memory |

**Troubleshooting Catalog (`slack_catalog.py`):**
- 6 categories: Database, Cost, Security, Infrastructure, Logs, Discovery
- 20+ quick actions with one-click execution
- 3 enterprise runbooks with guided steps
- Context-aware follow-up suggestions
- Error recovery blocks with actionable alternatives

**Conversation Memory (`conversation.py`):**
- Thread-based message tracking
- Redis persistence (24-hour TTL)
- Topic detection for routing hints
- Query type tracking for follow-ups

**Slack Commands:**
| Command | Description |
|---------|-------------|
| `help`, `?` | Show help menu with quick actions |
| `catalog`, `menu` | Open troubleshooting catalog |
| `runbook`, `runbooks` | Show available runbooks |

**Interactive Buttons:**
- Category navigation (Database, Cost, Security, etc.)
- Quick action execution (one-click workflows)
- Follow-up suggestions after responses
- Error recovery options

```python
# Catalog category selection triggers quick actions
# User clicks: Database → DB Health Check
# Handler executes: "database health check" query

# Follow-up suggestions appear after each response
# Based on query type (cost → "Show database costs specifically")
```

### 12. ToolConverter (`src/mcp/tools/converter.py`)
- Converts MCP tools to LangChain StructuredTools
- Dynamic Pydantic model generation from JSON schema
- Domain-based tool filtering via `DOMAIN_PREFIXES`
- Confirmation callback for high-risk (tier 4) tools

```python
from src.mcp.tools.converter import ToolConverter

converter = ToolConverter(tool_catalog)

# Get all tools up to tier 3
tools = converter.to_langchain_tools(max_tier=3)

# Get only database tools
db_tools = converter.get_domain_tools("database")

# Get safe tools only (tier 1-2, no risk)
safe_tools = converter.get_safe_tools()
```

### 13. RAG with OCI AI Studio (`src/rag/`)

Retrieval-Augmented Generation using OCI Generative AI service:

| Component | Purpose |
|-----------|---------|
| `OCIEmbeddings` | OCI GenAI embeddings (Cohere embed-v3) |
| `RedisVectorStore` | Vector storage with VSS or brute-force search |
| `RAGRetriever` | High-level retrieval with chunking |
| `DirectoryLoader` | Load documents from directories |
| `MarkdownLoader` | Parse markdown with frontmatter |

**Usage Example:**
```python
from src.rag import get_retriever, OCIDocumentationLoader

# Initialize retriever
retriever = await get_retriever("oci-knowledge")

# Ingest OCI documentation
loader = OCIDocumentationLoader()
docs = loader.load_all()
await retriever.ingest_documents(docs)

# Retrieve context for agent
result = await retriever.retrieve("How do I troubleshoot database performance?")
print(result.context)  # Combined context from relevant docs

# Use in agent prompt
agent_context = f"""
Based on the following documentation:
{result.context}

Answer the user's question: ...
"""
```

**Environment Variables:**
```bash
# OCI GenAI Embeddings
OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
OCI_GENAI_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com

# Vector Store
REDIS_URL=redis://localhost:6379
```

### 14. Resilience Infrastructure (`src/resilience/`)

Production-grade resilience patterns for fault tolerance and self-healing:

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `DeadLetterQueue` | Persist failed operations | Redis-backed, retry support, failure pattern tracking |
| `Bulkhead` | Resource isolation | Partitioned semaphores, per-domain limits |
| `HealthMonitor` | Component health tracking | Auto-restart, event callbacks, metrics |
| `RateLimitedLLM` | LLM concurrency control | Semaphore-based, timeout support |

**Bulkhead Partitions:**
| Partition | Max Concurrent | Tool Prefixes |
|-----------|----------------|---------------|
| DATABASE | 3 | `oci_database_`, `oci_opsi_`, `execute_sql` |
| INFRASTRUCTURE | 5 | `oci_compute_`, `oci_network_` |
| COST | 2 | `oci_cost_` |
| SECURITY | 3 | `oci_security_` |
| DISCOVERY | 2 | `oci_search_`, `oci_list_` |
| LLM | 5 | LLM calls |
| DEFAULT | 10 | Catch-all |

**Circuit Breaker Integration:**
The tool catalog includes circuit breaker logic that:
- Tracks consecutive failures per MCP server
- Opens circuit after 3 failures (60s cooldown)
- Rejects tool calls to unhealthy servers immediately
- Auto-closes circuit after successful health check

**Usage Example:**
```python
from src.resilience import (
    Bulkhead, DeadLetterQueue, HealthMonitor,
    HealthCheck, HealthStatus
)

# Bulkhead for resource isolation
bulkhead = Bulkhead.get_instance()
async with bulkhead.acquire("database"):
    result = await execute_database_operation()

# Deadletter queue for failed operations
dlq = DeadLetterQueue(redis_url="redis://localhost:6379")
await dlq.enqueue(
    failure_type="tool_call",
    operation="oci_cost_get_summary",
    error="Timeout after 30s",
    params={"days": 30},
)

# Health monitor with auto-restart
monitor = HealthMonitor.get_instance()
monitor.register_check(HealthCheck(
    name="mcp_database",
    check_func=check_mcp_health,
    restart_func=restart_mcp_server,
    failure_threshold=3,
))
await monitor.start()
```

### 15. Self-Healing Framework (`src/agents/self_healing/`)

LLM-powered error analysis and automatic recovery:

| Component | Purpose |
|-----------|---------|
| `ErrorAnalyzer` | Categorize errors, suggest recovery actions |
| `ParameterCorrector` | Fix incorrect tool parameters |
| `LogicValidator` | Pre-execution validation |
| `RetryStrategy` | Smart retry with exponential backoff |
| `SelfHealingMixin` | Mixin for any agent to inherit self-healing |

**Error Categories:**
| Category | Recovery Action |
|----------|-----------------|
| `PERMISSION` | Suggest IAM policy fix |
| `NOT_FOUND` | Parameter correction |
| `TIMEOUT` | Retry with backoff |
| `RATE_LIMIT` | Wait and retry |
| `VALIDATION` | Correct parameters |
| `TRANSIENT` | Simple retry |

**Using Self-Healing in Agents:**
```python
from src.agents.base import BaseAgent
from src.agents.self_healing import SelfHealingMixin

class MyAgent(BaseAgent, SelfHealingMixin):
    def __init__(self, llm, tool_catalog, ...):
        super().__init__(...)
        self.init_self_healing(llm, max_retries=3)

    async def invoke(self, query):
        # Use healing_call_tool for auto-retry with correction
        result = await self.healing_call_tool(
            "oci_database_execute_sql",
            {"query": "SELECT * FROM users"},
            user_intent=query,
        )
        return result
```

---

## Environment Variables

```bash
# OCI Configuration
OCI_CONFIG_FILE=~/.oci/config
OCI_CLI_PROFILE=DEFAULT

# LLM Provider Selection
# Options: oracle_code_assist, anthropic, openai, lm_studio
LLM_PROVIDER=oracle_code_assist

# Oracle Code Assist (OCA) - Primary LLM
OCA_MODEL=oca/gpt5
OCA_CALLBACK_HOST=127.0.0.1
OCA_CALLBACK_PORT=48801

# Alternative: Anthropic Claude
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-xxx
# ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Alternative: OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-xxx
# OPENAI_MODEL=gpt-4o-mini

# Slack Integration
SLACK_BOT_TOKEN=xoxb-xxx
SLACK_APP_TOKEN=xapp-xxx
SLACK_SIGNING_SECRET=xxx

# OCI APM (OpenTelemetry)
OCI_APM_ENDPOINT=https://xxx.apm-agt.region.oci.oraclecloud.com
OCI_APM_PRIVATE_DATA_KEY=xxx
OTEL_SERVICE_NAME=oci-coordinator-agent

# OCI Logging (Per-Agent Logs with trace correlation)
OCI_LOG_GROUP_ID=ocid1.loggroup.oc1.region.xxx
OCI_LOG_ID_COORDINATOR=ocid1.log.oc1.region.xxx
OCI_LOG_ID_DB_TROUBLESHOOT=ocid1.log.oc1.region.xxx
OCI_LOG_ID_LOG_ANALYTICS=ocid1.log.oc1.region.xxx
OCI_LOG_ID_SECURITY_THREAT=ocid1.log.oc1.region.xxx
OCI_LOG_ID_FINOPS=ocid1.log.oc1.region.xxx
OCI_LOG_ID_INFRASTRUCTURE=ocid1.log.oc1.region.xxx
OCI_LOGGING_REGION=eu-frankfurt-1
OCI_LOGGING_ENABLED=true

# ShowOCI Resource Discovery & Caching
SHOWOCI_CACHE_ENABLED=true              # Enable cache on startup
OCI_PROFILES=DEFAULT                     # Comma-separated OCI profiles
SHOWOCI_REFRESH_HOURS=4                  # Cache refresh interval (0=disabled)
REDIS_URL=redis://localhost:6379         # Redis connection

# LangGraph Coordinator Settings
USE_LANGGRAPH_COORDINATOR=true           # Enable LangGraph for Slack routing (default: true)

# RAG with OCI GenAI (AI Studio)
OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx  # Compartment for GenAI
OCI_GENAI_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com
```

### LLM Provider Usage

```python
from src.llm import get_llm

# Get LLM based on .env.local configuration
llm = get_llm()

# Use with agents
from src.agents.database.troubleshoot import DbTroubleshootAgent
agent = DbTroubleshootAgent(llm=llm, tool_catalog=catalog)
response = await agent.invoke("Check database performance")
```

### OCA Authentication (Oracle Code Assist)

OCA uses OAuth 2.0 with PKCE for authentication. The system handles this automatically.

**Automatic Flow (Slack Integration):**
1. When the Slack bot starts, an OAuth callback server starts automatically on port 48801
2. If OCA authentication is needed, Slack shows a "Login with Oracle SSO" button
3. User clicks the button → browser opens Oracle IDCS SSO
4. User completes SSO → browser redirects to `localhost:48801/auth/oca`
5. Callback server exchanges the code for tokens and caches them
6. User sees "Authentication Successful!" and can return to Slack

**Token Management:**
- Access tokens are cached in `~/.oca/token.json`
- Tokens auto-refresh using the refresh token (valid for 8 hours)
- Token expiry is handled automatically by the `OCATokenManager`

**Manual Login (Standalone):**
```bash
# Full login flow (opens browser, starts callback server)
poetry run python scripts/oca_auth.py

# Check token status
poetry run python scripts/oca_auth.py --status

# Callback-only mode (for debugging)
poetry run python scripts/oca_auth.py --callback-only
```

**Architecture:**
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Slack Bot     │────▶│  Oracle IDCS     │────▶│ Callback Server │
│  (generates     │     │  (SSO login)     │     │ (localhost:48801)│
│   PKCE + URL)   │     │                  │     │                  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                  ┌───────────────┐
                                                  │ Token Cache   │
                                                  │ ~/.oca/       │
                                                  │ token.json    │
                                                  └───────────────┘
```

**Key Files:**
| File | Purpose |
|------|---------|
| `src/llm/oca.py` | OCA LangChain model, token manager |
| `src/llm/oca_callback_server.py` | Background OAuth callback server |
| `src/channels/slack.py` | Slack auth UI (get_oca_auth_url) |
| `scripts/oca_auth.py` | Standalone login script |

---

## Coding Conventions

| Aspect | Convention |
|--------|------------|
| Imports | Absolute from `src.` |
| Types | Full type hints, Pydantic models |
| Logging | `structlog` via `get_logger()` |
| Errors | Specific exceptions, never bare `except` |
| Tests | Mirror source structure in `tests/` |
| Docstrings | Google-style |
| Coverage | 80%+ target |

---

## Agent Capabilities

### FinOps Agent (`src/agents/finops/agent.py`)

Cost management with built-in analysis:

| Feature | Description | MCP Tools |
|---------|-------------|-----------|
| Cost Analysis | Breakdown by service with 30s timeout | `oci_cost_get_summary` |
| Anomaly Detection | High-concentration analysis from cost data | Built-in analysis |
| Recommendations | Rightsizing suggestions based on spend | Built-in heuristics |

**Note**: The cost tool has a 30-second timeout. If the OCI Usage API is slow, check the OCI console directly.

```python
# Simple cost query (uses deterministic workflow)
result = await coordinator.invoke("How much am I spending?")
# Routes to cost_summary workflow → fast response

# Deep analysis (uses FinOps agent)
result = await finops_agent.invoke("Analyze cost trends and recommend optimizations")
# Returns: breakdown, trends, recommendations
```

### Security Agent (`src/agents/security/agent.py`)

Threat detection with MITRE ATT&CK mapping:

| MITRE Technique | Cloud Guard Problem | Tactic |
|-----------------|---------------------|--------|
| T1078 | SUSPICIOUS_LOGIN | Initial Access |
| T1098 | IAM_POLICY_CHANGE | Persistence |
| T1562 | SECURITY_GROUP_CHANGE | Defense Evasion |
| T1567 | DATA_EXFILTRATION | Exfiltration |
| T1496 | CRYPTO_MINING | Impact |

```python
# Security analysis with MITRE mapping
result = await security_agent.invoke("Analyze security threats")
# Returns: Cloud Guard problems with MITRE technique IDs
```

### Log Analytics Agent (`src/agents/log_analytics/agent.py`)

Pattern detection and cross-service correlation:

| Feature | Description |
|---------|-------------|
| Pattern Detection | Frequency analysis of error patterns (ORA-, timeout, auth failures) |
| Anomaly Detection | High-frequency bursts, error source concentration |
| Trace Correlation | Cross-service request tracing via trace_id |
| Temporal Correlation | Concurrent error detection across services |

```python
# Log analysis workflow
result = await log_agent.invoke("Search for errors in the last hour")
# Returns: patterns, anomalies, cross-service correlations
```

### Error Analysis Agent (`src/agents/error_analysis/agent.py`)

Log error detection with admin todo management:

| Feature | Description | MCP Tools |
|---------|-------------|-----------|
| Error Identification | Find errors in OCI logs | `oci_logging_search_logs` |
| Pattern Detection | Detect common error patterns (ORA-, OOM, auth failures) | Built-in regex |
| LLM Analysis | Analyze patterns for root cause | LangChain LLM |
| Admin Todo Management | Create action items for significant errors | `AdminTodoManager` |

**Error Patterns Detected:**
| Pattern | Severity | Category |
|---------|----------|----------|
| ORA-00060 (Deadlock) | CRITICAL | database |
| ORA-04031 (Shared Pool) | CRITICAL | database |
| OutOfMemory/OOM | CRITICAL | compute |
| Connection timeout/refused | HIGH | network |
| Authentication failed | HIGH | security |
| HTTP 4xx/5xx | MEDIUM | api |
| Rate limited | MEDIUM | api |

**Admin Todo Storage:** JSON file at `data/admin_todos.json`

```python
# Analyze logs for errors
result = await error_agent.invoke("Scan logs for errors", time_range_hours=1)
# Returns: patterns found, todos created, recommendations

# Get pending admin todos
todos = await error_agent.get_pending_todos()

# Resolve a todo
await error_agent.resolve_todo("todo-abc123", "Fixed by increasing connection pool")
```

---

## Agent System Prompts

| Agent | File | Triggers |
|-------|------|----------|
| Coordinator | `00-COORDINATOR-AGENT.md` | All requests (entry) |
| DB Troubleshoot | `01-DB-TROUBLESHOOT-AGENT.md` | database, slow query, AWR |
| Log Analytics | `02-LOG-ANALYTICS-AGENT.md` | logs, errors, audit |
| Security | `03-SECURITY-THREAT-AGENT.md` | security, threat, MITRE |
| FinOps | `04-FINOPS-AGENT.md` | cost, budget, spending |
| Infrastructure | `05-INFRASTRUCTURE-AGENT.md` | compute, network, VCN |
| Error Analysis | `src/agents/error_analysis/agent.py` | error scan, admin todos, patterns |

---

## Common Tasks

### Adding a New Agent

1. Create agent class in `src/agents/{domain}/{function}.py`
2. Implement `BaseAgent` with `get_definition()` classmethod
3. Agent auto-registers via catalog discovery
4. Add tests in `tests/test_{agent}.py`

### Adding a Skill/Workflow

```python
# In src/agents/skills.py or your agent module
from src.agents.skills import SkillDefinition, SkillStep, SkillRegistry

MY_SKILL = SkillDefinition(
    name="my_workflow",
    description="Description of what this workflow does",
    steps=[
        SkillStep("step_1", "First step", required_tools=["tool_a"]),
        SkillStep("step_2", "Second step", required_tools=["tool_b"]),
    ],
    required_tools=["tool_a", "tool_b"],
    tags=["domain", "category"],
)

# Register globally
SkillRegistry.get_instance().register(MY_SKILL)

# Execute from agent
result = await agent.execute_skill("my_workflow", context={"key": "value"})
```

### Adding an MCP Tool

1. Add tool function in `src/mcp/server/tools/{domain}.py`
2. Register via `register_{domain}_tools(mcp)` in `server/main.py`
3. Assign tool tier in `src/mcp/catalog.py` TOOL_TIERS
4. Tool auto-discovers via progressive disclosure

### Running Tests

```bash
# All tests
poetry run pytest --cov=src

# Specific test
poetry run pytest tests/test_mcp_server.py -v

# With coverage report
poetry run pytest --cov=src --cov-report=html

# Run evaluation
poetry run python -m src.evaluation.runner --mock --verbose
```

---

## Reference Documentation

| Document | Purpose |
|----------|---------|
| `docs/OCI_AGENT_REFERENCE.md` | Agent schema, naming, memory |
| `docs/ARCHITECTURE.md` | System architecture |
| `docs/ARCHITECTURE_DESIGN.md` | Unified tool naming and routing design |
| `docs/CODE_REVIEW.md` | Comprehensive code review and gap analysis |
| `docs/SHOWOCI_INTEGRATION.md` | ShowOCI discovery & caching |
| `AGENT.md` | Agent design patterns |
| `conductor/product.md` | Product requirements |
| `conductor/tech-stack.md` | Technology decisions |
| `conductor/workflow.md` | Development workflow |

---

## Security Notes

- Never commit OCIDs, API keys, or tokens
- Use `.env.local` for secrets (not in git)
- All sensitive config via OCI Vault in production
- Validate and sanitize all user inputs
- Audit logging enabled for all agent actions

---

## External References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [OCI APM/OTEL](https://docs.oracle.com/en-us/iaas/application-performance-monitoring/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [ShowOCI (Oracle)](https://github.com/oracle/oci-python-sdk/tree/master/examples/showoci)
- [Anthropic Agent Best Practices](https://www.anthropic.com/engineering/building-effective-agents)

## Feature Mapping Matrix

> **Full details**: See `docs/FEATURE_MAPPING.md` for comprehensive tool-to-feature mapping.

| Feature | Agent | Primary MCP Tools | Test Status |
|---------|-------|-------------------|-------------|
| **Identity Management** | `CoordinatorAgent` | `oci_list_compartments`, `oci_get_tenancy`, `oci_list_regions` | ✅ 6/6 |
| **Compute Operations** | `InfrastructureAgent` | `oci_compute_list_instances`, `oci_compute_find_instance` | ✅ 2/2 |
| **Network Operations** | `InfrastructureAgent` | `oci_network_list_vcns`, `oci_network_list_subnets` | ✅ 3/3 |
| **Cost Analysis** | `FinOpsAgent` | `oci_cost_get_summary` (30s timeout) | ✅ 3/3 |
| **Security Audit** | `SecurityThreatAgent` | `oci_security_list_users` | ✅ 1/1 |
| **DB Troubleshooting** | `DbTroubleshootAgent` | `oci_database_execute_sql`, `oci_opsi_get_fleet_summary` | ✅ |
| **Log Analysis** | `LogAnalyticsAgent` | `oci_observability_query_logs`, `oci_logan_search` | ✅ |
| **Resource Discovery** | `CoordinatorAgent` | `oci_discovery_summary`, `oci_discovery_search` | ✅ 3/3 |

**MCP Server Distribution**:
| Server | Tools | Domains |
|--------|-------|---------|
| oci-unified | 31 | identity, compute, network, cost, security, discovery |
| database-observatory | 50+ | database, opsi, logan |
| oci-infrastructure | 44 | Full OCI SDK wrapper |
| finopsai-mcp | 33 | Multicloud FinOps |

## Deployment

The system is deployed via GitHub Actions to OCI Cloud Run or Cloud Shell.
- **Workflow**: `.github/workflows/build-deploy.yaml`
- **Target**: OCI Cloud Run (Containerized) or Cloud Shell (Scripted)
