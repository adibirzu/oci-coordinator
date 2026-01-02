# OCI AI Agent Coordinator

An intelligent orchestration system that manages specialized AI agents for Oracle Cloud Infrastructure operations. Built with LangGraph, MCP (Model Context Protocol), and full OCI observability integration.

## Disclaimer

This software was created to showcase Oracle Cloud Infrastructure (OCI) AI Integration capabilities and demonstrate how to expand them using third-party services and AI tools. This is **NOT an official Oracle product** - it is a personal project demonstrating integration possibilities with OCI AI agents and assistants.

## Features

- **Multi-Agent Orchestration**: Coordinate 6 specialized agents for database, logging, security, cost, infrastructure, and error analysis
- **Workflow-First Design**: 70%+ deterministic workflows via MCP tools (16 pre-built workflows); LLM reasoning for complex analysis
- **Database Observatory Integration**: Full OPSI, SQLcl, and Logan Analytics via MCP
- **Slack Integration**: Real-time chatbot with Socket Mode and 3-second ack pattern
- **REST API**: FastAPI server with SSE streaming support
- **Self-Healing**: Automatic error recovery with parameter correction and smart retries
- **Resilience Infrastructure**: Bulkhead isolation, circuit breakers, dead letter queues
- **RAG**: OCI GenAI embeddings with Redis vector store for documentation context
- **OCI APM Tracing**: Full OpenTelemetry integration with trace-log correlation
- **OCI Logging**: Per-agent dedicated logs with trace ID injection
- **Multi-LLM Support**: Anthropic Claude, OpenAI GPT, OCI GenAI, Oracle Code Assist

## Architecture

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

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- OCI CLI configured (`~/.oci/config`)
- Slack App (for bot integration)
- Database Observatory MCP Server

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/oci-coordinator.git
cd oci-coordinator

# Install dependencies
poetry install

# Copy environment template
cp .env.example .env.local

# Edit .env.local with your configuration (see below)
```

### Configuration

Create `.env.local` with your settings:

```bash
# OCI Configuration
OCI_CONFIG_FILE=~/.oci/config
OCI_CLI_PROFILE=DEFAULT

# LLM Provider
LLM_DEFAULT_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxx

# Slack Integration
SLACK_BOT_TOKEN=xoxb-xxx
SLACK_APP_TOKEN=xapp-xxx
SLACK_SIGNING_SECRET=xxx

# OCI APM (OpenTelemetry)
OCI_APM_ENDPOINT=https://xxx.apm-agt.region.oci.oraclecloud.com
OCI_APM_PRIVATE_DATA_KEY=xxx
OTEL_SERVICE_NAME=oci-coordinator-agent

# OCI Logging
OCI_LOG_GROUP_ID=ocid1.loggroup.oc1.region.xxx
OCI_LOG_ID_COORDINATOR=ocid1.log.oc1.region.xxx
OCI_LOG_ID_DB_TROUBLESHOOT=ocid1.log.oc1.region.xxx
OCI_LOG_ID_LOG_ANALYTICS=ocid1.log.oc1.region.xxx
OCI_LOG_ID_SECURITY_THREAT=ocid1.log.oc1.region.xxx
OCI_LOG_ID_FINOPS=ocid1.log.oc1.region.xxx
OCI_LOG_ID_INFRASTRUCTURE=ocid1.log.oc1.region.xxx
OCI_LOGGING_REGION=eu-frankfurt-1
OCI_LOGGING_ENABLED=true
```

### Running

```bash
# Start Slack bot (Socket Mode)
poetry run python -m src.main --mode slack

# Start API server
poetry run python -m src.main --mode api --port 3001

# Start both
poetry run python -m src.main --mode both

# Run tests (212 tests)
poetry run pytest --cov=src

# Lint and format
poetry run ruff check src/
poetry run black src/
```

## Specialized Agents

| Agent | Capabilities | MCP Tools | Triggers |
|-------|--------------|-----------|----------|
| **DB Troubleshoot** | Performance RCA, SQL analysis | `get_performance_summary`, `analyze_cpu_usage`, `execute_sql` | database, slow, AWR, wait events |
| **Log Analytics** | Log search, pattern detection | `query_logs`, `search_audit` | logs, errors, audit |
| **Security Threat** | MITRE mapping, compliance | `list_problems`, `get_findings` | security, threat, compliance |
| **FinOps** | Cost analysis, optimization | `get_cost_summary`, `oci_cost_spikes`, `oci_cost_budget_status` | cost, budget, spending |
| **Infrastructure** | Compute, network management | `list_instances`, `manage_vcn`, `start_by_name` | instance, VM, VCN |
| **Error Analysis** | Log error detection, admin todos | `oci_logging_search_logs`, pattern detection | error scan, patterns, admin |

### FinOps AI Agent

The FinOps agent provides advanced cost management through the **finopsai-mcp** server:

```
Agent capabilities from finopsai-mcp:
- Cost spike detection with threshold alerts
- Rightsizing recommendations for compute resources
- Budget status tracking and alerts
- Service drilldown with optimization suggestions
- Multicloud support (OCI, AWS, Azure, GCP)
```

The FinOps agent workflow:
1. **Analyze Query** - Extract time range and service filters
2. **Get Costs** - Retrieve cost summary and breakdown by service
3. **Detect Anomalies** - Use `oci_cost_spikes` for anomaly detection
4. **Generate Recommendations** - Call `oci_cost_service_drilldown` for rightsizing

### Instance Operations by Name

The unified MCP server includes convenience tools for operating on instances by display name:

```bash
# Find instances by name (partial match supported)
oci_compute_find_instance(instance_name="prod", compartment_id="ocid1...")

# Start/Stop/Restart by name (auto-resolves to OCID)
oci_compute_start_by_name(instance_name="my-instance", compartment_id="ocid1...")
oci_compute_stop_by_name(instance_name="my-instance", compartment_id="ocid1...")
oci_compute_restart_by_name(instance_name="my-instance", compartment_id="ocid1...")
```

These tools handle automatic OCID resolution, partial name matching (case-insensitive), and disambiguation when multiple instances match.

## Database Observatory Skills

The DB Troubleshoot Agent uses tiered MCP tools for optimal response times:

| Tier | Response Time | Tools |
|------|---------------|-------|
| **1 (Cache)** | <100ms | `get_fleet_summary`, `search_databases`, `get_cached_database` |
| **2 (OPSI API)** | 1-5s | `analyze_cpu_usage`, `analyze_memory_usage`, `get_performance_summary` |
| **3 (SQL)** | 5-30s | `execute_sql`, `get_schema_info`, `database_status` |

### Available Workflows

- `db_rca_workflow` - 7-step root cause analysis
- `db_health_check_workflow` - Fast health check via cache
- `db_sql_analysis_workflow` - Deep SQL-level analysis

## MCP Server Configuration

Configure MCP servers in `config/mcp_servers.yaml`:

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
  timeout_seconds: 120  # Increased for large compartments (25+ resources)
  retry_attempts: 3
  backoff_multiplier: 2
```

## API Server

The FastAPI server provides REST endpoints for programmatic access:

```bash
# Start API server
poetry run python -m src.main --mode api --port 3001
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with component status |
| `/status` | GET | Detailed system status |
| `/chat` | POST | Process chat message through coordinator |
| `/chat/stream` | POST | Stream chat response (SSE) |
| `/tools` | GET | List available tools with filtering |
| `/tools/{name}` | GET | Get tool details |
| `/tools/execute` | POST | Execute a specific tool |
| `/agents` | GET | List registered agents |
| `/agents/{role}` | GET | Get agent details |
| `/mcp/servers` | GET | List MCP server status |
| `/stats` | GET | System statistics |

### Example Chat Request

```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the cost summary for this month?"}'
```

### External MCP Server References

> **Disclaimer**: The external MCP servers listed below are personal projects created to demonstrate OCI AI integration capabilities. These are **NOT official Oracle products** and are not endorsed by Oracle Corporation.

| Server | Description | GitHub |
|--------|-------------|--------|
| **oci-unified** | Built-in server with ShowOCI-style discovery | `src/mcp/server/` (this project) |
| **database-observatory** | OPSI, SQLcl, Logan Analytics | [adibirzu/mcp-oci-database-observatory](https://github.com/adibirzu/mcp-oci-database-observatory) |
| **oci-infrastructure** | Full OCI management (mcp-oci) | [adibirzu/mcp-oci](https://github.com/adibirzu/mcp-oci) |
| **finopsai-mcp** | Multicloud FinOps with anomaly detection | [adibirzu/finopsai-mcp](https://github.com/adibirzu/finopsai-mcp) |

### Why oci-unified vs mcp-oci?

| Feature | oci-unified | mcp-oci |
|---------|-------------|---------|
| ShowOCI-style discovery with Redis caching | ✅ | ❌ |
| Instance operations by name | ✅ | ❌ (requires OCID) |
| Troubleshoot skills | ✅ | ❌ |
| Full OCI API coverage | Basic | ✅ Comprehensive |

Use **oci-unified** for quick discovery and name-based operations. Use **mcp-oci** for comprehensive OCI management requiring full API access.

## Project Structure

```
oci-coordinator/
├── src/
│   ├── agents/                   # LangGraph agent implementations
│   │   ├── catalog.py            # Agent registry with auto-discovery
│   │   ├── base.py               # BaseAgent class
│   │   ├── skills.py             # Skill definitions and executor
│   │   ├── react_agent.py        # ReAct agent with dynamic tool discovery
│   │   ├── coordinator/          # LangGraph coordinator
│   │   │   ├── graph.py          # StateGraph with intent routing
│   │   │   ├── orchestrator.py   # Parallel orchestration with loop prevention
│   │   │   ├── workflows.py      # 16 pre-built deterministic workflows
│   │   │   └── state.py          # Conversation state management
│   │   ├── database/             # DB Troubleshoot Agent
│   │   ├── log_analytics/        # Log Analytics Agent
│   │   ├── security/             # Security Threat Agent
│   │   ├── finops/               # FinOps Agent
│   │   ├── infrastructure/       # Infrastructure Agent
│   │   ├── error_analysis/       # Error Analysis Agent with admin todos
│   │   └── self_healing/         # Self-healing framework (analyzer, corrector)
│   ├── channels/                 # Input channel integrations
│   │   └── slack.py              # Slack Bot with LangGraph integration
│   ├── mcp/                      # MCP client infrastructure
│   │   ├── client.py             # Multi-transport MCP client
│   │   ├── registry.py           # Server registry with health checks
│   │   ├── catalog.py            # Tool catalog with aliases & domain prefixes
│   │   ├── config.py             # YAML config loader
│   │   └── tools/                # Tool utilities
│   │       └── converter.py      # MCP → LangChain tool converter
│   ├── llm/                      # Multi-LLM factory
│   │   ├── factory.py            # LLM provider factory
│   │   └── oca.py                # Oracle Code Assist integration
│   ├── memory/                   # Shared memory layer
│   │   ├── manager.py            # Memory abstraction (Redis + ATP)
│   │   ├── checkpointer.py       # ATP-backed LangGraph checkpointer
│   │   └── context.py            # Context compression
│   ├── cache/                    # OCI resource caching
│   │   └── oci_resource_cache.py # Redis cache with tag invalidation
│   ├── rag/                      # RAG with OCI GenAI
│   │   ├── embeddings.py         # OCI GenAI embeddings
│   │   ├── vector_store.py       # Redis vector store
│   │   └── retriever.py          # High-level retrieval
│   ├── resilience/               # Resilience infrastructure
│   │   ├── bulkhead.py           # Resource isolation
│   │   ├── deadletter.py         # Failed operation queue
│   │   └── health.py             # Health monitoring
│   ├── observability/            # Tracing and logging
│   │   ├── tracing.py            # OpenTelemetry → OCI APM
│   │   └── oci_logging.py        # OCI Logging with trace correlation
│   ├── formatting/               # Response formatters
│   │   ├── base.py               # Structured response models
│   │   └── slack.py              # Slack Block Kit + native tables
│   ├── evaluation/               # Evaluation framework
│   │   └── judge.py              # LLM-as-a-Judge
│   ├── api/                      # FastAPI endpoints
│   │   └── main.py               # REST API with chat, tools, agents
│   └── main.py                   # Application entry point
├── tests/                        # Pytest test suite (212 tests)
├── config/                       # Configuration files
│   └── mcp_servers.yaml          # MCP server definitions
├── prompts/                      # Agent system prompts
└── docs/                         # Architecture documentation
```

## Observability

### OCI APM Integration

All agent operations are traced via OpenTelemetry:

```python
from src.observability import init_observability, get_trace_id

# Initialize on startup
init_observability(agent_name="db-troubleshoot-agent")

# Get current trace ID for correlation
trace_id = get_trace_id()
```

### OCI Logging

Per-agent dedicated logs with automatic trace correlation:

```python
# Logs automatically include trace_id for APM correlation
logger.info("Processing request", trace_id=get_trace_id())
```

View correlated logs in OCI Console:
- APM Traces: `cloud.oracle.com/apm/apm-traces`
- Logging: `cloud.oracle.com/logging`

## Development Status

### Completed ✅
- [x] Project structure and Poetry setup
- [x] Multi-LLM factory (OCA, Anthropic, OpenAI, OCI GenAI)
- [x] OpenTelemetry tracing → OCI APM
- [x] OCI Logging with trace correlation
- [x] MCP client infrastructure with retry logic
- [x] Tool-specific timeouts for large compartments
- [x] Database Observatory MCP integration
- [x] Agent catalog with auto-discovery
- [x] Skill system with step execution
- [x] 6 Specialized Agents (DB Troubleshoot, Log Analytics, Security, FinOps, Infrastructure, Error Analysis)
- [x] Slack Bot integration (Socket Mode + LangGraph + 3-second ack)
- [x] Instance operations by name (`oci_compute_*_by_name`)
- [x] Structured response formatting with Slack table blocks
- [x] LangGraph coordinator with intent routing
- [x] Multi-agent parallel orchestration with loop prevention
- [x] 16 pre-built deterministic workflows
- [x] FastAPI REST API server with SSE streaming
- [x] Tool aliases and domain-based dynamic discovery
- [x] ToolConverter for MCP → LangChain
- [x] RAG with OCI GenAI embeddings
- [x] LLM-as-a-Judge evaluation framework
- [x] Self-healing framework (error analysis, parameter correction)
- [x] Resilience infrastructure (Bulkhead, Circuit Breaker, DLQ)
- [x] Redis caching with tag-based invalidation
- [x] Context compression for long conversations
- [x] 212+ tests passing (80%+ coverage target)

### Planned 📋
- [ ] Microsoft Teams integration
- [ ] Web UI dashboard
- [ ] OKE deployment manifests

## Testing

```bash
# Run all 212 tests
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/test_all_agents.py -v

# Run with coverage report
poetry run pytest --cov=src --cov-report=html

# Run API server tests
poetry run pytest tests/test_api_server.py -v

# Run MCP tests
poetry run pytest tests/test_mcp_server.py tests/test_tool_converter.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow TDD workflow
4. Ensure tests pass with >80% coverage
5. Submit a pull request

## License

This project is for educational purposes. See [LICENSE](./LICENSE) for details.

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [Slack Bolt](https://slack.dev/bolt-python/) - Slack integration
- Oracle Code Assist, Claude (Anthropic), OCI GenAI

## Security

- Never commit secrets or OCIDs to version control
- Use `.env.local` for local development
- Use OCI Vault for production secrets
- Report security issues via GitHub Security Advisories
