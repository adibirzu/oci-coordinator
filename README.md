# OCI AI Agent Coordinator

An intelligent orchestration system that manages specialized AI agents for Oracle Cloud Infrastructure operations. Built with LangGraph, MCP (Model Context Protocol), and full OCI observability integration.

## Disclaimer

This software was created to showcase Oracle Cloud Infrastructure (OCI) AI Integration capabilities and demonstrate how to expand them using third-party services and AI tools. This is **NOT an official Oracle product** - it is a personal project demonstrating integration possibilities with OCI AI agents and assistants.

## Features

- **Multi-Agent Orchestration**: Coordinate 5 specialized agents for database, logging, security, cost, and infrastructure operations
- **Workflow-First Design**: 70%+ deterministic workflows via MCP tools; LLM reasoning for complex analysis
- **Database Observatory Integration**: Full OPSI, SQLcl, and Logan Analytics via MCP
- **Slack Integration**: Real-time chatbot with Socket Mode support
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

# Run tests (88 tests)
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
| **FinOps** | Cost analysis, optimization | `get_cost_summary`, `find_savings` | cost, budget, spending |
| **Infrastructure** | Compute, network management | `list_instances`, `manage_vcn` | instance, VM, VCN |

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
```

## Project Structure

```
oci-coordinator/
├── src/
│   ├── agents/              # LangGraph agent implementations
│   │   ├── catalog.py       # Agent registry with auto-discovery
│   │   ├── base.py          # BaseAgent class
│   │   ├── skills.py        # Skill definitions and executor
│   │   ├── database/        # DB Troubleshoot Agent
│   │   ├── log_analytics/   # Log Analytics Agent
│   │   ├── security/        # Security Threat Agent
│   │   ├── finops/          # FinOps Agent
│   │   └── infrastructure/  # Infrastructure Agent
│   ├── channels/            # Input channel integrations
│   │   └── slack.py         # Slack Bot handler
│   ├── mcp/                 # MCP client infrastructure
│   │   ├── client.py        # MCP client wrapper
│   │   ├── registry.py      # Server registry
│   │   ├── catalog.py       # Tool catalog
│   │   └── config.py        # YAML config loader
│   ├── llm/                 # Multi-LLM factory
│   │   ├── factory.py       # LLM provider factory
│   │   └── oca.py           # Oracle Code Assist integration
│   ├── observability/       # Tracing and logging
│   │   ├── tracing.py       # OpenTelemetry → OCI APM
│   │   └── oci_logging.py   # OCI Logging with trace correlation
│   ├── formatting/          # Response formatters
│   │   ├── base.py          # Structured response models
│   │   └── slack.py         # Slack Block Kit formatter
│   ├── api/                 # FastAPI endpoints
│   └── main.py              # Application entry point
├── tests/                   # Pytest test suite (88 tests)
├── config/                  # Configuration files
│   └── mcp_servers.yaml     # MCP server definitions
├── prompts/                 # Agent system prompts
└── conductor/               # Project management
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

### Completed
- [x] Project structure and Poetry setup
- [x] Multi-LLM factory (OCA, Anthropic, OpenAI)
- [x] OpenTelemetry tracing → OCI APM
- [x] OCI Logging with trace correlation
- [x] MCP client infrastructure
- [x] Database Observatory MCP integration
- [x] Agent catalog with auto-discovery
- [x] Skill system with step execution
- [x] DB Troubleshoot Agent with full workflow
- [x] Slack Bot integration (Socket Mode)
- [x] Structured response formatting
- [x] 88 tests passing (80%+ coverage target)

### In Progress
- [ ] LangGraph coordinator with intent routing
- [ ] Log Analytics agent workflows
- [ ] Security Threat agent workflows

### Planned
- [ ] Microsoft Teams integration
- [ ] Web UI dashboard
- [ ] LLM-as-a-Judge evaluation framework
- [ ] Redis state persistence
- [ ] Multi-agent collaboration

## Testing

```bash
# Run all tests
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/test_all_agents.py -v

# Run with coverage report
poetry run pytest --cov=src --cov-report=html
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
