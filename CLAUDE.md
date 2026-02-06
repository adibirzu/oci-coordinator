# CLAUDE.md - OCI AI Agent Coordinator

> **Token Optimization**: This file is kept minimal. For detailed info, use Serena memories:
> `project_overview`, `code_style`, `suggested_commands`, `task_completion`, `architecture_details`, `mcp_tools`

## Quick Start

```bash
poetry install                    # Install dependencies
poetry run pytest --cov=src       # Run tests (212 passing)
poetry run python -m src.main     # Start (Slack + API on port 3001)
```

## Project Summary

Python LangGraph orchestration for OCI operations. 7 agents, 5 MCP servers (395+ tools), 40+ workflows.

**Status**: Phase 4 - Production Readiness (Teams, OKE deployment)

## Agents Overview

| Agent | Module | Domain | Key Capabilities |
|-------|--------|--------|------------------|
| `DbTroubleshootAgent` | `database/troubleshoot.py` | Database | AWR, ASH, blocking, wait events, SQL tuning |
| `LogAnalyticsAgent` | `log_analytics/agent.py` | Observability | Log queries, pattern detection, anomaly correlation |
| `SecurityThreatAgent` | `security/agent.py` | Security | Cloud Guard, MITRE mapping, compliance |
| `FinOpsAgent` | `finops/agent.py` | Cost | Cost analysis, anomaly detection, optimization |
| `InfrastructureAgent` | `infrastructure/agent.py` | Compute/Network | Instance management, VCN, subnets |
| `ErrorAnalysisAgent` | `error_analysis/agent.py` | Debugging | Error classification, root cause analysis |
| `SelectAIAgent` | `selectai/agent.py` | Data/AI | NL2SQL, data chat, AI orchestration |

## DB Troubleshooting Workflow

The system supports full database performance troubleshooting:

| Step | Intent | Workflow | Primary Tool |
|------|--------|----------|--------------|
| Blocking Sessions | `check_blocking` | `db_blocking_sessions_workflow` | `oci_database_execute_sql` |
| CPU/Wait Events | `wait_events` | `db_wait_events_workflow` | `oci_dbmgmt_get_wait_events` |
| SQL Monitoring | `sql_monitoring` | `db_sql_monitoring_workflow` | `oci_database_execute_sql` |
| Long Running Ops | `long_running_ops` | `db_long_running_ops_workflow` | `oci_database_execute_sql` |
| Parallelism | `parallelism_stats` | `db_parallelism_stats_workflow` | `oci_database_execute_sql` |
| Full Table Scans | `full_table_scan` | `db_full_table_scan_workflow` | `oci_database_execute_sql` |
| AWR Report | `awr_report` | `db_awr_report_workflow` | `oci_dbmgmt_get_awr_report` |

See `docs/DB_TROUBLESHOOTING_WORKFLOW.md` for full mapping.

## Architecture (Simplified)

```
Slack/API → Coordinator (LangGraph) → Agents → MCP Servers → OCI
```

**Agents** (7): DB Troubleshoot, Log Analytics, Security, FinOps, Infrastructure, Error Analysis, SelectAI

## Key Conventions

| Aspect | Convention |
|--------|------------|
| Agent Class | `{Domain}{Function}Agent` (e.g., `DbTroubleshootAgent`) |
| Agent Module | `src/agents/{domain}/{function}.py` |
| MCP Tool | `oci_{domain}_{action}` (e.g., `oci_compute_list_instances`) |
| Imports | Absolute from `src.` |
| Logging | `structlog` via `get_logger()` |
| Types | Full type hints, Pydantic |
| Errors | Specific exceptions, never bare `except` |
| Coverage | 80%+ target |

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/agents/` | Agent implementations |
| `src/agents/coordinator/` | LangGraph coordinator |
| `src/mcp/` | MCP client & server |
| `src/channels/` | Slack handlers |
| `src/memory/` | Redis cache |
| `tests/` | Pytest suite |

## Common Tasks

**Add Agent**: Create in `src/agents/{domain}/{function}.py`, implement `BaseAgent` with `get_definition()` classmethod.

**Add MCP Tool**: Add in `src/mcp/server/tools/{domain}.py`, register in `server/main.py`.

**Run Tests**: `poetry run pytest --cov=src`

## Security Notes

- Never commit OCIDs, API keys, or tokens
- Use `.env.local` for secrets (not in git)
- All sensitive config via OCI Vault in production

## Essential Environment Variables

```bash
LLM_PROVIDER=oracle_code_assist  # oracle_code_assist, anthropic, openai
SLACK_BOT_TOKEN=xoxb-xxx
SLACK_APP_TOKEN=xapp-xxx
USE_LANGGRAPH_COORDINATOR=true
REDIS_URL=redis://localhost:6379
```

## SQLcl Database Connectivity

Direct SQL execution via SQLcl requires wallet configuration:

```bash
SQLCL_PATH=/Applications/sqlcl/bin/sql
SQLCL_TNS_ADMIN=~/oci_wallets_unified  # Unified wallet with all connections
SQLCL_DB_USERNAME=ADMIN
SQLCL_DB_PASSWORD="<password>"
SQLCL_DB_CONNECTION=th_high            # Default connection
SQLCL_FALLBACK_CONNECTION=ATPAdi       # Fallback if default fails
```

**Available Connections**: `th_high`, `th_medium`, `th_low`, `ATPAdi_high`, `ATPAdi`, `SelectAI_high`, `SelectAI`

## MCP Servers

| Server | Tools | Domains | Primary Use |
|--------|-------|---------|-------------|
| `oci-unified` | 77 | identity, compute, network, database, dbmgmt, opsi, logan | Core OCI + DB Mgmt + Log Analytics |
| `database-observatory` | 50+ | database, opsi, logan, observability | SQLcl/Logan queries |
| `oci-infrastructure` | 44 | compute, network, security, database | Full OCI SDK |
| `finopsai` | 33 | cost, budget, finops, anomaly | FinOps analysis |
| `oci-mcp-security` | 60+ | security, cloud-guard, waf, kms, bastion, datasafe | Comprehensive security |

**Environment Variables for External MCP Servers**:
```bash
DB_OBSERVATORY_PATH=/path/to/mcp-oci-database-observatory
OCI_INFRASTRUCTURE_PATH=/path/to/mcp-oci
FINOPSAI_PATH=/path/to/finopsai-mcp
OCI_SECURITY_PATH=/path/to/oci-mcp-security
```

## Observability

OpenTelemetry integration with OCI APM for full tracing and log correlation:

```python
from src.observability import init_observability, get_tracer, OracleCodeAssistInstrumentor

# Initialize on startup (enables tracing + OTLP log export + OCI Logging)
init_observability(agent_name="coordinator")

# Instrument LLM calls with GenAI semantic conventions
with OracleCodeAssistInstrumentor.chat_span(model="oca/gpt5") as llm_ctx:
    llm_ctx.set_prompt(content, role="user")
    llm_ctx.set_tokens(input=100, output=50)
```

**Key Tracers**: `oci-coordinator`, `mcp-oci-unified`, `mcp-oci-logan`, `oca-llm`

**Log Pipelines** (3 parallel destinations):
| Pipeline | Destination | Purpose |
|----------|-------------|---------|
| OTLP Log Export | OCI APM `/v1/logs` | Span-level log correlation (APM "Logs" tab) |
| OCI Logging | OCI Logging Service | Persistence, Log Analytics queries |
| Console | stdout | Local development |

Logs emitted within an active span automatically include `trace_id`/`span_id` for correlation.

## Reference

For detailed documentation, read Serena memories or see:
- `docs/OCI_AGENT_REFERENCE.md` - Agent schemas
- `docs/ARCHITECTURE.md` - Full architecture
- `docs/FEATURE_MAPPING.md` - Tool mapping
- `docs/DB_TROUBLESHOOTING_WORKFLOW.md` - DB workflow mapping
- `docs/agents.md` - Agent implementation guide
