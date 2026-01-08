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

Python LangGraph orchestration for OCI operations. 6 agents, 4 MCP servers (395+ tools), 35+ workflows.

**Status**: Phase 4 - Production Readiness (Teams, OKE deployment)

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

**Agents**: DB Troubleshoot, Log Analytics, Security, FinOps, Infrastructure, Error Analysis

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

## MCP Servers

| Server | Tools | Domains | Primary Use |
|--------|-------|---------|-------------|
| `oci-unified` | 51 | identity, compute, network, database, dbmgmt, opsi | Core OCI + DB Mgmt |
| `database-observatory` | 50+ | database, opsi, logan, observability | SQLcl/Logan queries |
| `oci-infrastructure` | 44 | compute, network, security, database | Full OCI SDK |
| `finopsai` | 33 | cost, budget, finops, anomaly | FinOps analysis |

## Reference

For detailed documentation, read Serena memories or see:
- `docs/OCI_AGENT_REFERENCE.md` - Agent schemas
- `docs/ARCHITECTURE.md` - Full architecture
- `docs/FEATURE_MAPPING.md` - Tool mapping
- `docs/DB_TROUBLESHOOTING_WORKFLOW.md` - DB workflow mapping
