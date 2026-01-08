# OCI AI Agent Coordinator

## Purpose

Python LangGraph orchestration system managing specialized AI agents for Oracle Cloud Infrastructure operations. Central hub for Slack, Teams, Web, and API requests, routing to agents via MCP servers.

**Status**: Phase 4 - Production Readiness (Teams integration, OKE deployment)

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11+ |
| Orchestration | LangGraph, LangChain |
| MCP Server | FastMCP |
| API | FastAPI, Uvicorn |
| LLM Providers | OCA, Anthropic, OpenAI, OCI GenAI |
| Cache | Redis (session state, tool results) |
| Observability | OpenTelemetry → OCI APM |
| Testing | Pytest (212 tests, 80%+ coverage) |
| Package Manager | Poetry |
| Linting | Ruff, Black, MyPy |

## Architecture Summary

- 6 specialized agents: DB Troubleshoot, Log Analytics, Security, FinOps, Infrastructure, Error Analysis
- 4 MCP servers with 158+ tools
- 35+ deterministic workflows, 100+ intent aliases
- LangGraph coordinator with intent routing and parallel orchestration

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/agents/` | Agent implementations |
| `src/agents/coordinator/` | LangGraph coordinator |
| `src/mcp/` | MCP client infrastructure |
| `src/mcp/server/` | Unified FastMCP server |
| `src/channels/` | Slack handlers |
| `src/memory/` | Redis cache layer |
| `tests/` | Pytest test suite |

## Environment Variables

```bash
# LLM Provider
LLM_PROVIDER=oracle_code_assist  # oracle_code_assist, anthropic, openai
OCA_MODEL=oca/gpt5
OCA_CALLBACK_PORT=48801

# Slack
SLACK_BOT_TOKEN=xoxb-xxx
SLACK_APP_TOKEN=xapp-xxx

# OCI
OCI_CONFIG_FILE=~/.oci/config
OCI_CLI_PROFILE=DEFAULT

# Cache & Coordinator
REDIS_URL=redis://localhost:6379
USE_LANGGRAPH_COORDINATOR=true

# Observability
OCI_APM_ENDPOINT=https://xxx.apm-agt.region.oci.oraclecloud.com
OCI_APM_PRIVATE_DATA_KEY=xxx

# Infrastructure Provisioning (requires ALLOW_MUTATIONS=true)
# DEFAULT_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
# DEFAULT_AVAILABILITY_DOMAIN=Uocm:EU-FRANKFURT-1-AD-1
# DEFAULT_SUBNET_ID=ocid1.subnet.oc1.xxx
ALLOW_MUTATIONS=false  # Safety flag for write operations
```

## Slack Integration

- **Socket Mode**: Real-time events via `SLACK_APP_TOKEN`
- **Commands**: `help`, `catalog`, `runbook`
- **Catalog**: 6 categories, 20+ quick actions
- **Conversation Memory**: Redis-backed, 24h TTL

## Agent Capabilities

| Agent | Key Features |
|-------|-------------|
| DB Troubleshoot | AWR, Top SQL, Wait Events, OPSI |
| Log Analytics | Pattern detection, anomaly detection |
| Security | MITRE ATT&CK mapping (T1078, T1098, T1562) |
| FinOps | Cost analysis (30s timeout), rightsizing |
| Infrastructure | Compute, network, VCN, instance provisioning |
| Error Analysis | ORA- patterns, OOM detection |
