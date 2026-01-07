# Track Plan: Foundation & Core Infrastructure

> **Note (2026-01):** ATP persistent memory references in this planning document are historical. ATP has been removed from the codebase; the memory layer now uses Redis cache with LangGraph MemorySaver.

**Track Started**: 2025-12-29
**Last Updated**: 2025-12-30
**Status**: Phase 4 IN PROGRESS 🔄 - Evaluation Framework Complete, Baseline Eval Pending

---

## Pre-Phase 0: Documentation & Architecture (COMPLETED)

Documentation and architecture planning completed to establish project foundation.

- [x] Task: Create CLAUDE.md for Claude Code context
- [x] Task: Create AGENT.md with agent architecture documentation
- [x] Task: Create README.md for project overview
- [x] Task: Create docs/ARCHITECTURE.md with comprehensive technical architecture
- [x] Task: Create .gitignore for GitHub safe publishing
- [x] Task: Update pyproject.toml with comprehensive dependencies and tool configs
- [x] Task: Create docs/OCI_AGENT_REFERENCE.md with agent schema and naming conventions

**Reference Materials**:
- `analysis/ORACLE_AGENTS.md` - TypeScript reference implementation
- `analysis/MCP_ARCHITECTURE.md` - MCP modernization plan
- `analysis/OBSERVATORY_graph.py` - Python LangGraph prototype
- `prompts/` - 6 agent system prompts (Coordinator + 5 specialized)

**MCP Server References** (from `/Users/abirzu/dev/MCP/`):
- `mcp-oci-new/` - Unified OCI MCP server (reference architecture)
- `opsi/` - Operations Insights MCP
- `logan/` - Logging Analytics MCP
- `finopsai-mcp/` - FinOps Analysis MCP
- `oci-mcp-security/` - Security & Compliance MCP

---

## Phase 1: Project Environment & LLM Setup [checkpoint: e188b88]

### Status: COMPLETED ✅

- [x] Task: Initialize Python project with Poetry and install dependencies (74d77a1)
- [x] Task: Configure Environment (Copy .env.local) (30d5ac2)
- [x] Task: Create src/ directory structure (5d3c47b)
- [x] Task: Implement OCI APM Tracing (Python port of OtelTracing.ts logic) (5581ec3)
- [x] Task: Implement OCA LangChain Wrapper (Python) based on referenced client (118ca1a)
- [x] Task: Implement Multi-LLM Factory (Factory pattern for OCA, OCI GenAI, Anthropic, OpenAI) (c0fe65e)
- [x] Task: Configure linting and formatting with Ruff and Black (f4b805b)
- [x] Task: Set up testing framework with Pytest and coverage reporting (94d5106)
- [x] Task: Conductor - User Manual Verification 'Project Environment' (Protocol in workflow.md) (e188b88)

**Deliverables**:
- `src/llm/providers/oca.py` - OCA LangChain wrapper with PKCE OAuth
- `src/llm/factory.py` - Multi-LLM factory pattern
- `src/observability/tracing.py` - OpenTelemetry → OCI APM integration
- Test coverage: 80%+

---

## Phase 2: Unified MCP Layer (Skills & Tools) [checkpoint: 647b9ab]

### Status: COMPLETED ✅

- [x] Task: Initialize Unified FastMCP Server structure (tools/, resources/, skills/) (ac1394a)
- [x] Task: Implement Progressive Disclosure (search_capabilities tool) (b99243c)
- [x] Task: Port Compute & Network domains to Unified Server (Markdown output, pagination) (721e4bd)
- [x] Task: Port Cost, Security & Observability domains to Unified Server (4c75bfb)
- [x] Task: Implement 'Troubleshoot' Workflow Skill (Deterministic logic) (4c75bfb)
- [x] Task: Conductor - User Manual Verification 'Unified MCP Layer' (Protocol in workflow.md) (647b9ab)

**Deliverables**:
- `src/mcp/server/main.py` - FastMCP server with progressive disclosure
- `src/mcp/server/tools/` - 5 domain tools (compute, network, cost, security, observability)
- `src/mcp/server/skills/troubleshoot.py` - Instance troubleshooting workflow
- Test coverage: 80%+

**Tool Inventory** (11+ tools):
| Domain | Tools |
|--------|-------|
| Compute | `list_instances`, `start_instance`, `stop_instance`, `restart_instance` |
| Network | `list_vcns`, `list_subnets` |
| Cost | `get_cost_summary`, `get_cost_by_service` |
| Security | `list_users`, `list_policies` |
| Observability | `get_metrics`, `query_logs` |
| Skills | `troubleshoot_instance` |

---

## Phase 3: LangGraph Coordinator Core (Workflow-First)

### Status: COMPLETE ✅

### Completed ✅
- [x] Task: Implement ToolCatalog for dynamic tool discovery
  - Target: `src/mcp/catalog.py`

- [x] Task: Implement BaseAgent class with auto-registration pattern
  - AgentDefinition, AgentStatus, AgentMetadata, KafkaTopics dataclasses
  - Abstract methods: get_definition(), invoke(), build_graph()
  - Memory and tool integration helpers
  - Target: `src/agents/base.py` (~280 lines)

- [x] Task: Implement AgentCatalog with auto-discovery
  - Auto-scan `src/agents/` directory via pkgutil/rglob
  - Capability-based agent lookup (get_by_capability, get_by_skill)
  - Health monitoring (health_check_all)
  - Singleton pattern with reset_instance() for testing
  - Target: `src/agents/catalog.py` (~320 lines)

- [x] Task: Implement SharedMemoryManager (Redis + ATP)
  - RedisMemoryStore for hot cache (session state, tool results)
  - ATPMemoryStore for persistent storage (conversation history, audit)
  - InMemoryStore for testing
  - Tiered get/set with cache warming
  - Target: `src/memory/manager.py` (~420 lines)

- [x] Task: Configure OCI ATP Connection
  - Wallet-based mTLS authentication support
  - ATPConfig from environment variables
  - Connection pool creation and schema initialization
  - Target: `src/memory/atp_config.py` (~200 lines)

- [x] Task: Define CoordinatorState schema (Messages, Skills, Context)
  - CoordinatorState with LangGraph message reducer
  - IntentClassification with category, confidence, domains
  - RoutingDecision with workflow-first thresholds
  - AgentContext, ToolCall, ToolResult dataclasses
  - determine_routing() function with 0.80/0.60/0.30 thresholds
  - Target: `src/agents/coordinator/state.py` (~300 lines)

- [x] Task: Implement Workflow Routing Node (Classifies Intent -> Workflow vs Agent)
  - CoordinatorNodes class with all graph nodes
  - classifier_node with LLM-based intent extraction
  - router_node with workflow-first routing
  - workflow_node, agent_node, action_node, output_node
  - Conditional edge functions for graph routing
  - Target: `src/agents/coordinator/nodes.py` (~450 lines)

- [x] Task: Implement LangGraph Coordinator graph
  - LangGraphCoordinator class with StateGraph
  - Graph: input → classifier → router → (workflow|agent) → (action →)* → output
  - MemorySaver checkpointing for conversation continuity
  - Tool binding via ToolConverter
  - invoke() and invoke_stream() methods
  - Factory function create_coordinator()
  - Target: `src/agents/coordinator/graph.py` (~350 lines)

- [x] Task: Implement specialized agents
  - ✅ DB Troubleshoot Agent (`src/agents/database/troubleshoot.py`)
  - ✅ Log Analytics Agent (`src/agents/log_analytics/agent.py`)
  - ✅ Security Threat Agent (`src/agents/security/agent.py`)
  - ✅ FinOps Agent (`src/agents/finops/agent.py`)
  - ✅ Infrastructure Agent (`src/agents/infrastructure/agent.py`)

### Enhancement: Slack Formatting & Catalog Improvements ✅

- [x] Task: Create src/formatting/ module with structured response types
  - `src/formatting/base.py` - StructuredResponse, ResponseHeader, Section, MetricValue, etc.
  - `src/formatting/slack.py` - SlackFormatter with Block Kit support
  - `src/formatting/markdown.py` - MarkdownFormatter for CLI/docs
  - `src/formatting/__init__.py` - Module exports

- [x] Task: Enhance AgentCatalog with domain lookup and priority selection
  - Domain-based agent lookup (`get_by_domain`)
  - Priority-based agent selection (`find_best_agent`, `find_agents_ranked`)
  - Performance metrics tracking (`record_invocation`, `get_metrics_summary`)
  - Agent scoring with weighted criteria (capability 40%, domain 30%, health 15%, perf 15%)

- [x] Task: Update BaseAgent with response formatting support
  - `create_response()` - Create StructuredResponse
  - `format_response()` - Format to Slack/Markdown/Teams
  - `format_error_response()` - Standard error formatting
  - `output_format` configuration via agent config

- [x] Task: Update CoordinatorState with output formatting
  - `output_format` field (markdown, slack, teams, etc.)
  - `channel_type` field (slack, teams, web, api, cli)
  - Metrics recording for agent invocations

- [x] Task: Update all specialized agents to use structured responses
  - DB Troubleshoot Agent - health scores, problem areas, recommendations
  - FinOps Agent - cost metrics, service breakdown tables, trends
  - Security Agent - risk assessment, Cloud Guard problems, compliance
  - Log Analytics Agent - patterns, anomalies, correlations
  - Infrastructure Agent - instance tables, VCN lists, action results

### Pending 🔄
- [ ] Task: Conductor - User Manual Verification 'LangGraph Coordinator Core' (Protocol in workflow.md)

### Files Implemented
| File | Lines | Description |
|------|-------|-------------|
| `src/agents/base.py` | ~500 | BaseAgent with auto-registration + formatting |
| `src/agents/catalog.py` | ~930 | AgentCatalog with domain/priority selection |
| `src/agents/__init__.py` | ~35 | Module exports |
| `src/memory/manager.py` | ~420 | SharedMemoryManager with Redis + ATP |
| `src/memory/atp_config.py` | ~200 | OCI ATP configuration |
| `src/memory/__init__.py` | ~20 | Module exports |
| `src/agents/coordinator/state.py` | ~340 | CoordinatorState with output format |
| `src/agents/coordinator/nodes.py` | ~490 | Graph nodes with metrics |
| `src/agents/coordinator/graph.py` | ~350 | LangGraph coordinator |
| `src/agents/coordinator/__init__.py` | ~35 | Module exports |
| `src/agents/database/troubleshoot.py` | ~690 | DB Troubleshoot Agent |
| `src/agents/log_analytics/agent.py` | ~310 | Log Analytics Agent |
| `src/agents/security/agent.py` | ~330 | Security Threat Agent |
| `src/agents/finops/agent.py` | ~350 | FinOps Agent |
| `src/agents/infrastructure/agent.py` | ~380 | Infrastructure Agent |
| `src/formatting/base.py` | ~280 | Structured response types |
| `src/formatting/slack.py` | ~330 | Slack Block Kit formatter |
| `src/formatting/markdown.py` | ~220 | Markdown formatter |
| `src/formatting/__init__.py` | ~50 | Module exports |
| **Total** | ~6,260 | Phase 3 + Enhancements Complete |

---

## Phase 4: Evaluation & Success

### Status: IN PROGRESS 🔄

### Completed ✅

- [x] Task: Define "Gold Standard" evaluation dataset (60 queries)
  - Database domain: 10 cases (troubleshooting, performance, scaling)
  - Security domain: 10 cases (Cloud Guard, compliance, threat detection)
  - FinOps domain: 10 cases (cost analysis, optimization, forecasting)
  - Logs domain: 10 cases (search, patterns, correlation, anomalies)
  - Infrastructure domain: 10 cases (compute, network, capacity)
  - Multi-domain: 5 cases (cross-domain troubleshooting)
  - Edge cases: 5 cases (ambiguous, malformed, adversarial)

- [x] Task: Implement LLM-as-a-Judge Evaluator (using OCI GenAI/Claude)
  - JudgmentCriteria: correctness (40%), quality (25%), safety (25%), efficiency (10%)
  - Deterministic checks: intent, routing, domains, tools, latency
  - LLM checks: relevance, completeness, accuracy, hallucinations
  - Supports skip_llm mode for fast testing

- [x] Task: Create evaluation runner and metrics collection
  - EvaluationRunner with sequential/concurrent modes
  - EvaluationMetrics with breakdowns by category, difficulty, domain
  - MetricsReport with markdown output and target validation
  - 12 tests passing

### Pending 🔄

- [ ] Task: Run baseline evaluation on Coordinator Intent Classification
  - Target: 70%+ workflow routing ratio
  - Target: 85%+ task success rate
  - Requires: Running coordinator with full MCP integration

- [ ] Task: Conductor - User Manual Verification 'Evaluation & Success' (Protocol in workflow.md)

### Files Implemented
| File | Lines | Description |
|------|-------|-------------|
| `src/evaluation/__init__.py` | ~25 | Module exports |
| `src/evaluation/dataset.py` | ~280 | Evaluation case schema and dataset loader |
| `src/evaluation/judge.py` | ~420 | LLM-as-a-Judge evaluator |
| `src/evaluation/metrics.py` | ~380 | Metrics collection and reporting |
| `src/evaluation/runner.py` | ~350 | Evaluation orchestrator |
| `src/evaluation/datasets/gold_standard.yaml` | ~500 | 60 evaluation cases |
| `tests/test_evaluation.py` | ~210 | 12 tests for evaluation framework |
| **Total** | ~2,165 | Phase 4 Evaluation Framework |

---

## Phase 5: Input Channels & Production (FUTURE)

- [ ] Task: Implement FastAPI endpoints
- [ ] Task: Slack integration (Bolt SDK)
- [ ] Task: Teams integration (Bot Framework)
- [ ] Task: OKE deployment configuration
- [ ] Task: OCI Vault integration for secrets

---

## Key Architecture Decisions

### 1. Workflow-First Design
- **Target**: 70%+ of requests handled by deterministic workflows
- **Fallback**: Agentic LLM reasoning for novel/complex requests
- **Implementation**: Confidence-based routing in classifier node

### 2. Agent Catalog Auto-Registration
- Agents auto-discover when placed in `src/agents/{domain}/`
- Implement `BaseAgent.get_definition()` for catalog registration
- Capability and skill-based agent lookup

### 3. Shared Memory Architecture
| Layer | Backend | Purpose | TTL |
|-------|---------|---------|-----|
| Hot Cache | Redis | Session state, tool results | 1 hour |
| Persistent | OCI ATP | Conversation history, audit | Permanent |
| Checkpoints | LangGraph | Graph state, iterations | Session |

### 4. Tool Tiers
| Tier | Latency | Risk | Examples |
|------|---------|------|----------|
| 1 | <100ms | None | `oci_ping`, `oci_search_tools` |
| 2 | 100ms-1s | Low | `list_instances`, `list_vcns` |
| 3 | 1-10s | Medium | `get_awr_report`, `query_logs` |
| 4 | 10s+ | High | `start_instance`, `stop_database` |

### 5. Naming Conventions
| Component | Convention | Example |
|-----------|------------|---------|
| Agent Role | `{domain}-{function}-agent` | `db-troubleshoot-agent` |
| Agent ID | `{role}-{uuid-suffix}` | `db-troubleshoot-agent-c5b6cd64b` |
| MCP Tool | `oci_{domain}_{action}_{resource}` | `oci_database_list_autonomous` |
| Kafka Topic | `commands.{agent-role}` | `commands.db-troubleshoot-agent` |

---

## Reference Implementation Mapping

| Component | Reference Source | Target Location |
|-----------|------------------|-----------------|
| OTEL Tracing | `analysis/observability_service/OtelTracing.ts` | `src/observability/tracing.py` ✅ |
| OCA Client | `analysis/oca-langchain-client/src/` | `src/llm/providers/oca.py` ✅ |
| MCP Server | `/Users/abirzu/dev/MCP/mcp-oci-new/` | `src/mcp/server/` ✅ |
| Agent Schema | `docs/OCI_AGENT_REFERENCE.md` | `src/agents/base.py` ✅ |
| Agent Catalog | `docs/OCI_AGENT_REFERENCE.md` | `src/agents/catalog.py` ✅ |
| Shared Memory | `docs/OCI_AGENT_REFERENCE.md` | `src/memory/manager.py` ✅ |
| LangGraph Coord | `analysis/OBSERVATORY_graph.py` | `src/agents/coordinator/graph.py` ✅ |
| Coordinator State | `analysis/OBSERVATORY_graph.py` | `src/agents/coordinator/state.py` ✅ |
| Coordinator Prompt | `prompts/00-COORDINATOR-AGENT.md` | System prompt config ⏳ |

---

## External Dependencies

### MCP Servers
| Server | Path | Transport | Status |
|--------|------|-----------|--------|
| oci-unified | `/Users/abirzu/dev/MCP/mcp-oci-new/` | stdio | Reference |
| opsi | `/Users/abirzu/dev/MCP/opsi/` | HTTP:8000 | Available |
| logan | `/Users/abirzu/dev/MCP/logan/` | HTTP:8001 | Available |
| finopsai | `/Users/abirzu/dev/MCP/finopsai-mcp/` | HTTP | Available |
| security | `/Users/abirzu/dev/MCP/oci-mcp-security/` | HTTP | Available |

### Infrastructure
- **Redis**: Required for hot cache (session state)
- **OCI ATP**: Required for persistent storage (conversation history)
- **OCI APM**: Required for observability
- **OCI Vault**: Required for production secrets

---

## Code Statistics

| Component | Files | Lines | Status |
|-----------|-------|-------|--------|
| `src/mcp/` | 12 | ~400 | ✅ Complete |
| `src/llm/` | 4 | ~250 | ✅ Complete |
| `src/observability/` | 2 | ~130 | ✅ Complete |
| `src/agents/` | 6 | ~1,500 | ✅ Core Complete |
| `src/memory/` | 3 | ~650 | ✅ Complete |
| `tests/` | 6 | ~300 | ✅ 80%+ coverage |
| **Total** | ~33 | ~3,230 | Phase 3 Core Complete |

### Phase 3 New Files
- `src/agents/base.py` - BaseAgent with auto-registration (~280 lines)
- `src/agents/catalog.py` - AgentCatalog with auto-discovery (~320 lines)
- `src/agents/coordinator/state.py` - CoordinatorState schema (~300 lines)
- `src/agents/coordinator/nodes.py` - Graph nodes (~450 lines)
- `src/agents/coordinator/graph.py` - LangGraph coordinator (~350 lines)
- `src/memory/manager.py` - SharedMemoryManager (~420 lines)
- `src/memory/atp_config.py` - ATP configuration (~200 lines)
