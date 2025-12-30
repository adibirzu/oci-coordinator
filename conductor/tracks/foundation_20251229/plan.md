# Track Plan: Foundation & Core Infrastructure

**Track Started**: 2025-12-29
**Last Updated**: 2025-12-30
**Status**: In Progress - Phase 1

---

## Pre-Phase 0: Documentation & Architecture (COMPLETED)

Documentation and architecture planning completed to establish project foundation.

- [x] Task: Create CLAUDE.md for Claude Code context
- [x] Task: Create AGENT.md with agent architecture documentation
- [x] Task: Create README.md for project overview
- [x] Task: Create docs/ARCHITECTURE.md with comprehensive technical architecture
- [x] Task: Create .gitignore for GitHub safe publishing
- [x] Task: Update pyproject.toml with comprehensive dependencies and tool configs

**Reference Materials Created**:
- `analysis/ORACLE_AGENTS.md` - TypeScript reference implementation (from existing project)
- `analysis/MCP_ARCHITECTURE.md` - MCP modernization plan
- `analysis/OBSERVATORY_graph.py` - Python LangGraph prototype
- `analysis/observability_service/` - TypeScript OTEL module (to port)
- `analysis/oca-langchain-client/` - TypeScript OCA client (to port)
- `prompts/` - 6 agent system prompts (Coordinator + 5 specialized)

---

## Phase 1: Project Environment & LLM Setup

### Completed
- [x] Task: Initialize Python project with Poetry and install dependencies (74d77a1)
- [x] Task: Configure Environment (Copy .env.local) (30d5ac2)

### In Progress
- [x] Task: Implement OCI APM Tracing (Python port of OtelTracing.ts logic) (5581ec3)
  - Port from: `analysis/observability_service/OtelTracing.ts`
  - Target: `src/observability/tracing.py`

### Pending
- [ ] Task: Create src/ directory structure
  - `src/core/` - Logger, config, exceptions
  - `src/agents/coordinator/` - LangGraph coordinator
  - `src/mcp/` - MCP client, registry, catalog
  - `src/llm/` - Multi-LLM factory
  - `src/observability/` - OTEL tracing
  - `src/api/` - FastAPI endpoints

- [x] Task: Implement OCA LangChain Wrapper (Python) based on referenced client (118ca1a)
  - Port from: `analysis/oca-langchain-client/`
  - Target: `src/llm/oca.py`

- [x] Task: Implement Multi-LLM Factory (Factory pattern for OCA, OCI GenAI, Anthropic, OpenAI) (c0fe65e)
  - Factory pattern for: OCA, OCI GenAI, Anthropic, OpenAI, Ollama
  - Target: `src/llm/factory.py`

- [x] Task: Configure linting and formatting with Ruff and Black (f4b805b)
  - Config: pyproject.toml (already configured)
  - Action: Run `ruff check src/` and `black src/`

- [x] Task: Set up testing framework with Pytest and coverage reporting (94d5106)
  - Config: pyproject.toml (already configured)
  - Action: Create `tests/conftest.py`, run `pytest --cov=src`

- [ ] Task: Conductor - User Manual Verification 'Project Environment' (Protocol in workflow.md)

---

## Phase 2: Unified MCP Layer (Skills & Tools)

- [ ] Task: Initialize Unified FastMCP Server structure (tools/, resources/, skills/)
- [ ] Task: Implement Progressive Disclosure (search_capabilities tool)
- [ ] Task: Port Compute & Network domains to Unified Server (Markdown output, pagination)
- [ ] Task: Port Cost, Security & Observability domains to Unified Server
- [ ] Task: Implement 'Troubleshoot' Workflow Skill (Deterministic logic)
- [ ] Task: Conductor - User Manual Verification 'Unified MCP Layer' (Protocol in workflow.md)

---

## Phase 3: LangGraph Coordinator Core (Workflow-First)

- [ ] Task: Implement ToolCatalog for dynamic tool discovery
- [ ] Task: Define CoordinatorState schema (Messages, Skills, Context)
  - Reference: `analysis/OBSERVATORY_graph.py` (CoordinatorState class)
- [ ] Task: Implement Workflow Routing Node (Classifies Intent -> Workflow vs Agent)
- [ ] Task: Integrate OCI Database Observatory Agent (as sub-graph)
- [ ] Task: Create basic runtime loop with OCA LLM binding
- [ ] Task: Conductor - User Manual Verification 'LangGraph Coordinator Core' (Protocol in workflow.md)

---

## Phase 4: Evaluation & Success

- [ ] Task: Define "Gold Standard" evaluation dataset (50+ queries)
- [ ] Task: Implement LLM-as-a-Judge Evaluator (using OCI GenAI/Claude)
- [ ] Task: Run baseline evaluation on Coordinator Intent Classification
- [ ] Task: Conductor - User Manual Verification 'Evaluation & Success' (Protocol in workflow.md)

---

## Implementation Notes

### Key Architecture Decisions
1. **Workflow-First**: Target 70%+ of requests handled by deterministic workflows
2. **Tool Tiers**: Classify tools by latency (Tier 1: <100ms, Tier 4: 10s+)
3. **Progressive Disclosure**: Agents discover tools dynamically via search_capabilities
4. **State Persistence**: MemorySaver (dev) → Redis (prod)

### Reference Files for Implementation
| Component | Reference | Target |
|-----------|-----------|--------|
| OTEL Tracing | `analysis/observability_service/OtelTracing.ts` | `src/observability/tracing.py` |
| OCA Client | `analysis/oca-langchain-client/src/` | `src/llm/oca.py` |
| LangGraph Coord | `analysis/OBSERVATORY_graph.py` | `src/agents/coordinator/graph.py` |
| Coordinator Prompt | `prompts/00-COORDINATOR-AGENT.md` | System prompt config |

### External Dependencies
- MCP Servers: OPSI (8000), Logan (8001), Unified (8002), SQLcl (stdio)
- Redis: Required for production state persistence
- OCI APM: Required for observability
