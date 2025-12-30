# Track Plan: Foundation & Core Infrastructure

## Phase 1: Project Environment & LLM Setup
- [x] Task: Initialize Python project with Poetry and install dependencies (langchain, langgraph, oci, fastapi, opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp) (74d77a1)
- [ ] Task: Configure Environment (Copy .env.local from /Users/abirzu/dev/oracle-db-autonomous-agent/.env.local)
- [ ] Task: Implement OCI APM Tracing (Python port of OtelTracing.ts logic)
- [ ] Task: Implement OCA LangChain Wrapper (Python) based on referenced client
- [ ] Task: Implement Multi-LLM Factory (Factory pattern for OCA, OCI GenAI, Anthropic, OpenAI)
- [ ] Task: Configure linting and formatting with Ruff and Black
- [ ] Task: Set up testing framework with Pytest and coverage reporting
- [ ] Task: Conductor - User Manual Verification 'Project Environment' (Protocol in workflow.md)

## Phase 2: Unified MCP Layer (Skills & Tools)
- [ ] Task: Initialize Unified FastMCP Server structure (tools/, resources/, skills/)
- [ ] Task: Implement Progressive Disclosure (search_capabilities tool)
- [ ] Task: Port Compute & Network domains to Unified Server (Markdown output, pagination)
- [ ] Task: Port Cost, Security & Observability domains to Unified Server
- [ ] Task: Implement 'Troubleshoot' Workflow Skill (Deterministic logic)
- [ ] Task: Conductor - User Manual Verification 'Unified MCP Layer' (Protocol in workflow.md)

## Phase 3: LangGraph Coordinator Core (Workflow-First)
- [ ] Task: Implement ToolCatalog for dynamic tool discovery
- [ ] Task: Define CoordinatorState schema (Messages, Skills, Context)
- [ ] Task: Implement Workflow Routing Node (Classifies Intent -> Workflow vs Agent)
- [ ] Task: Integrate OCI Database Observatory Agent (as sub-graph)
- [ ] Task: Create basic runtime loop with OCA LLM binding
- [ ] Task: Conductor - User Manual Verification 'LangGraph Coordinator Core' (Protocol in workflow.md)

## Phase 4: Evaluation & Success
- [ ] Task: Define "Gold Standard" evaluation dataset (50+ queries)
- [ ] Task: Implement LLM-as-a-Judge Evaluator (using OCI GenAI/Claude)
- [ ] Task: Run baseline evaluation on Coordinator Intent Classification
- [ ] Task: Conductor - User Manual Verification 'Evaluation & Success' (Protocol in workflow.md)
