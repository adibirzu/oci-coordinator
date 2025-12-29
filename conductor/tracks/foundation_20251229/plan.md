# Track Plan: Foundation & Core Infrastructure

## Phase 1: Project Environment
- [ ] Task: Initialize Python project with Poetry and install core dependencies (langchain, langgraph, oci, fastapi, redis, pydantic)
- [ ] Task: Configure linting and formatting with Ruff and Black
- [ ] Task: Set up testing framework with Pytest and coverage reporting
- [ ] Task: Create Docker development environment for local development
- [ ] Task: Conductor - User Manual Verification 'Project Environment' (Protocol in workflow.md)

## Phase 2: Unified MCP Layer
- [ ] Task: Initialize Unified FastMCP Server structure (tools/, resources/, skills/)
- [ ] Task: Implement Progressive Disclosure (search_capabilities tool)
- [ ] Task: Port Compute & Network domains to Unified Server with pagination/markdown
- [ ] Task: Port Cost, Security & Observability domains to Unified Server
- [ ] Task: Implement 'Troubleshoot' high-level skill
- [ ] Task: Conductor - User Manual Verification 'Unified MCP Layer' (Protocol in workflow.md)

## Phase 3: LangGraph Coordinator Core
- [ ] Task: Implement ToolCatalog for dynamic tool discovery
- [ ] Task: Define CoordinatorState schema matching reference architecture
- [ ] Task: Implement LangGraph nodes (Input, Agent, Action, Output)
- [ ] Task: Integrate OCI Database Observatory Agent (as sub-graph or tool)
- [ ] Task: Create basic runtime loop with ToolCatalog binding
- [ ] Task: Conductor - User Manual Verification 'LangGraph Coordinator Core' (Protocol in workflow.md)