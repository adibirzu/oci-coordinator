# Track Plan: Foundation & Core Infrastructure

## Phase 1: Project Environment
- [ ] Task: Initialize Python project with Poetry and install core dependencies (langchain, langgraph, oci, fastapi, redis, pydantic)
- [ ] Task: Configure linting and formatting with Ruff and Black
- [ ] Task: Set up testing framework with Pytest and coverage reporting
- [ ] Task: Create Docker development environment for local development
- [ ] Task: Conductor - User Manual Verification 'Project Environment' (Protocol in workflow.md)

## Phase 2: MCP Layer
- [ ] Task: Deploy and configure oci-mcp-cost and oci-mcp-db servers
- [ ] Task: Deploy and configure oci-mcp-compute and oci-mcp-network servers
- [ ] Task: Deploy and configure oci-mcp-security and oci-mcp-observability servers
- [ ] Task: Implement MCP server health monitoring utility
- [ ] Task: Conductor - User Manual Verification 'MCP Layer' (Protocol in workflow.md)

## Phase 3: LangGraph Coordinator Core
- [ ] Task: Define LangGraph State schema (messages, current_agent, slots)
- [ ] Task: Implement OCI GenAI Inference Node (Intent Classification)
- [ ] Task: Implement Router Node logic (Conditional Edges)
- [ ] Task: Implement Mock Agent Nodes for testing routing
- [ ] Task: Create basic LangGraph runtime loop and test entry point
- [ ] Task: Conductor - User Manual Verification 'LangGraph Coordinator Core' (Protocol in workflow.md)