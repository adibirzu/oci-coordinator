# Track Spec: Foundation & Core Infrastructure

## Overview
Establish the foundational infrastructure for the OCI AI Agent Coordinator, focusing on the LangGraph orchestration layer, OCI MCP server integration, and the Python development environment.

## Goals
- Initialize a robust Python 3.11 environment using Poetry with LangChain and LangGraph.
- Ensure high code quality with Ruff, Black, and Pytest.
- Deploy and verify health of 6 OCI MCP servers.
- Implement a functional LangGraph-based Coordinator that can classify intents and route to specialized agent nodes.

## Success Criteria
- Development environment is fully containerized (Docker).
- All MCP servers are reachable and reporting healthy.
- LangGraph application successfully executes a full routing loop (Entry -> Router -> Agent -> Exit) with mock agents.