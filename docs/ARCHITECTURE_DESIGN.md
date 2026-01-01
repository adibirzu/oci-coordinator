# Architecture Design: Unified Agent-Tool Integration

**Date**: 2025-12-31
**Phase**: 4 - Architecture Design
**Status**: Ready for Implementation

---

## 1. Design Goals

1. **Unified Tool Naming**: All tools follow `oci_{domain}_{action}` convention
2. **Single Request Path**: Slack → LangGraph Coordinator → Agent → Tools
3. **Agent-to-Agent Delegation**: Agents can invoke other agents via coordinator
4. **Dynamic Tool Discovery**: Agents discover tools from catalog, not hardcoded lists

---

## 2. Tool Naming Convention

### 2.1 Standard Format

```
oci_{domain}_{action}
```

**Examples:**
| Domain | Action | Tool Name |
|--------|--------|-----------|
| compute | list_instances | `oci_compute_list_instances` |
| database | execute_sql | `oci_database_execute_sql` |
| opsi | get_fleet_summary | `oci_opsi_get_fleet_summary` |
| cost | get_summary | `oci_cost_get_summary` |
| security | list_problems | `oci_security_list_problems` |
| network | list_vcns | `oci_network_list_vcns` |

### 2.2 Migration Map (database-observatory)

| Current Name | New Name |
|--------------|----------|
| `execute_sql` | `oci_database_execute_sql` |
| `get_schema_info` | `oci_database_get_schema` |
| `list_connections` | `oci_database_list_connections` |
| `database_status` | `oci_database_get_status` |
| `get_fleet_summary` | `oci_opsi_get_fleet_summary` |
| `search_databases` | `oci_opsi_search_databases` |
| `list_database_insights` | `oci_opsi_list_insights` |
| `analyze_cpu_usage` | `oci_opsi_analyze_cpu` |
| `analyze_memory_usage` | `oci_opsi_analyze_memory` |
| `analyze_io_usage` | `oci_opsi_analyze_io` |
| `get_blocking_sessions` | `oci_opsi_get_blocking_sessions` |
| `analyze_wait_events` | `oci_opsi_analyze_wait_events` |

### 2.3 Backward Compatibility

Add aliases in `ToolCatalog` for backward compatibility:

```python
TOOL_ALIASES = {
    # database-observatory legacy names
    "execute_sql": "oci_database_execute_sql",
    "get_fleet_summary": "oci_opsi_get_fleet_summary",
    "analyze_cpu_usage": "oci_opsi_analyze_cpu",
    ...
}
```

---

## 3. Request Flow Architecture

### 3.1 Current (Broken) Flow

```
Slack → SlackHandler._route_to_agent() → SpecializedReActAgent → Tools
        ↑ keyword-based routing         ↑ bypasses LangGraph
```

**Issues:**
- Slack uses keyword routing, not intent classification
- Bypasses LangGraph coordinator completely
- No workflow-first routing (70% deterministic)
- No agent-to-agent delegation

### 3.2 Proposed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         UNIFIED FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Slack/API → SlackHandler._invoke_coordinator()                 │
│                     │                                           │
│                     ▼                                           │
│        LangGraphCoordinator.invoke()                            │
│                     │                                           │
│        ┌───────────┼───────────┐                                │
│        ▼           ▼           ▼                                │
│    classifier   router      output                              │
│        │           │           │                                │
│        │    ┌──────┴──────┐    │                                │
│        │    ▼             ▼    │                                │
│        │  workflow     agent   │                                │
│        │    │            │     │                                │
│        │    │     SpecializedReActAgent                         │
│        │    │            │     │                                │
│        │    │     ToolCatalog.execute()                         │
│        │    │            │     │                                │
│        │    │     MCP Server   │                                │
│        │    │            │     │                                │
│        └────┴────────────┴─────┘                                │
│                     │                                           │
│                     ▼                                           │
│              ResponseFormatter                                  │
│                     │                                           │
│                     ▼                                           │
│              Slack/API Response                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Integration Points

1. **SlackHandler** (src/channels/slack.py)
   - Change `_invoke_coordinator()` to use `LangGraphCoordinator.invoke()`
   - Remove keyword-based `_route_to_agent()`

2. **LangGraphCoordinator** (src/agents/coordinator/graph.py)
   - Add `invoke_from_slack()` method for channel integration
   - Return structured response suitable for Slack formatting

3. **CoordinatorNodes** (src/agents/coordinator/nodes.py)
   - `agent_node` should use `SpecializedReActAgent`
   - Pass `ToolCatalog` instance to agents

---

## 4. Agent-to-Agent Delegation

### 4.1 Delegation Protocol

Agents can request help from other agents via the coordinator:

```python
class AgentDelegation:
    """Request to delegate to another agent."""
    target_agent: str      # e.g., "infrastructure-agent"
    query: str             # Sub-query to ask
    context: dict          # Shared context
    return_to: str         # Calling agent ID
```

### 4.2 Implementation

In `CoordinatorNodes.agent_node`:

```python
async def agent_node(self, state: CoordinatorState) -> CoordinatorState:
    # ... existing agent invocation ...

    # Check if agent requested delegation
    if result.delegation_request:
        state["pending_delegation"] = result.delegation_request
        state["routing"]["next"] = "delegate"
        return state
```

Add new `delegate_node`:

```python
async def delegate_node(self, state: CoordinatorState) -> CoordinatorState:
    delegation = state["pending_delegation"]

    # Get target agent
    target = self.agent_catalog.get_agent(delegation.target_agent)

    # Invoke with context
    result = await target.invoke(
        delegation.query,
        context=delegation.context,
    )

    # Return result to calling agent
    state["delegation_result"] = result
    state["routing"]["next"] = "agent"  # Return to calling agent
    return state
```

### 4.3 Use Cases

| Scenario | Flow |
|----------|------|
| "Check database cost" | FinOps → DB Agent for DB list → FinOps for cost |
| "Security of compute" | Security → Infra for instances → Security for analysis |
| "Why is DB slow?" | DB → Log Analytics for errors → DB for diagnosis |

---

## 5. Dynamic Tool Discovery

### 5.1 Remove Hardcoded Tool Lists

**Current** (react_agent.py:469-529):
```python
DOMAIN_PROMPTS = {
    "database": """...
Preferred tools:
- list_autonomous_databases  # HARDCODED - WRONG NAME
- database_status
...
```

**Proposed:**
```python
async def _get_tools_context(self, domain: str) -> str:
    """Dynamically get tools for domain from catalog."""
    await self.tool_catalog.ensure_fresh()

    tools = self.tool_catalog.search_tools(
        domain=domain,
        max_tier=3,
        limit=20,
    )

    return "\n".join(
        f"- **{t['name']}**: {t['description']}"
        for t in tools
    )
```

### 5.2 Domain-Tool Mapping

The catalog should support domain-based tool discovery:

```python
# In ToolCatalog
DOMAIN_PREFIXES = {
    "database": ["oci_database_", "oci_opsi_"],
    "infrastructure": ["oci_compute_", "oci_network_", "oci_list_"],
    "finops": ["oci_cost_"],
    "security": ["oci_security_"],
    "observability": ["oci_observability_", "oci_logan_"],
}

def get_tools_for_domain(self, domain: str) -> list[ToolDefinition]:
    """Get all tools for a domain."""
    prefixes = self.DOMAIN_PREFIXES.get(domain, [])
    return [
        tool for name, tool in self._tools.items()
        if any(name.startswith(p) for p in prefixes)
    ]
```

---

## 6. Configuration Updates

### 6.1 Agent-Domain-Server Mapping

Update `config/mcp_servers.yaml`:

```yaml
servers:
  oci-unified:
    transport: stdio
    command: python
    args: ["-m", "src.mcp.server"]
    enabled: true
    domains:
      - identity
      - compute
      - network
      - cost
      - security
      - observability

  database-observatory:
    transport: stdio
    command: python
    args: ["-m", "src.mcp_server"]
    working_dir: /path/to/mcp-oci-database-observatory
    enabled: true
    domains:
      - database
      - opsi
      - logan
```

### 6.2 Agent Definitions

Update agent definitions to reference domains, not tool lists:

```python
# In src/agents/database/troubleshoot.py
class DbTroubleshootAgent(BaseAgent):
    @classmethod
    def get_definition(cls) -> AgentDefinition:
        return AgentDefinition(
            agent_id="db-troubleshoot-agent",
            name="Database Troubleshooting Agent",
            domains=["database", "opsi"],  # NEW: domains not tools
            capabilities=[
                "database-analysis",
                "performance-diagnostics",
                "sql-tuning",
            ],
            ...
        )
```

---

## 7. Implementation Plan

### Phase 5.1: Tool Naming (src/mcp/catalog.py)
1. Add `TOOL_ALIASES` dictionary
2. Update `get_tool()` to check aliases
3. Add `DOMAIN_PREFIXES` mapping
4. Add `get_tools_for_domain()` method

### Phase 5.2: Dynamic Tool Discovery (src/agents/react_agent.py)
1. Remove `DOMAIN_PROMPTS` hardcoded tool lists
2. Add `_get_tools_context()` async method
3. Update `SpecializedReActAgent._get_compartment_context()`

### Phase 5.3: Slack → Coordinator (src/channels/slack.py)
1. Update `_invoke_coordinator()` to use `LangGraphCoordinator`
2. Remove `_route_to_agent()` keyword routing
3. Add response transformation for Slack blocks

### Phase 5.4: Agent Delegation (src/agents/coordinator/nodes.py)
1. Add `delegate_node`
2. Update graph edges for delegation
3. Add `AgentDelegation` protocol

---

## 8. Testing Strategy

| Test | File | Purpose |
|------|------|---------|
| Tool alias resolution | test_mcp_catalog.py | Verify aliases work |
| Domain tool discovery | test_mcp_catalog.py | Get tools by domain |
| Slack → Coordinator | test_slack.py | End-to-end flow |
| Agent delegation | test_coordinator.py | Multi-agent workflows |

---

## 9. Rollback Plan

If issues arise:
1. Tool aliases allow gradual migration
2. Keyword routing can be re-enabled as fallback
3. Feature flags for delegation

---

*End of Architecture Design Document*
