# AGENT.md - Agent Architecture Documentation

## Overview

The OCI AI Agent Coordinator implements a **Workflow-First** multi-agent architecture using LangGraph. The system prioritizes deterministic workflows for known tasks and falls back to agentic behavior for ambiguous requests.

## Architecture Principles

### 1. Workflow-First Design
- **70%+ of requests** should be handled by deterministic workflows
- Workflows are predefined LangGraph subgraphs with explicit steps
- Agentic fallback only for novel or complex requests

### 2. Tool-Centric Approach
- Agents reason and plan; MCP tools execute
- Tools are organized into tiers by latency and risk
- Progressive disclosure: agents discover tools as needed

### 3. State Persistence
- Conversation state persisted via LangGraph checkpoints
- Redis backend for production scalability
- Thread-based session management

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT CHANNELS                                    │
├──────────┬──────────┬──────────┬──────────┬──────────────────────────────────┤
│  Slack   │  Teams   │   Web    │   API    │           OCI Events             │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴─────────────────┬────────────────┘
     │          │          │          │                       │
     └──────────┴──────────┴────┬─────┴───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   INTENT CLASSIFIER   │
                    │   (Workflow Router)   │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
     ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
     │   WORKFLOW     │ │   AGENTIC    │ │    HUMAN     │
     │   (Predefined) │ │   (LLM-led)  │ │   (Approval) │
     └────────┬───────┘ └──────┬───────┘ └──────┬───────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │         TOOL CATALOG            │
              │   (MCP Tool Registry + Router)  │
              └────────────────┬────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  OCI OPSI     │     │  OCI Logan    │     │  OCI Unified  │
│  (DB Perf)    │     │  (Logs)       │     │  (Infra)      │
└───────────────┘     └───────────────┘     └───────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │   ORACLE CLOUD INFRASTRUCTURE   │
              └─────────────────────────────────┘
```

## Agent Types

### 1. Coordinator Agent

**Role**: Master orchestrator - routes requests, manages state, aggregates responses

**Implementation**: `src/agents/coordinator/graph.py`

```python
class CoordinatorState:
    messages: list[BaseMessage]       # Conversation history
    tool_calls: list[dict]            # Pending tool calls
    tool_results: list[dict]          # Tool execution results
    iteration: int                    # Loop counter
    max_iterations: int               # Guard against infinite loops
    error: str | None                 # Error state
```

**Graph Nodes**:
| Node | Purpose |
|------|---------|
| `input` | Normalize and prepare request |
| `classifier` | Route to workflow or agentic |
| `workflow` | Execute deterministic steps |
| `agent` | LLM reasoning with tools |
| `action` | Execute MCP tool calls |
| `human` | Await user approval |
| `output` | Format and return response |

**Routing Logic**:
```
input → classifier → [workflow|agent]
                         ↓
                      action ←→ agent (loop)
                         ↓
                       human (if needed)
                         ↓
                      output
```

### 2. Database Observatory Agent

**Role**: Specialized multi-database observability agent

**Triggers**: `database`, `slow query`, `AWR`, `wait events`, `blocking`, `performance`

**Capabilities**:
- Oracle, MySQL, PostgreSQL support
- AWR/ASH analysis
- Wait event interpretation
- SQL tuning recommendations
- Blocking session detection

**Workflow Steps** (7-step RCA):
1. Symptom Detection (LLM)
2. Blocking Sessions Check (SQLcl)
3. CPU & Wait Event Analysis (OPSI)
4. SQL Monitoring (SQLcl)
5. Long Operations Check (SQLcl)
6. Parallelism Analysis (SQLcl)
7. Archive & Report (LLM)

### 3. Log Analytics Agent

**Role**: Log search, correlation, and pattern analysis

**Triggers**: `logs`, `error logs`, `log search`, `audit`, `log patterns`

**Capabilities**:
- Log Analytics query construction
- Error pattern detection
- Service log correlation
- Audit log analysis
- Log-based alerting

**Tools**: `search_logs`, `get_log_sources`, `analyze_trends`, `execute_logan_query`

### 4. Security & Threat Agent

**Role**: Threat hunting and security posture assessment

**Triggers**: `security`, `threat`, `MITRE`, `attack`, `vulnerability`, `compliance`

**Capabilities**:
- MITRE ATT&CK mapping
- Cloud Guard problem analysis
- Security posture assessment
- Threat indicator correlation
- IOC analysis

**Tools**: `get_security_events`, `mitre_analysis`, `list_policies`, `get_audit_events`

### 5. FinOps Agent

**Role**: Cost analysis and optimization recommendations

**Triggers**: `cost`, `spending`, `budget`, `forecast`, `optimization`, `FinOps`

**Capabilities**:
- Cost breakdown by service/compartment
- Spending anomaly detection
- Budget tracking and alerts
- Rightsizing recommendations
- Reserved capacity analysis

**Tools**: `get_cost_summary`, `get_usage_report`, `analyze_cost_anomaly`

### 6. Infrastructure Agent

**Role**: Compute, network, and storage management

**Triggers**: `compute`, `instance`, `VM`, `network`, `VCN`, `storage`

**Capabilities**:
- Instance lifecycle management
- Network topology analysis
- Storage operations
- Capacity planning
- Resource inventory

**Tools**: `list_instances`, `start_instance`, `stop_instance`, `list_vcns`, `list_subnets`

## Tool Architecture

### Tool Tiers

| Tier | Latency | Risk Level | Examples | Approval |
|------|---------|------------|----------|----------|
| 1 | <100ms | None | `search_databases`, `ping` | Auto |
| 2 | 100ms-1s | Low | `list_instances`, `get_metrics` | Auto |
| 3 | 1-10s | Medium | `execute_sql`, `get_awr_report` | Auto |
| 4 | 10s+ | High | `start_instance`, `stop_database` | Human |

### Tool Catalog

The `ToolCatalog` aggregates tools from all MCP servers:

```python
class ToolCatalog:
    def refresh(self) -> None:
        """Discover tools from all connected servers."""

    def get_tool(self, name: str) -> ToolDefinition:
        """Get tool by name (handles namespacing)."""

    def search(self, query: str) -> list[ToolDefinition]:
        """Semantic search for relevant tools."""

    def to_langchain_tools(self) -> list[StructuredTool]:
        """Convert to LangChain format for LLM binding."""
```

### Progressive Disclosure

Instead of exposing 50+ tools, agents discover tools dynamically:

```python
# High-level skill (exposed by default)
def troubleshoot_instance(instance_id: str) -> str:
    """Comprehensive instance troubleshooting."""

# Low-level tools (discovered as needed)
def get_instance_metrics(instance_id: str) -> dict: ...
def get_instance_logs(instance_id: str) -> list: ...
def check_security_lists(instance_id: str) -> dict: ...
```

## State Management

### CoordinatorState Schema

```python
@dataclass
class CoordinatorState:
    # Message History
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Tool Execution
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    # Loop Control
    iteration: int
    max_iterations: int

    # Context
    oci_context: dict[str, str]  # compartment, region, tenancy
    conversation_topic: str | None

    # Error Handling
    error: str | None
```

### Persistence

```python
# Development: In-memory
checkpointer = MemorySaver()

# Production: Redis
from langgraph.checkpoint.redis import RedisSaver
checkpointer = RedisSaver(redis_url="redis://localhost:6379")
```

### Thread Management

```python
# Each conversation has a unique thread
config = {"configurable": {"thread_id": "user_123_session_456"}}

# Resume conversation
result = await coordinator.invoke("follow up question", config)
```

## Intent Classification

### Classification Categories

| Category | Pattern | Target |
|----------|---------|--------|
| `database.troubleshoot` | slow query, db hang, lock | DB Agent |
| `database.performance` | AWR, ASH, wait events | DB Agent |
| `logs.search` | find logs, show errors | Log Agent |
| `logs.analyze` | correlate, pattern, anomaly | Log Agent |
| `security.threat` | attack, breach, suspicious | Security Agent |
| `security.compliance` | compliance, audit, policy | Security Agent |
| `cost.analyze` | spending, cost breakdown | FinOps Agent |
| `cost.optimize` | reduce cost, rightsizing | FinOps Agent |
| `infra.manage` | start, stop, create | Infra Agent |
| `infra.analyze` | inventory, topology | Infra Agent |

### Confidence-Based Routing

```python
def classify_intent(query: str) -> tuple[str, float]:
    """Classify query into intent with confidence score."""
    # Returns: (intent_category, confidence)
    # e.g., ("database.troubleshoot", 0.95)
```

Routing rules:
- **confidence >= 0.8**: Route directly to workflow
- **0.5 <= confidence < 0.8**: Route to agentic fallback
- **confidence < 0.5**: Ask clarifying question

## Multi-Agent Workflows

### Sequential Workflow

```python
# Example: "Why did costs spike and is it a security issue?"
workflow = (
    finops_agent.analyze_spike()    # Step 1: Find cost spike
    >> extract_resources            # Step 2: Extract affected resources
    >> security_agent.check_threat  # Step 3: Check for threats
    >> aggregate_findings           # Step 4: Combine results
)
```

### Parallel Workflow

```python
# Example: "Give me a status overview"
results = await asyncio.gather(
    db_agent.get_health(),
    infra_agent.get_inventory(),
    finops_agent.get_cost_summary(),
    security_agent.get_posture(),
)
```

## Guardrails & Safety

### Confirmation Required

These actions require explicit user approval:
- `DELETE` operations
- `STOP` on production resources
- Budget modifications
- Security policy changes
- Access control modifications

### Rate Limiting

```python
RATE_LIMIT = {
    "requests_per_minute": 10,
    "cooldown_threshold": 8,
    "max_concurrent_agents": 5,
}
```

### Audit Logging

All actions are logged with:
- User ID and session
- Request intent classification
- Agent routing decisions
- Tool calls and results
- Errors and exceptions

## Evaluation Metrics

### Success Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| Task Success Rate | >85% | User accepts solution without retry |
| Workflow Ratio | >70% | Requests handled by deterministic workflows |
| Tool Accuracy | >90% | Correct tool selection and parameters |
| MTTR Reduction | >50% | vs. manual console operations |
| Human Intervention | <10% | Approval node triggers |

### LLM-as-a-Judge Evaluation

```python
evaluation_criteria = {
    "is_correct": "Did the final answer match ground truth?",
    "is_safe": "Did it respect guardrails?",
    "is_efficient": "Did it use minimum steps?",
}
```

## Extension Points

### Adding a New Agent

1. **Create System Prompt**: `prompts/NN-AGENT-NAME.md`
2. **Implement Agent Class**: `src/agents/agent_name/`
3. **Register Routing**: Update coordinator intent classification
4. **Add Integration Tests**: `tests/integration/test_agent_name.py`

### Adding a Workflow

1. **Define Subgraph**: `src/agents/coordinator/workflows/`
2. **Register Intent**: Map keywords to workflow
3. **Add Evaluation Cases**: Gold-standard test cases

### Adding an MCP Server

1. **Configure Server**: `config/mcp-servers.json`
2. **Set Environment**: Server URL and auth
3. **Test Connection**: `poetry run pytest tests/mcp/`
