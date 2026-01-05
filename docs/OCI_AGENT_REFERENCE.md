# OCI Agent Reference Document

## Overview

This document defines the standard specification for OCI AI Agents within the OCI Coordinator system. All agents—whether built-in, specialized, or custom—must conform to this specification to ensure proper registration, discovery, orchestration, and monitoring.

## Agent Object Schema

Every agent is defined by a structured object that enables automatic registration, health monitoring, and capability discovery.

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

class AgentStatus(str, Enum):
    """Agent lifecycle status."""
    REGISTERED = "registered"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"

@dataclass
class AgentMetadata:
    """Agent metadata for versioning and configuration."""
    version: str = "1.0.0"
    namespace: str = "oci-coordinator"
    max_iterations: int = 15
    timeout_seconds: int = 300
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_multiplier": 2,
        "initial_delay_ms": 1000
    })

@dataclass
class AgentDefinition:
    """
    Complete Agent Object Definition.

    This schema defines all attributes required for an agent to be
    registered in the Agent Catalog and orchestrated by the Coordinator.
    """
    # Identity
    agent_id: str                    # Unique identifier (e.g., "db-troubleshoot-agent-a1b2c3")
    role: str                        # Agent role name (e.g., "db-troubleshoot-agent")

    # Capabilities & Skills
    capabilities: List[str]          # High-level capability domains
    skills: List[str]                # Specific skills/workflows agent can execute

    # Health & Monitoring
    health_endpoint: str             # Health check URL

    # Metadata
    metadata: AgentMetadata          # Configuration and version info
    description: str                 # Human-readable description

    # Runtime State
    status: AgentStatus = AgentStatus.REGISTERED
    registered_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None

    # MCP Integration
    mcp_tools: List[str] = field(default_factory=list)  # MCP tools agent can invoke
    mcp_servers: List[str] = field(default_factory=list)  # MCP servers agent connects to
```

## Example Agent Definition

```json
{
  "agent_id": "db-troubleshoot-agent-c5b6cd64b",
  "role": "db-troubleshoot-agent",
  "capabilities": [
    "database-analysis",
    "performance-diagnostics",
    "sql-tuning",
    "blocking-analysis"
  ],
  "skills": [
    "rca_workflow",
    "blocking_analysis",
    "wait_event_analysis",
    "awr_analysis"
  ],
  "health_endpoint": "http://localhost:8010/health",
  "metadata": {
    "version": "1.0.0",
    "namespace": "oci-coordinator",
    "max_iterations": 15,
    "timeout_seconds": 300
  },
  "description": "Database Expert Agent for performance analysis, troubleshooting, and SQL tuning. Provides root cause analysis using AWR, ASH, and wait event diagnostics.",
  "status": "healthy",
  "registered_at": "2026-01-01T10:00:00.000000+00:00",
  "last_heartbeat": "2026-01-02T14:30:00.000000+00:00",
  "mcp_tools": [
    "oci_database_list_autonomous",
    "oci_database_get_autonomous",
    "oci_opsi_get_fleet_summary",
    "oci_opsi_analyze_cpu",
    "oci_database_execute_sql"
  ],
  "mcp_servers": [
    "oci-unified",
    "database-observatory"
  ]
}
```

---

## Naming Conventions

### Agent Naming

| Component | Convention | Example |
|-----------|------------|---------|
| Agent Role | `{domain}-{function}-agent` | `db-troubleshoot-agent` |
| Agent ID | `{role}-{uuid-suffix}` | `db-troubleshoot-agent-c5b6cd64b-2v8rr` |
| Agent Class | `{Domain}{Function}Agent` | `DbTroubleshootAgent` |
| Agent Module | `src/agents/{domain}/{function}.py` | `src/agents/database/troubleshoot.py` |

### Capability Naming

| Category | Convention | Example |
|----------|------------|---------|
| Domain Capability | `{domain}-{action}` | `database-analysis` |
| Skill Name | `{workflow}_workflow` | `rca_workflow` |
| Tool Name | `oci_{domain}_{action}_{resource}` | `oci_database_list_autonomous` |

### File Structure

```
src/agents/
├── __init__.py
├── base.py                      # BaseAgent class
├── catalog.py                   # AgentCatalog with auto-registration
├── skills.py                    # Skill definitions and executor
├── react_agent.py               # ReAct agent with tool discovery
├── coordinator/
│   ├── __init__.py
│   ├── graph.py                 # LangGraph coordinator
│   ├── orchestrator.py          # Parallel multi-agent orchestration
│   ├── workflows.py             # 16 pre-built deterministic workflows
│   ├── nodes.py                 # Graph node implementations
│   └── state.py                 # CoordinatorState
├── database/
│   ├── __init__.py
│   └── troubleshoot.py          # DbTroubleshootAgent
├── log_analytics/
│   ├── __init__.py
│   └── agent.py                 # LogAnalyticsAgent
├── security/
│   ├── __init__.py
│   └── agent.py                 # SecurityThreatAgent
├── finops/
│   ├── __init__.py
│   └── agent.py                 # FinOpsAgent
├── infrastructure/
│   ├── __init__.py
│   └── agent.py                 # InfrastructureAgent
├── error_analysis/
│   ├── __init__.py
│   ├── agent.py                 # ErrorAnalysisAgent
│   └── todo_manager.py          # AdminTodoManager for action items
└── self_healing/
    ├── __init__.py
    ├── analyzer.py              # ErrorAnalyzer - categorize errors
    ├── corrector.py             # ParameterCorrector - fix params
    ├── validator.py             # LogicValidator - pre-execution
    ├── retry.py                 # RetryStrategy - smart retries
    └── mixin.py                 # SelfHealingMixin for agents
```

---

## Agent Catalog

The Agent Catalog provides automatic registration, discovery, and lifecycle management for all agents.

### Catalog Interface

```python
from typing import Dict, List, Optional, Type
from pathlib import Path
import importlib
import pkgutil

class AgentCatalog:
    """
    Central registry for all agents with auto-discovery.

    Features:
    - Auto-registration from agents directory
    - Health monitoring and status tracking
    - Capability-based agent lookup
    - Dynamic agent loading
    """

    _instance: Optional['AgentCatalog'] = None
    _agents: Dict[str, AgentDefinition] = {}

    @classmethod
    def get_instance(cls) -> 'AgentCatalog':
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._agents = {}
        self._agent_classes: Dict[str, Type['BaseAgent']] = {}

    def auto_discover(self, agents_path: str = "src/agents") -> None:
        """
        Auto-discover and register agents from the agents directory.

        Scans all Python modules in the agents directory and registers
        any class that inherits from BaseAgent.
        """
        agents_dir = Path(agents_path)

        for module_info in pkgutil.walk_packages([str(agents_dir)]):
            if module_info.name.startswith('_'):
                continue

            try:
                module = importlib.import_module(f"src.agents.{module_info.name}")

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, BaseAgent) and
                        attr is not BaseAgent):
                        self.register_agent_class(attr)

            except ImportError as e:
                logger.warning(f"Failed to import agent module: {module_info.name}", error=str(e))

    def register(self, agent: AgentDefinition) -> None:
        """Register an agent definition."""
        self._agents[agent.agent_id] = agent
        logger.info(
            "Agent registered",
            agent_id=agent.agent_id,
            role=agent.role,
            capabilities=agent.capabilities
        )

    def register_agent_class(self, agent_class: Type['BaseAgent']) -> None:
        """Register an agent class for instantiation."""
        agent_def = agent_class.get_definition()
        self._agents[agent_def.role] = agent_def
        self._agent_classes[agent_def.role] = agent_class

    def get(self, agent_id: str) -> Optional[AgentDefinition]:
        """Get agent by ID or role."""
        return self._agents.get(agent_id)

    def get_by_capability(self, capability: str) -> List[AgentDefinition]:
        """Find agents that have a specific capability."""
        return [
            agent for agent in self._agents.values()
            if capability in agent.capabilities
        ]

    def get_by_skill(self, skill: str) -> List[AgentDefinition]:
        """Find agents that can execute a specific skill."""
        return [
            agent for agent in self._agents.values()
            if skill in agent.skills
        ]

    def list_all(self) -> List[AgentDefinition]:
        """List all registered agents."""
        return list(self._agents.values())

    def list_healthy(self) -> List[AgentDefinition]:
        """List only healthy agents."""
        return [
            agent for agent in self._agents.values()
            if agent.status == AgentStatus.HEALTHY
        ]

    async def health_check_all(self) -> Dict[str, AgentStatus]:
        """Run health checks on all agents."""
        results = {}
        for agent_id, agent in self._agents.items():
            try:
                # Check health endpoint
                status = await self._check_agent_health(agent)
                agent.status = status
                agent.last_heartbeat = datetime.utcnow()
                results[agent_id] = status
            except Exception as e:
                agent.status = AgentStatus.UNHEALTHY
                results[agent_id] = AgentStatus.UNHEALTHY
        return results

    async def _check_agent_health(self, agent: AgentDefinition) -> AgentStatus:
        """Check individual agent health."""
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(agent.health_endpoint)
            if response.status_code == 200:
                return AgentStatus.HEALTHY
            return AgentStatus.UNHEALTHY
```

### Auto-Registration Pattern

Agents are automatically discovered when the application starts:

```python
# In src/agents/__init__.py
from src.agents.catalog import AgentCatalog

def initialize_agents():
    """Initialize agent catalog with auto-discovery."""
    catalog = AgentCatalog.get_instance()
    catalog.auto_discover()
    return catalog

# In application startup
catalog = initialize_agents()
```

---

## Shared Memory Layer

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHARED MEMORY ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │  Coordinator     │      │   DB Agent       │      │   Log Agent      │  │
│  │      Agent       │      │                  │      │                  │  │
│  └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘  │
│           │                         │                         │            │
│           └─────────────────────────┼─────────────────────────┘            │
│                                     │                                       │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │                    MEMORY ABSTRACTION LAYER                          │   │
│  │                   (src/memory/manager.py)                            │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                    ┌────────────────────────────────┐                      │
│                    │                                │                      │
│                    ▼                                ▼                      │
│           ┌─────────────────┐             ┌─────────────────┐             │
│           │     REDIS       │             │  LangGraph      │             │
│           │   (Hot Cache)   │             │  MemorySaver    │             │
│           │                 │             │                 │             │
│           │ • Session state │             │ • Graph state   │             │
│           │ • Tool results  │             │ • Iteration     │             │
│           │ • Agent status  │             │   tracking      │             │
│           │ • TTL: 1hr      │             │ • Checkpoints   │             │
│           └─────────────────┘             └─────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Manager Implementation

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import timedelta
import json

class MemoryStore(ABC):
    """Abstract base for memory storage backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

class RedisMemoryStore(MemoryStore):
    """Redis-based hot cache for fast access."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        import redis.asyncio as redis
        self.client = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[Any]:
        value = await self.client.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        serialized = json.dumps(value, default=str)
        if ttl:
            await self.client.setex(key, ttl, serialized)
        else:
            await self.client.set(key, serialized)

    async def delete(self, key: str) -> None:
        await self.client.delete(key)

class SharedMemoryManager:
    """
    Unified memory manager with Redis cache.

    Uses Redis for hot cache (session state, recent results).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379"
    ):
        self.cache = RedisMemoryStore(redis_url)
        self.default_cache_ttl = timedelta(hours=1)

    async def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Get session state from cache."""
        return await self.cache.get(f"session:{session_id}")

    async def set_session_state(self, session_id: str, state: Dict) -> None:
        """Set session state in cache."""
        await self.cache.set(f"session:{session_id}", state, self.default_cache_ttl)

    async def get_conversation_history(self, thread_id: str) -> Optional[List[Dict]]:
        """Get full conversation history from persistent store."""
        if self.persistent:
            return await self.persistent.get(f"conversation:{thread_id}")
        return await self.cache.get(f"conversation:{thread_id}")

    async def append_conversation(self, thread_id: str, message: Dict) -> None:
        """Append message to conversation history."""
        history = await self.get_conversation_history(thread_id) or []
        history.append(message)

        # Update both cache and persistent
        await self.cache.set(f"conversation:{thread_id}", history, self.default_cache_ttl)
        if self.persistent:
            await self.persistent.set(f"conversation:{thread_id}", history)

    async def get_agent_memory(self, agent_id: str, memory_type: str) -> Optional[Any]:
        """Get agent-specific memory."""
        key = f"agent:{agent_id}:{memory_type}"
        # Try cache first
        result = await self.cache.get(key)
        if result is None and self.persistent:
            result = await self.persistent.get(key)
            if result:
                # Warm cache
                await self.cache.set(key, result, self.default_cache_ttl)
        return result

    async def set_agent_memory(self, agent_id: str, memory_type: str, value: Any) -> None:
        """Set agent-specific memory."""
        key = f"agent:{agent_id}:{memory_type}"
        await self.cache.set(key, value, self.default_cache_ttl)
        if self.persistent:
            await self.persistent.set(key, value)
```

---

## MCP Server Integration

### Reference MCP Servers

The OCI Coordinator integrates with MCP servers from `/Users/abirzu/dev/MCP/`:

| Server | Path | Purpose | Transport |
|--------|------|---------|-----------|
| **oci-unified** | `mcp-oci-new/` | Unified OCI operations | stdio/HTTP |
| **opsi** | `opsi/` | Operations Insights | HTTP |
| **logan** | `logan/` | Logging Analytics | HTTP |
| **finopsai** | `finopsai-mcp/` | FinOps Analysis | HTTP |
| **security** | `oci-mcp-security/` | Security & Compliance | HTTP |

### MCP Configuration

```json
{
  "mcp_servers": {
    "oci-unified": {
      "path": "/Users/abirzu/dev/MCP/mcp-oci-new",
      "command": "uv run python -m mcp_server_oci.server",
      "transport": "stdio",
      "env": {
        "OCI_CONFIG_FILE": "~/.oci/config",
        "OCI_CLI_PROFILE": "DEFAULT"
      }
    },
    "oci-opsi": {
      "path": "/Users/abirzu/dev/MCP/opsi",
      "url": "http://localhost:8000",
      "transport": "http"
    },
    "oci-logan": {
      "path": "/Users/abirzu/dev/MCP/logan",
      "url": "http://localhost:8001",
      "transport": "http"
    }
  }
}
```

---

## Base Agent Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph

class BaseAgent(ABC):
    """
    Base class for all OCI Agents.

    Provides common functionality:
    - Agent definition and registration
    - Memory access
    - MCP tool invocation
    - Observability integration
    """

    def __init__(
        self,
        memory_manager: SharedMemoryManager,
        tool_catalog: 'ToolCatalog',
        config: Dict[str, Any] = None
    ):
        self.memory = memory_manager
        self.tools = tool_catalog
        self.config = config or {}
        self._definition: Optional[AgentDefinition] = None

    @classmethod
    @abstractmethod
    def get_definition(cls) -> AgentDefinition:
        """Return the agent's definition for catalog registration."""
        pass

    @abstractmethod
    async def invoke(self, query: str, context: Dict[str, Any] = None) -> str:
        """Execute the agent's primary function."""
        pass

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the LangGraph for this agent."""
        pass

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke an MCP tool through the tool catalog."""
        tool_def = self.tools.get_tool(tool_name)
        if not tool_def:
            raise ValueError(f"Tool not found: {tool_name}")
        return await self.tools.execute(tool_name, arguments)

    async def save_memory(self, key: str, value: Any) -> None:
        """Save to agent's persistent memory."""
        await self.memory.set_agent_memory(
            self.get_definition().agent_id,
            key,
            value
        )

    async def load_memory(self, key: str) -> Optional[Any]:
        """Load from agent's persistent memory."""
        return await self.memory.get_agent_memory(
            self.get_definition().agent_id,
            key
        )
```

---

## Agent Implementation Example

```python
# src/agents/database/troubleshoot.py

from src.agents.base import BaseAgent, AgentDefinition, AgentMetadata, KafkaTopics
from langgraph.graph import StateGraph, END
from typing import Any, Dict

class DbTroubleshootAgent(BaseAgent):
    """
    Database Troubleshooting Agent.

    Specializes in Oracle database performance analysis,
    root cause analysis, and SQL tuning recommendations.
    """

    @classmethod
    def get_definition(cls) -> AgentDefinition:
        return AgentDefinition(
            agent_id="db-troubleshoot-agent",
            role="db-troubleshoot-agent",
            capabilities=[
                "database-analysis",
                "performance-diagnostics",
                "sql-tuning",
                "blocking-analysis"
            ],
            skills=[
                "rca_workflow",
                "blocking_analysis",
                "wait_event_analysis",
                "awr_analysis"
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.db-troubleshoot-agent"],
                produce=["results.db-troubleshoot-agent"]
            ),
            health_endpoint="http://localhost:8010/health",
            metadata=AgentMetadata(
                version="1.0.0",
                namespace="oci-coordinator",
                max_iterations=15
            ),
            description="Database Expert Agent for performance analysis and troubleshooting.",
            mcp_tools=[
                "oci_database_list_autonomous",
                "oci_database_get_autonomous",
                "oci_observability_get_metrics",
                "oci_observability_query_logs"
            ],
            mcp_servers=["oci-unified", "sqlcl"]
        )

    def build_graph(self) -> StateGraph:
        """Build the 7-step RCA workflow graph."""
        from langgraph.graph import StateGraph

        graph = StateGraph(RCAState)

        # Add nodes
        graph.add_node("detect_symptom", self._detect_symptom)
        graph.add_node("check_blocking", self._check_blocking)
        graph.add_node("analyze_cpu_wait", self._analyze_cpu_wait)
        graph.add_node("check_sql", self._check_sql)
        graph.add_node("check_longops", self._check_longops)
        graph.add_node("check_parallel", self._check_parallel)
        graph.add_node("generate_report", self._generate_report)

        # Add edges with conditional routing
        graph.set_entry_point("detect_symptom")
        graph.add_conditional_edges(
            "detect_symptom",
            self._route_after_symptom,
            {
                "blocking": "check_blocking",
                "cpu": "analyze_cpu_wait",
                "sql": "check_sql",
                "report": "generate_report"
            }
        )
        # ... additional edges

        return graph.compile()

    async def invoke(self, query: str, context: Dict[str, Any] = None) -> str:
        """Execute RCA workflow."""
        graph = self.build_graph()

        initial_state = RCAState(
            query=query,
            context=context or {},
            messages=[],
            findings=[]
        )

        result = await graph.ainvoke(initial_state)
        return result.get("report", "Analysis complete.")
```

---

## Agent Best Practices

This section establishes best practices for building effective OCI agents based on Anthropic's agent design patterns.

### 1. Agentic Loop Design

**Keep the loop simple:**
```python
# Recommended: Simple loop with clear exit conditions
async def agentic_loop(query: str, max_iterations: int = 15) -> str:
    """Simple agentic loop pattern."""
    state = AgentState(query=query, messages=[])

    for iteration in range(max_iterations):
        # Agent decides: tool call or final response
        response = await llm.invoke(state.messages)

        if response.tool_calls:
            # Execute tools and add results to context
            for tool_call in response.tool_calls:
                result = await execute_tool(tool_call)
                state.messages.append(ToolMessage(content=result))
        else:
            # No tools = final response
            return response.content

    return "Max iterations reached. Please refine your query."
```

**Avoid complex orchestration:**
- Prefer single-agent with good tools over multi-agent handoffs
- Use workflows (deterministic subgraphs) for well-defined tasks
- Reserve agentic mode for truly ambiguous or novel queries

### 2. Tool Design Principles

**Progressive Disclosure:**
```python
# Recommended: Start with discovery tool
INITIAL_TOOLS = [
    "oci_search_tools",      # Meta-tool to discover capabilities
    "oci_get_capabilities",  # What can this agent do?
    "oci_ping",              # Health check
]

# Tier-based expansion as needed
def get_tools_for_task(task_domain: str, max_tier: int = 3) -> list:
    """Expand toolset based on task requirements."""
    return catalog.search_tools(domain=task_domain, max_tier=max_tier)
```

**Clear Tool Descriptions:**
```python
# Good: Specific, actionable description
ToolDefinition(
    name="oci_database_get_awr_report",
    description="Retrieve AWR performance report for an Autonomous Database. "
                "Requires db_ocid and time range. Returns CPU, I/O, wait events.",
    input_schema={
        "type": "object",
        "properties": {
            "db_ocid": {"type": "string", "description": "Database OCID"},
            "start_time": {"type": "string", "format": "datetime"},
            "end_time": {"type": "string", "format": "datetime"}
        },
        "required": ["db_ocid"]
    }
)

# Bad: Vague, unhelpful description
# "Gets database info" - doesn't explain what or when to use
```

### 3. Error Handling Patterns

**Graceful Degradation:**
```python
async def call_tool_safely(
    tool_name: str,
    arguments: dict,
    fallback: str = None
) -> ToolCallResult:
    """Execute tool with fallback on failure."""
    try:
        result = await catalog.execute(tool_name, arguments)
        if result.success:
            return result

        # Log error but don't crash
        logger.warning(
            "Tool execution failed",
            tool=tool_name,
            error=result.error
        )

        if fallback:
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=fallback
            )

        return result

    except asyncio.TimeoutError:
        return ToolCallResult(
            tool_name=tool_name,
            success=False,
            error="Tool execution timed out"
        )
```

**Retry with Backoff:**
```python
async def execute_with_retry(
    tool_name: str,
    arguments: dict,
    max_retries: int = 3
) -> ToolCallResult:
    """Execute tool with exponential backoff."""
    for attempt in range(max_retries):
        result = await catalog.execute(tool_name, arguments)

        if result.success:
            return result

        if attempt < max_retries - 1:
            delay = (2 ** attempt) * 1.0  # 1s, 2s, 4s
            await asyncio.sleep(delay)

    return result
```

### 4. Confirmation Patterns for Risky Operations

**Human-in-the-Loop:**
```python
APPROVAL_REQUIRED_TOOLS = {
    "oci_compute_stop_instance",
    "oci_compute_terminate_instance",
    "oci_database_scale",
    "oci_database_delete",
}

async def execute_with_approval(
    tool_name: str,
    arguments: dict,
    channel: str
) -> ToolCallResult:
    """Execute tool, requesting approval for risky operations."""

    if tool_name in APPROVAL_REQUIRED_TOOLS:
        # Generate approval request
        approval_message = format_approval_request(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=get_risk_level(tool_name)
        )

        # Send to user and wait for confirmation
        approved = await request_user_approval(
            channel=channel,
            message=approval_message,
            timeout_seconds=300
        )

        if not approved:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error="Operation cancelled: User denied approval"
            )

    return await catalog.execute(tool_name, arguments)
```

### 5. Context Management

**Efficient Context Usage:**
```python
@dataclass
class AgentContext:
    """Managed context for agent execution."""

    # Essential context (always included)
    query: str
    session_id: str
    user_id: str

    # Domain context (included when relevant)
    compartment_id: str | None = None
    resource_focus: dict | None = None

    # Conversation summary (not full history)
    summary: str | None = None

    def to_system_prompt_context(self) -> str:
        """Format context for LLM system prompt."""
        parts = [f"User query: {self.query}"]

        if self.compartment_id:
            parts.append(f"OCI Compartment: {self.compartment_id}")

        if self.resource_focus:
            parts.append(f"Focus: {self.resource_focus}")

        if self.summary:
            parts.append(f"Previous context: {self.summary}")

        return "\n".join(parts)
```

**Conversation Summarization:**
```python
async def summarize_if_needed(
    messages: list[BaseMessage],
    max_messages: int = 10
) -> tuple[list[BaseMessage], str | None]:
    """Summarize older messages to manage context window."""

    if len(messages) <= max_messages:
        return messages, None

    # Keep recent messages
    recent = messages[-max_messages:]
    old = messages[:-max_messages]

    # Summarize old messages
    summary = await llm.invoke([
        SystemMessage(content="Summarize this conversation concisely:"),
        *old
    ])

    return recent, summary.content
```

### 6. Skill Execution Framework

**Skill Definition:**
```python
@dataclass
class SkillDefinition:
    """Definition of a reusable workflow/skill."""

    name: str                    # e.g., "rca_workflow"
    description: str             # What the skill does
    required_tools: list[str]    # MCP tools needed
    steps: list[str]             # Workflow steps
    estimated_duration: str      # e.g., "30-60 seconds"

    def validate_tools(self, catalog: ToolCatalog) -> bool:
        """Verify all required tools are available."""
        for tool in self.required_tools:
            if not catalog.get_tool(tool):
                return False
        return True

# Example skill
RCA_WORKFLOW = SkillDefinition(
    name="rca_workflow",
    description="7-step root cause analysis for database performance issues",
    required_tools=[
        "oci_database_get_autonomous",
        "oci_observability_get_metrics",
        "oci_observability_query_logs"
    ],
    steps=[
        "detect_symptom",
        "check_blocking",
        "analyze_wait_events",
        "check_sql_performance",
        "check_longops",
        "check_parallel_queries",
        "generate_report"
    ],
    estimated_duration="30-60 seconds"
)
```

**Skill Execution:**
```python
class SkillExecutor:
    """Executes defined skills/workflows."""

    def __init__(self, catalog: ToolCatalog, memory: SharedMemoryManager):
        self.catalog = catalog
        self.memory = memory
        self.skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill."""
        if skill.validate_tools(self.catalog):
            self.skills[skill.name] = skill

    async def execute(
        self,
        skill_name: str,
        context: dict
    ) -> dict:
        """Execute a registered skill."""
        skill = self.skills.get(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        results = {"skill": skill_name, "steps": []}

        for step in skill.steps:
            step_result = await self._execute_step(step, context, results)
            results["steps"].append(step_result)

            # Update context with step results
            context[f"{step}_result"] = step_result

        return results
```

### 7. Observability Best Practices

**Structured Logging:**
```python
import structlog

logger = structlog.get_logger(__name__)

async def invoke_with_logging(
    agent_id: str,
    query: str,
    context: dict
) -> str:
    """Agent invocation with comprehensive logging."""

    log = logger.bind(
        agent_id=agent_id,
        query_preview=query[:100],
        session_id=context.get("session_id")
    )

    log.info("agent_invoke_start")
    start_time = time.time()

    try:
        result = await agent.invoke(query, context)

        log.info(
            "agent_invoke_complete",
            duration_ms=(time.time() - start_time) * 1000,
            response_length=len(result)
        )

        return result

    except Exception as e:
        log.error(
            "agent_invoke_error",
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000
        )
        raise
```

**OpenTelemetry Tracing:**
```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

async def invoke_with_tracing(agent_id: str, query: str) -> str:
    """Agent invocation with OTEL tracing."""

    with tracer.start_as_current_span(
        f"agent.{agent_id}.invoke",
        attributes={
            "agent.id": agent_id,
            "query.length": len(query)
        }
    ) as span:
        try:
            result = await agent.invoke(query)
            span.set_attribute("response.length", len(result))
            span.set_status(Status(StatusCode.OK))
            return result

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
```

### 8. Testing Agents

**Unit Testing Pattern:**
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_catalog():
    """Mock tool catalog for testing."""
    catalog = MagicMock(spec=ToolCatalog)
    catalog.execute = AsyncMock(return_value=ToolCallResult(
        tool_name="test_tool",
        success=True,
        result={"data": "test"}
    ))
    return catalog

@pytest.fixture
def mock_memory():
    """Mock memory manager for testing."""
    memory = MagicMock(spec=SharedMemoryManager)
    memory.get_session_state = AsyncMock(return_value={})
    memory.set_session_state = AsyncMock()
    return memory

@pytest.mark.asyncio
async def test_agent_invoke(mock_catalog, mock_memory):
    """Test agent invocation."""
    agent = DbTroubleshootAgent(
        memory_manager=mock_memory,
        tool_catalog=mock_catalog
    )

    result = await agent.invoke(
        query="Why is my database slow?",
        context={"db_ocid": "ocid1.database.test"}
    )

    assert result is not None
    assert "analysis" in result.lower() or "performance" in result.lower()
```

**Integration Testing:**
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_mcp_server():
    """Test agent with actual MCP server connection."""

    # Start test MCP server
    registry = ServerRegistry.get_instance()
    registry.register_from_dict("test-server", {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "test_mcp_server"]
    })
    await registry.connect("test-server")

    try:
        catalog = ToolCatalog.get_instance(registry)
        await catalog.refresh()

        agent = DbTroubleshootAgent(
            memory_manager=SharedMemoryManager(),
            tool_catalog=catalog
        )

        result = await agent.invoke("List databases")
        assert "database" in result.lower()

    finally:
        await registry.disconnect_all()
```

---

## Error Analysis Agent

The Error Analysis Agent scans OCI logs for errors and creates admin todo items for significant issues.

### Capabilities

| Feature | Description |
|---------|-------------|
| Error Detection | Scan OCI logs for error patterns |
| Pattern Recognition | Detect ORA-, OOM, timeout, auth failures |
| LLM Analysis | Analyze patterns for root cause |
| Todo Management | Create action items for admins |

### Error Patterns

| Pattern | Severity | Category |
|---------|----------|----------|
| ORA-00060 (Deadlock) | CRITICAL | database |
| ORA-04031 (Shared Pool) | CRITICAL | database |
| OutOfMemory/OOM | CRITICAL | compute |
| Connection timeout | HIGH | network |
| Authentication failed | HIGH | security |
| HTTP 4xx/5xx | MEDIUM | api |

### Admin Todo Manager

```python
from src.agents.error_analysis import ErrorAnalysisAgent

agent = ErrorAnalysisAgent(llm=llm, tool_catalog=catalog)

# Analyze logs for errors
result = await agent.invoke("Scan logs for errors", time_range_hours=1)

# Get pending admin todos
todos = await agent.get_pending_todos()

# Resolve a todo
await agent.resolve_todo("todo-abc123", "Fixed by increasing connection pool")
```

---

## Self-Healing Framework

The self-healing framework provides automatic error recovery and parameter correction.

### Components

| Component | Purpose |
|-----------|---------|
| `ErrorAnalyzer` | Categorize errors, suggest recovery |
| `ParameterCorrector` | Fix incorrect tool parameters |
| `LogicValidator` | Pre-execution validation |
| `RetryStrategy` | Smart retry with backoff |
| `SelfHealingMixin` | Mixin for agent inheritance |

### Error Categories

| Category | Recovery Action |
|----------|-----------------|
| `PERMISSION` | Suggest IAM policy fix |
| `NOT_FOUND` | Parameter correction |
| `TIMEOUT` | Retry with backoff |
| `RATE_LIMIT` | Wait and retry |
| `VALIDATION` | Correct parameters |
| `TRANSIENT` | Simple retry |

### Using Self-Healing in Agents

```python
from src.agents.base import BaseAgent
from src.agents.self_healing import SelfHealingMixin

class MyAgent(BaseAgent, SelfHealingMixin):
    def __init__(self, llm, tool_catalog, ...):
        super().__init__(...)
        self.init_self_healing(llm, max_retries=3)

    async def invoke(self, query):
        # Use healing_call_tool for auto-retry with correction
        result = await self.healing_call_tool(
            "oci_database_execute_sql",
            {"query": "SELECT * FROM users"},
            user_intent=query,
        )
        return result
```

---

## Resilience Infrastructure

Production-grade resilience patterns for fault tolerance.

### Components

| Component | Purpose |
|-----------|---------|
| `DeadLetterQueue` | Persist failed operations for retry |
| `Bulkhead` | Resource isolation between domains |
| `HealthMonitor` | Component health with auto-restart |

### Bulkhead Partitions

| Partition | Max Concurrent | Tool Prefixes |
|-----------|----------------|---------------|
| DATABASE | 3 | `oci_database_`, `oci_opsi_` |
| INFRASTRUCTURE | 5 | `oci_compute_`, `oci_network_` |
| COST | 2 | `oci_cost_` |
| SECURITY | 3 | `oci_security_` |
| DISCOVERY | 2 | `oci_search_`, `oci_list_` |
| LLM | 5 | LLM calls |

### Circuit Breaker

The tool catalog includes circuit breaker logic:
- Tracks consecutive failures per MCP server
- Opens circuit after 3 failures (60s cooldown)
- Rejects tool calls to unhealthy servers immediately
- Auto-closes circuit after successful health check

```python
from src.resilience import Bulkhead, DeadLetterQueue, HealthMonitor

# Bulkhead for resource isolation
bulkhead = Bulkhead.get_instance()
async with bulkhead.acquire("database"):
    result = await execute_database_operation()

# Health monitor with auto-restart
monitor = HealthMonitor.get_instance()
monitor.register_check(HealthCheck(
    name="mcp_database",
    check_func=check_mcp_health,
    restart_func=restart_mcp_server,
    failure_threshold=3,
))
await monitor.start()
```

---

## Summary

This reference document establishes:

1. **Standard Agent Schema** - Consistent structure for all agents
2. **Naming Conventions** - Clear patterns for IDs, classes, and files
3. **Auto-Registration** - Automatic discovery and catalog management
4. **Shared Memory** - Redis cache for session state and tool results
5. **MCP Integration** - Reference servers and configuration patterns
6. **Base Implementation** - Reusable base class for agent development
7. **Best Practices** - Agentic loop design, error handling, observability
8. **Skill Framework** - Reusable workflow definitions and execution
9. **Testing Patterns** - Unit and integration testing strategies
10. **Error Analysis Agent** - Log scanning and admin todo management
11. **Self-Healing Framework** - Automatic error recovery and retries
12. **Resilience Infrastructure** - Bulkhead, circuit breaker, dead letter queues

All new agents must follow this specification to ensure proper integration with the OCI Coordinator orchestration system.
