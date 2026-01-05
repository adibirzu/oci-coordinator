# Phase 4: Multi-Agent Enhancement Plan

> **Note (2026-01):** ATP persistent memory references in this planning document are historical. ATP has been removed from the codebase; LangGraph now uses MemorySaver exclusively.

**Based on Best Practices From:**
- Anthropic Engineering: Multi-Agent Research System
- OCI AI Agent Platform: First Principles
- A2A + MCP Protocol Integration Patterns
- ClickHouse: 12 Agent Frameworks Analysis

---

## Executive Summary

The current implementation has solid foundations (LangGraph coordinator, MCP tools, Slack integration) but lacks several production-grade patterns identified in the best practices research:

| Area | Current State | Target State |
|------|---------------|--------------|
| **Agent Orchestration** | Sequential routing | Parallel orchestrator-worker pattern |
| **Agent Communication** | Direct invocation | A2A protocol with structured messages |
| **Memory (ATP)** | Schema defined, not connected | Full OCI ATP integration with checkpoints |
| **Slack Integration** | asyncio.run() per event | Proper async event loop management |
| **Error Recovery** | Basic try/catch | Checkpoints + graceful degradation |
| **Observability** | OCI APM + Logging | + Agent decision tracing |

---

## Gap Analysis

### 1. Coordinator Agent Gaps

**Current (`src/agents/coordinator/graph.py`):**
- Single-threaded sequential execution
- No parallel subagent spawning
- Intent classification relies on single LLM call
- MemorySaver is in-memory only (not persistent)

**Best Practice (Anthropic):**
- Lead agent coordinates 3-5 subagents in parallel
- Dynamic task decomposition based on complexity
- External memory preservation before context exceeds limits
- Token efficiency optimization (explains 80% of performance)

**Gaps Identified:**
1. No parallel agent execution capability
2. No dynamic complexity assessment
3. Checkpointer is not persistent (data lost on restart)
4. No context compression for long conversations

### 2. Agent Communication (A2A) Gaps

**Current (`src/agents/coordinator/nodes.py`):**
- Direct function calls between coordinator and agents
- No structured message format
- No clear task boundaries or output format specifications

**Best Practice (A2A Protocol):**
- Agents exchange JSON objects with sender, recipient, task intent, payload
- Schema-compliant payloads prevent hallucination
- Explicit task boundaries prevent work duplication

**Gaps Identified:**
1. No A2A message protocol
2. No schema validation for agent inputs/outputs
3. No task specification with explicit boundaries
4. No handoff protocol for multi-agent collaboration

### 3. Memory Layer Gaps

**Current (`src/memory/manager.py`):**
- Redis cache working
- ATP schema defined but `oracledb` async not tested in production
- No vector memory for semantic search
- No checkpoint persistence to ATP

**Best Practice (OCI AI Agent Platform):**
- RAG Tool for knowledge base retrieval
- SQL Tool for structured data queries
- Persistent checkpoints for error recovery

**Gaps Identified:**
1. ATP connection not validated in production path
2. No vector embeddings for conversation context
3. Checkpoints don't persist to ATP (in-memory only)
4. No conversation summarization for long threads

### 4. Slack Integration Gaps

**Current (`src/channels/slack.py`):**
- `asyncio.run()` creates new event loop per handler
- MCP connections reset on each message (expensive)
- No message queuing for rate limiting
- No conversation threading with memory

**Best Practice:**
- Single event loop for all async operations
- Connection pooling for external services
- Rate limiting and backpressure handling
- Thread context maintained across messages

**Gaps Identified:**
1. Event loop recreation causes MCP reconnections
2. No connection pooling strategy
3. Rate limiting not implemented
4. Thread history not leveraged for context

### 5. Error Recovery Gaps

**Current:**
- Basic try/except with error return
- No checkpoint save before risky operations
- No retry with exponential backoff
- No graceful degradation to simpler tools

**Best Practice (Anthropic):**
- Systems must resume from checkpoints rather than restarting
- Inform agents when tools fail, allowing adaptive recovery
- Combine model intelligence with deterministic safeguards

**Gaps Identified:**
1. No checkpoint save/restore mechanism
2. Tool failures cause immediate error return
3. No fallback tool tiers
4. No agent self-correction on failure

---

## Enhancement Roadmap

### Phase 4A: Core Infrastructure (Week 1-2)

#### 4A.1 Persistent Checkpointing to OCI ATP

**Implementation:**
```python
# src/memory/checkpointer.py
class ATPCheckpointer(BaseCheckpointSaver):
    """Persist LangGraph checkpoints to OCI ATP."""

    async def put(self, config: dict, checkpoint: Checkpoint, metadata: dict) -> None:
        """Save checkpoint to ATP."""
        await self.atp_store.set(
            key=f"checkpoint:{config['thread_id']}:{checkpoint.id}",
            value={
                "checkpoint": checkpoint.to_dict(),
                "metadata": metadata,
            }
        )

    async def get(self, config: dict) -> Checkpoint | None:
        """Restore checkpoint from ATP."""
        # Get latest checkpoint for thread
        ...
```

**Tasks:**
1. [ ] Create `ATPCheckpointer` implementing `BaseCheckpointSaver`
2. [ ] Add checkpoint table schema to ATP
3. [ ] Integrate with `LangGraphCoordinator._checkpointer`
4. [ ] Add checkpoint cleanup/retention policy
5. [ ] Write tests for checkpoint persistence

**Trade-off:** Checkpoint writes add latency (~50-100ms per save). Use configurable checkpoint frequency (every N iterations or on specific nodes).

#### 4A.2 A2A Message Protocol

**Implementation:**
```python
# src/agents/protocol.py
@dataclass
class AgentMessage:
    """Structured message between agents (A2A protocol)."""
    sender: str           # Sending agent ID
    recipient: str        # Target agent ID
    task_id: str          # Unique task identifier
    intent: str           # What needs to be done
    payload: dict         # Schema-validated input
    output_format: str    # Expected response format
    boundaries: list[str] # Explicit task boundaries
    context: dict         # Shared context

    def validate(self, schema: dict) -> bool:
        """Validate payload against schema."""
        ...
```

**Tasks:**
1. [ ] Define `AgentMessage` dataclass with validation
2. [ ] Create message schemas for each agent type
3. [ ] Update `CoordinatorNodes.agent_node()` to use protocol
4. [ ] Add message serialization for ATP persistence
5. [ ] Implement message routing in coordinator

#### 4A.3 Fix Slack Event Loop Architecture

**Current Problem:**
```python
# Current - creates new event loop each time
@app.event("message")
def handle_message(event, say, client):
    asyncio.run(self._process_message(event, say, client))  # BAD
```

**Solution:**
```python
# src/channels/slack.py
class SlackHandler:
    def __init__(self, ...):
        self._loop = asyncio.new_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _run_async(self, coro):
        """Run coroutine in shared event loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def start(self, ...):
        # Start event loop in background thread
        self._loop_thread = Thread(target=self._loop.run_forever)
        self._loop_thread.start()

        # Start Slack handler (it will use callbacks)
        ...
```

**Tasks:**
1. [ ] Create dedicated async event loop for Slack handler
2. [ ] Use `run_coroutine_threadsafe()` instead of `asyncio.run()`
3. [ ] Initialize MCP connections once at startup
4. [ ] Add connection health checks and reconnection
5. [ ] Test with concurrent Slack messages

---

### Phase 4B: Agent Orchestration (Week 2-3)

#### 4B.1 Parallel Subagent Execution

**Implementation:**
```python
# src/agents/coordinator/orchestrator.py
class ParallelOrchestrator:
    """Orchestrates parallel subagent execution."""

    async def execute_parallel(
        self,
        tasks: list[AgentMessage],
        max_concurrent: int = 5,
    ) -> list[AgentResult]:
        """Execute multiple agent tasks in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_execute(task: AgentMessage):
            async with semaphore:
                return await self._execute_agent(task)

        return await asyncio.gather(
            *[bounded_execute(t) for t in tasks],
            return_exceptions=True,
        )
```

**Tasks:**
1. [ ] Create `ParallelOrchestrator` class
2. [ ] Add complexity assessment for task decomposition
3. [ ] Implement result aggregation from parallel agents
4. [ ] Add timeout handling per-agent
5. [ ] Update coordinator graph to use parallel execution

#### 4B.2 Dynamic Task Decomposition

**Implementation:**
```python
# src/agents/coordinator/nodes.py (enhanced classifier)
async def classifier_node(self, state: CoordinatorState) -> dict:
    """Classify intent and determine complexity."""

    # Existing classification...
    intent = self._parse_classification(response.content, state.query)

    # NEW: Assess complexity
    complexity = await self._assess_complexity(state.query, intent)

    # Decompose complex tasks
    if complexity.score > 0.7 and complexity.decomposable:
        subtasks = await self._decompose_task(state.query, intent)
        return {
            "intent": intent,
            "complexity": complexity,
            "subtasks": subtasks,  # List of parallel tasks
        }

    return {"intent": intent}
```

**Tasks:**
1. [ ] Add complexity scoring (domains involved, entities, tool count)
2. [ ] Implement task decomposition prompt
3. [ ] Define decomposition heuristics (when to parallelize)
4. [ ] Add subtask coordination in router node
5. [ ] Implement result synthesis from subtasks

---

### Phase 4C: Memory & Context (Week 3-4)

#### 4C.1 Full ATP Integration

**Implementation:**
```sql
-- ATP Schema additions
CREATE TABLE conversation_summary (
    thread_id VARCHAR2(64) PRIMARY KEY,
    summary CLOB,  -- Compressed conversation summary
    key_entities JSON,
    last_updated TIMESTAMP DEFAULT SYSTIMESTAMP
);

CREATE TABLE agent_checkpoints (
    checkpoint_id VARCHAR2(64) PRIMARY KEY,
    thread_id VARCHAR2(64),
    agent_id VARCHAR2(64),
    state_json CLOB,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT fk_thread FOREIGN KEY (thread_id) REFERENCES conversation_history(thread_id)
);
```

**Tasks:**
1. [ ] Create ATP DDL migration script
2. [ ] Test `oracledb` async with ATP wallet
3. [ ] Implement conversation summarization
4. [ ] Add summary retrieval for long threads
5. [ ] Create checkpoint retention job (OCI Functions)

#### 4C.2 Context Compression

**Implementation:**
```python
# src/memory/context.py
class ContextManager:
    """Manages conversation context with compression."""

    MAX_TOKENS = 150_000  # Save before hitting 200k limit

    async def get_context(self, thread_id: str) -> str:
        """Get compressed context for thread."""
        # Check if recent messages fit
        recent = await self.memory.get_recent_messages(thread_id, limit=20)
        token_count = self._count_tokens(recent)

        if token_count < self.MAX_TOKENS:
            return self._format_messages(recent)

        # Need compression - retrieve summary
        summary = await self._get_or_create_summary(thread_id)
        return f"CONVERSATION SUMMARY:\n{summary}\n\nRECENT MESSAGES:\n{self._format_messages(recent[-5:])}"
```

**Tasks:**
1. [ ] Implement token counting (tiktoken or estimate)
2. [ ] Create conversation summarization with LLM
3. [ ] Add summary caching in Redis
4. [ ] Implement progressive summarization
5. [ ] Test with 100+ message threads

---

### Phase 4D: Error Recovery & Resilience (Week 4)

#### 4D.1 Tool Failure Handling

**Implementation:**
```python
# src/mcp/resilient_executor.py
class ResilientToolExecutor:
    """Execute tools with retry, fallback, and recovery."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute with resilience patterns."""
        tier = self.catalog.get_tool_tier(tool_call.name)

        for attempt in range(self.max_retries):
            try:
                result = await self._execute_with_timeout(
                    tool_call,
                    timeout=self.tier_timeouts[tier]
                )
                return ToolResult(success=True, result=result)

            except TimeoutError:
                if fallback := self._get_fallback_tool(tool_call.name):
                    # Try fallback tool (e.g., cache tier instead of API tier)
                    return await self.execute(fallback)

            except ToolError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

                # Inform agent about failure for adaptive recovery
                return ToolResult(
                    success=False,
                    error=str(e),
                    recovery_hint=self._get_recovery_hint(tool_call, e),
                )
```

**Tasks:**
1. [ ] Create `ResilientToolExecutor` with retry logic
2. [ ] Define fallback chains per tool tier
3. [ ] Add recovery hints to ToolResult
4. [ ] Update action node to handle recovery hints
5. [ ] Implement circuit breaker for flaky tools

#### 4D.2 Graceful Degradation

**Implementation:**
```python
# Degradation ladder
DEGRADATION_LADDER = [
    "full_mcp",      # All MCP tools available
    "cache_only",    # Only tier 1 cache tools
    "llm_only",      # No tools, LLM-only responses
    "static",        # Pre-canned responses
]

async def _invoke_with_degradation(self, state):
    """Try full capability, degrade on failures."""
    for level in DEGRADATION_LADDER:
        try:
            return await self._execute_at_level(state, level)
        except Exception as e:
            logger.warning(f"Degrading from {level}", error=str(e))
            continue

    return {"error": "All capability levels exhausted"}
```

**Tasks:**
1. [ ] Define degradation levels and capabilities
2. [ ] Implement level-aware execution
3. [ ] Add degradation metrics to APM
4. [ ] Create recovery logic (re-escalate when healthy)
5. [ ] Test degradation scenarios

---

## Implementation Priority Matrix

| Enhancement | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| Fix Slack event loop | High | Medium | P0 |
| ATP checkpoint persistence | High | Medium | P0 |
| A2A message protocol | High | High | P1 |
| Parallel agent execution | High | High | P1 |
| Tool resilience | Medium | Medium | P2 |
| Context compression | Medium | Low | P2 |
| Graceful degradation | Medium | High | P3 |

---

## OCI-Specific Best Practices to Adopt

Based on OCI AI Agent Platform first principles:

### 1. RAG Tool Pattern
```python
# Use for knowledge base queries
rag_tool = RAGTool(
    source=OCIObjectStorage(bucket="agent-knowledge"),
    embedding_model="cohere.embed-english-v3.0",
    vector_db="oci-opensearch",
)
```

### 2. SQL Tool Pattern
```python
# Use for structured ATP queries
sql_tool = SQLTool(
    connection=ATP(wallet_path="...", service_name="..._high"),
    allowed_tables=["agent_memory", "conversation_history"],
    read_only=True,  # Safety
)
```

### 3. Custom Tool Pattern
```python
# Use OCI Functions for specialized logic
custom_tool = OCIFunctionTool(
    function_ocid="ocid1.fnfunc...",
    timeout_seconds=30,
)
```

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Parallel agent latency | N/A (sequential) | -60% vs sequential | APM traces |
| Message persistence | 0% (in-memory) | 100% to ATP | ATP audit |
| Slack message handling | New loop/msg | Single loop | Memory profiling |
| Tool success rate | ~85% | >95% with retries | APM metrics |
| Long conversation support | ~20 messages | 100+ | Functional test |

---

## Next Steps

1. **Immediate (This Week):**
   - Fix Slack event loop architecture (P0)
   - Test ATP connection with production wallet
   - Create `ATPCheckpointer` class

2. **Next Sprint:**
   - Implement A2A message protocol
   - Add parallel orchestrator
   - Deploy ATP schema migrations

3. **Following Sprint:**
   - Context compression
   - Tool resilience patterns
   - End-to-end evaluation suite
