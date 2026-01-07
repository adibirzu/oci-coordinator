"""
LangGraph Coordinator for OCI AI Agents.

Implements the workflow-first orchestration pattern with enhancements:
- 70%+ requests → deterministic workflows
- Complex multi-domain requests → parallel multi-agent execution
- Remaining → agentic LLM reasoning with specialized agents

Graph Structure:
    input → classifier → router → [workflow|parallel|agent] → (action →)* → output

Phase 4 Enhancements:
- In-memory checkpointing for fault tolerance
- Parallel orchestrator for complex cross-domain queries
- Context compression for long conversations
- A2A protocol for structured agent communication
- Dynamic tool registration and usage tracking
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.coordinator.nodes import (
    CoordinatorNodes,
    should_continue_after_agent,
    should_continue_after_router,
    should_loop_from_action,
)
from src.agents.coordinator.state import (
    CoordinatorState,
    reset_thinking_callback,
    set_thinking_callback,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.agents.catalog import AgentCatalog
    from src.agents.coordinator.orchestrator import ParallelOrchestrator
    from src.mcp.catalog import ToolCatalog
    from src.memory.context import ContextManager
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


class NodeName(str, Enum):
    """Graph node names."""

    INPUT = "input"
    CLASSIFIER = "classifier"
    ROUTER = "router"
    WORKFLOW = "workflow"
    PARALLEL = "parallel"  # NEW: Parallel multi-agent execution
    AGENT = "agent"
    ACTION = "action"
    OUTPUT = "output"


class LangGraphCoordinator:
    """
    LangGraph-based Coordinator for OCI AI Agent orchestration.

    Implements a workflow-first design:
    - High-confidence requests route to deterministic workflows
    - Medium-confidence requests delegate to specialized agents
    - Low-confidence requests use agentic LLM reasoning or escalate

    Graph Flow:
        input → classifier → router →
            ├── workflow → output
            ├── agent → (action →)* → output
            └── output (escalate/direct)

    Usage:
        coordinator = LangGraphCoordinator(
            llm=llm,
            tool_catalog=tool_catalog,
            agent_catalog=agent_catalog,
            memory=memory,
        )
        await coordinator.initialize()
        result = await coordinator.invoke("List all running instances")
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tool_catalog: ToolCatalog,
        agent_catalog: AgentCatalog,
        memory: SharedMemoryManager,
        workflow_registry: dict[str, Any] | None = None,
        max_iterations: int = 15,
        checkpointer: BaseCheckpointSaver | None = None,
        enable_parallel: bool = True,
        enable_context_compression: bool = True,
    ):
        """
        Initialize the coordinator.

        Args:
            llm: LangChain chat model for reasoning
            tool_catalog: Catalog of MCP tools
            agent_catalog: Catalog of specialized agents
            memory: Shared memory manager
            workflow_registry: Map of workflow names to workflow functions
            max_iterations: Maximum tool calling iterations
            checkpointer: Custom checkpointer (defaults to MemorySaver)
            enable_parallel: Enable parallel orchestrator for complex queries
            enable_context_compression: Enable context compression for long conversations
        """
        self.llm = llm
        self.tool_catalog = tool_catalog
        self.agent_catalog = agent_catalog
        self.memory = memory
        self.workflow_registry = workflow_registry or {}
        self.max_iterations = max_iterations
        self.enable_parallel = enable_parallel
        self.enable_context_compression = enable_context_compression

        self.tool_catalog.set_memory_manager(memory)

        # Checkpointer: use provided or default to MemorySaver
        self._checkpointer = checkpointer or MemorySaver()

        # Context manager for long conversations
        self._context_manager: ContextManager | None = None

        # Parallel orchestrator for complex queries
        self._orchestrator: ParallelOrchestrator | None = None

        self._nodes = CoordinatorNodes(
            llm=llm,
            tool_catalog=tool_catalog,
            agent_catalog=agent_catalog,
            memory=memory,
            workflow_registry=workflow_registry,
            orchestrator=None,  # Set after orchestrator init
        )
        self._graph: StateGraph | None = None
        self._compiled_graph: Any = None
        self._logger = logger.bind(component="LangGraphCoordinator")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph.

        Creates the coordinator graph with:
        - Input processing
        - Intent classification
        - Workflow-first routing
        - Workflow execution
        - Agent delegation
        - Tool execution
        - Output preparation

        Returns:
            Compiled StateGraph
        """
        self._logger.info("Building coordinator graph")

        graph = StateGraph(CoordinatorState)

        # ─────────────────────────────────────────────────────────────────────
        # Add Nodes
        # ─────────────────────────────────────────────────────────────────────

        graph.add_node(NodeName.INPUT.value, self._nodes.input_node)
        graph.add_node(NodeName.CLASSIFIER.value, self._nodes.classifier_node)
        graph.add_node(NodeName.ROUTER.value, self._nodes.router_node)
        graph.add_node(NodeName.WORKFLOW.value, self._nodes.workflow_node)
        graph.add_node(NodeName.PARALLEL.value, self._nodes.parallel_node)
        graph.add_node(NodeName.AGENT.value, self._nodes.agent_node)
        graph.add_node(NodeName.ACTION.value, self._nodes.action_node)
        graph.add_node(NodeName.OUTPUT.value, self._nodes.output_node)

        # ─────────────────────────────────────────────────────────────────────
        # Set Entry Point
        # ─────────────────────────────────────────────────────────────────────

        graph.set_entry_point(NodeName.INPUT.value)

        # ─────────────────────────────────────────────────────────────────────
        # Add Edges
        # ─────────────────────────────────────────────────────────────────────

        # Input → Classifier (always)
        graph.add_edge(NodeName.INPUT.value, NodeName.CLASSIFIER.value)

        # Classifier → Router (always)
        graph.add_edge(NodeName.CLASSIFIER.value, NodeName.ROUTER.value)

        # Router → Conditional (workflow/parallel/agent/output)
        graph.add_conditional_edges(
            NodeName.ROUTER.value,
            should_continue_after_router,
            {
                "workflow": NodeName.WORKFLOW.value,
                "parallel": NodeName.PARALLEL.value,
                "agent": NodeName.AGENT.value,
                "output": NodeName.OUTPUT.value,
            },
        )

        # Workflow → Output (always, workflows are terminal)
        graph.add_edge(NodeName.WORKFLOW.value, NodeName.OUTPUT.value)

        # Parallel → Output (parallel execution synthesizes and returns)
        graph.add_edge(NodeName.PARALLEL.value, NodeName.OUTPUT.value)

        # Agent → Conditional (action/output)
        graph.add_conditional_edges(
            NodeName.AGENT.value,
            should_continue_after_agent,
            {
                "action": NodeName.ACTION.value,
                "output": NodeName.OUTPUT.value,
            },
        )

        # Action → Conditional (agent/output)
        graph.add_conditional_edges(
            NodeName.ACTION.value,
            should_loop_from_action,
            {
                "agent": NodeName.AGENT.value,
                "output": NodeName.OUTPUT.value,
            },
        )

        # Output → END
        graph.add_edge(NodeName.OUTPUT.value, END)

        self._logger.info("Coordinator graph built successfully")
        return graph

    async def initialize(self) -> None:
        """
        Initialize the coordinator.

        Builds the graph and optionally binds tools to the LLM.
        Sets up context manager and parallel orchestrator.
        Call this before using invoke().
        """
        # Initialize context manager for long conversations
        if self.enable_context_compression:
            await self._init_context_manager()

        # Initialize parallel orchestrator
        if self.enable_parallel:
            await self._init_parallel_orchestrator()

        # Build and compile the graph
        self._graph = self._build_graph()
        self._compiled_graph = self._graph.compile(checkpointer=self._checkpointer)

        # Refresh tool catalog if available
        if self.tool_catalog:
            await self.tool_catalog.refresh()
            self._bind_tools()

        # Discover agents
        if self.agent_catalog:
            self.agent_catalog.auto_discover()

        self._logger.info(
            "Coordinator initialized",
            tool_count=len(self.tool_catalog._tools) if self.tool_catalog else 0,
            agent_count=len(self.agent_catalog.list_all()) if self.agent_catalog else 0,
            workflow_count=len(self.workflow_registry),
            checkpointer_type=type(self._checkpointer).__name__,
            parallel_enabled=self._orchestrator is not None,
            context_compression=self._context_manager is not None,
        )

    async def _init_context_manager(self) -> None:
        """Initialize context manager for conversation compression."""
        try:
            from src.memory.context import ContextManager

            self._context_manager = ContextManager(
                memory=self.memory,
                llm=self.llm,
            )
            self._logger.info("Context manager initialized")

        except Exception as e:
            self._logger.warning(
                "Context manager init failed",
                error=str(e),
            )

    async def _init_parallel_orchestrator(self) -> None:
        """Initialize parallel orchestrator for complex queries."""
        try:
            from src.agents.coordinator.orchestrator import ParallelOrchestrator

            self._orchestrator = ParallelOrchestrator(
                agent_catalog=self.agent_catalog,
                tool_catalog=self.tool_catalog,
                llm=self.llm,
                memory=self.memory,
            )
            # Connect orchestrator to nodes for parallel execution
            self._nodes.orchestrator = self._orchestrator
            self._logger.info("Parallel orchestrator initialized and connected to nodes")

        except Exception as e:
            self._logger.warning(
                "Parallel orchestrator init failed",
                error=str(e),
            )

    def _bind_tools(self) -> None:
        """Bind MCP tools to the LLM if supported.

        Not all LLM providers support tool binding (e.g., OCA).
        This method gracefully skips binding if not supported.
        Tools are still available via direct MCP execution.
        """
        if not self.tool_catalog or not self.llm:
            return

        llm_type = type(self.llm).__name__

        # Skip tool binding for LLMs that don't support function calling
        # ChatOCA inherits bind_tools from BaseChatModel but doesn't support it
        unsupported_llms = {"ChatOCA", "ChatLiteLLM"}
        if llm_type in unsupported_llms:
            self._logger.info(
                "LLM does not support tool binding, tools available via MCP",
                llm_type=llm_type,
            )
            return

        # Check if LLM supports bind_tools
        if not hasattr(self.llm, "bind_tools"):
            self._logger.info(
                "LLM does not support bind_tools, tools available via MCP",
                llm_type=llm_type,
            )
            return

        try:
            from src.mcp.tools.converter import ToolConverter

            converter = ToolConverter(self.tool_catalog)
            tools = converter.to_langchain_tools()

            if tools:
                self.llm = self.llm.bind_tools(tools)
                self._nodes.llm = self.llm
                self._logger.info("Tools bound to LLM", count=len(tools))

        except ImportError:
            self._logger.warning("ToolConverter not available, skipping tool binding")
        except (AttributeError, NotImplementedError) as e:
            # LLM doesn't support native tool binding (common for OCA/custom LLMs)
            # This is expected - tools are available via MCP instead
            self._logger.info(
                "LLM does not support native tool binding, tools available via MCP",
                llm_type=type(self.llm).__name__,
                error_type=type(e).__name__,
            )
        except Exception as e:
            import traceback
            self._logger.error(
                "Failed to bind tools",
                error=str(e) or type(e).__name__,
                traceback=traceback.format_exc(),
            )

    async def invoke(
        self,
        query: str,
        thread_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        force_parallel: bool = False,
        on_thinking_update: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process a query through the coordinator graph.

        Args:
            query: User query
            thread_id: Optional thread ID for conversation continuity
            session_id: Optional session identifier
            user_id: Optional user identifier
            force_parallel: Force parallel execution for complex queries
            on_thinking_update: Optional callback for real-time thinking updates.
                Signature: async def callback(step: ThinkingStep) -> None
            metadata: Optional metadata dict (e.g., oci_profile, profile_context)

        Returns:
            Result dictionary with:
            - success: bool
            - response: str
            - routing_type: str (workflow/agent/parallel/direct/escalate)
            - iterations: int
            - error: str or None
            - thinking_trace: ThinkingTrace object
            - thinking_summary: str (compact summary)
        """
        import uuid

        if not self._compiled_graph:
            await self.initialize()

        # Create unique thread_id per query to avoid stale checkpoints
        if thread_id:
            effective_thread_id = thread_id
        else:
            effective_thread_id = str(uuid.uuid4())

        # Check for context compression if enabled
        context_str = ""
        if self._context_manager and thread_id:
            try:
                context_window = await self._context_manager.get_context(thread_id)
                if context_window.is_compressed:
                    context_str = self._context_manager.format_context_for_prompt(
                        context_window
                    )
                    self._logger.info(
                        "Using compressed context",
                        thread_id=thread_id,
                        total_messages=context_window.total_messages,
                        estimated_tokens=context_window.estimated_tokens,
                    )
            except Exception as e:
                self._logger.warning("Context compression failed", error=str(e))

        # Check if parallel execution should be used
        if force_parallel and self._orchestrator:
            return await self._invoke_parallel(
                query, effective_thread_id, session_id, context_str
            )

        # Create initial state with context and thinking callback
        full_query = f"{context_str}\n\n{query}" if context_str else query

        # Extract OCI profile from metadata for profile-aware operations
        oci_profile = "DEFAULT"
        if metadata:
            oci_profile = metadata.get("oci_profile", "DEFAULT")
            self._logger.debug("Using OCI profile", profile=oci_profile)

        initial_state = CoordinatorState(
            messages=[HumanMessage(content=full_query)],
            max_iterations=self.max_iterations,
            metadata=metadata or {},
            # Explicitly reset transient state for new turn
            final_response=None,
            error=None,
            tool_calls=[],
            tool_results=[],
            iteration=0,
            routing=None,
            current_agent=None,
            workflow_name=None,
            agent_context=None,
        )

        # Thread config for checkpointer
        config = {"configurable": {"thread_id": effective_thread_id}}

        self._logger.info(
            "Processing query",
            query_length=len(query),
            thread_id=effective_thread_id,
            has_context=bool(context_str),
        )

        # Set thinking callback in context variable (outside of serialized state)
        # This avoids msgpack serialization issues with LangGraph's MemorySaver
        callback_token = set_thinking_callback(on_thinking_update)
        try:
            result = await self._compiled_graph.ainvoke(initial_state, config)

            # Extract response from graph result
            final_response = result.get("final_response", "")
            if not final_response and result.get("messages"):
                # Fallback: get content from last AI message
                from langchain_core.messages import AIMessage
                for msg in reversed(result.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_response = str(msg.content)
                        break

            # Return successful result
            # Extract routing type and target from the routing decision object
            routing = result.get("routing")
            routing_type = "unknown"
            routing_target = None
            if routing:
                if hasattr(routing, "routing_type"):
                    routing_type = routing.routing_type.value if hasattr(routing.routing_type, "value") else str(routing.routing_type)
                    routing_target = getattr(routing, "target", None)
                elif isinstance(routing, dict):
                    routing_type = routing.get("routing_type", "unknown")
                    routing_target = routing.get("target")

            # Get workflow/agent name with fallbacks
            selected_workflow = result.get("workflow_name") or (routing_target if routing_type == "workflow" else None)
            selected_agent = result.get("current_agent") or (routing_target if routing_type == "agent" else None)

            # Get thinking trace for transparency
            thinking_trace = result.get("thinking_trace")
            thinking_summary = None
            if thinking_trace and hasattr(thinking_trace, "to_compact_summary"):
                thinking_summary = thinking_trace.to_compact_summary()

            return {
                "success": bool(final_response),
                "response": final_response or "No response generated",
                "routing_type": routing_type,
                "iterations": result.get("iteration", 0),
                "error": result.get("error"),
                "thread_id": effective_thread_id,
                "intent": result.get("intent"),
                "selected_agent": selected_agent,
                "selected_workflow": selected_workflow,
                "thinking_trace": thinking_trace,
                "thinking_summary": thinking_summary,
                "agent_candidates": result.get("agent_candidates", []),
            }

        except (TypeError, ValueError) as serde_error:
            # Handle serialization/deserialization errors from checkpointer
            # This can happen with MemorySaver when state schema changes
            error_msg = str(serde_error)
            self._logger.warning(
                "Checkpointer serialization error, retrying with fresh state",
                error=error_msg,
            )
            # Create a fresh thread_id to avoid stale checkpoint data
            import uuid
            fresh_thread_id = str(uuid.uuid4())
            fresh_config = {"configurable": {"thread_id": fresh_thread_id}}
            try:
                result = await self._compiled_graph.ainvoke(initial_state, fresh_config)
                effective_thread_id = fresh_thread_id  # Update to fresh thread

                # Extract response from graph result (same as success path)
                final_response = result.get("final_response", "")
                if not final_response and result.get("messages"):
                    from langchain_core.messages import AIMessage
                    for msg in reversed(result.get("messages", [])):
                        if isinstance(msg, AIMessage) and msg.content:
                            final_response = str(msg.content)
                            break

                # Extract routing type and target from the routing decision object
                routing = result.get("routing")
                routing_type = "unknown"
                routing_target = None
                if routing:
                    if hasattr(routing, "routing_type"):
                        routing_type = routing.routing_type.value if hasattr(routing.routing_type, "value") else str(routing.routing_type)
                        routing_target = getattr(routing, "target", None)
                    elif isinstance(routing, dict):
                        routing_type = routing.get("routing_type", "unknown")
                        routing_target = routing.get("target")

                # Get workflow/agent name with fallbacks
                selected_workflow = result.get("workflow_name") or (routing_target if routing_type == "workflow" else None)
                selected_agent = result.get("current_agent") or (routing_target if routing_type == "agent" else None)

                # Get thinking trace for transparency
                thinking_trace = result.get("thinking_trace")
                thinking_summary = None
                if thinking_trace and hasattr(thinking_trace, "to_compact_summary"):
                    thinking_summary = thinking_trace.to_compact_summary()

                return {
                    "success": bool(final_response),
                    "response": final_response or "No response generated",
                    "routing_type": routing_type,
                    "iterations": result.get("iteration", 0),
                    "error": result.get("error"),
                    "thread_id": effective_thread_id,
                    "intent": result.get("intent"),
                    "selected_agent": selected_agent,
                    "selected_workflow": selected_workflow,
                    "thinking_trace": thinking_trace,
                    "thinking_summary": thinking_summary,
                    "agent_candidates": result.get("agent_candidates", []),
                }

            except Exception as retry_error:
                self._logger.error("Retry also failed", error=str(retry_error))
                return {
                    "success": False,
                    "response": f"Error processing request: {retry_error}",
                    "routing_type": "error",
                    "iterations": 0,
                    "error": str(retry_error),
                    "thread_id": effective_thread_id,
                }

        except Exception as e:
            self._logger.error("Query processing failed", error=str(e))
            return {
                "success": False,
                "response": f"Error processing request: {e}",
                "routing_type": "error",
                "iterations": 0,
                "error": str(e),
                "thread_id": effective_thread_id,
            }
        finally:
            # Always reset the thinking callback context variable
            reset_thinking_callback(callback_token)

    async def _invoke_parallel(
        self,
        query: str,
        thread_id: str,
        session_id: str | None,
        context_str: str,
    ) -> dict[str, Any]:
        """
        Execute query using parallel orchestrator.

        Used for complex cross-domain queries that benefit from
        multiple agents working in parallel.
        """
        if not self._orchestrator:
            return {
                "success": False,
                "response": "Parallel orchestrator not available",
                "routing_type": "error",
                "iterations": 0,
                "error": "Orchestrator not initialized",
                "thread_id": thread_id,
            }

        self._logger.info(
            "Using parallel orchestrator",
            query_length=len(query),
            thread_id=thread_id,
        )

        try:
            result = await self._orchestrator.execute(
                query=query,
                context={"thread_id": thread_id, "history": context_str},
            )

            # Save to memory if thread_id provided (for conversation continuity)
            if thread_id and self.memory:
                await self.memory.append_conversation(
                    thread_id,
                    {"role": "user", "content": query},
                )
                await self.memory.append_conversation(
                    thread_id,
                    {"role": "assistant", "content": result.response},
                )

            return {
                "success": result.success,
                "response": result.response,
                "routing_type": "parallel",
                "iterations": len(result.agent_results),
                "error": result.error,
                "thread_id": thread_id,
                "agents_used": result.agents_used,
                "execution_time_ms": result.execution_time_ms,
            }

        except Exception as e:
            self._logger.error("Parallel execution failed", error=str(e))
            return {
                "success": False,
                "response": f"Parallel execution failed: {e}",
                "routing_type": "error",
                "iterations": 0,
                "error": str(e),
                "thread_id": thread_id,
            }

    async def invoke_stream(
        self,
        query: str,
        thread_id: str | None = None,
    ):
        """
        Stream the coordinator's processing.

        Yields state updates as the graph executes.

        Args:
            query: User query
            thread_id: Optional thread ID

        Yields:
            State updates from each node
        """
        import hashlib
        import uuid

        if not self._compiled_graph:
            await self.initialize()

        initial_state = CoordinatorState(
            messages=[HumanMessage(content=query)],
            max_iterations=self.max_iterations,
        )

        # Create unique thread_id per query to avoid stale checkpoints
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        if thread_id:
            effective_thread_id = f"{thread_id}_{query_hash}"
        else:
            effective_thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": effective_thread_id}}

        async for event in self._compiled_graph.astream(initial_state, config):
            yield event

    def register_workflow(
        self,
        name: str,
        workflow_fn: Any,
    ) -> None:
        """
        Register a deterministic workflow.

        Args:
            name: Workflow name (used in routing)
            workflow_fn: Async function that executes the workflow
                Signature: async def workflow(query, entities, tool_catalog, memory) -> str
        """
        self.workflow_registry[name] = workflow_fn
        self._nodes.workflow_registry = self.workflow_registry
        self._logger.info("Workflow registered", name=name)

    def get_graph(self) -> StateGraph | None:
        """Get the compiled graph for inspection."""
        return self._compiled_graph

    def get_graph_diagram(self) -> str:
        """
        Get a Mermaid diagram of the graph.

        Returns:
            Mermaid diagram string
        """
        if not self._compiled_graph:
            return "Graph not initialized"

        try:
            return self._compiled_graph.get_graph().draw_mermaid()
        except Exception:
            return "Diagram generation not available"


# ─────────────────────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────────────────────


async def create_coordinator(
    llm: BaseChatModel | None = None,
    redis_url: str = "redis://localhost:6379",
    max_iterations: int = 15,
    include_workflows: bool = True,
) -> LangGraphCoordinator:
    """
    Create and initialize a LangGraph Coordinator.

    Factory function that sets up all required components.

    Args:
        llm: LangChain chat model (creates default if not provided)
        redis_url: Redis URL for caching
        max_iterations: Maximum tool iterations
        include_workflows: Include pre-built deterministic workflows

    Returns:
        Initialized LangGraphCoordinator
    """
    from src.agents.catalog import AgentCatalog
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

    # Create LLM if not provided
    if llm is None:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    # Create components
    memory = SharedMemoryManager(redis_url=redis_url)

    tool_catalog = ToolCatalog(memory_manager=memory)
    agent_catalog = AgentCatalog.get_instance()

    # Load pre-built workflows if enabled
    workflow_registry = None
    if include_workflows:
        from src.agents.coordinator.workflows import get_workflow_registry

        workflow_registry = get_workflow_registry()
        logger.info("Loaded pre-built workflows", count=len(workflow_registry))

    # Create coordinator
    coordinator = LangGraphCoordinator(
        llm=llm,
        tool_catalog=tool_catalog,
        agent_catalog=agent_catalog,
        memory=memory,
        workflow_registry=workflow_registry,
        max_iterations=max_iterations,
    )

    # Initialize
    await coordinator.initialize()

    return coordinator
