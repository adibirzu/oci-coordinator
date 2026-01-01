"""
LangGraph Coordinator for OCI AI Agents.

Implements the workflow-first orchestration pattern with enhancements:
- 70%+ requests → deterministic workflows
- Complex multi-domain requests → parallel multi-agent execution
- Remaining → agentic LLM reasoning with specialized agents

Graph Structure:
    input → classifier → router → [workflow|parallel|agent] → (action →)* → output

Phase 4 Enhancements:
- ATP-persistent checkpointing for fault tolerance
- Parallel orchestrator for complex cross-domain queries
- Context compression for long conversations
- A2A protocol for structured agent communication
- Dynamic tool registration and usage tracking
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Any

import os

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
from src.agents.coordinator.state import CoordinatorState

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
            checkpointer: Custom checkpointer (uses ATP if env configured, else MemorySaver)
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

        # Checkpointer: use provided, or ATP if configured, else MemorySaver
        self._checkpointer = checkpointer or MemorySaver()
        self._use_atp_checkpointer = checkpointer is None and os.getenv("ATP_TNS_NAME")

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

        # Router → Conditional (workflow/agent/output)
        graph.add_conditional_edges(
            NodeName.ROUTER.value,
            should_continue_after_router,
            {
                "workflow": NodeName.WORKFLOW.value,
                "agent": NodeName.AGENT.value,
                "output": NodeName.OUTPUT.value,
            },
        )

        # Workflow → Output (always, workflows are terminal)
        graph.add_edge(NodeName.WORKFLOW.value, NodeName.OUTPUT.value)

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
        Sets up ATP checkpointer, context manager, and parallel orchestrator.
        Call this before using invoke().
        """
        # Initialize ATP checkpointer if configured
        if self._use_atp_checkpointer:
            await self._init_atp_checkpointer()

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

    async def _init_atp_checkpointer(self) -> None:
        """Initialize ATP-backed checkpointer."""
        try:
            from src.memory.checkpointer import ATPCheckpointer

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._checkpointer = await ATPCheckpointer.create(redis_url=redis_url)
            self._logger.info("ATP checkpointer initialized")

        except Exception as e:
            self._logger.warning(
                "ATP checkpointer init failed, using MemorySaver",
                error=str(e),
            )
            self._checkpointer = MemorySaver()

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
            self._logger.info("Parallel orchestrator initialized")

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
        except AttributeError as e:
            # LLM doesn't support tool binding
            self._logger.info(
                "LLM does not support tool binding",
                llm_type=type(self.llm).__name__,
                error=str(e),
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
    ) -> dict[str, Any]:
        """
        Process a query through the coordinator graph.

        Args:
            query: User query
            thread_id: Optional thread ID for conversation continuity
            session_id: Optional session identifier
            user_id: Optional user identifier
            force_parallel: Force parallel execution for complex queries

        Returns:
            Result dictionary with:
            - success: bool
            - response: str
            - routing_type: str (workflow/agent/parallel/direct/escalate)
            - iterations: int
            - error: str or None
        """
        import uuid

        if not self._compiled_graph:
            await self.initialize()

        effective_thread_id = thread_id or str(uuid.uuid4())

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

        # Create initial state with context
        full_query = f"{context_str}\n\n{query}" if context_str else query
        initial_state = CoordinatorState(
            messages=[HumanMessage(content=full_query)],
            max_iterations=self.max_iterations,
        )

        # Thread config for checkpointer
        config = {"configurable": {"thread_id": effective_thread_id}}

        self._logger.info(
            "Processing query",
            query_length=len(query),
            thread_id=effective_thread_id,
            has_context=bool(context_str),
        )

        try:
            result = await self._compiled_graph.ainvoke(initial_state, config)

            # Extract results
            routing_type = "direct"
            if result.get("routing"):
                routing_type = result["routing"].routing_type.value

            response = result.get("final_response", "")
            error = result.get("error")

            # Save to memory if session provided
            if session_id and self.memory:
                await self.memory.append_conversation(
                    effective_thread_id,
                    {
                        "role": "user",
                        "content": query,
                    },
                )
                await self.memory.append_conversation(
                    effective_thread_id,
                    {
                        "role": "assistant",
                        "content": response,
                    },
                )

            self._logger.info(
                "Query processed",
                success=error is None,
                routing_type=routing_type,
                iterations=result.get("iteration", 0),
            )

            return {
                "success": error is None,
                "response": response,
                "routing_type": routing_type,
                "iterations": result.get("iteration", 0),
                "error": error,
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

            # Save to memory
            if session_id and self.memory:
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
        import uuid

        if not self._compiled_graph:
            await self.initialize()

        initial_state = CoordinatorState(
            messages=[HumanMessage(content=query)],
            max_iterations=self.max_iterations,
        )

        effective_thread_id = thread_id or str(uuid.uuid4())
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
    atp_connection: str | None = None,
    max_iterations: int = 15,
) -> LangGraphCoordinator:
    """
    Create and initialize a LangGraph Coordinator.

    Factory function that sets up all required components.

    Args:
        llm: LangChain chat model (creates default if not provided)
        redis_url: Redis URL for caching
        atp_connection: ATP connection string for persistence
        max_iterations: Maximum tool iterations

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
    memory = SharedMemoryManager(
        redis_url=redis_url,
        atp_connection=atp_connection or os.getenv("ATP_CONNECTION_STRING"),
    )

    tool_catalog = ToolCatalog()
    agent_catalog = AgentCatalog.get_instance()

    # Create coordinator
    coordinator = LangGraphCoordinator(
        llm=llm,
        tool_catalog=tool_catalog,
        agent_catalog=agent_catalog,
        memory=memory,
        max_iterations=max_iterations,
    )

    # Initialize
    await coordinator.initialize()

    return coordinator
