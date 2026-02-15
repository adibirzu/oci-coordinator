"""
Coordinator State schema for LangGraph orchestration.

This module defines the state that flows through the coordinator graph,
including intent classification, workflow routing, and agent context.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.agents.coordinator.transparency import (
    AgentCandidate,
    ThinkingStep,
    ThinkingTrace,
)

# ─────────────────────────────────────────────────────────────────────────────
# Context Variable for Thinking Callback
# ─────────────────────────────────────────────────────────────────────────────
# This context variable stores the thinking update callback OUTSIDE of the
# CoordinatorState to avoid msgpack serialization issues with LangGraph's
# MemorySaver checkpointer. Callables cannot be serialized by msgpack.

_thinking_callback: contextvars.ContextVar[Callable[[ThinkingStep], None] | None] = (
    contextvars.ContextVar("thinking_callback", default=None)
)


def set_thinking_callback(callback: Callable[[ThinkingStep], None] | None) -> contextvars.Token:
    """
    Set the thinking update callback in the current context.

    This should be called before invoking the LangGraph coordinator.
    The callback receives ThinkingStep updates for real-time UI updates (e.g., Slack).

    Args:
        callback: Async or sync callback function, or None to clear

    Returns:
        Token that can be used to reset the context variable

    Example:
        token = set_thinking_callback(my_callback)
        try:
            result = await graph.ainvoke(state, config)
        finally:
            reset_thinking_callback(token)
    """
    return _thinking_callback.set(callback)


def get_thinking_callback() -> Callable[[ThinkingStep], None] | None:
    """Get the thinking update callback from the current context."""
    return _thinking_callback.get()


def reset_thinking_callback(token: contextvars.Token) -> None:
    """Reset the thinking callback to its previous value using the token."""
    _thinking_callback.reset(token)


class RoutingType(str, Enum):
    """Type of routing decision."""

    WORKFLOW = "workflow"  # Deterministic workflow execution
    AGENT = "agent"  # Delegate to specialized agent
    PARALLEL = "parallel"  # Parallel multi-agent execution for complex cross-domain queries
    DIRECT = "direct"  # Direct LLM response (no tools)
    ESCALATE = "escalate"  # Escalate to human


class IntentCategory(str, Enum):
    """High-level intent categories."""

    QUERY = "query"  # Information retrieval
    ACTION = "action"  # Perform an operation
    ANALYSIS = "analysis"  # Complex analysis task
    TROUBLESHOOT = "troubleshoot"  # Diagnose/fix issues
    UNKNOWN = "unknown"  # Cannot classify


@dataclass
class IntentClassification:
    """
    Result of intent classification.

    The classifier analyzes user input and determines:
    - What the user wants (intent)
    - Confidence in the classification
    - Which domain(s) are involved
    - Suggested routing type
    """

    intent: str  # Specific intent (e.g., "list_instances", "analyze_performance")
    category: IntentCategory  # High-level category
    confidence: float  # 0.0 to 1.0
    domains: list[str]  # Involved domains (e.g., ["compute", "network"])
    entities: dict[str, Any] = field(default_factory=dict)  # Extracted entities
    suggested_workflow: str | None = None  # If workflow match found
    suggested_agent: str | None = None  # If agent delegation needed

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "intent": self.intent,
            "category": self.category.value,
            "confidence": self.confidence,
            "domains": self.domains,
            "entities": self.entities,
            "suggested_workflow": self.suggested_workflow,
            "suggested_agent": self.suggested_agent,
        }


@dataclass
class RoutingDecision:
    """
    Routing decision made by the coordinator.

    Determines how the request will be processed:
    - WORKFLOW: Execute a deterministic workflow
    - AGENT: Delegate to a specialized agent
    - DIRECT: Answer directly without tools
    - ESCALATE: Human intervention needed
    """

    routing_type: RoutingType
    target: str | None  # Workflow name or agent role
    confidence: float  # Decision confidence
    reasoning: str  # Why this routing was chosen
    fallback: RoutingDecision | None = None  # Alternative if primary fails

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "routing_type": self.routing_type.value,
            "target": self.target,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "fallback": self.fallback.to_dict() if self.fallback else None,
        }


@dataclass
class ToolCall:
    """Representation of a tool call request."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ToolResult:
    """Result from tool execution."""

    tool_call_id: str
    tool_name: str
    result: Any
    success: bool
    duration_ms: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "result": self.result,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class AgentContext:
    """
    Context passed to specialized agents.

    Contains all information an agent needs to process a request.
    """

    query: str  # Original user query
    intent: IntentClassification | None = None  # Classified intent
    session_id: str | None = None  # Session identifier
    thread_id: str | None = None  # Conversation thread
    user_id: str | None = None  # User identifier
    previous_results: list[ToolResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "intent": self.intent.to_dict() if self.intent else None,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "previous_results": [r.to_dict() for r in self.previous_results],
            "metadata": self.metadata,
        }


class CoordinatorState(TypedDict, total=False):
    """
    State maintained through the LangGraph execution (TypedDict for LangGraph 1.0).

    This state flows through all nodes in the coordinator graph:
    input → classifier → router → (workflow|agent|direct) → output

    All fields use ``total=False`` so nodes can return partial state dicts.
    The ``messages`` field uses the ``add_messages`` reducer so new messages
    are *appended* instead of replaced.
    """

    # Message history (managed by LangGraph add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Query processing
    query: str
    intent: IntentClassification | None
    routing: RoutingDecision | None

    # Tool execution
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]

    # Agent delegation
    agent_context: AgentContext | None
    current_agent: str | None
    agent_response: str | None

    # Workflow execution
    workflow_state: dict[str, Any]
    workflow_name: str | None

    # Loop control
    iteration: int
    max_iterations: int

    # Completion
    error: str | None
    final_response: str | None

    # Output formatting
    output_format: str  # markdown, slack, teams, html, plain
    channel_type: str | None  # slack, teams, web, api, cli

    # Transparency layer - tracks thinking process for user visibility
    thinking_trace: ThinkingTrace | None
    agent_candidates: list[AgentCandidate]
    enhanced_query: str | None  # LLM-enhanced version of original query

    # Metadata for profile-aware operations (OCI profile, context, etc.)
    metadata: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper functions for CoordinatorState
# (TypedDict doesn't support methods, so these are module-level functions)
# ─────────────────────────────────────────────────────────────────────────────


def state_to_dict(state: CoordinatorState) -> dict[str, Any]:
    """Serialize state to dictionary (for debugging/logging)."""
    intent = state.get("intent")
    routing = state.get("routing")
    thinking_trace = state.get("thinking_trace")
    return {
        "query": state.get("query", ""),
        "enhanced_query": state.get("enhanced_query"),
        "intent": intent.to_dict() if intent else None,
        "routing": routing.to_dict() if routing else None,
        "tool_calls": [tc.to_dict() for tc in state.get("tool_calls", [])],
        "tool_results": [tr.to_dict() for tr in state.get("tool_results", [])],
        "current_agent": state.get("current_agent"),
        "workflow_name": state.get("workflow_name"),
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 15),
        "error": state.get("error"),
        "has_final_response": state.get("final_response") is not None,
        "thinking_trace": thinking_trace.to_dict() if thinking_trace else None,
        "agent_candidates": [c.to_dict() for c in state.get("agent_candidates", [])],
        "metadata": state.get("metadata", {}),
    }


def add_thinking_step(
    state: CoordinatorState,
    phase: ThinkingPhase,
    message: str,
    data: dict[str, Any] | None = None,
) -> None:
    """
    Add a thinking step to the trace and trigger update callback.

    Note: This mutates the thinking_trace in-place. Since ThinkingTrace is a
    mutable object stored by reference, this works with TypedDict state.

    Args:
        state: Current coordinator state
        phase: The thinking phase
        message: Human-readable message
        data: Additional phase-specific data
    """
    import asyncio
    import inspect

    thinking_trace = state.get("thinking_trace")
    if thinking_trace is None:
        from src.agents.coordinator.transparency import create_thinking_trace
        thinking_trace = create_thinking_trace()
        state["thinking_trace"] = thinking_trace

    step = thinking_trace.add_step(phase, message, data)

    # Trigger callback for real-time updates (e.g., Slack)
    callback = get_thinking_callback()
    if callback and callable(callback):
        try:
            result = callback(step)
            if inspect.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    pass
        except Exception:
            pass  # Don't let callback errors break the flow


def state_should_continue(state: CoordinatorState) -> bool:
    """Check if graph execution should continue."""
    if state.get("error"):
        return False
    if state.get("iteration", 0) >= state.get("max_iterations", 15):
        return False
    if state.get("final_response") is not None:
        return False
    return True


def state_has_pending_tools(state: CoordinatorState) -> bool:
    """Check if there are pending tool calls."""
    return len(state.get("tool_calls", [])) > 0


def state_get_routing_type(state: CoordinatorState) -> RoutingType:
    """Get the routing type from decision."""
    routing = state.get("routing")
    if routing:
        return routing.routing_type
    return RoutingType.DIRECT


# ─────────────────────────────────────────────────────────────────────────────
# Workflow-First Routing Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# UPDATED: Balance between fast workflows and LLM-powered reasoning
# Simple queries (list_compartments, tenancy_info) → workflows
# Complex queries (analysis, temporal, optimization) → LLM agents
WORKFLOW_CONFIDENCE_THRESHOLD = 0.90  # Only route to workflow with very high confidence
AGENT_CONFIDENCE_THRESHOLD = 0.50  # Lower threshold to use agents more often
ESCALATION_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold before escalating
PARALLEL_DOMAIN_THRESHOLD = 2  # Use parallel if 2+ domains involved

# Query complexity indicators that should trigger agent routing
COMPLEXITY_KEYWORDS = [
    # Temporal reasoning - ALL month names for consistent routing
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "last month", "this month", "previous", "yesterday", "last week", "this week",
    "year to date", "ytd", "quarter", "q1", "q2", "q3", "q4",
    "compared to", "comparison", "trend", "change", "growth", "decline",
    # Analysis keywords
    "why", "explain", "analyze", "analysis", "understand", "investigate",
    "root cause", "reason", "insight", "recommendation", "optimize", "suggest",
    # Complex operations
    "forecast", "predict", "anomaly", "unusual", "spike", "drop",
    "correlate", "correlation", "relationship", "impact",
]


def has_complexity_indicators(query: str) -> bool:
    """Check if query contains complexity indicators requiring LLM reasoning."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in COMPLEXITY_KEYWORDS)


def determine_routing(
    intent: IntentClassification,
    original_query: str | None = None,
) -> RoutingDecision:
    """
    Determine routing based on intent classification.

    UPDATED Routing Strategy:
    - Simple queries + very high confidence (>=0.90) → WORKFLOW
    - Complex queries (temporal, analytical) → AGENT (with LLM reasoning)
    - Multi-domain (2+) + analysis/troubleshoot → PARALLEL
    - Medium confidence (>=0.50) + agent match → AGENT
    - Low confidence (<0.25) → ESCALATE
    - Otherwise → DIRECT (LLM handles it)

    The key change: Complex queries that need reasoning (temporal, analytical,
    optimization) are now routed to LLM-powered agents even if a workflow exists.

    Args:
        intent: Classified intent
        original_query: Original query for complexity detection

    Returns:
        Routing decision
    """
    # PRIORITY 1: Pre-classified workflows with very high confidence (>=0.95)
    # These are deterministic matches from keyword pre-classification
    # They should ALWAYS use the workflow, even if complexity indicators exist
    # Example: "show cost trend" → monthly_trend workflow (even though "trend" is complex)
    #
    # Includes DB-specific workflows that need to run the workflow directly
    # instead of going through the agent's generic health check analysis
    DB_SPECIFIC_WORKFLOWS = {
        "full_table_scan", "top_sql", "sql_monitoring", "long_running_ops",
        "parallelism_stats", "awr_report", "blocking_sessions", "wait_events",
        "sql_plan_baselines", "addm_findings",
        # Fleet/overview workflows - run directly without agent LLM reasoning
        "db_performance_overview", "db_fleet_health", "managed_databases",
    }

    if (
        intent.confidence >= 0.95
        and intent.suggested_workflow
        and (
            intent.suggested_agent is None  # Pre-classified workflows without agents
            or intent.suggested_workflow in DB_SPECIFIC_WORKFLOWS  # DB workflows should run directly
        )
    ):
        return RoutingDecision(
            routing_type=RoutingType.WORKFLOW,
            target=intent.suggested_workflow,
            confidence=intent.confidence,
            reasoning=f"Pre-classified workflow match ({intent.confidence:.2f}) - using workflow '{intent.suggested_workflow}'",
        )

    # Check if query requires LLM reasoning (temporal, analytical, etc.)
    needs_llm_reasoning = False
    if original_query and has_complexity_indicators(original_query):
        needs_llm_reasoning = True

    # Also check if category suggests need for reasoning
    if intent.category in (IntentCategory.ANALYSIS, IntentCategory.TROUBLESHOOT):
        needs_llm_reasoning = True

    # Route complex queries to agents for LLM-powered reasoning
    if needs_llm_reasoning and intent.suggested_agent:
        return RoutingDecision(
            routing_type=RoutingType.AGENT,
            target=intent.suggested_agent,
            confidence=intent.confidence,
            reasoning=f"Query requires LLM reasoning (complexity indicators detected) - routing to agent '{intent.suggested_agent}'",
            fallback=RoutingDecision(
                routing_type=RoutingType.WORKFLOW,
                target=intent.suggested_workflow,
                confidence=intent.confidence * 0.8,
                reasoning="Fallback to workflow if agent fails",
            )
            if intent.suggested_workflow
            else None,
        )

    # Check for workflow match with high confidence (simple queries only)
    if (
        intent.confidence >= WORKFLOW_CONFIDENCE_THRESHOLD
        and intent.suggested_workflow
        and not needs_llm_reasoning
    ):
        return RoutingDecision(
            routing_type=RoutingType.WORKFLOW,
            target=intent.suggested_workflow,
            confidence=intent.confidence,
            reasoning=f"Simple query with high confidence ({intent.confidence:.2f}) - using workflow '{intent.suggested_workflow}'",
            fallback=RoutingDecision(
                routing_type=RoutingType.AGENT,
                target=intent.suggested_agent,
                confidence=intent.confidence * 0.8,
                reasoning="Fallback to agent if workflow fails",
            )
            if intent.suggested_agent
            else None,
        )

    # Check for multi-domain complex queries → PARALLEL execution
    # Only for analysis/troubleshoot categories that benefit from parallel agents
    if (
        len(intent.domains) >= PARALLEL_DOMAIN_THRESHOLD
        and intent.category in (IntentCategory.ANALYSIS, IntentCategory.TROUBLESHOOT)
        and intent.confidence >= AGENT_CONFIDENCE_THRESHOLD
    ):
        return RoutingDecision(
            routing_type=RoutingType.PARALLEL,
            target=None,  # Orchestrator determines agents
            confidence=intent.confidence,
            reasoning=f"Multi-domain ({len(intent.domains)} domains: {', '.join(intent.domains)}) {intent.category.value} - using parallel execution",
            fallback=RoutingDecision(
                routing_type=RoutingType.AGENT,
                target=intent.suggested_agent,
                confidence=intent.confidence * 0.8,
                reasoning="Fallback to single agent if parallel fails",
            )
            if intent.suggested_agent
            else None,
        )

    # Check for agent match with medium confidence
    if intent.confidence >= AGENT_CONFIDENCE_THRESHOLD and intent.suggested_agent:
        return RoutingDecision(
            routing_type=RoutingType.AGENT,
            target=intent.suggested_agent,
            confidence=intent.confidence,
            reasoning=f"Medium confidence ({intent.confidence:.2f}) - delegating to agent '{intent.suggested_agent}'",
        )

    # Very low confidence - escalate
    if intent.confidence < ESCALATION_CONFIDENCE_THRESHOLD:
        return RoutingDecision(
            routing_type=RoutingType.ESCALATE,
            target=None,
            confidence=intent.confidence,
            reasoning=f"Low confidence ({intent.confidence:.2f}) - escalating for human review",
        )

    # Default - direct LLM response
    return RoutingDecision(
        routing_type=RoutingType.DIRECT,
        target=None,
        confidence=intent.confidence,
        reasoning="No workflow/agent match - direct LLM response",
    )
