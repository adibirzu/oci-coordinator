"""
Coordinator State schema for LangGraph orchestration.

This module defines the state that flows through the coordinator graph,
including intent classification, workflow routing, and agent context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RoutingType(str, Enum):
    """Type of routing decision."""

    WORKFLOW = "workflow"  # Deterministic workflow execution
    AGENT = "agent"  # Delegate to specialized agent
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


@dataclass
class CoordinatorState:
    """
    State maintained through the LangGraph execution.

    This state flows through all nodes in the coordinator graph:
    input → classifier → router → (workflow|agent|direct) → output

    Attributes:
        messages: Conversation history (LangGraph managed)
        query: Original user query
        intent: Classified intent information
        routing: Routing decision
        tool_calls: Pending tool calls from LLM
        tool_results: Results from executed tools
        agent_context: Context for specialized agents
        current_agent: Currently executing agent role
        workflow_state: State for workflow execution
        iteration: Current iteration count
        max_iterations: Maximum allowed iterations (loop guard)
        error: Any error that occurred
        final_response: Final response to return
        output_format: Output format for response (markdown, slack, teams, etc.)
        channel_type: Input channel type for format detection
    """

    # Message history (managed by LangGraph add_messages reducer)
    messages: Annotated[Sequence[BaseMessage], add_messages] = field(
        default_factory=list
    )

    # Query processing
    query: str = ""
    intent: IntentClassification | None = None
    routing: RoutingDecision | None = None

    # Tool execution
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    # Agent delegation
    agent_context: AgentContext | None = None
    current_agent: str | None = None
    agent_response: str | None = None

    # Workflow execution
    workflow_state: dict[str, Any] = field(default_factory=dict)
    workflow_name: str | None = None

    # Loop control
    iteration: int = 0
    max_iterations: int = 15

    # Completion
    error: str | None = None
    final_response: str | None = None

    # Output formatting
    output_format: str = "markdown"  # markdown, slack, teams, html, plain
    channel_type: str | None = None  # slack, teams, web, api, cli

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary (for debugging/logging)."""
        return {
            "query": self.query,
            "intent": self.intent.to_dict() if self.intent else None,
            "routing": self.routing.to_dict() if self.routing else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "current_agent": self.current_agent,
            "workflow_name": self.workflow_name,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "error": self.error,
            "has_final_response": self.final_response is not None,
        }

    def should_continue(self) -> bool:
        """Check if graph execution should continue."""
        if self.error:
            return False
        if self.iteration >= self.max_iterations:
            return False
        if self.final_response is not None:
            return False
        return True

    def has_pending_tools(self) -> bool:
        """Check if there are pending tool calls."""
        return len(self.tool_calls) > 0

    def get_routing_type(self) -> RoutingType:
        """Get the routing type from decision."""
        if self.routing:
            return self.routing.routing_type
        return RoutingType.DIRECT


# ─────────────────────────────────────────────────────────────────────────────
# Workflow-First Routing Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Target: 70%+ of requests handled by deterministic workflows
WORKFLOW_CONFIDENCE_THRESHOLD = 0.80  # Route to workflow if confidence >= 0.80
AGENT_CONFIDENCE_THRESHOLD = 0.60  # Route to agent if confidence >= 0.60
ESCALATION_CONFIDENCE_THRESHOLD = 0.30  # Escalate if confidence < 0.30


def determine_routing(intent: IntentClassification) -> RoutingDecision:
    """
    Determine routing based on intent classification.

    Workflow-First Design:
    - High confidence (>=0.80) + workflow match → WORKFLOW
    - Medium confidence (>=0.60) + agent match → AGENT
    - Low confidence (<0.30) → ESCALATE
    - Otherwise → DIRECT (LLM handles it)

    Args:
        intent: Classified intent

    Returns:
        Routing decision
    """
    # Check for workflow match with high confidence
    if (
        intent.confidence >= WORKFLOW_CONFIDENCE_THRESHOLD
        and intent.suggested_workflow
    ):
        return RoutingDecision(
            routing_type=RoutingType.WORKFLOW,
            target=intent.suggested_workflow,
            confidence=intent.confidence,
            reasoning=f"High confidence ({intent.confidence:.2f}) match for workflow '{intent.suggested_workflow}'",
            fallback=RoutingDecision(
                routing_type=RoutingType.AGENT,
                target=intent.suggested_agent,
                confidence=intent.confidence * 0.8,
                reasoning="Fallback to agent if workflow fails",
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
        reasoning=f"No workflow/agent match - direct LLM response",
    )
