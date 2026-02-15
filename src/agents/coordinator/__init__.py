"""
LangGraph Coordinator for OCI AI Agents.

This module provides the central orchestration for the OCI Coordinator,
implementing a workflow-first design with agentic fallback.
"""

from src.agents.coordinator.graph import LangGraphCoordinator, create_coordinator
from src.agents.coordinator.nodes import CoordinatorNodes
from src.agents.coordinator.state import (
    AgentContext,
    CoordinatorState,
    IntentCategory,
    IntentClassification,
    RoutingDecision,
    RoutingType,
    ToolCall,
    ToolResult,
    add_thinking_step,
    determine_routing,
    state_get_routing_type,
    state_has_pending_tools,
    state_should_continue,
    state_to_dict,
)
from src.agents.coordinator.workflows import (
    WORKFLOW_REGISTRY,
    get_workflow_registry,
    list_workflows,
)

__all__ = [
    # Main coordinator
    "LangGraphCoordinator",
    "create_coordinator",
    "CoordinatorNodes",
    # State classes
    "CoordinatorState",
    "IntentClassification",
    "IntentCategory",
    "RoutingDecision",
    "RoutingType",
    "AgentContext",
    "ToolCall",
    "ToolResult",
    # State helper functions
    "add_thinking_step",
    "determine_routing",
    "state_get_routing_type",
    "state_has_pending_tools",
    "state_should_continue",
    "state_to_dict",
    # Workflows
    "WORKFLOW_REGISTRY",
    "get_workflow_registry",
    "list_workflows",
]
