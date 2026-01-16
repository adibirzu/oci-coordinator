"""
LangGraph Workflow Visualizer for OCI AI Agent Coordinator.

Provides interactive visualization of the coordinator's workflow graph,
including real-time execution tracing and troubleshooting support.

Features:
- Static graph structure visualization via Mermaid diagrams
- Live execution tracing with highlighted active nodes
- Agent routing visualization with confidence scores
- Example queries for different workflow paths

Inspired by: https://surma.dev/things/langgraph/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.coordinator.graph import LangGraphCoordinator
    from src.agents.coordinator.transparency import ThinkingTrace


class NodeStatus(str, Enum):
    """Status of a node in the workflow."""

    IDLE = "idle"           # Not yet reached
    ACTIVE = "active"       # Currently executing
    COMPLETED = "completed" # Finished successfully
    SKIPPED = "skipped"     # Not taken in this path
    ERROR = "error"         # Failed


class EdgeType(str, Enum):
    """Type of edge in the workflow graph."""

    SEQUENTIAL = "sequential"     # Always follows
    CONDITIONAL = "conditional"   # Based on condition
    LOOP = "loop"                 # Loop back edge


@dataclass
class WorkflowNode:
    """Represents a node in the workflow graph."""

    id: str
    name: str
    description: str
    status: NodeStatus = NodeStatus.IDLE
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowEdge:
    """Represents an edge in the workflow graph."""

    source: str
    target: str
    edge_type: EdgeType = EdgeType.SEQUENTIAL
    condition: str | None = None
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "condition": self.condition,
            "active": self.active,
        }


@dataclass
class ExecutionStep:
    """Represents a step in the execution trace."""

    node_id: str
    timestamp: datetime
    phase: str
    message: str
    duration_ms: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "data": self.data,
        }


@dataclass
class WorkflowVisualization:
    """Complete workflow visualization data."""

    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]
    execution_trace: list[ExecutionStep] = field(default_factory=list)
    active_node: str | None = None
    routing_type: str | None = None
    current_agent: str | None = None
    mermaid_diagram: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "execution_trace": [s.to_dict() for s in self.execution_trace],
            "active_node": self.active_node,
            "routing_type": self.routing_type,
            "current_agent": self.current_agent,
            "mermaid_diagram": self.mermaid_diagram,
            "timestamp": self.timestamp.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Static Graph Definition
# ─────────────────────────────────────────────────────────────────────────────

# Node definitions matching the coordinator graph
GRAPH_NODES: list[WorkflowNode] = [
    WorkflowNode(
        id="__start__",
        name="START",
        description="Entry point - query received",
    ),
    WorkflowNode(
        id="input",
        name="Input",
        description="Process and enhance the user query",
    ),
    WorkflowNode(
        id="classifier",
        name="Classifier",
        description="Classify intent, extract entities, identify domains",
    ),
    WorkflowNode(
        id="router",
        name="Router",
        description="Determine routing: workflow, parallel, agent, or escalate",
    ),
    WorkflowNode(
        id="workflow",
        name="Workflow",
        description="Execute deterministic workflow (70%+ of requests)",
    ),
    WorkflowNode(
        id="parallel",
        name="Parallel",
        description="Execute multiple agents in parallel for complex queries",
    ),
    WorkflowNode(
        id="agent",
        name="Agent",
        description="Delegate to specialized agent for LLM reasoning",
    ),
    WorkflowNode(
        id="action",
        name="Action",
        description="Execute tool calls from agent",
    ),
    WorkflowNode(
        id="output",
        name="Output",
        description="Format and return final response",
    ),
    WorkflowNode(
        id="__end__",
        name="END",
        description="Processing complete",
    ),
]

# Edge definitions matching the coordinator graph
GRAPH_EDGES: list[WorkflowEdge] = [
    # Entry
    WorkflowEdge("__start__", "input", EdgeType.SEQUENTIAL),

    # Input → Classifier (always)
    WorkflowEdge("input", "classifier", EdgeType.SEQUENTIAL),

    # Classifier → Router (always)
    WorkflowEdge("classifier", "router", EdgeType.SEQUENTIAL),

    # Router → Conditional branches
    WorkflowEdge("router", "workflow", EdgeType.CONDITIONAL, "routing_type == WORKFLOW"),
    WorkflowEdge("router", "parallel", EdgeType.CONDITIONAL, "routing_type == PARALLEL"),
    WorkflowEdge("router", "agent", EdgeType.CONDITIONAL, "routing_type == AGENT"),
    WorkflowEdge("router", "output", EdgeType.CONDITIONAL, "routing_type == ESCALATE/DIRECT"),

    # Workflow → Output (terminal)
    WorkflowEdge("workflow", "output", EdgeType.SEQUENTIAL),

    # Parallel → Output (terminal)
    WorkflowEdge("parallel", "output", EdgeType.SEQUENTIAL),

    # Agent → Conditional
    WorkflowEdge("agent", "action", EdgeType.CONDITIONAL, "has_tool_calls"),
    WorkflowEdge("agent", "output", EdgeType.CONDITIONAL, "final_response or max_iterations"),

    # Action → Loop back or finish
    WorkflowEdge("action", "agent", EdgeType.LOOP, "continue_loop"),
    WorkflowEdge("action", "output", EdgeType.CONDITIONAL, "error or max_iterations"),

    # Output → END
    WorkflowEdge("output", "__end__", EdgeType.SEQUENTIAL),
]


# ─────────────────────────────────────────────────────────────────────────────
# Example Queries for Each Path
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES: dict[str, list[dict[str, str]]] = {
    "workflow": [
        {
            "query": "List all compartments",
            "description": "Simple infrastructure query → deterministic workflow",
            "workflow": "list_compartments_workflow",
        },
        {
            "query": "Show cost summary",
            "description": "Cost query → cost_summary_workflow",
            "workflow": "cost_summary_workflow",
        },
        {
            "query": "Check blocking sessions",
            "description": "DB troubleshooting → db_blocking_sessions_workflow",
            "workflow": "db_blocking_sessions_workflow",
        },
        {
            "query": "Show wait events",
            "description": "DB performance → db_wait_events_workflow",
            "workflow": "db_wait_events_workflow",
        },
    ],
    "parallel": [
        {
            "query": "Analyze database performance and compare with monthly costs",
            "description": "Multi-domain: DB + FinOps → Parallel execution",
            "agents": ["DbTroubleshootAgent", "FinOpsAgent"],
        },
        {
            "query": "Check security issues and their impact on infrastructure",
            "description": "Multi-domain: Security + Infrastructure → Parallel execution",
            "agents": ["SecurityThreatAgent", "InfrastructureAgent"],
        },
    ],
    "agent": [
        {
            "query": "Why is my database slow today compared to yesterday?",
            "description": "Complex analytical query → DbTroubleshootAgent with reasoning",
            "agent": "DbTroubleshootAgent",
        },
        {
            "query": "Investigate the cost anomaly in November",
            "description": "Investigation query → FinOpsAgent with tool calls",
            "agent": "FinOpsAgent",
        },
        {
            "query": "Correlate security alerts with log patterns",
            "description": "Cross-reference query → LogAnalyticsAgent",
            "agent": "LogAnalyticsAgent",
        },
    ],
    "escalate": [
        {
            "query": "Deploy new infrastructure to production",
            "description": "Potentially dangerous action → Escalate for human review",
            "reason": "Action requires human approval",
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Mermaid Diagram Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_mermaid_diagram(
    active_node: str | None = None,
    active_path: list[str] | None = None,
    routing_type: str | None = None,
) -> str:
    """
    Generate a Mermaid flowchart diagram of the coordinator workflow.

    Args:
        active_node: Currently active node (highlighted)
        active_path: List of nodes in the execution path (highlighted)
        routing_type: The routing decision (workflow, parallel, agent, escalate)

    Returns:
        Mermaid diagram string
    """
    active_path = active_path or []

    lines = [
        "%%{init: {'theme': 'base', 'themeVariables': {"
        "'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', "
        "'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', "
        "'secondaryColor': '#f0fdf4', 'tertiaryColor': '#fef3c7'"
        "}}}%%",
        "flowchart TD",
        "",
        "    %% Nodes",
    ]

    # Node definitions with styling
    node_styles = {
        "__start__": "((START))",
        "__end__": "((END))",
        "input": "[📥 Input<br/>Process Query]",
        "classifier": "[🎯 Classifier<br/>Intent + Entities]",
        "router": "{🔀 Router<br/>Route Decision}",
        "workflow": "[⚡ Workflow<br/>Deterministic]",
        "parallel": "[🔄 Parallel<br/>Multi-Agent]",
        "agent": "[🤖 Agent<br/>LLM Reasoning]",
        "action": "[🔧 Action<br/>Tool Execution]",
        "output": "[📤 Output<br/>Format Response]",
    }

    for node_id, shape in node_styles.items():
        lines.append(f"    {node_id}{shape}")

    lines.append("")
    lines.append("    %% Edges")

    # Sequential edges
    lines.append("    __start__ --> input")
    lines.append("    input --> classifier")
    lines.append("    classifier --> router")
    lines.append("    workflow --> output")
    lines.append("    parallel --> output")
    lines.append("    output --> __end__")

    # Conditional edges from router
    lines.append("")
    lines.append("    %% Router conditional edges")
    lines.append("    router -->|WORKFLOW| workflow")
    lines.append("    router -->|PARALLEL| parallel")
    lines.append("    router -->|AGENT| agent")
    lines.append("    router -.->|ESCALATE| output")

    # Agent loop edges
    lines.append("")
    lines.append("    %% Agent-Action loop")
    lines.append("    agent -->|tool_calls| action")
    lines.append("    agent -->|done| output")
    lines.append("    action -->|continue| agent")
    lines.append("    action -.->|max_iter| output")

    # Styling for active nodes
    if active_node or active_path:
        lines.append("")
        lines.append("    %% Active node styling")

        if active_node:
            lines.append(f"    style {active_node} fill:#22c55e,stroke:#16a34a,stroke-width:3px")

        for node in active_path:
            if node != active_node:
                lines.append(f"    style {node} fill:#86efac,stroke:#22c55e,stroke-width:2px")

    # Highlight the routing path taken
    if routing_type:
        lines.append("")
        lines.append(f"    %% Routing path: {routing_type}")

        path_map = {
            "WORKFLOW": ["router", "workflow", "output"],
            "PARALLEL": ["router", "parallel", "output"],
            "AGENT": ["router", "agent"],
            "ESCALATE": ["router", "output"],
            "DIRECT": ["router", "output"],
        }

        if routing_type.upper() in path_map:
            for node in path_map[routing_type.upper()]:
                lines.append(f"    style {node} stroke:#f59e0b,stroke-width:3px")

    return "\n".join(lines)


def generate_execution_trace_diagram(
    execution_trace: list[ExecutionStep],
    include_timing: bool = True,
) -> str:
    """
    Generate a Mermaid sequence diagram showing execution flow.

    Args:
        execution_trace: List of execution steps
        include_timing: Whether to include timing information

    Returns:
        Mermaid sequence diagram string
    """
    if not execution_trace:
        return "sequenceDiagram\n    Note over User,Coordinator: No execution trace available"

    lines = [
        "sequenceDiagram",
        "    autonumber",
        "",
        "    participant User",
        "    participant Coordinator",
        "    participant Classifier",
        "    participant Router",
        "    participant Agent",
        "    participant Tools",
        "",
    ]

    # Map phases to participants
    phase_map = {
        "received": ("User", "Coordinator", "Query"),
        "enhancing": ("Coordinator", "Coordinator", "Enhance"),
        "enhanced": ("Coordinator", "Coordinator", "Enhanced"),
        "classifying": ("Coordinator", "Classifier", "Classify"),
        "classified": ("Classifier", "Coordinator", "Intent"),
        "routing": ("Coordinator", "Router", "Route"),
        "routed": ("Router", "Coordinator", "Decision"),
        "delegating": ("Coordinator", "Agent", "Delegate"),
        "executing": ("Agent", "Agent", "Execute"),
        "tool_call": ("Agent", "Tools", "Call"),
        "tool_result": ("Tools", "Agent", "Result"),
        "synthesizing": ("Agent", "Coordinator", "Synthesize"),
        "complete": ("Coordinator", "User", "Response"),
        "error": ("Coordinator", "User", "Error"),
    }

    for step in execution_trace:
        phase = step.phase.lower()
        if phase in phase_map:
            src, dst, action = phase_map[phase]
            msg = step.message[:50] + "..." if len(step.message) > 50 else step.message

            if include_timing and step.duration_ms:
                msg += f" ({step.duration_ms}ms)"

            lines.append(f"    {src}->>+{dst}: {action}: {msg}")
        else:
            lines.append(f"    Note over Coordinator: {step.phase}: {step.message[:30]}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Builder
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowVisualizer:
    """
    Builder for workflow visualizations.

    Provides methods to create visualizations from:
    - Static graph structure
    - Live execution traces
    - ThinkingTrace objects
    """

    def __init__(self, coordinator: LangGraphCoordinator | None = None):
        """
        Initialize the visualizer.

        Args:
            coordinator: Optional coordinator for getting live graph diagram
        """
        self._coordinator = coordinator

    def get_static_visualization(self) -> WorkflowVisualization:
        """
        Get the static workflow graph structure.

        Returns:
            WorkflowVisualization with nodes, edges, and Mermaid diagram
        """
        return WorkflowVisualization(
            nodes=[WorkflowNode(**n.__dict__) for n in GRAPH_NODES],
            edges=[WorkflowEdge(**e.__dict__) for e in GRAPH_EDGES],
            mermaid_diagram=generate_mermaid_diagram(),
        )

    def get_live_visualization(
        self,
        thinking_trace: ThinkingTrace | None = None,
        routing_type: str | None = None,
        current_agent: str | None = None,
    ) -> WorkflowVisualization:
        """
        Get a visualization with live execution state.

        Args:
            thinking_trace: ThinkingTrace from coordinator execution
            routing_type: The routing decision made
            current_agent: The currently executing agent

        Returns:
            WorkflowVisualization with execution state
        """
        nodes = [WorkflowNode(**n.__dict__) for n in GRAPH_NODES]
        edges = [WorkflowEdge(**e.__dict__) for e in GRAPH_EDGES]
        execution_trace: list[ExecutionStep] = []
        active_node: str | None = None
        active_path: list[str] = []

        if thinking_trace:
            # Convert ThinkingTrace to ExecutionSteps
            phase_to_node = {
                "received": "input",
                "enhancing": "input",
                "enhanced": "input",
                "classifying": "classifier",
                "classified": "classifier",
                "routing": "router",
                "routed": "router",
                "delegating": "agent",
                "executing": "agent",
                "tool_call": "action",
                "tool_result": "action",
                "synthesizing": "output",
                "complete": "output",
                "error": "output",
            }

            for step in thinking_trace.steps:
                node_id = phase_to_node.get(step.phase.value, "output")
                execution_trace.append(ExecutionStep(
                    node_id=node_id,
                    timestamp=step.timestamp,
                    phase=step.phase.value,
                    message=step.message,
                    duration_ms=step.duration_ms,
                    data=step.data,
                ))

                if node_id not in active_path:
                    active_path.append(node_id)

            # Determine active node from last step
            if execution_trace:
                last_step = execution_trace[-1]
                if last_step.phase not in ("complete", "error"):
                    active_node = last_step.node_id

            # Update node statuses
            for node in nodes:
                if node.id == active_node:
                    node.status = NodeStatus.ACTIVE
                elif node.id in active_path:
                    node.status = NodeStatus.COMPLETED
                elif node.id in ("__start__", "__end__"):
                    if "__start__" in active_path or active_path:
                        node.status = NodeStatus.COMPLETED if node.id == "__start__" else NodeStatus.IDLE

        # Generate diagram with active state
        mermaid = generate_mermaid_diagram(
            active_node=active_node,
            active_path=active_path,
            routing_type=routing_type,
        )

        return WorkflowVisualization(
            nodes=nodes,
            edges=edges,
            execution_trace=execution_trace,
            active_node=active_node,
            routing_type=routing_type,
            current_agent=current_agent,
            mermaid_diagram=mermaid,
        )

    def get_coordinator_diagram(self) -> str:
        """
        Get the Mermaid diagram directly from the coordinator.

        Returns:
            Mermaid diagram string from LangGraph's draw_mermaid()
        """
        if self._coordinator:
            return self._coordinator.get_graph_diagram()
        return generate_mermaid_diagram()

    def get_example_queries(self) -> dict[str, list[dict[str, str]]]:
        """
        Get example queries for each routing path.

        Returns:
            Dictionary of routing types to example queries
        """
        return EXAMPLE_QUERIES

    def get_agents_visualization(
        self,
        agent_candidates: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Generate a Mermaid diagram showing agent selection.

        Args:
            agent_candidates: List of candidate agents with confidence scores

        Returns:
            Mermaid diagram string
        """
        lines = [
            "%%{init: {'theme': 'base'}}%%",
            "flowchart LR",
            "",
            "    subgraph Agents[\"🤖 Available Agents\"]",
        ]

        # Define all agents
        agents = [
            ("db", "DbTroubleshoot", "Database"),
            ("log", "LogAnalytics", "Observability"),
            ("sec", "SecurityThreat", "Security"),
            ("fin", "FinOps", "Cost"),
            ("infra", "Infrastructure", "Compute/Network"),
            ("err", "ErrorAnalysis", "Debugging"),
            ("ai", "SelectAI", "Data/AI"),
        ]

        for agent_id, name, domain in agents:
            lines.append(f"        {agent_id}[{name}<br/>({domain})]")

        lines.append("    end")

        # If we have candidates, show selection
        if agent_candidates:
            lines.append("")
            lines.append("    subgraph Selection[\"🎯 Selection\"]")
            lines.append("        query((Query))")
            lines.append("    end")
            lines.append("")

            # Connect query to matching agents
            for candidate in agent_candidates:
                agent_role = candidate.get("agent_role", "")
                confidence = candidate.get("confidence", 0)
                selected = candidate.get("selected", False)

                # Find matching agent ID
                agent_id = None
                for aid, name, _ in agents:
                    if name.lower() in agent_role.lower():
                        agent_id = aid
                        break

                if agent_id:
                    style = "==>" if selected else "-->"
                    lines.append(f"    query {style}|{confidence:.0%}| {agent_id}")

                    if selected:
                        lines.append(f"    style {agent_id} fill:#22c55e,stroke:#16a34a")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_visualization_data(
    coordinator: LangGraphCoordinator | None = None,
    thinking_trace: ThinkingTrace | None = None,
    routing_type: str | None = None,
    current_agent: str | None = None,
    include_examples: bool = True,
) -> dict[str, Any]:
    """
    Get complete visualization data for the API response.

    Args:
        coordinator: Optional coordinator instance
        thinking_trace: Optional execution trace
        routing_type: The routing decision
        current_agent: Currently executing agent
        include_examples: Include example queries

    Returns:
        Dictionary with all visualization data
    """
    visualizer = WorkflowVisualizer(coordinator)

    # Get base visualization
    if thinking_trace:
        viz = visualizer.get_live_visualization(
            thinking_trace=thinking_trace,
            routing_type=routing_type,
            current_agent=current_agent,
        )
    else:
        viz = visualizer.get_static_visualization()

    result = viz.to_dict()

    # Add coordinator's native diagram if available
    if coordinator:
        result["langgraph_diagram"] = visualizer.get_coordinator_diagram()

    # Add example queries
    if include_examples:
        result["example_queries"] = visualizer.get_example_queries()

    return result
