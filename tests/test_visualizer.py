"""Tests for the LangGraph workflow visualizer."""

from datetime import datetime

import pytest

from src.observability.visualizer import (
    EXAMPLE_QUERIES,
    GRAPH_EDGES,
    GRAPH_NODES,
    EdgeType,
    ExecutionStep,
    NodeStatus,
    WorkflowEdge,
    WorkflowNode,
    WorkflowVisualization,
    WorkflowVisualizer,
    generate_execution_trace_diagram,
    generate_mermaid_diagram,
    get_visualization_data,
)


class TestWorkflowNode:
    """Tests for WorkflowNode dataclass."""

    def test_node_creation(self):
        """Test creating a workflow node."""
        node = WorkflowNode(
            id="test_node",
            name="Test Node",
            description="A test node",
        )
        assert node.id == "test_node"
        assert node.name == "Test Node"
        assert node.status == NodeStatus.IDLE

    def test_node_to_dict(self):
        """Test serializing node to dict."""
        node = WorkflowNode(
            id="input",
            name="Input",
            description="Process query",
            status=NodeStatus.ACTIVE,
            duration_ms=150,
        )
        data = node.to_dict()

        assert data["id"] == "input"
        assert data["status"] == "active"
        assert data["duration_ms"] == 150


class TestWorkflowEdge:
    """Tests for WorkflowEdge dataclass."""

    def test_edge_creation(self):
        """Test creating a workflow edge."""
        edge = WorkflowEdge(
            source="input",
            target="classifier",
            edge_type=EdgeType.SEQUENTIAL,
        )
        assert edge.source == "input"
        assert edge.target == "classifier"
        assert edge.edge_type == EdgeType.SEQUENTIAL

    def test_conditional_edge(self):
        """Test creating a conditional edge."""
        edge = WorkflowEdge(
            source="router",
            target="workflow",
            edge_type=EdgeType.CONDITIONAL,
            condition="routing_type == WORKFLOW",
        )
        assert edge.edge_type == EdgeType.CONDITIONAL
        assert edge.condition is not None


class TestGraphDefinitions:
    """Tests for static graph definitions."""

    def test_graph_nodes_exist(self):
        """Test that all required nodes are defined."""
        node_ids = {n.id for n in GRAPH_NODES}

        required_nodes = {
            "__start__",
            "__end__",
            "input",
            "classifier",
            "router",
            "workflow",
            "parallel",
            "agent",
            "action",
            "output",
        }

        assert required_nodes.issubset(node_ids)

    def test_graph_edges_exist(self):
        """Test that edges connect valid nodes."""
        node_ids = {n.id for n in GRAPH_NODES}

        for edge in GRAPH_EDGES:
            assert edge.source in node_ids, f"Source {edge.source} not in nodes"
            assert edge.target in node_ids, f"Target {edge.target} not in nodes"

    def test_example_queries_categories(self):
        """Test that example queries cover all routing types."""
        assert "workflow" in EXAMPLE_QUERIES
        assert "parallel" in EXAMPLE_QUERIES
        assert "agent" in EXAMPLE_QUERIES

        # Each category should have examples
        for category, examples in EXAMPLE_QUERIES.items():
            assert len(examples) > 0, f"No examples for {category}"


class TestMermaidDiagramGeneration:
    """Tests for Mermaid diagram generation."""

    def test_generate_basic_diagram(self):
        """Test generating a basic Mermaid diagram."""
        diagram = generate_mermaid_diagram()

        assert "flowchart TD" in diagram
        assert "__start__" in diagram
        assert "__end__" in diagram
        assert "input" in diagram
        assert "classifier" in diagram
        assert "router" in diagram

    def test_generate_diagram_with_routing(self):
        """Test generating diagram with routing type highlighted."""
        diagram = generate_mermaid_diagram(routing_type="WORKFLOW")

        assert "flowchart TD" in diagram
        # Should contain routing path comment
        assert "Routing path: WORKFLOW" in diagram

    def test_generate_diagram_with_active_node(self):
        """Test generating diagram with active node."""
        diagram = generate_mermaid_diagram(active_node="classifier")

        assert "style classifier fill:#22c55e" in diagram

    def test_generate_diagram_with_active_path(self):
        """Test generating diagram with active path."""
        diagram = generate_mermaid_diagram(
            active_path=["input", "classifier", "router"]
        )

        # Active path nodes should be styled
        assert "style input" in diagram or "style classifier" in diagram


class TestExecutionTraceDiagram:
    """Tests for execution trace sequence diagram."""

    def test_empty_trace(self):
        """Test generating diagram with empty trace."""
        diagram = generate_execution_trace_diagram([])

        assert "sequenceDiagram" in diagram
        assert "No execution trace available" in diagram

    def test_trace_with_steps(self):
        """Test generating diagram with execution steps."""
        steps = [
            ExecutionStep(
                node_id="input",
                timestamp=datetime.now(),
                phase="received",
                message="Query received",
            ),
            ExecutionStep(
                node_id="classifier",
                timestamp=datetime.now(),
                phase="classifying",
                message="Classifying intent",
            ),
        ]

        diagram = generate_execution_trace_diagram(steps)

        assert "sequenceDiagram" in diagram
        assert "User" in diagram
        assert "Coordinator" in diagram


class TestWorkflowVisualizer:
    """Tests for WorkflowVisualizer class."""

    def test_get_static_visualization(self):
        """Test getting static visualization."""
        visualizer = WorkflowVisualizer()
        viz = visualizer.get_static_visualization()

        assert isinstance(viz, WorkflowVisualization)
        assert len(viz.nodes) > 0
        assert len(viz.edges) > 0
        assert viz.mermaid_diagram != ""

    def test_get_example_queries(self):
        """Test getting example queries."""
        visualizer = WorkflowVisualizer()
        examples = visualizer.get_example_queries()

        assert "workflow" in examples
        assert "agent" in examples

    def test_visualization_to_dict(self):
        """Test serializing visualization to dict."""
        visualizer = WorkflowVisualizer()
        viz = visualizer.get_static_visualization()
        data = viz.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert "mermaid_diagram" in data
        assert "timestamp" in data


class TestGetVisualizationData:
    """Tests for get_visualization_data utility function."""

    def test_basic_visualization_data(self):
        """Test getting basic visualization data."""
        data = get_visualization_data()

        assert "nodes" in data
        assert "edges" in data
        assert "mermaid_diagram" in data

    def test_visualization_with_examples(self):
        """Test getting visualization data with examples."""
        data = get_visualization_data(include_examples=True)

        assert "example_queries" in data
        assert "workflow" in data["example_queries"]

    def test_visualization_without_examples(self):
        """Test getting visualization data without examples."""
        data = get_visualization_data(include_examples=False)

        assert "example_queries" not in data


class TestExecutionStep:
    """Tests for ExecutionStep dataclass."""

    def test_step_creation(self):
        """Test creating an execution step."""
        step = ExecutionStep(
            node_id="classifier",
            timestamp=datetime.now(),
            phase="classifying",
            message="Classifying intent",
            duration_ms=50,
        )

        assert step.node_id == "classifier"
        assert step.phase == "classifying"
        assert step.duration_ms == 50

    def test_step_to_dict(self):
        """Test serializing step to dict."""
        now = datetime.now()
        step = ExecutionStep(
            node_id="router",
            timestamp=now,
            phase="routing",
            message="Determining route",
            data={"candidates": 3},
        )
        data = step.to_dict()

        assert data["node_id"] == "router"
        assert data["phase"] == "routing"
        assert data["data"]["candidates"] == 3


class TestNodeStatus:
    """Tests for NodeStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert NodeStatus.IDLE.value == "idle"
        assert NodeStatus.ACTIVE.value == "active"
        assert NodeStatus.COMPLETED.value == "completed"
        assert NodeStatus.SKIPPED.value == "skipped"
        assert NodeStatus.ERROR.value == "error"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_edge_type_values(self):
        """Test that all expected edge types exist."""
        assert EdgeType.SEQUENTIAL.value == "sequential"
        assert EdgeType.CONDITIONAL.value == "conditional"
        assert EdgeType.LOOP.value == "loop"
