"""
Transparency Layer for Coordinator Decision Making.

This module provides data structures and utilities for tracking the coordinator's
thinking process, making decisions visible to users.

The transparency layer enables:
- Real-time thinking updates via Slack/Teams
- Agent selection visibility (why this agent was chosen)
- Query enhancement tracking (how the query was refined)
- Tool call tracing (what tools were invoked)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ThinkingPhase(str, Enum):
    """Phases in the coordinator's thinking process."""

    # Query understanding
    RECEIVED = "received"  # Query received from user
    ENHANCING = "enhancing"  # LLM enhancing/clarifying query
    ENHANCED = "enhanced"  # Query enhancement complete

    # Classification
    CLASSIFYING = "classifying"  # Classifying intent
    CLASSIFIED = "classified"  # Classification complete

    # Agent discovery
    DISCOVERING = "discovering"  # Finding matching agents
    DISCOVERED = "discovered"  # Agent candidates identified

    # Routing
    ROUTING = "routing"  # Determining route
    ROUTED = "routed"  # Route selected

    # Execution
    DELEGATING = "delegating"  # Delegating to agent
    EXECUTING = "executing"  # Agent/workflow executing
    TOOL_CALL = "tool_call"  # Tool being called
    TOOL_RESULT = "tool_result"  # Tool returned result

    # Completion
    SYNTHESIZING = "synthesizing"  # Combining results
    COMPLETE = "complete"  # Processing complete
    ERROR = "error"  # Error occurred


@dataclass
class ThinkingStep:
    """
    A single step in the coordinator's thinking process.

    Each step represents a decision point or action taken by the coordinator,
    providing transparency into how requests are processed.
    """

    phase: ThinkingPhase
    message: str  # Human-readable description
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)  # Phase-specific data
    duration_ms: int | None = None  # Duration if applicable

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "phase": self.phase.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "duration_ms": self.duration_ms,
        }

    def to_slack_text(self) -> str:
        """Format for Slack display."""
        emoji = PHASE_EMOJIS.get(self.phase, ":gear:")
        return f"{emoji} {self.message}"


@dataclass
class AgentCandidate:
    """
    A potential agent that could handle the request.

    Used to show users why certain agents were considered and which one was selected.
    """

    agent_id: str
    agent_role: str
    confidence: float  # 0.0 to 1.0
    capabilities: list[str]  # Matching capabilities
    match_reasons: list[str]  # Why this agent matched
    selected: bool = False  # Whether this was the chosen agent
    mcp_servers: list[str] = field(default_factory=list)  # Available MCP servers

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "confidence": self.confidence,
            "capabilities": self.capabilities,
            "match_reasons": self.match_reasons,
            "selected": self.selected,
            "mcp_servers": self.mcp_servers,
        }


@dataclass
class QueryEnhancement:
    """
    Record of query enhancement by LLM.

    Tracks how the original query was refined or clarified.
    """

    original_query: str
    enhanced_query: str
    enhancements: list[str]  # List of changes made
    entities_extracted: dict[str, Any] = field(default_factory=dict)
    clarifications: list[str] = field(default_factory=list)  # Questions resolved

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "original_query": self.original_query,
            "enhanced_query": self.enhanced_query,
            "enhancements": self.enhancements,
            "entities_extracted": self.entities_extracted,
            "clarifications": self.clarifications,
        }


# Emoji mapping for Slack display
PHASE_EMOJIS = {
    ThinkingPhase.RECEIVED: ":inbox_tray:",
    ThinkingPhase.ENHANCING: ":brain:",
    ThinkingPhase.ENHANCED: ":white_check_mark:",
    ThinkingPhase.CLASSIFYING: ":mag:",
    ThinkingPhase.CLASSIFIED: ":dart:",
    ThinkingPhase.DISCOVERING: ":busts_in_silhouette:",
    ThinkingPhase.DISCOVERED: ":raising_hand:",
    ThinkingPhase.ROUTING: ":railway_track:",
    ThinkingPhase.ROUTED: ":round_pushpin:",
    ThinkingPhase.DELEGATING: ":handshake:",
    ThinkingPhase.EXECUTING: ":gear:",
    ThinkingPhase.TOOL_CALL: ":wrench:",
    ThinkingPhase.TOOL_RESULT: ":package:",
    ThinkingPhase.SYNTHESIZING: ":sparkles:",
    ThinkingPhase.COMPLETE: ":white_check_mark:",
    ThinkingPhase.ERROR: ":x:",
}


@dataclass
class ThinkingTrace:
    """
    Manager for the coordinator's thinking trace.

    Provides methods to add steps, format for display, and track timing.
    Uses dataclass for JSON serialization compatibility with LangGraph checkpointing.
    """

    steps: list[ThinkingStep] = field(default_factory=list)
    agent_candidates: list[AgentCandidate] = field(default_factory=list)
    query_enhancement: QueryEnhancement | None = None
    _start_time: datetime | None = field(default=None, repr=False)

    def start(self, query: str) -> ThinkingStep:
        """Start the trace with the received query."""
        self._start_time = datetime.now()
        step = ThinkingStep(
            phase=ThinkingPhase.RECEIVED,
            message=f"Received query: \"{query[:100]}{'...' if len(query) > 100 else ''}\"",
            data={"query": query},
        )
        self.steps.append(step)
        return step

    def add_step(
        self,
        phase: ThinkingPhase,
        message: str,
        data: dict[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> ThinkingStep:
        """Add a thinking step."""
        step = ThinkingStep(
            phase=phase,
            message=message,
            data=data or {},
            duration_ms=duration_ms,
        )
        self.steps.append(step)
        return step

    def set_query_enhancement(
        self,
        original: str,
        enhanced: str,
        enhancements: list[str],
        entities: dict[str, Any] | None = None,
    ) -> None:
        """Record query enhancement."""
        self.query_enhancement = QueryEnhancement(
            original_query=original,
            enhanced_query=enhanced,
            enhancements=enhancements,
            entities_extracted=entities or {},
        )
        self.add_step(
            ThinkingPhase.ENHANCED,
            f"Query enhanced: {', '.join(enhancements[:3])}",
            data={"enhancements": enhancements},
        )

    def add_agent_candidate(self, candidate: AgentCandidate) -> None:
        """Add an agent candidate."""
        self.agent_candidates.append(candidate)

    def select_agent(self, agent_id: str) -> None:
        """Mark an agent as selected."""
        for candidate in self.agent_candidates:
            candidate.selected = candidate.agent_id == agent_id

    def complete(self, final_message: str = "Processing complete") -> None:
        """Mark the trace as complete."""
        if self._start_time:
            duration_ms = int((datetime.now() - self._start_time).total_seconds() * 1000)
        else:
            duration_ms = None

        self.add_step(
            ThinkingPhase.COMPLETE,
            final_message,
            duration_ms=duration_ms,
        )

    def error(self, error_message: str) -> None:
        """Record an error."""
        self.add_step(
            ThinkingPhase.ERROR,
            f"Error: {error_message}",
            data={"error": error_message},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full trace."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "agent_candidates": [c.to_dict() for c in self.agent_candidates],
            "query_enhancement": self.query_enhancement.to_dict()
            if self.query_enhancement
            else None,
            "_start_time": self._start_time.isoformat() if self._start_time else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThinkingTrace:
        """Restore from serialized dict (for JSON-based serialization)."""
        from datetime import datetime
        trace = cls()
        trace._start_time = (
            datetime.fromisoformat(data["_start_time"])
            if data.get("_start_time")
            else None
        )
        # Steps and candidates reconstructed minimally for checkpoint restore
        return trace

    def to_slack_blocks(self, include_details: bool = False) -> list[dict]:
        """
        Format thinking trace as Slack Block Kit.

        Args:
            include_details: Whether to include detailed data for each step

        Returns:
            List of Slack block dictionaries
        """
        blocks = []

        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":brain: Thinking Process",
                "emoji": True,
            }
        })

        # Steps as a single section with bullets
        if self.steps:
            step_lines = []
            for step in self.steps[-10:]:  # Last 10 steps max
                step_lines.append(step.to_slack_text())

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(step_lines),
                }
            })

        # Agent candidates (if any selected)
        selected = [c for c in self.agent_candidates if c.selected]
        if selected:
            agent = selected[0]
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": (
                        f":robot_face: *Selected:* {agent.agent_role} "
                        f"(confidence: {agent.confidence:.0%}) | "
                        f"Matched: {', '.join(agent.match_reasons[:2])}"
                    ),
                }]
            })

        return blocks

    def to_compact_summary(self) -> str:
        """
        Generate a compact one-line summary of the thinking process.

        Returns:
            Short summary string
        """
        if not self.steps:
            return "No processing steps recorded"

        # Get key decisions
        parts = []

        # Classification
        classified = [s for s in self.steps if s.phase == ThinkingPhase.CLASSIFIED]
        if classified:
            intent = classified[0].data.get("intent", "unknown")
            parts.append(f"Intent: {intent}")

        # Selected agent
        selected = [c for c in self.agent_candidates if c.selected]
        if selected:
            parts.append(f"Agent: {selected[0].agent_role}")

        # Tool calls
        tool_calls = [s for s in self.steps if s.phase == ThinkingPhase.TOOL_CALL]
        if tool_calls:
            parts.append(f"Tools: {len(tool_calls)}")

        # Duration
        complete = [s for s in self.steps if s.phase == ThinkingPhase.COMPLETE]
        if complete and complete[0].duration_ms:
            parts.append(f"Time: {complete[0].duration_ms}ms")

        return " | ".join(parts) if parts else "Processing complete"


def create_thinking_trace() -> ThinkingTrace:
    """Factory function to create a new thinking trace."""
    return ThinkingTrace()
