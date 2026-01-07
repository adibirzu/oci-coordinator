"""
Shared Memory module for OCI Coordinator.

Provides tiered storage:
- Redis: Hot cache for session state, tool results
- Neo4j: Persistent storage for conversation history, audit logs (optional)
- FileBasedPlanner: Persistent markdown files for multi-step operations (token optimization)
"""

from src.memory.manager import (
    InMemoryStore,
    MemoryStore,
    Neo4jMemoryStore,
    RedisMemoryStore,
    SharedMemoryManager,
)
from src.memory.planner import (
    ErrorRecord,
    FileBasedPlanner,
    PlanContext,
    PlanPhase,
    ToolOutput,
    create_planner_for_workflow,
    should_use_planner,
)

__all__ = [
    # Manager classes
    "InMemoryStore",
    "MemoryStore",
    "Neo4jMemoryStore",
    "RedisMemoryStore",
    "SharedMemoryManager",
    # File-based planner
    "FileBasedPlanner",
    "PlanContext",
    "PlanPhase",
    "ToolOutput",
    "ErrorRecord",
    "should_use_planner",
    "create_planner_for_workflow",
]
