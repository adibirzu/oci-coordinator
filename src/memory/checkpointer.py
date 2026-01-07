"""
Checkpointer factory for LangGraph.

Provides in-memory checkpointing for LangGraph graph state persistence.
For production persistence, consider implementing a database-backed
checkpointer (e.g., PostgreSQL, Redis).

Usage:
    from src.memory.checkpointer import create_checkpointer

    # Create checkpointer
    checkpointer = await create_checkpointer()

    # Use with LangGraph
    graph = StateGraph(MyState)
    compiled = graph.compile(checkpointer=checkpointer)
"""

from __future__ import annotations

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = structlog.get_logger(__name__)


async def create_checkpointer(
    redis_url: str | None = None,
) -> BaseCheckpointSaver:
    """
    Factory to create checkpointer for LangGraph state persistence.

    Currently uses in-memory checkpointing. For production deployments
    requiring persistence across restarts, implement a database-backed
    checkpointer (PostgreSQL, Redis, etc.).

    Args:
        redis_url: Reserved for future Redis-backed implementation

    Returns:
        Configured checkpointer instance
    """
    # Use LangGraph's built-in MemorySaver
    # Note: State is lost on process restart
    logger.info("Creating in-memory checkpointer for LangGraph")
    return MemorySaver()
