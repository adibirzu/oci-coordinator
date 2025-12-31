"""
Shared Memory module for OCI Coordinator.

Provides tiered storage:
- Redis: Hot cache for session state, tool results
- OCI ATP: Persistent storage for conversation history, audit logs
"""

from src.memory.atp_config import ATPConfig, create_atp_pool, init_atp_schema
from src.memory.manager import (
    ATPMemoryStore,
    InMemoryStore,
    MemoryStore,
    RedisMemoryStore,
    SharedMemoryManager,
)

__all__ = [
    "SharedMemoryManager",
    "MemoryStore",
    "RedisMemoryStore",
    "ATPMemoryStore",
    "InMemoryStore",
    "ATPConfig",
    "create_atp_pool",
    "init_atp_schema",
]
