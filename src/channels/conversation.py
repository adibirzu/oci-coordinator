"""
Conversation Memory for Slack Integration.

Tracks conversation history per thread to provide context-aware
responses and seamless multi-turn conversations.

Key Features:
- Thread-based conversation tracking
- Context compression for long conversations
- Topic detection for better routing
- Session state management
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Configuration
DEFAULT_LOCAL_CACHE_TTL = 3600  # 1 hour in seconds
DEFAULT_LOCAL_CACHE_MAX_SIZE = 1000  # Maximum threads to cache locally


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for a conversation thread."""
    thread_id: str
    channel_id: str
    user_id: str
    messages: list[ConversationMessage] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)  # Detected topics
    last_query_type: str | None = None  # Last workflow/agent used
    last_response_time: float | None = None
    session_start: float = field(default_factory=time.time)


@dataclass
class _CacheEntry:
    """Internal cache entry with TTL tracking."""
    context: ConversationContext
    expires_at: float


class ConversationManager:
    """
    Manages conversation state and history for Slack threads.

    Uses SharedMemoryManager for persistence and provides
    conversation-aware context for the coordinator.

    Thread-safe with asyncio.Lock and TTL-based local cache eviction.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_history: int = 20,
        context_window: int = 5,
        local_cache_ttl: int = DEFAULT_LOCAL_CACHE_TTL,
        local_cache_max_size: int = DEFAULT_LOCAL_CACHE_MAX_SIZE,
    ):
        """
        Initialize conversation manager.

        Args:
            redis_url: Redis URL for persistence (uses env if not provided)
            max_history: Maximum messages to keep per thread
            context_window: Recent messages to include in context
            local_cache_ttl: TTL for local cache entries in seconds
            local_cache_max_size: Maximum entries in local cache
        """
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._max_history = max_history
        self._context_window = context_window
        self._local_cache_ttl = local_cache_ttl
        self._local_cache_max_size = local_cache_max_size
        self._memory = None
        self._local_cache: dict[str, _CacheEntry] = {}
        self._cache_lock = asyncio.Lock()  # Thread-safe cache access

    async def _get_memory(self):
        """Get or create memory manager."""
        if self._memory is None:
            from src.memory.manager import SharedMemoryManager
            self._memory = SharedMemoryManager(redis_url=self._redis_url)
        return self._memory

    def _get_cache_key(self, thread_id: str) -> str:
        """Get Redis cache key for thread."""
        return f"conversation:{thread_id}"

    def _evict_expired_entries(self) -> None:
        """Evict expired entries from local cache (called within lock)."""
        now = time.time()
        expired = [k for k, v in self._local_cache.items() if v.expires_at < now]
        for key in expired:
            del self._local_cache[key]

        # Also evict oldest entries if cache is too large
        if len(self._local_cache) > self._local_cache_max_size:
            # Sort by expires_at and remove oldest 10%
            sorted_entries = sorted(
                self._local_cache.items(),
                key=lambda x: x[1].expires_at
            )
            to_remove = len(self._local_cache) - int(self._local_cache_max_size * 0.9)
            for key, _ in sorted_entries[:to_remove]:
                del self._local_cache[key]

    async def get_context(
        self,
        thread_id: str,
        channel_id: str,
        user_id: str,
    ) -> ConversationContext:
        """
        Get or create conversation context for a thread.

        Thread-safe with asyncio.Lock to prevent race conditions.

        Args:
            thread_id: Slack thread timestamp
            channel_id: Slack channel ID
            user_id: User ID

        Returns:
            ConversationContext for the thread
        """
        async with self._cache_lock:
            # Evict expired entries periodically
            self._evict_expired_entries()

            # Check local cache first (with TTL check)
            now = time.time()
            if thread_id in self._local_cache:
                entry = self._local_cache[thread_id]
                if entry.expires_at > now:
                    return entry.context
                else:
                    # Expired, remove from cache
                    del self._local_cache[thread_id]

            # Try to load from Redis
            try:
                memory = await self._get_memory()
                data = await memory.cache.get(self._get_cache_key(thread_id))

                if data:
                    context = ConversationContext(
                        thread_id=thread_id,
                        channel_id=data.get("channel_id", channel_id),
                        user_id=data.get("user_id", user_id),
                        messages=[
                            ConversationMessage(**msg)
                            for msg in data.get("messages", [])
                        ],
                        topics=data.get("topics", []),
                        last_query_type=data.get("last_query_type"),
                        last_response_time=data.get("last_response_time"),
                        session_start=data.get("session_start", time.time()),
                    )
                    self._local_cache[thread_id] = _CacheEntry(
                        context=context,
                        expires_at=now + self._local_cache_ttl,
                    )
                    return context
            except Exception as e:
                logger.exception("Failed to load conversation from Redis", error=str(e))

            # Create new context
            context = ConversationContext(
                thread_id=thread_id,
                channel_id=channel_id,
                user_id=user_id,
            )
            self._local_cache[thread_id] = _CacheEntry(
                context=context,
                expires_at=now + self._local_cache_ttl,
            )
            return context

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a message to the conversation history.

        Thread-safe with asyncio.Lock.

        Args:
            thread_id: Slack thread timestamp
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (agent_id, query_type, etc.)
        """
        async with self._cache_lock:
            if thread_id not in self._local_cache:
                logger.warning("Adding message to unknown thread", thread_id=thread_id)
                return

            entry = self._local_cache[thread_id]
            context = entry.context
            message = ConversationMessage(
                role=role,
                content=content,
                metadata=metadata or {},
            )
            context.messages.append(message)

            # Update topics based on content
            topics = self._detect_topics(content)
            for topic in topics:
                if topic not in context.topics:
                    context.topics.append(topic)

            # Trim history if needed
            if len(context.messages) > self._max_history:
                context.messages = context.messages[-self._max_history:]

            # Refresh TTL
            entry.expires_at = time.time() + self._local_cache_ttl

        # Persist to Redis (outside lock to avoid blocking)
        await self._persist_context(context)

    async def update_query_type(
        self,
        thread_id: str,
        query_type: str,
    ) -> None:
        """Update the last query type for follow-up suggestions."""
        async with self._cache_lock:
            if thread_id not in self._local_cache:
                return
            entry = self._local_cache[thread_id]
            context = entry.context
            context.last_query_type = query_type
            context.last_response_time = time.time()
            entry.expires_at = time.time() + self._local_cache_ttl

        await self._persist_context(context)

    async def _persist_context(self, context: ConversationContext) -> None:
        """Persist context to Redis."""
        try:
            memory = await self._get_memory()
            data = {
                "channel_id": context.channel_id,
                "user_id": context.user_id,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata,
                    }
                    for msg in context.messages
                ],
                "topics": context.topics,
                "last_query_type": context.last_query_type,
                "last_response_time": context.last_response_time,
                "session_start": context.session_start,
            }
            await memory.cache.set(
                self._get_cache_key(context.thread_id),
                data,
                ttl=timedelta(hours=24),  # Keep conversations for 24 hours
            )
        except Exception as e:
            logger.warning("Failed to persist conversation", error=str(e))

    def _detect_topics(self, content: str) -> list[str]:
        """Detect topics from message content."""
        content_lower = content.lower()
        topics = []

        topic_keywords = {
            "database": ["database", "db", "awr", "sql", "query", "autonomous"],
            "cost": ["cost", "spend", "budget", "finops", "billing", "price"],
            "security": ["security", "threat", "iam", "policy", "audit", "compliance"],
            "compute": ["instance", "vm", "compute", "shape", "memory", "cpu"],
            "network": ["network", "vcn", "subnet", "load balancer", "route"],
            "storage": ["storage", "bucket", "object", "block", "volume"],
            "logs": ["log", "error", "audit", "trace", "exception"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                topics.append(topic)

        return topics

    def get_recent_context(
        self,
        thread_id: str,
        max_messages: int | None = None,
    ) -> str:
        """
        Get formatted recent conversation context for LLM prompt.

        Note: This is a synchronous method for convenience.
        Cache reads are atomic for dict access in Python.

        Args:
            thread_id: Thread ID
            max_messages: Max messages to include (defaults to context_window)

        Returns:
            Formatted conversation history string
        """
        entry = self._local_cache.get(thread_id)
        if not entry:
            return ""

        context = entry.context
        limit = max_messages or self._context_window
        recent = context.messages[-limit:]

        if not recent:
            return ""

        lines = ["Previous conversation:"]
        for msg in recent:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def get_conversation_summary(self, thread_id: str) -> dict[str, Any]:
        """
        Get a summary of the conversation for analytics.

        Note: This is a synchronous method for convenience.
        Cache reads are atomic for dict access in Python.

        Args:
            thread_id: Thread ID

        Returns:
            Summary dict with message count, topics, duration
        """
        entry = self._local_cache.get(thread_id)
        if not entry:
            return {"exists": False}

        context = entry.context
        now = time.time()

        return {
            "exists": True,
            "message_count": len(context.messages),
            "user_messages": sum(1 for m in context.messages if m.role == "user"),
            "topics": context.topics,
            "duration_seconds": now - context.session_start,
            "last_query_type": context.last_query_type,
        }

    async def clear_context(self, thread_id: str) -> None:
        """Clear conversation context for a thread (thread-safe)."""
        async with self._cache_lock:
            if thread_id in self._local_cache:
                del self._local_cache[thread_id]

        try:
            memory = await self._get_memory()
            await memory.cache.delete(self._get_cache_key(thread_id))
        except Exception as e:
            logger.exception("Failed to clear conversation from Redis", error=str(e))


# Singleton instance
_conversation_manager: ConversationManager | None = None


def get_conversation_manager() -> ConversationManager:
    """Get the singleton ConversationManager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
