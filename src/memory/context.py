"""
Context Manager for Conversation Compression.

Handles long conversations by summarizing older messages while
preserving recent context. This prevents token limit issues
and maintains conversation coherence.

Usage:
    from src.memory.context import ContextManager

    ctx_manager = ContextManager(memory=shared_memory, llm=llm)

    # Get compressed context for a thread
    context = await ctx_manager.get_context(thread_id)

    # Check if compression is needed
    if await ctx_manager.needs_compression(thread_id):
        await ctx_manager.compress(thread_id)
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


@dataclass
class ContextSummary:
    """Summary of compressed conversation context."""

    thread_id: str
    summary: str
    key_entities: dict[str, Any]
    key_decisions: list[str]
    active_tasks: list[str]
    message_count_compressed: int
    created_at: datetime


@dataclass
class ContextWindow:
    """A window of conversation context."""

    thread_id: str
    summary: str | None  # Summary of older messages
    recent_messages: list[dict[str, Any]]  # Recent messages (uncompressed)
    total_messages: int
    estimated_tokens: int
    is_compressed: bool


class ContextManager:
    """
    Manages conversation context with automatic compression.

    Features:
    - Token estimation for context windows
    - Automatic summarization when context grows too large
    - Preservation of recent messages for accuracy
    - Extraction of key entities and decisions
    - Caching of summaries for efficiency
    """

    # Token thresholds
    MAX_CONTEXT_TOKENS = 150_000  # Trigger compression before 200k limit
    RECENT_MESSAGES_TO_KEEP = 10  # Keep last N messages uncompressed
    SUMMARY_CACHE_TTL = timedelta(hours=2)

    # Approximate tokens per character (varies by model)
    TOKENS_PER_CHAR = 0.25

    def __init__(
        self,
        memory: SharedMemoryManager,
        llm: BaseChatModel | None = None,
        max_tokens: int = MAX_CONTEXT_TOKENS,
        recent_messages_count: int = RECENT_MESSAGES_TO_KEEP,
    ):
        """
        Initialize context manager.

        Args:
            memory: Shared memory manager for storing conversations
            llm: LLM for generating summaries (optional, uses heuristics if not provided)
            max_tokens: Maximum tokens before compression
            recent_messages_count: Number of recent messages to keep uncompressed
        """
        self.memory = memory
        self.llm = llm
        self.max_tokens = max_tokens
        self.recent_count = recent_messages_count
        self._summary_cache: dict[str, ContextSummary] = {}
        self._logger = logger.bind(component="ContextManager")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.TOKENS_PER_CHAR)

    def _estimate_message_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total tokens for a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
            elif isinstance(content, dict):
                total += self._estimate_tokens(json.dumps(content))
            # Add overhead for role, metadata
            total += 10
        return total

    async def get_context(
        self,
        thread_id: str,
        include_summary: bool = True,
    ) -> ContextWindow:
        """
        Get the context window for a thread.

        Returns recent messages plus a summary of older messages
        if the conversation is long.

        Args:
            thread_id: Thread identifier
            include_summary: Whether to include summary of older messages

        Returns:
            ContextWindow with context data
        """
        # Get full conversation history with a short timeout to avoid blocking on persistent stores.
        timeout_s = float(os.getenv("CONTEXT_HISTORY_TIMEOUT_SECONDS", "3"))
        try:
            history = await asyncio.wait_for(
                self.memory.get_conversation_history(thread_id),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            self._logger.warning(
                "Conversation history load timed out",
                thread_id=thread_id,
                timeout_s=timeout_s,
            )
            history = []
        except Exception as e:
            self._logger.warning(
                "Conversation history load failed",
                thread_id=thread_id,
                error=str(e),
            )
            history = []
        history = history or []

        if not history:
            return ContextWindow(
                thread_id=thread_id,
                summary=None,
                recent_messages=[],
                total_messages=0,
                estimated_tokens=0,
                is_compressed=False,
            )

        total_tokens = self._estimate_message_tokens(history)

        # Check if compression is needed
        if total_tokens <= self.max_tokens:
            return ContextWindow(
                thread_id=thread_id,
                summary=None,
                recent_messages=history,
                total_messages=len(history),
                estimated_tokens=total_tokens,
                is_compressed=False,
            )

        # Need compression - split into old and recent
        recent_messages = history[-self.recent_count:]
        old_messages = history[:-self.recent_count]

        # Get or create summary
        summary = None
        if include_summary and old_messages:
            summary = await self._get_or_create_summary(thread_id, old_messages)

        # Calculate tokens for compressed window
        recent_tokens = self._estimate_message_tokens(recent_messages)
        summary_tokens = self._estimate_tokens(summary) if summary else 0

        return ContextWindow(
            thread_id=thread_id,
            summary=summary,
            recent_messages=recent_messages,
            total_messages=len(history),
            estimated_tokens=recent_tokens + summary_tokens,
            is_compressed=True,
        )

    async def needs_compression(self, thread_id: str) -> bool:
        """
        Check if a thread needs compression.

        Args:
            thread_id: Thread identifier

        Returns:
            True if compression is recommended
        """
        history = await self.memory.get_conversation_history(thread_id) or []
        tokens = self._estimate_message_tokens(history)
        return tokens > self.max_tokens

    async def compress(self, thread_id: str) -> ContextSummary | None:
        """
        Compress a thread's context by summarizing older messages.

        Args:
            thread_id: Thread identifier

        Returns:
            ContextSummary or None if no compression needed
        """
        history = await self.memory.get_conversation_history(thread_id) or []

        if len(history) <= self.recent_count:
            return None

        old_messages = history[:-self.recent_count]
        summary = await self._create_summary(thread_id, old_messages)

        # Cache the summary
        self._summary_cache[thread_id] = summary

        # Store in persistent memory
        await self.memory.set_agent_memory(
            agent_id="context-manager",
            memory_type=f"summary:{thread_id}",
            value=summary.__dict__,
        )

        self._logger.info(
            "Context compressed",
            thread_id=thread_id,
            messages_compressed=len(old_messages),
            summary_tokens=self._estimate_tokens(summary.summary),
        )

        return summary

    async def _get_or_create_summary(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """Get cached summary or create new one."""
        # Check cache
        if thread_id in self._summary_cache:
            cached = self._summary_cache[thread_id]
            if cached.message_count_compressed == len(messages):
                return cached.summary

        # Check persistent storage
        stored = await self.memory.get_agent_memory(
            agent_id="context-manager",
            memory_type=f"summary:{thread_id}",
        )
        if stored and stored.get("message_count_compressed") == len(messages):
            return stored["summary"]

        # Create new summary
        summary = await self._create_summary(thread_id, messages)
        self._summary_cache[thread_id] = summary

        return summary.summary

    async def _create_summary(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
    ) -> ContextSummary:
        """
        Create a summary of messages.

        Uses LLM if available, otherwise uses heuristic extraction.
        """
        if self.llm:
            return await self._create_llm_summary(thread_id, messages)
        else:
            return self._create_heuristic_summary(thread_id, messages)

    async def _create_llm_summary(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
    ) -> ContextSummary:
        """Create summary using LLM."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Format messages for summarization
        formatted = self._format_messages_for_summary(messages)

        prompt = f"""Summarize the following conversation history, extracting:
1. A concise summary of what was discussed (2-3 paragraphs max)
2. Key entities mentioned (resources, databases, users, etc.)
3. Important decisions or conclusions reached
4. Any active or pending tasks

Conversation History:
{formatted}

Provide a structured summary that captures the essential context needed to continue the conversation."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a conversation summarizer. Create concise, informative summaries that preserve essential context."),
                HumanMessage(content=prompt),
            ])

            summary_text = response.content

            # Extract structured data from summary
            key_entities = self._extract_entities(messages)
            key_decisions = self._extract_decisions(summary_text)
            active_tasks = self._extract_tasks(summary_text)

            return ContextSummary(
                thread_id=thread_id,
                summary=summary_text,
                key_entities=key_entities,
                key_decisions=key_decisions,
                active_tasks=active_tasks,
                message_count_compressed=len(messages),
                created_at=datetime.utcnow(),
            )

        except Exception as e:
            self._logger.error("LLM summary failed", error=str(e))
            return self._create_heuristic_summary(thread_id, messages)

    def _create_heuristic_summary(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
    ) -> ContextSummary:
        """Create summary using heuristics (no LLM)."""
        # Extract key information heuristically
        key_entities = self._extract_entities(messages)

        # Get first and last messages
        first_msg = messages[0] if messages else {}
        last_msg = messages[-1] if messages else {}

        # Create basic summary
        summary_parts = []

        if first_msg:
            summary_parts.append(
                f"Conversation started with: {first_msg.get('content', '')[:200]}..."
            )

        summary_parts.append(f"Total messages in context: {len(messages)}")

        if key_entities:
            entities_str = ", ".join(
                f"{k}: {v}" for k, v in list(key_entities.items())[:5]
            )
            summary_parts.append(f"Key entities mentioned: {entities_str}")

        if last_msg:
            summary_parts.append(
                f"Last discussed: {last_msg.get('content', '')[:200]}..."
            )

        return ContextSummary(
            thread_id=thread_id,
            summary="\n".join(summary_parts),
            key_entities=key_entities,
            key_decisions=[],
            active_tasks=[],
            message_count_compressed=len(messages),
            created_at=datetime.utcnow(),
        )

    def _format_messages_for_summary(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Format messages for LLM summarization."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content, indent=2)
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"[{role}]: {content}")

        return "\n".join(formatted)

    def _extract_entities(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract key entities from messages."""
        entities = {
            "ocids": [],
            "compartments": [],
            "databases": [],
            "instances": [],
            "users": [],
        }

        for msg in messages:
            content = str(msg.get("content", ""))

            # Extract OCIDs
            import re
            ocids = re.findall(r"ocid1\.[a-z]+\.[a-z0-9]+\.[a-z0-9-]+\.[a-z0-9]+", content)
            entities["ocids"].extend(ocids)

            # Extract compartment names
            comp_matches = re.findall(r"compartment[:\s]+([A-Za-z0-9_-]+)", content, re.I)
            entities["compartments"].extend(comp_matches)

            # Extract database names
            db_matches = re.findall(r"database[:\s]+([A-Za-z0-9_-]+)", content, re.I)
            entities["databases"].extend(db_matches)

        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]  # Limit to 10 each

        # Remove empty lists
        return {k: v for k, v in entities.items() if v}

    def _extract_decisions(self, summary: str) -> list[str]:
        """Extract key decisions from summary text."""
        decisions = []
        decision_keywords = [
            "decided to", "will ", "agreed to", "concluded that",
            "recommendation:", "action:", "resolved to",
        ]

        lines = summary.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in decision_keywords):
                decisions.append(line.strip())

        return decisions[:5]  # Limit to 5

    def _extract_tasks(self, summary: str) -> list[str]:
        """Extract active tasks from summary text."""
        tasks = []
        task_keywords = [
            "todo:", "task:", "pending:", "next step:",
            "need to", "should ", "must ",
        ]

        lines = summary.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in task_keywords):
                tasks.append(line.strip())

        return tasks[:5]  # Limit to 5

    def format_context_for_prompt(
        self,
        context_window: ContextWindow,
    ) -> str:
        """
        Format context window for inclusion in LLM prompt.

        Args:
            context_window: Context window to format

        Returns:
            Formatted string for prompt
        """
        parts = []

        if context_window.is_compressed and context_window.summary:
            parts.append("=== CONVERSATION SUMMARY (older messages) ===")
            parts.append(context_window.summary)
            parts.append("")

        if context_window.recent_messages:
            parts.append("=== RECENT MESSAGES ===")
            for msg in context_window.recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"[{role}]: {content}")

        return "\n".join(parts)

    def clear_cache(self, thread_id: str | None = None) -> None:
        """
        Clear summary cache.

        Args:
            thread_id: Specific thread or None for all
        """
        if thread_id:
            if thread_id in self._summary_cache:
                del self._summary_cache[thread_id]
        else:
            self._summary_cache.clear()
