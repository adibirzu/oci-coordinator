"""
A2A (Agent-to-Agent) Communication Protocol.

Implements structured message passing between agents following
best practices from Anthropic and A2A protocol specifications.

Features:
- Schema-validated messages
- Clear task boundaries
- Explicit output formats
- Context sharing
- Result aggregation

Usage:
    from src.agents.protocol import AgentMessage, AgentResult, MessageBus

    # Create message
    message = AgentMessage(
        sender="coordinator",
        recipient="db-troubleshoot-agent",
        task_id="task-123",
        intent="analyze_performance",
        payload={"database_id": "ocid1..."},
        boundaries=["focus on CPU usage", "check last 24 hours"],
    )

    # Send via message bus
    result = await message_bus.send(message)
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class MessagePriority(str, Enum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageStatus(str, Enum):
    """Message processing status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AgentMessage(BaseModel):
    """
    Structured message for agent-to-agent communication.

    Based on A2A protocol best practices:
    - Explicit sender/recipient for routing
    - Task boundaries prevent work duplication
    - Output format ensures consistent responses
    - Shared context enables collaboration
    """

    # Routing
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = Field(..., description="Sending agent ID")
    recipient: str = Field(..., description="Target agent ID or 'broadcast'")

    # Task specification
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent: str = Field(..., description="What needs to be done")
    payload: dict[str, Any] = Field(default_factory=dict, description="Input data")

    # Task boundaries (prevents work duplication)
    boundaries: list[str] = Field(
        default_factory=list,
        description="Explicit scope boundaries for the task",
    )

    # Output specification
    output_format: str = Field(
        default="json",
        description="Expected response format: json, markdown, table",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for expected output",
    )

    # Shared context
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Shared context from previous agents",
    )
    parent_task_id: str | None = Field(
        default=None,
        description="Parent task for subtask tracking",
    )

    # Metadata
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    timeout_seconds: int = Field(default=120)  # 2 min default, agents can override
    created_at: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)

    @field_validator("boundaries", mode="before")
    @classmethod
    def ensure_boundaries(cls, v):
        """Ensure boundaries is a list."""
        if v is None:
            return []
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        """Deserialize from dictionary."""
        return cls(**data)

    def create_subtask(
        self,
        recipient: str,
        intent: str,
        payload: dict[str, Any],
        boundaries: list[str] | None = None,
    ) -> AgentMessage:
        """Create a subtask message from this message."""
        return AgentMessage(
            sender=self.recipient,  # Current recipient becomes sender
            recipient=recipient,
            task_id=str(uuid.uuid4()),
            parent_task_id=self.task_id,
            intent=intent,
            payload=payload,
            boundaries=boundaries or [],
            context={**self.context, "parent_intent": self.intent},
            output_format=self.output_format,
            priority=self.priority,
            timeout_seconds=self.timeout_seconds,
        )

    def create_response(
        self,
        success: bool,
        result: Any,
        error: str | None = None,
    ) -> AgentResult:
        """Create a response to this message."""
        return AgentResult(
            message_id=self.message_id,
            task_id=self.task_id,
            sender=self.recipient,
            recipient=self.sender,
            success=success,
            result=result,
            error=error,
        )


class AgentResult(BaseModel):
    """
    Result from agent task execution.

    Contains the response data and metadata for
    result aggregation and error handling.
    """

    # Reference to original message
    message_id: str
    task_id: str

    # Routing (reversed from message)
    sender: str  # Agent that produced the result
    recipient: str  # Original sender (coordinator)

    # Result data
    success: bool
    result: Any = Field(default=None)
    error: str | None = Field(default=None)

    # Execution metadata
    execution_time_ms: int | None = Field(default=None)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tokens_used: int | None = Field(default=None)

    # For aggregation
    subtask_results: list[AgentResult] = Field(default_factory=list)

    # Status tracking
    status: MessageStatus = Field(default=MessageStatus.COMPLETED)
    completed_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentResult:
        """Deserialize from dictionary."""
        return cls(**data)

    def aggregate_with(self, other: AgentResult) -> AgentResult:
        """Aggregate this result with another (for parallel tasks)."""
        combined_result = {
            "results": [self.result, other.result],
        }
        return AgentResult(
            message_id=self.message_id,
            task_id=self.task_id,
            sender=self.sender,
            recipient=self.recipient,
            success=self.success and other.success,
            result=combined_result,
            error=self.error or other.error,
            subtask_results=[*self.subtask_results, *other.subtask_results, other],
        )


@dataclass
class TaskSpecification:
    """
    Detailed task specification for complex operations.

    Used by the coordinator to decompose and delegate tasks
    with explicit boundaries and resource allocations.
    """

    task_id: str
    intent: str
    description: str

    # Decomposition
    subtasks: list[TaskSpecification] = field(default_factory=list)
    parallel: bool = False  # Can subtasks run in parallel?

    # Resource allocation
    estimated_tokens: int = 1000
    timeout_seconds: int = 120  # 2 min default for subtasks
    max_tool_calls: int = 5

    # Boundaries
    boundaries: list[str] = field(default_factory=list)
    excluded_domains: list[str] = field(default_factory=list)

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Task IDs

    def to_messages(
        self, sender: str, context: dict[str, Any]
    ) -> list[AgentMessage]:
        """Convert task specification to agent messages."""
        messages = []

        if self.subtasks:
            # Create messages for subtasks
            for subtask in self.subtasks:
                messages.extend(subtask.to_messages(sender, context))
        else:
            # Leaf task - create single message
            messages.append(
                AgentMessage(
                    sender=sender,
                    recipient="auto",  # Will be routed by coordinator
                    task_id=self.task_id,
                    intent=self.intent,
                    payload={"description": self.description},
                    boundaries=self.boundaries,
                    context=context,
                    timeout_seconds=self.timeout_seconds,
                )
            )

        return messages


class MessageBus:
    """
    Central message bus for agent communication.

    Handles:
    - Message routing between agents
    - Priority queuing
    - Timeout management
    - Result aggregation
    """

    def __init__(self):
        self._handlers: dict[str, Callable] = {}
        self._pending: dict[str, AgentMessage] = {}
        self._results: dict[str, AgentResult] = {}
        self._queues: dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }
        self._logger = logger.bind(component="MessageBus")

    def register_handler(
        self,
        agent_id: str,
        handler: Callable[[AgentMessage], AgentResult],
    ) -> None:
        """Register a message handler for an agent."""
        self._handlers[agent_id] = handler
        self._logger.info("Handler registered", agent_id=agent_id)

    def unregister_handler(self, agent_id: str) -> None:
        """Unregister a message handler."""
        if agent_id in self._handlers:
            del self._handlers[agent_id]
            self._logger.info("Handler unregistered", agent_id=agent_id)

    async def send(
        self,
        message: AgentMessage,
        wait_for_result: bool = True,
    ) -> AgentResult | None:
        """
        Send a message to an agent.

        Args:
            message: Message to send
            wait_for_result: If True, wait for response

        Returns:
            AgentResult if wait_for_result, else None
        """
        self._logger.info(
            "Sending message",
            message_id=message.message_id,
            sender=message.sender,
            recipient=message.recipient,
            intent=message.intent,
        )

        # Store pending message
        self._pending[message.message_id] = message

        # Get handler for recipient
        handler = self._handlers.get(message.recipient)
        if not handler:
            self._logger.error(
                "No handler for recipient",
                recipient=message.recipient,
            )
            return AgentResult(
                message_id=message.message_id,
                task_id=message.task_id,
                sender=message.recipient,
                recipient=message.sender,
                success=False,
                error=f"No handler registered for {message.recipient}",
                status=MessageStatus.FAILED,
            )

        if not wait_for_result:
            # Fire and forget - add to queue
            await self._queues[message.priority].put(message)
            return None

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                self._execute_handler(handler, message),
                timeout=message.timeout_seconds,
            )

            # Store result
            self._results[message.message_id] = result

            # Clear pending
            del self._pending[message.message_id]

            return result

        except TimeoutError:
            self._logger.error(
                "Message timeout",
                message_id=message.message_id,
                timeout=message.timeout_seconds,
            )
            result = AgentResult(
                message_id=message.message_id,
                task_id=message.task_id,
                sender=message.recipient,
                recipient=message.sender,
                success=False,
                error=f"Timeout after {message.timeout_seconds}s",
                status=MessageStatus.TIMEOUT,
            )
            self._results[message.message_id] = result
            return result

    async def _execute_handler(
        self,
        handler: Callable,
        message: AgentMessage,
    ) -> AgentResult:
        """Execute a handler with proper error handling."""
        import time

        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(message)
            else:
                result = handler(message)

            execution_time = int((time.time() - start_time) * 1000)

            if isinstance(result, AgentResult):
                result.execution_time_ms = execution_time
                return result
            else:
                return AgentResult(
                    message_id=message.message_id,
                    task_id=message.task_id,
                    sender=message.recipient,
                    recipient=message.sender,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time,
                )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self._logger.error(
                "Handler execution failed",
                message_id=message.message_id,
                error=str(e),
            )
            return AgentResult(
                message_id=message.message_id,
                task_id=message.task_id,
                sender=message.recipient,
                recipient=message.sender,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
                status=MessageStatus.FAILED,
            )

    async def send_parallel(
        self,
        messages: list[AgentMessage],
        max_concurrent: int = 5,
    ) -> list[AgentResult]:
        """
        Send multiple messages in parallel.

        Args:
            messages: Messages to send
            max_concurrent: Maximum concurrent executions

        Returns:
            List of results in same order as messages
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_send(msg: AgentMessage) -> AgentResult:
            async with semaphore:
                result = await self.send(msg, wait_for_result=True)
                return result or AgentResult(
                    message_id=msg.message_id,
                    task_id=msg.task_id,
                    sender=msg.recipient,
                    recipient=msg.sender,
                    success=False,
                    error="No result returned",
                )

        results = await asyncio.gather(
            *[bounded_send(msg) for msg in messages],
            return_exceptions=True,
        )

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    AgentResult(
                        message_id=messages[i].message_id,
                        task_id=messages[i].task_id,
                        sender=messages[i].recipient,
                        recipient=messages[i].sender,
                        success=False,
                        error=str(result),
                        status=MessageStatus.FAILED,
                    )
                )
            else:
                final_results.append(result)

        return final_results

    def get_pending_count(self) -> int:
        """Get count of pending messages."""
        return len(self._pending)

    def get_result(self, message_id: str) -> AgentResult | None:
        """Get cached result for a message."""
        return self._results.get(message_id)

    def clear_results(self) -> None:
        """Clear cached results."""
        self._results.clear()


# Global message bus instance
_message_bus: MessageBus | None = None


def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus


def reset_message_bus() -> None:
    """Reset the global message bus (for testing)."""
    global _message_bus
    _message_bus = None
