"""
Base Agent implementation for OCI Coordinator.

This module provides the foundational classes for all agents:
- AgentDefinition: Complete agent specification
- BaseAgent: Abstract base class with common functionality
- Supporting dataclasses: AgentStatus, AgentMetadata, KafkaTopics

Enhanced Features:
- Structured response formatting with Slack Block Kit support
- Output format configuration (Slack, Markdown, Teams, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import structlog

if TYPE_CHECKING:
    from langgraph.graph import StateGraph

    from src.agents.protocol import AgentMessage, AgentResult, MessageBus
    from src.agents.skills import SkillExecutionResult, SkillExecutor
    from src.formatting.base import StructuredResponse
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


class AgentStatus(str, Enum):
    """Agent lifecycle status."""

    REGISTERED = "registered"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class AgentMetadata:
    """Agent metadata for versioning and configuration."""

    version: str = "1.0.0"
    namespace: str = "oci-coordinator"
    max_iterations: int = 15
    timeout_seconds: int = 300
    retry_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "max_retries": 3,
            "backoff_multiplier": 2,
            "initial_delay_ms": 1000,
        }
    )


@dataclass
class KafkaTopics:
    """Message topics for event-driven communication."""

    consume: list[str] = field(default_factory=list)
    produce: list[str] = field(default_factory=list)


@dataclass
class AgentDefinition:
    """
    Complete Agent Object Definition.

    This schema defines all attributes required for an agent to be
    registered in the Agent Catalog and orchestrated by the Coordinator.

    Naming Conventions:
    - agent_id: {role}-{uuid-suffix} (e.g., "db-troubleshoot-agent-c5b6cd64b")
    - role: {domain}-{function}-agent (e.g., "db-troubleshoot-agent")
    """

    # Identity
    agent_id: str
    role: str

    # Capabilities & Skills
    capabilities: list[str]
    skills: list[str]

    # Communication
    kafka_topics: KafkaTopics
    health_endpoint: str

    # Metadata
    metadata: AgentMetadata
    description: str

    # Runtime State
    status: AgentStatus = AgentStatus.REGISTERED
    registered_at: datetime | None = None
    last_heartbeat: datetime | None = None

    # MCP Integration
    mcp_tools: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent definition to dictionary."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "capabilities": self.capabilities,
            "skills": self.skills,
            "kafka_topics": {
                "consume": self.kafka_topics.consume,
                "produce": self.kafka_topics.produce,
            },
            "health_endpoint": self.health_endpoint,
            "metadata": {
                "version": self.metadata.version,
                "namespace": self.metadata.namespace,
                "max_iterations": self.metadata.max_iterations,
                "timeout_seconds": self.metadata.timeout_seconds,
                "retry_policy": self.metadata.retry_policy,
            },
            "description": self.description,
            "status": self.status.value,
            "registered_at": (
                self.registered_at.isoformat() if self.registered_at else None
            ),
            "last_heartbeat": (
                self.last_heartbeat.isoformat() if self.last_heartbeat else None
            ),
            "mcp_tools": self.mcp_tools,
            "mcp_servers": self.mcp_servers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentDefinition":
        """Deserialize agent definition from dictionary."""
        kafka_data = data.get("kafka_topics", {})
        metadata_data = data.get("metadata", {})

        return cls(
            agent_id=data["agent_id"],
            role=data["role"],
            capabilities=data.get("capabilities", []),
            skills=data.get("skills", []),
            kafka_topics=KafkaTopics(
                consume=kafka_data.get("consume", []),
                produce=kafka_data.get("produce", []),
            ),
            health_endpoint=data.get("health_endpoint", ""),
            metadata=AgentMetadata(
                version=metadata_data.get("version", "1.0.0"),
                namespace=metadata_data.get("namespace", "oci-coordinator"),
                max_iterations=metadata_data.get("max_iterations", 15),
                timeout_seconds=metadata_data.get("timeout_seconds", 300),
                retry_policy=metadata_data.get(
                    "retry_policy",
                    {
                        "max_retries": 3,
                        "backoff_multiplier": 2,
                        "initial_delay_ms": 1000,
                    },
                ),
            ),
            description=data.get("description", ""),
            status=AgentStatus(data.get("status", "registered")),
            registered_at=(
                datetime.fromisoformat(data["registered_at"])
                if data.get("registered_at")
                else None
            ),
            last_heartbeat=(
                datetime.fromisoformat(data["last_heartbeat"])
                if data.get("last_heartbeat")
                else None
            ),
            mcp_tools=data.get("mcp_tools", []),
            mcp_servers=data.get("mcp_servers", []),
        )


class BaseAgent(ABC):
    """
    Base class for all OCI Agents.

    Provides common functionality:
    - Agent definition and registration
    - Memory access via SharedMemoryManager
    - MCP tool invocation via ToolCatalog
    - Observability integration
    - Structured response formatting (Slack, Markdown, Teams, etc.)

    Usage:
        class MyAgent(BaseAgent):
            @classmethod
            def get_definition(cls) -> AgentDefinition:
                return AgentDefinition(...)

            async def invoke(self, query: str, context: dict = None) -> str:
                # Agent logic
                pass

            def build_graph(self) -> StateGraph:
                # LangGraph construction
                pass

    Formatting Usage:
        # In your agent's invoke method:
        response = self.create_response("Analysis Complete")
        response.add_metrics("Performance", [
            MetricValue(label="CPU", value=45.2, unit="%"),
        ])
        return self.format_response(response)
    """

    # Class-level definition cache
    _definition_cache: ClassVar[dict[type, AgentDefinition]] = {}

    # Default output format
    DEFAULT_OUTPUT_FORMAT = "markdown"

    def __init__(
        self,
        memory_manager: "SharedMemoryManager | None" = None,
        tool_catalog: "ToolCatalog | None" = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize agent with shared resources.

        Args:
            memory_manager: Shared memory for cross-agent state
            tool_catalog: Catalog of available MCP tools
            config: Agent-specific configuration
                - output_format: Output format (markdown, slack, teams, etc.)
        """
        self.memory = memory_manager
        self.tools = tool_catalog
        self.config = config or {}
        self._output_format = self.config.get("output_format", self.DEFAULT_OUTPUT_FORMAT)
        self._logger = logger.bind(
            agent_role=self.get_definition().role,
            agent_id=self.get_definition().agent_id,
        )
        if self.memory and self.tools:
            try:
                self.tools.set_memory_manager(self.memory)
            except Exception as exc:
                self._logger.debug(
                    "Failed to attach memory manager to tool catalog",
                    error=str(exc),
                )

    @classmethod
    @abstractmethod
    def get_definition(cls) -> AgentDefinition:
        """
        Return the agent's definition for catalog registration.

        This method must be implemented by all agent subclasses.
        The definition is cached at class level for efficiency.

        Returns:
            AgentDefinition with all agent metadata
        """
        pass

    @abstractmethod
    async def invoke(self, query: str, context: dict[str, Any] | None = None) -> str:
        """
        Execute the agent's primary function.

        Args:
            query: User query or task description
            context: Additional context (session info, previous results)

        Returns:
            Agent's response as string
        """
        pass

    @abstractmethod
    def build_graph(self) -> "StateGraph":
        """
        Build the LangGraph for this agent.

        Returns:
            Compiled StateGraph ready for execution
        """
        pass

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Invoke an MCP tool through the tool catalog.

        Args:
            tool_name: Name of the MCP tool (e.g., "oci_compute_list_instances")
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found in catalog
        """
        if not self.tools:
            raise RuntimeError("Tool catalog not initialized")

        tool_def = self.tools.get_tool(tool_name)
        if not tool_def:
            raise ValueError(f"Tool not found: {tool_name}")

        self._logger.info("Invoking tool", tool_name=tool_name)
        return await self.tools.execute(tool_name, arguments)

    async def save_memory(self, key: str, value: Any) -> None:
        """
        Save to agent's persistent memory.

        Args:
            key: Memory key (e.g., "last_analysis", "user_preferences")
            value: Value to store (must be JSON-serializable)
        """
        if not self.memory:
            self._logger.warning("Memory manager not initialized, skipping save")
            return

        await self.memory.set_agent_memory(
            self.get_definition().agent_id,
            key,
            value,
        )
        self._logger.debug("Saved to memory", key=key)

    async def load_memory(self, key: str) -> Any | None:
        """
        Load from agent's persistent memory.

        Args:
            key: Memory key to retrieve

        Returns:
            Stored value or None if not found
        """
        if not self.memory:
            self._logger.warning("Memory manager not initialized")
            return None

        return await self.memory.get_agent_memory(
            self.get_definition().agent_id,
            key,
        )

    def get_available_tools(self) -> list[str]:
        """Get list of MCP tools this agent can use."""
        return self.get_definition().mcp_tools

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities."""
        return self.get_definition().capabilities

    def get_skills(self) -> list[str]:
        """Get agent skills/workflows."""
        return self.get_definition().skills

    # ─────────────────────────────────────────────────────────────────────────
    # Inter-Agent Communication
    # ─────────────────────────────────────────────────────────────────────────

    def _get_message_bus(self) -> "MessageBus":
        """Get the global message bus for inter-agent communication."""
        from src.agents.protocol import get_message_bus

        return get_message_bus()

    async def request_agent_assistance(
        self,
        target_agent_id: str,
        query: str,
        context: dict[str, Any] | None = None,
        boundaries: list[str] | None = None,
        timeout_seconds: int = 60,
    ) -> "AgentResult":
        """
        Request assistance from another agent via the message bus.

        Use this when your agent needs specialized knowledge or capabilities
        from another agent. The request is routed through the coordinator's
        message bus with proper tracking and timeout handling.

        Args:
            target_agent_id: ID of the agent to request help from
                (e.g., "db-troubleshoot-agent", "finops-agent")
            query: The question or task for the target agent
            context: Shared context data to pass to the target agent
            boundaries: Task boundaries to prevent scope creep
            timeout_seconds: Maximum time to wait for response

        Returns:
            AgentResult with success status and result data

        Example:
            # From infrastructure agent, request database analysis
            result = await self.request_agent_assistance(
                target_agent_id="db-troubleshoot-agent",
                query="What is the CPU usage for database X?",
                context={"database_id": "ocid1..."},
                boundaries=["Focus only on CPU metrics"],
            )
            if result.success:
                cpu_analysis = result.result
        """
        from src.agents.protocol import AgentMessage, MessagePriority

        message_bus = self._get_message_bus()
        my_id = self.get_definition().agent_id

        self._logger.info(
            "Requesting agent assistance",
            from_agent=my_id,
            target_agent=target_agent_id,
            query_length=len(query),
        )

        message = AgentMessage(
            sender=my_id,
            recipient=target_agent_id,
            intent=query,
            payload=context or {},
            boundaries=boundaries or [],
            context={
                "requesting_agent": my_id,
                "request_type": "assistance",
                **(context or {}),
            },
            priority=MessagePriority.NORMAL,
            timeout_seconds=timeout_seconds,
        )

        result = await message_bus.send(message, wait_for_result=True)

        if result:
            self._logger.info(
                "Agent assistance response received",
                target_agent=target_agent_id,
                success=result.success,
                execution_time_ms=result.execution_time_ms,
            )
        else:
            self._logger.warning(
                "No response from agent assistance request",
                target_agent=target_agent_id,
            )

        return result

    async def delegate_subtask(
        self,
        capability: str,
        query: str,
        context: dict[str, Any] | None = None,
        boundaries: list[str] | None = None,
    ) -> "AgentResult | None":
        """
        Delegate a subtask to an agent with the specified capability.

        This method finds an appropriate agent by capability rather than
        by explicit ID, making it more flexible for task delegation.

        Args:
            capability: Required capability (e.g., "cost-analysis", "log-search")
            query: The subtask to delegate
            context: Context data for the subtask
            boundaries: Scope boundaries for the subtask

        Returns:
            AgentResult if an agent was found and responded, None otherwise

        Example:
            # Delegate cost analysis to whoever has that capability
            result = await self.delegate_subtask(
                capability="cost-analysis",
                query="What are the costs for compartment X?",
                context={"compartment_id": "ocid1..."},
            )
        """
        from src.agents.catalog import AgentCatalog

        # Try to get agent catalog to find agent by capability
        try:
            catalog = AgentCatalog.get_instance()
            agents = catalog.get_by_capability(capability)

            if not agents:
                self._logger.warning(
                    "No agent found for capability",
                    capability=capability,
                )
                return None

            # Use first matching agent
            target_agent = agents[0]
            return await self.request_agent_assistance(
                target_agent_id=target_agent.agent_id,
                query=query,
                context=context,
                boundaries=boundaries,
            )

        except Exception as e:
            self._logger.error(
                "Failed to delegate subtask",
                capability=capability,
                error=str(e),
            )
            return None

    async def broadcast_context(
        self,
        context_key: str,
        context_value: Any,
    ) -> None:
        """
        Share context with all agents via shared memory.

        Use this to propagate discoveries or state that other agents
        might find useful (e.g., resolved compartment IDs, discovered
        resource relationships).

        Args:
            context_key: Key to store the context under
            context_value: Value to share (must be JSON-serializable)

        Example:
            # Share discovered database-to-compartment mapping
            await self.broadcast_context(
                "discovered_mapping",
                {"database_id": "ocid1...", "compartment_name": "Production"},
            )
        """
        if not self.memory:
            self._logger.warning("Memory not available for context broadcast")
            return

        # Store in shared namespace accessible to all agents
        await self.memory.set(
            f"shared_context:{context_key}",
            context_value,
            ttl=3600,  # 1 hour TTL
        )

        self._logger.debug(
            "Context broadcasted",
            key=context_key,
            from_agent=self.get_definition().agent_id,
        )

    async def get_shared_context(self, context_key: str) -> Any | None:
        """
        Retrieve context shared by other agents.

        Args:
            context_key: Key to retrieve

        Returns:
            Shared context value or None if not found
        """
        if not self.memory:
            return None

        return await self.memory.get(f"shared_context:{context_key}")

    def discover_agents(
        self,
        capability: str | None = None,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Discover available agents to request help from.

        Args:
            capability: Filter by capability (e.g., "cost-analysis")
            domain: Filter by domain (e.g., "database")

        Returns:
            List of agent info dicts with id, role, capabilities, skills

        Example:
            # Find all agents
            all_agents = self.discover_agents()

            # Find agents that can do cost analysis
            cost_agents = self.discover_agents(capability="cost-analysis")

            # Find database-related agents
            db_agents = self.discover_agents(domain="database")
        """
        from src.agents.catalog import AgentCatalog

        try:
            catalog = AgentCatalog.get_instance()

            if capability:
                agents = catalog.get_by_capability(capability)
            elif domain:
                agents = catalog.get_by_domain(domain)
            else:
                agents = catalog.list_all()

            # Filter out self
            my_id = self.get_definition().agent_id

            return [
                {
                    "agent_id": agent.agent_id,
                    "role": agent.role,
                    "capabilities": agent.capabilities,
                    "skills": agent.skills,
                    "description": agent.description,
                }
                for agent in agents
                if agent.agent_id != my_id
            ]

        except Exception as e:
            self._logger.error("Failed to discover agents", error=str(e))
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Skill Execution
    # ─────────────────────────────────────────────────────────────────────────

    def create_skill_executor(self) -> "SkillExecutor":
        """
        Create a skill executor for this agent.

        Returns:
            SkillExecutor initialized with this agent's tool catalog
        """
        from src.agents.skills import SkillExecutor

        if not self.tools:
            raise RuntimeError("Tool catalog not initialized for skill execution")

        return SkillExecutor(self.tools)

    async def execute_skill(
        self,
        skill_name: str,
        context: dict[str, Any] | None = None,
        handlers: dict[str, Any] | None = None,
    ) -> "SkillExecutionResult":
        """
        Execute a registered skill.

        Args:
            skill_name: Name of the skill to execute
            context: Context data for the skill execution
            handlers: Optional custom step handlers

        Returns:
            SkillExecutionResult with step results and status
        """
        from src.agents.skills import SkillRegistry

        # Get skill from global registry
        registry = SkillRegistry.get_instance()
        skill = registry.get(skill_name)

        if not skill:
            from src.agents.skills import SkillExecutionResult, SkillStatus

            return SkillExecutionResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED,
                error=f"Skill not found in registry: {skill_name}",
            )

        # Create executor and register skill
        executor = self.create_skill_executor()
        if not executor.register(skill, handlers):
            from src.agents.skills import SkillExecutionResult, SkillStatus

            return SkillExecutionResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED,
                error="Skill registration failed - required tools unavailable",
            )

        self._logger.info("Executing skill", skill=skill_name)
        return await executor.execute(skill_name, context)

    def can_execute_skill(self, skill_name: str) -> tuple[bool, list[str]]:
        """
        Check if this agent can execute a skill.

        Args:
            skill_name: Name of the skill to check

        Returns:
            Tuple of (can_execute, missing_tools)
        """
        from src.agents.skills import SkillRegistry

        registry = SkillRegistry.get_instance()
        skill = registry.get(skill_name)

        if not skill:
            return False, [f"Skill not found: {skill_name}"]

        if not self.tools:
            return False, ["Tool catalog not initialized"]

        return skill.validate_tools(self.tools)

    async def health_check(self) -> bool:
        """
        Perform agent health check.

        Override in subclasses for custom health logic.

        Returns:
            True if agent is healthy
        """
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Response Formatting
    # ─────────────────────────────────────────────────────────────────────────

    def set_output_format(self, format_name: str) -> None:
        """
        Set the output format for this agent.

        Args:
            format_name: Format name (markdown, slack, teams, etc.)
        """
        self._output_format = format_name

    def get_output_format(self) -> str:
        """Get the current output format."""
        return self._output_format

    def create_response(
        self,
        title: str,
        subtitle: str | None = None,
        severity: str | None = None,
        icon: str | None = None,
    ) -> "StructuredResponse":
        """
        Create a structured response.

        Args:
            title: Response title
            subtitle: Optional subtitle
            severity: Severity level (critical, high, medium, low, info, success)
            icon: Icon emoji

        Returns:
            StructuredResponse object for building the response
        """
        from src.formatting.base import (
            ResponseHeader,
            Severity,
            StructuredResponse,
        )

        # Convert severity string to enum
        severity_enum = None
        if severity:
            try:
                severity_enum = Severity(severity.lower())
            except ValueError:
                pass

        header = ResponseHeader(
            title=title,
            subtitle=subtitle,
            icon=icon,
            severity=severity_enum,
            agent_name=self.get_definition().role,
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        return StructuredResponse(header=header)

    def format_response(
        self,
        response: "StructuredResponse",
        format_name: str | None = None,
    ) -> str | dict[str, Any]:
        """
        Format a structured response to the specified format.

        Args:
            response: Structured response to format
            format_name: Override output format (uses agent's default if not specified)

        Returns:
            Formatted response (string for text formats, dict for JSON-based formats)
        """
        from src.formatting.base import FormatterRegistry, OutputFormat

        format_name = format_name or self._output_format

        try:
            output_format = OutputFormat(format_name.lower())
        except ValueError:
            self._logger.warning(
                "Unknown output format, falling back to markdown",
                format_name=format_name,
            )
            output_format = OutputFormat.MARKDOWN

        return FormatterRegistry.format(response, output_format)

    def format_error_response(
        self,
        error: str,
        title: str = "Error",
        format_name: str | None = None,
    ) -> str | dict[str, Any]:
        """
        Format an error response.

        Args:
            error: Error message
            title: Error title
            format_name: Override output format

        Returns:
            Formatted error response
        """
        from src.formatting.base import FormatterRegistry, OutputFormat

        format_name = format_name or self._output_format

        try:
            output_format = OutputFormat(format_name.lower())
        except ValueError:
            output_format = OutputFormat.MARKDOWN

        formatter = FormatterRegistry.get(output_format)
        if formatter:
            return formatter.format_error(error, title)

        return f"## ❌ {title}\n\n{error}"

    def __repr__(self) -> str:
        definition = self.get_definition()
        return f"<{self.__class__.__name__} role={definition.role} status={definition.status.value}>"
