"""
DeepSkill Framework - Advanced skills combining MCP tools with code execution.

DeepSkills enable agents to:
1. Execute MCP tools directly
2. Run sandboxed Python code for data processing
3. Chain multiple operations with intelligent error handling
4. Self-test their capabilities without external dependencies

Example usage:
    class DatabaseAnalysisSkill(DeepSkill):
        async def execute(self, context: SkillContext) -> SkillResult:
            # 1. Query database via MCP
            data = await self.call_mcp_tool("oci_database_execute_sql",
                sql="SELECT * FROM v$session WHERE status='ACTIVE'")

            # 2. Process with code execution
            analysis = await self.execute_code('''
                import pandas as pd
                df = pd.DataFrame(data)
                return df.groupby('status').count()
            ''', {'data': data})

            return SkillResult(success=True, data=analysis)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from datetime import datetime
import asyncio
import structlog

logger = structlog.get_logger()

T = TypeVar("T", bound="DeepSkill")


class SkillPriority(Enum):
    """Priority levels for skill execution."""
    CRITICAL = 1    # Use best model, full resources
    HIGH = 2        # Important but can use standard resources
    STANDARD = 3    # Normal priority
    BACKGROUND = 4  # Can be deferred or use minimal resources


class SkillStatus(Enum):
    """Status of skill execution."""
    PENDING = "pending"       # Not yet started
    RUNNING = "running"       # Currently executing
    COMPLETED = "completed"   # Successfully completed
    FAILED = "failed"         # Execution failed
    CANCELLED = "cancelled"   # Cancelled before completion
    TIMEOUT = "timeout"       # Timed out during execution


@dataclass
class SkillContext:
    """Context passed to skill execution."""

    # Input parameters
    parameters: Dict[str, Any]

    # Agent context
    agent_id: str
    agent_type: str

    # Conversation context
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None

    # Execution context
    priority: SkillPriority = SkillPriority.STANDARD
    timeout_seconds: int = 300
    max_retries: int = 3

    # MCP client reference (injected by agent)
    mcp_client: Optional[Any] = None

    # Code executor reference (injected by agent)
    code_executor: Optional[Any] = None

    # Parent skill for chaining
    parent_skill_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SkillResult:
    """Result from skill execution."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Execution metrics
    execution_time_ms: int = 0
    mcp_calls_count: int = 0
    code_executions_count: int = 0
    tokens_used: int = 0

    # For chained skills
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations for retry or fallback
    retry_recommended: bool = False
    fallback_skill: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeepSkillConfig:
    """Configuration for a DeepSkill."""

    # Identity
    skill_id: str
    name: str
    description: str
    version: str = "1.0.0"

    # Capabilities
    required_mcp_tools: List[str] = field(default_factory=list)
    optional_mcp_tools: List[str] = field(default_factory=list)
    requires_code_execution: bool = False

    # Resource requirements
    priority: SkillPriority = SkillPriority.STANDARD
    timeout_seconds: int = 300
    max_retries: int = 3

    # Model requirements
    min_model_tier: str = "haiku"  # haiku, sonnet, opus

    # Agent compatibility
    compatible_agents: List[str] = field(default_factory=list)

    # Self-test configuration
    has_self_test: bool = True
    test_timeout_seconds: int = 60

    # Metadata
    tags: List[str] = field(default_factory=list)


class DeepSkill(ABC):
    """
    Base class for advanced skills with MCP tool integration and code execution.

    DeepSkills are designed to be:
    1. Modular: Each skill handles a specific capability
    2. Self-testing: Skills can verify their own functionality
    3. Agent-native: Testing is done BY agents, not external tools
    4. Error-aware: Built-in retry and fallback mechanisms
    """

    def __init__(self, config: DeepSkillConfig):
        self.config = config
        self._mcp_call_count = 0
        self._code_exec_count = 0
        self._start_time: Optional[datetime] = None

    @property
    def skill_id(self) -> str:
        return self.config.skill_id

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the skill with given context.

        This is the main entry point for skill execution.
        Implementations should use call_mcp_tool() and execute_code()
        for their operations.
        """
        pass

    async def validate_requirements(self, context: SkillContext) -> tuple[bool, str]:
        """
        Validate that all requirements are met before execution.

        Returns (is_valid, error_message).
        """
        # Check MCP client
        if self.config.required_mcp_tools and not context.mcp_client:
            return False, "MCP client required but not provided"

        # Check code executor if needed
        if self.config.requires_code_execution and not context.code_executor:
            return False, "Code executor required but not provided"

        # Check required parameters (subclasses can override)
        missing = self._check_required_parameters(context.parameters)
        if missing:
            return False, f"Missing required parameters: {missing}"

        return True, ""

    def _check_required_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Check for required parameters. Override in subclasses."""
        return []

    async def call_mcp_tool(
        self,
        context: SkillContext,
        tool_name: str,
        **kwargs
    ) -> Any:
        """
        Call an MCP tool through the agent's MCP client.

        This wraps MCP tool calls with:
        - Logging and metrics
        - Error handling
        - Retry logic
        """
        if not context.mcp_client:
            raise RuntimeError("MCP client not available in context")

        self._mcp_call_count += 1

        logger.debug(
            "skill_mcp_call",
            skill_id=self.skill_id,
            tool_name=tool_name,
            call_number=self._mcp_call_count
        )

        try:
            # Call through the MCP client
            result = await context.mcp_client.call_tool(tool_name, kwargs)
            return result
        except Exception as e:
            logger.error(
                "skill_mcp_call_failed",
                skill_id=self.skill_id,
                tool_name=tool_name,
                error=str(e)
            )
            raise

    async def execute_code(
        self,
        context: SkillContext,
        code: str,
        variables: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 30
    ) -> Any:
        """
        Execute Python code in a sandboxed environment.

        Args:
            context: Skill execution context
            code: Python code to execute
            variables: Variables to inject into execution namespace
            timeout_seconds: Maximum execution time

        Returns:
            Result of code execution
        """
        if not context.code_executor:
            raise RuntimeError("Code executor not available in context")

        self._code_exec_count += 1

        logger.debug(
            "skill_code_execution",
            skill_id=self.skill_id,
            code_length=len(code),
            exec_number=self._code_exec_count
        )

        try:
            result = await context.code_executor.execute(
                code=code,
                variables=variables or {},
                timeout_seconds=timeout_seconds
            )
            return result
        except Exception as e:
            logger.error(
                "skill_code_execution_failed",
                skill_id=self.skill_id,
                error=str(e)
            )
            raise

    async def run(self, context: SkillContext) -> SkillResult:
        """
        Run the skill with full lifecycle management.

        This handles:
        1. Requirement validation
        2. Execution with timing
        3. Error handling and retries
        4. Metrics collection
        """
        self._start_time = datetime.utcnow()
        self._mcp_call_count = 0
        self._code_exec_count = 0

        # Validate requirements
        is_valid, error = await self.validate_requirements(context)
        if not is_valid:
            return SkillResult(
                success=False,
                error=error,
                error_type="ValidationError"
            )

        # Execute with retries
        last_error: Optional[Exception] = None
        for attempt in range(context.max_retries):
            try:
                result = await asyncio.wait_for(
                    self.execute(context),
                    timeout=context.timeout_seconds
                )

                # Add metrics
                result.execution_time_ms = int(
                    (datetime.utcnow() - self._start_time).total_seconds() * 1000
                )
                result.mcp_calls_count = self._mcp_call_count
                result.code_executions_count = self._code_exec_count

                return result

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(
                    f"Skill execution timed out after {context.timeout_seconds}s"
                )
                logger.warning(
                    "skill_timeout",
                    skill_id=self.skill_id,
                    attempt=attempt + 1
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "skill_execution_error",
                    skill_id=self.skill_id,
                    attempt=attempt + 1,
                    error=str(e)
                )

                # Check if retry is recommended
                if not self._should_retry(e):
                    break

        # All retries exhausted
        return SkillResult(
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            error_type=type(last_error).__name__ if last_error else "UnknownError",
            execution_time_ms=int(
                (datetime.utcnow() - self._start_time).total_seconds() * 1000
            ),
            mcp_calls_count=self._mcp_call_count,
            code_executions_count=self._code_exec_count,
            retry_recommended=True
        )

    def _should_retry(self, error: Exception) -> bool:
        """Determine if error is retryable."""
        # Transient errors should retry
        retryable_types = (
            asyncio.TimeoutError,
            ConnectionError,
            TimeoutError,
        )
        return isinstance(error, retryable_types)

    # ==================== Self-Testing ====================

    async def self_test(self, context: SkillContext) -> SkillResult:
        """
        Run self-test to verify skill functionality.

        This is called BY agents to verify their skills work correctly.
        NOT meant to be called by external tools like Claude Code.

        Override this method to implement skill-specific tests.
        """
        # Default implementation checks basic requirements
        is_valid, error = await self.validate_requirements(context)
        if not is_valid:
            return SkillResult(
                success=False,
                error=f"Self-test failed: {error}",
                error_type="SelfTestValidationError"
            )

        # Check MCP tool availability
        if self.config.required_mcp_tools:
            for tool_name in self.config.required_mcp_tools:
                try:
                    # Verify tool exists by checking metadata
                    # This doesn't execute the tool, just verifies it's available
                    if context.mcp_client:
                        await context.mcp_client.get_tool_info(tool_name)
                except Exception as e:
                    return SkillResult(
                        success=False,
                        error=f"Required MCP tool '{tool_name}' not available: {e}",
                        error_type="SelfTestMCPError"
                    )

        return SkillResult(
            success=True,
            data={"message": f"Self-test passed for {self.name}"},
            metadata={"tested_at": datetime.utcnow().isoformat()}
        )

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of this skill."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.config.version,
            "status": "healthy",
            "required_tools": self.config.required_mcp_tools,
            "requires_code_execution": self.config.requires_code_execution,
        }


class DeepSkillRegistry:
    """
    Registry for managing DeepSkills.

    The registry allows:
    1. Registration of skills by ID
    2. Discovery of skills by capability
    3. Agent-specific skill filtering
    4. Batch self-testing
    """

    _instance: Optional["DeepSkillRegistry"] = None
    _skills: Dict[str, DeepSkill] = {}
    _agent_skills: Dict[str, List[str]] = {}

    def __new__(cls) -> "DeepSkillRegistry":
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._skills = {}
            cls._agent_skills = {}
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing)."""
        cls._instance = None
        cls._skills = {}
        cls._agent_skills = {}

    def register(self, skill: DeepSkill) -> None:
        """Register a skill in the registry."""
        self._skills[skill.skill_id] = skill

        # Index by compatible agents
        for agent_type in skill.config.compatible_agents:
            if agent_type not in self._agent_skills:
                self._agent_skills[agent_type] = []
            if skill.skill_id not in self._agent_skills[agent_type]:
                self._agent_skills[agent_type].append(skill.skill_id)

        logger.info(
            "skill_registered",
            skill_id=skill.skill_id,
            name=skill.name,
            compatible_agents=skill.config.compatible_agents
        )

    def unregister(self, skill_id: str) -> None:
        """Remove a skill from the registry."""
        if skill_id in self._skills:
            skill = self._skills[skill_id]
            del self._skills[skill_id]

            # Remove from agent index
            for agent_type in skill.config.compatible_agents:
                if agent_type in self._agent_skills:
                    self._agent_skills[agent_type] = [
                        sid for sid in self._agent_skills[agent_type]
                        if sid != skill_id
                    ]

    def get(self, skill_id: str) -> Optional[DeepSkill]:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def get_for_agent(self, agent_type: str) -> List[DeepSkill]:
        """Get all skills compatible with an agent type."""
        skill_ids = self._agent_skills.get(agent_type, [])
        return [self._skills[sid] for sid in skill_ids if sid in self._skills]

    def get_by_mcp_tool(self, tool_name: str) -> List[DeepSkill]:
        """Find skills that use a specific MCP tool."""
        return [
            skill for skill in self._skills.values()
            if tool_name in skill.config.required_mcp_tools
            or tool_name in skill.config.optional_mcp_tools
        ]

    def list_all(self) -> List[DeepSkill]:
        """List all registered skills."""
        return list(self._skills.values())

    async def run_all_self_tests(
        self,
        context: SkillContext,
        agent_type: Optional[str] = None
    ) -> Dict[str, SkillResult]:
        """
        Run self-tests for all skills (or agent-specific skills).

        This is designed to be called BY agents to verify their capabilities.
        """
        skills = (
            self.get_for_agent(agent_type) if agent_type
            else self.list_all()
        )

        results = {}
        for skill in skills:
            if skill.config.has_self_test:
                results[skill.skill_id] = await skill.self_test(context)

        return results

    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status."""
        return {
            "total_skills": len(self._skills),
            "skills_by_agent": {
                agent: len(skills)
                for agent, skills in self._agent_skills.items()
            },
            "skill_ids": list(self._skills.keys()),
        }


# Decorator for easy skill registration
def register_skill(
    skill_id: str,
    name: str,
    description: str,
    compatible_agents: List[str],
    required_mcp_tools: Optional[List[str]] = None,
    requires_code_execution: bool = False,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a DeepSkill class.

    Example:
        @register_skill(
            skill_id="db_blocking_analysis",
            name="Database Blocking Analysis",
            description="Analyze database blocking sessions",
            compatible_agents=["db_troubleshoot"],
            required_mcp_tools=["oci_database_execute_sql"],
            requires_code_execution=True
        )
        class DatabaseBlockingSkill(DeepSkill):
            async def execute(self, context: SkillContext) -> SkillResult:
                ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        config = DeepSkillConfig(
            skill_id=skill_id,
            name=name,
            description=description,
            compatible_agents=compatible_agents,
            required_mcp_tools=required_mcp_tools or [],
            requires_code_execution=requires_code_execution,
            **kwargs
        )

        # Store config on class for later instantiation
        cls._default_config = config

        # Auto-register on import if registry exists
        registry = DeepSkillRegistry()
        instance = cls(config)
        registry.register(instance)

        return cls

    return decorator
