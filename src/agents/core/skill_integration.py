"""
DeepSkill Integration for Agents.

This module provides integration between the DeepSkill framework
and existing agent implementations. It enables agents to:

1. Discover compatible DeepSkills
2. Execute DeepSkills using their MCP connections
3. Run self-tests for capability verification
4. Bridge the agent's tool catalog with SkillContext

Usage:
    class MyAgent(BaseAgent, DeepSkillMixin):
        async def invoke(self, query: str, context: dict = None) -> str:
            # Use DeepSkills directly
            result = await self.execute_deep_skill(
                "database_blocking_analysis",
                parameters={"database_id": "..."}
            )
            return result.data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import structlog

from src.agents.core.deep_skills import (
    DeepSkill,
    DeepSkillRegistry,
    SkillContext,
    SkillPriority,
    SkillResult,
)
from src.agents.core.code_executor import CodeExecutor

if TYPE_CHECKING:
    from src.agents.base import BaseAgent
    from src.mcp.catalog import ToolCatalog

logger = structlog.get_logger()


@dataclass
class MCPClientAdapter:
    """
    Adapter that wraps an agent's tool catalog for use with DeepSkills.

    This bridges the gap between:
    - Agent's ToolCatalog (async execute method)
    - DeepSkill's expected MCP client interface (call_tool method)

    The adapter ensures DeepSkills can use MCP tools without coupling
    to specific tool catalog implementations.
    """

    tool_catalog: Any  # ToolCatalog
    _call_history: List[Dict[str, Any]] = field(default_factory=list)

    async def call_tool(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Call an MCP tool through the catalog.

        Args:
            tool_name: Name of the MCP tool
            params: Tool parameters

        Returns:
            Tool execution result
        """
        params = params or {}

        # Record the call for debugging/testing
        self._call_history.append({
            "tool_name": tool_name,
            "params": params,
            "timestamp": datetime.utcnow().isoformat()
        })

        logger.debug(
            "mcp_client_adapter_call",
            tool_name=tool_name,
            param_count=len(params)
        )

        # Use the tool catalog's execute method
        if hasattr(self.tool_catalog, "execute"):
            return await self.tool_catalog.execute(tool_name, params)
        elif hasattr(self.tool_catalog, "call_tool"):
            return await self.tool_catalog.call_tool(tool_name, params)
        else:
            raise RuntimeError(
                f"Tool catalog does not have execute or call_tool method"
            )

    async def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool (for validation).

        Args:
            tool_name: Name of the tool

        Returns:
            Tool metadata if found
        """
        if hasattr(self.tool_catalog, "get_tool"):
            tool = self.tool_catalog.get_tool(tool_name)
            if tool:
                return {
                    "name": tool_name,
                    "available": True,
                    "description": getattr(tool, "description", ""),
                }
        return None

    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of tool calls for debugging."""
        return self._call_history.copy()

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()


class DeepSkillMixin:
    """
    Mixin class that adds DeepSkill capabilities to agents.

    This mixin provides:
    1. DeepSkill discovery and execution
    2. Automatic context creation from agent resources
    3. Self-test execution for capability verification
    4. Integration with agent's existing MCP tools

    Usage:
        class InfrastructureAgent(BaseAgent, DeepSkillMixin):
            def __init__(self, ...):
                super().__init__(...)
                self.init_deep_skills()

            async def invoke(self, query: str, context: dict = None) -> str:
                # Execute a DeepSkill
                result = await self.execute_deep_skill(
                    "instance_health_check",
                    parameters={"compartment_id": "..."}
                )
    """

    # Instance attributes (set by init_deep_skills)
    _deep_skill_registry: Optional[DeepSkillRegistry] = None
    _mcp_adapter: Optional[MCPClientAdapter] = None
    _code_executor: Optional[CodeExecutor] = None
    _deep_skill_initialized: bool = False

    def init_deep_skills(
        self,
        enable_code_execution: bool = True,
        code_execution_timeout: int = 30,
    ) -> None:
        """
        Initialize DeepSkill capabilities for this agent.

        Should be called in the agent's __init__ after calling super().__init__.

        Args:
            enable_code_execution: Whether to enable sandboxed code execution
            code_execution_timeout: Default timeout for code execution
        """
        # Get the registry singleton
        self._deep_skill_registry = DeepSkillRegistry()

        # Create MCP adapter if tool catalog is available
        if hasattr(self, "tools") and self.tools:
            self._mcp_adapter = MCPClientAdapter(tool_catalog=self.tools)

        # Create code executor if enabled
        if enable_code_execution:
            self._code_executor = CodeExecutor(
                timeout_seconds=code_execution_timeout
            )

        self._deep_skill_initialized = True

        logger.info(
            "deep_skills_initialized",
            agent_id=self._get_agent_id(),
            mcp_available=self._mcp_adapter is not None,
            code_execution=enable_code_execution,
            compatible_skills=len(self.get_compatible_deep_skills())
        )

    def _get_agent_id(self) -> str:
        """Get the agent ID from the definition."""
        if hasattr(self, "get_definition"):
            return self.get_definition().agent_id
        return "unknown_agent"

    def _get_agent_type(self) -> str:
        """Get the agent type for skill compatibility matching."""
        if hasattr(self, "get_definition"):
            definition = self.get_definition()
            # Extract agent type from role (e.g., "db-troubleshoot-agent" -> "db_troubleshoot")
            role = definition.role.replace("-agent", "").replace("-", "_")
            return role
        return "unknown"

    def get_compatible_deep_skills(self) -> List[DeepSkill]:
        """
        Get all DeepSkills compatible with this agent.

        Returns:
            List of compatible DeepSkill instances
        """
        if not self._deep_skill_registry:
            return []

        agent_type = self._get_agent_type()
        return self._deep_skill_registry.get_for_agent(agent_type)

    def get_deep_skill(self, skill_id: str) -> Optional[DeepSkill]:
        """
        Get a specific DeepSkill by ID.

        Args:
            skill_id: The skill identifier

        Returns:
            DeepSkill if found and compatible, None otherwise
        """
        if not self._deep_skill_registry:
            return None

        skill = self._deep_skill_registry.get(skill_id)
        if skill:
            # Verify compatibility
            agent_type = self._get_agent_type()
            if agent_type in skill.config.compatible_agents or "coordinator" in skill.config.compatible_agents:
                return skill

        return None

    def create_skill_context(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        priority: SkillPriority = SkillPriority.STANDARD,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> SkillContext:
        """
        Create a SkillContext from this agent's resources.

        Args:
            parameters: Parameters to pass to the skill
            priority: Execution priority
            conversation_id: Optional conversation tracking ID
            user_id: Optional user ID
            timeout_seconds: Skill execution timeout

        Returns:
            SkillContext configured for this agent
        """
        return SkillContext(
            parameters=parameters or {},
            agent_id=self._get_agent_id(),
            agent_type=self._get_agent_type(),
            conversation_id=conversation_id,
            user_id=user_id,
            priority=priority,
            timeout_seconds=timeout_seconds,
            mcp_client=self._mcp_adapter,
            code_executor=self._code_executor,
        )

    async def execute_deep_skill(
        self,
        skill_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: SkillPriority = SkillPriority.STANDARD,
        timeout_seconds: int = 300,
        conversation_id: Optional[str] = None,
    ) -> SkillResult:
        """
        Execute a DeepSkill by ID.

        This is the primary method for running DeepSkills from agents.
        It handles context creation, validation, and execution.

        Args:
            skill_id: ID of the skill to execute
            parameters: Skill input parameters
            priority: Execution priority
            timeout_seconds: Maximum execution time
            conversation_id: Optional conversation tracking

        Returns:
            SkillResult with execution outcome
        """
        if not self._deep_skill_initialized:
            return SkillResult(
                success=False,
                error="DeepSkills not initialized. Call init_deep_skills() first.",
                error_type="InitializationError"
            )

        # Get the skill
        skill = self.get_deep_skill(skill_id)
        if not skill:
            return SkillResult(
                success=False,
                error=f"Skill not found or not compatible: {skill_id}",
                error_type="SkillNotFoundError"
            )

        # Create context
        context = self.create_skill_context(
            parameters=parameters,
            priority=priority,
            timeout_seconds=timeout_seconds,
            conversation_id=conversation_id,
        )

        logger.info(
            "executing_deep_skill",
            skill_id=skill_id,
            agent_id=self._get_agent_id(),
            param_count=len(parameters or {}),
            priority=priority.name
        )

        # Run the skill
        try:
            result = await skill.run(context)

            logger.info(
                "deep_skill_completed",
                skill_id=skill_id,
                success=result.success,
                execution_time_ms=result.execution_time_ms,
                mcp_calls=result.mcp_calls_count,
                code_executions=result.code_executions_count
            )

            return result

        except Exception as e:
            logger.error(
                "deep_skill_error",
                skill_id=skill_id,
                error=str(e)
            )
            return SkillResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )

    async def run_deep_skill_self_tests(
        self,
        skill_ids: Optional[List[str]] = None,
    ) -> Dict[str, SkillResult]:
        """
        Run self-tests for DeepSkills.

        This method enables agents to verify their skill capabilities
        without external testing dependencies.

        Args:
            skill_ids: Specific skills to test (None = all compatible)

        Returns:
            Dict mapping skill_id to SkillResult
        """
        if not self._deep_skill_initialized:
            return {"_error": SkillResult(
                success=False,
                error="DeepSkills not initialized"
            )}

        # Get skills to test
        if skill_ids:
            skills = [
                self.get_deep_skill(sid)
                for sid in skill_ids
                if self.get_deep_skill(sid)
            ]
        else:
            skills = self.get_compatible_deep_skills()

        results = {}
        context = self.create_skill_context(
            parameters={"_self_test": True},
            timeout_seconds=60
        )

        for skill in skills:
            logger.info(
                "running_skill_self_test",
                skill_id=skill.skill_id,
                agent_id=self._get_agent_id()
            )

            try:
                result = await skill.self_test(context)
                results[skill.skill_id] = result
            except Exception as e:
                results[skill.skill_id] = SkillResult(
                    success=False,
                    error=f"Self-test exception: {e}",
                    error_type="SelfTestException"
                )

        # Log summary
        passed = sum(1 for r in results.values() if r.success)
        logger.info(
            "deep_skill_self_tests_completed",
            agent_id=self._get_agent_id(),
            total=len(results),
            passed=passed,
            failed=len(results) - passed
        )

        return results

    def get_deep_skill_status(self) -> Dict[str, Any]:
        """
        Get status of DeepSkill integration for this agent.

        Returns:
            Status dict with skill counts and capabilities
        """
        compatible_skills = self.get_compatible_deep_skills()

        return {
            "initialized": self._deep_skill_initialized,
            "agent_type": self._get_agent_type(),
            "mcp_available": self._mcp_adapter is not None,
            "code_execution_available": self._code_executor is not None,
            "compatible_skill_count": len(compatible_skills),
            "compatible_skills": [
                {
                    "skill_id": s.skill_id,
                    "name": s.name,
                    "requires_code": s.config.requires_code_execution,
                    "required_tools": s.config.required_mcp_tools,
                }
                for s in compatible_skills
            ],
        }


# Convenience function for quick skill execution
async def execute_skill_for_agent(
    agent: "BaseAgent",
    skill_id: str,
    parameters: Dict[str, Any],
    tool_catalog: Optional["ToolCatalog"] = None,
) -> SkillResult:
    """
    Execute a DeepSkill for an agent without requiring the mixin.

    This is a utility function for ad-hoc skill execution when
    the full mixin isn't needed.

    Args:
        agent: The agent executing the skill
        skill_id: Skill to execute
        parameters: Skill parameters
        tool_catalog: Optional tool catalog (uses agent's if not provided)

    Returns:
        SkillResult from skill execution
    """
    registry = DeepSkillRegistry()
    skill = registry.get(skill_id)

    if not skill:
        return SkillResult(
            success=False,
            error=f"Skill not found: {skill_id}",
            error_type="SkillNotFoundError"
        )

    # Create MCP adapter
    catalog = tool_catalog or getattr(agent, "tools", None)
    mcp_adapter = MCPClientAdapter(tool_catalog=catalog) if catalog else None

    # Create context
    agent_id = agent.get_definition().agent_id if hasattr(agent, "get_definition") else "unknown"
    agent_type = agent_id.replace("-agent", "").replace("-", "_")

    context = SkillContext(
        parameters=parameters,
        agent_id=agent_id,
        agent_type=agent_type,
        mcp_client=mcp_adapter,
        code_executor=CodeExecutor() if skill.config.requires_code_execution else None,
    )

    return await skill.run(context)
