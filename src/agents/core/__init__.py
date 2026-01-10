"""
Core framework components for advanced agent capabilities.

This module provides:
- DeepSkill: Advanced skills combining MCP tools with code execution
- CodeExecutor: Sandboxed Python execution for agent use
- ModelTiers: Intelligent LLM model selection strategy
- SQLclExecutor: Direct SQL execution via MCP

All components are designed to be used BY agents, not external tools.
"""

from src.agents.core.deep_skills import (
    DeepSkill,
    DeepSkillConfig,
    DeepSkillRegistry,
    SkillContext,
    SkillResult,
)
from src.agents.core.code_executor import (
    CodeExecutor,
    CodeExecutionResult,
    ExecutionConfig,
)
from src.agents.core.model_tiers import (
    ModelTier,
    ModelTierStrategy,
    TaskComplexity,
    get_model_for_task,
)
from src.agents.core.sqlcl_executor import (
    SQLclExecutor,
    SQLclConfig,
    QueryResult,
)

__all__ = [
    # Deep Skills
    "DeepSkill",
    "DeepSkillConfig",
    "DeepSkillRegistry",
    "SkillContext",
    "SkillResult",
    # Code Execution
    "CodeExecutor",
    "CodeExecutionResult",
    "ExecutionConfig",
    # Model Tiers
    "ModelTier",
    "ModelTierStrategy",
    "TaskComplexity",
    "get_model_for_task",
    # SQLcl Execution
    "SQLclExecutor",
    "SQLclConfig",
    "QueryResult",
]
