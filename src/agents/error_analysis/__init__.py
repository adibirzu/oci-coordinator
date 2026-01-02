"""
Error Analysis Agent Package.

Provides log error detection, pattern analysis, and admin todo management.
"""

from src.agents.error_analysis.agent import ErrorAnalysisAgent
from src.agents.error_analysis.todo_manager import (
    AdminTodo,
    AdminTodoManager,
    TodoSeverity,
    TodoStatus,
)

__all__ = [
    "ErrorAnalysisAgent",
    "AdminTodo",
    "AdminTodoManager",
    "TodoSeverity",
    "TodoStatus",
]
