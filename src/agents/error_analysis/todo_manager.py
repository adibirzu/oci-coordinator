"""
Admin Todo Manager for Error Analysis Agent.

Manages a JSON-based todo list for admin action items generated
from error analysis and log pattern detection.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default path for admin todos
DEFAULT_TODO_PATH = Path(__file__).parent.parent.parent.parent / "data" / "admin_todos.json"


class TodoSeverity(str, Enum):
    """Severity levels for admin todos."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TodoStatus(str, Enum):
    """Status of an admin todo."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    IGNORED = "ignored"


@dataclass
class AdminTodo:
    """An admin action item generated from error analysis."""

    id: str
    title: str
    description: str
    severity: TodoSeverity
    error_pattern: str
    source: str  # Which agent/service detected this
    created_at: str
    status: TodoStatus = TodoStatus.PENDING
    assigned_to: str | None = None
    resolution: str | None = None
    resolved_at: str | None = None
    occurrence_count: int = 1
    last_occurrence: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdminTodo:
        """Create AdminTodo from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            severity=TodoSeverity(data["severity"]),
            error_pattern=data["error_pattern"],
            source=data["source"],
            created_at=data["created_at"],
            status=TodoStatus(data.get("status", "pending")),
            assigned_to=data.get("assigned_to"),
            resolution=data.get("resolution"),
            resolved_at=data.get("resolved_at"),
            occurrence_count=data.get("occurrence_count", 1),
            last_occurrence=data.get("last_occurrence"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["severity"] = self.severity.value
        data["status"] = self.status.value
        return data


class AdminTodoManager:
    """
    Manages admin todos stored in a JSON file.

    Provides CRUD operations and pattern deduplication for admin action items.
    """

    def __init__(self, todo_path: Path | str | None = None):
        """Initialize the manager with the todo file path."""
        self.todo_path = Path(todo_path) if todo_path else DEFAULT_TODO_PATH
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Ensure the todo file exists with valid structure."""
        if not self.todo_path.exists():
            self.todo_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_data({
                "version": "1.0",
                "last_updated": None,
                "todos": [],
                "statistics": self._empty_statistics(),
            })

    def _empty_statistics(self) -> dict[str, Any]:
        """Return empty statistics structure."""
        return {
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "resolved": 0,
            "by_severity": {s.value: 0 for s in TodoSeverity},
        }

    def _load_data(self) -> dict[str, Any]:
        """Load todos from JSON file."""
        try:
            with open(self.todo_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load todos", error=str(e))
            return {
                "version": "1.0",
                "last_updated": None,
                "todos": [],
                "statistics": self._empty_statistics(),
            }

    def _save_data(self, data: dict[str, Any]) -> None:
        """Save todos to JSON file."""
        data["last_updated"] = datetime.utcnow().isoformat()
        try:
            with open(self.todo_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save todos", error=str(e))

    def _recalculate_statistics(self, todos: list[dict[str, Any]]) -> dict[str, Any]:
        """Recalculate statistics from todos list."""
        stats = self._empty_statistics()
        stats["total"] = len(todos)

        for todo in todos:
            status = todo.get("status", "pending")
            severity = todo.get("severity", "low")

            if status == "pending":
                stats["pending"] += 1
            elif status == "in_progress":
                stats["in_progress"] += 1
            elif status == "resolved":
                stats["resolved"] += 1

            if severity in stats["by_severity"]:
                stats["by_severity"][severity] += 1

        return stats

    def add_todo(
        self,
        title: str,
        description: str,
        severity: TodoSeverity | str,
        error_pattern: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> AdminTodo:
        """
        Add a new todo or increment occurrence count if pattern exists.

        Returns:
            The created or updated AdminTodo
        """
        if isinstance(severity, str):
            severity = TodoSeverity(severity)

        data = self._load_data()
        todos = data.get("todos", [])
        now = datetime.utcnow().isoformat()

        # Check for existing todo with same pattern
        for todo in todos:
            if todo["error_pattern"] == error_pattern and todo["status"] != "resolved":
                # Update existing todo
                todo["occurrence_count"] = todo.get("occurrence_count", 1) + 1
                todo["last_occurrence"] = now
                data["todos"] = todos
                data["statistics"] = self._recalculate_statistics(todos)
                self._save_data(data)

                logger.info(
                    "Updated existing todo",
                    todo_id=todo["id"],
                    pattern=error_pattern,
                    occurrences=todo["occurrence_count"],
                )
                return AdminTodo.from_dict(todo)

        # Create new todo
        new_todo = AdminTodo(
            id=f"todo-{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            severity=severity,
            error_pattern=error_pattern,
            source=source,
            created_at=now,
            metadata=metadata or {},
        )

        todos.append(new_todo.to_dict())
        data["todos"] = todos
        data["statistics"] = self._recalculate_statistics(todos)
        self._save_data(data)

        logger.info(
            "Created new admin todo",
            todo_id=new_todo.id,
            title=title,
            severity=severity.value,
            source=source,
        )

        return new_todo

    def get_todos(
        self,
        status: TodoStatus | str | None = None,
        severity: TodoSeverity | str | None = None,
    ) -> list[AdminTodo]:
        """Get todos with optional filtering."""
        data = self._load_data()
        todos = data.get("todos", [])

        if status:
            status_val = status.value if isinstance(status, TodoStatus) else status
            todos = [t for t in todos if t.get("status") == status_val]

        if severity:
            sev_val = severity.value if isinstance(severity, TodoSeverity) else severity
            todos = [t for t in todos if t.get("severity") == sev_val]

        return [AdminTodo.from_dict(t) for t in todos]

    def get_todo(self, todo_id: str) -> AdminTodo | None:
        """Get a specific todo by ID."""
        data = self._load_data()
        for todo in data.get("todos", []):
            if todo["id"] == todo_id:
                return AdminTodo.from_dict(todo)
        return None

    def update_status(
        self,
        todo_id: str,
        status: TodoStatus | str,
        resolution: str | None = None,
    ) -> AdminTodo | None:
        """Update the status of a todo."""
        if isinstance(status, str):
            status = TodoStatus(status)

        data = self._load_data()
        todos = data.get("todos", [])

        for todo in todos:
            if todo["id"] == todo_id:
                todo["status"] = status.value
                if resolution:
                    todo["resolution"] = resolution
                if status == TodoStatus.RESOLVED:
                    todo["resolved_at"] = datetime.utcnow().isoformat()

                data["todos"] = todos
                data["statistics"] = self._recalculate_statistics(todos)
                self._save_data(data)

                logger.info(
                    "Updated todo status",
                    todo_id=todo_id,
                    status=status.value,
                )
                return AdminTodo.from_dict(todo)

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get todo statistics."""
        data = self._load_data()
        return data.get("statistics", self._empty_statistics())

    def get_critical_todos(self) -> list[AdminTodo]:
        """Get all pending critical todos."""
        return self.get_todos(status=TodoStatus.PENDING, severity=TodoSeverity.CRITICAL)

    def get_summary(self) -> str:
        """Get a formatted summary of admin todos."""
        stats = self.get_statistics()
        critical = len(self.get_critical_todos())

        lines = [
            f"Admin Todo Summary:",
            f"  Total: {stats['total']}",
            f"  Pending: {stats['pending']} ({critical} critical)",
            f"  In Progress: {stats['in_progress']}",
            f"  Resolved: {stats['resolved']}",
        ]

        if stats["pending"] > 0:
            lines.append("\nBy Severity (Pending):")
            for sev, count in stats["by_severity"].items():
                if count > 0:
                    lines.append(f"  {sev.upper()}: {count}")

        return "\n".join(lines)
