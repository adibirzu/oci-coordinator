"""
File-Based Planner for OCI Operations.

Implements the "planning-with-files" pattern from Manus context engineering.
Uses persistent markdown files as external memory to reduce LLM token usage
and maintain goal awareness across long tool call sequences.

The 3-File Pattern:
    .plan/task_plan.md - Track phases and progress
    .plan/notes.md - Store findings, tool outputs
    .plan/[deliverable].md - Final output/report

Key Principles:
    1. Filesystem as External Memory - Store outputs in files, not context
    2. Attention Manipulation - Re-read plan before decisions
    3. Keep Failure Traces - Log all errors for learning
    4. Append-Only Context - Never modify previous messages

Usage:
    from src.memory.planner import FileBasedPlanner

    planner = FileBasedPlanner(work_dir="/tmp/oci-operations")

    # Start a new operation
    await planner.create_plan(
        operation="Cost Spike Investigation",
        goal="Identify root cause of cost increase",
        phases=["Gather data", "Analyze anomalies", "Generate report"],
        oci_context={"tenancy": "default", "time_range": "7d"},
    )

    # Save tool output
    await planner.save_tool_output(
        tool_name="oci_cost_by_compartment",
        output={"total": 12450, "compartments": [...]},
        summary="Total: $12,450. Top: production (+45%)",
    )

    # Update progress
    await planner.update_phase(phase=1, status="completed")
    await planner.update_phase(phase=2, status="in_progress")

    # Read plan before major decision
    context = await planner.get_current_context()

    # Log error
    await planner.log_error("API timeout", resolution="Reduced time range to 7d")

    # Generate final report
    await planner.write_deliverable("cost_report.md", content)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PlanPhase:
    """A phase in the task plan."""

    name: str
    status: str = "pending"  # pending, in_progress, completed
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class ToolOutput:
    """Record of a tool execution."""

    tool_name: str
    summary: str
    output_location: str  # section in notes.md
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ErrorRecord:
    """Record of an error encountered during operation."""

    error: str
    resolution: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PlanContext:
    """Current context from plan files."""

    operation: str
    goal: str
    current_phase: int
    current_phase_name: str
    phases_completed: int
    total_phases: int
    tool_outputs: list[ToolOutput]
    errors: list[ErrorRecord]
    oci_context: dict[str, Any]
    is_stale: bool  # True if many tool calls since last read


class FileBasedPlanner:
    """
    Manages file-based planning for OCI operations.

    Uses persistent markdown files to:
    - Track multi-phase operations
    - Store tool outputs without bloating LLM context
    - Maintain goal awareness across long sessions
    - Record errors for learning
    """

    PLAN_DIR = ".plan"
    TASK_PLAN_FILE = "task_plan.md"
    NOTES_FILE = "notes.md"

    def __init__(
        self,
        work_dir: str | Path | None = None,
        thread_id: str | None = None,
    ):
        """
        Initialize the file-based planner.

        Args:
            work_dir: Working directory for plan files. Defaults to current dir.
            thread_id: Optional thread ID for namespacing plans.
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.thread_id = thread_id

        # Plan directory path
        if thread_id:
            self.plan_dir = self.work_dir / self.PLAN_DIR / thread_id
        else:
            self.plan_dir = self.work_dir / self.PLAN_DIR

        # File paths
        self.task_plan_path = self.plan_dir / self.TASK_PLAN_FILE
        self.notes_path = self.plan_dir / self.NOTES_FILE

        # In-memory state
        self._phases: list[PlanPhase] = []
        self._tool_outputs: list[ToolOutput] = []
        self._errors: list[ErrorRecord] = []
        self._operation: str = ""
        self._goal: str = ""
        self._oci_context: dict[str, Any] = {}
        self._tool_calls_since_read: int = 0

        self._logger = logger.bind(component="FileBasedPlanner", thread_id=thread_id)

    async def create_plan(
        self,
        operation: str,
        goal: str,
        phases: list[str],
        oci_context: dict[str, Any] | None = None,
        key_questions: list[str] | None = None,
    ) -> str:
        """
        Create a new task plan.

        Args:
            operation: Name of the operation (e.g., "Cost Spike Investigation")
            goal: One sentence describing the end state
            phases: List of phase names
            oci_context: OCI-specific context (tenancy, compartment, profile)
            key_questions: Questions to answer during investigation

        Returns:
            Path to the created task_plan.md
        """
        # Ensure plan directory exists
        self.plan_dir.mkdir(parents=True, exist_ok=True)

        # Store in memory
        self._operation = operation
        self._goal = goal
        self._oci_context = oci_context or {}
        self._phases = [PlanPhase(name=p) for p in phases]
        self._tool_outputs = []
        self._errors = []
        self._tool_calls_since_read = 0

        # Mark first phase as in_progress
        if self._phases:
            self._phases[0].status = "in_progress"
            self._phases[0].started_at = datetime.utcnow()

        # Write task_plan.md
        await self._write_task_plan()

        # Initialize notes.md
        await self._init_notes()

        self._logger.info(
            "Plan created",
            operation=operation,
            phases=len(phases),
            plan_path=str(self.task_plan_path),
        )

        return str(self.task_plan_path)

    async def save_tool_output(
        self,
        tool_name: str,
        output: Any,
        summary: str,
        section: str | None = None,
    ) -> str:
        """
        Save tool output to notes.md.

        Args:
            tool_name: Name of the MCP tool
            output: Raw output (will be stored in file, not context)
            summary: One-line summary to keep in context
            section: Optional section name in notes.md

        Returns:
            Reference to the output location (e.g., "notes.md#section")
        """
        section = section or tool_name.replace("oci_", "").replace("_", "-")
        timestamp = datetime.utcnow()

        # Format output for storage
        if isinstance(output, dict | list):
            output_str = json.dumps(output, indent=2, default=str)
        else:
            output_str = str(output)

        # Truncate very long outputs
        max_chars = 5000
        if len(output_str) > max_chars:
            output_str = output_str[:max_chars] + f"\n... (truncated, {len(output_str) - max_chars} chars omitted)"

        # Append to notes.md
        notes_content = f"""
### {tool_name} - {timestamp.strftime("%Y-%m-%d %H:%M:%S")}

```
{output_str}
```

**Summary**: {summary}

---
"""
        self._append_to_file(self.notes_path, notes_content)

        # Record in memory
        output_location = f"notes.md#{section}"
        self._tool_outputs.append(ToolOutput(
            tool_name=tool_name,
            summary=summary,
            output_location=output_location,
            timestamp=timestamp,
        ))

        # Track tool calls since last read
        self._tool_calls_since_read += 1

        # Update task_plan.md with tool output reference
        await self._write_task_plan()

        self._logger.debug(
            "Tool output saved",
            tool=tool_name,
            location=output_location,
            summary_len=len(summary),
        )

        return output_location

    async def update_phase(
        self,
        phase: int,
        status: str,
    ) -> None:
        """
        Update phase status.

        Args:
            phase: Phase number (0-indexed)
            status: New status (pending, in_progress, completed)
        """
        if phase < 0 or phase >= len(self._phases):
            self._logger.warning("Invalid phase index", phase=phase, total=len(self._phases))
            return

        p = self._phases[phase]
        p.status = status

        if status == "in_progress" and not p.started_at:
            p.started_at = datetime.utcnow()
        elif status == "completed" and not p.completed_at:
            p.completed_at = datetime.utcnow()

            # Auto-start next phase
            if phase + 1 < len(self._phases):
                self._phases[phase + 1].status = "in_progress"
                self._phases[phase + 1].started_at = datetime.utcnow()

        await self._write_task_plan()

        self._logger.info(
            "Phase updated",
            phase=phase,
            name=p.name,
            status=status,
        )

    async def log_error(
        self,
        error: str,
        resolution: str,
    ) -> None:
        """
        Log an error and its resolution.

        Args:
            error: Description of the error
            resolution: How it was resolved or worked around
        """
        self._errors.append(ErrorRecord(
            error=error,
            resolution=resolution,
            timestamp=datetime.utcnow(),
        ))

        await self._write_task_plan()

        self._logger.info(
            "Error logged",
            error=error[:50],
            resolution=resolution[:50],
        )

    async def get_current_context(self) -> PlanContext:
        """
        Get current context from plan files.

        This is the key method for "attention manipulation" - reading the plan
        refreshes goals in the LLM's attention window.

        Returns:
            Current plan context
        """
        # Reset tool calls counter
        self._tool_calls_since_read = 0

        # Find current phase
        current_phase = 0
        current_phase_name = "Unknown"
        for i, p in enumerate(self._phases):
            if p.status == "in_progress":
                current_phase = i
                current_phase_name = p.name
                break

        phases_completed = sum(1 for p in self._phases if p.status == "completed")

        context = PlanContext(
            operation=self._operation,
            goal=self._goal,
            current_phase=current_phase,
            current_phase_name=current_phase_name,
            phases_completed=phases_completed,
            total_phases=len(self._phases),
            tool_outputs=self._tool_outputs.copy(),
            errors=self._errors.copy(),
            oci_context=self._oci_context.copy(),
            is_stale=False,
        )

        self._logger.debug(
            "Context retrieved",
            phase=current_phase,
            completed=phases_completed,
            total=len(self._phases),
        )

        return context

    def needs_context_refresh(self, threshold: int = 5) -> bool:
        """
        Check if context refresh is recommended.

        After many tool calls, the original goal may drift out of
        the LLM's attention window. Call get_current_context() to refresh.

        Args:
            threshold: Number of tool calls before refresh is recommended

        Returns:
            True if refresh is recommended
        """
        return self._tool_calls_since_read >= threshold

    async def write_deliverable(
        self,
        filename: str,
        content: str,
    ) -> str:
        """
        Write a final deliverable file.

        Args:
            filename: Name of the deliverable file (e.g., "cost_report.md")
            content: Content of the deliverable

        Returns:
            Path to the created file
        """
        path = self.plan_dir / filename
        path.write_text(content)

        self._logger.info(
            "Deliverable written",
            filename=filename,
            path=str(path),
            size=len(content),
        )

        return str(path)

    async def read_notes(self, section: str | None = None) -> str:
        """
        Read notes file content.

        Args:
            section: Optional section to extract

        Returns:
            Notes content or section content
        """
        if not self.notes_path.exists():
            return ""

        content = self.notes_path.read_text()

        if section:
            # Extract specific section
            import re
            pattern = rf"### {section}.*?(?=###|\Z)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(0)
            return ""

        return content

    async def get_summary(self) -> str:
        """
        Get a compact summary of the current plan state.

        Returns:
            Markdown summary string
        """
        summary_parts = [
            f"## {self._operation}",
            f"**Goal**: {self._goal}",
            "",
            "### Progress",
        ]

        for i, p in enumerate(self._phases):
            icon = "✅" if p.status == "completed" else "🔄" if p.status == "in_progress" else "⏳"
            summary_parts.append(f"- {icon} Phase {i + 1}: {p.name}")

        if self._tool_outputs:
            summary_parts.append("")
            summary_parts.append("### Recent Tool Outputs")
            for to in self._tool_outputs[-5:]:  # Last 5
                summary_parts.append(f"- `{to.tool_name}`: {to.summary[:50]}...")

        if self._errors:
            summary_parts.append("")
            summary_parts.append("### Errors Encountered")
            for e in self._errors:
                summary_parts.append(f"- {e.error} → {e.resolution}")

        return "\n".join(summary_parts)

    async def cleanup(self) -> None:
        """Clean up plan directory after operation completes."""
        import shutil
        if self.plan_dir.exists():
            shutil.rmtree(self.plan_dir)
            self._logger.info("Plan directory cleaned up", path=str(self.plan_dir))

    # ─────────────────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────────────────

    async def _write_task_plan(self) -> None:
        """Write the task_plan.md file."""
        lines = [
            f"# Task Plan: {self._operation}",
            "",
            "## Goal",
            self._goal,
            "",
        ]

        # OCI Context
        if self._oci_context:
            lines.extend([
                "## OCI Context",
            ])
            for key, value in self._oci_context.items():
                lines.append(f"- **{key.title()}**: {value}")
            lines.append("")

        # Phases
        lines.append("## Phases")
        for i, p in enumerate(self._phases):
            checkbox = "[x]" if p.status == "completed" else "[ ]"
            suffix = " (CURRENT)" if p.status == "in_progress" else ""
            lines.append(f"- {checkbox} Phase {i + 1}: {p.name}{suffix}")
        lines.append("")

        # Tool Outputs
        if self._tool_outputs:
            lines.extend([
                "## Tool Outputs",
                "| Tool | Summary | Location |",
                "|------|---------|----------|",
            ])
            for to in self._tool_outputs:
                lines.append(f"| {to.tool_name} | {to.summary[:40]}... | {to.output_location} |")
            lines.append("")

        # Errors
        if self._errors:
            lines.extend([
                "## Errors Encountered",
            ])
            for e in self._errors:
                lines.append(f"- {e.error} → {e.resolution}")
            lines.append("")

        # Status
        current_phase = next((p for p in self._phases if p.status == "in_progress"), None)
        phase_name = current_phase.name if current_phase else "Not started"
        lines.extend([
            "## Status",
            f"**Currently**: {phase_name}",
        ])

        self.task_plan_path.write_text("\n".join(lines))

    async def _init_notes(self) -> None:
        """Initialize the notes.md file."""
        content = f"""# Notes: {self._operation}

## Tool Outputs

*Tool outputs will be appended below*

---
"""
        self.notes_path.write_text(content)

    def _append_to_file(self, path: Path, content: str) -> None:
        """Append content to a file."""
        with open(path, "a") as f:
            f.write(content)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Helpers
# ─────────────────────────────────────────────────────────────────────────────


def should_use_planner(
    query: str,
    estimated_tool_calls: int = 3,
) -> bool:
    """
    Determine if file-based planning should be used for a query.

    Args:
        query: User query
        estimated_tool_calls: Estimated number of tool calls needed

    Returns:
        True if planner should be used
    """
    # Use planner for complex operations
    complex_keywords = [
        "investigate", "troubleshoot", "analyze", "audit", "compare",
        "spike", "anomaly", "report", "security", "performance",
        "why", "root cause", "trend",
    ]

    query_lower = query.lower()
    has_complex_keyword = any(kw in query_lower for kw in complex_keywords)

    return has_complex_keyword or estimated_tool_calls >= 3


async def create_planner_for_workflow(
    workflow_name: str,
    query: str,
    thread_id: str | None = None,
) -> FileBasedPlanner | None:
    """
    Create a planner for a specific workflow type.

    Args:
        workflow_name: Name of the workflow
        query: User query
        thread_id: Optional thread ID

    Returns:
        Configured planner or None if not needed
    """
    workflow_phases = {
        "cost_analysis": [
            "Get cost summary",
            "Drill down by service",
            "Analyze anomalies",
            "Generate report",
        ],
        "database_troubleshoot": [
            "Get database status",
            "Check metrics",
            "Query logs",
            "Generate AWR",
            "Synthesize findings",
        ],
        "security_audit": [
            "Define scope",
            "Check Cloud Guard",
            "Review IAM",
            "Check security zones",
            "Generate report",
        ],
        "infrastructure_check": [
            "List resources",
            "Check status",
            "Analyze metrics",
            "Generate summary",
        ],
    }

    phases = workflow_phases.get(workflow_name)
    if not phases:
        return None

    planner = FileBasedPlanner(thread_id=thread_id)

    await planner.create_plan(
        operation=workflow_name.replace("_", " ").title(),
        goal=f"Complete {workflow_name} for: {query[:100]}",
        phases=phases,
        oci_context={"workflow": workflow_name},
    )

    return planner
