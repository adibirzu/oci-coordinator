"""Tests for file-based planner integration.

Tests the FileBasedPlanner class and helper functions that provide
token optimization through filesystem-based context management.
"""

import tempfile
from pathlib import Path

import pytest

from src.memory.planner import (
    ErrorRecord,
    FileBasedPlanner,
    PlanPhase,
    ToolOutput,
    create_planner_for_workflow,
    should_use_planner,
)


class TestShouldUsePlanner:
    """Tests for should_use_planner function."""

    def test_complex_workflow_types_trigger_planner(self):
        """Complex workflows should always use planner."""
        complex_workflows = [
            "database_troubleshoot",
            "cost_analysis",
            "security_audit",
            "log_analysis",
            "error_analysis",
        ]
        for workflow in complex_workflows:
            assert should_use_planner("simple query", workflow) is True, \
                f"Workflow {workflow} should trigger planner"

    def test_general_workflow_does_not_trigger_without_keywords(self):
        """General workflow without keywords should not use planner."""
        assert should_use_planner("show databases", "general") is False
        assert should_use_planner("list instances", "infrastructure_check") is False

    def test_keywords_trigger_planner_for_any_workflow(self):
        """Complex keywords should trigger planner regardless of workflow."""
        keywords = [
            "investigate", "troubleshoot", "analyze", "audit",
            "compare", "spike", "anomaly", "report", "security",
            "performance", "why", "root cause", "trend", "diagnose", "debug",
        ]
        for keyword in keywords:
            query = f"please {keyword} this issue"
            assert should_use_planner(query, "general") is True, \
                f"Keyword '{keyword}' should trigger planner"

    def test_case_insensitive_keyword_matching(self):
        """Keywords should match case-insensitively."""
        assert should_use_planner("TROUBLESHOOT the database", None) is True
        assert should_use_planner("Analyze performance", None) is True

    def test_none_workflow_type_handled(self):
        """None workflow type should not cause errors."""
        assert should_use_planner("simple query", None) is False
        assert should_use_planner("troubleshoot issue", None) is True


class TestPlanPhase:
    """Tests for PlanPhase dataclass."""

    def test_phase_creation(self):
        """Test phase creation with default values."""
        phase = PlanPhase(name="Get data")
        assert phase.name == "Get data"
        assert phase.status == "pending"
        assert phase.started_at is None
        assert phase.completed_at is None

    def test_phase_with_status(self):
        """Test phase with explicit status."""
        phase = PlanPhase(name="Test phase", status="completed")
        assert phase.status == "completed"


class TestToolOutput:
    """Tests for ToolOutput dataclass."""

    def test_tool_output_creation(self):
        """Test tool output creation."""
        output = ToolOutput(
            tool_name="oci_list_databases",
            summary="Found 5 databases",
            output_location="notes.md#databases",
        )
        assert output.tool_name == "oci_list_databases"
        assert output.summary == "Found 5 databases"
        assert output.output_location == "notes.md#databases"
        assert output.timestamp is not None


class TestErrorRecord:
    """Tests for ErrorRecord dataclass."""

    def test_error_record_creation(self):
        """Test error record creation."""
        error = ErrorRecord(
            error="Timeout after 30s",
            resolution="Reduced time range",
        )
        assert error.error == "Timeout after 30s"
        assert error.resolution == "Reduced time range"
        assert error.timestamp is not None


class TestFileBasedPlanner:
    """Tests for FileBasedPlanner class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for plan files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def planner(self, temp_dir):
        """Create a planner instance for testing."""
        return FileBasedPlanner(work_dir=temp_dir)

    @pytest.mark.asyncio
    async def test_planner_create_plan(self, planner):
        """Test planner creates plan and directory."""
        await planner.create_plan(
            operation="Test Operation",
            goal="Test goal",
            phases=["Phase 1", "Phase 2"],
        )
        assert planner.plan_dir.exists()
        assert (planner.plan_dir / "task_plan.md").exists()

    @pytest.mark.asyncio
    async def test_update_phase(self, planner):
        """Test updating phase status."""
        await planner.create_plan(
            operation="Test",
            goal="Test",
            phases=["Phase 1", "Phase 2"],
        )
        await planner.update_phase(0, status="completed")

        # Verify phase updated
        plan_content = (planner.plan_dir / "task_plan.md").read_text()
        assert "Phase 1" in plan_content

    @pytest.mark.asyncio
    async def test_save_tool_output(self, planner):
        """Test saving tool output to notes."""
        await planner.create_plan(
            operation="Test",
            goal="Test",
            phases=["Phase 1"],
        )
        await planner.save_tool_output(
            tool_name="test_tool",
            output={"key": "value"},
            summary="Test summary",
        )

        notes_file = planner.plan_dir / "notes.md"
        assert notes_file.exists()
        notes_content = notes_file.read_text()
        assert "test_tool" in notes_content
        assert "Test summary" in notes_content

    @pytest.mark.asyncio
    async def test_log_error(self, planner):
        """Test logging error to plan."""
        await planner.create_plan(
            operation="Test",
            goal="Test",
            phases=["Phase 1"],
        )
        await planner.log_error(
            error="Test error occurred",
            resolution="Will retry with smaller scope",
        )

        plan_content = (planner.plan_dir / "task_plan.md").read_text()
        assert "Test error occurred" in plan_content

    @pytest.mark.asyncio
    async def test_get_summary(self, planner):
        """Test getting plan summary."""
        await planner.create_plan(
            operation="Test Operation",
            goal="Test goal",
            phases=["Phase 1", "Phase 2"],
        )
        await planner.save_tool_output("tool1", {"data": 1}, "Summary 1")
        await planner.save_tool_output("tool2", {"data": 2}, "Summary 2")

        summary = await planner.get_summary()
        assert "Test Operation" in summary
        assert "tool1" in summary


class TestCreatePlannerForWorkflow:
    """Tests for create_planner_for_workflow factory function."""

    @pytest.mark.asyncio
    async def test_creates_planner_for_cost_analysis(self):
        """Test planner creation for cost analysis workflow."""
        planner = await create_planner_for_workflow(
            workflow_name="cost_analysis",
            query="Why did costs spike?",
        )
        assert planner is not None
        assert planner._operation == "Cost Analysis"
        assert len(planner._phases) == 4

    @pytest.mark.asyncio
    async def test_creates_planner_for_database_troubleshoot(self):
        """Test planner creation for database troubleshoot workflow."""
        planner = await create_planner_for_workflow(
            workflow_name="database_troubleshoot",
            query="Database is slow",
        )
        assert planner is not None
        assert planner._operation == "Database Troubleshoot"
        assert len(planner._phases) == 5

    @pytest.mark.asyncio
    async def test_creates_planner_for_security_audit(self):
        """Test planner creation for security audit workflow."""
        planner = await create_planner_for_workflow(
            workflow_name="security_audit",
            query="Run security check",
        )
        assert planner is not None
        assert planner._operation == "Security Audit"
        assert len(planner._phases) == 5

    @pytest.mark.asyncio
    async def test_creates_planner_for_log_analysis(self):
        """Test planner creation for log analysis workflow."""
        planner = await create_planner_for_workflow(
            workflow_name="log_analysis",
            query="Find error patterns",
        )
        assert planner is not None
        assert planner._operation == "Log Analysis"
        assert len(planner._phases) == 5

    @pytest.mark.asyncio
    async def test_creates_planner_with_thread_id(self):
        """Test planner creation includes thread ID in path."""
        planner = await create_planner_for_workflow(
            workflow_name="cost_analysis",
            query="Test query",
            thread_id="T123456",
        )
        assert planner is not None
        assert "T123456" in str(planner.plan_dir)

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_workflow(self):
        """Test returns None for unknown workflow."""
        planner = await create_planner_for_workflow(
            workflow_name="unknown_workflow",
            query="Generic query",
        )
        assert planner is None


class TestPlannerIntegration:
    """Integration tests for planner with coordinator-like usage."""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """Simulate a full workflow with planner."""
        # Create planner for cost analysis
        planner = await create_planner_for_workflow(
            workflow_name="cost_analysis",
            query="Why did costs increase last week?",
            thread_id="test-thread",
        )
        assert planner is not None

        # Simulate tool calls
        await planner.save_tool_output(
            tool_name="oci_cost_by_compartment",
            output={"total": 12450, "change_pct": 45},
            summary="Total cost: $12,450 (+45%)",
        )

        await planner.update_phase(0, status="completed")
        await planner.update_phase(1, status="in_progress")

        await planner.save_tool_output(
            tool_name="oci_cost_service_drilldown",
            output={"top_service": "Compute", "cost": 8200},
            summary="Top service: Compute ($8,200)",
        )

        await planner.update_phase(1, status="completed")

        # Get summary
        summary = await planner.get_summary()
        assert "Cost Analysis" in summary
        assert "oci_cost" in summary

        # Verify files exist
        assert planner.plan_dir.exists()
        assert (planner.plan_dir / "task_plan.md").exists()
        assert (planner.plan_dir / "notes.md").exists()

    @pytest.mark.asyncio
    async def test_error_recovery_pattern(self):
        """Test error logging and recovery pattern."""
        planner = await create_planner_for_workflow(
            workflow_name="database_troubleshoot",
            query="Database is slow",
        )
        assert planner is not None

        # Simulate timeout error
        await planner.log_error(
            error="Timeout after 30s querying AWR",
            resolution="Reducing time range to 1 hour",
        )

        # Verify error is logged
        plan_content = (planner.plan_dir / "task_plan.md").read_text()
        assert "Timeout" in plan_content

        # Simulate successful retry
        await planner.save_tool_output(
            tool_name="get_awr_report",
            output={"time_range": "1h"},
            summary="AWR report generated for 1 hour window",
        )

        notes_content = (planner.plan_dir / "notes.md").read_text()
        assert "AWR" in notes_content
