"""
Test Suite Definitions for Agent-Based Testing.

Provides pre-defined test suites that agents can run to verify:
- MCP tool connectivity
- Skill execution
- End-to-end workflows

Test suites are organized by category:
- CONNECTIVITY: Basic MCP server communication
- SKILL_BASIC: Individual skill execution
- SKILL_ADVANCED: Complex multi-step skills
- INTEGRATION: End-to-end workflow tests

Example usage:
    from src.agents.testing import TestSuite, get_test_suite

    # Get a predefined suite
    suite = get_test_suite(TestSuite.BASIC_CONNECTIVITY)

    # Run with agent's test runner
    results = await runner.run_suite(suite)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import structlog

logger = structlog.get_logger()


class TestSuite(Enum):
    """Pre-defined test suite categories."""

    # Connectivity tests
    BASIC_CONNECTIVITY = "basic_connectivity"
    MCP_HEALTH = "mcp_health"

    # Skill tests
    SKILL_BASIC = "skill_basic"
    SKILL_ADVANCED = "skill_advanced"

    # Domain-specific tests
    DATABASE = "database"
    SECURITY = "security"
    FINOPS = "finops"
    LOG_ANALYTICS = "log_analytics"
    INFRASTRUCTURE = "infrastructure"

    # Integration tests
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"


@dataclass
class TestCase:
    """
    Definition of a single test case.

    Test cases can be:
    1. MCP tool calls (tool_name + tool_params)
    2. Skill executions (skill_id + skill_params)
    3. Custom assertions (assertion_fn)
    """

    name: str
    description: str

    # What to test
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    skill_id: Optional[str] = None
    skill_params: Dict[str, Any] = field(default_factory=dict)

    # Validation
    expected_status: str = "success"  # "success", "error", "timeout"
    assertion_fn: Optional[Callable[[Any], bool]] = None
    expected_keys: List[str] = field(default_factory=list)

    # Metadata
    timeout_seconds: int = 30
    tags: Set[str] = field(default_factory=set)
    requires_backend: bool = True  # If False, can run with mocks

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure tags is a set."""
        if not isinstance(self.tags, set):
            self.tags = set(self.tags)


@dataclass
class TestSuiteDefinition:
    """
    Definition of a test suite containing multiple test cases.
    """

    suite_id: TestSuite
    name: str
    description: str
    test_cases: List[TestCase]

    # Configuration
    stop_on_failure: bool = False
    parallel_execution: bool = False
    timeout_seconds: int = 300  # Total suite timeout

    # Metadata
    tags: Set[str] = field(default_factory=set)
    compatible_agents: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Ensure sets are sets."""
        if not isinstance(self.tags, set):
            self.tags = set(self.tags)
        if not isinstance(self.compatible_agents, set):
            self.compatible_agents = set(self.compatible_agents)


# Registry for test suites
_SUITE_REGISTRY: Dict[TestSuite, TestSuiteDefinition] = {}


def register_test_suite(suite: TestSuiteDefinition) -> TestSuiteDefinition:
    """
    Register a test suite in the global registry.

    Args:
        suite: Test suite definition to register

    Returns:
        The registered suite
    """
    _SUITE_REGISTRY[suite.suite_id] = suite
    logger.debug(
        "test_suite_registered",
        suite_id=suite.suite_id.value,
        test_count=len(suite.test_cases)
    )
    return suite


def get_test_suite(suite_id: TestSuite) -> Optional[TestSuiteDefinition]:
    """
    Get a test suite by ID.

    Args:
        suite_id: TestSuite enum value

    Returns:
        TestSuiteDefinition if found, None otherwise
    """
    return _SUITE_REGISTRY.get(suite_id)


def get_all_suites() -> Dict[TestSuite, TestSuiteDefinition]:
    """Get all registered test suites."""
    return _SUITE_REGISTRY.copy()


def get_suites_for_agent(agent_id: str) -> List[TestSuiteDefinition]:
    """
    Get all test suites compatible with an agent.

    Args:
        agent_id: Agent identifier

    Returns:
        List of compatible test suites
    """
    compatible = []
    for suite in _SUITE_REGISTRY.values():
        if not suite.compatible_agents or agent_id in suite.compatible_agents:
            compatible.append(suite)
    return compatible


def get_suites_by_tag(tag: str) -> List[TestSuiteDefinition]:
    """
    Get all test suites with a specific tag.

    Args:
        tag: Tag to filter by

    Returns:
        List of matching test suites
    """
    return [
        suite for suite in _SUITE_REGISTRY.values()
        if tag in suite.tags
    ]


# =============================================================================
# Pre-defined Test Suites
# =============================================================================

# Basic connectivity tests
BASIC_CONNECTIVITY_SUITE = register_test_suite(TestSuiteDefinition(
    suite_id=TestSuite.BASIC_CONNECTIVITY,
    name="Basic Connectivity Tests",
    description="Verify basic MCP server connectivity and health",
    tags={"connectivity", "health", "quick"},
    test_cases=[
        TestCase(
            name="mcp_ping",
            description="Verify MCP server responds to ping",
            tool_name="oci_database_ping",
            expected_status="success",
            timeout_seconds=10,
            tags={"connectivity", "quick"},
        ),
        TestCase(
            name="health_check",
            description="Verify MCP server health endpoint",
            tool_name="oci_database_health_check",
            expected_status="success",
            timeout_seconds=10,
            tags={"health", "quick"},
        ),
    ]
))

# MCP health tests
MCP_HEALTH_SUITE = register_test_suite(TestSuiteDefinition(
    suite_id=TestSuite.MCP_HEALTH,
    name="MCP Server Health Tests",
    description="Comprehensive health checks for MCP servers",
    tags={"health", "mcp"},
    test_cases=[
        TestCase(
            name="database_observatory_health",
            description="Check Database Observatory MCP health",
            tool_name="oci_database_health_check",
            expected_status="success",
            tags={"database", "health"},
        ),
        TestCase(
            name="finops_ping",
            description="Check FinOps MCP connectivity",
            tool_name="oci_cost_ping",
            expected_status="success",
            tags={"finops", "health"},
        ),
        TestCase(
            name="security_ping",
            description="Check Security MCP connectivity",
            tool_name="oci_security_ping",
            expected_status="success",
            tags={"security", "health"},
        ),
        TestCase(
            name="logan_health",
            description="Check Logan MCP health",
            tool_name="oci_logan_health",
            expected_status="success",
            tags={"logan", "health"},
        ),
    ]
))

# Database test suite
DATABASE_SUITE = register_test_suite(TestSuiteDefinition(
    suite_id=TestSuite.DATABASE,
    name="Database Operations Tests",
    description="Test database-related MCP tools and skills",
    tags={"database", "db"},
    compatible_agents={"db_troubleshoot", "coordinator"},
    test_cases=[
        TestCase(
            name="list_connections",
            description="List available database connections",
            tool_name="oci_database_list_connections",
            expected_status="success",
            tags={"database", "connectivity"},
        ),
        TestCase(
            name="cache_status",
            description="Check OPSI cache status",
            tool_name="oci_database_cache_status",
            expected_status="success",
            tags={"database", "cache"},
        ),
        TestCase(
            name="fleet_summary",
            description="Get database fleet summary",
            tool_name="oci_opsi_get_fleet_summary",
            tool_params={},
            expected_status="success",
            tags={"database", "opsi"},
        ),
        TestCase(
            name="list_skills",
            description="List available OPSI skills",
            tool_name="oci_opsi_list_skills",
            expected_status="success",
            tags={"database", "skills"},
        ),
    ]
))

# Security test suite
SECURITY_SUITE = register_test_suite(TestSuiteDefinition(
    suite_id=TestSuite.SECURITY,
    name="Security Operations Tests",
    description="Test security-related MCP tools",
    tags={"security"},
    compatible_agents={"security", "coordinator"},
    test_cases=[
        TestCase(
            name="security_health",
            description="Check security MCP server health",
            tool_name="oci_security_health",
            tool_params={"deep": False},
            expected_status="success",
            tags={"security", "health"},
        ),
    ]
))

# FinOps test suite
FINOPS_SUITE = register_test_suite(TestSuiteDefinition(
    suite_id=TestSuite.FINOPS,
    name="FinOps Operations Tests",
    description="Test cost and budget MCP tools",
    tags={"finops", "cost"},
    compatible_agents={"finops", "coordinator"},
    test_cases=[
        TestCase(
            name="finops_ping",
            description="Check FinOps MCP connectivity",
            tool_name="oci_cost_ping",
            expected_status="success",
            tags={"finops", "connectivity"},
        ),
        TestCase(
            name="list_providers",
            description="List configured cloud providers",
            tool_name="finops_list_providers",
            expected_status="success",
            tags={"finops", "providers"},
        ),
        TestCase(
            name="cost_templates",
            description="Get available cost analysis templates",
            tool_name="oci_cost_templates",
            expected_status="success",
            tags={"finops", "templates"},
        ),
    ]
))

# Log Analytics test suite
LOG_ANALYTICS_SUITE = register_test_suite(TestSuiteDefinition(
    suite_id=TestSuite.LOG_ANALYTICS,
    name="Log Analytics Tests",
    description="Test Logan MCP tools",
    tags={"logan", "logs"},
    compatible_agents={"log_analytics", "coordinator"},
    test_cases=[
        TestCase(
            name="logan_health",
            description="Check Logan MCP health",
            tool_name="oci_logan_health",
            tool_params={"detail": False},
            expected_status="success",
            tags={"logan", "health"},
        ),
        TestCase(
            name="usage_guide",
            description="Get Logan usage guide",
            tool_name="oci_logan_usage_guide",
            expected_status="success",
            tags={"logan", "docs"},
        ),
        TestCase(
            name="list_skills",
            description="List Logan skills",
            tool_name="oci_logan_list_skills",
            expected_status="success",
            tags={"logan", "skills"},
        ),
    ]
))

# Infrastructure test suite
INFRASTRUCTURE_SUITE = register_test_suite(TestSuiteDefinition(
    suite_id=TestSuite.INFRASTRUCTURE,
    name="Infrastructure Tests",
    description="Test compute, network, and storage MCP tools",
    tags={"infrastructure", "compute", "network"},
    compatible_agents={"infrastructure", "coordinator"},
    test_cases=[
        TestCase(
            name="mcp_ping",
            description="Check OCI MCP connectivity",
            tool_name="oci_ping",
            expected_status="success",
            tags={"infrastructure", "connectivity"},
        ),
        TestCase(
            name="list_domains",
            description="List available OCI tool domains",
            tool_name="oci_list_domains",
            expected_status="success",
            tags={"infrastructure", "domains"},
        ),
        TestCase(
            name="cache_stats",
            description="Get MCP cache statistics",
            tool_name="oci_get_cache_stats",
            expected_status="success",
            tags={"infrastructure", "cache"},
        ),
    ]
))


# =============================================================================
# Test Suite Builder
# =============================================================================

class TestSuiteBuilder:
    """
    Builder for creating custom test suites.

    Example:
        suite = (TestSuiteBuilder()
            .with_id(TestSuite.DATABASE)
            .with_name("Custom DB Tests")
            .add_test(TestCase(name="test1", ...))
            .add_test(TestCase(name="test2", ...))
            .build())
    """

    def __init__(self):
        self._suite_id: Optional[TestSuite] = None
        self._name: str = ""
        self._description: str = ""
        self._test_cases: List[TestCase] = []
        self._tags: Set[str] = set()
        self._compatible_agents: Set[str] = set()
        self._stop_on_failure: bool = False
        self._parallel_execution: bool = False
        self._timeout_seconds: int = 300

    def with_id(self, suite_id: TestSuite) -> "TestSuiteBuilder":
        """Set suite ID."""
        self._suite_id = suite_id
        return self

    def with_name(self, name: str) -> "TestSuiteBuilder":
        """Set suite name."""
        self._name = name
        return self

    def with_description(self, description: str) -> "TestSuiteBuilder":
        """Set suite description."""
        self._description = description
        return self

    def add_test(self, test_case: TestCase) -> "TestSuiteBuilder":
        """Add a test case."""
        self._test_cases.append(test_case)
        return self

    def add_tests(self, test_cases: List[TestCase]) -> "TestSuiteBuilder":
        """Add multiple test cases."""
        self._test_cases.extend(test_cases)
        return self

    def with_tags(self, *tags: str) -> "TestSuiteBuilder":
        """Add tags."""
        self._tags.update(tags)
        return self

    def for_agents(self, *agent_ids: str) -> "TestSuiteBuilder":
        """Set compatible agents."""
        self._compatible_agents.update(agent_ids)
        return self

    def stop_on_failure(self, stop: bool = True) -> "TestSuiteBuilder":
        """Set stop on failure behavior."""
        self._stop_on_failure = stop
        return self

    def parallel(self, parallel: bool = True) -> "TestSuiteBuilder":
        """Enable parallel execution."""
        self._parallel_execution = parallel
        return self

    def with_timeout(self, seconds: int) -> "TestSuiteBuilder":
        """Set suite timeout."""
        self._timeout_seconds = seconds
        return self

    def build(self) -> TestSuiteDefinition:
        """Build and return the test suite."""
        if not self._suite_id:
            raise ValueError("Suite ID is required")
        if not self._name:
            raise ValueError("Suite name is required")

        return TestSuiteDefinition(
            suite_id=self._suite_id,
            name=self._name,
            description=self._description,
            test_cases=self._test_cases,
            tags=self._tags,
            compatible_agents=self._compatible_agents,
            stop_on_failure=self._stop_on_failure,
            parallel_execution=self._parallel_execution,
            timeout_seconds=self._timeout_seconds,
        )

    def build_and_register(self) -> TestSuiteDefinition:
        """Build and register the test suite."""
        suite = self.build()
        return register_test_suite(suite)
