"""
Agent-Based Testing Framework.

This framework enables agents to test their own capabilities
WITHOUT relying on external tools like Claude Code's MCP connections.

Key principles:
1. Tests are executed BY agents, not external systems
2. Tests use the same MCP clients agents use in production
3. Tests are self-contained and can run in isolation
4. Tests report results back to the coordinator

Usage:
    # From within an agent
    from src.agents.testing import AgentTestRunner, TestSuite

    runner = AgentTestRunner(agent=self)
    results = await runner.run_suite(TestSuite.BASIC_CONNECTIVITY)

    # Or run all tests
    results = await runner.run_all_tests()
"""

from src.agents.testing.runner import (
    AgentTestRunner,
    TestResult,
    TestSuiteResult,
)
from src.agents.testing.suites import (
    TestSuite,
    TestCase,
    get_test_suite,
    register_test_suite,
)
from src.agents.testing.mocks import (
    MockMCPClient,
    MockResponse,
    create_mock_context,
)

__all__ = [
    # Runner
    "AgentTestRunner",
    "TestResult",
    "TestSuiteResult",
    # Suites
    "TestSuite",
    "TestCase",
    "get_test_suite",
    "register_test_suite",
    # Mocks
    "MockMCPClient",
    "MockResponse",
    "create_mock_context",
]
