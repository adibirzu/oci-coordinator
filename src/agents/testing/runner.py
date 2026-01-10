"""
Agent Test Runner - Executes tests from within agents.

This runner is designed to be used BY agents to test their own
capabilities without external dependencies like Claude Code.

Example usage from within an agent:

    class DbTroubleshootAgent(BaseAgent):
        async def self_test(self) -> Dict[str, Any]:
            runner = AgentTestRunner(agent=self)
            results = await runner.run_all_tests()
            return results.to_dict()

        async def test_specific_skill(self, skill_id: str):
            runner = AgentTestRunner(agent=self)
            return await runner.test_skill(skill_id)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
import structlog
import traceback

logger = structlog.get_logger()


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TestResult:
    """Result from a single test execution."""

    test_id: str
    test_name: str
    status: TestStatus
    duration_ms: int = 0

    # Success details
    message: Optional[str] = None
    data: Optional[Any] = None

    # Error details
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback_str: Optional[str] = None

    # Metadata
    executed_at: datetime = field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None
    skill_id: Optional[str] = None

    def is_success(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "error": self.error,
            "executed_at": self.executed_at.isoformat(),
            "agent_id": self.agent_id,
            "skill_id": self.skill_id,
        }


@dataclass
class TestSuiteResult:
    """Result from running a test suite."""

    suite_name: str
    total_tests: int
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    timeouts: int = 0

    total_duration_ms: int = 0
    results: List[TestResult] = field(default_factory=list)

    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Metadata
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.passed == self.total_tests

    def add_result(self, result: TestResult) -> None:
        """Add a test result and update counters."""
        self.results.append(result)
        self.total_duration_ms += result.duration_ms

        if result.status == TestStatus.PASSED:
            self.passed += 1
        elif result.status == TestStatus.FAILED:
            self.failed += 1
        elif result.status == TestStatus.SKIPPED:
            self.skipped += 1
        elif result.status == TestStatus.ERROR:
            self.errors += 1
        elif result.status == TestStatus.TIMEOUT:
            self.timeouts += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "success_rate": self.success_rate,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": [r.to_dict() for r in self.results],
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
        }

    def get_failures(self) -> List[TestResult]:
        """Get all failed tests."""
        return [r for r in self.results if r.status in [TestStatus.FAILED, TestStatus.ERROR]]


# Type for test functions
TestFunc = Callable[..., Coroutine[Any, Any, TestResult]]


@dataclass
class RegisteredTest:
    """A registered test case."""
    test_id: str
    test_name: str
    test_func: TestFunc
    suite: str = "default"
    timeout_seconds: int = 60
    requires_mcp: bool = False
    requires_db: bool = False
    tags: List[str] = field(default_factory=list)


class AgentTestRunner:
    """
    Test runner that executes from within an agent.

    This runner uses the agent's own MCP client and resources,
    ensuring tests reflect actual agent capabilities.

    Usage:
        runner = AgentTestRunner(agent=self)

        # Run all tests
        results = await runner.run_all_tests()

        # Run specific suite
        results = await runner.run_suite("connectivity")

        # Run single test
        result = await runner.run_test("test_mcp_connectivity")

        # Test a skill
        result = await runner.test_skill("db_blocking_analysis")
    """

    # Class-level test registry
    _registered_tests: Dict[str, RegisteredTest] = {}

    def __init__(
        self,
        agent: Any,
        default_timeout: int = 60,
        fail_fast: bool = False
    ):
        """
        Initialize test runner.

        Args:
            agent: The agent instance running tests
            default_timeout: Default timeout for tests in seconds
            fail_fast: Stop on first failure
        """
        self.agent = agent
        self.default_timeout = default_timeout
        self.fail_fast = fail_fast

        # Extract agent info
        self.agent_id = getattr(agent, 'agent_id', str(id(agent)))
        self.agent_type = getattr(agent, 'agent_type', type(agent).__name__)

        # Get MCP client from agent if available
        self.mcp_client = getattr(agent, 'mcp_client', None)

    @classmethod
    def register_test(
        cls,
        test_id: str,
        test_name: str,
        suite: str = "default",
        timeout_seconds: int = 60,
        requires_mcp: bool = False,
        requires_db: bool = False,
        tags: Optional[List[str]] = None
    ) -> Callable[[TestFunc], TestFunc]:
        """
        Decorator to register a test function.

        Example:
            @AgentTestRunner.register_test(
                test_id="test_mcp_connectivity",
                test_name="MCP Connectivity Test",
                suite="connectivity",
                requires_mcp=True
            )
            async def test_mcp_connectivity(runner: AgentTestRunner) -> TestResult:
                ...
        """
        def decorator(func: TestFunc) -> TestFunc:
            registered = RegisteredTest(
                test_id=test_id,
                test_name=test_name,
                test_func=func,
                suite=suite,
                timeout_seconds=timeout_seconds,
                requires_mcp=requires_mcp,
                requires_db=requires_db,
                tags=tags or []
            )
            cls._registered_tests[test_id] = registered
            return func
        return decorator

    async def run_test(
        self,
        test_id: str,
        **kwargs
    ) -> TestResult:
        """
        Run a single registered test.

        Args:
            test_id: ID of the test to run
            **kwargs: Additional arguments for the test

        Returns:
            TestResult with test outcome
        """
        if test_id not in self._registered_tests:
            return TestResult(
                test_id=test_id,
                test_name="Unknown",
                status=TestStatus.ERROR,
                error=f"Test '{test_id}' not found in registry",
                agent_id=self.agent_id
            )

        registered = self._registered_tests[test_id]

        # Check prerequisites
        if registered.requires_mcp and not self.mcp_client:
            return TestResult(
                test_id=test_id,
                test_name=registered.test_name,
                status=TestStatus.SKIPPED,
                message="Skipped: MCP client not available",
                agent_id=self.agent_id
            )

        # Run the test
        return await self._execute_test(registered, **kwargs)

    async def run_suite(
        self,
        suite_name: str,
        **kwargs
    ) -> TestSuiteResult:
        """
        Run all tests in a suite.

        Args:
            suite_name: Name of the test suite
            **kwargs: Additional arguments for tests

        Returns:
            TestSuiteResult with all test outcomes
        """
        # Get tests for this suite
        tests = [
            t for t in self._registered_tests.values()
            if t.suite == suite_name
        ]

        result = TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(tests),
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )

        for test in tests:
            test_result = await self._execute_test(test, **kwargs)
            result.add_result(test_result)

            if self.fail_fast and not test_result.is_success():
                break

        result.completed_at = datetime.utcnow()
        return result

    async def run_all_tests(self, **kwargs) -> TestSuiteResult:
        """
        Run all registered tests.

        Returns:
            TestSuiteResult with all test outcomes
        """
        tests = list(self._registered_tests.values())

        result = TestSuiteResult(
            suite_name="all",
            total_tests=len(tests),
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )

        for test in tests:
            test_result = await self._execute_test(test, **kwargs)
            result.add_result(test_result)

            if self.fail_fast and not test_result.is_success():
                break

        result.completed_at = datetime.utcnow()
        return result

    async def run_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        **kwargs
    ) -> TestSuiteResult:
        """
        Run tests matching specified tags.

        Args:
            tags: Tags to match
            match_all: If True, test must have all tags; if False, any tag matches
            **kwargs: Additional arguments for tests

        Returns:
            TestSuiteResult with matching test outcomes
        """
        def matches(test: RegisteredTest) -> bool:
            if match_all:
                return all(t in test.tags for t in tags)
            else:
                return any(t in test.tags for t in tags)

        tests = [t for t in self._registered_tests.values() if matches(t)]

        result = TestSuiteResult(
            suite_name=f"tags:{','.join(tags)}",
            total_tests=len(tests),
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )

        for test in tests:
            test_result = await self._execute_test(test, **kwargs)
            result.add_result(test_result)

            if self.fail_fast and not test_result.is_success():
                break

        result.completed_at = datetime.utcnow()
        return result

    async def test_skill(self, skill_id: str) -> TestResult:
        """
        Test a specific DeepSkill.

        This runs the skill's self_test() method.
        """
        from src.agents.core import DeepSkillRegistry, SkillContext

        registry = DeepSkillRegistry()
        skill = registry.get(skill_id)

        if not skill:
            return TestResult(
                test_id=f"skill_test_{skill_id}",
                test_name=f"Skill Test: {skill_id}",
                status=TestStatus.ERROR,
                error=f"Skill '{skill_id}' not found in registry",
                agent_id=self.agent_id,
                skill_id=skill_id
            )

        start_time = datetime.utcnow()

        try:
            # Create context for skill testing
            context = SkillContext(
                parameters={},
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                mcp_client=self.mcp_client,
            )

            # Run skill self-test
            skill_result = await asyncio.wait_for(
                skill.self_test(context),
                timeout=skill.config.test_timeout_seconds
            )

            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            if skill_result.success:
                return TestResult(
                    test_id=f"skill_test_{skill_id}",
                    test_name=f"Skill Test: {skill.name}",
                    status=TestStatus.PASSED,
                    message=skill_result.data.get("message") if skill_result.data else None,
                    duration_ms=duration,
                    agent_id=self.agent_id,
                    skill_id=skill_id
                )
            else:
                return TestResult(
                    test_id=f"skill_test_{skill_id}",
                    test_name=f"Skill Test: {skill.name}",
                    status=TestStatus.FAILED,
                    error=skill_result.error,
                    duration_ms=duration,
                    agent_id=self.agent_id,
                    skill_id=skill_id
                )

        except asyncio.TimeoutError:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return TestResult(
                test_id=f"skill_test_{skill_id}",
                test_name=f"Skill Test: {skill_id}",
                status=TestStatus.TIMEOUT,
                error=f"Skill test timed out after {skill.config.test_timeout_seconds}s",
                duration_ms=duration,
                agent_id=self.agent_id,
                skill_id=skill_id
            )
        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return TestResult(
                test_id=f"skill_test_{skill_id}",
                test_name=f"Skill Test: {skill_id}",
                status=TestStatus.ERROR,
                error=str(e),
                error_type=type(e).__name__,
                traceback_str=traceback.format_exc(),
                duration_ms=duration,
                agent_id=self.agent_id,
                skill_id=skill_id
            )

    async def test_all_skills(self) -> TestSuiteResult:
        """
        Test all skills registered for this agent type.
        """
        from src.agents.core import DeepSkillRegistry

        registry = DeepSkillRegistry()
        skills = registry.get_for_agent(self.agent_type)

        result = TestSuiteResult(
            suite_name="skills",
            total_tests=len(skills),
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )

        for skill in skills:
            test_result = await self.test_skill(skill.skill_id)
            result.add_result(test_result)

            if self.fail_fast and not test_result.is_success():
                break

        result.completed_at = datetime.utcnow()
        return result

    async def _execute_test(
        self,
        test: RegisteredTest,
        **kwargs
    ) -> TestResult:
        """Execute a single test with error handling."""
        start_time = datetime.utcnow()

        try:
            # Run test with timeout
            result = await asyncio.wait_for(
                test.test_func(self, **kwargs),
                timeout=test.timeout_seconds
            )

            # Ensure duration is set
            if result.duration_ms == 0:
                result.duration_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )

            result.agent_id = self.agent_id
            return result

        except asyncio.TimeoutError:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return TestResult(
                test_id=test.test_id,
                test_name=test.test_name,
                status=TestStatus.TIMEOUT,
                error=f"Test timed out after {test.timeout_seconds}s",
                duration_ms=duration,
                agent_id=self.agent_id
            )

        except AssertionError as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return TestResult(
                test_id=test.test_id,
                test_name=test.test_name,
                status=TestStatus.FAILED,
                error=str(e) or "Assertion failed",
                error_type="AssertionError",
                duration_ms=duration,
                agent_id=self.agent_id
            )

        except Exception as e:
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(
                "test_execution_error",
                test_id=test.test_id,
                error=str(e)
            )
            return TestResult(
                test_id=test.test_id,
                test_name=test.test_name,
                status=TestStatus.ERROR,
                error=str(e),
                error_type=type(e).__name__,
                traceback_str=traceback.format_exc(),
                duration_ms=duration,
                agent_id=self.agent_id
            )

    def get_available_tests(self) -> Dict[str, List[str]]:
        """Get available tests organized by suite."""
        suites: Dict[str, List[str]] = {}
        for test in self._registered_tests.values():
            if test.suite not in suites:
                suites[test.suite] = []
            suites[test.suite].append(test.test_id)
        return suites

    @classmethod
    def get_all_registered_tests(cls) -> List[str]:
        """Get all registered test IDs."""
        return list(cls._registered_tests.keys())
