"""
Mock Objects for Agent-Based Testing.

Provides mock implementations of MCP clients and contexts
for isolated testing without backend connectivity.

Key components:
- MockMCPClient: Simulates MCP tool calls with configurable responses
- MockResponse: Pre-defined response objects
- create_mock_context: Factory for creating test contexts

Example usage:
    from src.agents.testing import MockMCPClient, create_mock_context

    # Create a mock client with responses
    mock = MockMCPClient()
    mock.add_response("oci_database_ping", {"status": "ok"})

    # Create test context
    context = create_mock_context(mcp_client=mock)

    # Use in tests
    result = await context.mcp_client.call_tool("oci_database_ping", {})
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import asyncio
import structlog

logger = structlog.get_logger()


class MockResponseType(Enum):
    """Types of mock responses."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"


@dataclass
class MockResponse:
    """
    Pre-configured mock response for tool calls.

    Supports various response types:
    - SUCCESS: Returns the configured data
    - ERROR: Returns an error response
    - TIMEOUT: Simulates a timeout
    - EXCEPTION: Raises an exception
    """

    response_type: MockResponseType = MockResponseType.SUCCESS
    data: Any = None
    error_message: Optional[str] = None
    delay_seconds: float = 0.0  # Simulate latency

    # For dynamic responses
    response_fn: Optional[Callable[[Dict[str, Any]], Any]] = None

    def __post_init__(self):
        """Set default data for success responses."""
        if self.response_type == MockResponseType.SUCCESS and self.data is None:
            self.data = {"status": "success"}

    @classmethod
    def success(cls, data: Any = None) -> "MockResponse":
        """Create a success response."""
        return cls(response_type=MockResponseType.SUCCESS, data=data)

    @classmethod
    def error(cls, message: str = "Mock error") -> "MockResponse":
        """Create an error response."""
        return cls(
            response_type=MockResponseType.ERROR,
            error_message=message
        )

    @classmethod
    def timeout(cls, delay: float = 5.0) -> "MockResponse":
        """Create a timeout response."""
        return cls(
            response_type=MockResponseType.TIMEOUT,
            delay_seconds=delay
        )

    @classmethod
    def exception(cls, message: str = "Mock exception") -> "MockResponse":
        """Create an exception-raising response."""
        return cls(
            response_type=MockResponseType.EXCEPTION,
            error_message=message
        )

    @classmethod
    def dynamic(cls, fn: Callable[[Dict[str, Any]], Any]) -> "MockResponse":
        """Create a dynamic response based on input parameters."""
        return cls(
            response_type=MockResponseType.SUCCESS,
            response_fn=fn
        )


@dataclass
class ToolCallRecord:
    """Record of a tool call made to the mock client."""

    tool_name: str
    params: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0


class MockMCPClient:
    """
    Mock MCP client for testing without backend connectivity.

    Features:
    - Configure responses per tool
    - Record all calls for assertions
    - Simulate errors and timeouts
    - Dynamic responses based on parameters

    Example:
        mock = MockMCPClient()

        # Add specific responses
        mock.add_response("oci_database_ping", {"status": "ok"})
        mock.add_response("oci_compute_list", MockResponse.error("Not found"))

        # Add dynamic response
        mock.add_response("oci_database_execute_sql", MockResponse.dynamic(
            lambda params: {"rows": [{"id": 1}]} if "SELECT" in params.get("sql", "") else {}
        ))

        # Use the mock
        result = await mock.call_tool("oci_database_ping", {})
        assert result == {"status": "ok"}

        # Check call history
        assert mock.was_called("oci_database_ping")
        assert mock.call_count("oci_database_ping") == 1
    """

    def __init__(self, default_response: Optional[MockResponse] = None):
        """
        Initialize mock client.

        Args:
            default_response: Response for tools without configured responses
        """
        self._responses: Dict[str, MockResponse] = {}
        self._call_history: List[ToolCallRecord] = []
        self._default_response = default_response or MockResponse.success({"status": "mock"})

    def add_response(
        self,
        tool_name: str,
        response: Union[MockResponse, Dict[str, Any], Any]
    ) -> "MockMCPClient":
        """
        Add a response for a tool.

        Args:
            tool_name: Name of the tool
            response: MockResponse or raw data (will be wrapped)

        Returns:
            self for chaining
        """
        if isinstance(response, MockResponse):
            self._responses[tool_name] = response
        else:
            self._responses[tool_name] = MockResponse.success(response)
        return self

    def add_responses(self, responses: Dict[str, Any]) -> "MockMCPClient":
        """
        Add multiple responses.

        Args:
            responses: Dict mapping tool names to responses

        Returns:
            self for chaining
        """
        for tool_name, response in responses.items():
            self.add_response(tool_name, response)
        return self

    async def call_tool(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Simulate a tool call.

        Args:
            tool_name: Name of the tool to call
            params: Tool parameters

        Returns:
            Configured response data

        Raises:
            Exception: If configured to raise an exception
            asyncio.TimeoutError: If configured to timeout
        """
        params = params or {}
        start_time = datetime.utcnow()

        # Get response config
        response = self._responses.get(tool_name, self._default_response)

        # Record the call
        record = ToolCallRecord(
            tool_name=tool_name,
            params=params.copy(),
            timestamp=start_time
        )

        try:
            # Apply delay
            if response.delay_seconds > 0:
                await asyncio.sleep(response.delay_seconds)

            # Handle response types
            if response.response_type == MockResponseType.TIMEOUT:
                record.error = "Timeout"
                raise asyncio.TimeoutError(f"Mock timeout for {tool_name}")

            if response.response_type == MockResponseType.EXCEPTION:
                record.error = response.error_message
                raise Exception(response.error_message or "Mock exception")

            if response.response_type == MockResponseType.ERROR:
                result = {
                    "error": True,
                    "message": response.error_message or "Mock error"
                }
                record.response = result
                return result

            # SUCCESS response
            if response.response_fn:
                result = response.response_fn(params)
            else:
                result = response.data

            record.response = result
            return result

        finally:
            # Calculate duration
            record.duration_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            self._call_history.append(record)

    def get_call_history(self) -> List[ToolCallRecord]:
        """Get all recorded calls."""
        return self._call_history.copy()

    def get_calls_for_tool(self, tool_name: str) -> List[ToolCallRecord]:
        """Get calls for a specific tool."""
        return [c for c in self._call_history if c.tool_name == tool_name]

    def was_called(self, tool_name: str) -> bool:
        """Check if a tool was called."""
        return any(c.tool_name == tool_name for c in self._call_history)

    def call_count(self, tool_name: Optional[str] = None) -> int:
        """
        Get call count.

        Args:
            tool_name: If provided, count for specific tool; otherwise total

        Returns:
            Number of calls
        """
        if tool_name:
            return len(self.get_calls_for_tool(tool_name))
        return len(self._call_history)

    def last_call(self, tool_name: Optional[str] = None) -> Optional[ToolCallRecord]:
        """Get the last call (optionally for a specific tool)."""
        history = self.get_calls_for_tool(tool_name) if tool_name else self._call_history
        return history[-1] if history else None

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()

    def reset(self) -> None:
        """Reset all responses and history."""
        self._responses.clear()
        self._call_history.clear()


@dataclass
class MockCodeExecutionResult:
    """Mock result from code execution."""
    success: bool = True
    result: Any = None
    output: str = ""
    error: Optional[str] = None
    execution_time_ms: int = 10


class MockCodeExecutor:
    """
    Mock code executor for testing.

    Simulates code execution without actually running code.
    """

    def __init__(self, default_result: Any = None):
        self._results: Dict[str, Any] = {}
        self._default_result = default_result
        self._call_history: List[Dict[str, Any]] = []

    def add_result(self, code_pattern: str, result: Any) -> "MockCodeExecutor":
        """
        Add a result for code matching a pattern.

        Args:
            code_pattern: Substring to match in code
            result: Result to return

        Returns:
            self for chaining
        """
        self._results[code_pattern] = result
        return self

    async def execute(
        self,
        code: str,
        variables: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> MockCodeExecutionResult:
        """Simulate code execution."""
        self._call_history.append({
            "code": code,
            "variables": variables,
            "timeout": timeout_seconds
        })

        # Check for matching pattern
        for pattern, result in self._results.items():
            if pattern in code:
                return MockCodeExecutionResult(
                    success=True,
                    result=result
                )

        # Return default
        return MockCodeExecutionResult(
            success=True,
            result=self._default_result
        )

    def was_called_with(self, code_substring: str) -> bool:
        """Check if execute was called with code containing substring."""
        return any(
            code_substring in call["code"]
            for call in self._call_history
        )

    def clear_history(self) -> None:
        """Clear execution history."""
        self._call_history.clear()


@dataclass
class MockSkillContext:
    """
    Mock context for skill execution.

    Provides all dependencies a skill needs in a mock form.
    """

    mcp_client: MockMCPClient
    code_executor: MockCodeExecutor
    agent_id: str = "test_agent"
    session_id: str = "test_session"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_params(self, **kwargs) -> "MockSkillContext":
        """Create new context with additional parameters."""
        new_params = {**self.parameters, **kwargs}
        return MockSkillContext(
            mcp_client=self.mcp_client,
            code_executor=self.code_executor,
            agent_id=self.agent_id,
            session_id=self.session_id,
            parameters=new_params,
            metadata=self.metadata.copy()
        )


def create_mock_context(
    mcp_client: Optional[MockMCPClient] = None,
    code_executor: Optional[MockCodeExecutor] = None,
    agent_id: str = "test_agent",
    session_id: str = "test_session",
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    # Convenience: pre-configure common responses
    ping_response: Optional[Dict[str, Any]] = None,
    health_response: Optional[Dict[str, Any]] = None,
) -> MockSkillContext:
    """
    Factory function to create a mock skill context.

    Args:
        mcp_client: Mock MCP client (created if not provided)
        code_executor: Mock code executor (created if not provided)
        agent_id: Agent identifier
        session_id: Session identifier
        parameters: Skill parameters
        metadata: Additional metadata
        ping_response: Pre-configured response for ping tools
        health_response: Pre-configured response for health tools

    Returns:
        MockSkillContext ready for testing

    Example:
        # Basic context
        ctx = create_mock_context()

        # With pre-configured responses
        ctx = create_mock_context(
            ping_response={"status": "ok"},
            health_response={"healthy": True}
        )

        # Custom MCP client
        mock = MockMCPClient()
        mock.add_response("my_tool", {"data": "test"})
        ctx = create_mock_context(mcp_client=mock)
    """
    # Create defaults
    if mcp_client is None:
        mcp_client = MockMCPClient()

    if code_executor is None:
        code_executor = MockCodeExecutor()

    # Add convenience responses
    if ping_response:
        for tool in ["oci_database_ping", "oci_cost_ping", "oci_security_ping"]:
            mcp_client.add_response(tool, ping_response)

    if health_response:
        for tool in ["oci_database_health_check", "oci_security_health", "oci_logan_health"]:
            mcp_client.add_response(tool, health_response)

    return MockSkillContext(
        mcp_client=mcp_client,
        code_executor=code_executor,
        agent_id=agent_id,
        session_id=session_id,
        parameters=parameters or {},
        metadata=metadata or {}
    )


# =============================================================================
# Pre-built Mock Responses for Common Tools
# =============================================================================

COMMON_MOCK_RESPONSES = {
    # Ping/Health
    "oci_database_ping": {"status": "pong", "timestamp": "2024-01-01T00:00:00Z"},
    "oci_cost_ping": {"status": "ok"},
    "oci_security_ping": {"status": "healthy"},
    "oci_logan_health": {"status": "ok", "detail": False},

    # Database tools
    "oci_database_health_check": {
        "status": "healthy",
        "cache_size": 100,
        "connections": 1
    },
    "oci_database_list_connections": {
        "connections": [
            {"name": "default", "status": "available"}
        ]
    },
    "oci_database_cache_status": {
        "entries": 50,
        "hit_rate": 0.85
    },

    # OPSI tools
    "oci_opsi_get_fleet_summary": {
        "total_databases": 10,
        "by_type": {"AUTONOMOUS": 5, "EXTERNAL": 5},
        "by_status": {"AVAILABLE": 8, "STOPPED": 2}
    },
    "oci_opsi_list_skills": {
        "skills": [
            {"name": "analyze_cpu", "tier": 2},
            {"name": "get_fleet_summary", "tier": 1}
        ]
    },

    # FinOps tools
    "oci_cost_templates": {
        "templates": [
            {"name": "monthly_summary", "description": "Monthly cost summary"}
        ]
    },
    "finops_list_providers": {
        "providers": [{"name": "oci", "enabled": True}]
    },

    # Infrastructure tools
    "oci_list_domains": {
        "domains": ["compute", "network", "database", "observability"]
    },
    "oci_get_cache_stats": {
        "total_requests": 1000,
        "cache_hits": 850,
        "hit_rate": 0.85
    },
}


def create_fully_mocked_client() -> MockMCPClient:
    """
    Create a mock client pre-configured with common responses.

    Returns:
        MockMCPClient with all common tool responses configured
    """
    client = MockMCPClient()
    client.add_responses(COMMON_MOCK_RESPONSES)
    return client


def create_database_test_context() -> MockSkillContext:
    """Create a context configured for database testing."""
    client = create_fully_mocked_client()

    # Add database-specific responses
    client.add_response("oci_database_execute_sql", MockResponse.dynamic(
        lambda params: {
            "success": True,
            "rows": [],
            "columns": ["id", "name"],
            "row_count": 0
        }
    ))

    return create_mock_context(mcp_client=client, agent_id="db_troubleshoot")


def create_finops_test_context() -> MockSkillContext:
    """Create a context configured for FinOps testing."""
    client = create_fully_mocked_client()

    # Add finops-specific responses
    client.add_response("oci_cost_by_compartment", {
        "total_cost": 1000.00,
        "currency": "USD",
        "compartments": []
    })

    return create_mock_context(mcp_client=client, agent_id="finops")
