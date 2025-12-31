"""
MCP Client Implementation.

Provides connectivity to MCP servers using different transport protocols
(stdio, HTTP, SSE) for the OCI AI Agent Coordinator.

Security Note: This module uses asyncio.create_subprocess_exec() for spawning
MCP server processes. This is the safe alternative to shell-based execution
as it does NOT use shell interpolation - arguments are passed directly to
the executable without shell parsing, preventing command injection.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class TransportType(str, Enum):
    """MCP transport types."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


class MCPError(Exception):
    """MCP protocol error."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    server_id: str
    transport: TransportType
    # stdio transport - command must be a validated executable path
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None  # Working directory for stdio transport
    # HTTP/SSE transport
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # Common
    timeout_seconds: int = 120  # Increased default for large compartments
    retry_attempts: int = 3
    retry_backoff: float = 1.5  # Exponential backoff multiplier
    capabilities: dict[str, Any] = field(default_factory=dict)
    # Tool-specific timeouts
    tool_timeouts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "server_id": self.server_id,
            "transport": self.transport.value,
            "command": self.command,
            "args": self.args,
            "url": self.url,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_id: str
    namespace: str | None = None  # e.g., "oci_compute"

    @property
    def full_name(self) -> str:
        """Get namespaced tool name."""
        if self.namespace:
            return f"{self.namespace}:{self.name}"
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_id": self.server_id,
            "namespace": self.namespace,
        }


@dataclass
class ResourceDefinition:
    """Definition of an MCP resource."""

    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None
    server_id: str = ""


@dataclass
class ToolCallResult:
    """Result from executing an MCP tool."""

    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: int = 0


class MCPTransport(ABC):
    """Abstract base for MCP transport implementations."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._connected = False
        self._logger = logger.bind(server_id=config.server_id)

    @property
    def connected(self) -> bool:
        """Check if transport is connected."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""
        pass

    @abstractmethod
    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send a JSON-RPC request and get response."""
        pass


class StdioTransport(MCPTransport):
    """
    stdio transport for MCP servers.

    Spawns a subprocess and communicates via stdin/stdout using JSON-RPC.

    Security: Uses create_subprocess_exec which passes arguments directly
    to the executable WITHOUT shell interpolation, preventing injection.
    """

    # Allowlist of known safe MCP server commands
    ALLOWED_COMMANDS = {
        "python",
        "python3",
        "node",
        "npx",
        "uv",
        "uvx",
        "sql",  # SQLcl
    }

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    def _validate_command(self, command: str) -> str:
        """
        Validate the command is safe to execute.

        Only allows known MCP server executables from allowlist.
        """
        # Extract base command name
        base_cmd = os.path.basename(command)

        # Check against allowlist
        if base_cmd not in self.ALLOWED_COMMANDS:
            # Check if it's an absolute path to an allowed command
            allowed = False
            for allowed_cmd in self.ALLOWED_COMMANDS:
                if base_cmd == allowed_cmd or base_cmd.startswith(f"{allowed_cmd}."):
                    allowed = True
                    break

            if not allowed:
                raise MCPError(
                    -32600,
                    f"Command '{base_cmd}' not in allowed MCP server commands. "
                    f"Allowed: {self.ALLOWED_COMMANDS}",
                )

        return command

    async def connect(self) -> None:
        """Spawn the MCP server process using safe subprocess execution."""
        if self._connected:
            return

        if not self.config.command:
            raise MCPError(-32600, "No command specified for stdio transport")

        # Validate command against allowlist
        validated_command = self._validate_command(self.config.command)

        # Resolve working directory
        cwd = None
        if self.config.working_dir:
            cwd = os.path.expanduser(os.path.expandvars(self.config.working_dir))
            if not os.path.isdir(cwd):
                raise MCPError(-32600, f"Working directory does not exist: {cwd}")

        self._logger.info(
            "Starting MCP server process",
            command=validated_command,
            args=self.config.args,
            working_dir=cwd,
        )

        # Build environment - merge with current env
        env = {**os.environ, **self.config.env}

        # SECURITY: Using create_subprocess_exec (NOT create_subprocess_shell)
        # This passes arguments directly to the executable without shell parsing,
        # similar to Node.js execFile vs exec, preventing command injection.
        self._process = await asyncio.create_subprocess_exec(
            validated_command,
            *self.config.args,  # Arguments passed as list, not string
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,  # Set working directory
            # Set limit to 10MB to handle large MCP responses (e.g., many tools)
            limit=10 * 1024 * 1024,
        )

        # Start reader task
        self._reader_task = asyncio.create_task(self._read_responses())

        # Send initialize request
        result = await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {
                    "name": "oci-coordinator",
                    "version": "1.0.0",
                },
            },
        )

        self._connected = True
        self._logger.info("MCP server connected", capabilities=result.get("capabilities"))

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()

        self._connected = False
        self._logger.info("MCP server disconnected")

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send JSON-RPC request via stdin."""
        if not self._process or not self._process.stdin:
            raise MCPError(-32600, "Not connected")

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        # Send request
        request_data = json.dumps(request) + "\n"
        self._process.stdin.write(request_data.encode())
        await self._process.stdin.drain()

        # Wait for response
        try:
            response = await asyncio.wait_for(
                future, timeout=self.config.timeout_seconds
            )
            return response
        except asyncio.TimeoutError:
            del self._pending_requests[request_id]
            raise MCPError(-32000, f"Request timeout: {method}")

    async def _read_responses(self) -> None:
        """Read responses from stdout."""
        if not self._process or not self._process.stdout:
            return

        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break

                data = json.loads(line.decode())
                request_id = data.get("id")

                if request_id and request_id in self._pending_requests:
                    future = self._pending_requests.pop(request_id)
                    if "error" in data:
                        error = data["error"]
                        future.set_exception(
                            MCPError(
                                error.get("code", -32000),
                                error.get("message", "Unknown error"),
                                error.get("data"),
                            )
                        )
                    else:
                        future.set_result(data.get("result"))

            except json.JSONDecodeError:
                continue
            except asyncio.CancelledError:
                break


class HTTPTransport(MCPTransport):
    """
    HTTP transport for MCP servers.

    Sends JSON-RPC requests over HTTP POST.
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._client: Any = None
        self._request_id = 0

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._connected:
            return

        import httpx

        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            headers=self.config.headers,
        )

        # Test connection with initialize
        result = await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {
                    "name": "oci-coordinator",
                    "version": "1.0.0",
                },
            },
        )

        self._connected = True
        self._logger.info("HTTP MCP server connected")

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
        self._connected = False

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send JSON-RPC request over HTTP."""
        if not self._client:
            raise MCPError(-32600, "Not connected")

        self._request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        response = await self._client.post(
            self.config.url,
            json=request,
        )
        response.raise_for_status()

        data = response.json()
        if "error" in data:
            error = data["error"]
            raise MCPError(
                error.get("code", -32000),
                error.get("message", "Unknown error"),
            )

        return data.get("result")


class MCPClient:
    """
    High-level MCP client for interacting with a server.

    Provides tool discovery, execution, and resource access.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._transport: MCPTransport | None = None
        self._tools: dict[str, ToolDefinition] = {}
        self._resources: dict[str, ResourceDefinition] = {}
        self._logger = logger.bind(server_id=config.server_id)

    @property
    def connected(self) -> bool:
        """Check if client is connected."""
        return self._transport is not None and self._transport.connected

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        """Get discovered tools."""
        return self._tools

    @property
    def resources(self) -> dict[str, ResourceDefinition]:
        """Get discovered resources."""
        return self._resources

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self.connected:
            return

        # Create appropriate transport
        if self.config.transport == TransportType.STDIO:
            self._transport = StdioTransport(self.config)
        elif self.config.transport in (TransportType.HTTP, TransportType.SSE):
            self._transport = HTTPTransport(self.config)
        else:
            raise MCPError(-32600, f"Unknown transport: {self.config.transport}")

        await self._transport.connect()

        # Discover tools and resources
        await self._discover_tools()
        await self._discover_resources()

        self._logger.info(
            "MCP client connected",
            tools=len(self._tools),
            resources=len(self._resources),
        )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._transport:
            await self._transport.disconnect()
        self._tools.clear()
        self._resources.clear()

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        if not self._transport:
            return

        try:
            result = await self._transport.send_request("tools/list")
            tools = result.get("tools", [])

            for tool in tools:
                tool_def = ToolDefinition(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_id=self.config.server_id,
                )
                self._tools[tool["name"]] = tool_def

            self._logger.debug("Discovered tools", count=len(self._tools))

        except MCPError as e:
            self._logger.warning("Tool discovery failed", error=str(e))

    async def _discover_resources(self) -> None:
        """Discover available resources from the server."""
        if not self._transport:
            return

        try:
            result = await self._transport.send_request("resources/list")
            resources = result.get("resources", [])

            for resource in resources:
                resource_def = ResourceDefinition(
                    uri=resource["uri"],
                    name=resource.get("name", resource["uri"]),
                    description=resource.get("description"),
                    mime_type=resource.get("mimeType"),
                    server_id=self.config.server_id,
                )
                self._resources[resource["uri"]] = resource_def

            self._logger.debug("Discovered resources", count=len(self._resources))

        except MCPError as e:
            self._logger.debug("Resource discovery failed", error=str(e))

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """
        Execute a tool on the server with retry logic.

        Features:
        - Tool-specific timeouts for slow operations
        - Exponential backoff retry on timeout errors
        - Tracing integration for observability
        """
        import time

        # Get tracer for MCP operations - use observability module for consistency
        from src.observability import get_tracer
        tracer = get_tracer("mcp-executor")

        # Get tool-specific timeout or default
        timeout = self.config.tool_timeouts.get(
            tool_name, self.config.timeout_seconds
        )

        with tracer.start_as_current_span(f"mcp.tool.{tool_name}") as span:
            start_time = time.time()

            # Set span attributes for MCP tool call
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.server.id", self.config.server_id)
            span.set_attribute("mcp.tool.arguments", str(arguments)[:500])
            span.set_attribute("mcp.timeout_seconds", timeout)

            if not self._transport:
                span.set_attribute("mcp.error", True)
                span.set_attribute("mcp.error_type", "not_connected")
                return ToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    error="Not connected",
                )

            # Retry loop with exponential backoff
            last_error = None
            for attempt in range(self.config.retry_attempts):
                try:
                    span.set_attribute("mcp.attempt", attempt + 1)

                    # Use tool-specific timeout
                    result = await asyncio.wait_for(
                        self._transport.send_request(
                            "tools/call",
                            {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        ),
                        timeout=timeout,
                    )

                    duration_ms = int((time.time() - start_time) * 1000)
                    span.set_attribute("mcp.duration_ms", duration_ms)
                    span.set_attribute("mcp.success", True)

                    # Extract content from result
                    content = result.get("content", [])
                    if content and len(content) > 0:
                        # Get text content
                        text_content = next(
                            (c.get("text") for c in content if c.get("type") == "text"),
                            None,
                        )
                        result_preview = str(text_content or content)[:200]
                        span.set_attribute("mcp.result_preview", result_preview)
                        return ToolCallResult(
                            tool_name=tool_name,
                            success=True,
                            result=text_content or content,
                            duration_ms=duration_ms,
                        )

                    span.set_attribute("mcp.result_type", "raw")
                    return ToolCallResult(
                        tool_name=tool_name,
                        success=True,
                        result=result,
                        duration_ms=duration_ms,
                    )

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {timeout}s (attempt {attempt + 1}/{self.config.retry_attempts})"
                    self._logger.warning(
                        "Tool call timeout, retrying",
                        tool=tool_name,
                        attempt=attempt + 1,
                        timeout=timeout,
                    )
                    # Exponential backoff before retry
                    if attempt < self.config.retry_attempts - 1:
                        backoff = (self.config.retry_backoff ** attempt) * 2
                        await asyncio.sleep(backoff)
                        # Increase timeout for retry
                        timeout = int(timeout * 1.5)

                except MCPError as e:
                    # Don't retry non-timeout MCP errors
                    if e.code == -32000 and "timeout" in str(e).lower():
                        last_error = str(e)
                        if attempt < self.config.retry_attempts - 1:
                            backoff = (self.config.retry_backoff ** attempt) * 2
                            await asyncio.sleep(backoff)
                            timeout = int(timeout * 1.5)
                        continue
                    # Other errors - fail immediately
                    duration_ms = int((time.time() - start_time) * 1000)
                    span.set_attribute("mcp.duration_ms", duration_ms)
                    span.set_attribute("mcp.error", True)
                    span.set_attribute("mcp.error_message", str(e)[:200])
                    return ToolCallResult(
                        tool_name=tool_name,
                        success=False,
                        error=str(e),
                        duration_ms=duration_ms,
                    )

            # All retries exhausted
            duration_ms = int((time.time() - start_time) * 1000)
            span.set_attribute("mcp.duration_ms", duration_ms)
            span.set_attribute("mcp.error", True)
            span.set_attribute("mcp.error_message", last_error or "Unknown error")
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool Error: {last_error}",
                duration_ms=duration_ms,
            )

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the server."""
        if not self._transport:
            raise MCPError(-32600, "Not connected")

        result = await self._transport.send_request(
            "resources/read",
            {"uri": uri},
        )

        contents = result.get("contents", [])
        if contents:
            return contents[0].get("text") or contents[0].get("blob")
        return None
