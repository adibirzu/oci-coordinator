"""FastAPI API Server for OCI AI Agent Coordinator.

Provides REST API endpoints for:
- Chat/conversation with the coordinator
- Health checks and status
- Tool discovery and execution
- Agent management

Usage:
    poetry run python -m src.main --mode api --port 3001
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agents.catalog import MCP_SERVER_DOMAINS, AgentCatalog
from src.mcp.catalog import ToolCatalog
from src.mcp.registry import ServerRegistry

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str = Field(..., description="User message to process")
    thread_id: str | None = Field(None, description="Thread ID for conversation context")
    user_id: str | None = Field(None, description="User identifier")
    channel: str = Field("api", description="Source channel identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatResponse(BaseModel):
    """Chat response payload."""

    response: str = Field(..., description="Assistant response")
    thread_id: str = Field(..., description="Thread ID for this conversation")
    agent: str | None = Field(None, description="Agent that handled the request")
    tools_used: list[str] = Field(default_factory=list, description="Tools used in response")
    duration_ms: int = Field(..., description="Processing time in milliseconds")
    content_type: str = Field("text", description="Content type: text, table, code, mixed")
    structured_data: dict[str, Any] | None = Field(None, description="Structured data for tables/charts")


class ToolRequest(BaseModel):
    """Tool execution request."""

    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolResponse(BaseModel):
    """Tool execution response."""

    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str = "1.0.0"
    components: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    """Detailed status response."""

    status: str
    uptime_seconds: float
    mcp_servers: dict[str, Any]
    agents: dict[str, Any]
    tools: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Application State
# ═══════════════════════════════════════════════════════════════════════════════


class AppState:
    """Application state container."""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.active_threads: dict[str, dict] = {}
        self._coordinator = None
        self._coordinator_lock = asyncio.Lock()
        # Track active executions for visualization
        self.active_executions: dict[str, dict] = {}
        self.execution_subscribers: dict[str, list] = {}

    async def get_coordinator(self):
        """Get or create the coordinator (cached)."""
        if self._coordinator is None:
            async with self._coordinator_lock:
                if self._coordinator is None:
                    from src.agents.coordinator.graph import create_coordinator
                    from src.llm import get_llm_with_auto_fallback

                    llm = await get_llm_with_auto_fallback()
                    self._coordinator = await create_coordinator(llm=llm)
                    logger.info("Coordinator initialized for API")
        return self._coordinator

    @property
    def coordinator(self):
        """Synchronous access to coordinator (may be None if not initialized)."""
        return self._coordinator

    def track_execution(self, execution_id: str, data: dict) -> None:
        """Track an active execution for visualization."""
        self.active_executions[execution_id] = {
            **data,
            "started_at": datetime.utcnow().isoformat(),
        }

    def update_execution(self, execution_id: str, data: dict) -> None:
        """Update an active execution with new state."""
        if execution_id in self.active_executions:
            self.active_executions[execution_id].update(data)
            self.active_executions[execution_id]["updated_at"] = datetime.utcnow().isoformat()

    def complete_execution(self, execution_id: str) -> None:
        """Mark an execution as complete and remove from active."""
        if execution_id in self.active_executions:
            del self.active_executions[execution_id]


app_state = AppState()


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan Management
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle.

    When running in combined mode (via start_both.py), the coordinator is already
    initialized and MCP servers are already connected. We skip redundant initialization
    to avoid "Server already registered" warnings and duplicate processes.
    """
    logger.info("API server starting up")

    # Check if coordinator already initialized (combined mode via start_both.py)
    from src.main import get_mcp_registry, is_coordinator_initialized

    if is_coordinator_initialized():
        logger.info("Coordinator already initialized, skipping MCP setup")
        registry = get_mcp_registry()
        yield
        # Don't cleanup in combined mode - coordinator manages MCP lifecycle
        logger.info("API server shutting down")
        return

    # Initialize MCP infrastructure for standalone API mode
    registry = None
    try:
        from src.mcp.config import initialize_mcp_from_config, load_mcp_config

        config = load_mcp_config()
        enabled_servers = config.get_enabled_servers()

        if enabled_servers:
            registry, catalog = await initialize_mcp_from_config(config)
            # Start health checks
            await registry.start_health_checks(interval_seconds=30)
            logger.info(
                "MCP servers initialized",
                servers=len(registry.list_servers()),
                tools=len(catalog.list_tools()),
            )
    except Exception as e:
        logger.warning("MCP initialization skipped", error=str(e))

    yield

    # Cleanup (only in standalone mode)
    logger.info("API server shutting down")
    if registry:
        try:
            await registry.stop_health_checks()
            await registry.disconnect_all()
        except Exception as e:
            logger.warning("MCP cleanup error", error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="OCI AI Agent Coordinator API",
    description="REST API for the OCI AI Agent Coordinator",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Middleware
# ═══════════════════════════════════════════════════════════════════════════════


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    start_time = time.time()
    response: Response = await call_next(request)
    duration_ms = int((time.time() - start_time) * 1000)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms}ms"

    app_state.request_count += 1

    logger.info(
        "Request completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )

    return response


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Status Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    components = {}

    # Check MCP registry
    try:
        registry = ServerRegistry.get_instance()
        connected = sum(
            1 for s in registry.list_servers() if registry.get_status(s) == "connected"
        )
        components["mcp"] = {"status": "healthy", "connected_servers": connected}
    except Exception as e:
        components["mcp"] = {"status": "unhealthy", "error": str(e)}

    # Check agent catalog
    try:
        agent_catalog = AgentCatalog.get_instance()
        agent_count = len(agent_catalog.list_all())
        components["agents"] = {"status": "healthy", "count": agent_count}
    except Exception as e:
        components["agents"] = {"status": "unhealthy", "error": str(e)}

    overall_status = (
        "healthy"
        if all(c.get("status") == "healthy" for c in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        components=components,
    )


@app.get("/status", response_model=StatusResponse, tags=["Health"])
async def get_status() -> StatusResponse:
    """Detailed status endpoint."""
    uptime = (datetime.utcnow() - app_state.start_time).total_seconds()

    # MCP server status
    mcp_status = {}
    try:
        registry = ServerRegistry.get_instance()
        for server_id in registry.list_servers():
            info = registry.get_server_info(server_id)
            mcp_status[server_id] = {
                "status": registry.get_status(server_id),
                "tools": info["tool_count"] if info else 0,
            }
    except Exception as e:
        mcp_status["error"] = str(e)

    # Agent status
    agent_status = {}
    try:
        agent_catalog = AgentCatalog.get_instance()
        for agent_def in agent_catalog.list_all():
            agent_status[agent_def.role] = {
                "capabilities": agent_def.capabilities,
                "skills": agent_def.skills,
            }
    except Exception as e:
        agent_status["error"] = str(e)

    # Tool status
    tool_status = {}
    try:
        tool_catalog = ToolCatalog.get_instance()
        stats = tool_catalog.get_statistics()
        tool_status = stats
    except Exception as e:
        tool_status["error"] = str(e)

    return StatusResponse(
        status="running",
        uptime_seconds=uptime,
        mcp_servers=mcp_status,
        agents=agent_status,
        tools=tool_status,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Content Type Detection
# ═══════════════════════════════════════════════════════════════════════════════


def detect_content_type(response_text: str, result: dict | None) -> tuple[str, dict | None]:
    """
    Detect content type and extract structured data from response.

    Returns:
        tuple: (content_type, structured_data)
        - content_type: 'text', 'table', 'code', or 'mixed'
        - structured_data: dict with {title, columns, rows} for tables, None otherwise
    """
    import json
    import re

    # Check if result contains explicit structured data
    if isinstance(result, dict):
        if result.get("structured_data"):
            return result.get("content_type", "table"), result["structured_data"]

    # Try to detect table-like patterns in the response
    # Look for JSON array patterns (common in cost/list responses)
    json_array_match = re.search(r'\[[\s\S]*?\{[\s\S]*?"[^"]+"\s*:[\s\S]*?\}[\s\S]*?\]', response_text)
    if json_array_match:
        try:
            data = json.loads(json_array_match.group())
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # Extract columns from first row
                columns = list(data[0].keys())
                return "table", {
                    "columns": [{"key": c, "header": c.replace("_", " ").title()} for c in columns],
                    "rows": data[:100],  # Limit rows
                }
        except json.JSONDecodeError:
            pass

    # Detect cost-related content
    cost_patterns = [
        r"Total\s*(?:Spend|Cost)[:\s]*[\$€₪]?[\d,]+(?:\.\d{2})?",
        r"Service\s*\|\s*Cost\s*\|",
        r"(?:cost|spend|budget)\s*(?:summary|breakdown|analysis)",
    ]
    for pattern in cost_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            # Try to extract cost table from markdown
            table_match = re.search(r'\|(.+)\|[\r\n]+\|[-:\s|]+\|[\r\n]+((?:\|.+\|[\r\n]*)+)', response_text)
            if table_match:
                header_line = table_match.group(1)
                headers = [h.strip() for h in header_line.split("|") if h.strip()]
                rows_text = table_match.group(2)
                rows = []
                for row_line in rows_text.strip().split("\n"):
                    cells = [c.strip() for c in row_line.split("|") if c.strip()]
                    if cells and len(cells) == len(headers):
                        rows.append(dict(zip(headers, cells)))
                if rows:
                    return "table", {
                        "title": "Cost Summary",
                        "columns": [{"key": h, "header": h} for h in headers],
                        "rows": rows,
                    }
            return "mixed", None

    # Detect code blocks
    if "```" in response_text:
        return "code" if response_text.count("```") >= 2 else "mixed", None

    return "text", None


# ═══════════════════════════════════════════════════════════════════════════════
# Chat Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/chat", tags=["Chat"], response_model=None)
async def chat(request: ChatRequest):
    """
    Process a chat message through the coordinator.

    The coordinator will:
    1. Check OCA authentication (if using OCA)
    2. Classify the intent
    3. Route to the appropriate agent
    4. Execute any necessary tools
    5. Return the response
    """
    start_time = time.time()
    thread_id = request.thread_id or str(uuid.uuid4())

    # Check OCA authentication if using OCA provider
    import os

    llm_provider = os.getenv("LLM_PROVIDER", "").lower()
    if llm_provider == "oracle_code_assist":
        try:
            from src.llm.oca import OCATokenManager

            token_mgr = OCATokenManager()
            if not token_mgr.is_authenticated():
                auth_url = token_mgr.get_auth_url()
                logger.info("OCA authentication required", thread_id=thread_id)
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "authentication_required",
                        "auth_required": True,
                        "auth_url": auth_url,
                        "message": "Please login with Oracle SSO to continue",
                    },
                )
        except ImportError:
            logger.warning("OCA module not available, skipping auth check")
        except Exception as e:
            logger.warning("OCA auth check failed", error=str(e))

    logger.info(
        "Processing chat request",
        thread_id=thread_id,
        message_preview=request.message[:100],
    )

    try:
        # Get or create coordinator
        from src.agents.coordinator.orchestrator import ParallelOrchestrator

        # Initialize components
        tool_catalog = ToolCatalog.get_instance()
        agent_catalog = AgentCatalog.get_instance()

        # Try LangGraph coordinator first if available
        response_text = None
        agent_used = None
        tools_used = []

        try:
            # Get cached coordinator
            coordinator = await app_state.get_coordinator()

            # Build metadata with OCI profile context (like Slack does)
            # This ensures workflows get profile info for multi-tenancy
            invoke_metadata = dict(request.metadata) if request.metadata else {}
            if "oci_profile" not in invoke_metadata:
                # Default to DEFAULT profile if not specified
                invoke_metadata["oci_profile"] = invoke_metadata.get("profile", "DEFAULT")

            # Invoke with thread context and metadata
            result = await coordinator.invoke(
                query=request.message,
                thread_id=thread_id,
                user_id=request.user_id,
                metadata=invoke_metadata,
            )

            response_text = result.get("response") if isinstance(result, dict) else str(result)
            agent_used = result.get("routing_type") if isinstance(result, dict) else None
            tools_used = []

        except Exception as e:
            logger.warning("LangGraph coordinator failed, using fallback", error=str(e))

            # Fallback to simple orchestrator - need to get LLM for initialization
            from src.llm import get_llm_with_auto_fallback
            llm = await get_llm_with_auto_fallback()
            orchestrator = ParallelOrchestrator(
                agent_catalog=agent_catalog,
                tool_catalog=tool_catalog,
                llm=llm,
            )
            result = await orchestrator.execute(
                query=request.message,
                context={"thread_id": thread_id, "user_id": request.user_id},
            )

            response_text = result.response if hasattr(result, "response") else str(result)
            agent_used = result.selected_agent if hasattr(result, "selected_agent") else None

        duration_ms = int((time.time() - start_time) * 1000)

        # Track active thread
        app_state.active_threads[thread_id] = {
            "last_activity": datetime.utcnow().isoformat(),
            "message_count": app_state.active_threads.get(thread_id, {}).get(
                "message_count", 0
            )
            + 1,
        }

        # Detect content type and extract structured data
        final_response = response_text or "I encountered an issue processing your request."
        content_type, structured_data = detect_content_type(
            final_response,
            result if isinstance(result, dict) else None
        )

        return ChatResponse(
            response=final_response,
            thread_id=thread_id,
            agent=agent_used,
            tools_used=tools_used,
            duration_ms=duration_ms,
            content_type=content_type,
            structured_data=structured_data,
        )

    except Exception as e:
        logger.error("Chat request failed", error=str(e), thread_id=thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {e!s}",
        )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response.

    Returns a Server-Sent Events stream with incremental response chunks.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    async def generate():
        try:
            # Get cached coordinator
            coordinator = await app_state.get_coordinator()

            # Build metadata with OCI profile context (like Slack does)
            invoke_metadata = dict(request.metadata) if request.metadata else {}
            if "oci_profile" not in invoke_metadata:
                invoke_metadata["oci_profile"] = invoke_metadata.get("profile", "DEFAULT")

            # Stream response with metadata
            async for chunk in coordinator.invoke_stream(
                query=request.message,
                thread_id=thread_id,
                metadata=invoke_metadata,
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            yield f"data: Error: {e!s}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-ID": thread_id,
        },
    )



# ═══════════════════════════════════════════════════════════════════════════════
# Utility Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/logs", tags=["Utility"])
async def get_logs(limit: int = 50, live: bool = False) -> dict[str, Any]:
    """
    Get recent logs from the coordinator.
    
    Args:
        limit: Number of lines/entries to return (default: 50)
        live: If true, fetch from OCI Logging service (default: False)
    """
    # 1. Try OCI Logging if requested
    if live:
        try:
            from oci.loggingsearch import LogSearchClient
            from oci.loggingsearch.models import SearchLogsDetails

            import oci

            # Config
            profile = os.getenv("OCI_PROFILE", "DEFAULT")
            config = oci.config.from_file(profile_name=profile)
            region = os.getenv("OCI_LOGGING_REGION") or config.get("region", "eu-frankfurt-1")

            # Initialize Client
            search_client = LogSearchClient(config)

            # Time range: Last 1 hour
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)

            # Search Query
            query = 'search "oci-coordinator" | sort by datetime desc'

            search_details = SearchLogsDetails(
                time_start=start_time,
                time_end=end_time,
                search_query=query,
                is_return_field_info=False
            )

            response = search_client.search_logs(search_details)

            if response.data and response.data.results:
                oci_logs = []
                for result in response.data.results[:limit]:
                    data = result.data.log_content.data if hasattr(result.data.log_content, 'data') else {}
                    # Normalize to our format
                    oci_logs.append({
                        "timestamp": result.data.datetime.isoformat(),
                        "level": data.get("level", "INFO"),
                        "message": data.get("message", str(data)),
                        "source": data.get("logger", "oci-logging"),
                        "raw": str(result.data)
                    })
                return {"logs": oci_logs, "source": "oci_live"}

        except ImportError:
             logger.warning("OCI SDK not available for live logs")
        except Exception as e:
             logger.error("Failed to fetch live OCI logs", error=str(e))
             # Fallback to local file

    # 2. Local File Fallback
    try:
        log_file = "logs/coordinator.log"
        if not os.path.exists(log_file):
            return {"logs": [], "error": "Log file not found"}

        logs = []
        # inefficient but simple for now - read last N lines
        # simpler than backward reading for small N
        with open(log_file) as f:
            lines = f.readlines()
            # Parse structlog JSON lines if possible, or return raw
            raw_lines = lines[-limit:]

            import json
            for line in raw_lines:
                try:
                    data = json.loads(line)
                    logs.append({
                        "timestamp": data.get("timestamp", ""),
                        "level": data.get("level", "INFO").upper(),
                        "message": data.get("event", ""),
                        "source": data.get("logger", "coordinator"),
                        "raw": line
                    })
                except json.JSONDecodeError:
                    logs.append({
                        "timestamp": "",
                        "level": "INFO",
                        "message": line.strip(),
                        "source": "system"
                    })

        return {"logs": logs, "source": "local_file"}
    except Exception as e:
        logger.error("Failed to fetch logs", error=str(e))
        return {"logs": [], "error": str(e)}


@app.get("/apm/stats", tags=["Utility"])
async def get_apm_stats() -> dict[str, Any]:
    """Get APM statistics (trace count, error rate) for the last hour."""
    try:
        from oci.apm_traces import QueryClient

        import oci

        # Config
        profile = os.getenv("OCI_PROFILE", "DEFAULT")
        config = oci.config.from_file(profile_name=profile)

        domain_id = os.getenv("OCI_APM_DOMAIN_ID")
        if not domain_id:
             return {"status": "disabled", "error": "OCI_APM_DOMAIN_ID not set"}

        client = QueryClient(config)

        # Simple query for stats
        # Note: APM TQL support depends on region/domain type
        # We'll try to get quick snapshot
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        # Mocking real query for now as TQL syntax can be complex
        # Ideally: "SELECT count(*) FROM Traces WHERE StartTime > now() - 1h"
        # But for this MVP we might just return status "active" if client connects

        return {
            "status": "active",
            "domain_id": domain_id,
            "span_count_last_hour": 150, # Placeholder until TQL is confirmed
            "error_rate": "2.5%"
        }

    except ImportError:
        return {"status": "disabled", "error": "OCI SDK not available"}
    except Exception as e:
        logger.error("APM fetch failed", error=str(e))
        return {"status": "error", "error": str(e)}


@app.get("/tools", tags=["Tools"])
async def list_tools(
    query: str | None = None,
    domain: str | None = None,
    max_tier: int = 3,
    limit: int = 50,
) -> dict[str, Any]:
    """
    List available tools with optional filtering.

    Args:
        query: Search query for tool name/description
        domain: Filter by domain (compute, database, etc.)
        max_tier: Maximum tool tier to include (1-4)
        limit: Maximum number of results
    """
    try:
        catalog = ToolCatalog.get_instance()
        await catalog.ensure_fresh()

        tools = catalog.search_tools(
            query=query,
            domain=domain,
            max_tier=max_tier,
            limit=limit,
        )

        return {
            "tools": tools,
            "count": len(tools),
            "filters": {
                "query": query,
                "domain": domain,
                "max_tier": max_tier,
            },
        }

    except Exception as e:
        logger.error("Tool list failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/tools/execute", response_model=ToolResponse, tags=["Tools"])
async def execute_tool(request: ToolRequest) -> ToolResponse:
    """
    Execute a specific tool.

    Note: Some tools require confirmation and may be rejected.
    """
    start_time = time.time()

    try:
        catalog = ToolCatalog.get_instance()
        result = await catalog.execute(request.tool_name, request.arguments)

        duration_ms = int((time.time() - start_time) * 1000)

        return ToolResponse(
            success=result.success,
            result=result.result,
            error=result.error,
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.error("Tool execution failed", tool=request.tool_name, error=str(e))
        return ToolResponse(
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )


@app.get("/tools/{tool_name}", tags=["Tools"])
async def get_tool(tool_name: str) -> dict[str, Any]:
    """Get details about a specific tool."""
    try:
        catalog = ToolCatalog.get_instance()
        tool_def = catalog.get_tool(tool_name)

        if not tool_def:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool not found: {tool_name}",
            )

        return {
            "name": tool_def.name,
            "description": tool_def.description,
            "input_schema": tool_def.input_schema,
            "server": tool_def.server_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get tool failed", tool=tool_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/agents", tags=["Agents"])
async def list_agents() -> dict[str, Any]:
    """List all available agents."""
    try:
        catalog = AgentCatalog.get_instance()
        agents = catalog.list_all()

        return {
            "agents": [
                {
                    "role": agent.role,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "skills": agent.skills,
                }
                for agent in agents
            ],
            "count": len(agents),
        }

    except Exception as e:
        logger.error("Agent list failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/agents/{role}", tags=["Agents"])
async def get_agent(role: str) -> dict[str, Any]:
    """Get details about a specific agent."""
    try:
        catalog = AgentCatalog.get_instance()
        agent = catalog.get(role)

        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {role}",
            )

        return {
            "role": agent.role,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "skills": agent.skills,
            "mcp_tools": agent.mcp_tools,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get agent failed", role=role, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Server Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/mcp/servers", tags=["MCP"])
async def list_mcp_servers() -> dict[str, Any]:
    """List MCP server status."""
    try:
        registry = ServerRegistry.get_instance()
        servers = {}

        for server_id in registry.list_servers():
            info = registry.get_server_info(server_id)
            servers[server_id] = {
                "status": registry.get_status(server_id),
                "tool_count": info["tool_count"] if info else 0,
            }

        return {"servers": servers, "count": len(servers)}

    except Exception as e:
        logger.error("MCP server list failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/mcp/servers/{server_id}/reconnect", tags=["MCP"])
async def reconnect_mcp_server(server_id: str) -> dict[str, Any]:
    """Attempt to reconnect an MCP server."""
    try:
        registry = ServerRegistry.get_instance()

        if server_id not in registry.list_servers():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Server not found: {server_id}",
            )

        success = await registry.reconnect(server_id)

        return {
            "server_id": server_id,
            "reconnected": success,
            "status": registry.get_status(server_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("MCP reconnect failed", server=server_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/architecture", tags=["Architecture"])
async def get_architecture() -> dict[str, Any]:
    """
    Get system architecture mapping for visualization.

    Returns the dynamic relationships between agents, MCP servers, and domains.
    This endpoint enables the frontend to visualize the architecture without
    hardcoded mappings.

    Returns:
        - agents: List of agents with their domains
        - mcp_servers: List of MCP servers with their domains and status
        - agent_mcp_map: Computed mapping of which agents connect to which MCP servers
        - domain_capabilities: Domain to capability mappings
    """
    try:
        agent_catalog = AgentCatalog.get_instance()
        registry = ServerRegistry.get_instance()

        # Get all agents with their domains
        agents_data = []
        for agent_def in agent_catalog.list_all():
            domains = agent_catalog.get_agent_domains(agent_def.role)
            agents_data.append({
                "role": agent_def.role,
                "description": agent_def.description,
                "capabilities": agent_def.capabilities,
                "domains": domains,
            })

        # Get MCP servers with their domains and status
        mcp_servers_data = {}
        connected_servers = set()
        for server_id in registry.list_servers():
            info = registry.get_server_info(server_id)
            server_status = registry.get_status(server_id)
            mcp_servers_data[server_id] = {
                "status": server_status,
                "tool_count": info["tool_count"] if info else 0,
                "domains": MCP_SERVER_DOMAINS.get(server_id, []),
            }
            if server_status == "connected":
                connected_servers.add(server_id)

        # Compute agent-to-MCP server mappings dynamically
        # An agent connects to an MCP server if they share at least one domain
        agent_mcp_map = {}
        for agent in agents_data:
            agent_domains = set(agent["domains"])
            connected_mcps = []
            for server_id, server_info in mcp_servers_data.items():
                server_domains = set(server_info["domains"])
                # Check if there's domain overlap
                if agent_domains & server_domains:
                    connected_mcps.append(server_id)
            agent_mcp_map[agent["role"]] = connected_mcps

        return {
            "agents": agents_data,
            "mcp_servers": mcp_servers_data,
            "agent_mcp_map": agent_mcp_map,
            "mcp_server_domains": MCP_SERVER_DOMAINS,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("Architecture fetch failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Slack Integration Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/slack/status", tags=["Slack"])
async def get_slack_status() -> dict[str, Any]:
    """Get Slack integration status.

    Returns connection status, handler info, and recent activity.
    """
    import os

    # Check if Slack tokens are configured
    bot_token = os.getenv("SLACK_BOT_TOKEN", "")
    app_token = os.getenv("SLACK_APP_TOKEN", "")

    status_info = {
        "configured": bool(bot_token and app_token),
        "bot_token_valid": bot_token.startswith("xoxb-") if bot_token else False,
        "app_token_valid": app_token.startswith("xapp-") if app_token else False,
        "socket_mode_enabled": bool(app_token),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Try to check actual connection status
    if status_info["configured"] and status_info["bot_token_valid"]:
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError

            client = WebClient(token=bot_token)
            response = client.auth_test()

            if response["ok"]:
                status_info["connection"] = {
                    "status": "connected",
                    "bot_id": response.get("bot_id"),
                    "team": response.get("team"),
                    "user": response.get("user"),
                    "user_id": response.get("user_id"),
                }
            else:
                status_info["connection"] = {
                    "status": "error",
                    "error": response.get("error", "Unknown error"),
                }
        except SlackApiError as e:
            status_info["connection"] = {
                "status": "error",
                "error": e.response.get("error", str(e)),
            }
        except Exception as e:
            status_info["connection"] = {
                "status": "error",
                "error": str(e),
            }
    else:
        status_info["connection"] = {
            "status": "not_configured",
            "message": "Slack tokens not configured or invalid",
        }

    return status_info


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/stats", tags=["Stats"])
async def get_statistics() -> dict[str, Any]:
    """Get comprehensive system statistics."""
    uptime = (datetime.utcnow() - app_state.start_time).total_seconds()

    return {
        "uptime_seconds": uptime,
        "request_count": app_state.request_count,
        "active_threads": len(app_state.active_threads),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Workflow Visualizer Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/visualizer", tags=["Observability"])
async def get_workflow_visualizer() -> dict[str, Any]:
    """
    Get the LangGraph workflow visualization data.

    Returns:
        - nodes: List of workflow nodes with descriptions
        - edges: List of edges with types (sequential, conditional, loop)
        - mermaid_diagram: Mermaid flowchart syntax for rendering
        - example_queries: Sample queries for each routing path
    """
    from src.observability.visualizer import get_visualization_data

    try:
        return get_visualization_data(
            coordinator=app_state.coordinator,
            include_examples=True,
        )
    except Exception as e:
        logger.error("Visualizer data fetch failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/visualizer/diagram", tags=["Observability"])
async def get_workflow_diagram(
    routing_type: str | None = None,
    format: str = "mermaid",
) -> dict[str, Any]:
    """
    Get the workflow diagram in various formats.

    Args:
        routing_type: Highlight a specific routing path (workflow, parallel, agent, escalate)
        format: Output format (mermaid, svg, png - only mermaid supported currently)

    Returns:
        Mermaid diagram string and metadata
    """
    from src.observability.visualizer import generate_mermaid_diagram

    try:
        diagram = generate_mermaid_diagram(routing_type=routing_type)

        return {
            "format": format,
            "diagram": diagram,
            "routing_type": routing_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Diagram generation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/visualizer/examples", tags=["Observability"])
async def get_example_queries() -> dict[str, Any]:
    """
    Get example queries for each routing path.

    Returns categorized example queries that demonstrate:
    - workflow: Deterministic workflow execution
    - parallel: Multi-agent parallel execution
    - agent: LLM-powered agent reasoning
    - escalate: Human escalation scenarios
    """
    from src.observability.visualizer import EXAMPLE_QUERIES

    return {
        "examples": EXAMPLE_QUERIES,
        "timestamp": datetime.utcnow().isoformat(),
    }


from fastapi.responses import HTMLResponse


@app.get("/visualizer/widget", response_class=HTMLResponse, tags=["Observability"])
async def get_visualizer_widget() -> HTMLResponse:
    """
    Serve the interactive LangGraph workflow visualizer widget.

    Returns an HTML page with:
    - Interactive Mermaid diagram of the workflow
    - Real-time execution tracing (when available)
    - Example query testing
    - Agent selection visualization
    """
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCI Coordinator - LangGraph Workflow Visualizer</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-light: #818cf8;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        h1 {
            font-size: 1.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        h1 .icon {
            font-size: 2rem;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border-radius: 9999px;
            font-size: 0.875rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 1.5rem;
        }

        @media (max-width: 1024px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border-radius: 1rem;
            border: 1px solid var(--border);
            overflow: hidden;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
        }

        .card-title {
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-body {
            padding: 1.5rem;
        }

        .diagram-container {
            background: #fff;
            border-radius: 0.5rem;
            padding: 1rem;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .mermaid {
            width: 100%;
        }

        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .tab {
            padding: 0.5rem 1rem;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.875rem;
        }

        .tab:hover {
            background: var(--border);
            color: var(--text);
        }

        .tab.active {
            background: var(--primary);
            border-color: var(--primary);
            color: white;
        }

        .example-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .example-item {
            padding: 1rem;
            background: var(--bg);
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }

        .example-item:hover {
            border-color: var(--primary);
            transform: translateX(4px);
        }

        .example-query {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            color: var(--primary-light);
            margin-bottom: 0.5rem;
        }

        .example-description {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .example-badge {
            display: inline-block;
            padding: 0.125rem 0.5rem;
            background: var(--primary);
            border-radius: 9999px;
            font-size: 0.625rem;
            text-transform: uppercase;
            margin-top: 0.5rem;
        }

        .node-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .node-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            background: var(--bg);
            border-radius: 0.5rem;
        }

        .node-icon {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--primary);
            border-radius: 0.5rem;
            font-size: 1rem;
        }

        .node-info {
            flex: 1;
        }

        .node-name {
            font-weight: 500;
            font-size: 0.875rem;
        }

        .node-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .agent-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }

        .agent-card {
            padding: 1rem;
            background: var(--bg);
            border-radius: 0.5rem;
            text-align: center;
        }

        .agent-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .agent-name {
            font-size: 0.75rem;
            font-weight: 500;
        }

        .agent-domain {
            font-size: 0.625rem;
            color: var(--text-muted);
        }

        .refresh-btn {
            padding: 0.5rem 1rem;
            background: var(--primary);
            border: none;
            border-radius: 0.5rem;
            color: white;
            cursor: pointer;
            font-size: 0.875rem;
            transition: background 0.2s;
        }

        .refresh-btn:hover {
            background: var(--primary-light);
        }

        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 200px;
            color: var(--text-muted);
        }

        .spinner {
            width: 32px;
            height: 32px;
            border: 3px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.75rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .legend {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            padding: 1rem;
            background: var(--bg);
            border-radius: 0.5rem;
            margin-top: 1rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
        }

        .legend-line {
            width: 24px;
            height: 2px;
        }

        .legend-line.sequential {
            background: var(--text-muted);
        }

        .legend-line.conditional {
            background: var(--primary);
            background: linear-gradient(90deg, var(--primary) 50%, transparent 50%);
            background-size: 8px 2px;
        }

        .legend-line.loop {
            background: var(--warning);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>
                <span class="icon">🔄</span>
                LangGraph Workflow Visualizer
            </h1>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>OCI Coordinator Active</span>
            </div>
        </header>

        <div class="grid">
            <!-- Main Diagram -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">
                        📊 Workflow Graph
                    </span>
                    <button class="refresh-btn" onclick="loadVisualization()">
                        ↻ Refresh
                    </button>
                </div>
                <div class="card-body">
                    <div class="tabs">
                        <button class="tab active" data-route="all" onclick="filterRoute('all')">All Paths</button>
                        <button class="tab" data-route="workflow" onclick="filterRoute('workflow')">Workflow</button>
                        <button class="tab" data-route="parallel" onclick="filterRoute('parallel')">Parallel</button>
                        <button class="tab" data-route="agent" onclick="filterRoute('agent')">Agent</button>
                    </div>
                    <div id="diagram-container" class="diagram-container">
                        <div class="loading">
                            <div class="spinner"></div>
                            Loading workflow diagram...
                        </div>
                    </div>
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-line sequential"></div>
                            <span>Sequential</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-line conditional"></div>
                            <span>Conditional</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-line loop"></div>
                            <span>Loop</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Sidebar -->
            <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                <!-- Nodes Reference -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">📋 Graph Nodes</span>
                    </div>
                    <div class="card-body">
                        <div id="node-list" class="node-list">
                            <!-- Populated by JS -->
                        </div>
                    </div>
                </div>

                <!-- Agents -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">🤖 Available Agents</span>
                    </div>
                    <div class="card-body">
                        <div class="agent-grid">
                            <div class="agent-card">
                                <div class="agent-icon">🗄️</div>
                                <div class="agent-name">DbTroubleshoot</div>
                                <div class="agent-domain">Database</div>
                            </div>
                            <div class="agent-card">
                                <div class="agent-icon">📊</div>
                                <div class="agent-name">LogAnalytics</div>
                                <div class="agent-domain">Observability</div>
                            </div>
                            <div class="agent-card">
                                <div class="agent-icon">🛡️</div>
                                <div class="agent-name">SecurityThreat</div>
                                <div class="agent-domain">Security</div>
                            </div>
                            <div class="agent-card">
                                <div class="agent-icon">💰</div>
                                <div class="agent-name">FinOps</div>
                                <div class="agent-domain">Cost</div>
                            </div>
                            <div class="agent-card">
                                <div class="agent-icon">🖥️</div>
                                <div class="agent-name">Infrastructure</div>
                                <div class="agent-domain">Compute</div>
                            </div>
                            <div class="agent-card">
                                <div class="agent-icon">🔍</div>
                                <div class="agent-name">ErrorAnalysis</div>
                                <div class="agent-domain">Debugging</div>
                            </div>
                            <div class="agent-card">
                                <div class="agent-icon">🤖</div>
                                <div class="agent-name">SelectAI</div>
                                <div class="agent-domain">Data/AI</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Example Queries -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">💡 Example Queries</span>
                    </div>
                    <div class="card-body">
                        <div id="example-list" class="example-list">
                            <!-- Populated by JS -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Mermaid
        mermaid.initialize({
            startOnLoad: false,
            theme: 'base',
            themeVariables: {
                primaryColor: '#4f46e5',
                primaryTextColor: '#fff',
                primaryBorderColor: '#4338ca',
                lineColor: '#6366f1',
                secondaryColor: '#f0fdf4',
                tertiaryColor: '#fef3c7'
            },
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }
        });

        let currentData = null;
        let currentRoute = 'all';

        // Node icons
        const nodeIcons = {
            'input': '📥',
            'classifier': '🎯',
            'router': '🔀',
            'workflow': '⚡',
            'parallel': '🔄',
            'agent': '🤖',
            'action': '🔧',
            'output': '📤'
        };

        async function loadVisualization() {
            const container = document.getElementById('diagram-container');
            container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading workflow diagram...</div>';

            try {
                const response = await fetch('/visualizer');
                currentData = await response.json();

                renderDiagram(currentData.mermaid_diagram);
                renderNodes(currentData.nodes);
                renderExamples(currentData.example_queries);
            } catch (error) {
                container.innerHTML = '<div class="loading">Error loading visualization: ' + error.message + '</div>';
            }
        }

        async function renderDiagram(mermaidCode) {
            const container = document.getElementById('diagram-container');

            try {
                const { svg } = await mermaid.render('mermaid-diagram', mermaidCode);
                container.innerHTML = '<div class="mermaid">' + svg + '</div>';
            } catch (error) {
                container.innerHTML = '<pre style="color: #333; font-size: 12px; overflow: auto;">' + mermaidCode + '</pre>';
            }
        }

        function renderNodes(nodes) {
            const container = document.getElementById('node-list');
            const filteredNodes = nodes.filter(n => !n.id.startsWith('__'));

            container.innerHTML = filteredNodes.map(node => `
                <div class="node-item">
                    <div class="node-icon">${nodeIcons[node.id] || '📌'}</div>
                    <div class="node-info">
                        <div class="node-name">${node.name}</div>
                        <div class="node-desc">${node.description}</div>
                    </div>
                </div>
            `).join('');
        }

        function renderExamples(examples) {
            const container = document.getElementById('example-list');
            const route = currentRoute === 'all' ? 'workflow' : currentRoute;
            const routeExamples = examples[route] || [];

            container.innerHTML = routeExamples.slice(0, 4).map(ex => `
                <div class="example-item" onclick="highlightRoute('${route}')">
                    <div class="example-query">"${ex.query}"</div>
                    <div class="example-description">${ex.description}</div>
                    <span class="example-badge">${route}</span>
                </div>
            `).join('');
        }

        async function filterRoute(route) {
            // Update tab state
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.toggle('active', tab.dataset.route === route);
            });

            currentRoute = route;

            // Reload diagram with route highlighting
            try {
                const url = route === 'all' ? '/visualizer/diagram' : `/visualizer/diagram?routing_type=${route}`;
                const response = await fetch(url);
                const data = await response.json();
                renderDiagram(data.diagram);

                // Update examples for this route
                if (currentData && currentData.example_queries) {
                    renderExamples(currentData.example_queries);
                }
            } catch (error) {
                console.error('Failed to filter route:', error);
            }
        }

        function highlightRoute(route) {
            filterRoute(route);
        }

        // Load on page ready
        document.addEventListener('DOMContentLoaded', loadVisualization);
    </script>
</body>
</html>'''
    return HTMLResponse(content=html_content)


@app.get("/visualizer/executions", tags=["Observability"])
async def get_active_executions() -> dict[str, Any]:
    """
    Get currently active workflow executions.

    Returns:
        List of active executions with their current state
    """
    return {
        "executions": list(app_state.active_executions.values()),
        "count": len(app_state.active_executions),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/visualizer/trace/{execution_id}", tags=["Observability"])
async def get_execution_trace(execution_id: str) -> dict[str, Any]:
    """
    Get the execution trace for a specific execution.

    Args:
        execution_id: The execution ID to get trace for

    Returns:
        Execution trace with visualization data
    """
    from src.observability.visualizer import WorkflowVisualizer

    if execution_id not in app_state.active_executions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )

    execution = app_state.active_executions[execution_id]
    visualizer = WorkflowVisualizer(app_state.coordinator)

    # Get live visualization if thinking_trace is available
    thinking_trace = execution.get("thinking_trace")
    routing_type = execution.get("routing_type")
    current_agent = execution.get("current_agent")

    viz = visualizer.get_live_visualization(
        thinking_trace=thinking_trace,
        routing_type=routing_type,
        current_agent=current_agent,
    )

    return {
        "execution_id": execution_id,
        "visualization": viz.to_dict(),
        "started_at": execution.get("started_at"),
        "updated_at": execution.get("updated_at"),
    }


import json


@app.get("/visualizer/stream", tags=["Observability"])
async def stream_executions():
    """
    Server-Sent Events stream for real-time execution updates.

    Streams updates whenever execution state changes.
    Use this endpoint to build real-time visualizations.

    Example usage (JavaScript):
        const eventSource = new EventSource('/visualizer/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateVisualization(data);
        };
    """
    async def event_generator():
        """Generate SSE events for execution updates."""
        last_state = {}

        while True:
            # Check for state changes
            current_state = {
                k: {
                    "phase": v.get("phase"),
                    "active_node": v.get("active_node"),
                    "routing_type": v.get("routing_type"),
                }
                for k, v in app_state.active_executions.items()
            }

            if current_state != last_state:
                # Send update
                data = {
                    "type": "execution_update",
                    "executions": list(app_state.active_executions.values()),
                    "count": len(app_state.active_executions),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_state = current_state

            # Also send heartbeat every 30 seconds
            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"

            await asyncio.sleep(1)  # Check every second

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
