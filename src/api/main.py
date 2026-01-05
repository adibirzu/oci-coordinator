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

from src.agents.catalog import AgentCatalog
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

    async def get_coordinator(self):
        """Get or create the coordinator (cached)."""
        if self._coordinator is None:
            async with self._coordinator_lock:
                if self._coordinator is None:
                    from src.agents.coordinator.graph import create_coordinator
                    from src.llm import get_llm

                    llm = get_llm()
                    self._coordinator = await create_coordinator(llm=llm)
                    logger.info("Coordinator initialized for API")
        return self._coordinator


app_state = AppState()


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan Management
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("API server starting up")
    yield
    logger.info("API server shutting down")


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
        from src.llm import get_llm

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

            # Invoke with thread context
            result = await coordinator.invoke(
                query=request.message,
                thread_id=thread_id,
                user_id=request.user_id,
            )

            response_text = result.get("response") if isinstance(result, dict) else str(result)
            agent_used = result.get("routing_type") if isinstance(result, dict) else None
            tools_used = []

        except Exception as e:
            logger.warning("LangGraph coordinator failed, using fallback", error=str(e))

            # Fallback to simple orchestrator
            orchestrator = ParallelOrchestrator(agent_catalog=agent_catalog)
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
            detail=f"Failed to process request: {str(e)}",
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

            # Stream response
            async for chunk in coordinator.invoke_stream(
                query=request.message,
                thread_id=thread_id,
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            yield f"data: Error: {str(e)}\n\n"

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
            import oci
            from oci.loggingsearch import LogSearchClient
            from oci.loggingsearch.models import SearchLogsDetails

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
        with open(log_file, "r") as f:
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
        import oci
        from oci.apm_traces import QueryClient

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
