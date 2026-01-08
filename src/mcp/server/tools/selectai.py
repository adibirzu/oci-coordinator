"""
MCP Tools for Oracle Autonomous Database SelectAI.

Provides tools for:
- Natural language to SQL translation (NL2SQL)
- Chat with database context
- Text summarization and narration
- AI profile management
- SelectAI agent orchestration

Connection Strategy:
Supports two connection methods with automatic fallback:
1. Wallet-based: Direct connection using oracledb (preferred, faster)
2. ORDS REST API: HTTP-based connection (fallback, requires ORDS enabled)

The tools automatically choose the best available method.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

import httpx
from opentelemetry import trace

# Get tracer for SelectAI tools
_tracer = trace.get_tracer("mcp-oci-selectai")
logger = logging.getLogger(__name__)

# Default timeout for SelectAI operations (seconds)
SELECTAI_TIMEOUT = 60

# Connection pool for wallet-based connections
_connection_pool = None


def _get_ords_config() -> dict:
    """Get ORDS configuration from environment."""
    return {
        "base_url": os.getenv("SELECTAI_ORDS_BASE_URL", ""),
        "schema": os.getenv("SELECTAI_ORDS_SCHEMA", "ADMIN"),
        "auth_token": os.getenv("SELECTAI_ORDS_AUTH_TOKEN", ""),
        "default_profile": os.getenv("SELECTAI_DEFAULT_PROFILE", "OCI_GENAI"),
    }


def _get_wallet_config() -> dict:
    """Get wallet-based ATP configuration from environment."""
    return {
        "tns_name": os.getenv("ATP_TNS_NAME", ""),
        "user": os.getenv("ATP_USER", "ADMIN"),
        "password": os.getenv("ATP_PASSWORD", ""),
        "wallet_dir": os.getenv("ATP_WALLET_DIR", ""),
        "wallet_password": os.getenv("ATP_WALLET_PASSWORD", ""),
        "default_profile": os.getenv("SELECTAI_DEFAULT_PROFILE", "OCI_GENAI"),
    }


def _is_wallet_configured() -> bool:
    """Check if wallet-based connection is configured."""
    config = _get_wallet_config()
    return bool(config["tns_name"] and config["password"] and config["wallet_dir"])


def _read_dsn_from_tnsnames(wallet_dir: str, tns_name: str) -> str | None:
    """Read full DSN string from tnsnames.ora in the wallet directory.

    The full DSN is needed to preserve security settings like ssl_server_dn_match.
    """
    import os

    tnsnames_path = os.path.join(wallet_dir, "tnsnames.ora")
    if not os.path.exists(tnsnames_path):
        return None

    try:
        with open(tnsnames_path, "r") as f:
            content = f.read()

        # Parse the TNS entry - entries are separated by blank lines
        for entry in content.split("\n\n"):
            if entry.strip().lower().startswith(tns_name.lower()):
                # Extract the description part after the "="
                eq_pos = entry.find("=")
                if eq_pos > 0:
                    return entry[eq_pos + 1 :].strip()
        return None
    except Exception as e:
        logger.warning(f"Failed to read tnsnames.ora: {e}")
        return None


async def _get_wallet_connection():
    """Get a connection from the wallet-based pool."""
    global _connection_pool

    try:
        import oracledb
    except ImportError:
        logger.warning("oracledb not installed, wallet connections unavailable")
        return None

    config = _get_wallet_config()

    if not _is_wallet_configured():
        return None

    try:
        if _connection_pool is None:
            # Read the full DSN from tnsnames.ora to preserve security settings
            dsn = _read_dsn_from_tnsnames(config["wallet_dir"], config["tns_name"])
            if not dsn:
                dsn = config["tns_name"]
                logger.warning(
                    f"Could not read DSN from tnsnames.ora, using TNS name directly"
                )

            # Create connection pool with full DSN
            _connection_pool = oracledb.create_pool(
                user=config["user"],
                password=config["password"],
                dsn=dsn,
                config_dir=config["wallet_dir"],
                wallet_location=config["wallet_dir"],
                wallet_password=config.get("wallet_password"),
                min=1,
                max=5,
                increment=1,
            )
            logger.info(f"Created SelectAI connection pool for {config['tns_name']}")

        return _connection_pool.acquire()
    except Exception as e:
        logger.warning(f"Failed to get wallet connection: {e}")
        return None


async def _execute_sql_via_wallet(sql: str, fetch: bool = True) -> dict:
    """Execute SQL via wallet-based connection.

    Args:
        sql: SQL statement to execute
        fetch: Whether to fetch results (True for SELECT, False for DML)

    Returns:
        Dict with execution result or error
    """
    conn = await _get_wallet_connection()
    if conn is None:
        return {"error": "Wallet connection not available"}

    try:
        cursor = conn.cursor()

        # Execute the SQL
        cursor.execute(sql)

        if fetch:
            # Get column names
            columns = [col[0].lower() for col in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            # Convert to list of dicts
            items = []
            for row in rows:
                items.append(dict(zip(columns, row)))

            return {"items": items, "row_count": len(items)}
        else:
            conn.commit()
            return {"status": "ok", "rows_affected": cursor.rowcount}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


async def _execute_sql_via_ords(
    sql: str,
    base_url: str,
    schema: str,
    auth_token: str,
    timeout: int = SELECTAI_TIMEOUT,
) -> dict:
    """Execute SQL via ORDS REST API.

    Args:
        sql: SQL statement to execute
        base_url: ORDS base URL (e.g., https://xxx.adb.region.oraclecloudapps.com)
        schema: Database schema name
        auth_token: OAuth token for ORDS authentication
        timeout: Request timeout in seconds

    Returns:
        Dict with execution result or error
    """
    endpoint = f"{base_url.rstrip('/')}/ords/{schema}/_/sql"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    payload = {
        "statementText": sql,
        "limit": 1000,
        "offset": 0,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"ORDS API error: {e.response.status_code}",
                "detail": e.response.text[:500] if e.response.text else str(e),
            }
        except httpx.RequestError as e:
            return {
                "error": f"Connection error: {type(e).__name__}",
                "detail": str(e),
            }


async def _execute_sql(
    sql: str,
    ords_base_url: str | None = None,
    ords_schema: str | None = None,
    ords_auth_token: str | None = None,
    fetch: bool = True,
) -> tuple[dict, str]:
    """Execute SQL using best available connection method.

    Tries wallet-based connection first, falls back to ORDS if not available.

    Args:
        sql: SQL statement to execute
        ords_base_url: Override ORDS base URL
        ords_schema: Override ORDS schema
        ords_auth_token: Override ORDS auth token
        fetch: Whether to fetch results

    Returns:
        Tuple of (result dict, connection_method used)
    """
    # Try wallet connection first
    if _is_wallet_configured():
        result = await _execute_sql_via_wallet(sql, fetch=fetch)
        if "error" not in result:
            return result, "wallet"
        logger.debug(f"Wallet execution failed, trying ORDS: {result.get('error')}")

    # Fall back to ORDS
    ords_config = _get_ords_config()
    base_url = ords_base_url or ords_config["base_url"]
    schema = ords_schema or ords_config["schema"]
    auth_token = ords_auth_token or ords_config["auth_token"]

    if base_url:
        result = await _execute_sql_via_ords(sql, base_url, schema, auth_token)
        return result, "ords"

    # Neither connection available
    return {
        "error": "No database connection configured",
        "hint": "Configure ATP_TNS_NAME/ATP_PASSWORD/ATP_WALLET_DIR for wallet connection, "
                "or SELECTAI_ORDS_BASE_URL for ORDS connection.",
    }, "none"


async def _selectai_generate_logic(
    prompt: str,
    profile_name: str | None = None,
    action: Literal["showsql", "runsql", "explainsql", "narrate", "chat", "summarize"] = "runsql",
    ords_base_url: str | None = None,
    ords_schema: str | None = None,
    ords_auth_token: str | None = None,
) -> str:
    """Internal logic for SelectAI GENERATE function.

    Args:
        prompt: Natural language query or chat message
        profile_name: Name of the AI profile to use
        action: SelectAI action type
        ords_base_url: Override ORDS base URL
        ords_schema: Override ORDS schema
        ords_auth_token: Override ORDS auth token

    Returns:
        JSON string with generated SQL and/or query results
    """
    with _tracer.start_as_current_span("mcp.selectai.generate") as span:
        span.set_attribute("action", action)
        span.set_attribute("prompt_length", len(prompt))

        # Get default profile from config
        wallet_config = _get_wallet_config()
        ords_config = _get_ords_config()
        profile = profile_name or wallet_config["default_profile"] or ords_config["default_profile"]

        # Escape single quotes in prompt
        safe_prompt = prompt.replace("'", "''")

        # Build the DBMS_CLOUD_AI.GENERATE SQL
        sql = f"""
SELECT DBMS_CLOUD_AI.GENERATE(
    prompt => '{safe_prompt}',
    profile_name => '{profile}',
    action => '{action}'
) AS response FROM DUAL
"""

        span.set_attribute("profile", profile)

        result, connection_method = await _execute_sql(
            sql, ords_base_url, ords_schema, ords_auth_token
        )
        span.set_attribute("connection_method", connection_method)

        if "error" in result:
            return json.dumps({
                "type": "selectai_generate",
                "action": action,
                "error": result["error"],
                "detail": result.get("detail", result.get("hint", "")),
            })

        # Extract the response from result
        try:
            items = result.get("items", [])
            if items and len(items) > 0:
                # Get the response column value
                first_item = items[0]
                response_data = first_item.get("response", first_item.get("RESPONSE", first_item))
            else:
                response_data = result

            return json.dumps({
                "type": "selectai_generate",
                "action": action,
                "profile": profile,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response_data,
                "connection": connection_method,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        except Exception as e:
            return json.dumps({
                "type": "selectai_generate",
                "action": action,
                "error": f"Failed to parse response: {e}",
                "raw_result": str(result)[:500],
            })


async def _selectai_list_profiles_logic(
    ords_base_url: str | None = None,
    ords_schema: str | None = None,
    ords_auth_token: str | None = None,
) -> str:
    """Internal logic for listing AI profiles.

    Returns:
        JSON string with list of AI profiles
    """
    with _tracer.start_as_current_span("mcp.selectai.list_profiles") as span:
        # Query to list AI profiles
        # Note: user_cloud_ai_profiles has basic columns; provider/model are in attributes
        sql = """
SELECT profile_id, profile_name, status, description, created, last_modified
FROM user_cloud_ai_profiles
ORDER BY created DESC
"""

        result, connection_method = await _execute_sql(
            sql, ords_base_url, ords_schema, ords_auth_token
        )
        span.set_attribute("connection_method", connection_method)

        if "error" in result:
            return json.dumps({
                "type": "selectai_profiles",
                "error": result["error"],
                "detail": result.get("detail", result.get("hint", "")),
            })

        try:
            items = result.get("items", [])
            profiles = []
            for item in items:
                profiles.append({
                    "id": item.get("profile_id", item.get("PROFILE_ID")),
                    "name": item.get("profile_name", item.get("PROFILE_NAME")),
                    "status": item.get("status", item.get("STATUS")),
                    "description": item.get("description", item.get("DESCRIPTION")),
                    "created": str(item.get("created", item.get("CREATED", ""))),
                    "last_modified": str(item.get("last_modified", item.get("LAST_MODIFIED", ""))),
                })

            return json.dumps({
                "type": "selectai_profiles",
                "count": len(profiles),
                "profiles": profiles,
                "connection": connection_method,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        except Exception as e:
            return json.dumps({
                "type": "selectai_profiles",
                "error": f"Failed to parse profiles: {e}",
                "raw_result": str(result)[:500],
            })


async def _selectai_run_agent_logic(
    team_name: str,
    user_prompt: str,
    session_id: str | None = None,
    ords_base_url: str | None = None,
    ords_schema: str | None = None,
    ords_auth_token: str | None = None,
) -> str:
    """Internal logic for running a SelectAI agent team.

    Args:
        team_name: Name of the agent team to run
        user_prompt: User's prompt/question for the agent
        session_id: Optional session ID for conversation continuity
        ords_base_url: Override ORDS base URL
        ords_schema: Override ORDS schema
        ords_auth_token: Override ORDS auth token

    Returns:
        JSON string with agent execution results
    """
    with _tracer.start_as_current_span("mcp.selectai.run_agent") as span:
        span.set_attribute("team_name", team_name)
        span.set_attribute("prompt_length", len(user_prompt))

        # Escape single quotes
        safe_prompt = user_prompt.replace("'", "''")
        safe_team = team_name.replace("'", "''")

        # Build the DBMS_CLOUD_AI_AGENT.RUN_TEAM SQL
        # Use a SELECT wrapper for the function call
        if session_id:
            safe_session = session_id.replace("'", "''")
            sql = f"""
SELECT DBMS_CLOUD_AI_AGENT.RUN_TEAM(
    team_name => '{safe_team}',
    user_prompt => '{safe_prompt}',
    session_id => '{safe_session}'
) AS response FROM DUAL
"""
        else:
            sql = f"""
SELECT DBMS_CLOUD_AI_AGENT.RUN_TEAM(
    team_name => '{safe_team}',
    user_prompt => '{safe_prompt}'
) AS response FROM DUAL
"""

        span.set_attribute("team", team_name)

        result, connection_method = await _execute_sql(
            sql, ords_base_url, ords_schema, ords_auth_token
        )
        span.set_attribute("connection_method", connection_method)

        if "error" in result:
            return json.dumps({
                "type": "selectai_agent",
                "team": team_name,
                "error": result["error"],
                "detail": result.get("detail", result.get("hint", "")),
            })

        try:
            # Extract the result
            items = result.get("items", [])
            if items and len(items) > 0:
                output = items[0].get("response", items[0].get("RESPONSE", items[0]))
            else:
                output = result.get("result", result)

            return json.dumps({
                "type": "selectai_agent",
                "team": team_name,
                "prompt": user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt,
                "response": output,
                "session_id": session_id,
                "connection": connection_method,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        except Exception as e:
            return json.dumps({
                "type": "selectai_agent",
                "team": team_name,
                "error": f"Failed to parse agent response: {e}",
                "raw_result": str(result)[:500],
            })


async def _selectai_get_profile_tables_logic(
    profile_name: str,
    ords_base_url: str | None = None,
    ords_schema: str | None = None,
    ords_auth_token: str | None = None,
) -> str:
    """Get tables associated with an AI profile.

    Args:
        profile_name: Name of the AI profile
        ords_base_url: Override ORDS base URL
        ords_schema: Override ORDS schema
        ords_auth_token: Override ORDS auth token

    Returns:
        JSON string with profile table information
    """
    with _tracer.start_as_current_span("mcp.selectai.get_profile_tables") as span:
        span.set_attribute("profile", profile_name)

        safe_profile = profile_name.replace("'", "''")

        # Query to get profile attributes including object_list
        sql = f"""
SELECT profile_name, attributes
FROM user_cloud_ai_profiles
WHERE profile_name = '{safe_profile}'
"""

        result, connection_method = await _execute_sql(
            sql, ords_base_url, ords_schema, ords_auth_token
        )
        span.set_attribute("connection_method", connection_method)

        if "error" in result:
            return json.dumps({
                "type": "selectai_profile_tables",
                "profile": profile_name,
                "error": result["error"],
                "detail": result.get("detail", result.get("hint", "")),
            })

        try:
            items = result.get("items", [])
            if items:
                attrs = items[0].get("attributes", items[0].get("ATTRIBUTES", "{}"))
                if isinstance(attrs, str):
                    attrs = json.loads(attrs)
                object_list = attrs.get("object_list", [])
                return json.dumps({
                    "type": "selectai_profile_tables",
                    "profile": profile_name,
                    "tables": object_list,
                    "table_count": len(object_list),
                    "connection": connection_method,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            else:
                return json.dumps({
                    "type": "selectai_profile_tables",
                    "profile": profile_name,
                    "error": f"Profile '{profile_name}' not found",
                })
        except Exception as e:
            return json.dumps({
                "type": "selectai_profile_tables",
                "profile": profile_name,
                "error": f"Failed to parse profile: {e}",
            })


def register_selectai_tools(mcp):
    """Register SelectAI tools with the MCP server."""

    @mcp.tool()
    async def oci_selectai_generate(
        prompt: str,
        profile_name: str | None = None,
        action: str = "runsql",
        ords_base_url: str | None = None,
        ords_schema: str | None = None,
        ords_auth_token: str | None = None,
    ) -> str:
        """Execute SelectAI GENERATE function for natural language to SQL.

        Translates natural language queries into SQL and optionally executes them
        using Oracle Autonomous Database's SelectAI feature.

        Args:
            prompt: Natural language query (e.g., "Show top 10 customers by revenue")
            profile_name: AI profile name (defaults to SELECTAI_DEFAULT_PROFILE env var)
            action: Action type - one of:
                - showsql: Return generated SQL without executing
                - runsql: Execute generated SQL and return results
                - explainsql: Generate SQL with explanation
                - narrate: Natural language summary of data
                - chat: General conversation with database context
                - summarize: Summarize text/documents
            ords_base_url: Override ORDS base URL (defaults to SELECTAI_ORDS_BASE_URL)
            ords_schema: Override ORDS schema (defaults to SELECTAI_ORDS_SCHEMA)
            ords_auth_token: Override ORDS auth token (defaults to SELECTAI_ORDS_AUTH_TOKEN)

        Returns:
            JSON with generated SQL and/or query results

        Example:
            oci_selectai_generate(
                prompt="Show me customers with orders over $10000",
                action="runsql"
            )
        """
        return await _selectai_generate_logic(
            prompt=prompt,
            profile_name=profile_name,
            action=action,
            ords_base_url=ords_base_url,
            ords_schema=ords_schema,
            ords_auth_token=ords_auth_token,
        )

    @mcp.tool()
    async def oci_selectai_list_profiles(
        ords_base_url: str | None = None,
        ords_schema: str | None = None,
        ords_auth_token: str | None = None,
    ) -> str:
        """List available SelectAI profiles in the database.

        Returns all AI profiles configured in the database, including their
        provider, model, and status information.

        Args:
            ords_base_url: Override ORDS base URL (defaults to SELECTAI_ORDS_BASE_URL)
            ords_schema: Override ORDS schema (defaults to SELECTAI_ORDS_SCHEMA)
            ords_auth_token: Override ORDS auth token (defaults to SELECTAI_ORDS_AUTH_TOKEN)

        Returns:
            JSON with list of AI profiles
        """
        return await _selectai_list_profiles_logic(
            ords_base_url=ords_base_url,
            ords_schema=ords_schema,
            ords_auth_token=ords_auth_token,
        )

    @mcp.tool()
    async def oci_selectai_get_profile_tables(
        profile_name: str,
        ords_base_url: str | None = None,
        ords_schema: str | None = None,
        ords_auth_token: str | None = None,
    ) -> str:
        """Get tables associated with a SelectAI profile.

        Returns the list of tables that a profile has been configured to access
        for natural language queries.

        Args:
            profile_name: Name of the AI profile
            ords_base_url: Override ORDS base URL (defaults to SELECTAI_ORDS_BASE_URL)
            ords_schema: Override ORDS schema (defaults to SELECTAI_ORDS_SCHEMA)
            ords_auth_token: Override ORDS auth token (defaults to SELECTAI_ORDS_AUTH_TOKEN)

        Returns:
            JSON with profile tables information
        """
        return await _selectai_get_profile_tables_logic(
            profile_name=profile_name,
            ords_base_url=ords_base_url,
            ords_schema=ords_schema,
            ords_auth_token=ords_auth_token,
        )

    @mcp.tool()
    async def oci_selectai_run_agent(
        team_name: str,
        user_prompt: str,
        session_id: str | None = None,
        ords_base_url: str | None = None,
        ords_schema: str | None = None,
        ords_auth_token: str | None = None,
    ) -> str:
        """Execute a SelectAI agent team.

        Runs a pre-configured SelectAI agent team with the given prompt.
        Agent teams can orchestrate multiple tools and perform complex
        multi-step workflows.

        Args:
            team_name: Name of the agent team to run (e.g., "SALES_ANALYST_TEAM")
            user_prompt: User's question or request for the agent
            session_id: Optional session ID for conversation continuity
            ords_base_url: Override ORDS base URL (defaults to SELECTAI_ORDS_BASE_URL)
            ords_schema: Override ORDS schema (defaults to SELECTAI_ORDS_SCHEMA)
            ords_auth_token: Override ORDS auth token (defaults to SELECTAI_ORDS_AUTH_TOKEN)

        Returns:
            JSON with agent execution results

        Example:
            oci_selectai_run_agent(
                team_name="QUARTERLY_ANALYSIS_TEAM",
                user_prompt="What were last quarter's top performing regions?"
            )
        """
        return await _selectai_run_agent_logic(
            team_name=team_name,
            user_prompt=user_prompt,
            session_id=session_id,
            ords_base_url=ords_base_url,
            ords_schema=ords_schema,
            ords_auth_token=ords_auth_token,
        )

    @mcp.tool()
    async def oci_selectai_ping() -> dict:
        """Health check for SelectAI configuration.

        Returns the current SelectAI configuration status and connectivity.
        Shows both wallet-based and ORDS connection options.
        """
        wallet_config = _get_wallet_config()
        ords_config = _get_ords_config()

        wallet_configured = _is_wallet_configured()
        ords_configured = bool(ords_config["base_url"])

        # Determine overall status
        if wallet_configured or ords_configured:
            status = "ok"
        else:
            status = "not_configured"

        return {
            "status": status,
            "connections": {
                "wallet": {
                    "configured": wallet_configured,
                    "tns_name": wallet_config["tns_name"] if wallet_configured else None,
                    "user": wallet_config["user"] if wallet_configured else None,
                },
                "ords": {
                    "configured": ords_configured,
                    "base_url": ords_config["base_url"][:50] + "..." if ords_config["base_url"] and len(ords_config["base_url"]) > 50 else ords_config["base_url"],
                    "schema": ords_config["schema"],
                },
            },
            "default_profile": wallet_config["default_profile"] or ords_config["default_profile"],
            "preferred_connection": "wallet" if wallet_configured else ("ords" if ords_configured else "none"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    @mcp.tool()
    async def oci_selectai_test_connection() -> str:
        """Test database connectivity for SelectAI.

        Attempts to connect to the database and run a simple query.
        Tests both wallet and ORDS connections if configured.

        Returns:
            JSON with connection test results
        """
        results = {
            "type": "selectai_connection_test",
            "tests": [],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Test wallet connection
        if _is_wallet_configured():
            sql = "SELECT 'connected' AS status, SYS_CONTEXT('USERENV', 'DB_NAME') AS db_name FROM DUAL"
            wallet_result = await _execute_sql_via_wallet(sql)

            if "error" in wallet_result:
                results["tests"].append({
                    "method": "wallet",
                    "success": False,
                    "error": wallet_result["error"],
                })
            else:
                items = wallet_result.get("items", [])
                db_name = items[0].get("db_name", "unknown") if items else "unknown"
                results["tests"].append({
                    "method": "wallet",
                    "success": True,
                    "database": db_name,
                })

        # Test ORDS connection
        ords_config = _get_ords_config()
        if ords_config["base_url"]:
            sql = "SELECT 'connected' AS status FROM DUAL"
            ords_result = await _execute_sql_via_ords(
                sql,
                ords_config["base_url"],
                ords_config["schema"],
                ords_config["auth_token"],
            )

            if "error" in ords_result:
                results["tests"].append({
                    "method": "ords",
                    "success": False,
                    "error": ords_result["error"],
                })
            else:
                results["tests"].append({
                    "method": "ords",
                    "success": True,
                })

        if not results["tests"]:
            results["error"] = "No connection methods configured"

        return json.dumps(results)
