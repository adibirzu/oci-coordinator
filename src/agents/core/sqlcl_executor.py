"""
SQLcl Direct Executor for Database Operations.

Provides direct SQL execution via MCP tools for database troubleshooting.
Designed for the DB Observability agent to execute performance queries.

Features:
- Direct SQL execution via oci_database_execute_sql MCP tool
- Pre-built query templates for common DB operations
- Result parsing and formatting
- Connection management via MCP

Example usage:
    executor = SQLclExecutor(mcp_client)

    # Execute raw SQL
    result = await executor.execute_sql("SELECT * FROM v$session WHERE status='ACTIVE'")

    # Use pre-built template
    result = await executor.get_blocking_sessions()

    # Execute with parameters
    result = await executor.execute_template(
        "top_sql_by_cpu",
        top_n=10,
        time_range_hours=24
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import structlog

logger = structlog.get_logger()


class QueryType(Enum):
    """Types of database queries."""
    BLOCKING_SESSIONS = "blocking_sessions"
    WAIT_EVENTS = "wait_events"
    SQL_MONITORING = "sql_monitoring"
    LONG_RUNNING_OPS = "long_running_ops"
    PARALLELISM_STATS = "parallelism_stats"
    FULL_TABLE_SCANS = "full_table_scans"
    SESSION_STATS = "session_stats"
    LOCK_ANALYSIS = "lock_analysis"
    TOP_SQL = "top_sql"
    TABLESPACE_USAGE = "tablespace_usage"
    UNDO_USAGE = "undo_usage"
    TEMP_USAGE = "temp_usage"
    CUSTOM = "custom"


@dataclass
class SQLclConfig:
    """Configuration for SQLcl executor."""

    # MCP tool name for SQL execution
    mcp_tool_name: str = "oci_database_execute_sql"

    # Connection settings
    connection_name: str = "default"
    timeout_seconds: int = 120

    # Result settings
    max_rows: int = 1000
    format_results: bool = True

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class QueryResult:
    """Result from SQL query execution."""

    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    row_count: int = 0

    # Error info
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Execution metadata
    query_type: QueryType = QueryType.CUSTOM
    execution_time_ms: int = 0
    executed_at: datetime = field(default_factory=datetime.utcnow)

    # Raw response (for debugging)
    raw_response: Optional[Any] = None


# Pre-built query templates for common DB operations
QUERY_TEMPLATES = {
    QueryType.BLOCKING_SESSIONS: """
        SELECT
            s1.sid AS blocking_sid,
            s1.serial# AS blocking_serial,
            s1.username AS blocking_user,
            s1.machine AS blocking_machine,
            s1.program AS blocking_program,
            s2.sid AS blocked_sid,
            s2.serial# AS blocked_serial,
            s2.username AS blocked_user,
            s2.event AS wait_event,
            s2.seconds_in_wait,
            l1.type AS lock_type,
            DECODE(l1.lmode,
                0, 'None',
                1, 'Null',
                2, 'Row Share',
                3, 'Row Exclusive',
                4, 'Share',
                5, 'Share Row Exclusive',
                6, 'Exclusive') AS lock_mode
        FROM
            v$lock l1
            JOIN v$session s1 ON l1.sid = s1.sid
            JOIN v$lock l2 ON l1.id1 = l2.id1 AND l1.id2 = l2.id2 AND l1.lmode > 0 AND l2.request > 0
            JOIN v$session s2 ON l2.sid = s2.sid
        WHERE
            s1.sid != s2.sid
        ORDER BY
            s2.seconds_in_wait DESC
    """,

    QueryType.WAIT_EVENTS: """
        SELECT
            event,
            wait_class,
            total_waits,
            total_timeouts,
            time_waited / 100 AS time_waited_secs,
            average_wait / 100 AS avg_wait_secs
        FROM
            v$system_event
        WHERE
            wait_class != 'Idle'
            AND total_waits > 0
        ORDER BY
            time_waited DESC
        FETCH FIRST {top_n} ROWS ONLY
    """,

    QueryType.SQL_MONITORING: """
        SELECT
            sql_id,
            sql_exec_id,
            username,
            status,
            sql_text,
            elapsed_time / 1000000 AS elapsed_secs,
            cpu_time / 1000000 AS cpu_secs,
            buffer_gets,
            disk_reads,
            sql_plan_hash_value
        FROM
            v$sql_monitor
        WHERE
            status IN ('EXECUTING', 'DONE (ERROR)')
            OR elapsed_time > {min_elapsed_secs} * 1000000
        ORDER BY
            elapsed_time DESC
        FETCH FIRST {top_n} ROWS ONLY
    """,

    QueryType.LONG_RUNNING_OPS: """
        SELECT
            sid,
            serial#,
            username,
            opname,
            target,
            sofar,
            totalwork,
            ROUND(sofar/NULLIF(totalwork, 0) * 100, 2) AS pct_complete,
            elapsed_seconds,
            time_remaining AS est_remaining_secs,
            message
        FROM
            v$session_longops
        WHERE
            sofar != totalwork
            AND totalwork > 0
        ORDER BY
            elapsed_seconds DESC
    """,

    QueryType.PARALLELISM_STATS: """
        SELECT
            qcsid,
            qcserial#,
            qcinst_id,
            server_set,
            server#,
            degree,
            req_degree,
            sid,
            serial#,
            inst_id,
            status
        FROM
            v$px_session
        ORDER BY
            qcsid, server_set, server#
    """,

    QueryType.FULL_TABLE_SCANS: """
        SELECT
            sql_id,
            plan_hash_value,
            executions,
            buffer_gets,
            disk_reads,
            rows_processed,
            elapsed_time / 1000000 AS elapsed_secs,
            SUBSTR(sql_text, 1, 200) AS sql_text
        FROM
            v$sql
        WHERE
            sql_text LIKE '%TABLE ACCESS FULL%'
            OR sql_id IN (
                SELECT DISTINCT sql_id
                FROM v$sql_plan
                WHERE operation = 'TABLE ACCESS'
                AND options = 'FULL'
            )
        ORDER BY
            buffer_gets DESC
        FETCH FIRST {top_n} ROWS ONLY
    """,

    QueryType.SESSION_STATS: """
        SELECT
            s.sid,
            s.serial#,
            s.username,
            s.status,
            s.machine,
            s.program,
            s.sql_id,
            s.event,
            s.wait_class,
            s.seconds_in_wait,
            s.state,
            s.last_call_et AS last_activity_secs
        FROM
            v$session s
        WHERE
            s.type = 'USER'
            AND s.username IS NOT NULL
        ORDER BY
            s.last_call_et DESC
        FETCH FIRST {top_n} ROWS ONLY
    """,

    QueryType.LOCK_ANALYSIS: """
        SELECT
            o.owner,
            o.object_name,
            o.object_type,
            l.session_id,
            s.serial#,
            s.username,
            s.machine,
            DECODE(l.locked_mode,
                0, 'None',
                1, 'Null',
                2, 'Row Share',
                3, 'Row Exclusive',
                4, 'Share',
                5, 'Share Row Exclusive',
                6, 'Exclusive') AS lock_mode,
            s.seconds_in_wait
        FROM
            v$locked_object l
            JOIN dba_objects o ON l.object_id = o.object_id
            JOIN v$session s ON l.session_id = s.sid
        ORDER BY
            s.seconds_in_wait DESC
    """,

    QueryType.TOP_SQL: """
        SELECT
            sql_id,
            plan_hash_value,
            executions,
            elapsed_time / 1000000 AS elapsed_secs,
            cpu_time / 1000000 AS cpu_secs,
            buffer_gets,
            disk_reads,
            rows_processed,
            SUBSTR(sql_text, 1, 200) AS sql_text
        FROM
            v$sql
        WHERE
            executions > 0
        ORDER BY
            {order_by} DESC
        FETCH FIRST {top_n} ROWS ONLY
    """,

    QueryType.TABLESPACE_USAGE: """
        SELECT
            tablespace_name,
            ROUND(used_space * 8192 / 1024 / 1024, 2) AS used_mb,
            ROUND(tablespace_size * 8192 / 1024 / 1024, 2) AS total_mb,
            ROUND(used_percent, 2) AS used_pct
        FROM
            dba_tablespace_usage_metrics
        ORDER BY
            used_percent DESC
    """,

    QueryType.UNDO_USAGE: """
        SELECT
            tablespace_name,
            status,
            COUNT(*) AS extent_count,
            SUM(bytes) / 1024 / 1024 AS total_mb
        FROM
            dba_undo_extents
        GROUP BY
            tablespace_name, status
        ORDER BY
            tablespace_name, status
    """,

    QueryType.TEMP_USAGE: """
        SELECT
            s.sid,
            s.serial#,
            s.username,
            s.program,
            u.tablespace,
            u.segtype,
            u.contents,
            u.blocks * 8192 / 1024 / 1024 AS size_mb
        FROM
            v$tempseg_usage u
            JOIN v$session s ON u.session_addr = s.saddr
        ORDER BY
            u.blocks DESC
    """,
}


class SQLclExecutor:
    """
    Direct SQL executor for database operations via MCP.

    This executor is designed to be used BY agents (like DB Troubleshoot)
    to execute SQL queries against Oracle databases through the
    oci_database_execute_sql MCP tool.

    Usage:
        # With MCP client
        executor = SQLclExecutor(mcp_client)

        # Execute raw SQL
        result = await executor.execute_sql("SELECT * FROM v$session")

        # Use template
        result = await executor.execute_template(QueryType.BLOCKING_SESSIONS)

        # Get specific analysis
        result = await executor.get_blocking_sessions()
    """

    def __init__(
        self,
        mcp_client: Any,
        config: Optional[SQLclConfig] = None
    ):
        self.mcp_client = mcp_client
        self.config = config or SQLclConfig()

    async def execute_sql(
        self,
        sql: str,
        connection: Optional[str] = None,
        timeout_seconds: Optional[int] = None
    ) -> QueryResult:
        """
        Execute raw SQL query.

        Args:
            sql: SQL statement to execute
            connection: Database connection name (default: from config)
            timeout_seconds: Query timeout (default: from config)

        Returns:
            QueryResult with data or error
        """
        start_time = datetime.utcnow()
        connection = connection or self.config.connection_name
        timeout = timeout_seconds or self.config.timeout_seconds

        try:
            # Call MCP tool
            response = await self.mcp_client.call_tool(
                self.config.mcp_tool_name,
                {
                    "sql": sql.strip(),
                    "connection": connection,
                }
            )

            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Parse response
            return self._parse_response(response, execution_time)

        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error("sqlcl_execution_error", error=str(e), sql=sql[:100])

            return QueryResult(
                success=False,
                error=str(e),
                error_code="EXECUTION_ERROR",
                execution_time_ms=execution_time,
                query_type=QueryType.CUSTOM
            )

    async def execute_template(
        self,
        query_type: QueryType,
        **params
    ) -> QueryResult:
        """
        Execute a pre-built query template.

        Args:
            query_type: Type of query to execute
            **params: Parameters to substitute in template

        Returns:
            QueryResult with data or error
        """
        if query_type not in QUERY_TEMPLATES:
            return QueryResult(
                success=False,
                error=f"Unknown query type: {query_type}",
                error_code="INVALID_QUERY_TYPE",
                query_type=query_type
            )

        # Get template and substitute parameters
        template = QUERY_TEMPLATES[query_type]

        # Default parameters
        defaults = {
            "top_n": 20,
            "min_elapsed_secs": 60,
            "order_by": "elapsed_time",
        }
        defaults.update(params)

        try:
            sql = template.format(**defaults)
        except KeyError as e:
            return QueryResult(
                success=False,
                error=f"Missing required parameter: {e}",
                error_code="MISSING_PARAMETER",
                query_type=query_type
            )

        result = await self.execute_sql(sql)
        result.query_type = query_type
        return result

    # Convenience methods for common queries

    async def get_blocking_sessions(self) -> QueryResult:
        """Get current blocking sessions."""
        return await self.execute_template(QueryType.BLOCKING_SESSIONS)

    async def get_wait_events(self, top_n: int = 20) -> QueryResult:
        """Get top wait events."""
        return await self.execute_template(QueryType.WAIT_EVENTS, top_n=top_n)

    async def get_sql_monitoring(
        self,
        top_n: int = 20,
        min_elapsed_secs: int = 60
    ) -> QueryResult:
        """Get SQL monitoring data."""
        return await self.execute_template(
            QueryType.SQL_MONITORING,
            top_n=top_n,
            min_elapsed_secs=min_elapsed_secs
        )

    async def get_long_running_ops(self) -> QueryResult:
        """Get long-running operations."""
        return await self.execute_template(QueryType.LONG_RUNNING_OPS)

    async def get_parallelism_stats(self) -> QueryResult:
        """Get parallel execution statistics."""
        return await self.execute_template(QueryType.PARALLELISM_STATS)

    async def get_full_table_scans(self, top_n: int = 20) -> QueryResult:
        """Get queries with full table scans."""
        return await self.execute_template(QueryType.FULL_TABLE_SCANS, top_n=top_n)

    async def get_session_stats(self, top_n: int = 50) -> QueryResult:
        """Get session statistics."""
        return await self.execute_template(QueryType.SESSION_STATS, top_n=top_n)

    async def get_lock_analysis(self) -> QueryResult:
        """Get lock analysis."""
        return await self.execute_template(QueryType.LOCK_ANALYSIS)

    async def get_top_sql(
        self,
        order_by: str = "elapsed_time",
        top_n: int = 20
    ) -> QueryResult:
        """Get top SQL by various metrics."""
        valid_order_by = ["elapsed_time", "cpu_time", "buffer_gets", "disk_reads", "executions"]
        if order_by not in valid_order_by:
            order_by = "elapsed_time"
        return await self.execute_template(QueryType.TOP_SQL, order_by=order_by, top_n=top_n)

    async def get_tablespace_usage(self) -> QueryResult:
        """Get tablespace usage."""
        return await self.execute_template(QueryType.TABLESPACE_USAGE)

    async def get_undo_usage(self) -> QueryResult:
        """Get undo segment usage."""
        return await self.execute_template(QueryType.UNDO_USAGE)

    async def get_temp_usage(self) -> QueryResult:
        """Get temporary segment usage."""
        return await self.execute_template(QueryType.TEMP_USAGE)

    def _parse_response(
        self,
        response: Any,
        execution_time_ms: int
    ) -> QueryResult:
        """Parse MCP tool response into QueryResult."""
        # Handle different response formats
        if isinstance(response, dict):
            # Standard dict response
            if "error" in response:
                return QueryResult(
                    success=False,
                    error=response.get("error"),
                    error_code=response.get("error_code"),
                    execution_time_ms=execution_time_ms,
                    raw_response=response
                )

            data = response.get("data", response.get("rows", []))
            columns = response.get("columns", [])

            if isinstance(data, list) and data:
                # If data is list of lists, convert to list of dicts
                if isinstance(data[0], list) and columns:
                    data = [dict(zip(columns, row)) for row in data]
                elif isinstance(data[0], dict):
                    columns = list(data[0].keys()) if not columns else columns

            return QueryResult(
                success=True,
                data=data if isinstance(data, list) else [data] if data else [],
                columns=columns,
                row_count=len(data) if isinstance(data, list) else (1 if data else 0),
                execution_time_ms=execution_time_ms,
                raw_response=response
            )

        elif isinstance(response, list):
            # Direct list response
            columns = list(response[0].keys()) if response and isinstance(response[0], dict) else []
            return QueryResult(
                success=True,
                data=response,
                columns=columns,
                row_count=len(response),
                execution_time_ms=execution_time_ms,
                raw_response=response
            )

        elif isinstance(response, str):
            # String response (possibly error or formatted output)
            if "error" in response.lower() or "ora-" in response.lower():
                return QueryResult(
                    success=False,
                    error=response,
                    execution_time_ms=execution_time_ms,
                    raw_response=response
                )
            return QueryResult(
                success=True,
                data=[{"output": response}],
                columns=["output"],
                row_count=1,
                execution_time_ms=execution_time_ms,
                raw_response=response
            )

        else:
            return QueryResult(
                success=False,
                error=f"Unexpected response type: {type(response)}",
                execution_time_ms=execution_time_ms,
                raw_response=response
            )

    # Self-test capability

    async def self_test(self) -> QueryResult:
        """
        Test SQLcl connectivity and basic functionality.

        This is designed to be called BY agents to verify their
        database connectivity without external tools.
        """
        test_sql = "SELECT 1 AS test_value FROM DUAL"

        result = await self.execute_sql(test_sql)

        if result.success:
            # Verify we got expected result
            if result.data and len(result.data) > 0:
                test_value = result.data[0].get("TEST_VALUE", result.data[0].get("test_value"))
                if test_value == 1:
                    result.error = None
                    return result

            result.success = False
            result.error = "Self-test query succeeded but returned unexpected data"

        return result

    async def get_available_templates(self) -> Dict[str, str]:
        """Get list of available query templates with descriptions."""
        descriptions = {
            QueryType.BLOCKING_SESSIONS: "Find sessions blocking other sessions",
            QueryType.WAIT_EVENTS: "Top wait events by time waited",
            QueryType.SQL_MONITORING: "Active and recent SQL executions",
            QueryType.LONG_RUNNING_OPS: "Long-running operations in progress",
            QueryType.PARALLELISM_STATS: "Parallel query execution statistics",
            QueryType.FULL_TABLE_SCANS: "Queries performing full table scans",
            QueryType.SESSION_STATS: "Session statistics and activity",
            QueryType.LOCK_ANALYSIS: "Object-level lock analysis",
            QueryType.TOP_SQL: "Top SQL by various performance metrics",
            QueryType.TABLESPACE_USAGE: "Tablespace space usage",
            QueryType.UNDO_USAGE: "Undo segment usage",
            QueryType.TEMP_USAGE: "Temporary segment usage per session",
        }
        return {qt.value: desc for qt, desc in descriptions.items()}
