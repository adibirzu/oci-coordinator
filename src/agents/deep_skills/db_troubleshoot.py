"""
Database Troubleshooting DeepSkills.

Advanced skills for database performance analysis that combine:
- MCP tool calls (SQLcl, Database Management, OPSI)
- Code execution for data analysis
- Self-testing capabilities

Workflow mapping (from CLAUDE.md):
| Intent           | Skill                    | Primary Tool               |
|------------------|--------------------------|----------------------------|
| check_blocking   | DatabaseBlockingSkill    | oci_database_execute_sql   |
| wait_events      | WaitEventsSkill          | oci_dbmgmt_get_wait_events |
| sql_monitoring   | TopSQLSkill              | oci_database_execute_sql   |
| awr_report       | AWRReportSkill           | oci_dbmgmt_get_awr_report  |
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import structlog

from src.agents.core.deep_skills import (
    DeepSkill,
    DeepSkillConfig,
    SkillContext,
    SkillResult,
    register_skill,
)

logger = structlog.get_logger()


# =============================================================================
# Database Blocking Analysis Skill
# =============================================================================

@register_skill(
    skill_id="db_blocking_analysis",
    name="Database Blocking Session Analysis",
    description="Analyze blocking sessions, lock contention, and wait chains",
    compatible_agents=["db_troubleshoot", "coordinator"],
    required_mcp_tools=["oci_database_execute_sql"],
    requires_code_execution=True,
    tags=["database", "blocking", "performance", "troubleshooting"],
)
class DatabaseBlockingSkill(DeepSkill):
    """
    Analyze database blocking sessions.

    Steps:
    1. Query V$SESSION for blocking relationships
    2. Identify blocker/waiter chains
    3. Analyze lock types and wait times
    4. Generate recommendations
    """

    BLOCKING_QUERY = """
    SELECT
        s.sid as waiter_sid,
        s.serial# as waiter_serial,
        s.username as waiter_user,
        s.sql_id as waiter_sql_id,
        s.event as wait_event,
        s.seconds_in_wait,
        s.blocking_session as blocker_sid,
        bs.username as blocker_user,
        bs.sql_id as blocker_sql_id,
        bs.status as blocker_status,
        l.type as lock_type,
        l.lmode,
        l.request
    FROM v$session s
    LEFT JOIN v$session bs ON s.blocking_session = bs.sid
    LEFT JOIN v$lock l ON s.sid = l.sid AND l.request > 0
    WHERE s.blocking_session IS NOT NULL
    ORDER BY s.seconds_in_wait DESC
    """

    ANALYSIS_CODE = """
import json

def analyze_blocking(data):
    if not data.get('rows'):
        return {
            'has_blocking': False,
            'summary': 'No blocking sessions found',
            'blockers': [],
            'recommendations': []
        }

    rows = data['rows']

    # Find root blockers (sessions that block others but aren't blocked)
    blocker_sids = {r.get('blocker_sid') for r in rows if r.get('blocker_sid')}
    waiter_sids = {r.get('waiter_sid') for r in rows}
    root_blockers = blocker_sids - waiter_sids

    # Calculate total wait time
    total_wait = sum(r.get('seconds_in_wait', 0) for r in rows)

    # Group by blocker
    blocker_impact = {}
    for r in rows:
        blocker = r.get('blocker_sid')
        if blocker:
            if blocker not in blocker_impact:
                blocker_impact[blocker] = {
                    'sid': blocker,
                    'user': r.get('blocker_user'),
                    'sql_id': r.get('blocker_sql_id'),
                    'waiters': 0,
                    'total_wait_seconds': 0
                }
            blocker_impact[blocker]['waiters'] += 1
            blocker_impact[blocker]['total_wait_seconds'] += r.get('seconds_in_wait', 0)

    # Sort by impact
    top_blockers = sorted(
        blocker_impact.values(),
        key=lambda x: x['total_wait_seconds'],
        reverse=True
    )[:5]

    # Generate recommendations
    recommendations = []
    if total_wait > 300:  # More than 5 minutes total wait
        recommendations.append('HIGH: Significant blocking detected - consider killing blocker sessions')
    if len(root_blockers) == 1:
        recommendations.append(f'Single root blocker (SID {list(root_blockers)[0]}) - investigate this session')
    if any(r.get('lock_type') == 'TX' for r in rows):
        recommendations.append('Row-level locks detected - check for long-running transactions')

    return {
        'has_blocking': True,
        'summary': f'{len(rows)} blocked sessions, {len(root_blockers)} root blockers',
        'total_wait_seconds': total_wait,
        'root_blockers': list(root_blockers),
        'top_blockers': top_blockers,
        'recommendations': recommendations
    }

result = analyze_blocking(data)
return result
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute blocking session analysis."""
        try:
            # Step 1: Execute blocking query
            query_result = await self.call_mcp_tool(
                context,
                "oci_database_execute_sql",
                {
                    "sql": self.BLOCKING_QUERY,
                    "connection": context.parameters.get("connection", "default")
                }
            )

            if not query_result.get("success", True):
                return SkillResult(
                    success=False,
                    error=query_result.get("error", "Query execution failed")
                )

            # Step 2: Analyze with code execution
            analysis = await self.execute_code(
                context,
                self.ANALYSIS_CODE,
                variables={"data": query_result}
            )

            if not analysis.success:
                # Return raw data if analysis fails
                return SkillResult(
                    success=True,
                    data={
                        "raw_data": query_result,
                        "analysis_error": analysis.error
                    },
                    metadata={"analysis_failed": True}
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "query_rows": len(query_result.get("rows", [])),
                    "execution_time_ms": analysis.execution_time_ms
                }
            )

        except Exception as e:
            logger.error("blocking_analysis_failed", error=str(e))
            return SkillResult(success=False, error=str(e))

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test skill functionality."""
        try:
            # Test that we can execute SQL
            result = await self.call_mcp_tool(
                context,
                "oci_database_execute_sql",
                {"sql": "SELECT 1 FROM DUAL", "connection": "default"}
            )
            success = result.get("success", False) or "rows" in result
            return SkillResult(
                success=success,
                data={"test": "sql_execution", "result": result} if success else None,
                error=None if success else "SQL execution test failed"
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e), error_type="SelfTestError")


# =============================================================================
# Wait Events Analysis Skill
# =============================================================================

@register_skill(
    skill_id="db_wait_events",
    name="Database Wait Events Analysis",
    description="Analyze database wait events and performance bottlenecks",
    compatible_agents=["db_troubleshoot", "coordinator"],
    required_mcp_tools=["oci_dbmgmt_summarize_awr_wait_events"],
    requires_code_execution=True,
    tags=["database", "wait_events", "performance", "awr"],
)
class WaitEventsSkill(DeepSkill):
    """
    Analyze wait events from AWR data.

    Steps:
    1. Get AWR snapshots for time range
    2. Summarize wait events
    3. Categorize and prioritize
    4. Generate recommendations
    """

    ANALYSIS_CODE = """
import json

def analyze_wait_events(data):
    if not data.get('items'):
        return {
            'has_issues': False,
            'summary': 'No significant wait events found',
            'categories': {},
            'recommendations': []
        }

    items = data['items']

    # Group by category
    categories = {}
    for item in items:
        category = item.get('wait_category', 'Other')
        if category not in categories:
            categories[category] = {
                'total_time_ms': 0,
                'events': []
            }
        categories[category]['total_time_ms'] += item.get('time_waited_ms', 0)
        categories[category]['events'].append({
            'name': item.get('event_name'),
            'time_ms': item.get('time_waited_ms', 0),
            'count': item.get('wait_count', 0)
        })

    # Sort events within categories
    for cat in categories.values():
        cat['events'] = sorted(cat['events'], key=lambda x: x['time_ms'], reverse=True)[:5]

    # Find top category
    top_category = max(categories.items(), key=lambda x: x[1]['total_time_ms'])

    # Generate recommendations
    recommendations = []
    if 'User I/O' in categories and categories['User I/O']['total_time_ms'] > 10000:
        recommendations.append('HIGH: User I/O waits indicate storage performance issues')
    if 'CPU' in categories and categories['CPU']['total_time_ms'] > 5000:
        recommendations.append('CPU contention detected - review parallel operations')
    if 'Concurrency' in categories:
        recommendations.append('Lock contention detected - review transaction design')

    return {
        'has_issues': len(items) > 0,
        'summary': f'Top wait category: {top_category[0]}',
        'total_events': len(items),
        'categories': categories,
        'top_category': top_category[0],
        'recommendations': recommendations
    }

result = analyze_wait_events(data)
return result
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute wait events analysis."""
        try:
            # Get parameters
            database_id = context.parameters.get("database_id")
            begin_snapshot = context.parameters.get("begin_snapshot_id")
            end_snapshot = context.parameters.get("end_snapshot_id")

            if not all([database_id, begin_snapshot, end_snapshot]):
                return SkillResult(
                    success=False,
                    error="Required parameters: database_id, begin_snapshot_id, end_snapshot_id"
                )

            # Step 1: Get wait events summary
            wait_result = await self.call_mcp_tool(
                context,
                "oci_dbmgmt_summarize_awr_wait_events",
                {
                    "managed_database_id": database_id,
                    "begin_snapshot_id": begin_snapshot,
                    "end_snapshot_id": end_snapshot,
                    "top_n": 20
                }
            )

            # Step 2: Analyze with code execution
            analysis = await self.execute_code(
                context,
                self.ANALYSIS_CODE,
                variables={"data": wait_result}
            )

            if not analysis.success:
                return SkillResult(
                    success=True,
                    data={"raw_data": wait_result, "analysis_error": analysis.error}
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={"database_id": database_id}
            )

        except Exception as e:
            logger.error("wait_events_analysis_failed", error=str(e))
            return SkillResult(success=False, error=str(e))


# =============================================================================
# Top SQL Analysis Skill
# =============================================================================

@register_skill(
    skill_id="db_top_sql",
    name="Top SQL Analysis",
    description="Identify and analyze top SQL by CPU, elapsed time, or executions",
    compatible_agents=["db_troubleshoot", "coordinator"],
    required_mcp_tools=["oci_opsi_get_sql_statistics"],
    requires_code_execution=True,
    tags=["database", "sql", "performance", "tuning"],
)
class TopSQLSkill(DeepSkill):
    """
    Analyze top SQL statements.

    Steps:
    1. Get SQL statistics from OPSI
    2. Rank by selected metric
    3. Identify optimization opportunities
    4. Generate tuning recommendations
    """

    ANALYSIS_CODE = """
import json

def analyze_top_sql(data, sort_by='cpu_time'):
    items = data.get('items', data.get('sql_stats', []))
    if not items:
        return {
            'has_issues': False,
            'summary': 'No SQL statistics available',
            'top_sql': [],
            'recommendations': []
        }

    # Normalize field names
    normalized = []
    for item in items:
        normalized.append({
            'sql_id': item.get('sql_id', item.get('sqlIdentifier', '')),
            'cpu_time_sec': item.get('cpu_time_in_sec', item.get('cpuTimeInSec', 0)),
            'elapsed_time_sec': item.get('elapsed_time_in_sec', item.get('elapsedTimeInSec', 0)),
            'executions': item.get('executions_count', item.get('executionsCount', 0)),
            'buffer_gets': item.get('buffer_gets', item.get('bufferGets', 0)),
            'disk_reads': item.get('disk_reads', item.get('diskReads', 0)),
            'sql_text': item.get('sql_text', item.get('sqlText', ''))[:200]
        })

    # Calculate efficiency metrics
    for sql in normalized:
        if sql['executions'] > 0:
            sql['cpu_per_exec'] = sql['cpu_time_sec'] / sql['executions']
            sql['elapsed_per_exec'] = sql['elapsed_time_sec'] / sql['executions']
            sql['gets_per_exec'] = sql['buffer_gets'] / sql['executions']
        else:
            sql['cpu_per_exec'] = 0
            sql['elapsed_per_exec'] = 0
            sql['gets_per_exec'] = 0

    # Sort by specified metric
    sort_key = 'cpu_time_sec' if sort_by == 'cpu_time' else sort_by
    top_sql = sorted(normalized, key=lambda x: x.get(sort_key, 0), reverse=True)[:10]

    # Generate recommendations
    recommendations = []
    total_cpu = sum(s['cpu_time_sec'] for s in normalized)
    if top_sql and top_sql[0]['cpu_time_sec'] > total_cpu * 0.5:
        recommendations.append(f"SQL {top_sql[0]['sql_id']} consumes >50% of CPU - prioritize tuning")

    high_buffer_gets = [s for s in top_sql if s['gets_per_exec'] > 100000]
    if high_buffer_gets:
        recommendations.append(f"{len(high_buffer_gets)} SQL statements with high buffer gets - check indexes")

    return {
        'has_issues': len(top_sql) > 0,
        'summary': f'Analyzed {len(normalized)} SQL statements',
        'total_cpu_seconds': total_cpu,
        'top_sql': top_sql,
        'recommendations': recommendations
    }

result = analyze_top_sql(data, sort_by)
return result
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute top SQL analysis."""
        try:
            # Get parameters
            database_id = context.parameters.get("database_id")
            hours_back = context.parameters.get("hours_back", 24)
            sort_by = context.parameters.get("sort_by", "cpuTimeInSec")
            limit = context.parameters.get("limit", 20)

            # Step 1: Get SQL statistics
            sql_result = await self.call_mcp_tool(
                context,
                "oci_opsi_get_sql_statistics",
                {
                    "database_id": database_id,
                    "hours_back": hours_back,
                    "sort_by": sort_by,
                    "limit": limit
                }
            )

            # Step 2: Analyze with code execution
            analysis = await self.execute_code(
                context,
                self.ANALYSIS_CODE,
                variables={"data": sql_result, "sort_by": sort_by}
            )

            if not analysis.success:
                return SkillResult(
                    success=True,
                    data={"raw_data": sql_result, "analysis_error": analysis.error}
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={"database_id": database_id, "hours_back": hours_back}
            )

        except Exception as e:
            logger.error("top_sql_analysis_failed", error=str(e))
            return SkillResult(success=False, error=str(e))


# =============================================================================
# AWR Report Skill
# =============================================================================

@register_skill(
    skill_id="db_awr_report",
    name="AWR Report Generation",
    description="Generate and analyze AWR reports for performance tuning",
    compatible_agents=["db_troubleshoot", "coordinator"],
    required_mcp_tools=["oci_dbmgmt_get_awr_report_auto", "oci_dbmgmt_list_awr_snapshots"],
    requires_code_execution=False,
    tags=["database", "awr", "performance", "report"],
)
class AWRReportSkill(DeepSkill):
    """
    Generate AWR reports.

    Steps:
    1. Find available snapshots
    2. Generate AWR report
    3. Return report content
    """

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute AWR report generation."""
        try:
            database_id = context.parameters.get("database_id")
            hours_back = context.parameters.get("hours_back", 1)
            report_format = context.parameters.get("format", "HTML")

            if not database_id:
                return SkillResult(
                    success=False,
                    error="Required parameter: database_id"
                )

            # Use auto-generated report (finds snapshots automatically)
            report_result = await self.call_mcp_tool(
                context,
                "oci_dbmgmt_get_awr_report_auto",
                {
                    "managed_database_id": database_id,
                    "hours_back": hours_back,
                    "report_format": report_format
                }
            )

            if report_result.get("error"):
                return SkillResult(
                    success=False,
                    error=report_result.get("error")
                )

            return SkillResult(
                success=True,
                data={
                    "report_content": report_result.get("content"),
                    "report_format": report_format,
                    "database_id": database_id,
                    "time_range_hours": hours_back
                },
                metadata={
                    "begin_snapshot": report_result.get("begin_snapshot_id"),
                    "end_snapshot": report_result.get("end_snapshot_id")
                }
            )

        except Exception as e:
            logger.error("awr_report_failed", error=str(e))
            return SkillResult(success=False, error=str(e))


# =============================================================================
# Database Health Check Skill
# =============================================================================

@register_skill(
    skill_id="db_health_check",
    name="Database Health Check",
    description="Comprehensive database health assessment",
    compatible_agents=["db_troubleshoot", "coordinator"],
    required_mcp_tools=["oci_opsi_get_performance_summary", "oci_dbmgmt_count_attention_logs"],
    requires_code_execution=True,
    tags=["database", "health", "monitoring"],
)
class DatabaseHealthCheckSkill(DeepSkill):
    """
    Comprehensive database health check.

    Steps:
    1. Get performance metrics (CPU, memory, I/O)
    2. Check attention logs for alerts
    3. Analyze overall health
    4. Generate health score and recommendations
    """

    HEALTH_ANALYSIS_CODE = """
import json

def analyze_health(perf_data, attention_data):
    # Initialize health score (100 = perfect)
    health_score = 100
    issues = []
    recommendations = []

    # Analyze performance metrics
    cpu_analysis = perf_data.get('cpu', {})
    memory_analysis = perf_data.get('memory', {})
    io_analysis = perf_data.get('io', {})

    # CPU health
    cpu_avg = cpu_analysis.get('avg_utilization', 0)
    cpu_max = cpu_analysis.get('max_utilization', 0)
    if cpu_avg > 80:
        health_score -= 20
        issues.append(f'HIGH CPU: Average {cpu_avg}%')
        recommendations.append('Consider adding CPU resources or optimizing queries')
    elif cpu_avg > 60:
        health_score -= 10
        issues.append(f'MODERATE CPU: Average {cpu_avg}%')

    # Memory health
    mem_avg = memory_analysis.get('avg_utilization', 0)
    if mem_avg > 90:
        health_score -= 15
        issues.append(f'HIGH MEMORY: Average {mem_avg}%')
        recommendations.append('Review memory settings or add memory')

    # I/O health
    io_wait = io_analysis.get('avg_io_wait', 0)
    if io_wait > 20:
        health_score -= 15
        issues.append(f'HIGH I/O WAIT: {io_wait}%')
        recommendations.append('Check storage performance')

    # Attention logs
    urgent_count = attention_data.get('IMMEDIATE', 0)
    warning_count = attention_data.get('SOON', 0)

    if urgent_count > 0:
        health_score -= (urgent_count * 10)
        issues.append(f'{urgent_count} urgent attention logs')
        recommendations.append('Review urgent attention logs immediately')

    if warning_count > 5:
        health_score -= 5
        issues.append(f'{warning_count} warning attention logs')

    # Ensure score is within bounds
    health_score = max(0, min(100, health_score))

    # Determine status
    if health_score >= 90:
        status = 'HEALTHY'
    elif health_score >= 70:
        status = 'WARNING'
    elif health_score >= 50:
        status = 'DEGRADED'
    else:
        status = 'CRITICAL'

    return {
        'health_score': health_score,
        'status': status,
        'issues': issues,
        'recommendations': recommendations,
        'metrics': {
            'cpu_avg': cpu_avg,
            'memory_avg': mem_avg,
            'io_wait': io_wait,
            'urgent_alerts': urgent_count,
            'warnings': warning_count
        }
    }

result = analyze_health(perf_data, attention_data)
return result
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute database health check."""
        try:
            database_id = context.parameters.get("database_id")
            hours_back = context.parameters.get("hours_back", 24)

            if not database_id:
                return SkillResult(
                    success=False,
                    error="Required parameter: database_id"
                )

            # Get performance summary and attention logs in parallel
            perf_result = await self.call_mcp_tool(
                context,
                "oci_opsi_get_performance_summary",
                {"database_id": database_id, "hours_back": hours_back}
            )

            attention_result = await self.call_mcp_tool(
                context,
                "oci_dbmgmt_count_attention_logs",
                {"managed_database_id": database_id, "hours_back": hours_back}
            )

            # Analyze health
            analysis = await self.execute_code(
                context,
                self.HEALTH_ANALYSIS_CODE,
                variables={
                    "perf_data": perf_result,
                    "attention_data": attention_result
                }
            )

            if not analysis.success:
                return SkillResult(
                    success=True,
                    data={
                        "performance": perf_result,
                        "attention_logs": attention_result,
                        "analysis_error": analysis.error
                    }
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "database_id": database_id,
                    "hours_back": hours_back
                }
            )

        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return SkillResult(success=False, error=str(e))

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test skill functionality."""
        try:
            # Test OPSI connectivity
            result = await self.call_mcp_tool(
                context,
                "oci_opsi_list_skills",
                {}
            )
            success = "skills" in result or isinstance(result, list)
            return SkillResult(
                success=success,
                data={"test": "opsi_connectivity", "result": result} if success else None,
                error=None if success else "OPSI connectivity test failed"
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e), error_type="SelfTestError")
