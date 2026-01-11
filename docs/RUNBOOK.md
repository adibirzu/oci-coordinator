# Operations Runbook

## Overview

This runbook provides operational procedures for the OCI AI Agent Coordinator, covering common issues, diagnostics, and recovery procedures.

**Last Updated**: 2026-01-11

---

## Quick Reference

| Issue | First Check | Recovery |
|-------|-------------|----------|
| MCP server disconnected | Health monitor status | Auto-restart (wait 60s) or manual restart |
| LLM timeouts | Rate limiter queue | Check provider status, increase timeout |
| Slack not responding | Socket mode status | Restart Slack handler |
| Tool execution failing | Circuit breaker status | Wait for cooldown or force reset |
| Memory/Redis errors | Redis connection | Restart Redis or check connectivity |

---

## 1. MCP Server Diagnostics

### 1.1 Check Server Status

```bash
# Via API
curl http://localhost:3001/mcp/servers

# Expected response
{
  "servers": [
    {"id": "oci-unified", "status": "connected", "tool_count": 30},
    {"id": "database-observatory", "status": "connected", "tool_count": 32},
    ...
  ]
}
```

### 1.2 MCP Server Not Connecting

**Symptoms:**
- Tools returning "Server temporarily unavailable"
- Health monitor shows UNHEALTHY status
- Circuit breaker is OPEN

**Diagnosis:**
```bash
# Check if server process is running
ps aux | grep "mcp-oci"

# Check server logs
tail -f logs/mcp_database-observatory.log

# Test server manually
python -m src.mcp_server --test
```

**Recovery:**
```python
# Manual restart via Python
from src.mcp.registry import ServerRegistry

registry = ServerRegistry.get_instance()
await registry.disconnect("database-observatory")
await asyncio.sleep(2)
await registry.connect("database-observatory")
```

### 1.3 Circuit Breaker Open

**Symptoms:**
- Tools failing immediately with "Server temporarily unavailable"
- No actual API calls being made

**Check Circuit State:**
```python
from src.mcp.registry import ServerRegistry

registry = ServerRegistry.get_instance()
for server_id in registry.list_servers():
    print(f"{server_id}: circuit_open={registry._is_circuit_open(server_id)}")
```

**Force Reset (Use with caution):**
```python
# Reset failure count to close circuit
registry._failure_counts["database-observatory"] = 0
registry._circuit_open_until["database-observatory"] = datetime.min
```

### 1.4 FinOps Coordinator Registration

**Symptoms:**
- FinOps tools appear but coordinator shows "not registered"
- `finops_register_with_coordinator` returns "skipped"

**Required configuration:**
```bash
# Coordinator callback endpoint for FinOps MCP registration
# Use your coordinator base URL or registration endpoint.
export FINOPS_COORDINATOR_ENDPOINT=http://localhost:3001
```

**Recovery:**
```bash
# Trigger registration (from finopsai-mcp)
poetry run python -c "import asyncio; from finopsai_mcp.coordinator import register_with_coordinator; print(asyncio.run(register_with_coordinator(force=True)))"
```
Note: The current FinOps registration stores metadata locally; a coordinator HTTP callback is a future integration.

### 1.5 SQLcl Database Connectivity

**Environment Variables Required** (`.env.local`):
```bash
SQLCL_PATH=/Applications/sqlcl/bin/sql
SQLCL_TNS_ADMIN=~/oci_wallets_unified
SQLCL_DB_USERNAME=ADMIN
SQLCL_DB_PASSWORD="<password>"
SQLCL_DB_CONNECTION=th_high
SQLCL_FALLBACK_CONNECTION=ATPAdi
```

**Symptoms:**
- "No database connection available" errors
- SQL execution timeouts
- Empty connection list from `oci_database_list_connections`

**Diagnosis:**
```bash
# List available connections
poetry run python -c "
import asyncio
from src.mcp.server.tools.database import _list_database_connections_logic
result = asyncio.run(_list_database_connections_logic())
print(result)
"

# Test SQL execution
poetry run python -c "
import asyncio
from src.mcp.server.tools.database import _execute_sql_logic
result = asyncio.run(_execute_sql_logic(
    sql='SELECT NAME FROM V\$DATABASE',
    connection_name='th_high',
    timeout_seconds=30
))
print(result)
"
```

**Available Connections** (after unified wallet setup):
| Connection | Database | Status |
|------------|----------|--------|
| `th_high` | FCEK9UA6 | Default |
| `th_medium` | FCEK9UA6 | Available |
| `th_low` | FCEK9UA6 | Available |
| `ATPAdi_high` | FCECIC39 | Available |
| `ATPAdi` | FCECIC39 | Available |
| `SelectAI_high` | FCECIC39 | Available |
| `SelectAI` | FCECIC39 | Available |

**Recovery:**
1. Verify wallet files exist: `ls ~/oci_wallets_unified/`
2. Verify tnsnames.ora has entries: `cat ~/oci_wallets_unified/tnsnames.ora`
3. Export environment variables: `set -a && source .env.local && set +a`
4. Test SQLcl directly: `$SQLCL_PATH -L ADMIN/<password>@th_high`

---

## 2. LLM Provider Issues

### 2.1 Check Rate Limiter Status

```python
from src.llm import get_llm

llm = get_llm()
if hasattr(llm, 'get_metrics'):
    metrics = llm.get_metrics()
    print(f"Queue size: {metrics.current_queue_size}")
    print(f"Total requests: {metrics.total_requests}")
    print(f"Timeouts: {metrics.timeout_count}")
```

### 2.2 LLM Timeout Errors

**Symptoms:**
- "Tool call timed out" errors
- Slow response times
- Queue buildup in rate limiter

**Diagnosis:**
```bash
# Check LLM provider status
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -d '{"model": "claude-sonnet-4-20250514", "max_tokens": 10, "messages": [{"role": "user", "content": "ping"}]}'
```

**Mitigation:**
```python
# Increase timeout for specific operations
from src.llm.rate_limiter import wrap_with_rate_limiter

llm = wrap_with_rate_limiter(
    base_llm,
    max_concurrent=5,
    timeout_seconds=600,  # Increase from default 300
)
```

### 2.3 OCA Authentication Expired

**Symptoms:**
- "Token expired" errors
- OCA calls failing with 401

**Recovery:**
```bash
# Refresh OCA token
poetry run python scripts/oca_auth.py

# Check token status
poetry run python scripts/oca_auth.py --status
```

---

## 3. Slack Integration Issues

### 3.1 Check Slack Connection

```bash
# Via API
curl http://localhost:3001/slack/status

# Expected response
{
  "connected": true,
  "bot_id": "U12345",
  "team": "my-workspace",
  "socket_mode": true
}
```

### 3.2 Slack Not Responding

**Symptoms:**
- Messages not being acknowledged
- 3-second timeout messages appearing
- Socket disconnections

**Diagnosis:**
```bash
# Check Slack app tokens
echo $SLACK_BOT_TOKEN | head -c 10
echo $SLACK_APP_TOKEN | head -c 10

# Verify token validity
curl -X POST https://slack.com/api/auth.test \
  -H "Authorization: Bearer $SLACK_BOT_TOKEN"
```

**Recovery:**
```bash
# Restart with Slack only
poetry run python -m src.main --mode slack

# Or restart both modes
poetry run python -m src.main --mode both
```

### 3.3 Duplicate Message Handling

If Slack retries are causing duplicate processing:

```python
# Check if message was already processed
from src.memory import SharedMemoryManager

memory = SharedMemoryManager.get_instance()
processed = await memory.redis.get(f"processed:{message_ts}")
if processed:
    return  # Skip duplicate
await memory.redis.setex(f"processed:{message_ts}", 300, "1")
```

---

## 4. Redis/Memory Issues

### 4.1 Check Redis Connection

```bash
# Test Redis connectivity
redis-cli ping
# Expected: PONG

# Check memory usage
redis-cli info memory
```

### 4.2 Redis Connection Errors

**Symptoms:**
- "Connection refused" errors
- Deadletter queue not persisting
- Session state not saving

**Diagnosis:**
```bash
# Check Redis server status
systemctl status redis

# Check connection from coordinator
python -c "import redis; r = redis.from_url('redis://localhost:6379'); print(r.ping())"
```

**Recovery:**
```bash
# Restart Redis
systemctl restart redis

# Fallback: Run with in-memory storage
export REDIS_URL=""
poetry run python -m src.main
```

### 4.3 Clear Stale Cache

```bash
# Clear all coordinator keys
redis-cli KEYS "oci:*" | xargs redis-cli DEL

# Clear only session state
redis-cli KEYS "session:*" | xargs redis-cli DEL

# Clear deadletter queue
redis-cli DEL "oci:deadletter:queue"
```

---

## 5. Deadletter Queue Management

### 5.1 View Failed Operations

```python
from src.resilience import DeadLetterQueue

dlq = DeadLetterQueue(redis_url="redis://localhost:6379")

# Get failed operations
failures = await dlq.get_failed(limit=10)
for f in failures:
    print(f"Operation: {f.operation}")
    print(f"Error: {f.error}")
    print(f"Timestamp: {f.timestamp}")
    print("---")
```

### 5.2 Retry Failed Operations

```python
from src.resilience import DeadLetterQueue, FailureType

dlq = DeadLetterQueue(redis_url="redis://localhost:6379")

# Retry all timeouts
result = await dlq.retry_failed(
    failure_types=[FailureType.TIMEOUT],
    max_retries=3,
)
print(f"Retried: {result.retried}, Failed: {result.failed}")
```

### 5.3 Discard Failed Operations

```python
# Discard specific operation
await dlq.discard(operation_id="op_12345")

# Discard all older than 7 days (automatic via TTL)
```

---

## 6. Health Monitor Operations

### 6.1 Check System Health

```python
from src.resilience import HealthMonitor

monitor = HealthMonitor.get_instance()
status = await monitor.get_status()

for component, health in status.items():
    print(f"{component}: {health.status.value}")
    if health.message:
        print(f"  Message: {health.message}")
    print(f"  Last check: {health.last_check}")
```

### 6.2 Force Health Check

```python
# Run immediate health check
from src.resilience import HealthMonitor

monitor = HealthMonitor.get_instance()
await monitor.check_all()
```

### 6.3 Disable Auto-Restart

```python
# Temporarily disable auto-restart for a component
from src.resilience import HealthMonitor

monitor = HealthMonitor.get_instance()
check = monitor._checks.get("mcp_database-observatory")
if check:
    check.critical = False  # Disable auto-restart
```

---

## 7. Bulkhead Operations

### 7.1 Check Partition Usage

```python
from src.resilience import Bulkhead

bulkhead = Bulkhead.get_instance()

for partition_name in ["database", "cost", "infrastructure", "security"]:
    metrics = await bulkhead.get_metrics(partition_name)
    print(f"{partition_name}:")
    print(f"  Active: {metrics.active_count}/{metrics.max_concurrent}")
    print(f"  Waiting: {metrics.waiting_count}")
    print(f"  Timeouts: {metrics.timeout_count}")
```

### 7.2 Adjust Partition Limits

```python
# Increase limit for heavy workloads (temporary)
from src.resilience import Bulkhead, BulkheadPartition

bulkhead = Bulkhead.get_instance()
partition = bulkhead._partitions[BulkheadPartition.DATABASE]
partition._semaphore = asyncio.Semaphore(5)  # Increase from 3 to 5
```

---

## 8. Tool Catalog Validation

### 8.1 Validate All Tools

```python
from src.mcp.validation import validate_tool_catalog
from src.mcp.catalog import ToolCatalog

catalog = ToolCatalog.get_instance()
result = await validate_tool_catalog(catalog)

print(result.summary())
if not result.valid:
    for error in result.get_errors():
        print(f"ERROR: {error.tool_name} - {error.message}")
```

### 8.2 Startup Health Verification

```python
from src.mcp.validation import verify_startup_health

result = await verify_startup_health(catalog)
for category, status in result.items():
    print(f"{category}: {status['status']}")
    if status.get('error'):
        print(f"  Error: {status['error']}")
```

---

## 9. Common Error Patterns

### 9.1 OCI API Rate Limiting

**Symptoms:**
- 429 errors from OCI APIs
- "Too many requests" messages

**Mitigation:**
```python
# Add exponential backoff
import asyncio

async def call_with_backoff(func, *args, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await func(*args)
        except RateLimitError:
            wait = 2 ** attempt
            await asyncio.sleep(wait)
    raise Exception("Max retries exceeded")
```

### 9.2 OCI Service Errors

| Error Code | Meaning | Action |
|------------|---------|--------|
| 400 | Bad request | Check parameters |
| 401 | Unauthorized | Refresh OCI config |
| 403 | Forbidden | Check IAM policies |
| 404 | Not found | Verify OCID exists |
| 429 | Rate limited | Wait and retry |
| 500 | Server error | Retry with backoff |
| 503 | Service unavailable | Wait for OCI recovery |

### 9.3 Timeout Patterns

| Operation | Default Timeout | Recommended |
|-----------|-----------------|-------------|
| Cost summary | 30s | 30s (OCI Usage API slow) |
| List instances | 120s | 180s for large compartments |
| SQL execution | 60s | 120s for complex queries |
| Discovery scan | 60s | 120s for full tenancy |

---

## 10. Emergency Procedures

### 10.1 Full System Restart

```bash
# Stop all services
pkill -f "python -m src.main"

# Clear stale state
redis-cli FLUSHDB

# Restart
poetry run python -m src.main --mode both
```

### 10.2 Rollback to Previous Version

```bash
# Check current version
git log -1 --format="%H %s"

# Rollback
git checkout HEAD~1
poetry install
poetry run python -m src.main
```

### 10.3 Disable All Auto-Healing

```bash
# Disable rate limiting
export LLM_RATE_LIMITING=false

# Disable health monitor auto-restart
export HEALTH_MONITOR_AUTO_RESTART=false

# Start in degraded mode
poetry run python -m src.main
```

---

## 11. Monitoring Dashboards

### 11.1 Key Metrics to Watch

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| Rate limiter queue | > 10 | > 20 |
| Circuit breakers open | > 1 | > 2 |
| Health check failures | > 2 | > 3 |
| DLQ size | > 50 | > 100 |
| LLM latency p95 | > 10s | > 30s |

### 11.2 API Health Endpoint

```bash
# Comprehensive health check
curl http://localhost:3001/health

# Expected response
{
  "status": "healthy",
  "components": {
    "mcp_servers": "healthy",
    "redis": "healthy",
    "llm": "healthy",
    "slack": "connected"
  },
  "uptime_seconds": 3600
}
```

---

## 12. Log Analysis

### 12.1 Find Errors

```bash
# Recent errors
grep -i "error" logs/coordinator.log | tail -20

# Tool failures
grep "tool_call_failed" logs/coordinator.log | tail -20

# Circuit breaker events
grep "circuit" logs/coordinator.log
```

### 12.2 Trace Correlation

```bash
# Find all logs for a trace
TRACE_ID="abc123"
grep "$TRACE_ID" logs/*.log
```

### 12.3 Performance Analysis

```bash
# Slow tool calls (> 10s)
grep "duration_ms" logs/coordinator.log | awk -F'duration_ms=' '{if ($2 > 10000) print}'

# LLM latency
grep "llm.invoke" logs/coordinator.log | grep "duration"
```

---

## Appendix: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_RATE_LIMITING` | `true` | Enable LLM rate limiting |
| `LLM_MAX_CONCURRENT` | `5` | Max concurrent LLM calls |
| `CIRCUIT_BREAKER_THRESHOLD` | `3` | Failures before circuit opens |
| `CIRCUIT_BREAKER_TIMEOUT` | `60` | Seconds before half-open |
| `HEALTH_CHECK_INTERVAL` | `60` | Seconds between health checks |
| `DLQ_RETENTION_DAYS` | `7` | Deadletter queue retention |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |

---

*End of Operations Runbook*
