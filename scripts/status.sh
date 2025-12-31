#!/bin/bash
#
# OCI AI Agent Coordinator - Status Script
#
# Usage:
#   ./scripts/status.sh         # Show status
#   ./scripts/status.sh --logs  # Show status and recent logs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.coordinator.pid"
LOG_FILE="$PROJECT_DIR/logs/coordinator.log"

SHOW_LOGS=false
LOG_LINES=20

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --logs|-l)
            SHOW_LOGS=true
            shift
            ;;
        --lines|-n)
            LOG_LINES="$2"
            shift 2
            ;;
        --help|-h)
            echo "OCI AI Agent Coordinator - Status Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --logs, -l       Show recent log entries"
            echo "  --lines N, -n N  Number of log lines to show (default: 20)"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "  OCI AI Agent Coordinator Status"
echo "========================================="
echo ""

# Check main coordinator process
echo "Coordinator Process:"
echo "-------------------"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "  Status:  RUNNING"
        echo "  PID:     $PID"

        # Get process details
        PS_INFO=$(ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,command --no-headers 2>/dev/null || true)
        if [ -n "$PS_INFO" ]; then
            CPU=$(echo "$PS_INFO" | awk '{print $3}')
            MEM=$(echo "$PS_INFO" | awk '{print $4}')
            UPTIME=$(echo "$PS_INFO" | awk '{print $5}')
            echo "  CPU:     ${CPU}%"
            echo "  Memory:  ${MEM}%"
            echo "  Uptime:  $UPTIME"
        fi
    else
        echo "  Status:  STOPPED (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    # Try to find by process name
    PIDS=$(pgrep -f "python -m src.main" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "  Status:  RUNNING (no PID file)"
        echo "  PIDs:    $PIDS"
    else
        echo "  Status:  STOPPED"
    fi
fi

echo ""

# Check MCP server processes
echo "MCP Server Processes:"
echo "--------------------"

MCP_PIDS=$(pgrep -f "src.mcp.server" 2>/dev/null || true)
if [ -n "$MCP_PIDS" ]; then
    for pid in $MCP_PIDS; do
        CMD=$(ps -p "$pid" -o command --no-headers 2>/dev/null | head -c 60 || echo "unknown")
        echo "  PID $pid: $CMD..."
    done
else
    echo "  No MCP servers running"
fi

echo ""

# Check Redis
echo "Redis Status:"
echo "-------------"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        REDIS_INFO=$(redis-cli info keyspace 2>/dev/null | grep "^db0" || echo "no keys")
        echo "  Status:  CONNECTED"
        echo "  Keys:    $REDIS_INFO"
    else
        echo "  Status:  NOT CONNECTED"
    fi
else
    echo "  Status:  redis-cli not found"
fi

echo ""

# Check log file
echo "Log File:"
echo "---------"
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LOG_LINES_COUNT=$(wc -l < "$LOG_FILE")
    echo "  Path:    $LOG_FILE"
    echo "  Size:    $LOG_SIZE"
    echo "  Lines:   $LOG_LINES_COUNT"
else
    echo "  No log file found"
fi

echo ""

# Show recent errors
echo "Recent Errors (last 10):"
echo "------------------------"
if [ -f "$LOG_FILE" ]; then
    ERRORS=$(grep -i "error\|exception\|failed" "$LOG_FILE" 2>/dev/null | tail -10 || echo "  No errors found")
    if [ -n "$ERRORS" ]; then
        echo "$ERRORS" | head -10
    else
        echo "  No errors found"
    fi
else
    echo "  No log file"
fi

# Show logs if requested
if [ "$SHOW_LOGS" = true ] && [ -f "$LOG_FILE" ]; then
    echo ""
    echo "========================================="
    echo "  Recent Logs (last $LOG_LINES lines)"
    echo "========================================="
    echo ""
    tail -"$LOG_LINES" "$LOG_FILE"
fi

echo ""
