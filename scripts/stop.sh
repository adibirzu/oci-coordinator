#!/bin/bash
#
# OCI AI Agent Coordinator - Stop Script
#
# Usage:
#   ./scripts/stop.sh           # Graceful stop
#   ./scripts/stop.sh --force   # Force kill
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.coordinator.pid"

FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "OCI AI Agent Coordinator - Stop Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force, -f  Force kill (SIGKILL instead of SIGTERM)"
            echo "  --help, -h   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to stop process
stop_process() {
    local pid=$1
    local name=$2
    local signal=$3

    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping $name (PID: $pid)..."
        kill "$signal" "$pid" 2>/dev/null || true

        # Wait for process to terminate
        local count=0
        while ps -p "$pid" > /dev/null 2>&1; do
            sleep 1
            count=$((count + 1))
            if [ $count -ge 10 ]; then
                echo "Process didn't stop gracefully, forcing..."
                kill -9 "$pid" 2>/dev/null || true
                sleep 1
                break
            fi
        done

        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Failed to stop $name"
            return 1
        else
            echo "$name stopped"
            return 0
        fi
    else
        echo "$name is not running"
        return 0
    fi
}

# Check PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")

    if [ "$FORCE" = true ]; then
        stop_process "$PID" "Coordinator" "-9"
    else
        stop_process "$PID" "Coordinator" "-15"
    fi

    rm -f "$PID_FILE"
else
    # Try to find process by name
    PIDS=$(pgrep -f "python -m src.main" 2>/dev/null || true)

    if [ -z "$PIDS" ]; then
        echo "Coordinator is not running"
        exit 0
    fi

    echo "Found coordinator processes: $PIDS"

    for pid in $PIDS; do
        if [ "$FORCE" = true ]; then
            stop_process "$pid" "Coordinator ($pid)" "-9"
        else
            stop_process "$pid" "Coordinator ($pid)" "-15"
        fi
    done
fi

# Also stop any MCP server processes (all patterns)
echo "Stopping MCP server processes..."
for pattern in "src.mcp.server.main" "src.mcp_server" "mcp_server_oci.server"; do
    MCP_PIDS=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$MCP_PIDS" ]; then
        for pid in $MCP_PIDS; do
            stop_process "$pid" "MCP Server ($pid)" "-15"
        done
    fi
done

echo "All processes stopped"
