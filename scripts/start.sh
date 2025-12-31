#!/bin/bash
#
# OCI AI Agent Coordinator - Start Script
#
# Usage:
#   ./scripts/start.sh              # Start in Slack mode (default)
#   ./scripts/start.sh slack        # Start in Slack mode
#   ./scripts/start.sh api          # Start in API mode
#   ./scripts/start.sh both         # Start both Slack and API
#   ./scripts/start.sh --cache      # Start with ShowOCI cache enabled
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.coordinator.pid"
LOG_FILE="$PROJECT_DIR/logs/coordinator.log"

# Default values
MODE="slack"
PORT=3001
ENABLE_CACHE=false
FOREGROUND=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        slack|api|both)
            MODE="$1"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --cache)
            ENABLE_CACHE=true
            shift
            ;;
        --foreground|-f)
            FOREGROUND=true
            shift
            ;;
        --help|-h)
            echo "OCI AI Agent Coordinator - Start Script"
            echo ""
            echo "Usage: $0 [MODE] [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  slack       Start in Slack integration mode (default)"
            echo "  api         Start in API server mode"
            echo "  both        Start both Slack and API"
            echo ""
            echo "Options:"
            echo "  --port N    API server port (default: 3001)"
            echo "  --cache     Enable ShowOCI resource cache"
            echo "  --foreground, -f  Run in foreground (don't daemonize)"
            echo "  --help, -h  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Coordinator is already running (PID: $OLD_PID)"
        echo "Use ./scripts/stop.sh to stop it first"
        exit 1
    else
        # Stale PID file
        rm -f "$PID_FILE"
    fi
fi

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Change to project directory
cd "$PROJECT_DIR"

# Load environment
if [ -f ".env.local" ]; then
    echo "Loading environment from .env.local"
    set -a
    source .env.local
    set +a
fi

# Enable ShowOCI cache if requested
if [ "$ENABLE_CACHE" = true ]; then
    export SHOWOCI_CACHE_ENABLED=true
    echo "ShowOCI cache enabled"
fi

echo "Starting OCI AI Agent Coordinator..."
echo "  Mode: $MODE"
echo "  Port: $PORT (if API mode)"
echo "  Cache: $ENABLE_CACHE"
echo "  Log: $LOG_FILE"

if [ "$FOREGROUND" = true ]; then
    # Run in foreground
    echo "Running in foreground (Ctrl+C to stop)..."
    poetry run python -m src.main --mode "$MODE" --port "$PORT"
else
    # Run in background
    nohup poetry run python -m src.main --mode "$MODE" --port "$PORT" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"

    # Wait a moment and check if it started
    sleep 3

    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Coordinator started successfully (PID: $PID)"
        echo ""
        echo "View logs: tail -f $LOG_FILE"
        echo "Stop:      ./scripts/stop.sh"
        echo "Status:    ./scripts/status.sh"
    else
        echo "Failed to start coordinator. Check logs:"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
fi
