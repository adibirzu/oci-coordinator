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

# Load environment from .env.local
# Search multiple locations to support worktrees
load_env_file() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        echo "Loading environment from: $env_file"
        set -a
        source "$env_file"
        set +a
        # Export for child processes (including MCP servers)
        export OCI_COORDINATOR_ENV_FILE="$env_file"
        return 0
    fi
    return 1
}

ENV_LOADED=false

# Try in order of preference:
# 1. Current project directory
# 2. Main project directory (for worktrees)
# 3. Home-relative path
# 4. Explicit environment variable

if [ -n "$OCI_COORDINATOR_ENV_FILE" ] && [ -f "$OCI_COORDINATOR_ENV_FILE" ]; then
    load_env_file "$OCI_COORDINATOR_ENV_FILE" && ENV_LOADED=true
elif [ -f "$PROJECT_DIR/.env.local" ]; then
    load_env_file "$PROJECT_DIR/.env.local" && ENV_LOADED=true
elif [ -f "$HOME/dev/oci-coordinator/.env.local" ]; then
    load_env_file "$HOME/dev/oci-coordinator/.env.local" && ENV_LOADED=true
fi

if [ "$ENV_LOADED" = false ]; then
    echo "Warning: No .env.local found. Using system environment variables only."
    echo "Searched:"
    echo "  - $PROJECT_DIR/.env.local"
    echo "  - $HOME/dev/oci-coordinator/.env.local"
fi

# Enable ShowOCI cache if requested
if [ "$ENABLE_CACHE" = true ]; then
    export SHOWOCI_CACHE_ENABLED=true
    echo "ShowOCI cache enabled"
fi

# Determine Python executable - prefer venv, fallback to poetry
PYTHON_CMD=""
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON_CMD="$PROJECT_DIR/.venv/bin/python"
elif command -v poetry &> /dev/null; then
    PYTHON_CMD="poetry run python"
else
    echo "Error: Neither .venv/bin/python nor poetry found"
    exit 1
fi

echo "Starting OCI AI Agent Coordinator..."
echo "  Mode: $MODE"
echo "  Port: $PORT (if API mode)"
echo "  Cache: $ENABLE_CACHE"
echo "  Log: $LOG_FILE"

if [ "$FOREGROUND" = true ]; then
    # Run in foreground
    echo "Running in foreground (Ctrl+C to stop)..."
    if [ "$MODE" = "both" ]; then
        # Use start_both.py for combined mode (works around asyncio interaction)
        # Must use 'poetry run python' for proper environment setup
        poetry run python "$SCRIPT_DIR/start_both.py"
    else
        $PYTHON_CMD -m src.main --mode "$MODE" --port "$PORT"
    fi
else
    # Run in background
    if [ "$MODE" = "both" ]; then
        # Use start_both.py for combined mode (works around asyncio interaction)
        # Must use 'poetry run python' for proper environment setup
        nohup poetry run python "$SCRIPT_DIR/start_both.py" > "$LOG_FILE" 2>&1 &
    else
        nohup $PYTHON_CMD -m src.main --mode "$MODE" --port "$PORT" > "$LOG_FILE" 2>&1 &
    fi
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
