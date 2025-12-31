#!/bin/bash
#
# OCI AI Agent Coordinator - Restart Script
#
# Usage:
#   ./scripts/restart.sh              # Restart in current mode
#   ./scripts/restart.sh slack        # Restart in Slack mode
#   ./scripts/restart.sh --cache      # Restart with cache enabled
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Restarting OCI AI Agent Coordinator..."
echo ""

# Stop first
"$SCRIPT_DIR/stop.sh"

echo ""
sleep 2

# Start with same arguments
"$SCRIPT_DIR/start.sh" "$@"
