#!/bin/bash
# Launch with Monitoring Enabled
# Negative Space Imaging Project
# Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Negative Space Imaging - Launch with Monitoring"
echo "=============================================="

# Configuration
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
APP_PORT="${APP_PORT:-8080}"

# Activate virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# Start monitoring stack if Docker is available
if command -v docker-compose &> /dev/null; then
    echo "Starting monitoring stack..."
    docker-compose -f docker-compose.performance.yml up -d prometheus grafana 2>/dev/null || true
fi

# Set monitoring environment variables
export ENABLE_METRICS=true
export METRICS_PORT=9100
export PROMETHEUS_MULTIPROC_DIR="/tmp/prometheus_multiproc"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Start the application with monitoring
echo ""
echo "Monitoring Configuration:"
echo "  Prometheus: http://localhost:$PROMETHEUS_PORT"
echo "  Grafana: http://localhost:$GRAFANA_PORT"
echo "  Metrics: http://localhost:$METRICS_PORT/metrics"
echo ""

# Launch main application
if [[ -f "cli.py" ]]; then
    python cli.py "$@"
elif [[ -f "api/server.py" ]]; then
    python api/server.py "$@"
else
    echo "Starting demo mode..."
    python pipeline_demo.py --mode full "$@"
fi
