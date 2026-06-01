#!/bin/bash
# Launch Spark Interface
# Negative Space Imaging Project
# Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Negative Space Imaging - Spark Interface"
echo "=============================================="

# Check for Spark
if ! command -v spark-submit &> /dev/null; then
    echo "Warning: Apache Spark not found"
    echo "Install Spark or set SPARK_HOME environment variable"
    exit 1
fi

# Set Spark configuration
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-4g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-4g}"
export SPARK_EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-4}"

# Activate virtual environment if exists
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f ".venv-hpc/bin/activate" ]]; then
    source .venv-hpc/bin/activate
fi

# Launch Spark interface
echo "Starting Spark interface..."
echo "Driver Memory: $SPARK_DRIVER_MEMORY"
echo "Executor Memory: $SPARK_EXECUTOR_MEMORY"

if [[ -f "spark_interface/main.py" ]]; then
    spark-submit \
        --driver-memory "$SPARK_DRIVER_MEMORY" \
        --executor-memory "$SPARK_EXECUTOR_MEMORY" \
        --executor-cores "$SPARK_EXECUTOR_CORES" \
        spark_interface/main.py "$@"
else
    echo "Spark interface module not found"
    echo "Run: python -c 'from spark_interface import start; start()'"
fi
