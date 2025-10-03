#!/bin/bash
set -e

# This script will be executed when the PostgreSQL container starts
# It copies all SQL files from the /init-scripts directory into the Docker entrypoint directory
# PostgreSQL will automatically execute scripts in that directory in alphabetical order

echo "Initializing Negative Space Imaging Project database..."

# Copy initialization scripts to PostgreSQL entrypoint directory
cp /init-scripts/*.sql /docker-entrypoint-initdb.d/

echo "Database initialization scripts copied successfully."
