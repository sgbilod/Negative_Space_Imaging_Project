#!/bin/bash
# Setup Database Integration Environment for Negative Space Imaging Project
# Linux/macOS Shell Script

echo "==================================="
echo "Database Integration Setup Utility"
echo "==================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH."
    echo "Please install Python 3.7 or higher."
    exit 1
fi

echo "Python is installed, proceeding with setup..."
echo

# Default values
CONFIG="deployment/config/database.yaml"

# Parse command-line arguments
ALL=""
INSTALL_DEPS=""
CREATE_DIRS=""
INIT_DB=""
TEST=""
FORCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL="--all"
            shift
            ;;
        --install-deps)
            INSTALL_DEPS="--install-deps"
            shift
            ;;
        --create-dirs)
            CREATE_DIRS="--create-dirs"
            shift
            ;;
        --init-db)
            INIT_DB="--init-db"
            shift
            ;;
        --test)
            TEST="--test"
            shift
            ;;
        --force)
            FORCE="--force"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python3 deployment/setup_database.py"

if [ -n "$ALL" ]; then
    CMD="$CMD $ALL"
fi
if [ -n "$INSTALL_DEPS" ]; then
    CMD="$CMD $INSTALL_DEPS"
fi
if [ -n "$CREATE_DIRS" ]; then
    CMD="$CMD $CREATE_DIRS"
fi
if [ -n "$INIT_DB" ]; then
    CMD="$CMD $INIT_DB"
fi
if [ -n "$TEST" ]; then
    CMD="$CMD $TEST"
fi
if [ -n "$FORCE" ]; then
    CMD="$CMD $FORCE"
fi
CMD="$CMD --config $CONFIG"

echo "Running setup with command:"
echo "$CMD"
echo

# Execute the setup script
eval $CMD

if [ $? -ne 0 ]; then
    echo
    echo "Setup failed. Please check the logs for details."
    exit 1
else
    echo
    echo "Setup completed successfully."
fi

echo
echo "To manage the database, use the following commands:"
echo
echo "python3 deployment/database_deploy.py --deploy    : Deploy the database"
echo "python3 deployment/database_deploy.py --verify    : Verify the database"
echo "python3 deployment/database_deploy.py --migrate   : Run migrations"
echo "python3 deployment/database_deploy.py --backup    : Backup the database"
echo "python3 deployment/database_deploy.py --restore   : Restore from backup"
echo
echo "python3 deployment/test_database_deployment.py    : Test the database"
echo
