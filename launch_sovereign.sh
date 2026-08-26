#!/bin/bash
# Sovereign Control System Launcher
# © 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

echo
echo "SOVEREIGN CONTROL SYSTEM"
echo "© 2025 Negative Space Imaging, Inc."
echo
echo "INITIALIZING..."
echo

python sovereign_cli.py initialize --mode SOVEREIGN

echo
echo "EXECUTING SOVEREIGN OPERATION..."
echo

python sovereign_cli.py sovereign

echo
echo "SYSTEM STATUS:"
echo

python sovereign_cli.py status

echo
echo "OPERATION COMPLETE"
echo

read -p "Press Enter to continue..."
