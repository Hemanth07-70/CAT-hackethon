#!/bin/bash
# Run the CAT Operator Guardian backend
# Usage: bash run.sh

PYTHON=/opt/anaconda3/envs/cat_guardian/bin/python
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== CAT Operator Guardian ==="
echo "Starting backend..."
echo "Swagger UI → http://localhost:8000/docs"
echo "WebSocket  → ws://localhost:8000/ws/alerts"
echo ""

cd "$PROJECT_DIR/backend"
$PYTHON -m uvicorn main:app --reload --port 8000 --host 0.0.0.0
