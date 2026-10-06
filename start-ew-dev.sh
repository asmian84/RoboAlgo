#!/usr/bin/env bash
# start-ew-dev.sh — start both dev servers with EW v2 engine
# Backend runs from the ew-engine-v2 worktree; frontend from the main repo.
set -e

WORKTREE="/Users/asmian/RoboAlgo/.worktrees/ew-engine-v2"
FRONTEND="/Users/asmian/RoboAlgo/frontend"
VENV="/Users/asmian/RoboAlgo/venv/bin"

echo "Stopping existing servers..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :5174 | xargs kill -9 2>/dev/null || true
sleep 1

echo "Starting backend (EW v2) on :8000 from $WORKTREE..."
(cd "$WORKTREE" && "$VENV/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --reload >> /tmp/uvicorn-ew-v2.log 2>&1) &
BACKEND_PID=$!

echo "Starting frontend on :5174..."
(cd "$FRONTEND" && npm exec vite -- --port 5174 >> /tmp/vite.log 2>&1) &
FRONTEND_PID=$!

sleep 4

# Verify
BACKEND_CWD=$(lsof -p $(lsof -ti :8000 | tail -1) -a -d cwd 2>/dev/null | awk 'NR>1{print $NF}')
echo ""
echo "✓ Backend  http://localhost:8000  (CWD: $BACKEND_CWD)"
echo "✓ Frontend http://localhost:5174"
echo ""
echo "Logs: /tmp/uvicorn-ew-v2.log  /tmp/vite.log"
