#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONPATH="$PWD/backend/src${PYTHONPATH:+:$PYTHONPATH}"
python -m uvicorn consearch.api:app --host 127.0.0.1 --port 8001 &
API_PID=$!
cleanup() {
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
pnpm exec vite --host 0.0.0.0 --port "${PORT:-3000}" --strictPort
