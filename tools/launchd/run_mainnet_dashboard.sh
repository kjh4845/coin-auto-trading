#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/opt/local/bin/python3"

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/data/runtime/launchd"

set -a
source "$REPO_ROOT/.env"
set +a

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m ai_auto_trading mainnet-dashboard \
  --host 127.0.0.1 \
  --port 8765 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT
