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

exec /usr/bin/caffeinate -i -m "$PYTHON_BIN" -u -m ai_auto_trading mainnet-priority-auto-cycle-loop \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --entry-margin-fraction 0.98 \
  --interval-seconds 5.0 \
  --idle-reconcile-seconds 60.0 \
  --confirm-mainnet-live
