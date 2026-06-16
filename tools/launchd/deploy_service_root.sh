#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SERVICE_ROOT="${AI_AUTO_TRADING_SERVICE_ROOT:-$HOME/ai-auto-trading-service}"

mkdir -p \
  "$SERVICE_ROOT/src" \
  "$SERVICE_ROOT/config" \
  "$SERVICE_ROOT/policy" \
  "$SERVICE_ROOT/docs" \
  "$SERVICE_ROOT/tests" \
  "$SERVICE_ROOT/tools/launchd" \
  "$SERVICE_ROOT/data/runtime/launchd" \
  "$SERVICE_ROOT/data/runtime/execution" \
  "$SERVICE_ROOT/data/runtime/logs" \
  "$SERVICE_ROOT/data/runtime/mark_price"

rsync -a --delete "$REPO_ROOT/src/" "$SERVICE_ROOT/src/"
rsync -a --delete "$REPO_ROOT/config/" "$SERVICE_ROOT/config/"
rsync -a --delete "$REPO_ROOT/policy/" "$SERVICE_ROOT/policy/"
rsync -a --delete "$REPO_ROOT/docs/" "$SERVICE_ROOT/docs/"
rsync -a --delete "$REPO_ROOT/tests/" "$SERVICE_ROOT/tests/"
rsync -a --delete "$REPO_ROOT/tools/launchd/" "$SERVICE_ROOT/tools/launchd/"

cp "$REPO_ROOT/.env" "$SERVICE_ROOT/.env"
cp "$REPO_ROOT/.env.example" "$SERVICE_ROOT/.env.example"

if [ -f "$REPO_ROOT/data/runtime/execution/testnet_execution.sqlite3" ] && [ ! -f "$SERVICE_ROOT/data/runtime/execution/testnet_execution.sqlite3" ]; then
  cp "$REPO_ROOT/data/runtime/execution/testnet_execution.sqlite3" \
    "$SERVICE_ROOT/data/runtime/execution/testnet_execution.sqlite3"
fi

chmod +x \
  "$SERVICE_ROOT/tools/launchd/run_testnet_auto_cycle.sh" \
  "$SERVICE_ROOT/tools/launchd/run_mainnet_auto_cycle.sh" \
  "$SERVICE_ROOT/tools/launchd/run_mainnet_dashboard.sh" \
  "$SERVICE_ROOT/tools/launchd/run_testnet_dashboard.sh" \
  "$SERVICE_ROOT/tools/launchd/run_local_ai_server.sh" \
  "$SERVICE_ROOT/tools/launchd/deploy_service_root.sh"

printf '%s\n' "$SERVICE_ROOT"
