#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_LOCAL_PYTHON="$HOME/llm/gemma4-e2b/.venv/bin/python"

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/data/runtime/launchd"

set -a
source "$REPO_ROOT/.env"
set +a

PYTHON_BIN="${LOCAL_MODEL_PYTHON:-$DEFAULT_LOCAL_PYTHON}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m ai_auto_trading.ai.local_server
