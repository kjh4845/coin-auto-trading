from __future__ import annotations

import asyncio
import argparse
from dataclasses import asdict, is_dataclass, replace
from decimal import Decimal
import json
from pathlib import Path
import sys
import time

from .ai.inference import LocalInferenceTradeAssistant
from .backtest.profile_runner import run_profiled_backtest
from .backtest.replay import (
    BacktestConfig,
    load_funding_rate_parquet,
    load_kline_parquet,
    resample_klines,
    run_hybrid_backtest,
    write_decision_report_json,
    write_trade_logs_jsonl,
)
from .data.historical import BinanceHistoricalDownloader
from .data.mark_price_recorder import MarkPriceRecorder
from .execution.testnet import (
    BinanceFuturesMainnetClient,
    BinanceFuturesTestnetClient,
    TestnetExecutionEngine,
)
from .features.snapshot import FEATURE_SCHEMA_VERSION, FeatureSnapshot, TimeframeFeatureSnapshot
from .models import OrderIntent
from .runtime.dashboard import serve_dashboard
from .runtime.orchestrator import MonitorResult, TestnetRuntimeOrchestrator
from .runtime.testnet import TestnetExecutionRuntime
from .settings import load_settings, required_layout
from .strategy.runtime_profiles import (
    build_runtime_entry_profiles,
    build_runtime_entry_profiles_for_symbol,
    required_runtime_timeframes,
    runtime_strategy_context,
    runtime_stream_interval,
)
from .strategy.rule_based import RuleStrategyParameters, SignalDecision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_auto_trading",
        description="BTCUSDT futures auto trading bot scaffold CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "show-config",
        help="Print the current resolved configuration in JSON.",
    )
    subparsers.add_parser(
        "local-ai-check",
        help="Ping the configured local AI endpoint and run one sample entry-gate review.",
    )
    local_ai_bench = subparsers.add_parser(
        "local-ai-bench",
        help="Measure local AI latency using the sample entry-gate review prompt.",
    )
    local_ai_bench.add_argument("--iterations", type=int, default=3)
    local_ai_bench.add_argument("--warmup", type=int, default=1)
    subparsers.add_parser(
        "check-layout",
        help="Validate that the required Phase 0 directories exist.",
    )

    fetch = subparsers.add_parser(
        "fetch-historical",
        help="Fetch a historical Binance futures dataset and store it as parquet.",
    )
    fetch.add_argument(
        "--dataset",
        required=True,
        choices=[
            "contract_klines",
            "mark_price_klines",
            "funding_rate",
            "open_interest_hist",
        ],
    )
    fetch.add_argument("--symbol", default="BTCUSDT")
    fetch.add_argument("--interval")
    fetch.add_argument("--limit", type=int, default=500)
    fetch.add_argument("--start-time", type=int)
    fetch.add_argument("--end-time", type=int)
    fetch.add_argument("--output")

    backfill = subparsers.add_parser(
        "backfill-historical",
        help="Backfill a Binance futures dataset over a time range and store it as parquet.",
    )
    backfill.add_argument(
        "--dataset",
        required=True,
        choices=[
            "contract_klines",
            "mark_price_klines",
            "funding_rate",
            "open_interest_hist",
        ],
    )
    backfill.add_argument("--symbol", default="BTCUSDT")
    backfill.add_argument("--interval")
    backfill.add_argument("--limit", type=int, default=500)
    backfill.add_argument("--start-time", type=int, required=True)
    backfill.add_argument("--end-time", type=int, required=True)
    backfill.add_argument("--output")

    backtest = subparsers.add_parser(
        "backtest-run",
        help="Run the hybrid backtest from parquet candle data and export reports.",
    )
    backtest.add_argument("--symbol", default="BTCUSDT")
    backtest.add_argument("--contract-parquet", required=True)
    backtest.add_argument("--mark-parquet", required=True)
    backtest.add_argument("--funding-parquet")
    backtest.add_argument(
        "--strategy-mode",
        choices=["single_profile", "best_pair_v1"],
        help="Override the configured strategy mode for this backtest run.",
    )
    backtest.add_argument("--contract-base-interval", default="1m")
    backtest.add_argument("--mark-interval", default="1m")
    backtest.add_argument("--execution-timeframe")
    backtest.add_argument("--micro-timeframe")
    backtest.add_argument("--confirmation-timeframe")
    backtest.add_argument("--macro-timeframe")
    backtest.add_argument("--leverage", type=int)
    backtest.add_argument("--entry-notional-usdt", type=float)
    backtest.add_argument("--fee-bps", type=float, default=4.0)
    backtest.add_argument("--slippage-bps", type=float, default=2.0)
    backtest.add_argument("--max-long-funding-rate", type=float)
    backtest.add_argument("--min-short-funding-rate", type=float)
    backtest.add_argument("--min-long-taker-buy-ratio", type=float)
    backtest.add_argument("--max-short-taker-buy-ratio", type=float)
    backtest.add_argument("--min-micro-long-ema-spread-pct", type=float)
    backtest.add_argument("--min-micro-short-ema-spread-pct", type=float)
    backtest.add_argument("--early-scratch-min-bars", type=int, default=0)
    backtest.add_argument("--early-scratch-min-mfe-r", type=float, default=0.0)
    backtest.add_argument("--early-scratch-max-adverse-r", type=float, default=0.0)
    backtest.add_argument("--min-trades-for-decision", type=int, default=1)
    backtest.add_argument("--min-profit-factor", type=float, default=1.0)
    backtest.add_argument("--max-allowed-drawdown-usdt", type=float, default=100.0)
    backtest.add_argument("--start-time", type=int)
    backtest.add_argument("--end-time", type=int)
    backtest.add_argument("--ai-mode", choices=["disabled", "local"], default="disabled")
    backtest.add_argument("--output-dir")

    recorder = subparsers.add_parser(
        "record-mark-price",
        help="Record live mark price websocket events into SQLite.",
    )
    recorder.add_argument("--symbol", default="BTCUSDT")
    recorder.add_argument("--stream-speed", choices=["1s", "3s"], default="1s")
    recorder.add_argument("--max-messages", type=int)
    recorder.add_argument("--duration-seconds", type=float)
    recorder.add_argument("--output")

    testnet_check = subparsers.add_parser(
        "testnet-check",
        help="Check signed Binance Futures testnet connectivity and current exchange state.",
    )
    testnet_check.add_argument("--symbol", default="BTCUSDT")

    ensure_config = subparsers.add_parser(
        "testnet-ensure-config",
        help="Enforce one-way mode, isolated margin, and expected leverage on testnet.",
    )
    ensure_config.add_argument("--symbol", default="BTCUSDT")
    ensure_config.add_argument("--leverage", type=int)
    ensure_config.add_argument(
        "--margin-mode",
        choices=["ISOLATED", "CROSSED"],
        default="ISOLATED",
    )
    ensure_config.add_argument(
        "--position-mode",
        choices=["ONE_WAY", "HEDGE"],
        default="ONE_WAY",
    )

    order_test = subparsers.add_parser(
        "testnet-order-test",
        help="Normalize and preflight an entry plus protective stop bundle on testnet.",
    )
    order_test.add_argument("--symbol", default="BTCUSDT")
    order_test.add_argument("--side", required=True, choices=["BUY", "SELL"])
    order_test.add_argument("--quantity", required=True, type=float)
    order_test.add_argument("--stop-price", required=True, type=float)
    order_test.add_argument("--leverage", type=int)
    order_test.add_argument(
        "--margin-mode",
        choices=["ISOLATED", "CROSSED"],
        default="ISOLATED",
    )
    order_test.add_argument(
        "--position-mode",
        choices=["ONE_WAY", "HEDGE"],
        default="ONE_WAY",
    )
    order_test.add_argument(
        "--skip-account-config",
        action="store_true",
        help="Only run exchange rule normalization and /order/test preflight.",
    )

    place_bundle = subparsers.add_parser(
        "testnet-place-bundle",
        help="Place a testnet entry order and protective stop after preflight.",
    )
    place_bundle.add_argument("--symbol", default="BTCUSDT")
    place_bundle.add_argument("--side", required=True, choices=["BUY", "SELL"])
    place_bundle.add_argument("--quantity", required=True, type=float)
    place_bundle.add_argument("--stop-price", required=True, type=float)
    place_bundle.add_argument("--leverage", type=int)
    place_bundle.add_argument(
        "--margin-mode",
        choices=["ISOLATED", "CROSSED"],
        default="ISOLATED",
    )
    place_bundle.add_argument(
        "--position-mode",
        choices=["ONE_WAY", "HEDGE"],
        default="ONE_WAY",
    )
    place_bundle.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip /order/test preflight after normalization.",
    )
    place_bundle.add_argument(
        "--confirm-testnet",
        action="store_true",
        help="Required flag before placing testnet orders.",
    )

    reconcile = subparsers.add_parser(
        "testnet-reconcile",
        help="Compare expected local testnet state against the exchange state.",
    )
    reconcile.add_argument("--symbol", default="BTCUSDT")
    reconcile.add_argument("--expected-position-qty", type=float)
    reconcile.add_argument("--expected-stop-price", type=float)
    reconcile.add_argument("--expected-leverage", type=int)
    reconcile.add_argument(
        "--expected-margin-mode",
        choices=["ISOLATED", "CROSSED"],
    )
    reconcile.add_argument(
        "--expected-position-mode",
        choices=["ONE_WAY", "HEDGE"],
    )

    mainnet_check = subparsers.add_parser(
        "mainnet-check",
        help="Check signed Binance Futures mainnet connectivity and current exchange state.",
    )
    mainnet_check.add_argument("--symbol", default="BTCUSDT")

    mainnet_ensure_config = subparsers.add_parser(
        "mainnet-ensure-config",
        help="Enforce one-way mode, isolated margin, and expected leverage on mainnet.",
    )
    mainnet_ensure_config.add_argument("--symbol", default="BTCUSDT")
    mainnet_ensure_config.add_argument("--leverage", type=int)
    mainnet_ensure_config.add_argument(
        "--margin-mode",
        choices=["ISOLATED", "CROSSED"],
        default="ISOLATED",
    )
    mainnet_ensure_config.add_argument(
        "--position-mode",
        choices=["ONE_WAY", "HEDGE"],
        default="ONE_WAY",
    )
    mainnet_ensure_config.add_argument(
        "--confirm-mainnet-config",
        action="store_true",
        help="Required flag because this changes live account symbol settings.",
    )

    mainnet_order_test = subparsers.add_parser(
        "mainnet-order-test",
        help="Normalize and /order/test a live-market entry intent without placing an order.",
    )
    mainnet_order_test.add_argument("--symbol", default="BTCUSDT")
    mainnet_order_test.add_argument("--side", required=True, choices=["BUY", "SELL"])
    mainnet_order_test.add_argument("--quantity", required=True, type=float)
    mainnet_order_test.add_argument("--stop-price", required=True, type=float)
    mainnet_order_test.add_argument("--leverage", type=int)
    mainnet_order_test.add_argument(
        "--margin-mode",
        choices=["ISOLATED", "CROSSED"],
        default="ISOLATED",
    )
    mainnet_order_test.add_argument(
        "--position-mode",
        choices=["ONE_WAY", "HEDGE"],
        default="ONE_WAY",
    )
    mainnet_order_test.add_argument(
        "--ensure-account-config",
        action="store_true",
        help="Also enforce live account configuration before preflight.",
    )

    mainnet_place_bundle = subparsers.add_parser(
        "mainnet-place-bundle",
        help="Place a live mainnet entry order and protective stop after preflight.",
    )
    mainnet_place_bundle.add_argument("--symbol", default="BTCUSDT")
    mainnet_place_bundle.add_argument("--side", required=True, choices=["BUY", "SELL"])
    mainnet_place_bundle.add_argument("--quantity", required=True, type=float)
    mainnet_place_bundle.add_argument("--stop-price", required=True, type=float)
    mainnet_place_bundle.add_argument("--leverage", type=int)
    mainnet_place_bundle.add_argument(
        "--margin-mode",
        choices=["ISOLATED", "CROSSED"],
        default="ISOLATED",
    )
    mainnet_place_bundle.add_argument(
        "--position-mode",
        choices=["ONE_WAY", "HEDGE"],
        default="ONE_WAY",
    )
    mainnet_place_bundle.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip /order/test preflight after normalization.",
    )
    mainnet_place_bundle.add_argument(
        "--confirm-mainnet-live",
        action="store_true",
        help="Required flag before placing live mainnet orders.",
    )

    mainnet_reconcile = subparsers.add_parser(
        "mainnet-reconcile",
        help="Compare expected local mainnet state against the exchange state.",
    )
    mainnet_reconcile.add_argument("--symbol", default="BTCUSDT")
    mainnet_reconcile.add_argument("--expected-position-qty", type=float)
    mainnet_reconcile.add_argument("--expected-stop-price", type=float)
    mainnet_reconcile.add_argument("--expected-leverage", type=int)
    mainnet_reconcile.add_argument(
        "--expected-margin-mode",
        choices=["ISOLATED", "CROSSED"],
    )
    mainnet_reconcile.add_argument(
        "--expected-position-mode",
        choices=["ONE_WAY", "HEDGE"],
    )

    mainnet_auto_cycle_once = subparsers.add_parser(
        "mainnet-auto-cycle-once",
        help="Run one automated mainnet cycle. Requires explicit live confirmation.",
    )
    mainnet_auto_cycle_once.add_argument("--symbol", default="BTCUSDT")
    mainnet_auto_cycle_once.add_argument(
        "--entry-notional-usdt",
        type=float,
        help="Disabled on mainnet; sizing uses available USDT * entry margin fraction * leverage.",
    )
    mainnet_auto_cycle_once.add_argument("--entry-margin-fraction", type=float, default=0.98)
    mainnet_auto_cycle_once.add_argument("--candle-limit", type=int, default=120)
    mainnet_auto_cycle_once.add_argument(
        "--confirm-mainnet-live",
        action="store_true",
        help="Required flag because this can place live mainnet orders.",
    )

    mainnet_auto_cycle_loop = subparsers.add_parser(
        "mainnet-auto-cycle-loop",
        help="Run repeated automated mainnet cycles using the rule engine and position manager.",
    )
    mainnet_auto_cycle_loop.add_argument("--symbol", default="BTCUSDT")
    mainnet_auto_cycle_loop.add_argument(
        "--entry-notional-usdt",
        type=float,
        help="Disabled on mainnet; sizing uses available USDT * entry margin fraction * leverage.",
    )
    mainnet_auto_cycle_loop.add_argument("--entry-margin-fraction", type=float, default=0.98)
    mainnet_auto_cycle_loop.add_argument("--candle-limit", type=int, default=120)
    mainnet_auto_cycle_loop.add_argument("--interval-seconds", type=float, default=5.0)
    mainnet_auto_cycle_loop.add_argument("--iterations", type=int)
    mainnet_auto_cycle_loop.add_argument("--duration-seconds", type=float)
    mainnet_auto_cycle_loop.add_argument(
        "--confirm-mainnet-live",
        action="store_true",
        help="Required flag because this can place live mainnet orders.",
    )

    mainnet_priority_cycle_once = subparsers.add_parser(
        "mainnet-priority-auto-cycle-once",
        help="Run one priority-ordered multi-symbol mainnet cycle with one account-level position.",
    )
    mainnet_priority_cycle_once.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    mainnet_priority_cycle_once.add_argument(
        "--entry-notional-usdt",
        type=float,
        help="Disabled on mainnet; sizing uses available USDT * entry margin fraction * leverage.",
    )
    mainnet_priority_cycle_once.add_argument("--entry-margin-fraction", type=float, default=0.98)
    mainnet_priority_cycle_once.add_argument("--candle-limit", type=int, default=120)
    mainnet_priority_cycle_once.add_argument(
        "--confirm-mainnet-live",
        action="store_true",
        help="Required flag because this can place live mainnet orders.",
    )

    mainnet_priority_cycle_loop = subparsers.add_parser(
        "mainnet-priority-auto-cycle-loop",
        help="Run a priority-ordered multi-symbol mainnet loop with one account-level position.",
    )
    mainnet_priority_cycle_loop.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    mainnet_priority_cycle_loop.add_argument(
        "--entry-notional-usdt",
        type=float,
        help="Disabled on mainnet; sizing uses available USDT * entry margin fraction * leverage.",
    )
    mainnet_priority_cycle_loop.add_argument("--entry-margin-fraction", type=float, default=0.98)
    mainnet_priority_cycle_loop.add_argument("--candle-limit", type=int, default=120)
    mainnet_priority_cycle_loop.add_argument("--interval-seconds", type=float, default=5.0)
    mainnet_priority_cycle_loop.add_argument("--idle-reconcile-seconds", type=float, default=60.0)
    mainnet_priority_cycle_loop.add_argument("--iterations", type=int)
    mainnet_priority_cycle_loop.add_argument("--duration-seconds", type=float)
    mainnet_priority_cycle_loop.add_argument(
        "--confirm-mainnet-live",
        action="store_true",
        help="Required flag because this can place live mainnet orders.",
    )

    mainnet_runtime_status = subparsers.add_parser(
        "mainnet-runtime-status",
        help="Print stored mainnet runtime status, expected state, lockouts, and recent incidents.",
    )
    mainnet_runtime_status.add_argument("--symbol", default="BTCUSDT")
    mainnet_runtime_status.add_argument("--incident-limit", type=int, default=10)

    mainnet_manual_pause = subparsers.add_parser(
        "mainnet-manual-pause",
        help="Acquire a manual pause lockout for mainnet without touching exchange state.",
    )
    mainnet_manual_pause.add_argument("--symbol", default="BTCUSDT")

    mainnet_manual_resume = subparsers.add_parser(
        "mainnet-manual-resume",
        help="Clear manual pause and manual review lockouts for mainnet without touching exchange state.",
    )
    mainnet_manual_resume.add_argument("--symbol", default="BTCUSDT")

    runtime_recover = subparsers.add_parser(
        "testnet-runtime-recover",
        help="Recover expected runtime state from SQLite and reconcile it against testnet.",
    )
    runtime_recover.add_argument("--symbol", default="BTCUSDT")

    user_stream_start = subparsers.add_parser(
        "testnet-user-stream-start",
        help="Create a Binance Futures testnet user data stream listen key and store it in runtime state.",
    )
    user_stream_start.add_argument("--symbol", default="BTCUSDT")

    user_stream_keepalive = subparsers.add_parser(
        "testnet-user-stream-keepalive",
        help="Send keepalive for the active testnet user data stream listen key.",
    )
    user_stream_keepalive.add_argument("--symbol", default="BTCUSDT")

    user_stream_close = subparsers.add_parser(
        "testnet-user-stream-close",
        help="Close the active testnet user data stream listen key and clear runtime state.",
    )
    user_stream_close.add_argument("--symbol", default="BTCUSDT")

    user_stream_run = subparsers.add_parser(
        "testnet-user-stream-run",
        help="Consume Binance Futures testnet user data stream events and persist them locally.",
    )
    user_stream_run.add_argument("--symbol", default="BTCUSDT")
    user_stream_run.add_argument("--max-messages", type=int)
    user_stream_run.add_argument("--duration-seconds", type=float)
    user_stream_run.add_argument("--reconnect-delay-seconds", type=float, default=1.0)
    user_stream_run.add_argument(
        "--keep-open",
        action="store_true",
        help="Do not close the listen key when the local consumer exits.",
    )

    monitor_once = subparsers.add_parser(
        "testnet-monitor-once",
        help="Run one reconciliation/risk tick and update runtime status.",
    )
    monitor_once.add_argument("--symbol", default="BTCUSDT")

    monitor_loop = subparsers.add_parser(
        "testnet-monitor-loop",
        help="Run repeated reconciliation/risk ticks for a fixed number of iterations or duration.",
    )
    monitor_loop.add_argument("--symbol", default="BTCUSDT")
    monitor_loop.add_argument("--interval-seconds", type=float, default=5.0)
    monitor_loop.add_argument("--iterations", type=int)
    monitor_loop.add_argument("--duration-seconds", type=float)

    manage_once = subparsers.add_parser(
        "testnet-manage-position-once",
        help="Evaluate ATR/time-stop rules once against the managed testnet position and exit if needed.",
    )
    manage_once.add_argument("--symbol", default="BTCUSDT")
    manage_once.add_argument("--candle-limit", type=int, default=120)

    manage_loop = subparsers.add_parser(
        "testnet-manage-position-loop",
        help="Run repeated ATR/time-stop management checks for the managed testnet position.",
    )
    manage_loop.add_argument("--symbol", default="BTCUSDT")
    manage_loop.add_argument("--candle-limit", type=int, default=120)
    manage_loop.add_argument("--interval-seconds", type=float, default=5.0)
    manage_loop.add_argument("--iterations", type=int)
    manage_loop.add_argument("--duration-seconds", type=float)

    auto_cycle_once = subparsers.add_parser(
        "testnet-auto-cycle-once",
        help="Run one automated testnet cycle: reconcile, then either place a rule-based entry or manage the open position.",
    )
    auto_cycle_once.add_argument("--symbol", default="BTCUSDT")
    auto_cycle_once.add_argument("--entry-notional-usdt", type=float, default=1000.0)
    auto_cycle_once.add_argument("--candle-limit", type=int, default=120)

    auto_cycle_loop = subparsers.add_parser(
        "testnet-auto-cycle-loop",
        help="Run repeated automated testnet cycles using the rule engine and position manager.",
    )
    auto_cycle_loop.add_argument("--symbol", default="BTCUSDT")
    auto_cycle_loop.add_argument("--entry-notional-usdt", type=float, default=1000.0)
    auto_cycle_loop.add_argument("--candle-limit", type=int, default=120)
    auto_cycle_loop.add_argument("--interval-seconds", type=float, default=5.0)
    auto_cycle_loop.add_argument("--iterations", type=int)
    auto_cycle_loop.add_argument("--duration-seconds", type=float)

    runtime_status = subparsers.add_parser(
        "testnet-runtime-status",
        help="Print stored runtime status, expected state, active lockouts, and recent incidents.",
    )
    runtime_status.add_argument("--symbol", default="BTCUSDT")
    runtime_status.add_argument("--incident-limit", type=int, default=10)

    forward_review = subparsers.add_parser(
        "testnet-forward-review",
        help="Report forward closed-trade progress and sizing review readiness.",
    )
    forward_review.add_argument("--symbol", default="BTCUSDT")
    forward_review.add_argument("--min-trades", type=int, default=30)
    forward_review.add_argument("--target-trades", type=int, default=50)
    forward_review.add_argument("--entry-notional-usdt", type=float, default=1000.0)
    forward_review.add_argument("--leverage", type=float)
    forward_review.add_argument("--account-equity-usdt", type=float)

    incident_tail = subparsers.add_parser(
        "testnet-incident-tail",
        help="Print recent execution incidents from the SQLite runtime store.",
    )
    incident_tail.add_argument("--limit", type=int, default=20)

    risk_status = subparsers.add_parser(
        "testnet-risk-status",
        help="Print active runtime risk lockouts for the symbol.",
    )
    risk_status.add_argument("--symbol", default="BTCUSDT")

    manual_pause = subparsers.add_parser(
        "testnet-manual-pause",
        help="Acquire a manual pause lockout without touching exchange state.",
    )
    manual_pause.add_argument("--symbol", default="BTCUSDT")

    manual_resume = subparsers.add_parser(
        "testnet-manual-resume",
        help="Clear manual pause and manual review lockouts without touching exchange state.",
    )
    manual_resume.add_argument("--symbol", default="BTCUSDT")

    dashboard = subparsers.add_parser(
        "testnet-dashboard",
        help="Run a local admin dashboard for runtime status and testnet controls.",
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--symbol", default="BTCUSDT")
    dashboard.add_argument("--symbols")

    mainnet_dashboard = subparsers.add_parser(
        "mainnet-dashboard",
        help="Run a local admin dashboard for runtime status and mainnet controls.",
    )
    mainnet_dashboard.add_argument("--host", default="127.0.0.1")
    mainnet_dashboard.add_argument("--port", type=int, default=8765)
    mainnet_dashboard.add_argument("--symbol", default="BTCUSDT")
    mainnet_dashboard.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")

    return parser


def _to_jsonable(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}...{value[-2:]}"


def _settings_payload() -> dict[str, object]:
    settings = load_settings()
    payload: dict[str, object] = {}
    for key, value in settings.__dict__.items():
        if key.endswith("_api_key"):
            payload[key] = _mask_secret(str(value))
            payload[f"{key}_configured"] = bool(value)
            continue
        if key.endswith("_api_secret"):
            payload[key] = "***" if value else ""
            payload[f"{key}_configured"] = bool(value)
            continue
        payload[key] = _to_jsonable(value)
    payload["runtime_strategy_context"] = _to_jsonable(runtime_strategy_context(settings))
    return payload


def cmd_show_config() -> int:
    print(json.dumps(_settings_payload(), indent=2, sort_keys=True))
    return 0


def cmd_check_layout() -> int:
    settings = load_settings()
    missing = [str(path) for path in required_layout(settings) if not path.exists()]
    if missing:
        print(json.dumps({"status": "missing", "paths": missing}, indent=2))
        return 1
    print(json.dumps({"status": "ok"}, indent=2))
    return 0


def cmd_local_ai_check() -> int:
    settings = load_settings()
    assistant = LocalInferenceTradeAssistant.from_settings(settings)
    snapshot, rule_decision = _sample_ai_review_inputs(settings)
    decision = assistant.review_entry(
        snapshot=snapshot,
        rule_decision=rule_decision,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "endpoint": settings.local_model_endpoint,
                "model_id": settings.local_model_id,
                "model_path": settings.local_model_path,
                "decision": decision.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_local_ai_bench(args: argparse.Namespace) -> int:
    settings = load_settings()
    assistant = LocalInferenceTradeAssistant.from_settings(settings)
    snapshot, rule_decision = _sample_ai_review_inputs(settings)
    latencies_ms: list[float] = []
    last_decision = None
    for _ in range(max(0, args.warmup)):
        assistant.review_entry(snapshot=snapshot, rule_decision=rule_decision)
    for _ in range(max(1, args.iterations)):
        started = time.perf_counter()
        last_decision = assistant.review_entry(snapshot=snapshot, rule_decision=rule_decision)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(latencies_ms)
    p50_index = min(len(ordered) - 1, len(ordered) // 2)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    print(
        json.dumps(
            {
                "status": "ok",
                "endpoint": settings.local_model_endpoint,
                "model_id": settings.local_model_id,
                "iterations": len(latencies_ms),
                "warmup": max(0, args.warmup),
                "latency_ms": {
                    "min": round(min(latencies_ms), 2),
                    "max": round(max(latencies_ms), 2),
                    "avg": round(sum(latencies_ms) / len(latencies_ms), 2),
                    "p50": round(ordered[p50_index], 2),
                    "p95": round(ordered[p95_index], 2),
                },
                "last_decision": last_decision.to_dict() if last_decision is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _sample_ai_review_inputs(settings):
    execution = TimeframeFeatureSnapshot(
        timeframe=settings.execution_timeframe,
        last_open_time=1,
        candle_count=30,
        last_close=100000.0,
        current_volume=1600.0,
        taker_buy_ratio=0.56,
        ema_fast_9=99980.0,
        ema_slow_21=99940.0,
        rsi_14=58.0,
        atr_14=120.0,
        cumulative_vwap=99970.0,
        swing_high_20=100120.0,
        swing_low_20=99780.0,
        roc_5=0.15,
        volume_sma_20=1400.0,
        volume_ratio_20=1.1428571428571428,
    )
    snapshot = FeatureSnapshot(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        symbol=settings.trading_symbol,
        generated_from="contract_price",
        timeframes={
            settings.execution_timeframe: execution,
            settings.micro_timeframe: TimeframeFeatureSnapshot(
                timeframe=settings.micro_timeframe,
                last_open_time=1,
                candle_count=30,
                last_close=100010.0,
                current_volume=2300.0,
                taker_buy_ratio=0.58,
                ema_fast_9=100000.0,
                ema_slow_21=99920.0,
                rsi_14=57.0,
                atr_14=160.0,
                cumulative_vwap=99990.0,
                swing_high_20=100180.0,
                swing_low_20=99850.0,
                roc_5=0.18,
                volume_sma_20=2100.0,
                volume_ratio_20=1.0952380952380953,
            ),
            settings.confirmation_timeframe: TimeframeFeatureSnapshot(
                timeframe=settings.confirmation_timeframe,
                last_open_time=1,
                candle_count=30,
                last_close=100020.0,
                current_volume=4200.0,
                taker_buy_ratio=0.55,
                ema_fast_9=99990.0,
                ema_slow_21=99920.0,
                rsi_14=57.0,
                atr_14=220.0,
                cumulative_vwap=99980.0,
                swing_high_20=100220.0,
                swing_low_20=99720.0,
                roc_5=0.12,
                volume_sma_20=3900.0,
                volume_ratio_20=1.0769230769230769,
            ),
            settings.macro_timeframe: TimeframeFeatureSnapshot(
                timeframe=settings.macro_timeframe,
                last_open_time=1,
                candle_count=30,
                last_close=100050.0,
                current_volume=12800.0,
                taker_buy_ratio=0.54,
                ema_fast_9=100010.0,
                ema_slow_21=99910.0,
                rsi_14=60.0,
                atr_14=480.0,
                cumulative_vwap=99960.0,
                swing_high_20=100400.0,
                swing_low_20=99500.0,
                roc_5=0.2,
                volume_sma_20=12000.0,
                volume_ratio_20=1.0666666666666667,
            ),
            settings.regime_timeframe: TimeframeFeatureSnapshot(
                timeframe=settings.regime_timeframe,
                last_open_time=1,
                candle_count=40,
                last_close=100200.0,
                current_volume=22000.0,
                taker_buy_ratio=0.53,
                ema_fast_9=100120.0,
                ema_slow_21=99960.0,
                rsi_14=61.0,
                atr_14=760.0,
                cumulative_vwap=100010.0,
                swing_high_20=100600.0,
                swing_low_20=99350.0,
                roc_5=0.35,
                volume_sma_20=21000.0,
                volume_ratio_20=1.0476190476190477,
            ),
            settings.anchor_timeframe: TimeframeFeatureSnapshot(
                timeframe=settings.anchor_timeframe,
                last_open_time=1,
                candle_count=40,
                last_close=100500.0,
                current_volume=56000.0,
                taker_buy_ratio=0.52,
                ema_fast_9=100250.0,
                ema_slow_21=99880.0,
                rsi_14=63.0,
                atr_14=1400.0,
                cumulative_vwap=99920.0,
                swing_high_20=101200.0,
                swing_low_20=99000.0,
                roc_5=0.55,
                volume_sma_20=53000.0,
                volume_ratio_20=1.0566037735849056,
            ),
        },
    )
    return snapshot, SignalDecision(action="LONG", reason_codes=["long_trend_alignment"])


def cmd_fetch_historical(args: argparse.Namespace) -> int:
    downloader = BinanceHistoricalDownloader()
    result = downloader.fetch_and_store(
        dataset=args.dataset,
        symbol=args.symbol,
        interval_or_period=args.interval,
        limit=args.limit,
        start_time=args.start_time,
        end_time=args.end_time,
        output_path=Path(args.output) if args.output else None,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": result.dataset,
                "interval": result.interval,
                "rows": result.rows,
                "path": str(result.path),
            },
            indent=2,
        )
    )
    return 0


def cmd_backfill_historical(args: argparse.Namespace) -> int:
    downloader = BinanceHistoricalDownloader()
    result = downloader.backfill_and_store(
        dataset=args.dataset,
        symbol=args.symbol,
        interval_or_period=args.interval,
        limit=args.limit,
        start_time=args.start_time,
        end_time=args.end_time,
        output_path=Path(args.output) if args.output else None,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": result.dataset,
                "interval": result.interval,
                "total_rows": result.total_rows,
                "batches": result.batches,
                "path": str(result.path),
            },
            indent=2,
        )
    )
    return 0


def cmd_backtest_run(args: argparse.Namespace) -> int:
    settings = load_settings()
    strategy_mode = args.strategy_mode or settings.strategy_mode
    settings_for_run = replace(settings, strategy_mode=strategy_mode)
    if strategy_mode == "best_pair_v1":
        _validate_profiled_backtest_args(args)
        if not args.funding_parquet:
            raise ValueError("best_pair_v1 backtests require --funding-parquet")
    execution_timeframe = args.execution_timeframe or settings.execution_timeframe
    micro_timeframe = args.micro_timeframe or settings.micro_timeframe
    confirmation_timeframe = args.confirmation_timeframe or settings.confirmation_timeframe
    macro_timeframe = args.macro_timeframe or settings.macro_timeframe
    contract_base = Path(args.contract_parquet)
    mark_path = Path(args.mark_parquet)
    contract_base_candles = load_kline_parquet(
        contract_base,
        symbol=args.symbol,
        interval=args.contract_base_interval,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    lower_mark_candles = load_kline_parquet(
        mark_path,
        symbol=args.symbol,
        interval=args.mark_interval,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    funding_rows = (
        load_funding_rate_parquet(
            Path(args.funding_parquet),
            symbol=args.symbol,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        if args.funding_parquet
        else []
    )
    if not contract_base_candles:
        raise ValueError("no contract candles matched the requested filters")
    if not lower_mark_candles:
        raise ValueError("no mark-price candles matched the requested filters")
    if args.funding_parquet and not funding_rows:
        raise ValueError("no funding rows matched the requested filters")

    if strategy_mode == "best_pair_v1":
        required_timeframes = required_runtime_timeframes(settings_for_run)
    else:
        strategy_params = RuleStrategyParameters(
            execution_timeframe=execution_timeframe,
            micro_timeframe=micro_timeframe,
            confirmation_timeframe=confirmation_timeframe,
            macro_timeframe=macro_timeframe,
            regime_timeframe=settings.regime_timeframe,
            anchor_timeframe=settings.anchor_timeframe,
            min_micro_long_ema_spread_pct=(
                float(args.min_micro_long_ema_spread_pct)
                if args.min_micro_long_ema_spread_pct is not None
                else settings.min_micro_long_ema_spread_pct
            ),
            min_micro_short_ema_spread_pct=(
                float(args.min_micro_short_ema_spread_pct)
                if args.min_micro_short_ema_spread_pct is not None
                else settings.min_micro_short_ema_spread_pct
            ),
            min_higher_tf_ema_spread_pct=settings.min_higher_tf_ema_spread_pct,
            min_volume_ratio_20=settings.min_volume_ratio_20,
            max_long_funding_rate=(
                float(args.max_long_funding_rate)
                if args.max_long_funding_rate is not None
                else settings.max_long_funding_rate
            ),
            min_short_funding_rate=(
                float(args.min_short_funding_rate)
                if args.min_short_funding_rate is not None
                else settings.min_short_funding_rate
            ),
            min_long_taker_buy_ratio=(
                float(args.min_long_taker_buy_ratio)
                if args.min_long_taker_buy_ratio is not None
                else settings.min_long_taker_buy_ratio
            ),
            max_short_taker_buy_ratio=(
                float(args.max_short_taker_buy_ratio)
                if args.max_short_taker_buy_ratio is not None
                else settings.max_short_taker_buy_ratio
            ),
            allow_long_entries=settings.allow_long_entries,
            allow_short_entries=settings.allow_short_entries,
        )
        required_timeframes = strategy_params.required_timeframes()

    contract_candles_by_timeframe = {
        timeframe: resample_klines(
            contract_base_candles,
            target_interval=timeframe,
            source_interval=args.contract_base_interval,
        )
        for timeframe in required_timeframes
    }
    if any(not candles for candles in contract_candles_by_timeframe.values()):
        raise ValueError("resampled contract candles are empty for at least one required timeframe")

    config = BacktestConfig(
        symbol=args.symbol,
        execution_timeframe=(
            runtime_stream_interval(settings_for_run)
            if strategy_mode == "best_pair_v1"
            else execution_timeframe
        ),
        confirmation_timeframe=confirmation_timeframe,
        macro_timeframe=macro_timeframe,
        lower_mark_timeframe=args.mark_interval,
        leverage_at_entry=float(args.leverage or settings.live_start_leverage),
        entry_notional_usdt=float(args.entry_notional_usdt or 1000.0),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        atr_trailing_multiplier=settings.atr_trailing_multiplier,
        atr_trail_activation_profit_r=settings.atr_trail_activation_profit_r,
        atr_trail_min_bars=settings.atr_trail_min_bars,
        exit_policy=settings.exit_policy,
        fixed_take_profit_r=settings.fixed_take_profit_r,
        early_scratch_min_bars=max(0, int(args.early_scratch_min_bars)),
        early_scratch_min_mfe_r=max(0.0, float(args.early_scratch_min_mfe_r)),
        early_scratch_max_adverse_r=max(0.0, float(args.early_scratch_max_adverse_r)),
        max_holding_bars=settings.max_holding_bars,
        model_base="rule_only" if args.ai_mode == "disabled" else settings.local_model_base,
        min_trades_for_decision=int(args.min_trades_for_decision),
        min_profit_factor=float(args.min_profit_factor),
        max_allowed_drawdown_usdt=float(args.max_allowed_drawdown_usdt),
    )
    assistant = None
    ai_gate_config = None
    if args.ai_mode == "local":
        assistant = LocalInferenceTradeAssistant.from_settings(settings)
        from .ai.gate import AIGateConfig

        ai_gate_config = AIGateConfig.from_settings(settings)
    profile_results_payload = None
    combination_payload = None
    if strategy_mode == "best_pair_v1":
        combined = run_profiled_backtest(
            profiles=build_runtime_entry_profiles(settings_for_run),
            contract_candles_by_timeframe=contract_candles_by_timeframe,
            lower_mark_price_candles=lower_mark_candles,
            funding_rate_rows=funding_rows,
            base_config=config,
            ai_trade_assistant=assistant,
            ai_gate_config=ai_gate_config,
        )
        result = combined.combined_result
        profile_results_payload = {
            run.profile_name: {
                "priority": run.priority,
                "metrics": _to_jsonable(run.result.metrics),
                "decision_report": _to_jsonable(run.result.decision_report),
            }
            for run in combined.standalone_runs
        }
        combination_payload = {
            "accepted_by_profile": combined.accepted_by_profile,
            "rejected_by_profile": combined.rejected_by_profile,
            "exact_time_conflicts": combined.exact_time_conflicts,
        }
    else:
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract_candles_by_timeframe,
            lower_mark_price_candles=lower_mark_candles,
            funding_rate_rows=funding_rows,
            config=config,
            strategy_params=strategy_params,
            ai_trade_assistant=assistant,
            ai_gate_config=ai_gate_config,
        )

    output_dir = Path(args.output_dir) if args.output_dir else (
        settings.data_dir
        / "backtests"
        / f"{args.symbol.lower()}-{int(time.time())}"
    )
    trade_logs_path = write_trade_logs_jsonl(result.trade_records, output_dir / "trade_logs.jsonl")
    decision_report_path = write_decision_report_json(result.decision_report, output_dir / "decision_report.json")
    profile_results_path = None
    if profile_results_payload is not None:
        profile_results_path = output_dir / "profile_results.json"
        profile_results_path.write_text(
            json.dumps(profile_results_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    summary_payload = {
        "status": "ok",
        "symbol": args.symbol,
        "ai_mode": args.ai_mode,
        "strategy_mode": strategy_mode,
        "runtime_strategy_context": _to_jsonable(runtime_strategy_context(settings_for_run)),
        "contract_parquet": str(contract_base),
        "mark_parquet": str(mark_path),
        "funding_parquet": str(Path(args.funding_parquet)) if args.funding_parquet else None,
        "resampled_rows": {
            timeframe: len(candles)
            for timeframe, candles in contract_candles_by_timeframe.items()
        },
        "mark_rows": len(lower_mark_candles),
        "funding_rows": len(funding_rows),
        "config": _to_jsonable(config),
        "metrics": _to_jsonable(result.metrics),
        "decision_report": _to_jsonable(result.decision_report),
        "profile_results": profile_results_payload,
        "combination": combination_payload,
        "artifacts": {
            "output_dir": str(output_dir),
            "trade_logs_jsonl": str(trade_logs_path),
            "decision_report_json": str(decision_report_path),
            "profile_results_json": str(profile_results_path) if profile_results_path else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


def _validate_profiled_backtest_args(args: argparse.Namespace) -> None:
    unsupported = {
        "execution_timeframe": args.execution_timeframe,
        "micro_timeframe": args.micro_timeframe,
        "confirmation_timeframe": args.confirmation_timeframe,
        "macro_timeframe": args.macro_timeframe,
        "max_long_funding_rate": args.max_long_funding_rate,
        "min_short_funding_rate": args.min_short_funding_rate,
        "min_long_taker_buy_ratio": args.min_long_taker_buy_ratio,
        "max_short_taker_buy_ratio": args.max_short_taker_buy_ratio,
        "min_micro_long_ema_spread_pct": args.min_micro_long_ema_spread_pct,
        "min_micro_short_ema_spread_pct": args.min_micro_short_ema_spread_pct,
    }
    used = [name for name, value in unsupported.items() if value is not None]
    if used:
        raise ValueError(
            "best_pair_v1 does not accept single-profile overrides: "
            + ", ".join(sorted(used))
            + "; rerun with --strategy-mode single_profile if you want custom profile tuning"
        )


def cmd_record_mark_price(args: argparse.Namespace) -> int:
    recorder = MarkPriceRecorder(
        symbol=args.symbol,
        stream_speed=args.stream_speed,
        database_path=Path(args.output) if args.output else None,
    )
    result = asyncio.run(
        recorder.run(
            max_messages=args.max_messages,
            duration_seconds=args.duration_seconds,
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "symbol": result.symbol,
                "messages_recorded": result.messages_recorded,
                "health_events_recorded": result.health_events_recorded,
                "database_path": str(result.database_path),
            },
            indent=2,
        )
    )
    return 0


def _build_testnet_engine() -> tuple[object, BinanceFuturesTestnetClient, TestnetExecutionEngine]:
    settings = load_settings()
    client = BinanceFuturesTestnetClient(settings=settings)
    if not client.api_key or not client.api_secret:
        raise ValueError(
            "missing testnet credentials; set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET"
        )
    return settings, client, TestnetExecutionEngine(client)


def _build_testnet_runtime() -> tuple[
    object,
    BinanceFuturesTestnetClient,
    TestnetExecutionEngine,
    TestnetExecutionRuntime,
    TestnetRuntimeOrchestrator,
]:
    settings, client, engine = _build_testnet_engine()
    runtime = TestnetExecutionRuntime(engine, settings=settings)
    orchestrator = TestnetRuntimeOrchestrator(engine, runtime)
    return settings, client, engine, runtime, orchestrator


def _mainnet_runtime_database_path(settings) -> Path:
    return settings.data_dir / "runtime" / "execution" / "mainnet_execution.sqlite3"


def _build_mainnet_engine() -> tuple[object, BinanceFuturesMainnetClient, TestnetExecutionEngine]:
    settings = load_settings()
    client = BinanceFuturesMainnetClient(settings=settings)
    if not client.api_key or not client.api_secret:
        raise ValueError(
            "missing mainnet credentials; set BINANCE_API_KEY and BINANCE_API_SECRET"
        )
    return settings, client, TestnetExecutionEngine(client)


def _build_mainnet_runtime() -> tuple[
    object,
    BinanceFuturesMainnetClient,
    TestnetExecutionEngine,
    TestnetExecutionRuntime,
    TestnetRuntimeOrchestrator,
]:
    settings, client, engine = _build_mainnet_engine()
    runtime = TestnetExecutionRuntime(
        engine,
        settings=settings,
        database_path=_mainnet_runtime_database_path(settings),
    )
    orchestrator = TestnetRuntimeOrchestrator(engine, runtime)
    return settings, client, engine, runtime, orchestrator


def _build_runtime_store() -> tuple[object, TestnetExecutionRuntime]:
    settings = load_settings()
    runtime = TestnetExecutionRuntime(None, settings=settings)
    return settings, runtime


def _build_mainnet_runtime_store() -> tuple[object, TestnetExecutionRuntime]:
    settings = load_settings()
    runtime = TestnetExecutionRuntime(
        None,
        settings=settings,
        database_path=_mainnet_runtime_database_path(settings),
    )
    return settings, runtime


def _bundle_intents(args: argparse.Namespace) -> tuple[OrderIntent, OrderIntent]:
    entry_intent = OrderIntent(
        side=args.side,
        quantity=args.quantity,
        order_type="MARKET",
        symbol=args.symbol,
    )
    stop_side = "SELL" if args.side == "BUY" else "BUY"
    hard_stop_intent = OrderIntent(
        side=stop_side,
        quantity=args.quantity,
        order_type="STOP_MARKET",
        symbol=args.symbol,
        reduce_only=True,
        working_type="MARK_PRICE",
        stop_price=args.stop_price,
    )
    return entry_intent, hard_stop_intent


def cmd_testnet_check(args: argparse.Namespace) -> int:
    _, _, engine = _build_testnet_engine()
    timestamp = engine.server_timestamp()
    state = engine.fetch_remote_state(symbol=args.symbol, timestamp=timestamp)
    print(
        json.dumps(
            {
                "status": "ok",
                "server_time_ms": timestamp,
                "symbol": args.symbol,
                "remote_state": _to_jsonable(state),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_ensure_config(args: argparse.Namespace) -> int:
    settings, _, engine = _build_testnet_engine()
    timestamp = engine.server_timestamp()
    result = engine.ensure_account_configuration(
        symbol=args.symbol,
        expected_leverage=args.leverage or settings.live_start_leverage,
        expected_margin_mode=args.margin_mode,
        expected_position_mode=args.position_mode,
        timestamp=timestamp,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "server_time_ms": timestamp,
                "symbol": args.symbol,
                "reconciliation": _to_jsonable(result),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_order_test(args: argparse.Namespace) -> int:
    settings, _, engine = _build_testnet_engine()
    timestamp = engine.server_timestamp()
    if not args.skip_account_config:
        engine.ensure_account_configuration(
            symbol=args.symbol,
            expected_leverage=args.leverage or settings.live_start_leverage,
            expected_margin_mode=args.margin_mode,
            expected_position_mode=args.position_mode,
            timestamp=timestamp,
        )

    entry_intent, hard_stop_intent = _bundle_intents(args)
    prepared = engine.prepare_entry_and_protection(
        entry_intent=entry_intent,
        hard_stop_intent=hard_stop_intent,
        timestamp=timestamp,
        preflight=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "server_time_ms": timestamp,
                "prepared_bundle": _to_jsonable(prepared),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_place_bundle(args: argparse.Namespace) -> int:
    if not args.confirm_testnet:
        raise ValueError("testnet-place-bundle requires --confirm-testnet")

    settings, _, engine, runtime, orchestrator = _build_testnet_runtime()
    timestamp = engine.server_timestamp()
    entry_intent, hard_stop_intent = _bundle_intents(args)
    result = orchestrator.guarded_place_entry_bundle(
        symbol=args.symbol,
        timestamp=timestamp,
        place_fn=lambda: runtime.place_entry_and_protection(
            entry_intent=entry_intent,
            hard_stop_intent=hard_stop_intent,
            timestamp=timestamp,
            expected_leverage=args.leverage or settings.live_start_leverage,
            expected_margin_mode=args.margin_mode,
            expected_position_mode=args.position_mode,
            skip_preflight=args.skip_preflight,
        ),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "server_time_ms": timestamp,
                "bundle_result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_reconcile(args: argparse.Namespace) -> int:
    settings, _, engine = _build_testnet_engine()
    runtime = TestnetExecutionRuntime(engine, settings=settings)
    timestamp = engine.server_timestamp()
    result = engine.reconcile_state(
        symbol=args.symbol,
        expected_position_qty=args.expected_position_qty,
        expected_stop_price_mark=args.expected_stop_price,
        expected_leverage=args.expected_leverage,
        expected_margin_mode=args.expected_margin_mode,
        expected_position_mode=args.expected_position_mode,
        timestamp=timestamp,
    )
    runtime.record_incident(
        level="INFO" if result.ok else "WARNING",
        event_type="manual_reconciliation",
        message="manual testnet reconciliation completed",
        details={
            "symbol": args.symbol,
            "ok": result.ok,
            "mismatches": result.mismatches,
        },
    )
    print(
        json.dumps(
            {
                "status": "ok" if result.ok else "mismatch",
                "server_time_ms": timestamp,
                "reconciliation": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.ok else 1


def cmd_mainnet_check(args: argparse.Namespace) -> int:
    _, _, engine = _build_mainnet_engine()
    timestamp = engine.server_timestamp()
    state = engine.fetch_remote_state(symbol=args.symbol, timestamp=timestamp)
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "server_time_ms": timestamp,
                "symbol": args.symbol,
                "remote_state": _to_jsonable(state),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_mainnet_ensure_config(args: argparse.Namespace) -> int:
    if not args.confirm_mainnet_config:
        raise ValueError("mainnet-ensure-config requires --confirm-mainnet-config")
    settings, _, engine = _build_mainnet_engine()
    timestamp = engine.server_timestamp()
    result = engine.ensure_account_configuration(
        symbol=args.symbol,
        expected_leverage=args.leverage or settings.live_start_leverage,
        expected_margin_mode=args.margin_mode,
        expected_position_mode=args.position_mode,
        timestamp=timestamp,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "server_time_ms": timestamp,
                "symbol": args.symbol,
                "reconciliation": _to_jsonable(result),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_mainnet_order_test(args: argparse.Namespace) -> int:
    settings, _, engine = _build_mainnet_engine()
    timestamp = engine.server_timestamp()
    if args.ensure_account_config:
        engine.ensure_account_configuration(
            symbol=args.symbol,
            expected_leverage=args.leverage or settings.live_start_leverage,
            expected_margin_mode=args.margin_mode,
            expected_position_mode=args.position_mode,
            timestamp=timestamp,
        )

    entry_intent, hard_stop_intent = _bundle_intents(args)
    prepared = engine.prepare_entry_and_protection(
        entry_intent=entry_intent,
        hard_stop_intent=hard_stop_intent,
        timestamp=timestamp,
        preflight=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "account_config_applied": bool(args.ensure_account_config),
                "server_time_ms": timestamp,
                "prepared_bundle": _to_jsonable(prepared),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_mainnet_place_bundle(args: argparse.Namespace) -> int:
    if not args.confirm_mainnet_live:
        raise ValueError("mainnet-place-bundle requires --confirm-mainnet-live")

    settings, _, engine, runtime, orchestrator = _build_mainnet_runtime()
    timestamp = engine.server_timestamp()
    entry_intent, hard_stop_intent = _bundle_intents(args)
    result = orchestrator.guarded_place_entry_bundle(
        symbol=args.symbol,
        timestamp=timestamp,
        place_fn=lambda: runtime.place_entry_and_protection(
            entry_intent=entry_intent,
            hard_stop_intent=hard_stop_intent,
            timestamp=timestamp,
            expected_leverage=args.leverage or settings.live_start_leverage,
            expected_margin_mode=args.margin_mode,
            expected_position_mode=args.position_mode,
            skip_preflight=args.skip_preflight,
        ),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "server_time_ms": timestamp,
                "bundle_result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_mainnet_reconcile(args: argparse.Namespace) -> int:
    settings, _, engine = _build_mainnet_engine()
    runtime = TestnetExecutionRuntime(
        engine,
        settings=settings,
        database_path=_mainnet_runtime_database_path(settings),
    )
    timestamp = engine.server_timestamp()
    result = engine.reconcile_state(
        symbol=args.symbol,
        expected_position_qty=args.expected_position_qty,
        expected_stop_price_mark=args.expected_stop_price,
        expected_leverage=args.expected_leverage,
        expected_margin_mode=args.expected_margin_mode,
        expected_position_mode=args.expected_position_mode,
        timestamp=timestamp,
    )
    runtime.record_incident(
        level="INFO" if result.ok else "WARNING",
        event_type="manual_reconciliation",
        message="manual mainnet reconciliation completed",
        details={
            "symbol": args.symbol,
            "ok": result.ok,
            "mismatches": result.mismatches,
        },
    )
    print(
        json.dumps(
            {
                "status": "ok" if result.ok else "mismatch",
                "environment": "mainnet",
                "server_time_ms": timestamp,
                "reconciliation": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.ok else 1


def cmd_testnet_runtime_recover(args: argparse.Namespace) -> int:
    settings, _, engine = _build_testnet_engine()
    runtime = TestnetExecutionRuntime(engine, settings=settings)
    timestamp = engine.server_timestamp()
    result = runtime.recover_and_reconcile(symbol=args.symbol, timestamp=timestamp)
    print(
        json.dumps(
            {
                "status": "ok" if result.ok else "mismatch",
                "server_time_ms": timestamp,
                "recovery_result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.ok else 1


def cmd_testnet_user_stream_start(args: argparse.Namespace) -> int:
    _, _, engine, runtime, orchestrator = _build_testnet_runtime()
    timestamp = engine.server_timestamp()
    session = orchestrator.start_user_stream(symbol=args.symbol, timestamp=timestamp)
    print(
        json.dumps(
            {
                "status": "ok",
                "server_time_ms": timestamp,
                "session": _to_jsonable(session),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_user_stream_keepalive(args: argparse.Namespace) -> int:
    _, _, engine, runtime, orchestrator = _build_testnet_runtime()
    timestamp = engine.server_timestamp()
    session = orchestrator.keepalive_user_stream(symbol=args.symbol, timestamp=timestamp)
    print(
        json.dumps(
            {
                "status": "ok",
                "server_time_ms": timestamp,
                "session": _to_jsonable(session),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_user_stream_close(args: argparse.Namespace) -> int:
    _, _, _, runtime, orchestrator = _build_testnet_runtime()
    orchestrator.close_user_stream(symbol=args.symbol)
    print(
        json.dumps(
            {
                "status": "ok",
                "symbol": args.symbol,
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_user_stream_run(args: argparse.Namespace) -> int:
    _, _, engine, runtime, orchestrator = _build_testnet_runtime()
    timestamp = engine.server_timestamp()
    result = asyncio.run(
        orchestrator.user_stream_service.run_stream(
            symbol=args.symbol,
            timestamp=timestamp,
            max_messages=args.max_messages,
            duration_seconds=args.duration_seconds,
            reconnect_delay_seconds=args.reconnect_delay_seconds,
            close_on_exit=not args.keep_open,
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "server_time_ms": timestamp,
                "result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_monitor_once(args: argparse.Namespace) -> int:
    _, _, engine, runtime, orchestrator = _build_testnet_runtime()
    timestamp = engine.server_timestamp()
    result = orchestrator.reconcile_once(symbol=args.symbol, timestamp=timestamp)
    print(
        json.dumps(
            {
                "status": "ok" if not result.active_lockouts else "paused",
                "server_time_ms": timestamp,
                "monitor_result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result.active_lockouts else 1


def cmd_testnet_monitor_loop(args: argparse.Namespace) -> int:
    _, _, engine, runtime, orchestrator = _build_testnet_runtime()
    start = time.monotonic()
    completed = 0
    last_result = None
    while True:
        if args.iterations is not None and completed >= args.iterations:
            break
        if args.duration_seconds is not None and time.monotonic() - start >= args.duration_seconds:
            break
        timestamp = engine.server_timestamp()
        last_result = orchestrator.reconcile_once(symbol=args.symbol, timestamp=timestamp)
        completed += 1
        if args.iterations is not None and completed >= args.iterations:
            break
        if args.duration_seconds is not None and time.monotonic() - start >= args.duration_seconds:
            break
        time.sleep(args.interval_seconds)
    print(
        json.dumps(
            {
                "status": "ok" if last_result is None or not last_result.active_lockouts else "paused",
                "iterations_completed": completed,
                "last_result": _to_jsonable(last_result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if last_result is None:
        return 0
    return 0 if not last_result.active_lockouts else 1


def cmd_testnet_manage_position_once(args: argparse.Namespace) -> int:
    _, _, engine, runtime, orchestrator = _build_testnet_runtime()
    timestamp = engine.server_timestamp()
    result = orchestrator.manage_open_position_once(
        symbol=args.symbol,
        timestamp=timestamp,
        candle_limit=args.candle_limit,
    )
    print(
        json.dumps(
            {
                "status": "ok" if result.action != "EXIT" else "exited",
                "server_time_ms": timestamp,
                "result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_manage_position_loop(args: argparse.Namespace) -> int:
    _, _, engine, runtime, orchestrator = _build_testnet_runtime()
    start = time.monotonic()
    completed = 0
    last_result = None
    while True:
        if args.iterations is not None and completed >= args.iterations:
            break
        if args.duration_seconds is not None and time.monotonic() - start >= args.duration_seconds:
            break
        timestamp = engine.server_timestamp()
        last_result = orchestrator.manage_open_position_once(
            symbol=args.symbol,
            timestamp=timestamp,
            candle_limit=args.candle_limit,
        )
        completed += 1
        if last_result.action == "EXIT":
            break
        if args.iterations is not None and completed >= args.iterations:
            break
        if args.duration_seconds is not None and time.monotonic() - start >= args.duration_seconds:
            break
        time.sleep(args.interval_seconds)
    print(
        json.dumps(
            {
                "status": "ok",
                "iterations_completed": completed,
                "last_result": _to_jsonable(last_result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_auto_cycle_once(args: argparse.Namespace) -> int:
    settings, _, engine, runtime, orchestrator = _build_testnet_runtime()
    timestamp = engine.server_timestamp()
    monitor = orchestrator.reconcile_once(symbol=args.symbol, timestamp=timestamp)
    managed_state = runtime.load_managed_trade_state(args.symbol)
    if monitor.safety_action is not None:
        print(
            json.dumps(
                {
                    "status": "paused" if monitor.active_lockouts else "ok",
                    "server_time_ms": timestamp,
                    "monitor_result": _to_jsonable(monitor),
                    "result": None,
                    "runtime_db_path": str(runtime.database_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if not monitor.active_lockouts else 1
    if managed_state is not None:
        result = orchestrator.manage_open_position_once(
            symbol=args.symbol,
            timestamp=timestamp,
            candle_limit=args.candle_limit,
        )
    else:
        if monitor.active_lockouts:
            print(
                json.dumps(
                    {
                        "status": "paused",
                        "server_time_ms": timestamp,
                        "monitor_result": _to_jsonable(monitor),
                        "result": None,
                        "runtime_db_path": str(runtime.database_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        trigger_close_time_ms = orchestrator.latest_runtime_trigger_close_time(
            symbol=args.symbol,
            timestamp=timestamp,
        )
        result = orchestrator.attempt_rule_entry_once(
            symbol=args.symbol,
            timestamp=timestamp,
            entry_notional_usdt=args.entry_notional_usdt,
            leverage=settings.live_start_leverage,
            candle_limit=args.candle_limit,
            trigger_close_time_ms=trigger_close_time_ms,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "server_time_ms": timestamp,
                "monitor_result": _to_jsonable(monitor),
                "result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_auto_cycle_loop(args: argparse.Namespace) -> int:
    settings, _, engine, runtime, orchestrator = _build_testnet_runtime()
    result = asyncio.run(
        orchestrator.run_event_driven_cycle(
            symbol=args.symbol,
            entry_notional_usdt=args.entry_notional_usdt,
            leverage=settings.live_start_leverage,
            candle_limit=args.candle_limit,
            duration_seconds=args.duration_seconds,
            max_closed_candles=args.iterations,
            reconnect_delay_seconds=args.interval_seconds,
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_mainnet_auto_cycle_once(args: argparse.Namespace) -> int:
    if not args.confirm_mainnet_live:
        raise ValueError("mainnet-auto-cycle-once requires --confirm-mainnet-live")

    settings, _, engine, runtime, orchestrator = _build_mainnet_runtime()
    timestamp = engine.server_timestamp()
    monitor = orchestrator.reconcile_once(symbol=args.symbol, timestamp=timestamp)
    managed_state = runtime.load_managed_trade_state(args.symbol)
    if monitor.safety_action is not None:
        print(
            json.dumps(
                {
                    "status": "paused" if monitor.active_lockouts else "ok",
                    "environment": "mainnet",
                    "server_time_ms": timestamp,
                    "monitor_result": _to_jsonable(monitor),
                    "result": None,
                    "runtime_db_path": str(runtime.database_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if not monitor.active_lockouts else 1
    if managed_state is not None:
        result = orchestrator.manage_open_position_once(
            symbol=args.symbol,
            timestamp=timestamp,
            candle_limit=args.candle_limit,
        )
    else:
        if monitor.active_lockouts:
            print(
                json.dumps(
                    {
                        "status": "paused",
                        "environment": "mainnet",
                        "server_time_ms": timestamp,
                        "monitor_result": _to_jsonable(monitor),
                        "result": None,
                        "runtime_db_path": str(runtime.database_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        trigger_close_time_ms = orchestrator.latest_runtime_trigger_close_time(
            symbol=args.symbol,
            timestamp=timestamp,
        )
        entry_notional_usdt, sizing = _mainnet_entry_notional_usdt(
            engine,
            fixed_entry_notional_usdt=args.entry_notional_usdt,
            leverage=settings.live_start_leverage,
            entry_margin_fraction=args.entry_margin_fraction,
        )
        result = orchestrator.attempt_rule_entry_once(
            symbol=args.symbol,
            timestamp=timestamp,
            entry_notional_usdt=entry_notional_usdt,
            leverage=settings.live_start_leverage,
            candle_limit=args.candle_limit,
            trigger_close_time_ms=trigger_close_time_ms,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "server_time_ms": timestamp,
                "sizing": _to_jsonable(sizing) if managed_state is None else None,
                "monitor_result": _to_jsonable(monitor),
                "result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_mainnet_auto_cycle_loop(args: argparse.Namespace) -> int:
    if not args.confirm_mainnet_live:
        raise ValueError("mainnet-auto-cycle-loop requires --confirm-mainnet-live")

    settings, _, engine, runtime, orchestrator = _build_mainnet_runtime()

    def _entry_notional_usdt() -> float:
        entry_notional, _ = _mainnet_entry_notional_usdt(
            engine,
            fixed_entry_notional_usdt=args.entry_notional_usdt,
            leverage=settings.live_start_leverage,
            entry_margin_fraction=args.entry_margin_fraction,
        )
        return entry_notional

    _, initial_sizing = _mainnet_entry_notional_usdt(
        engine,
        fixed_entry_notional_usdt=args.entry_notional_usdt,
        leverage=settings.live_start_leverage,
        entry_margin_fraction=args.entry_margin_fraction,
    )
    result = asyncio.run(
        orchestrator.run_event_driven_cycle(
            symbol=args.symbol,
            entry_notional_usdt=_entry_notional_usdt,
            leverage=settings.live_start_leverage,
            candle_limit=args.candle_limit,
            duration_seconds=args.duration_seconds,
            max_closed_candles=args.iterations,
            reconnect_delay_seconds=args.interval_seconds,
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "sizing": _to_jsonable(initial_sizing),
                "result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _priority_symbols(value: str) -> list[str]:
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    expected = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    if symbols != expected:
        raise ValueError("priority symbols are fixed and must be BTCUSDT,ETHUSDT,SOLUSDT")
    return symbols


def _dashboard_symbols(value: str | None, fallback_symbol: str) -> list[str]:
    if not value:
        return [fallback_symbol.upper()]
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not symbols:
        return [fallback_symbol.upper()]
    return symbols


def _mainnet_priority_settings(settings, symbol: str):
    exit_policy = settings.exit_policy
    fixed_take_profit_r = settings.fixed_take_profit_r
    if symbol in {"ETHUSDT", "SOLUSDT"}:
        exit_policy = "time_stop_only"
        fixed_take_profit_r = 1.0
    return replace(
        settings,
        strategy_mode="multi_symbol_priority_v1",
        exit_policy=exit_policy,
        fixed_take_profit_r=fixed_take_profit_r,
    )


def _mainnet_priority_leverage(settings, symbol: str) -> int:
    return settings.live_start_leverage


def _mainnet_entry_notional_usdt(
    engine: TestnetExecutionEngine,
    *,
    fixed_entry_notional_usdt: float | None,
    leverage: int,
    entry_margin_fraction: float,
) -> tuple[float, dict[str, object]]:
    if fixed_entry_notional_usdt is not None:
        raise ValueError("--entry-notional-usdt is disabled on mainnet; use --entry-margin-fraction")
    if not 0.0 < entry_margin_fraction <= 1.0:
        raise ValueError("--entry-margin-fraction must be > 0 and <= 1")

    timestamp = engine.server_timestamp()
    account = engine.client.query_account(timestamp=timestamp)
    available_balance = _account_available_usdt(account)
    if available_balance is None:
        raise ValueError("account availableBalance is missing; cannot size priority entry")
    entry_notional = available_balance * entry_margin_fraction * leverage
    return entry_notional, {
        "mode": "available_balance_fraction",
        "available_balance_usdt": available_balance,
        "entry_margin_fraction": entry_margin_fraction,
        "leverage": leverage,
        "entry_notional_usdt": entry_notional,
        "server_time_ms": timestamp,
    }


def _account_available_usdt(account: dict[str, object]) -> float | None:
    assets = account.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, dict) and str(asset.get("asset", "")).upper() == "USDT":
                value = asset.get("availableBalance")
                if value not in (None, ""):
                    return float(value)
    value = account.get("availableBalance")
    if value in (None, ""):
        return None
    return float(value)


def _mainnet_priority_entry_notional_usdt(
    engine: TestnetExecutionEngine,
    *,
    fixed_entry_notional_usdt: float | None,
    leverage: int,
    entry_margin_fraction: float,
) -> tuple[float, dict[str, object]]:
    return _mainnet_entry_notional_usdt(
        engine,
        fixed_entry_notional_usdt=fixed_entry_notional_usdt,
        leverage=leverage,
        entry_margin_fraction=entry_margin_fraction,
    )


def _remote_has_open_position(monitor: MonitorResult | None) -> bool:
    if monitor is None or monitor.reconciliation is None:
        return False
    return any(
        abs(float(position.get("positionAmt", "0"))) > 1e-9
        for position in monitor.reconciliation.remote_state.positions
    )


def _remote_account_open_position_symbols(engine: TestnetExecutionEngine) -> list[str]:
    timestamp = engine.server_timestamp()
    positions = engine.client.query_position_risk(timestamp=timestamp)
    symbols: list[str] = []
    for position in positions:
        if abs(float(position.get("positionAmt", "0"))) > 1e-9:
            symbol = str(position.get("symbol", "")).upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _managed_symbol(runtime: TestnetExecutionRuntime, symbols: list[str]) -> str | None:
    for symbol in symbols:
        if runtime.load_managed_trade_state(symbol) is not None:
            return symbol
    return None


def _ensure_priority_user_stream_heartbeat(
    orchestrator: TestnetRuntimeOrchestrator,
    *,
    symbol: str,
    timestamp: int,
) -> None:
    orchestrator.ensure_user_stream_heartbeat(symbol=symbol, timestamp=timestamp)


def _run_mainnet_priority_cycle(
    *,
    settings,
    engine: TestnetExecutionEngine,
    runtime: TestnetExecutionRuntime,
    orchestrator: TestnetRuntimeOrchestrator,
    symbols: list[str],
    entry_notional_usdt: float | None,
    entry_margin_fraction: float,
    candle_limit: int,
    last_close_times: dict[str, int] | None = None,
    last_reconcile_times: dict[str, int] | None = None,
    idle_reconcile_seconds: float = 60.0,
) -> dict[str, object]:
    cycle_started_ms = engine.server_timestamp()
    monitors: dict[str, MonitorResult] = {}
    symbol_results: dict[str, object] = {}

    managed = _managed_symbol(runtime, symbols)
    if managed is not None:
        runtime.settings = _mainnet_priority_settings(settings, managed)
        managed_timestamp = engine.server_timestamp()
        _ensure_priority_user_stream_heartbeat(
            orchestrator,
            symbol=managed,
            timestamp=managed_timestamp,
        )
        result = orchestrator.manage_open_position_once(
            symbol=managed,
            timestamp=managed_timestamp,
            candle_limit=candle_limit,
        )
        monitor: MonitorResult | None = None
        idle_reconcile_ms = max(0, int(idle_reconcile_seconds * 1000))
        reconcile_due = (
            last_reconcile_times is None
            or idle_reconcile_ms <= 0
            or managed_timestamp - last_reconcile_times.get(managed, 0) >= idle_reconcile_ms
        )
        if reconcile_due:
            monitor_timestamp = engine.server_timestamp()
            monitor = orchestrator.reconcile_once(
                symbol=managed,
                timestamp=monitor_timestamp,
            )
            if last_reconcile_times is not None:
                last_reconcile_times[managed] = monitor_timestamp
        return {
            "server_time_ms": managed_timestamp,
            "cycle_started_ms": cycle_started_ms,
            "action": "MANAGE_OPEN_POSITION",
            "managed_symbol": managed,
            "result": _to_jsonable(result),
            "monitor": _to_jsonable(monitor),
        }

    account_open_symbols = _remote_account_open_position_symbols(engine)
    if account_open_symbols:
        return {
            "server_time_ms": engine.server_timestamp(),
            "cycle_started_ms": cycle_started_ms,
            "action": "SKIP_ACCOUNT_POSITION",
            "remote_open_symbols": account_open_symbols,
            "result": None,
        }

    idle_reconcile_ms = max(0, int(idle_reconcile_seconds * 1000))
    for symbol in symbols:
        runtime.settings = _mainnet_priority_settings(settings, symbol)
        trigger_timestamp = engine.server_timestamp()
        trigger_close_time_ms = orchestrator.latest_runtime_trigger_close_time(
            symbol=symbol,
            timestamp=trigger_timestamp,
        )
        reconcile_due = (
            last_reconcile_times is None
            or idle_reconcile_ms <= 0
            or trigger_timestamp - last_reconcile_times.get(symbol, 0) >= idle_reconcile_ms
        )
        if last_close_times is not None and last_close_times.get(symbol) == trigger_close_time_ms:
            if reconcile_due:
                monitor_timestamp = engine.server_timestamp()
                monitors[symbol] = orchestrator.reconcile_once(
                    symbol=symbol,
                    timestamp=monitor_timestamp,
                )
                if last_reconcile_times is not None:
                    last_reconcile_times[symbol] = monitor_timestamp
            symbol_results[symbol] = {
                "action": "SKIP_ALREADY_PROCESSED_CANDLE",
                "trigger_close_time_ms": trigger_close_time_ms,
            }
            continue
        if last_close_times is not None:
            last_close_times[symbol] = trigger_close_time_ms

        monitor_timestamp = engine.server_timestamp()
        monitor = orchestrator.reconcile_once(
            symbol=symbol,
            timestamp=monitor_timestamp,
        )
        monitors[symbol] = monitor
        if last_reconcile_times is not None:
            last_reconcile_times[symbol] = monitor_timestamp
        if monitor.safety_action is not None:
            symbol_results[symbol] = {
                "action": "SKIP_SAFETY_ACTION",
                "safety_action": monitor.safety_action,
            }
            continue
        if monitor.active_lockouts:
            symbol_results[symbol] = {
                "action": "SKIP_LOCKOUT",
                "active_lockouts": monitor.active_lockouts,
            }
            continue

        leverage = _mainnet_priority_leverage(settings, symbol)
        resolved_entry_notional_usdt, sizing = _mainnet_priority_entry_notional_usdt(
            engine,
            fixed_entry_notional_usdt=entry_notional_usdt,
            leverage=leverage,
            entry_margin_fraction=entry_margin_fraction,
        )
        entry_timestamp = engine.server_timestamp()
        result = orchestrator.attempt_rule_entry_once(
            symbol=symbol,
            timestamp=entry_timestamp,
            entry_notional_usdt=resolved_entry_notional_usdt,
            leverage=leverage,
            candle_limit=candle_limit,
            trigger_close_time_ms=trigger_close_time_ms,
        )
        symbol_results[symbol] = {
            "sizing": _to_jsonable(sizing),
            "result": _to_jsonable(result),
        }
        if result.action == "PLACED":
            monitor_timestamp = engine.server_timestamp()
            monitors[symbol] = orchestrator.reconcile_once(
                symbol=symbol,
                timestamp=monitor_timestamp,
            )
            if last_reconcile_times is not None:
                last_reconcile_times[symbol] = monitor_timestamp
            return {
                "server_time_ms": monitor_timestamp,
                "cycle_started_ms": cycle_started_ms,
                "action": "PLACED",
                "selected_symbol": symbol,
                "monitors": _to_jsonable(monitors),
                "symbol_results": symbol_results,
            }

    return {
        "server_time_ms": engine.server_timestamp(),
        "cycle_started_ms": cycle_started_ms,
        "action": "NO_TRADE",
        "monitors": _to_jsonable(monitors),
        "symbol_results": symbol_results,
    }


def cmd_mainnet_priority_auto_cycle_once(args: argparse.Namespace) -> int:
    if not args.confirm_mainnet_live:
        raise ValueError("mainnet-priority-auto-cycle-once requires --confirm-mainnet-live")

    settings, _, engine, runtime, orchestrator = _build_mainnet_runtime()
    symbols = _priority_symbols(args.symbols)
    runtime.settings = replace(settings, strategy_mode="multi_symbol_priority_v1")
    result = _run_mainnet_priority_cycle(
        settings=settings,
        engine=engine,
        runtime=runtime,
        orchestrator=orchestrator,
        symbols=symbols,
        entry_notional_usdt=args.entry_notional_usdt,
        entry_margin_fraction=args.entry_margin_fraction,
        candle_limit=args.candle_limit,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "strategy_mode": "multi_symbol_priority_v1",
                "symbols": symbols,
                "result": _to_jsonable(result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_mainnet_priority_auto_cycle_loop(args: argparse.Namespace) -> int:
    if not args.confirm_mainnet_live:
        raise ValueError("mainnet-priority-auto-cycle-loop requires --confirm-mainnet-live")

    settings, _, engine, runtime, orchestrator = _build_mainnet_runtime()
    symbols = _priority_symbols(args.symbols)
    runtime.settings = replace(settings, strategy_mode="multi_symbol_priority_v1")
    started = time.monotonic()
    completed = 0
    last_result: dict[str, object] | None = None
    last_close_times: dict[str, int] = {}
    last_reconcile_times: dict[str, int] = {}

    while True:
        if args.iterations is not None and completed >= args.iterations:
            break
        if args.duration_seconds is not None and time.monotonic() - started >= args.duration_seconds:
            break
        last_result = _run_mainnet_priority_cycle(
            settings=settings,
            engine=engine,
            runtime=runtime,
            orchestrator=orchestrator,
            symbols=symbols,
            entry_notional_usdt=args.entry_notional_usdt,
            entry_margin_fraction=args.entry_margin_fraction,
            candle_limit=args.candle_limit,
            last_close_times=last_close_times,
            last_reconcile_times=last_reconcile_times,
            idle_reconcile_seconds=args.idle_reconcile_seconds,
        )
        completed += 1
        if args.iterations is not None and completed >= args.iterations:
            break
        if args.duration_seconds is not None and time.monotonic() - started >= args.duration_seconds:
            break
        time.sleep(max(1.0, args.interval_seconds))

    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "strategy_mode": "multi_symbol_priority_v1",
                "symbols": symbols,
                "iterations_completed": completed,
                "idle_reconcile_seconds": args.idle_reconcile_seconds,
                "last_result": _to_jsonable(last_result),
                "runtime_db_path": str(runtime.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_testnet_runtime_status(args: argparse.Namespace) -> int:
    _, runtime = _build_runtime_store()
    summary = runtime.runtime_summary(args.symbol, incident_limit=args.incident_limit)
    print(json.dumps(_to_jsonable(summary), indent=2, sort_keys=True))
    return 0


def cmd_mainnet_runtime_status(args: argparse.Namespace) -> int:
    _, runtime = _build_mainnet_runtime_store()
    summary = runtime.runtime_summary(args.symbol, incident_limit=args.incident_limit)
    print(json.dumps(_to_jsonable(summary), indent=2, sort_keys=True))
    return 0


def cmd_testnet_forward_review(args: argparse.Namespace) -> int:
    settings, runtime = _build_runtime_store()
    payload = {
        "symbol": args.symbol,
        "strategy_context": runtime_strategy_context(settings),
        "progress": runtime.forward_trade_progress(
            args.symbol,
            target_min_trades=args.min_trades,
            target_max_trades=args.target_trades,
        ),
        "trade_performance_summary": runtime.trade_performance_summary(args.symbol),
        "sizing_review": runtime.position_sizing_review(
            args.symbol,
            min_trade_count=args.min_trades,
            target_trade_count=args.target_trades,
            entry_notional_usdt=args.entry_notional_usdt,
            leverage=args.leverage or float(settings.live_start_leverage),
            account_equity_usdt=args.account_equity_usdt,
        ),
        "database_path": str(runtime.database_path),
    }
    print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))
    return 0


def cmd_testnet_incident_tail(args: argparse.Namespace) -> int:
    _, runtime = _build_runtime_store()
    incidents = [incident.__dict__ for incident in runtime.recent_incidents(limit=args.limit)]
    print(json.dumps(_to_jsonable({"incidents": incidents}), indent=2, sort_keys=True))
    return 0


def cmd_testnet_risk_status(args: argparse.Namespace) -> int:
    _, runtime = _build_runtime_store()
    lockouts = [lockout.to_dict() for lockout in runtime.active_lockouts(args.symbol)]
    print(json.dumps(_to_jsonable({"symbol": args.symbol, "lockouts": lockouts}), indent=2, sort_keys=True))
    return 0


def cmd_testnet_manual_pause(args: argparse.Namespace) -> int:
    _, runtime = _build_runtime_store()
    runtime.acquire_lockout(
        symbol=args.symbol,
        code="manual_pause",
        reason="paused by cli operator",
        details={"source": "cli"},
    )
    state = _runtime_state_after_manual_lockout_change(runtime, symbol=args.symbol)
    print(json.dumps(_to_jsonable({"status": "ok", "symbol": args.symbol, "runtime_state": state}), indent=2, sort_keys=True))
    return 0


def cmd_testnet_manual_resume(args: argparse.Namespace) -> int:
    _, runtime = _build_runtime_store()
    runtime.release_lockout(
        symbol=args.symbol,
        code="manual_pause",
        reason="resumed by cli operator",
    )
    runtime.release_lockout(
        symbol=args.symbol,
        code="manual_review_required",
        reason="manual review cleared by cli operator",
    )
    runtime.release_lockout(
        symbol=args.symbol,
        code="repeated_loss_pattern_review",
        reason="repeated loss pattern review cleared by cli operator",
    )
    state = _runtime_state_after_manual_lockout_change(runtime, symbol=args.symbol)
    print(json.dumps(_to_jsonable({"status": "ok", "symbol": args.symbol, "runtime_state": state}), indent=2, sort_keys=True))
    return 0


def cmd_mainnet_manual_pause(args: argparse.Namespace) -> int:
    _, runtime = _build_mainnet_runtime_store()
    runtime.acquire_lockout(
        symbol=args.symbol,
        code="manual_pause",
        reason="paused by cli operator",
        details={"source": "cli", "environment": "mainnet"},
    )
    state = _runtime_state_after_manual_lockout_change(runtime, symbol=args.symbol)
    print(
        json.dumps(
            _to_jsonable(
                {
                    "status": "ok",
                    "environment": "mainnet",
                    "symbol": args.symbol,
                    "runtime_state": state,
                    "runtime_db_path": str(runtime.database_path),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_mainnet_manual_resume(args: argparse.Namespace) -> int:
    _, runtime = _build_mainnet_runtime_store()
    runtime.release_lockout(
        symbol=args.symbol,
        code="manual_pause",
        reason="resumed by cli operator",
    )
    runtime.release_lockout(
        symbol=args.symbol,
        code="manual_review_required",
        reason="manual review cleared by cli operator",
    )
    runtime.release_lockout(
        symbol=args.symbol,
        code="repeated_loss_pattern_review",
        reason="repeated loss pattern review cleared by cli operator",
    )
    state = _runtime_state_after_manual_lockout_change(runtime, symbol=args.symbol)
    print(
        json.dumps(
            _to_jsonable(
                {
                    "status": "ok",
                    "environment": "mainnet",
                    "symbol": args.symbol,
                    "runtime_state": state,
                    "runtime_db_path": str(runtime.database_path),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _runtime_state_after_manual_lockout_change(runtime: TestnetExecutionRuntime, *, symbol: str) -> str:
    remaining = runtime.active_lockouts(symbol)
    expected_state = runtime.load_expected_state(symbol)
    managed_state = runtime.load_managed_trade_state(symbol)
    state = "PROTECTED" if expected_state is not None or managed_state is not None else ("PAUSED" if remaining else "READY")
    runtime.update_runtime_status(symbol=symbol, state=state)
    return state


def cmd_testnet_dashboard(args: argparse.Namespace) -> int:
    symbols = _dashboard_symbols(args.symbols, args.symbol)
    server = serve_dashboard(
        host=args.host,
        port=args.port,
        symbol=symbols[0],
        symbols=symbols,
        environment="testnet",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "testnet",
                "url": f"http://{args.host}:{args.port}",
                "symbol": symbols[0],
                "symbols": symbols,
            },
            indent=2,
            sort_keys=True,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_mainnet_dashboard(args: argparse.Namespace) -> int:
    symbols = _dashboard_symbols(args.symbols, args.symbol)
    server = serve_dashboard(
        host=args.host,
        port=args.port,
        symbol=symbols[0],
        symbols=symbols,
        environment="mainnet",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "mainnet",
                "url": f"http://{args.host}:{args.port}",
                "symbol": symbols[0],
                "symbols": symbols,
            },
            indent=2,
            sort_keys=True,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "show-config":
        return cmd_show_config()
    if args.command == "local-ai-check":
        return cmd_local_ai_check()
    if args.command == "local-ai-bench":
        return cmd_local_ai_bench(args)
    if args.command == "check-layout":
        return cmd_check_layout()
    if args.command == "fetch-historical":
        return cmd_fetch_historical(args)
    if args.command == "backfill-historical":
        return cmd_backfill_historical(args)
    if args.command == "backtest-run":
        return cmd_backtest_run(args)
    if args.command == "record-mark-price":
        return cmd_record_mark_price(args)
    if args.command == "testnet-check":
        return cmd_testnet_check(args)
    if args.command == "testnet-ensure-config":
        return cmd_testnet_ensure_config(args)
    if args.command == "testnet-order-test":
        return cmd_testnet_order_test(args)
    if args.command == "testnet-place-bundle":
        return cmd_testnet_place_bundle(args)
    if args.command == "testnet-reconcile":
        return cmd_testnet_reconcile(args)
    if args.command == "mainnet-check":
        return cmd_mainnet_check(args)
    if args.command == "mainnet-ensure-config":
        return cmd_mainnet_ensure_config(args)
    if args.command == "mainnet-order-test":
        return cmd_mainnet_order_test(args)
    if args.command == "mainnet-place-bundle":
        return cmd_mainnet_place_bundle(args)
    if args.command == "mainnet-reconcile":
        return cmd_mainnet_reconcile(args)
    if args.command == "testnet-runtime-recover":
        return cmd_testnet_runtime_recover(args)
    if args.command == "testnet-user-stream-start":
        return cmd_testnet_user_stream_start(args)
    if args.command == "testnet-user-stream-keepalive":
        return cmd_testnet_user_stream_keepalive(args)
    if args.command == "testnet-user-stream-close":
        return cmd_testnet_user_stream_close(args)
    if args.command == "testnet-user-stream-run":
        return cmd_testnet_user_stream_run(args)
    if args.command == "testnet-monitor-once":
        return cmd_testnet_monitor_once(args)
    if args.command == "testnet-monitor-loop":
        return cmd_testnet_monitor_loop(args)
    if args.command == "testnet-manage-position-once":
        return cmd_testnet_manage_position_once(args)
    if args.command == "testnet-manage-position-loop":
        return cmd_testnet_manage_position_loop(args)
    if args.command == "testnet-auto-cycle-once":
        return cmd_testnet_auto_cycle_once(args)
    if args.command == "testnet-auto-cycle-loop":
        return cmd_testnet_auto_cycle_loop(args)
    if args.command == "mainnet-auto-cycle-once":
        return cmd_mainnet_auto_cycle_once(args)
    if args.command == "mainnet-auto-cycle-loop":
        return cmd_mainnet_auto_cycle_loop(args)
    if args.command == "mainnet-priority-auto-cycle-once":
        return cmd_mainnet_priority_auto_cycle_once(args)
    if args.command == "mainnet-priority-auto-cycle-loop":
        return cmd_mainnet_priority_auto_cycle_loop(args)
    if args.command == "testnet-runtime-status":
        return cmd_testnet_runtime_status(args)
    if args.command == "mainnet-runtime-status":
        return cmd_mainnet_runtime_status(args)
    if args.command == "testnet-forward-review":
        return cmd_testnet_forward_review(args)
    if args.command == "testnet-incident-tail":
        return cmd_testnet_incident_tail(args)
    if args.command == "testnet-risk-status":
        return cmd_testnet_risk_status(args)
    if args.command == "testnet-manual-pause":
        return cmd_testnet_manual_pause(args)
    if args.command == "testnet-manual-resume":
        return cmd_testnet_manual_resume(args)
    if args.command == "mainnet-manual-pause":
        return cmd_mainnet_manual_pause(args)
    if args.command == "mainnet-manual-resume":
        return cmd_mainnet_manual_resume(args)
    if args.command == "testnet-dashboard":
        return cmd_testnet_dashboard(args)
    if args.command == "mainnet-dashboard":
        return cmd_mainnet_dashboard(args)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
