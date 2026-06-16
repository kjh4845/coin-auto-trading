from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from ai_auto_trading.cli import (
    build_parser,
    main,
    _mainnet_entry_notional_usdt,
    _run_mainnet_priority_cycle,
)
from ai_auto_trading.runtime.orchestrator import ManagedPositionResult, MonitorResult
from ai_auto_trading.settings import load_settings


class CliTest(unittest.TestCase):
    def test_parser_has_expected_commands(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.prog, "ai_auto_trading")
        self.assertEqual(parser.parse_args(["testnet-check"]).command, "testnet-check")
        self.assertEqual(parser.parse_args(["local-ai-bench"]).command, "local-ai-bench")
        self.assertEqual(
            parser.parse_args(["testnet-user-stream-start"]).command,
            "testnet-user-stream-start",
        )
        self.assertEqual(
            parser.parse_args(["testnet-user-stream-run"]).command,
            "testnet-user-stream-run",
        )
        self.assertEqual(
            parser.parse_args(["testnet-monitor-once"]).command,
            "testnet-monitor-once",
        )
        self.assertEqual(
            parser.parse_args(["testnet-monitor-loop"]).command,
            "testnet-monitor-loop",
        )
        self.assertEqual(
            parser.parse_args(["testnet-manage-position-once"]).command,
            "testnet-manage-position-once",
        )
        self.assertEqual(
            parser.parse_args(["testnet-manage-position-loop"]).command,
            "testnet-manage-position-loop",
        )
        self.assertEqual(
            parser.parse_args(["testnet-auto-cycle-once"]).command,
            "testnet-auto-cycle-once",
        )
        self.assertEqual(
            parser.parse_args(["testnet-auto-cycle-loop"]).command,
            "testnet-auto-cycle-loop",
        )
        self.assertEqual(
            parser.parse_args(["testnet-runtime-status"]).command,
            "testnet-runtime-status",
        )
        self.assertEqual(
            parser.parse_args(["testnet-forward-review"]).command,
            "testnet-forward-review",
        )
        self.assertEqual(
            parser.parse_args(["testnet-dashboard"]).command,
            "testnet-dashboard",
        )
        self.assertEqual(
            parser.parse_args(["mainnet-dashboard"]).command,
            "mainnet-dashboard",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "mainnet-dashboard",
                    "--symbols",
                    "BTCUSDT,ETHUSDT,SOLUSDT",
                ]
            ).symbols,
            "BTCUSDT,ETHUSDT,SOLUSDT",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "backtest-run",
                    "--contract-parquet",
                    "contract.parquet",
                    "--mark-parquet",
                    "mark.parquet",
                ]
            ).command,
            "backtest-run",
        )
        self.assertEqual(
            parser.parse_args(["testnet-manual-pause"]).command,
            "testnet-manual-pause",
        )
        self.assertEqual(
            parser.parse_args(["testnet-manual-resume"]).command,
            "testnet-manual-resume",
        )
        self.assertEqual(
            parser.parse_args(["testnet-runtime-recover"]).command,
            "testnet-runtime-recover",
        )
        self.assertEqual(parser.parse_args(["mainnet-check"]).command, "mainnet-check")
        self.assertEqual(
            parser.parse_args(
                [
                    "mainnet-order-test",
                    "--side",
                    "SELL",
                    "--quantity",
                    "0.002",
                    "--stop-price",
                    "80000",
                ]
            ).command,
            "mainnet-order-test",
        )
        self.assertEqual(
            parser.parse_args(["mainnet-auto-cycle-loop"]).command,
            "mainnet-auto-cycle-loop",
        )
        self.assertEqual(
            parser.parse_args(["mainnet-priority-auto-cycle-once"]).command,
            "mainnet-priority-auto-cycle-once",
        )
        self.assertEqual(
            parser.parse_args(["mainnet-priority-auto-cycle-loop"]).command,
            "mainnet-priority-auto-cycle-loop",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "mainnet-priority-auto-cycle-loop",
                    "--idle-reconcile-seconds",
                    "60",
                ]
            ).idle_reconcile_seconds,
            60.0,
        )
        priority_args = parser.parse_args(["mainnet-priority-auto-cycle-loop"])
        self.assertIsNone(priority_args.entry_notional_usdt)
        self.assertEqual(priority_args.entry_margin_fraction, 0.98)
        mainnet_args = parser.parse_args(["mainnet-auto-cycle-loop"])
        self.assertIsNone(mainnet_args.entry_notional_usdt)
        self.assertEqual(mainnet_args.entry_margin_fraction, 0.98)
        self.assertEqual(
            parser.parse_args(["mainnet-runtime-status"]).command,
            "mainnet-runtime-status",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "testnet-order-test",
                    "--side",
                    "BUY",
                    "--quantity",
                    "0.01",
                    "--stop-price",
                    "97500",
                ]
            ).command,
            "testnet-order-test",
        )

    def test_check_layout_command_returns_success(self) -> None:
        exit_code = main(["check-layout"])
        self.assertEqual(exit_code, 0)

    def test_mainnet_sizing_rejects_fixed_entry_notional(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry-notional-usdt is disabled"):
            _mainnet_entry_notional_usdt(
                None,
                fixed_entry_notional_usdt=200.0,
                leverage=5,
                entry_margin_fraction=0.5,
            )

    def test_mainnet_priority_cycle_reconciles_managed_position(self) -> None:
        class Engine:
            def __init__(self) -> None:
                self.next_timestamp = 1_000_000

            def server_timestamp(self) -> int:
                self.next_timestamp += 1_000
                return self.next_timestamp

        class Runtime:
            def __init__(self) -> None:
                self.settings = None

            def load_managed_trade_state(self, symbol: str) -> object | None:
                return object() if symbol == "ETHUSDT" else None

        class Orchestrator:
            def __init__(self) -> None:
                self.heartbeat_calls: list[str] = []
                self.managed_calls: list[str] = []
                self.reconcile_calls: list[str] = []

            def ensure_user_stream_heartbeat(self, *, symbol: str, timestamp: int):
                self.heartbeat_calls.append(symbol)

            def manage_open_position_once(self, *, symbol: str, timestamp: int, candle_limit: int):
                self.managed_calls.append(symbol)
                return ManagedPositionResult(symbol=symbol, action="HOLD", runtime_state="PROTECTED")

            def reconcile_once(self, *, symbol: str, timestamp: int):
                self.reconcile_calls.append(symbol)
                return MonitorResult(
                    symbol=symbol,
                    reconciliation=None,
                    active_lockouts=[],
                    runtime_state="PROTECTED",
                )

        orchestrator = Orchestrator()
        result = _run_mainnet_priority_cycle(
            settings=load_settings(),
            engine=Engine(),
            runtime=Runtime(),
            orchestrator=orchestrator,
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            entry_notional_usdt=None,
            entry_margin_fraction=0.5,
            candle_limit=120,
            last_reconcile_times={},
            idle_reconcile_seconds=60.0,
        )

        self.assertEqual(result["action"], "MANAGE_OPEN_POSITION")
        self.assertEqual(orchestrator.heartbeat_calls, ["ETHUSDT"])
        self.assertEqual(orchestrator.managed_calls, ["ETHUSDT"])
        self.assertEqual(orchestrator.reconcile_calls, ["ETHUSDT"])
        self.assertIsNotNone(result["monitor"])

    def test_show_config_masks_exchange_credentials(self) -> None:
        buffer = io.StringIO()
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "mainnet123456",
                "BINANCE_API_SECRET": "mainnet-secret",
                "BINANCE_TESTNET_API_KEY": "abcdef123456",
                "BINANCE_TESTNET_API_SECRET": "secret-xyz",
            },
            clear=False,
        ):
            with redirect_stdout(buffer):
                exit_code = main(["show-config"])
        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["binance_api_key"], "mai...56")
        self.assertEqual(payload["binance_api_secret"], "***")
        self.assertTrue(payload["binance_api_key_configured"])
        self.assertTrue(payload["binance_api_secret_configured"])
        self.assertEqual(payload["binance_testnet_api_key"], "abc...56")
        self.assertEqual(payload["binance_testnet_api_secret"], "***")
        self.assertTrue(payload["binance_testnet_api_key_configured"])
        self.assertTrue(payload["binance_testnet_api_secret_configured"])
        self.assertIn("runtime_strategy_context", payload)

    def test_backtest_run_exports_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            contract_path = tmp_path / "contract.parquet"
            mark_path = tmp_path / "mark.parquet"
            output_dir = tmp_path / "out"
            contract_rows = []
            mark_rows = []
            base_time = 0
            for index in range(180):
                open_time = base_time + (index * 60_000)
                close_time = open_time + 59_999
                close = 100.0 + (index * 0.01)
                contract_rows.append(
                    {
                        "dataset": "contract_klines",
                        "symbol": "BTCUSDT",
                        "interval": "1m",
                        "open_time": open_time,
                        "open": close - 0.02,
                        "high": close + 0.04,
                        "low": close - 0.04,
                        "close": close,
                        "volume": 10.0,
                        "close_time": close_time,
                        "quote_asset_volume": 1000.0,
                        "number_of_trades": 10,
                        "taker_buy_base_asset_volume": 5.0,
                        "taker_buy_quote_asset_volume": 500.0,
                        "ignore": "0",
                        "collected_at_ms": 1,
                    }
                )
                mark_rows.append(
                    {
                        "dataset": "mark_price_klines",
                        "symbol": "BTCUSDT",
                        "interval": "1m",
                        "open_time": open_time,
                        "open": close - 0.01,
                        "high": close + 0.05,
                        "low": close - 0.05,
                        "close": close,
                        "volume": 0.0,
                        "close_time": close_time,
                        "quote_asset_volume": 0.0,
                        "number_of_trades": 1,
                        "taker_buy_base_asset_volume": 0.0,
                        "taker_buy_quote_asset_volume": 0.0,
                        "ignore": "0",
                        "collected_at_ms": 1,
                    }
                )
            pq.write_table(pa.Table.from_pylist(contract_rows), contract_path)
            pq.write_table(pa.Table.from_pylist(mark_rows), mark_path)
            buffer = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "STRATEGY_MODE": "single_profile",
                    "REGIME_TIMEFRAME": "1h",
                    "ANCHOR_TIMEFRAME": "1h",
                    "MIN_HIGHER_TF_EMA_SPREAD_PCT": "0",
                    "MIN_VOLUME_RATIO_20": "0",
                },
                clear=False,
            ):
                with redirect_stdout(buffer):
                    exit_code = main(
                        [
                            "backtest-run",
                            "--symbol",
                            "BTCUSDT",
                            "--contract-parquet",
                            str(contract_path),
                            "--mark-parquet",
                            str(mark_path),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ok")
            self.assertIn("metrics", payload)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "decision_report.json").exists())
            self.assertTrue((output_dir / "trade_logs.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
