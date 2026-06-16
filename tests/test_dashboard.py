from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from ai_auto_trading.runtime.dashboard import (
    DashboardController,
    _account_income_summary,
    _account_usdt_status_from_account,
    _mainnet_entry_notional_usdt as _dashboard_mainnet_entry_notional_usdt,
    _normalize_income_history_rows,
    serve_dashboard,
)
from ai_auto_trading.runtime.testnet import ExpectedRuntimeState, ManagedTradeState, TestnetExecutionRuntime
from ai_auto_trading.settings import Settings, load_settings


def _settings_with_temp_runtime(temp_root: Path) -> Settings:
    base = load_settings()
    return Settings(
        app_env=base.app_env,
        log_level=base.log_level,
        strategy_mode=base.strategy_mode,
        repo_root=base.repo_root,
        config_dir=base.config_dir,
        policy_dir=base.policy_dir,
        data_dir=temp_root,
        tests_dir=base.tests_dir,
        trading_symbol=base.trading_symbol,
        execution_timeframe=base.execution_timeframe,
        micro_timeframe=base.micro_timeframe,
        confirmation_timeframe=base.confirmation_timeframe,
        macro_timeframe=base.macro_timeframe,
        regime_timeframe=base.regime_timeframe,
        anchor_timeframe=base.anchor_timeframe,
        atr_trailing_multiplier=base.atr_trailing_multiplier,
        atr_trail_activation_profit_r=base.atr_trail_activation_profit_r,
        atr_trail_min_bars=base.atr_trail_min_bars,
        exit_policy=base.exit_policy,
        fixed_take_profit_r=base.fixed_take_profit_r,
        max_holding_bars=base.max_holding_bars,
        live_start_leverage=base.live_start_leverage,
        system_leverage_cap=base.system_leverage_cap,
        binance_futures_base_url=base.binance_futures_base_url,
        binance_futures_ws_base_url=base.binance_futures_ws_base_url,
        binance_futures_testnet_base_url=base.binance_futures_testnet_base_url,
        binance_futures_testnet_ws_base_url=base.binance_futures_testnet_ws_base_url,
        binance_api_key="",
        binance_api_secret="",
        binance_testnet_api_key="",
        binance_testnet_api_secret="",
        local_model_id=base.local_model_id,
        local_model_base=base.local_model_base,
        local_model_path=base.local_model_path,
        local_model_python=base.local_model_python,
        local_model_endpoint=base.local_model_endpoint,
        local_model_mps_max_memory=base.local_model_mps_max_memory,
        local_model_cpu_max_memory=base.local_model_cpu_max_memory,
        ai_gate_enabled=base.ai_gate_enabled,
        ai_gate_fail_open=base.ai_gate_fail_open,
        ai_gate_min_setup_quality=base.ai_gate_min_setup_quality,
        ai_reduce_size_fraction=base.ai_reduce_size_fraction,
        ai_request_timeout_seconds=base.ai_request_timeout_seconds,
        ai_max_tokens=base.ai_max_tokens,
        ai_max_latency_ms=base.ai_max_latency_ms,
        allow_long_entries=base.allow_long_entries,
        allow_short_entries=base.allow_short_entries,
        max_long_funding_rate=base.max_long_funding_rate,
        min_short_funding_rate=base.min_short_funding_rate,
        min_long_taker_buy_ratio=base.min_long_taker_buy_ratio,
        max_short_taker_buy_ratio=base.max_short_taker_buy_ratio,
        min_micro_long_ema_spread_pct=base.min_micro_long_ema_spread_pct,
        min_micro_short_ema_spread_pct=base.min_micro_short_ema_spread_pct,
        min_higher_tf_ema_spread_pct=base.min_higher_tf_ema_spread_pct,
        min_volume_ratio_20=base.min_volume_ratio_20,
        max_daily_loss_r=base.max_daily_loss_r,
        max_consecutive_losses=base.max_consecutive_losses,
        cooldown_after_loss_minutes=base.cooldown_after_loss_minutes,
    )


class DashboardTest(unittest.TestCase):
    def test_account_usdt_status_extracts_futures_balances(self) -> None:
        payload = {
            "totalWalletBalance": "67.10",
            "availableBalance": "55.00",
            "totalMarginBalance": "70.00",
            "totalUnrealizedProfit": "2.90",
            "totalInitialMargin": "15.00",
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "67.67",
                    "availableBalance": "55.50",
                    "marginBalance": "70.57",
                    "unrealizedProfit": "2.90",
                    "initialMargin": "15.00",
                    "maintMargin": "1.50",
                    "maxWithdrawAmount": "52.00",
                }
            ],
        }

        status = _account_usdt_status_from_account(payload, server_time_ms=123)

        self.assertEqual(status["asset"], "USDT")
        self.assertEqual(status["server_time_ms"], 123)
        self.assertEqual(status["wallet_balance_usdt"], 67.67)
        self.assertEqual(status["available_balance_usdt"], 55.50)
        self.assertEqual(status["margin_balance_usdt"], 70.57)
        self.assertEqual(status["unrealized_pnl_usdt"], 2.90)

    def test_account_equity_history_summarizes_seed_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = _settings_with_temp_runtime(Path(tmp_dir))
            runtime = TestnetExecutionRuntime(None, settings=settings)
            for index, equity in enumerate([67.0, 68.5, 69.0]):
                runtime.record_account_snapshot(
                    symbol="ACCOUNT",
                    snapshot_type="rest_account_usdt_balance",
                    payload={
                        "usdt": {
                            "wallet_balance_usdt": equity - 0.25,
                            "margin_balance_usdt": equity,
                            "available_balance_usdt": equity,
                            "unrealized_pnl_usdt": 0.25,
                        }
                    },
                    recorded_at_ms=1_000 + index,
                )
            controller = DashboardController(settings=settings)

            history = controller.account_equity_history(hours=0, max_points=2)

            self.assertEqual(history["point_count"], 3)
            self.assertEqual(len(history["points"]), 2)
            self.assertEqual(history["summary"]["start_equity_usdt"], 67.0)
            self.assertEqual(history["summary"]["current_equity_usdt"], 69.0)
            self.assertEqual(history["summary"]["change_usdt"], 2.0)

    def test_account_income_summary_uses_exchange_income_net(self) -> None:
        rows = _normalize_income_history_rows(
            [
                {
                    "symbol": "ETHUSDT",
                    "incomeType": "REALIZED_PNL",
                    "income": "0.20356999",
                    "asset": "USDT",
                    "time": 1_000,
                    "tranId": "1",
                },
                {
                    "symbol": "ETHUSDT",
                    "incomeType": "COMMISSION",
                    "income": "-0.83476558",
                    "asset": "USDT",
                    "time": 1_001,
                    "tranId": "2",
                },
                {
                    "symbol": "ETHUSDT",
                    "incomeType": "FUNDING_FEE",
                    "income": "0.00064474",
                    "asset": "USDT",
                    "time": 1_002,
                    "tranId": "3",
                },
            ]
        )

        summary = _account_income_summary(rows, symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])

        self.assertAlmostEqual(summary["realized_pnl_usdt"], 0.20356999)
        self.assertAlmostEqual(summary["commission_usdt"], -0.83476558)
        self.assertAlmostEqual(summary["funding_fee_usdt"], 0.00064474)
        self.assertAlmostEqual(summary["trading_net_income_usdt"], -0.63055085)
        self.assertAlmostEqual(
            summary["by_symbol"]["ETHUSDT"]["trading_net_income_usdt"],
            -0.63055085,
        )

    def test_mainnet_dashboard_sizing_rejects_fixed_entry_notional(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry_notional_usdt is disabled"):
            _dashboard_mainnet_entry_notional_usdt(
                None,
                fixed_entry_notional_usdt=200.0,
                leverage=5,
                entry_margin_fraction=0.5,
            )

    def test_controller_summary_and_manual_pause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = _settings_with_temp_runtime(Path(tmp_dir))
            runtime = TestnetExecutionRuntime(None, settings=settings)
            runtime.update_runtime_status(symbol="BTCUSDT", state="READY")
            controller = DashboardController(settings=settings)

            summary = controller.summary(symbol="BTCUSDT", incident_limit=5)
            self.assertEqual(summary["environment"], "testnet")
            self.assertEqual(summary["summary"]["runtime_status"]["state"], "READY")
            self.assertEqual(summary["summary"]["trade_performance_summary"]["total_trade_count"], 0)

            controller.handle_action(action="manual_pause", payload={"symbol": "BTCUSDT"})
            paused = controller.summary(symbol="BTCUSDT", incident_limit=5)
            self.assertEqual(paused["summary"]["active_lockouts"][0]["code"], "manual_pause")

            controller.handle_action(action="manual_resume", payload={"symbol": "BTCUSDT"})
            resumed = controller.summary(symbol="BTCUSDT", incident_limit=5)
            self.assertEqual(resumed["summary"]["active_lockouts"], [])

    def test_manual_resume_clears_manual_review_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = _settings_with_temp_runtime(Path(tmp_dir))
            runtime = TestnetExecutionRuntime(None, settings=settings)
            runtime.acquire_lockout(
                symbol="BTCUSDT",
                code="manual_review_required",
                reason="safety exit requires operator review",
            )
            runtime.update_runtime_status(symbol="BTCUSDT", state="PAUSED")
            controller = DashboardController(settings=settings)

            controller.handle_action(action="manual_resume", payload={"symbol": "BTCUSDT"})
            resumed = controller.summary(symbol="BTCUSDT", incident_limit=5)
            self.assertEqual(resumed["summary"]["active_lockouts"], [])

    def test_clear_expected_state_also_clears_managed_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = _settings_with_temp_runtime(Path(tmp_dir))
            runtime = TestnetExecutionRuntime(None, settings=settings)
            runtime.update_runtime_status(symbol="BTCUSDT", state="PROTECTED")
            runtime.save_expected_state(
                ExpectedRuntimeState(
                    symbol="BTCUSDT",
                    expected_position_qty=0.01,
                    expected_stop_price_mark=97500.0,
                    expected_leverage=2,
                    expected_margin_mode="ISOLATED",
                    expected_position_mode="ONE_WAY",
                    updated_at_ms=1,
                )
            )
            runtime.save_managed_trade_state(
                ManagedTradeState(
                    symbol="BTCUSDT",
                    side="LONG",
                    quantity=0.01,
                    leverage_at_entry=2.0,
                    entry_contract_price_avg=100000.0,
                    entry_mark_price=100000.0,
                    execution_timeframe="3m",
                    atr_trailing_multiplier=2.5,
                    max_holding_bars=8,
                    opened_at_ms=1,
                    signal_reason_codes=[],
                    model_base="rule_only",
                    adapter_version=None,
                    ai_snapshot=None,
                    bars_held=0,
                    highest_high=100000.0,
                    lowest_low=100000.0,
                    last_processed_candle_close_time_ms=None,
                    atr_trail_history=[],
                )
            )
            controller = DashboardController(settings=settings)
            result = controller.handle_action(action="clear_expected_state", payload={"symbol": "BTCUSDT"})
            self.assertEqual(result["status"], "ok")
            self.assertIsNone(runtime.load_expected_state("BTCUSDT"))
            self.assertIsNone(runtime.load_managed_trade_state("BTCUSDT"))
            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.state, "READY")

    def test_server_serves_html_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = _settings_with_temp_runtime(Path(tmp_dir))
            runtime = TestnetExecutionRuntime(None, settings=settings)
            runtime.update_runtime_status(symbol="BTCUSDT", state="READY")
            runtime.persist_exchange_income_records(
                [
                    {
                        "symbol": "BTCUSDT",
                        "incomeType": "REALIZED_PNL",
                        "income": "1.25",
                        "asset": "USDT",
                        "time": 2_000,
                        "tranId": "1",
                    },
                    {
                        "symbol": "BTCUSDT",
                        "incomeType": "COMMISSION",
                        "income": "-0.10",
                        "asset": "USDT",
                        "time": 2_001,
                        "tranId": "2",
                    },
                ],
                synced_at_ms=2_010,
                related_trade_id="trade-1",
            )

            controller = DashboardController(settings=settings, symbol="BTCUSDT")
            server = serve_dashboard(
                host="127.0.0.1",
                port=0,
                symbol="BTCUSDT",
                controller=controller,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                html = urlopen(base_url, timeout=5).read().decode("utf-8")
                self.assertIn("자동매매 허브", html)
                self.assertIn("USDT 시드 추이", html)
                self.assertIn("누적 성과 요약", html)
                self.assertIn("Binance 실제 income", html)
                self.assertIn("최근 거래 추정손익", html)
                self.assertIn("symbolOverview", html)

                service_status = json.loads(
                    urlopen(f"{base_url}/api/service-status", timeout=5).read().decode("utf-8")
                )
                self.assertEqual(service_status["status"], "ok")
                self.assertIn("services", service_status["data"])

                equity_history = json.loads(
                    urlopen(f"{base_url}/api/account-equity-history", timeout=5).read().decode("utf-8")
                )
                self.assertEqual(equity_history["status"], "ok")
                self.assertIn("summary", equity_history["data"])

                income_summary = json.loads(
                    urlopen(f"{base_url}/api/account-income-summary?hours=0", timeout=5).read().decode("utf-8")
                )
                self.assertEqual(income_summary["status"], "ok")
                self.assertEqual(income_summary["data"]["source"], "stored_exchange_income_records")
                self.assertAlmostEqual(
                    income_summary["data"]["summary"]["trading_net_income_usdt"],
                    1.15,
                )

                req = Request(
                    f"{base_url}/api/action",
                    data=json.dumps({"action": "manual_pause", "symbol": "BTCUSDT"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                payload = json.loads(urlopen(req, timeout=5).read().decode("utf-8"))
                self.assertEqual(payload["status"], "ok")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_mainnet_controller_uses_mainnet_runtime_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = _settings_with_temp_runtime(Path(tmp_dir))
            runtime = TestnetExecutionRuntime(
                None,
                settings=settings,
                database_path=Path(tmp_dir) / "runtime" / "execution" / "mainnet_execution.sqlite3",
            )
            runtime.update_runtime_status(symbol="BTCUSDT", state="READY")

            controller = DashboardController(
                settings=settings,
                symbol="BTCUSDT",
                environment="mainnet",
            )
            summary = controller.summary(symbol="BTCUSDT", incident_limit=5)

            self.assertEqual(summary["environment"], "mainnet")
            self.assertEqual(summary["summary"]["runtime_status"]["state"], "READY")
            self.assertTrue(
                summary["summary"]["database_path"].endswith("mainnet_execution.sqlite3")
            )

    def test_mainnet_controller_supports_priority_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = _settings_with_temp_runtime(Path(tmp_dir))
            runtime = TestnetExecutionRuntime(
                None,
                settings=settings,
                database_path=Path(tmp_dir) / "runtime" / "execution" / "mainnet_execution.sqlite3",
            )
            for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                runtime.update_runtime_status(symbol=symbol, state="READY")
            runtime.save_expected_state(
                ExpectedRuntimeState(
                    symbol="ETHUSDT",
                    expected_position_qty=0.073,
                    expected_stop_price_mark=2279.16,
                    expected_leverage=5,
                    expected_margin_mode="ISOLATED",
                    expected_position_mode="ONE_WAY",
                    updated_at_ms=123,
                )
            )

            controller = DashboardController(
                settings=settings,
                symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                environment="mainnet",
            )

            summary = controller.summary(symbol="SOLUSDT", incident_limit=5)

            self.assertEqual(summary["symbols"], ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            self.assertEqual(summary["summary"]["account_position_symbol"], "ETHUSDT")
            self.assertEqual(summary["summary"]["account_position_blocked_by"], "ETHUSDT")
            eth_summary = controller.summary(symbol="ETHUSDT", incident_limit=5)
            self.assertEqual(eth_summary["summary"]["account_position_symbol"], "ETHUSDT")
            self.assertIsNone(eth_summary["summary"]["account_position_blocked_by"])
            strategy_context = summary["summary"]["strategy_context"]
            self.assertEqual(strategy_context["strategy_mode"], "multi_symbol_priority_v1")
            self.assertIn("ETHUSDT", strategy_context["profiles_by_symbol"])
            self.assertIn("SOLUSDT", strategy_context["profiles_by_symbol"])
            self.assertEqual(
                strategy_context["runtime_stream_interval_by_symbol"]["BTCUSDT"],
                "5m",
            )
            self.assertEqual(
                strategy_context["runtime_stream_interval_by_symbol"]["SOLUSDT"],
                "15m",
            )


if __name__ == "__main__":
    unittest.main()
