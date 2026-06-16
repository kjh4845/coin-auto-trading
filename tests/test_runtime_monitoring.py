from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import sqlite3
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs
from unittest.mock import patch

from ai_auto_trading.ai.inference import AIDecision
from ai_auto_trading.execution.testnet import BinanceFuturesTestnetClient, TestnetExecutionEngine
from ai_auto_trading.models import OrderIntent, PolicyVersionInfo, PositionState, TradeRecord
from ai_auto_trading.runtime.kline_stream import KlineCloseEvent
from ai_auto_trading.runtime.mark_price_stream import MarkPriceEvent
from ai_auto_trading.runtime.orchestrator import TestnetRuntimeOrchestrator
from ai_auto_trading.runtime.risk_manager import RuntimeRiskConfig, TestnetRiskManager
from ai_auto_trading.runtime.testnet import TestnetExecutionRuntime
from ai_auto_trading.runtime.user_stream import TestnetUserStreamService
from ai_auto_trading.settings import load_settings
from ai_auto_trading.strategy.rule_based import SignalDecision


class MonitoringTransport:
    def __init__(self) -> None:
        self.listen_key = "listen-key-1"
        self.position_mode = False
        self.symbol_config = {
            "symbol": "BTCUSDT",
            "marginType": "ISOLATED",
            "leverage": "2",
        }
        self.positions: list[dict[str, object]] = []
        self.open_orders: list[dict[str, object]] = []
        self.open_algo_orders: list[dict[str, object]] = []
        self.klines: list[dict[str, float | int]] = []
        self.available_balance = "100000.0"
        self.income_history: list[dict[str, object]] = [
            {
                "symbol": "BTCUSDT",
                "incomeType": "REALIZED_PNL",
                "income": "1.25",
                "asset": "USDT",
                "time": 1234567890,
                "tranId": 1,
                "tradeId": "income-1",
            },
            {
                "symbol": "BTCUSDT",
                "incomeType": "COMMISSION",
                "income": "-0.10",
                "asset": "USDT",
                "time": 1234567890,
                "tranId": 2,
                "tradeId": "income-2",
            },
        ]

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ):
        decoded_body = body.decode("utf-8") if body is not None else ""
        if url.endswith("/fapi/v1/time"):
            return {"serverTime": 1234567890}
        if "/fapi/v1/listenKey" in url:
            if method == "POST":
                return {"listenKey": self.listen_key}
            return {"listenKey": self.listen_key}
        if "/fapi/v1/exchangeInfo" in url:
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                            {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "notional": "100"},
                        ],
                    }
                ]
            }
        if "/fapi/v1/premiumIndex" in url:
            return {
                "symbol": "BTCUSDT",
                "markPrice": "100000.0",
                "lastFundingRate": "0.0",
            }
        if "/fapi/v1/klines" in url:
            if self.klines:
                return [
                    [
                        row["open_time"],
                        str(row["open"]),
                        str(row["high"]),
                        str(row["low"]),
                        str(row["close"]),
                        str(row["volume"]),
                        row["close_time"],
                        "0",
                        0,
                        "0",
                        "0",
                        "0",
                    ]
                    for row in self.klines
                ]
            return []
        if "/fapi/v1/positionSide/dual" in url:
            if method == "GET":
                return {"dualSidePosition": self.position_mode}
            params = parse_qs(decoded_body or "")
            self.position_mode = params["dualSidePosition"][0] == "true"
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/symbolConfig" in url:
            return [dict(self.symbol_config)]
        if "/fapi/v3/positionRisk" in url:
            return list(self.positions)
        if "/fapi/v3/account" in url:
            return {"availableBalance": self.available_balance}
        if "/fapi/v1/income" in url:
            return list(self.income_history)
        if "/fapi/v1/openOrders" in url:
            return list(self.open_orders)
        if "/fapi/v1/openAlgoOrders" in url:
            return list(self.open_algo_orders)
        if "/fapi/v1/order/test" in url:
            return {}
        if "/fapi/v1/algoOpenOrders" in url and method == "DELETE":
            self.open_algo_orders = []
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/allOpenOrders" in url and method == "DELETE":
            self.open_orders = []
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/leverage" in url or "/fapi/v1/marginType" in url:
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/order" in url:
            params = parse_qs(decoded_body or "")
            if params.get("reduceOnly", ["false"])[0].lower() == "true":
                self.positions = []
            return {
                "status": "FILLED",
                "symbol": "BTCUSDT",
                "executedQty": params.get("quantity", ["0.01"])[0],
            }
        if "/fapi/v1/algoOrder" in url:
            params = parse_qs(decoded_body or "")
            algo_order = {
                "symbol": "BTCUSDT",
                "orderType": params["type"][0],
                "workingType": params["workingType"][0],
                "triggerPrice": params["triggerPrice"][0],
                "algoStatus": "NEW",
            }
            self.open_algo_orders = [algo_order]
            return algo_order
        raise AssertionError(f"unexpected url {url}")


class RuntimeMonitoringTest(unittest.TestCase):
    class _Assistant:
        model_base = "google/gemma-4-E2B-it"

        def __init__(self, decision: AIDecision) -> None:
            self._decision = decision

        def review_entry(self, **_: object) -> AIDecision:
            return self._decision

    def _stack(self, db_path: Path):
        settings = replace(
            load_settings(),
            strategy_mode="single_profile",
            ai_gate_enabled=False,
            allow_long_entries=True,
            allow_short_entries=True,
            micro_timeframe="3m",
            regime_timeframe="1h",
            anchor_timeframe="1h",
            max_long_funding_rate=1.0,
            min_short_funding_rate=-1.0,
            min_long_taker_buy_ratio=0.0,
            max_short_taker_buy_ratio=1.0,
            min_micro_long_ema_spread_pct=0.0,
            min_micro_short_ema_spread_pct=0.0,
            min_higher_tf_ema_spread_pct=0.0,
            min_volume_ratio_20=0.0,
        )
        client = BinanceFuturesTestnetClient(
            api_key="test-key",
            api_secret="test-secret",
            settings=settings,
            transport=MonitoringTransport(),
        )
        engine = TestnetExecutionEngine(client)
        runtime = TestnetExecutionRuntime(engine, settings=settings, database_path=db_path)
        risk_manager = TestnetRiskManager(
            RuntimeRiskConfig(
                user_stream_stale_ms=1000,
                mark_price_stale_ms=1000,
                api_error_window_ms=60000,
                max_error_incidents_in_window=2,
            )
        )
        orchestrator = TestnetRuntimeOrchestrator(engine, runtime, risk_manager)
        user_stream = TestnetUserStreamService(client, runtime)
        return client, engine, runtime, orchestrator, user_stream

    def _persist_trade_record(
        self,
        runtime: TestnetExecutionRuntime,
        *,
        trade_id: str,
        opened_at_ms: int,
        closed_at_ms: int,
        pnl_usdt: float,
        exit_reason: str = "HARD_STOP_MARK_PRICE",
        symbol: str = "BTCUSDT",
    ) -> None:
        position = PositionState(
            side="LONG",
            quantity=1.0,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100.0,
            entry_mark_price=100.0,
            symbol=symbol,
        )
        record = TradeRecord.from_closed_position(
            trade_id=trade_id,
            opened_at=datetime.fromtimestamp(opened_at_ms / 1000.0, tz=timezone.utc),
            closed_at=datetime.fromtimestamp(closed_at_ms / 1000.0, tz=timezone.utc),
            position=position,
            policy=PolicyVersionInfo(
                policy_version="policy_v1",
                strategy_version="strategy_v1",
                feature_schema_version="features_v1",
                runtime_version="testnet_runtime_v1",
            ),
            exit_reason=exit_reason,
            exit_contract_price_avg=100.0 + pnl_usdt,
            exit_mark_price=100.0 + pnl_usdt,
            max_favorable_excursion_usdt=max(0.0, pnl_usdt),
            max_adverse_excursion_usdt=abs(min(0.0, pnl_usdt)),
            signal_reason_codes=["long_trend_alignment"],
            ai_snapshot={"entry_action": "allow"},
        )
        runtime.persist_closed_trade_record(trade_record=record, closed_at_ms=closed_at_ms)

    def test_user_stream_lifecycle_and_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, engine, runtime, _, user_stream = self._stack(db_path)
            session = user_stream.start_session(symbol="BTCUSDT", timestamp=engine.server_timestamp())
            self.assertEqual(session.listen_key, "listen-key-1")
            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.active_listen_key, "listen-key-1")

            user_stream.keepalive_session(symbol="BTCUSDT", timestamp=1234567899)
            user_stream.ingest_payload(
                symbol="BTCUSDT",
                payload={
                    "e": "ORDER_TRADE_UPDATE",
                    "E": 1234567900,
                    "o": {
                        "i": 1,
                        "c": "cid",
                        "S": "BUY",
                        "o": "MARKET",
                        "X": "FILLED",
                        "R": False,
                        "q": "0.01",
                        "p": "0",
                        "sp": "0",
                    },
                },
            )
            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.last_user_stream_event_ms, 1234567900)

            with sqlite3.connect(db_path) as conn:
                order_events = conn.execute("SELECT COUNT(*) FROM order_events").fetchone()[0]
            self.assertEqual(order_events, 1)

            user_stream.close_session(symbol="BTCUSDT")
            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertIsNone(status.active_listen_key)

    def test_monitor_once_acquires_missing_stop_lockout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, engine, runtime, orchestrator, _ = self._stack(db_path)
            runtime.save_expected_state(
                runtime.load_expected_state("BTCUSDT")
                or runtime.place_entry_and_protection(
                    entry_intent=OrderIntent(side="BUY", quantity=0.01, order_type="MARKET"),
                    hard_stop_intent=OrderIntent(
                        side="SELL",
                        quantity=0.01,
                        order_type="STOP_MARKET",
                        reduce_only=True,
                        working_type="MARK_PRICE",
                        stop_price=97500.0,
                    ),
                    timestamp=1234567890,
                    expected_leverage=2,
                    skip_preflight=True,
                ).expected_state
            )
            transport = client.transport  # type: ignore[assignment]
            transport.positions = [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.01",
                    "markPrice": "100000.0",
                    "unRealizedProfit": "0.0",
                }
            ]
            transport.open_algo_orders = []
            result = orchestrator.reconcile_once(symbol="BTCUSDT", timestamp=1234568890)
            self.assertEqual(result.runtime_state, "PAUSED")
            self.assertEqual(result.active_lockouts, ["manual_review_required"])
            self.assertEqual(result.safety_action, "system_failsafe_exit")
            self.assertIsNone(runtime.load_expected_state("BTCUSDT"))
            self.assertIsNone(runtime.load_managed_trade_state("BTCUSDT"))
            records = runtime.recent_trade_records("BTCUSDT", limit=1)
            self.assertEqual(records[0]["exit_reason"], "SYSTEM_FAILSAFE_EXIT")

    def test_runtime_summary_exposes_lockouts_and_incidents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, _, _ = self._stack(db_path)
            runtime.update_runtime_status(symbol="BTCUSDT", state="READY")
            runtime.acquire_lockout(
                symbol="BTCUSDT",
                code="api_error_threshold",
                reason="too many recent execution errors",
            )
            summary = runtime.runtime_summary("BTCUSDT", incident_limit=5)
            self.assertEqual(summary["runtime_status"]["state"], "READY")
            self.assertEqual(summary["active_lockouts"][0]["code"], "api_error_threshold")
            self.assertGreaterEqual(len(summary["recent_incidents"]), 1)

    def test_kline_no_message_watchdog_marks_error_and_lockout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, _ = self._stack(db_path)

            with self.assertRaisesRegex(RuntimeError, "execution kline websocket"):
                orchestrator.kline_stream_service._raise_no_message_watchdog(
                    symbol="BTCUSDT",
                    interval="5m",
                    elapsed_seconds=91.0,
                    threshold_seconds=90.0,
                )

            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.state, "ERROR")
            self.assertIn("execution kline websocket", status.last_error or "")
            self.assertIn("kline_stream_stale", [lockout.code for lockout in runtime.active_lockouts("BTCUSDT")])
            self.assertEqual(runtime.recent_incidents(level="ERROR", limit=1)[0].event_type, "kline_stream_stale")

    def test_mark_price_no_message_watchdog_marks_error_and_lockout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, _ = self._stack(db_path)

            with self.assertRaisesRegex(RuntimeError, "mark price websocket"):
                orchestrator.mark_price_stream_service._raise_no_message_watchdog(
                    symbol="BTCUSDT",
                    elapsed_seconds=91.0,
                    threshold_seconds=90.0,
                )

            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.state, "ERROR")
            self.assertIn("mark price websocket", status.last_error or "")
            self.assertIn("mark_price_stream_stale", [lockout.code for lockout in runtime.active_lockouts("BTCUSDT")])
            self.assertEqual(runtime.recent_incidents(level="ERROR", limit=1)[0].event_type, "mark_price_stream_stale")

    def test_monitor_watchdog_marks_error_when_reconcile_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, _ = self._stack(db_path)
            runtime.update_runtime_status(
                symbol="BTCUSDT",
                state="READY",
                last_reconcile_ms=1,
            )

            with self.assertRaisesRegex(RuntimeError, "runtime monitor tick is stale"):
                asyncio.run(
                    orchestrator._monitor_watchdog_loop(
                        symbol="BTCUSDT",
                        stale_after_seconds=0.001,
                        check_interval_seconds=0.001,
                        duration_seconds=None,
                    )
                )

            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.state, "ERROR")
            self.assertIn("runtime_monitor_stale", [lockout.code for lockout in runtime.active_lockouts("BTCUSDT")])
            self.assertEqual(runtime.recent_incidents(level="ERROR", limit=1)[0].event_type, "runtime_monitor_stale")

    def test_reconcile_once_releases_monitor_watchdog_lockouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, _ = self._stack(db_path)
            runtime.acquire_lockout(
                symbol="BTCUSDT",
                code="runtime_monitor_stale",
                reason="stale before successful monitor tick",
            )
            runtime.acquire_lockout(
                symbol="BTCUSDT",
                code="runtime_background_task_failed",
                reason="previous background task failed",
            )

            result = orchestrator.reconcile_once(symbol="BTCUSDT", timestamp=1234568890)

            self.assertEqual(result.runtime_state, "READY")
            self.assertNotIn("runtime_monitor_stale", [lockout.code for lockout in runtime.active_lockouts("BTCUSDT")])
            self.assertNotIn(
                "runtime_background_task_failed",
                [lockout.code for lockout in runtime.active_lockouts("BTCUSDT")],
            )

    def test_event_loop_exits_when_background_task_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, _ = self._stack(db_path)

            async def failing_stream(**_: object) -> dict[str, object]:
                raise RuntimeError("stream died")

            async def idle_stream(**_: object) -> dict[str, object]:
                await asyncio.sleep(60)
                return {}

            orchestrator.user_stream_service.run_stream = failing_stream  # type: ignore[method-assign]
            orchestrator.mark_price_stream_service.run_stream = idle_stream  # type: ignore[method-assign]
            orchestrator.kline_stream_service.run_stream = idle_stream  # type: ignore[method-assign]

            with self.assertRaisesRegex(RuntimeError, "user_stream failed"):
                asyncio.run(
                    orchestrator.run_event_driven_cycle(
                        symbol="BTCUSDT",
                        entry_notional_usdt=1000.0,
                        leverage=2,
                        duration_seconds=None,
                    )
                )

            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.state, "ERROR")
            self.assertIn("user_stream failed", status.last_error or "")
            self.assertIn(
                "runtime_background_task_failed",
                [lockout.code for lockout in runtime.active_lockouts("BTCUSDT")],
            )
            self.assertEqual(
                runtime.recent_incidents(level="ERROR", limit=1)[0].event_type,
                "runtime_background_task_failed",
            )

    def test_trade_performance_summary_aggregates_closed_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, _, _ = self._stack(db_path)
            self._persist_trade_record(
                runtime,
                trade_id="win-1",
                opened_at_ms=1_000,
                closed_at_ms=2_000,
                pnl_usdt=5.0,
            )
            self._persist_trade_record(
                runtime,
                trade_id="loss-1",
                opened_at_ms=3_000,
                closed_at_ms=4_000,
                pnl_usdt=-2.5,
                exit_reason="TIME_STOP",
            )
            self._persist_trade_record(
                runtime,
                trade_id="flat-1",
                opened_at_ms=5_000,
                closed_at_ms=6_000,
                pnl_usdt=0.0,
                exit_reason="OTHER",
            )

            performance = runtime.trade_performance_summary("BTCUSDT")
            self.assertEqual(performance["total_trade_count"], 3)
            self.assertEqual(performance["decisive_trade_count"], 2)
            self.assertEqual(performance["win_count"], 1)
            self.assertEqual(performance["loss_count"], 1)
            self.assertEqual(performance["flat_count"], 1)
            self.assertAlmostEqual(performance["win_rate_pct"], 50.0)
            self.assertAlmostEqual(performance["total_realized_pnl_after_fees_usdt"], 2.5)
            self.assertAlmostEqual(performance["total_realized_r"], 1.0)
            self.assertAlmostEqual(performance["profit_factor"], 2.0)
            self.assertEqual(performance["best_trade_id"], "win-1")
            self.assertEqual(performance["worst_trade_id"], "loss-1")

            summary = runtime.runtime_summary("BTCUSDT", incident_limit=5)
            self.assertEqual(summary["trade_performance_summary"]["total_trade_count"], 3)

            recent_performance = runtime.trade_performance_summary("BTCUSDT", since_ms=4_000)
            self.assertEqual(recent_performance["total_trade_count"], 2)
            self.assertEqual(recent_performance["win_count"], 0)
            self.assertEqual(recent_performance["loss_count"], 1)
            self.assertEqual(recent_performance["flat_count"], 1)
            self.assertAlmostEqual(
                recent_performance["total_realized_pnl_after_fees_usdt"],
                -2.5,
            )

            recent_summary = runtime.runtime_summary(
                "BTCUSDT",
                incident_limit=5,
                display_since_ms=4_000,
            )
            self.assertEqual(recent_summary["trade_performance_summary"]["total_trade_count"], 2)
            self.assertEqual(
                [record["trade_id"] for record in recent_summary["recent_trade_records"]],
                ["flat-1", "loss-1"],
            )

    def test_forward_trade_progress_reports_milestone_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, _, _ = self._stack(db_path)
            base_ms = 1_700_000_000_000
            self._persist_trade_record(
                runtime,
                trade_id="t1",
                opened_at_ms=base_ms - 60_000,
                closed_at_ms=base_ms,
                pnl_usdt=2.5,
            )
            self._persist_trade_record(
                runtime,
                trade_id="t2",
                opened_at_ms=base_ms + 86_400_000 - 60_000,
                closed_at_ms=base_ms + 86_400_000,
                pnl_usdt=-2.5,
            )

            progress = runtime.forward_trade_progress(
                "BTCUSDT",
                target_min_trades=3,
                target_max_trades=5,
                now_ms=base_ms + 2 * 86_400_000,
            )
            self.assertEqual(progress["closed_trade_count"], 2)
            self.assertEqual(progress["remaining_to_min"], 1)
            self.assertEqual(progress["remaining_to_max"], 3)
            self.assertFalse(progress["ready_for_sizing_review"])
            self.assertEqual(progress["trades_last_7d"], 2)

    def test_position_sizing_review_uses_forward_trade_r_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, _, _ = self._stack(db_path)
            base_ms = 1_700_000_000_000
            for index, pnl in enumerate((2.5, -2.5, 5.0, 2.5), start=1):
                closed_at_ms = base_ms + (index * 86_400_000)
                self._persist_trade_record(
                    runtime,
                    trade_id=f"t{index}",
                    opened_at_ms=closed_at_ms - 60_000,
                    closed_at_ms=closed_at_ms,
                    pnl_usdt=pnl,
                )

            review = runtime.position_sizing_review(
                "BTCUSDT",
                min_trade_count=4,
                target_trade_count=6,
                entry_notional_usdt=1000.0,
                leverage=5.0,
                account_equity_usdt=5000.0,
            )
            self.assertTrue(review["eligible"])
            self.assertEqual(review["trade_count"], 4)
            self.assertEqual(review["current_one_r_usdt"], 10.0)
            self.assertAlmostEqual(review["edge_stats"]["total_realized_r"], 3.0)
            self.assertGreater(review["edge_stats"]["kelly_fraction_of_equity"], 0.0)
            self.assertIn("balanced", review["candidate_position_sizes"])
            self.assertGreater(
                review["candidate_position_sizes"]["balanced"]["implied_entry_notional_usdt"],
                0.0,
            )

    def test_recently_started_user_stream_is_not_immediately_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, user_stream = self._stack(db_path)
            user_stream.start_session(symbol="BTCUSDT", timestamp=1_000_000)
            result = orchestrator.reconcile_once(symbol="BTCUSDT", timestamp=1_000_500)
            self.assertEqual(result.active_lockouts, [])
            self.assertEqual(result.runtime_state, "READY")

    def test_monitor_once_pauses_when_exchange_has_orphan_remote_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, _, orchestrator, _ = self._stack(db_path)
            transport = client.transport  # type: ignore[assignment]
            transport.positions = [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.01",
                    "markPrice": "100000.0",
                    "unRealizedProfit": "0.0",
                }
            ]
            result = orchestrator.reconcile_once(symbol="BTCUSDT", timestamp=1234568890)
            assert result.reconciliation is not None
            self.assertTrue(result.reconciliation.ok)
            self.assertEqual(result.reconciliation.mismatches, [])
            self.assertEqual(result.runtime_state, "PAUSED")
            self.assertEqual(result.active_lockouts, ["manual_review_required"])
            self.assertEqual(result.safety_action, "system_failsafe_exit")

    def test_manage_open_position_once_exits_and_clears_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            runtime.place_entry_and_protection(
                entry_intent=OrderIntent(side="BUY", quantity=0.01, order_type="MARKET"),
                hard_stop_intent=OrderIntent(
                    side="SELL",
                    quantity=0.01,
                    order_type="STOP_MARKET",
                    reduce_only=True,
                    working_type="MARK_PRICE",
                    stop_price=97500.0,
                ),
                timestamp=1234567890,
                expected_leverage=2,
                skip_preflight=True,
            )
            managed = runtime.load_managed_trade_state("BTCUSDT")
            assert managed is not None
            runtime.save_managed_trade_state(
                managed.__class__(
                    symbol=managed.symbol,
                    side=managed.side,
                    quantity=managed.quantity,
                    leverage_at_entry=managed.leverage_at_entry,
                    entry_contract_price_avg=managed.entry_contract_price_avg,
                    entry_mark_price=managed.entry_mark_price,
                    execution_timeframe=managed.execution_timeframe,
                    atr_trailing_multiplier=managed.atr_trailing_multiplier,
                    max_holding_bars=1,
                    opened_at_ms=managed.opened_at_ms,
                    signal_reason_codes=managed.signal_reason_codes,
                    model_base=managed.model_base,
                    adapter_version=managed.adapter_version,
                    ai_snapshot=managed.ai_snapshot,
                    bars_held=0,
                    highest_high=managed.highest_high,
                    lowest_low=managed.lowest_low,
                    last_processed_candle_close_time_ms=None,
                    atr_trail_history=[],
                )
            )
            transport = client.transport  # type: ignore[assignment]
            transport.positions = [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.01",
                    "markPrice": "100000.0",
                    "unRealizedProfit": "0.0",
                }
            ]
            base_time = 1234567890
            transport.klines = [
                {
                    "open_time": base_time + (i * 180000),
                    "open": 100.0,
                    "high": 100.2,
                    "low": 99.8,
                    "close": 100.0 if i == 29 else 100.1,
                    "volume": 10.0 + i,
                    "close_time": base_time + ((i + 1) * 180000) - 1,
                }
                for i in range(30)
            ]
            timestamp = int(transport.klines[-1]["close_time"]) + 1  # type: ignore[index]
            result = orchestrator.manage_open_position_once(
                symbol="BTCUSDT",
                timestamp=timestamp,
                candle_limit=30,
            )
            self.assertEqual(result.action, "EXIT")
            self.assertEqual(result.exit_reason, "TIME_STOP")
            self.assertIsNone(runtime.load_expected_state("BTCUSDT"))
            self.assertIsNone(runtime.load_managed_trade_state("BTCUSDT"))
            income_rows = runtime.exchange_income_records_since(since_ms=0, symbol="BTCUSDT")
            self.assertTrue(any(row["income_type"] == "REALIZED_PNL" for row in income_rows))
            self.assertEqual(income_rows[0]["related_trade_id"], runtime.recent_trade_records("BTCUSDT", limit=1)[0]["trade_id"])

    def test_attempt_rule_entry_once_places_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            transport = client.transport  # type: ignore[assignment]
            transport.klines = [
                {
                    "open_time": i * 180000,
                    "open": 100.0,
                    "high": 100.2 if i < 29 else 100.25,
                    "low": 99.95,
                    "close": 100.0 if i < 29 else 100.1,
                    "volume": 10.0 + i,
                    "close_time": (i + 1) * 180000 - 1,
                }
                for i in range(30)
            ]
            result = orchestrator.attempt_rule_entry_once(
                symbol="BTCUSDT",
                timestamp=1234567890,
                entry_notional_usdt=1000.0,
                leverage=2,
                candle_limit=30,
            )
            self.assertEqual(result.action, "PLACED")
            self.assertEqual(result.signal_action, "LONG")
            self.assertIsNotNone(runtime.load_expected_state("BTCUSDT"))
            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.active_listen_key, "listen-key-1")
            self.assertEqual(status.last_user_stream_event_ms, 1234567890)

    def test_attempt_rule_entry_once_refreshes_stale_user_stream_before_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            runtime.update_runtime_status(
                symbol="BTCUSDT",
                active_listen_key="listen-key-1",
                last_user_stream_event_ms=1234460000,
            )
            transport = client.transport  # type: ignore[assignment]
            transport.klines = [
                {
                    "open_time": i * 180000,
                    "open": 100.0,
                    "high": 100.2 if i < 29 else 100.25,
                    "low": 99.95,
                    "close": 100.0 if i < 29 else 100.1,
                    "volume": 10.0 + i,
                    "close_time": (i + 1) * 180000 - 1,
                }
                for i in range(30)
            ]
            result = orchestrator.attempt_rule_entry_once(
                symbol="BTCUSDT",
                timestamp=1234567890,
                entry_notional_usdt=1000.0,
                leverage=2,
                candle_limit=30,
            )
            self.assertEqual(result.action, "PLACED")
            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.active_listen_key, "listen-key-1")
            self.assertEqual(status.last_user_stream_event_ms, 1234567890)

    def test_attempt_rule_entry_once_skips_when_available_balance_is_too_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            transport = client.transport  # type: ignore[assignment]
            transport.available_balance = "50.0"
            transport.klines = [
                {
                    "open_time": i * 180000,
                    "open": 100.0,
                    "high": 100.2 if i < 29 else 100.25,
                    "low": 99.95,
                    "close": 100.0 if i < 29 else 100.1,
                    "volume": 10.0 + i,
                    "close_time": (i + 1) * 180000 - 1,
                }
                for i in range(30)
            ]

            result = orchestrator.attempt_rule_entry_once(
                symbol="BTCUSDT",
                timestamp=1234567890,
                entry_notional_usdt=200.0,
                leverage=2,
                candle_limit=30,
            )

            self.assertEqual(result.action, "SKIP_INSUFFICIENT_MARGIN")
            self.assertIn("blocked_insufficient_available_balance", result.reason_codes)
            self.assertIsNone(runtime.load_expected_state("BTCUSDT"))
            incidents = runtime.recent_incidents(limit=1)
            self.assertEqual(incidents[0].event_type, "signal_entry_skipped_insufficient_margin")

    def test_attempt_rule_entry_once_best_pair_uses_selected_profile_timeframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, engine, runtime, orchestrator, _ = self._stack(db_path)
            runtime.settings = replace(runtime.settings, strategy_mode="best_pair_v1")
            client.settings = runtime.settings
            engine.client.settings = runtime.settings
            candle_map = {
                timeframe: [
                    {
                        "open_time": 0,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 101.0 if timeframe == "30m" else 100.0,
                        "volume": 10.0,
                        "close_time": 1_800_000 - 1,
                        "taker_buy_base_asset_volume": 5.0,
                    }
                ]
                for timeframe in ["5m", "15m", "1h", "4h", "1d", "30m"]
            }

            def _decision(_context, *, params, **_kwargs):
                if params.execution_timeframe == "30m":
                    return SignalDecision(action="LONG", reason_codes=["long_profile_hit"])
                return SignalDecision(action="NO_TRADE", reason_codes=["blocked_no_valid_setup"])

            with patch.object(orchestrator, "_closed_candles_by_timeframe", return_value=candle_map):
                with patch("ai_auto_trading.runtime.orchestrator.evaluate_rule_signal", side_effect=_decision):
                    result = orchestrator.attempt_rule_entry_once(
                        symbol="BTCUSDT",
                        timestamp=1_800_000 - 1,
                        entry_notional_usdt=1000.0,
                        leverage=2,
                        candle_limit=1,
                    )

            self.assertEqual(result.action, "PLACED")
            self.assertEqual(result.signal_action, "LONG")
            managed = runtime.load_managed_trade_state("BTCUSDT")
            assert managed is not None
            self.assertEqual(managed.execution_timeframe, "30m")
            self.assertIn("profile:long_best_30m", managed.signal_reason_codes)

    def test_attempt_rule_entry_once_rejects_leverage_above_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            transport = client.transport  # type: ignore[assignment]
            transport.klines = [
                {
                    "open_time": i * 180000,
                    "open": 100.0,
                    "high": 100.2 if i < 29 else 100.25,
                    "low": 99.95,
                    "close": 100.0 if i < 29 else 100.1,
                    "volume": 10.0 + i,
                    "close_time": (i + 1) * 180000 - 1,
                }
                for i in range(30)
            ]
            with self.assertRaisesRegex(ValueError, "system cap"):
                orchestrator.attempt_rule_entry_once(
                    symbol="BTCUSDT",
                    timestamp=1234567890,
                    entry_notional_usdt=1000.0,
                    leverage=11,
                    candle_limit=30,
                )
            self.assertIsNone(runtime.load_expected_state("BTCUSDT"))

    def test_attempt_rule_entry_once_respects_ai_reduce_size_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, engine, runtime, _, _ = self._stack(db_path)
            orchestrator = TestnetRuntimeOrchestrator(
                engine,
                runtime,
                ai_trade_assistant=self._Assistant(
                    AIDecision(
                        regime="trend_up",
                        setup_quality=0.9,
                        entry_action="reduce_size",
                        exit_action="hold",
                        confidence=0.8,
                        reason_codes=["thin_confirmation"],
                    )
                ),
            )
            transport = client.transport  # type: ignore[assignment]
            transport.klines = [
                {
                    "open_time": i * 180000,
                    "open": 100.0,
                    "high": 100.2 if i < 29 else 100.25,
                    "low": 99.95,
                    "close": 100.0 if i < 29 else 100.1,
                    "volume": 10.0 + i,
                    "close_time": (i + 1) * 180000 - 1,
                }
                for i in range(30)
            ]
            result = orchestrator.attempt_rule_entry_once(
                symbol="BTCUSDT",
                timestamp=1234567890,
                entry_notional_usdt=1000.0,
                leverage=2,
                candle_limit=30,
            )
            self.assertEqual(result.action, "PLACED")
            managed = runtime.load_managed_trade_state("BTCUSDT")
            assert managed is not None
            self.assertEqual(managed.model_base, "google/gemma-4-E2B-it")
            self.assertIsNotNone(managed.ai_snapshot)
            self.assertAlmostEqual(managed.quantity, 4.995, places=3)

    def test_handle_execution_kline_close_triggers_entry_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            transport = client.transport  # type: ignore[assignment]
            transport.klines = [
                {
                    "open_time": i * 180000,
                    "open": 100.0,
                    "high": 100.2 if i < 29 else 100.25,
                    "low": 99.95,
                    "close": 100.0 if i < 29 else 100.1,
                    "volume": 10.0 + i,
                    "close_time": (i + 1) * 180000 - 1,
                }
                for i in range(30)
            ]
            result = asyncio.run(
                orchestrator.handle_execution_kline_close(
                    event=KlineCloseEvent(
                        symbol="BTCUSDT",
                        interval="3m",
                        open_time=transport.klines[-1]["open_time"],  # type: ignore[index]
                        close_time=transport.klines[-1]["close_time"],  # type: ignore[index]
                        open_price=100.0,
                        high_price=100.25,
                        low_price=99.95,
                        close_price=100.1,
                        volume=39.0,
                    ),
                    entry_notional_usdt=1000.0,
                    leverage=2,
                    candle_limit=30,
                )
            )
            self.assertEqual(result["trigger"], "execution_kline_close")
            self.assertEqual(result["result"]["action"], "PLACED")

    def test_handle_mark_price_tick_exits_on_hard_stop_breach(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            runtime.place_entry_and_protection(
                entry_intent=OrderIntent(side="BUY", quantity=0.01, order_type="MARKET"),
                hard_stop_intent=OrderIntent(
                    side="SELL",
                    quantity=0.01,
                    order_type="STOP_MARKET",
                    reduce_only=True,
                    working_type="MARK_PRICE",
                    stop_price=97500.0,
                ),
                timestamp=1234567890,
                expected_leverage=2,
                skip_preflight=True,
            )
            transport = client.transport  # type: ignore[assignment]
            transport.positions = [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.01",
                    "markPrice": "97000.0",
                    "unRealizedProfit": "-30.0",
                }
            ]
            result = asyncio.run(
                orchestrator.handle_mark_price_tick(
                    event=MarkPriceEvent(
                        symbol="BTCUSDT",
                        event_time_ms=1234568890,
                        mark_price=97000.0,
                        index_price=97010.0,
                    )
                )
            )
            assert result is not None
            self.assertEqual(result["result"]["exit_reason"], "HARD_STOP_MARK_PRICE")
            self.assertIsNone(runtime.load_expected_state("BTCUSDT"))
            self.assertIsNone(runtime.load_managed_trade_state("BTCUSDT"))

    def test_monitor_once_flattens_when_mark_price_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            runtime.place_entry_and_protection(
                entry_intent=OrderIntent(side="BUY", quantity=0.01, order_type="MARKET"),
                hard_stop_intent=OrderIntent(
                    side="SELL",
                    quantity=0.01,
                    order_type="STOP_MARKET",
                    reduce_only=True,
                    working_type="MARK_PRICE",
                    stop_price=97500.0,
                ),
                timestamp=1234567890,
                expected_leverage=2,
                skip_preflight=True,
            )
            runtime.update_runtime_status(symbol="BTCUSDT", last_mark_price_event_ms=None)
            transport = client.transport  # type: ignore[assignment]
            transport.positions = [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.01",
                    "markPrice": "100000.0",
                    "unRealizedProfit": "0.0",
                }
            ]
            result = orchestrator.reconcile_once(symbol="BTCUSDT", timestamp=1234568890)
            self.assertEqual(result.runtime_state, "PAUSED")
            self.assertEqual(result.active_lockouts, ["manual_review_required"])
            self.assertEqual(result.safety_action, "system_failsafe_exit")

    def test_monitor_once_acquires_account_risk_lockouts_after_recent_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, _ = self._stack(db_path)
            base_time = 1_234_567_890
            self._persist_trade_record(
                runtime,
                trade_id="loss-1",
                opened_at_ms=base_time - 60_000,
                closed_at_ms=base_time - 30_000,
                pnl_usdt=-2.5,
            )
            self._persist_trade_record(
                runtime,
                trade_id="loss-2",
                opened_at_ms=base_time - 180_000,
                closed_at_ms=base_time - 120_000,
                pnl_usdt=-2.5,
            )
            self._persist_trade_record(
                runtime,
                trade_id="loss-3",
                opened_at_ms=base_time - 300_000,
                closed_at_ms=base_time - 240_000,
                pnl_usdt=-2.5,
            )
            result = orchestrator.reconcile_once(symbol="BTCUSDT", timestamp=base_time)
            self.assertEqual(result.runtime_state, "PAUSED")
            self.assertIn("daily_loss_limit_breached", result.active_lockouts)
            self.assertIn("consecutive_loss_limit", result.active_lockouts)
            self.assertIn("recent_loss_cooldown", result.active_lockouts)
            overview = runtime.account_risk_overview("BTCUSDT", now_ms=base_time)
            self.assertAlmostEqual(overview["recent_24h_realized_r"], -3.0)
            self.assertEqual(overview["max_daily_loss_r"], 3.0)
            self.assertEqual(overview["scope"], "account")

    def test_account_risk_counts_cross_symbol_consecutive_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, _ = self._stack(db_path)
            base_time = 1_234_567_890
            for index, symbol in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
                self._persist_trade_record(
                    runtime,
                    trade_id=f"cross-loss-{index}",
                    opened_at_ms=base_time - 90_000 + index * 20_000,
                    closed_at_ms=base_time - 80_000 + index * 20_000,
                    pnl_usdt=-2.5,
                    symbol=symbol,
                )

            overview = runtime.account_risk_overview("BTCUSDT", now_ms=base_time)
            self.assertEqual(overview["consecutive_losses"], 3)
            self.assertAlmostEqual(overview["recent_24h_realized_r"], -3.0)

            result = orchestrator.reconcile_once(symbol="BTCUSDT", timestamp=base_time)

            self.assertEqual(result.runtime_state, "PAUSED")
            self.assertIn("consecutive_loss_limit", result.active_lockouts)

    def test_latest_runtime_trigger_updates_kline_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            transport = client.transport  # type: ignore[assignment]
            transport.klines = [
                {
                    "open_time": 0,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 10.0,
                    "close_time": 299_999,
                },
                {
                    "open_time": 300_000,
                    "open": 100.5,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.5,
                    "volume": 11.0,
                    "close_time": 599_999,
                },
            ]

            close_time = orchestrator.latest_runtime_trigger_close_time(
                symbol="BTCUSDT",
                timestamp=600_000,
            )

            self.assertEqual(close_time, 599_999)
            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.last_execution_kline_close_ms, 599_999)

    def test_closed_trade_record_persists_trade_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, _, _ = self._stack(db_path)
            self._persist_trade_record(
                runtime,
                trade_id="review-loss",
                opened_at_ms=1_000,
                closed_at_ms=2_000,
                pnl_usdt=-5.0,
                exit_reason="TIME_STOP",
            )
            reviews = runtime.recent_trade_reviews("BTCUSDT", limit=1)
            self.assertEqual(reviews[0]["trade_id"], "review-loss")
            self.assertEqual(reviews[0]["primary_cause"], "no_follow_through")
            self.assertIn("continuation", reviews[0]["explanation"])
            self.assertEqual(reviews[0]["review_version"], "trade_review_v2")
            self.assertEqual(reviews[0]["rule_change_candidates"][0]["scope"], "entry_filter")
            self.assertFalse(reviews[0]["rule_change_candidates"][0]["auto_apply"])
            self.assertEqual(reviews[0]["handling_decision"]["action"], "cooldown_and_monitor_pattern")
            self.assertFalse(reviews[0]["handling_decision"]["auto_apply_rule_changes"])

    def test_exchange_income_records_are_persisted_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, _, _ = self._stack(db_path)
            rows = [
                {
                    "symbol": "ETHUSDT",
                    "incomeType": "REALIZED_PNL",
                    "income": "0.20",
                    "asset": "USDT",
                    "time": 2_000,
                    "tranId": "1",
                    "tradeId": "10",
                },
                {
                    "symbol": "ETHUSDT",
                    "incomeType": "COMMISSION",
                    "income": "-0.05",
                    "asset": "USDT",
                    "time": 2_001,
                    "tranId": "2",
                    "tradeId": "11",
                },
            ]

            first = runtime.persist_exchange_income_records(
                rows,
                synced_at_ms=2_010,
                related_trade_id="local-trade-1",
            )
            second = runtime.persist_exchange_income_records(
                rows,
                synced_at_ms=2_020,
                related_trade_id="local-trade-1",
            )
            stored = runtime.exchange_income_records_since(since_ms=0)

            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 2)
            self.assertEqual(len(stored), 2)
            self.assertEqual(stored[0]["symbol"], "ETHUSDT")
            self.assertEqual(stored[0]["income_type"], "REALIZED_PNL")
            self.assertEqual(stored[0]["related_trade_id"], "local-trade-1")
            self.assertAlmostEqual(sum(row["income_usdt"] for row in stored), 0.15)

    def test_repeated_loss_pattern_acquires_review_lockout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, _, _ = self._stack(db_path)
            for index in range(3):
                self._persist_trade_record(
                    runtime,
                    trade_id=f"repeat-loss-{index}",
                    opened_at_ms=1_000 + index * 10_000,
                    closed_at_ms=2_000 + index * 10_000,
                    pnl_usdt=-5.0,
                    exit_reason="TIME_STOP",
                )

            lockouts = runtime.active_lockouts("BTCUSDT")
            self.assertIn("repeated_loss_pattern_review", [lockout.code for lockout in lockouts])
            status = runtime.load_runtime_status("BTCUSDT")
            assert status is not None
            self.assertEqual(status.state, "PAUSED")
            self.assertIn("no_follow_through", status.last_error or "")

    def test_attempt_rule_entry_once_ignores_future_confirmation_candles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            client, _, runtime, orchestrator, _ = self._stack(db_path)
            transport = client.transport  # type: ignore[assignment]
            base = [
                {
                    "open_time": i * 180000,
                    "open": 100.0,
                    "high": 100.2 if i < 29 else 100.25,
                    "low": 99.95,
                    "close": 100.0 if i < 29 else 100.1,
                    "volume": 10.0 + i,
                    "close_time": (i + 1) * 180000 - 1,
                }
                for i in range(30)
            ]
            future = {
                "open_time": 99999999,
                "open": 90.0,
                "high": 90.1,
                "low": 89.9,
                "close": 90.0,
                "volume": 1.0,
                "close_time": 99999999 + 180000,
            }
            transport.klines = base + [future]
            result = orchestrator.attempt_rule_entry_once(
                symbol="BTCUSDT",
                timestamp=base[-1]["close_time"],
                entry_notional_usdt=1000.0,
                leverage=2,
                candle_limit=31,
            )
            self.assertEqual(result.action, "PLACED")

    def test_manage_open_position_once_persists_exchange_side_flat_trade_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, user_stream = self._stack(db_path)
            runtime.place_entry_and_protection(
                entry_intent=OrderIntent(side="BUY", quantity=0.01, order_type="MARKET"),
                hard_stop_intent=OrderIntent(
                    side="SELL",
                    quantity=0.01,
                    order_type="STOP_MARKET",
                    reduce_only=True,
                    working_type="MARK_PRICE",
                    stop_price=97500.0,
                ),
                timestamp=1234567890,
                expected_leverage=2,
                skip_preflight=True,
            )
            user_stream.ingest_payload(
                symbol="BTCUSDT",
                payload={
                    "e": "ORDER_TRADE_UPDATE",
                    "E": 1234568890,
                    "o": {
                        "i": 1,
                        "c": "cid",
                        "S": "SELL",
                        "o": "STOP_MARKET",
                        "X": "FILLED",
                        "R": True,
                        "q": "0.01",
                        "p": "0",
                        "sp": "97500",
                        "ap": "97500",
                    },
                },
            )
            result = orchestrator.manage_open_position_once(
                symbol="BTCUSDT",
                timestamp=1234569999,
                candle_limit=30,
            )
            self.assertEqual(result.action, "NO_POSITION")
            records = runtime.recent_trade_records("BTCUSDT", limit=1)
            self.assertEqual(records[0]["exit_reason"], "HARD_STOP_MARK_PRICE")

    def test_manage_open_position_once_clears_stale_local_state_when_exchange_is_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            _, _, runtime, orchestrator, _ = self._stack(db_path)
            runtime.save_expected_state(
                runtime.place_entry_and_protection(
                    entry_intent=OrderIntent(side="BUY", quantity=0.01, order_type="MARKET"),
                    hard_stop_intent=OrderIntent(
                        side="SELL",
                        quantity=0.01,
                        order_type="STOP_MARKET",
                        reduce_only=True,
                        working_type="MARK_PRICE",
                        stop_price=97500.0,
                    ),
                    timestamp=1234567890,
                    expected_leverage=2,
                    skip_preflight=True,
                ).expected_state
            )
            result = orchestrator.manage_open_position_once(
                symbol="BTCUSDT",
                timestamp=1234569999,
                candle_limit=30,
            )
            self.assertEqual(result.action, "NO_POSITION")
            self.assertIsNone(runtime.load_expected_state("BTCUSDT"))
            self.assertIsNone(runtime.load_managed_trade_state("BTCUSDT"))


if __name__ == "__main__":
    unittest.main()
