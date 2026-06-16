from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ai_auto_trading.ai.inference import AIDecision
from ai_auto_trading.backtest.replay import (
    BacktestConfig,
    load_kline_parquet,
    resample_klines,
    run_hybrid_backtest,
    write_decision_report_json,
    write_trade_logs_jsonl,
)
from ai_auto_trading.strategy.rule_based import RuleStrategyParameters


def _contract_candle(index: int, close: float, step_ms: int = 180000) -> dict[str, float | int]:
    open_price = close - 0.01
    return {
        "open_time": index * step_ms,
        "open": open_price,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": 10.0 + index,
        "close_time": (index + 1) * step_ms - 1,
    }


def _mark_candle(
    open_time: int,
    close_time: int,
    *,
    open_price: float,
    high: float,
    low: float,
    close_price: float,
) -> dict[str, float | int]:
    return {
        "open_time": open_time,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close_price,
        "volume": 0.0,
        "close_time": close_time,
    }


class BacktestTest(unittest.TestCase):
    class _Assistant:
        model_base = "google/gemma-4-E2B-it"

        def __init__(self, decision: AIDecision) -> None:
            self._decision = decision

        def review_entry(self, **_: object) -> AIDecision:
            return self._decision

    def _build_contract_data(self) -> dict[str, list[dict[str, float | int]]]:
        base_3m = [_contract_candle(i, 100.0) for i in range(32)]
        base_15m = [_contract_candle(i, 100.0) for i in range(32)]
        base_1h = [_contract_candle(i, 100.0) for i in range(32)]
        base_3m[30]["close"] = 100.10
        base_3m[30]["open"] = 100.00
        base_3m[30]["high"] = 100.20
        base_3m[30]["low"] = 99.95
        base_15m[30]["close"] = 100.12
        base_15m[30]["open"] = 100.00
        base_15m[30]["high"] = 100.22
        base_15m[30]["low"] = 99.95
        base_1h[30]["close"] = 100.15
        base_1h[30]["open"] = 100.00
        base_1h[30]["high"] = 100.25
        base_1h[30]["low"] = 99.95
        # after signal, entry happens on candle 31 open and hard stop should breach intrabar
        base_3m[31]["open"] = 100.61
        base_3m[31]["high"] = 100.80
        base_3m[31]["low"] = 98.50
        base_3m[31]["close"] = 99.00
        return {"3m": base_3m, "15m": base_15m, "1h": base_1h}

    def _build_mark_data(self, execution_candles: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for candle in execution_candles[:-1]:
            rows.append(
                _mark_candle(
                    int(candle["open_time"]),
                    int(candle["close_time"]),
                    open_price=float(candle["open"]),
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    close_price=float(candle["close"]),
                )
            )
        last = execution_candles[-1]
        rows.extend(
            [
                _mark_candle(
                    int(last["open_time"]),
                    int(last["open_time"]) + 59999,
                    open_price=100.61,
                    high=100.70,
                    low=99.80,
                    close_price=100.00,
                ),
                _mark_candle(
                    int(last["open_time"]) + 60000,
                    int(last["open_time"]) + 119999,
                    open_price=100.00,
                    high=100.10,
                    low=97.90,
                    close_price=98.20,
                ),
            ]
        )
        return rows

    def _build_generic_mark_data(
        self, execution_candles: list[dict[str, float | int]]
    ) -> list[dict[str, float | int]]:
        return [
            _mark_candle(
                int(candle["open_time"]),
                int(candle["close_time"]),
                open_price=float(candle["open"]),
                high=float(candle["high"]),
                low=float(candle["low"]),
                close_price=float(candle["close"]),
            )
            for candle in execution_candles
        ]

    def _build_early_scratch_contract_data(self) -> dict[str, list[dict[str, float | int]]]:
        base_3m = [_contract_candle(i, 100.0) for i in range(35)]
        base_15m = [_contract_candle(i, 100.0) for i in range(35)]
        base_1h = [_contract_candle(i, 100.0) for i in range(35)]
        for rows in (base_3m, base_15m, base_1h):
            rows[30]["close"] = 100.10
            rows[30]["open"] = 100.00
            rows[30]["high"] = 100.20
            rows[30]["low"] = 99.95
        base_3m[31]["open"] = 100.50
        base_3m[31]["high"] = 100.55
        base_3m[31]["low"] = 100.20
        base_3m[31]["close"] = 100.25
        base_3m[32]["open"] = 100.20
        base_3m[32]["high"] = 100.24
        base_3m[32]["low"] = 100.05
        base_3m[32]["close"] = 100.10
        base_3m[33]["open"] = 100.10
        base_3m[33]["high"] = 100.12
        base_3m[33]["low"] = 99.95
        base_3m[33]["close"] = 100.00
        base_3m[34]["open"] = 100.00
        base_3m[34]["high"] = 100.02
        base_3m[34]["low"] = 99.90
        base_3m[34]["close"] = 99.98
        return {"3m": base_3m, "15m": base_15m, "1h": base_1h}

    def _build_exit_policy_contract_data(self) -> dict[str, list[dict[str, float | int]]]:
        base_3m = [_contract_candle(i, 100.0) for i in range(35)]
        base_15m = [_contract_candle(i, 100.0) for i in range(35)]
        base_1h = [_contract_candle(i, 100.0) for i in range(35)]
        for rows in (base_3m, base_15m, base_1h):
            rows[30]["close"] = 100.10
            rows[30]["open"] = 100.00
            rows[30]["high"] = 100.20
            rows[30]["low"] = 99.95
        return {"3m": base_3m, "15m": base_15m, "1h": base_1h}

    def test_hybrid_backtest_triggers_hard_stop_and_rejects_strategy(self) -> None:
        contract = self._build_contract_data()
        mark = self._build_mark_data(contract["3m"])
        config = BacktestConfig(max_holding_bars=3)
        strategy_params = RuleStrategyParameters(long_roc_min=0.01, short_roc_max=-0.01)
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            config=config,
            strategy_params=strategy_params,
        )
        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0].exit_reason, "HARD_STOP_MARK_PRICE")
        self.assertEqual(result.decision_report.recommendation, "REJECT")
        self.assertIn("rejected_negative_total_pnl", result.decision_report.reasons)

    def test_trade_log_and_report_export(self) -> None:
        contract = self._build_contract_data()
        mark = self._build_mark_data(contract["3m"])
        strategy_params = RuleStrategyParameters(long_roc_min=0.01, short_roc_max=-0.01)
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            config=BacktestConfig(),
            strategy_params=strategy_params,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_path = Path(tmp_dir) / "trades.jsonl"
            report_path = Path(tmp_dir) / "decision_report.json"
            write_trade_logs_jsonl(result.trade_records, logs_path)
            write_decision_report_json(result.decision_report, report_path)
            self.assertTrue(logs_path.exists())
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text())
            self.assertIn("recommendation", report)
            self.assertIn("explanation", report)

    def test_backtest_records_ai_metadata_when_gate_allows_trade(self) -> None:
        contract = self._build_contract_data()
        mark = self._build_mark_data(contract["3m"])
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            config=BacktestConfig(),
            strategy_params=RuleStrategyParameters(long_roc_min=0.01, short_roc_max=-0.01),
            ai_trade_assistant=self._Assistant(
                AIDecision(
                    regime="trend_up",
                    setup_quality=0.8,
                    entry_action="allow",
                    exit_action="hold",
                    confidence=0.7,
                    reason_codes=["trend_alignment"],
                )
            ),
        )
        self.assertEqual(result.trade_records[0].model_base, "google/gemma-4-E2B-it")
        self.assertIsNotNone(result.trade_records[0].ai_snapshot)

    def test_backtest_blocks_long_entry_when_funding_is_overheated(self) -> None:
        contract = self._build_contract_data()
        mark = self._build_mark_data(contract["3m"])
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            funding_rate_rows=[{"funding_time": int(contract["3m"][30]["close_time"]), "funding_rate": 0.0002}],
            config=BacktestConfig(),
            strategy_params=RuleStrategyParameters(
                long_roc_min=0.01,
                short_roc_max=-0.01,
                max_long_funding_rate=0.0001,
            ),
        )
        self.assertEqual(len(result.trade_records), 0)

    def test_backtest_exits_weak_trade_with_early_fail_rule(self) -> None:
        contract = self._build_early_scratch_contract_data()
        mark = self._build_generic_mark_data(contract["3m"])
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            config=BacktestConfig(
                max_holding_bars=8,
                early_scratch_min_bars=3,
                early_scratch_min_mfe_r=0.15,
                early_scratch_max_adverse_r=0.2,
            ),
            strategy_params=RuleStrategyParameters(long_roc_min=0.01, short_roc_max=-0.01),
        )
        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0].exit_reason, "EARLY_FAIL_EXIT")

    def test_time_stop_only_policy_skips_atr_trail(self) -> None:
        contract = self._build_exit_policy_contract_data()
        contract["3m"][31]["open"] = 100.61
        contract["3m"][31]["high"] = 102.20
        contract["3m"][31]["low"] = 99.90
        contract["3m"][31]["close"] = 101.20
        mark = self._build_generic_mark_data(contract["3m"])
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            config=BacktestConfig(
                exit_policy="time_stop_only",
                max_holding_bars=1,
            ),
            strategy_params=RuleStrategyParameters(long_roc_min=0.01, short_roc_max=-0.01),
        )
        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0].exit_reason, "TIME_STOP")

    def test_break_even_policy_exits_on_reversal_after_arm(self) -> None:
        contract = self._build_exit_policy_contract_data()
        contract["3m"][31]["open"] = 100.61
        contract["3m"][31]["high"] = 102.20
        contract["3m"][31]["low"] = 99.80
        contract["3m"][31]["close"] = 100.40
        mark = self._build_generic_mark_data(contract["3m"])
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            config=BacktestConfig(
                exit_policy="break_even_time_stop",
                break_even_activation_profit_r=0.5,
                break_even_min_bars=1,
                max_holding_bars=8,
            ),
            strategy_params=RuleStrategyParameters(long_roc_min=0.01, short_roc_max=-0.01),
        )
        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0].exit_reason, "BREAK_EVEN_STOP_EXIT")

    def test_fixed_take_profit_policy_exits_at_target(self) -> None:
        contract = self._build_exit_policy_contract_data()
        contract["3m"][31]["open"] = 100.61
        contract["3m"][31]["high"] = 103.80
        contract["3m"][31]["low"] = 100.50
        contract["3m"][31]["close"] = 103.50
        mark = self._build_generic_mark_data(contract["3m"])
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            config=BacktestConfig(
                exit_policy="fixed_tp_time_stop",
                fixed_take_profit_r=1.0,
                max_holding_bars=8,
            ),
            strategy_params=RuleStrategyParameters(long_roc_min=0.01, short_roc_max=-0.01),
        )
        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0].exit_reason, "FIXED_TAKE_PROFIT")

    def test_partial_take_profit_policy_splits_trade(self) -> None:
        contract = self._build_exit_policy_contract_data()
        contract["3m"][31]["open"] = 100.61
        contract["3m"][31]["high"] = 103.80
        contract["3m"][31]["low"] = 100.50
        contract["3m"][31]["close"] = 103.50
        contract["3m"][32]["open"] = 103.50
        contract["3m"][32]["high"] = 103.70
        contract["3m"][32]["low"] = 102.80
        contract["3m"][32]["close"] = 103.00
        mark = self._build_generic_mark_data(contract["3m"])
        result = run_hybrid_backtest(
            contract_candles_by_timeframe=contract,
            lower_mark_price_candles=mark,
            config=BacktestConfig(
                exit_policy="partial_tp_runner",
                partial_take_profit_r=1.0,
                partial_take_profit_fraction=0.5,
                max_holding_bars=2,
            ),
            strategy_params=RuleStrategyParameters(long_roc_min=0.01, short_roc_max=-0.01),
        )
        self.assertEqual(len(result.trade_records), 2)
        self.assertEqual(result.trade_records[0].exit_reason, "PARTIAL_TAKE_PROFIT")
        self.assertEqual(result.trade_records[1].exit_reason, "TIME_STOP")

    def test_resample_klines_aggregates_one_minute_rows(self) -> None:
        candles = []
        for index, close in enumerate([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]):
            candles.append(
                {
                    "dataset": "contract_klines",
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "open_time": index * 60_000,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 10.0 + index,
                    "close_time": ((index + 1) * 60_000) - 1,
                    "quote_asset_volume": 1000.0 + index,
                    "number_of_trades": 100 + index,
                    "taker_buy_base_asset_volume": 4.0 + index,
                    "taker_buy_quote_asset_volume": 400.0 + index,
                    "ignore": "0",
                    "collected_at_ms": 1_000 + index,
                }
            )
        resampled = resample_klines(candles, target_interval="3m", source_interval="1m")
        self.assertEqual(len(resampled), 2)
        self.assertEqual(resampled[0]["open_time"], 0)
        self.assertEqual(resampled[0]["open"], 99.5)
        self.assertEqual(resampled[0]["close"], 102.0)
        self.assertEqual(resampled[0]["high"], 103.0)
        self.assertEqual(resampled[0]["low"], 99.0)
        self.assertEqual(resampled[0]["volume"], 33.0)

    def test_load_kline_parquet_filters_by_symbol_interval_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "candles.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "dataset": "contract_klines",
                            "symbol": "BTCUSDT",
                            "interval": "1m",
                            "open_time": 0,
                            "open": 100.0,
                            "high": 101.0,
                            "low": 99.0,
                            "close": 100.5,
                            "volume": 1.0,
                            "close_time": 59_999,
                            "quote_asset_volume": 1.0,
                            "number_of_trades": 1,
                            "taker_buy_base_asset_volume": 0.5,
                            "taker_buy_quote_asset_volume": 0.5,
                            "ignore": "0",
                            "collected_at_ms": 1,
                        },
                        {
                            "dataset": "contract_klines",
                            "symbol": "ETHUSDT",
                            "interval": "1m",
                            "open_time": 60_000,
                            "open": 200.0,
                            "high": 201.0,
                            "low": 199.0,
                            "close": 200.5,
                            "volume": 2.0,
                            "close_time": 119_999,
                            "quote_asset_volume": 2.0,
                            "number_of_trades": 2,
                            "taker_buy_base_asset_volume": 1.0,
                            "taker_buy_quote_asset_volume": 1.0,
                            "ignore": "0",
                            "collected_at_ms": 2,
                        },
                    ]
                ),
                path,
            )
            rows = load_kline_parquet(
                path,
                symbol="BTCUSDT",
                interval="1m",
                start_time=0,
                end_time=30_000,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "BTCUSDT")
            self.assertEqual(rows[0]["close"], 100.5)


if __name__ == "__main__":
    unittest.main()
