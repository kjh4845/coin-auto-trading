from __future__ import annotations

import unittest

from ai_auto_trading.runtime.position_manager import evaluate_managed_trade_exit
from ai_auto_trading.runtime.testnet import ManagedTradeState


def _candle(index: int, close: float, *, step_ms: int = 180000) -> dict[str, float | int]:
    return {
        "open_time": index * step_ms,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": 10.0 + index,
        "close_time": (index + 1) * step_ms - 1,
    }


class PositionManagerTest(unittest.TestCase):
    def test_atr_trail_exit_for_long(self) -> None:
        candles = [_candle(i, 100.0 + (0.1 * i)) for i in range(30)]
        candles[-1]["close"] = 100.0
        candles[-1]["high"] = 100.1
        candles[-1]["low"] = 99.7
        state = ManagedTradeState(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100.0,
            entry_mark_price=100.0,
            execution_timeframe="3m",
            atr_trailing_multiplier=1.0,
            max_holding_bars=100,
            opened_at_ms=0,
            signal_reason_codes=[],
            model_base="rule_only",
            adapter_version=None,
            ai_snapshot=None,
            bars_held=0,
            highest_high=100.0,
            lowest_low=100.0,
            last_processed_candle_close_time_ms=None,
            atr_trail_history=[],
        )
        decision = evaluate_managed_trade_exit(trade_state=state, execution_candles=candles)
        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.exit_reason, "ATR_TRAIL_EXIT")

    def test_time_stop_exit_after_max_holding_bars(self) -> None:
        candles = [_candle(i, 100.0 + (0.05 * i)) for i in range(20)]
        state = ManagedTradeState(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100.0,
            entry_mark_price=100.0,
            execution_timeframe="3m",
            atr_trailing_multiplier=10.0,
            max_holding_bars=1,
            opened_at_ms=0,
            signal_reason_codes=[],
            model_base="rule_only",
            adapter_version=None,
            ai_snapshot=None,
            bars_held=0,
            highest_high=100.0,
            lowest_low=100.0,
            last_processed_candle_close_time_ms=None,
            atr_trail_history=[],
        )
        decision = evaluate_managed_trade_exit(trade_state=state, execution_candles=candles)
        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.exit_reason, "TIME_STOP")

    def test_fixed_take_profit_exit(self) -> None:
        candles = [_candle(i, 100.0 + (0.05 * i)) for i in range(20)]
        candles[-1]["close"] = 103.0
        candles[-1]["high"] = 103.2
        candles[-1]["low"] = 102.7
        state = ManagedTradeState(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100.0,
            entry_mark_price=100.0,
            execution_timeframe="3m",
            atr_trailing_multiplier=10.0,
            max_holding_bars=20,
            opened_at_ms=0,
            signal_reason_codes=[],
            model_base="rule_only",
            adapter_version=None,
            ai_snapshot=None,
            bars_held=0,
            highest_high=100.0,
            lowest_low=100.0,
            last_processed_candle_close_time_ms=None,
            atr_trail_history=[],
            exit_policy="fixed_tp_time_stop",
            fixed_take_profit_r=1.0,
        )
        decision = evaluate_managed_trade_exit(trade_state=state, execution_candles=candles)
        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.exit_reason, "FIXED_TAKE_PROFIT")

    def test_pre_entry_candles_do_not_advance_trade_state(self) -> None:
        candles = [_candle(i, 100.0 + (0.05 * i)) for i in range(10)]
        state = ManagedTradeState(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100.0,
            entry_mark_price=100.0,
            execution_timeframe="3m",
            atr_trailing_multiplier=2.5,
            max_holding_bars=1,
            opened_at_ms=int(candles[-1]["close_time"]) + 1,
            signal_reason_codes=[],
            model_base="rule_only",
            adapter_version=None,
            ai_snapshot=None,
            bars_held=0,
            highest_high=100.0,
            lowest_low=100.0,
            last_processed_candle_close_time_ms=None,
            atr_trail_history=[],
        )
        decision = evaluate_managed_trade_exit(trade_state=state, execution_candles=candles)
        self.assertEqual(decision.action, "NO_ACTION")
        assert decision.updated_trade_state is not None
        self.assertEqual(decision.updated_trade_state.bars_held, 0)
        self.assertEqual(decision.updated_trade_state.highest_high, 100.0)

    def test_future_candle_is_ignored_until_closed(self) -> None:
        candle = _candle(0, 100.0)
        state = ManagedTradeState(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100.0,
            entry_mark_price=100.0,
            execution_timeframe="3m",
            atr_trailing_multiplier=2.5,
            max_holding_bars=1,
            opened_at_ms=0,
            signal_reason_codes=[],
            model_base="rule_only",
            adapter_version=None,
            ai_snapshot=None,
            bars_held=0,
            highest_high=100.0,
            lowest_low=100.0,
            last_processed_candle_close_time_ms=None,
            atr_trail_history=[],
        )
        decision = evaluate_managed_trade_exit(
            trade_state=state,
            execution_candles=[candle],
            latest_allowed_close_time_ms=int(candle["close_time"]) - 10,
        )
        self.assertEqual(decision.action, "NO_ACTION")
        assert decision.updated_trade_state is not None
        self.assertEqual(decision.updated_trade_state.bars_held, 0)


if __name__ == "__main__":
    unittest.main()
